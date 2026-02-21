import gc
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ..inference.sampling import sample, sample_discrete_euler, sample_flow_pingpong
from .songscriber_clap import (
    DEFAULT_CLAP_MODEL_NAME,
    ClapEmbedder,
    clap_alignment,
    clap_fad,
    load_precomputed_song_describer_embeddings,
)


class SongDescriberClapEvalCallback(pl.Callback):
    def __init__(
        self,
        precomputed_embeddings_path: str,
        sample_size: int,
        eval_every: int = 2000,
        num_prompts: int = 64,
        eval_steps: int = 100,
        eval_cfg_scale: float = 7.0,
        gen_batch_size: int = 4,
        clap_batch_size: int = 8,
        clap_model_name: str = DEFAULT_CLAP_MODEL_NAME,
        clap_device: str = "cuda",
        metric_prefix: str = "songscriber",
    ) -> None:
        super().__init__()
        self.precomputed_embeddings_path = Path(precomputed_embeddings_path)
        self.sample_size = sample_size
        self.eval_every = eval_every
        self.num_prompts = num_prompts
        self.eval_steps = eval_steps
        self.eval_cfg_scale = eval_cfg_scale
        self.gen_batch_size = gen_batch_size
        self.clap_batch_size = clap_batch_size
        self.clap_model_name = clap_model_name
        self.clap_device = clap_device
        self.metric_prefix = metric_prefix
        self.last_eval_step = -1

        precomputed = load_precomputed_song_describer_embeddings(self.precomputed_embeddings_path)
        self.prompts = [str(x) for x in precomputed["prompts"]]
        self.durations = precomputed["durations"].astype(np.float32)
        self.ref_audio_embeddings = precomputed["audio_embeddings"].astype(np.float32)
        self.text_embeddings = precomputed["text_embeddings"].astype(np.float32)
        self.clap_embedder: ClapEmbedder | None = None

    def _autocast_context(self, device: torch.device):
        if device.type == "cuda":
            return torch.cuda.amp.autocast()
        return nullcontext()

    def _resolve_clap_device(self, module_device: torch.device) -> torch.device:
        if self.clap_device == "auto":
            return module_device if module_device.type == "cuda" else torch.device("cpu")
        return torch.device(self.clap_device)

    def _ensure_clap_embedder(self, module_device: torch.device) -> None:
        if self.clap_embedder is None:
            clap_device = self._resolve_clap_device(module_device)
            self.clap_embedder = ClapEmbedder(
                model_name=self.clap_model_name,
                device=clap_device,
            )

    def _build_conditioning(self, start: int, end: int, max_duration: float) -> list[dict[str, float | str]]:
        cond = []
        for idx in range(start, end):
            seconds_total = float(self.durations[idx])
            if seconds_total <= 0:
                seconds_total = max_duration
            seconds_total = float(min(seconds_total, max_duration))
            cond.append(
                {
                    "prompt": self.prompts[idx],
                    "seconds_start": 0.0,
                    "seconds_total": seconds_total,
                }
            )
        return cond

    def _generate_audio_batch(self, module, conditioning: list[dict[str, float | str]]) -> torch.Tensor:
        batch_size = len(conditioning)
        sample_size = self.sample_size

        if module.diffusion.pretransform is not None:
            sample_size = sample_size // module.diffusion.pretransform.downsampling_ratio

        noise = torch.randn(
            [batch_size, module.diffusion.io_channels, sample_size],
            device=module.device,
        )

        with self._autocast_context(module.device):
            cond_tensors = module.diffusion.conditioner(conditioning, module.device)
            cond_inputs = module.diffusion.get_conditioning_inputs(cond_tensors)
            model = module.diffusion_ema.ema_model if module.diffusion_ema is not None else module.diffusion.model

            if module.diffusion_objective == "v":
                fakes = sample(
                    model,
                    noise,
                    self.eval_steps,
                    0,
                    **cond_inputs,
                    cfg_scale=self.eval_cfg_scale,
                    dist_shift=module.diffusion.dist_shift,
                    batch_cfg=True,
                )
            elif module.diffusion_objective == "rectified_flow":
                fakes = sample_discrete_euler(
                    model,
                    noise,
                    self.eval_steps,
                    **cond_inputs,
                    cfg_scale=self.eval_cfg_scale,
                    dist_shift=module.diffusion.dist_shift,
                    batch_cfg=True,
                )
            elif module.diffusion_objective == "rf_denoiser":
                logsnr = torch.linspace(-6, 2, self.eval_steps + 1).to(module.device)
                sigmas = torch.sigmoid(-logsnr)
                sigmas[0] = 1.0
                sigmas[-1] = 0.0
                fakes = sample_flow_pingpong(
                    model,
                    noise,
                    sigmas=sigmas,
                    **cond_inputs,
                    cfg_scale=self.eval_cfg_scale,
                    dist_shift=module.diffusion.dist_shift,
                    batch_cfg=True,
                )
            else:
                raise ValueError(f"Unsupported diffusion objective: {module.diffusion_objective}")

            if module.diffusion.pretransform is not None:
                fakes = module.diffusion.pretransform.decode(fakes)

        return fakes

    def _log_metrics(self, trainer, metrics: dict[str, float]) -> None:
        logger = trainer.logger
        if logger is None:
            return

        step = trainer.global_step
        from pytorch_lightning.loggers import CometLogger, WandbLogger

        if isinstance(logger, WandbLogger):
            logger.experiment.log(metrics, step=step)
        elif isinstance(logger, CometLogger):
            logger.experiment.log_metrics(metrics, step=step)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if self.eval_every <= 0:
            return

        if (trainer.global_step - 1) % self.eval_every != 0 or self.last_eval_step == trainer.global_step:
            return

        self.last_eval_step = trainer.global_step
        module.eval()

        try:
            n = min(self.num_prompts, len(self.prompts))
            if n == 0:
                return

            self._ensure_clap_embedder(module.device)
            max_duration = self.sample_size / module.diffusion.sample_rate
            generated_embeddings: list[np.ndarray] = []

            for start in range(0, n, self.gen_batch_size):
                end = min(start + self.gen_batch_size, n)
                conditioning = self._build_conditioning(start, end, max_duration)
                generated_audio = self._generate_audio_batch(module, conditioning)
                audio_embeddings = self.clap_embedder.embed_audio_tensor(
                    generated_audio,
                    sample_rate=module.diffusion.sample_rate,
                    batch_size=self.clap_batch_size,
                )
                generated_embeddings.append(audio_embeddings)
                del generated_audio
                del audio_embeddings
                gc.collect()
                torch.cuda.empty_cache()

            if not generated_embeddings:
                return

            generated_embeddings_np = np.concatenate(generated_embeddings, axis=0)
            reference_embeddings = self.ref_audio_embeddings[:n]
            text_embeddings = self.text_embeddings[:n]

            fad_value = clap_fad(reference_embeddings, generated_embeddings_np)
            align_mean, align_std = clap_alignment(text_embeddings, generated_embeddings_np)

            metrics = {
                f"{self.metric_prefix}/clap_fad": float(fad_value),
                f"{self.metric_prefix}/clap_alignment_mean": float(align_mean),
                f"{self.metric_prefix}/clap_alignment_std": float(align_std),
                f"{self.metric_prefix}/num_prompts": float(n),
            }
            self._log_metrics(trainer, metrics)
            print(
                f"[{self.metric_prefix}] step={trainer.global_step} n={n} "
                f"clap_fad={fad_value:.6f} align_mean={align_mean:.6f} align_std={align_std:.6f}"
            )
        except Exception as e:
            print(f"SongDescriberClapEvalCallback failed: {type(e).__name__}: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            module.train()
