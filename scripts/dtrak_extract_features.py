#!/usr/bin/env python3

import argparse
import inspect
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap  # vmap/grad kept for reference; not used in standard-backward path

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.inference.sampling import get_alphas_sigmas
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import (
    copy_state_dict,
    load_ckpt_state_dict,
    remove_weight_norm_from_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract D-TRAK style projected per-example gradients for stable-audio diffusion models."
    )
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True)
    parser.add_argument("--pretrained-ckpt-path", type=str, required=True)
    parser.add_argument("--pretransform-ckpt-path", type=str, default=None)
    parser.add_argument(
        "--remove-pretransform-weight-norm",
        type=str,
        default="none",
        choices=["none", "pre_load", "post_load"],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--proj-dim", type=int, required=True)
    parser.add_argument("--used-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--e-seed", type=int, default=0)
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument(
        "--t-strategy",
        type=str,
        default="uniform",
        choices=["uniform", "cumulative"],
    )
    parser.add_argument("--num-train-steps", type=int, default=1000)
    parser.add_argument(
        "--f",
        type=str,
        default="loss",
        choices=[
            "loss",
            "mean-squared-l2-norm",
            "mean",
            "l1-norm",
            "l2-norm",
            "linf-norm",
        ],
    )
    parser.add_argument("--cfg-dropout-prob", type=float, default=0.0)
    parser.add_argument("--feature-path", type=str, required=True)
    parser.add_argument("--ids-path", type=str, default=None)
    parser.add_argument("--meta-path", type=str, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--param-regex", type=str, default=None)
    parser.add_argument("--disable-random-crop", action="store_true")
    parser.add_argument("--autocast-pretransform", action="store_true")
    parser.add_argument("--disable-sdpa", action="store_true")
    parser.add_argument("--model-id", type=int, default=0)
    parser.add_argument("--pre-encoded", action="store_true")
    parser.add_argument("--no-pre-encoded", action="store_true")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_pre_encoded(args: argparse.Namespace, model_config: Dict[str, Any]) -> bool:
    if args.pre_encoded and args.no_pre_encoded:
        raise ValueError("Use only one of --pre-encoded or --no-pre-encoded.")
    if args.pre_encoded:
        return True
    if args.no_pre_encoded:
        return False
    return bool(model_config.get("training", {}).get("pre_encoded", False))


def resolve_num_examples(loader: torch.utils.data.DataLoader, max_examples: Optional[int]) -> int:
    if max_examples is not None:
        if max_examples <= 0:
            raise ValueError("--max-examples must be positive when provided.")
        return max_examples
    try:
        return len(loader.dataset)  # type: ignore[arg-type]
    except (TypeError, AttributeError) as exc:
        raise ValueError("Iterable dataset detected. Please pass --max-examples.") from exc


def select_timesteps(k: int, num_train_steps: int, strategy: str) -> List[int]:
    if k <= 0:
        raise ValueError("K must be positive.")
    if strategy == "uniform":
        stride = max(1, num_train_steps // k)
        steps = list(range(0, num_train_steps, stride))[:k]
    elif strategy == "cumulative":
        steps = list(range(k))
    else:
        raise ValueError(f"Unknown t_strategy: {strategy}")
    return steps


def get_sample_id(metadata_item: Dict[str, Any], fallback_index: int) -> str:
    for key in ("path", "relpath", "latent_filename", "__key__"):
        value = metadata_item.get(key, None)
        if value is not None:
            return str(value)
    return f"sample_{fallback_index}"


def vectorize_and_ignore_buffers(g: Sequence[torch.Tensor], device: str = "cpu") -> torch.Tensor:
    """Flatten per-sample gradients and stack them.

    Args:
        g: sequence of tensors, each of shape [batch_size, ...].
        device: target device for the output ("cpu" saves GPU memory).
    """
    batch_size = len(g[0])
    out = []
    for batch_ix in range(batch_size):
        out.append(torch.cat([x[batch_ix].flatten().to(device) for x in g]))
    return torch.stack(out)


def loss_from_prediction(prediction: torch.Tensor, target: torch.Tensor, f_name: str) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    if f_name == "mean-squared-l2-norm":
        value = F.mse_loss(prediction, torch.zeros_like(target), reduction="none")
        return value.reshape(1, -1).mean()
    if f_name == "loss":
        value = F.mse_loss(prediction, target, reduction="none")
        return value.reshape(1, -1).mean()
    flat_pred = prediction.reshape(1, -1)
    if f_name == "mean":
        return flat_pred.mean()
    if f_name == "l1-norm":
        return torch.norm(flat_pred, p=1.0, dim=-1).mean()
    if f_name == "l2-norm":
        return torch.norm(flat_pred, p=2.0, dim=-1).mean()
    if f_name == "linf-norm":
        return torch.norm(flat_pred, p=float("inf"), dim=-1).mean()
    raise ValueError(f"Unknown f: {f_name}")


def get_condition_kwargs(diffusion, metadata: List[Dict[str, Any]], device: torch.device) -> Dict[str, Optional[torch.Tensor]]:
    if diffusion.conditioner is None:
        return {
            "cross_attn_cond": None,
            "cross_attn_mask": None,
            "global_cond": None,
            "input_concat_cond": None,
            "prepend_cond": None,
            "prepend_cond_mask": None,
        }
    diffusion.conditioner.to(device)
    conditioning = diffusion.conditioner(metadata, device)
    model_kwargs = diffusion.get_conditioning_inputs(conditioning)
    return {
        "cross_attn_cond": model_kwargs.get("cross_attn_cond", None),
        "cross_attn_mask": model_kwargs.get("cross_attn_mask", None),
        "global_cond": model_kwargs.get("global_cond", None),
        "input_concat_cond": model_kwargs.get("input_concat_cond", None),
        "prepend_cond": model_kwargs.get("prepend_cond", None),
        "prepend_cond_mask": model_kwargs.get("prepend_cond_mask", None),
    }


def maybe_encode_input(
    diffusion,
    reals: torch.Tensor,
    pre_encoded: bool,
    use_autocast: bool,
    device: torch.device,
) -> torch.Tensor:
    diffusion_input = reals
    if diffusion.pretransform is None:
        return diffusion_input
    diffusion.pretransform.to(device)
    if not pre_encoded:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(use_autocast and device.type == "cuda")):
                diffusion_input = diffusion.pretransform.encode(diffusion_input)
    else:
        if hasattr(diffusion.pretransform, "scale") and diffusion.pretransform.scale != 1.0:
            diffusion_input = diffusion_input / diffusion.pretransform.scale
    return diffusion_input


def create_projector(
    grad_dim: int,
    proj_dim: int,
    seed: int,
    device: torch.device,
    max_batch_size: int,
):
    if device.type != "cuda":
        raise ValueError("D-TRAK CudaProjector requires a CUDA device.")
    try:
        from trak.projectors import CudaProjector, ProjectionType
    except Exception as exc:
        raise ImportError(
            "Cannot import `trak.projectors`. Install `traker==0.1.3` to match D-TRAK projector code."
        ) from exc
    kwargs = {
        "grad_dim": grad_dim,
        "proj_dim": proj_dim,
        "seed": seed,
        "proj_type": ProjectionType.normal,
        "device": str(device),
    }
    sig = inspect.signature(CudaProjector.__init__)
    if "max_batch_size" in sig.parameters:
        # fast_jl CUDA kernels only exist for batch sizes that are multiples of
        # 16 (project_normal_16, _32, …).  Clamp to at least 16.
        kwargs["max_batch_size"] = max(16, int(max_batch_size))
    return CudaProjector(**kwargs)


def main() -> None:
    args = parse_args()
    set_seeds(args.seed)

    if args.disable_sdpa and hasattr(F, "scaled_dot_product_attention"):
        # NOTE: deleting SDPA works for diffusers UNets (they have a manual-attention fallback)
        # but breaks stable-audio-tools (no fallback). In PyTorch 2.1+, SDPA automatically
        # falls back to the math kernel under vmap, so disabling is generally unnecessary.
        print("WARNING: --disable-sdpa is set but stable-audio-tools has no fallback attention. "
              "Ignoring this flag. SDPA will auto-select a vmap-compatible kernel.")

    model_config = load_json(args.model_config)
    dataset_config = load_json(args.dataset_config)
    if args.disable_random_crop and "random_crop" in dataset_config:
        dataset_config["random_crop"] = False
    dataset_config["drop_last"] = False

    pre_encoded = resolve_pre_encoded(args, model_config)
    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    audio_channels = model_config.get("audio_channels", 2)

    dataloader = create_dataloader_from_config(
        dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=sample_rate,
        sample_size=sample_size,
        audio_channels=audio_channels,
        shuffle=False,
    )

    num_examples = resolve_num_examples(dataloader, args.max_examples)
    selected_timesteps = select_timesteps(args.K, args.num_train_steps, args.t_strategy)

    diffusion = create_model_from_config(model_config)
    copy_state_dict(diffusion, load_ckpt_state_dict(args.pretrained_ckpt_path))

    if args.remove_pretransform_weight_norm == "pre_load" and diffusion.pretransform is not None:
        remove_weight_norm_from_model(diffusion.pretransform)

    if args.pretransform_ckpt_path and diffusion.pretransform is not None:
        diffusion.pretransform.load_state_dict(load_ckpt_state_dict(args.pretransform_ckpt_path))

    if args.remove_pretransform_weight_norm == "post_load" and diffusion.pretransform is not None:
        remove_weight_norm_from_model(diffusion.pretransform)

    device = torch.device(args.device)
    diffusion = diffusion.to(device)
    diffusion.eval()
    if diffusion.pretransform is not None:
        diffusion.pretransform.eval()

    model = diffusion.model
    # Using standard backward (not vmap+grad), so gradient checkpointing stays
    # ENABLED inside the transformer — this saves ~30 GiB of activation memory.
    print("Using standard backward with gradient checkpointing (no vmap)")
    regex = re.compile(args.param_regex) if args.param_regex else None
    # Build a dict that references the ACTUAL model parameters (not detached
    # copies).  We need them in the computation graph for autograd.grad().
    selected_param_names = set()
    for name, tensor in model.named_parameters():
        if tensor.requires_grad and (regex is None or regex.search(name)):
            selected_param_names.add(name)
    if not selected_param_names:
        raise ValueError("No trainable parameters selected. Check --param-regex.")

    grad_dim = sum(
        p.numel() for n, p in model.named_parameters() if n in selected_param_names
    )
    print(f"Selected {len(selected_param_names)} parameter groups, grad_dim={grad_dim:,} ({grad_dim*4/1024**3:.2f} GiB)")
    used_dim = args.used_dim if args.used_dim is not None else args.proj_dim
    if used_dim > args.proj_dim:
        raise ValueError("--used-dim cannot be larger than --proj-dim.")
    projector = create_projector(
        grad_dim=grad_dim,
        proj_dim=args.proj_dim,
        seed=args.seed,
        device=device,
        max_batch_size=args.batch_size,
    )

    # ---- Gradient computation strategy -----------------------------------------
    # We use standard torch.autograd.grad instead of vmap+grad so that
    # torch.utils.checkpoint.checkpoint (gradient checkpointing) can be used
    # inside the transformer layers.  This trades a small amount of extra
    # forward-pass compute for a *massive* reduction in activation memory
    # (~35 GiB → ~2 GiB for a 24-layer transformer).
    #
    # For batch_size > 1 we loop over samples; each sample gets its own
    # forward+backward with checkpointing, producing its per-sample gradient.
    # -------------------------------------------------------------------------

    # Collect the actual parameter tensors we want gradients for, in a stable order.
    param_list = [p for n, p in model.named_parameters() if n in selected_param_names]
    param_names = [n for n, p in model.named_parameters() if n in selected_param_names]

    feature_path = Path(args.feature_path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    dstore = np.memmap(
        str(feature_path),
        dtype=np.float32,
        mode="w+",
        shape=(num_examples, args.proj_dim),
    )

    ids_path = Path(args.ids_path) if args.ids_path else feature_path.with_suffix(feature_path.suffix + ".ids.txt")
    ids_path.parent.mkdir(parents=True, exist_ok=True)
    sample_ids: List[str] = []

    diffusion_objective = diffusion.diffusion_objective
    saved = 0
    import time as _time
    _t_start = _time.time()

    for batch_idx, batch in enumerate(dataloader):
        if saved >= num_examples:
            break

        reals, metadata = batch
        metadata_list = list(metadata)
        if isinstance(reals, np.ndarray):
            reals = torch.from_numpy(reals)
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]
        reals = reals.to(device)

        diffusion_input = maybe_encode_input(
            diffusion=diffusion,
            reals=reals,
            pre_encoded=pre_encoded,
            use_autocast=args.autocast_pretransform,
            device=device,
        )

        cond_kwargs = get_condition_kwargs(diffusion, metadata_list, device)

        # Offload conditioner & pretransform to CPU — they are no longer needed
        # until the next batch.  This frees ~1-2 GiB of GPU memory for the
        # vmap+grad computation that follows.
        if diffusion.conditioner is not None:
            diffusion.conditioner.to("cpu")
        if diffusion.pretransform is not None:
            diffusion.pretransform.to("cpu")
        torch.cuda.empty_cache()

        cond_kw = {
            "cross_attn_cond": cond_kwargs["cross_attn_cond"],
            "cross_attn_mask": cond_kwargs["cross_attn_mask"],
            "global_cond": cond_kwargs["global_cond"],
            "input_concat_cond": cond_kwargs["input_concat_cond"],
            "prepend_cond": cond_kwargs["prepend_cond"],
            "prepend_cond_mask": cond_kwargs["prepend_cond_mask"],
        }

        batch_size_actual = diffusion_input.shape[0]

        emb = None
        _t_batch = _time.time()
        for t_idx, timestep_int in enumerate(selected_timesteps):
            timestep_seed = args.e_seed * args.num_train_steps + timestep_int
            set_seeds(timestep_seed)

            timestep_value = float(timestep_int) / float(args.num_train_steps)
            t = torch.full(
                (batch_size_actual,),
                timestep_value,
                device=device,
                dtype=torch.float32,
            )

            if diffusion.dist_shift is not None:
                t = diffusion.dist_shift.time_shift(t, diffusion_input.shape[2])

            if diffusion_objective == "v":
                alphas, sigmas = get_alphas_sigmas(t)
                targets_noise = torch.randn_like(diffusion_input)
                alphas_b = alphas[:, None, None]
                sigmas_b = sigmas[:, None, None]
                noised_inputs = diffusion_input * alphas_b + targets_noise * sigmas_b
                targets = targets_noise * alphas_b - diffusion_input * sigmas_b
            elif diffusion_objective in ["rectified_flow", "rf_denoiser"]:
                targets_noise = torch.randn_like(diffusion_input)
                alphas_b = (1 - t)[:, None, None]
                sigmas_b = t[:, None, None]
                noised_inputs = diffusion_input * alphas_b + targets_noise * sigmas_b
                targets = targets_noise - diffusion_input
            else:
                raise ValueError(f"Unsupported diffusion objective: {diffusion_objective}")

            # --- Per-sample gradient via standard backward + checkpointing ----
            sample_grads_list = []
            _t_ts = _time.time()
            for si in range(batch_size_actual):
                # Build per-sample kwargs (slice batch dim)
                fwd_kwargs = {"cfg_dropout_prob": args.cfg_dropout_prob}
                # NOTE: we do NOT set use_checkpointing=False here, so the
                # transformer WILL use gradient checkpointing → huge memory saving
                for k, v in cond_kw.items():
                    if v is not None:
                        fwd_kwargs[k] = v[si : si + 1]

                # Enable grad only for selected params
                for p in model.parameters():
                    p.requires_grad_(False)
                for p in param_list:
                    p.requires_grad_(True)

                prediction = model(
                    noised_inputs[si : si + 1],
                    t[si : si + 1],
                    **fwd_kwargs,
                )
                if hasattr(prediction, "sample"):
                    prediction = prediction.sample
                loss = loss_from_prediction(prediction, targets[si : si + 1], args.f)

                grads = torch.autograd.grad(loss, param_list)
                # Flatten to a single vector on CPU immediately
                flat = torch.cat([g.detach().flatten().cpu() for g in grads])
                sample_grads_list.append(flat)

                # Free graph
                del prediction, loss, grads

            per_sample_grads = torch.stack(sample_grads_list)  # [B, grad_dim] on CPU
            del sample_grads_list
            torch.cuda.empty_cache()

            _elapsed_ts = _time.time() - _t_ts
            print(f"  batch {batch_idx} | timestep {t_idx+1}/{len(selected_timesteps)} "
                  f"(t={timestep_int}) | {batch_size_actual} samples | {_elapsed_ts:.1f}s",
                  flush=True)

            emb = per_sample_grads if emb is None else emb + per_sample_grads

        emb = emb / float(len(selected_timesteps))
        # Move back to GPU for projection.
        # CudaProjector (fast_jl) requires batch size to be a multiple of 16.
        # Pad if necessary, project, then slice back.
        emb_gpu = emb.to(device)
        actual_bs = emb_gpu.shape[0]
        if actual_bs % 16 != 0:
            pad_bs = ((actual_bs // 16) + 1) * 16
            emb_gpu = F.pad(emb_gpu, (0, 0, 0, pad_bs - actual_bs))
        emb = projector.project(emb_gpu, model_id=args.model_id)[:actual_bs]
        del emb_gpu
        if used_dim != args.proj_dim:
            emb[:, used_dim:] = 0

        batch_size = emb.shape[0]
        keep = min(batch_size, num_examples - saved)
        dstore[saved:saved + keep] = emb[:keep].detach().cpu().numpy().astype(np.float32)

        for local_ix in range(keep):
            sample_ids.append(get_sample_id(metadata_list[local_ix], fallback_index=saved + local_ix))

        saved += keep
        _elapsed_total = _time.time() - _t_start
        _elapsed_batch = _time.time() - _t_batch
        _eta = (_elapsed_total / saved) * (num_examples - saved) if saved > 0 else 0
        print(f"saved={saved}/{num_examples} | batch {_elapsed_batch:.1f}s | "
              f"total {_elapsed_total:.0f}s | ETA {_eta:.0f}s ({_eta/60:.1f}min)",
              flush=True)

    dstore.flush()
    with open(ids_path, "w", encoding="utf-8") as handle:
        for sample_id in sample_ids:
            handle.write(sample_id + "\n")

    meta_path = Path(args.meta_path) if args.meta_path else feature_path.with_suffix(feature_path.suffix + ".meta.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "feature_path": str(feature_path),
        "ids_path": str(ids_path),
        "num_rows_allocated": int(num_examples),
        "valid_rows": int(saved),
        "proj_dim": int(args.proj_dim),
        "used_dim": int(used_dim),
        "K": int(args.K),
        "t_strategy": args.t_strategy,
        "num_train_steps": int(args.num_train_steps),
        "f": args.f,
        "cfg_dropout_prob": float(args.cfg_dropout_prob),
        "model_id": int(args.model_id),
        "param_regex": args.param_regex,
        "grad_dim": int(grad_dim),
        "pre_encoded": bool(pre_encoded),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=True)

    print(f"feature_path={feature_path}")
    print(f"ids_path={ids_path}")
    print(f"meta_path={meta_path}")


if __name__ == "__main__":
    main()
