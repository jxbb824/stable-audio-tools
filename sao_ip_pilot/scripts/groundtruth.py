#!/usr/bin/env python3

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from stable_audio_tools.data.dataset import create_dataloader_from_config
from stable_audio_tools.inference.sampling import get_alphas_sigmas
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict, remove_weight_norm_from_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ground-truth query scores for subset stable-audio models."
    )
    parser.add_argument("--models-root", type=str, default="sao_ip_pilot/outputs/models")
    parser.add_argument("--model-indices", type=int, nargs="*", default=None)
    parser.add_argument("--model-config", type=str, default="model_config_freeze_vae.json")
    parser.add_argument(
        "--query-dataset-config",
        type=str,
        default="sao_ip_pilot/configs/query_dataset.json",
    )
    parser.add_argument("--output-path", type=str, default="sao_ip_pilot/outputs/losses_model_x_query.pt")
    parser.add_argument("--output-meta-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
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
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--remove-pretransform-weight-norm",
        type=str,
        default="none",
        choices=["none", "pre_load", "post_load"],
    )
    parser.add_argument("--pretransform-ckpt-path", type=str, default=None)
    parser.add_argument("--autocast-pretransform", action="store_true")
    parser.add_argument("--pre-encoded", action="store_true")
    parser.add_argument("--no-pre-encoded", action="store_true")
    return parser.parse_args()


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def resolve_pre_encoded(args: argparse.Namespace, model_config: Dict[str, Any]) -> bool:
    if args.pre_encoded and args.no_pre_encoded:
        raise ValueError("Use only one of --pre-encoded or --no-pre-encoded.")
    if args.pre_encoded:
        return True
    if args.no_pre_encoded:
        return False
    return bool(model_config.get("training", {}).get("pre_encoded", False))


def select_timesteps(k: int, num_train_steps: int, strategy: str) -> List[int]:
    if k <= 0:
        raise ValueError("K must be positive.")
    if strategy == "uniform":
        stride = max(1, num_train_steps // k)
        return list(range(0, num_train_steps, stride))[:k]
    if strategy == "cumulative":
        return list(range(k))
    raise ValueError(f"Unknown t_strategy: {strategy}")


def get_sample_id(metadata_item: Dict[str, Any], fallback_index: int) -> str:
    for key in ("path", "relpath", "latent_filename", "__key__"):
        value = metadata_item.get(key, None)
        if value is not None:
            return str(value)
    return f"sample_{fallback_index}"


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
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=(use_autocast and device.type == "cuda")):
                diffusion_input = diffusion.pretransform.encode(diffusion_input)
    else:
        if hasattr(diffusion.pretransform, "scale") and diffusion.pretransform.scale != 1.0:
            diffusion_input = diffusion_input / diffusion.pretransform.scale
    return diffusion_input


def score_from_prediction_per_example(
    prediction: torch.Tensor,
    target: torch.Tensor,
    f_name: str,
) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    batch_size = prediction.shape[0]
    if f_name == "mean-squared-l2-norm":
        value = F.mse_loss(prediction, torch.zeros_like(target), reduction="none")
        return value.reshape(batch_size, -1).mean(dim=1)
    if f_name == "loss":
        value = F.mse_loss(prediction, target, reduction="none")
        return value.reshape(batch_size, -1).mean(dim=1)
    flat_pred = prediction.reshape(batch_size, -1)
    if f_name == "mean":
        return flat_pred.mean(dim=1)
    if f_name == "l1-norm":
        return torch.norm(flat_pred, p=1.0, dim=-1)
    if f_name == "l2-norm":
        return torch.norm(flat_pred, p=2.0, dim=-1)
    if f_name == "linf-norm":
        return torch.norm(flat_pred, p=float("inf"), dim=-1)
    raise ValueError(f"Unknown f: {f_name}")


def discover_model_dirs(models_root: Path, model_indices: Optional[Sequence[int]]) -> List[Tuple[int, Path]]:
    if model_indices is not None and len(model_indices) > 0:
        pairs = []
        for model_index in model_indices:
            model_dir = models_root / str(model_index)
            if not model_dir.is_dir():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            pairs.append((int(model_index), model_dir))
        return pairs

    discovered = []
    for child in models_root.iterdir():
        if child.is_dir() and child.name.isdigit():
            discovered.append((int(child.name), child))
    discovered.sort(key=lambda item: item[0])
    if not discovered:
        raise FileNotFoundError(f"No numeric model directories found under {models_root}")
    return discovered


def build_query_cache(
    query_dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    max_examples: Optional[int],
) -> Tuple[List[Tuple[torch.Tensor, List[Dict[str, Any]]]], List[str]]:
    loader = create_dataloader_from_config(
        query_dataset_config,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=int(model_config["sample_rate"]),
        sample_size=int(model_config["sample_size"]),
        audio_channels=int(model_config.get("audio_channels", 2)),
        shuffle=False,
    )

    cached_batches: List[Tuple[torch.Tensor, List[Dict[str, Any]]]] = []
    query_ids: List[str] = []
    saved = 0

    for batch in loader:
        reals, metadata = batch
        metadata_list = list(metadata)
        if isinstance(reals, np.ndarray):
            reals = torch.from_numpy(reals)
        if reals.ndim == 4 and reals.shape[0] == 1:
            reals = reals[0]

        keep = reals.shape[0]
        if max_examples is not None:
            remaining = max_examples - saved
            if remaining <= 0:
                break
            keep = min(keep, remaining)

        reals = reals[:keep].contiguous().cpu()
        metadata_list = metadata_list[:keep]

        cached_batches.append((reals, metadata_list))
        for local_index, metadata_item in enumerate(metadata_list):
            query_ids.append(get_sample_id(metadata_item, fallback_index=saved + local_index))
        saved += keep

        if max_examples is not None and saved >= max_examples:
            break

    if not cached_batches:
        raise RuntimeError("No query samples were loaded.")
    return cached_batches, query_ids


def load_model_for_scoring(
    model_config: Dict[str, Any],
    checkpoint_path: Path,
    pretransform_ckpt_path: Optional[str],
    remove_pretransform_weight_norm: str,
    device: torch.device,
):
    diffusion = create_model_from_config(model_config)
    copy_state_dict(diffusion, load_ckpt_state_dict(str(checkpoint_path)))

    if remove_pretransform_weight_norm == "pre_load" and diffusion.pretransform is not None:
        remove_weight_norm_from_model(diffusion.pretransform)

    if pretransform_ckpt_path and diffusion.pretransform is not None:
        diffusion.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path))

    if remove_pretransform_weight_norm == "post_load" and diffusion.pretransform is not None:
        remove_weight_norm_from_model(diffusion.pretransform)

    diffusion = diffusion.to(device)
    diffusion.eval()
    if diffusion.pretransform is not None:
        diffusion.pretransform.eval()
    return diffusion


def score_queries_for_model(
    diffusion,
    cached_batches: Sequence[Tuple[torch.Tensor, List[Dict[str, Any]]]],
    selected_timesteps: Sequence[int],
    num_train_steps: int,
    f_name: str,
    cfg_dropout_prob: float,
    e_seed: int,
    pre_encoded: bool,
    autocast_pretransform: bool,
    device: torch.device,
) -> torch.Tensor:
    results: List[torch.Tensor] = []
    diffusion_objective = diffusion.diffusion_objective
    model = diffusion.model

    with torch.inference_mode():
        for batch_index, (reals_cpu, metadata_list) in enumerate(cached_batches):
            reals = reals_cpu.to(device, non_blocking=True)
            diffusion_input = maybe_encode_input(
                diffusion=diffusion,
                reals=reals,
                pre_encoded=pre_encoded,
                use_autocast=autocast_pretransform,
                device=device,
            )
            cond_kwargs = get_condition_kwargs(diffusion, metadata_list, device)

            batch_scores = torch.zeros(diffusion_input.shape[0], device=device, dtype=torch.float32)
            for timestep_int in selected_timesteps:
                timestep_seed = e_seed * num_train_steps + timestep_int
                set_seeds(timestep_seed)

                timestep_value = float(timestep_int) / float(num_train_steps)
                t = torch.full(
                    (diffusion_input.shape[0],),
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

                prediction = model(
                    noised_inputs,
                    t,
                    cfg_dropout_prob=cfg_dropout_prob,
                    cross_attn_cond=cond_kwargs["cross_attn_cond"],
                    cross_attn_mask=cond_kwargs["cross_attn_mask"],
                    global_cond=cond_kwargs["global_cond"],
                    input_concat_cond=cond_kwargs["input_concat_cond"],
                    prepend_cond=cond_kwargs["prepend_cond"],
                    prepend_cond_mask=cond_kwargs["prepend_cond_mask"],
                )
                if hasattr(prediction, "sample"):
                    prediction = prediction.sample
                batch_scores.add_(score_from_prediction_per_example(prediction, targets, f_name))

            batch_scores.div_(float(len(selected_timesteps)))
            results.append(batch_scores.cpu())
            print(
                f"  scored batch {batch_index + 1}/{len(cached_batches)} "
                f"({len(metadata_list)} queries)",
                flush=True,
            )

    return torch.cat(results, dim=0)


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parse_args()
    set_seeds(args.seed)

    models_root = Path(args.models_root).resolve()
    model_pairs = discover_model_dirs(models_root, args.model_indices)

    output_path = Path(args.output_path).resolve() if args.output_path else (models_root / "gt_gen.pt").resolve()
    output_meta_path = (
        Path(args.output_meta_path).resolve()
        if args.output_meta_path
        else output_path.with_suffix(output_path.suffix + ".meta.json")
    )
    query_ids_path = output_path.with_suffix(output_path.suffix + ".query_ids.txt")
    model_ids_path = output_path.with_suffix(output_path.suffix + ".model_ids.txt")

    device_name = args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)

    reference_model_config_path = model_pairs[0][1] / "resolved_model_config.json"
    if not reference_model_config_path.exists():
        reference_model_config_path = Path(args.model_config).resolve()
    reference_model_config = load_json(str(reference_model_config_path))

    query_dataset_config = deepcopy(load_json(args.query_dataset_config))
    query_dataset_config["drop_last"] = False
    query_dataset_config["random_crop"] = False

    selected_timesteps = select_timesteps(args.K, args.num_train_steps, args.t_strategy)
    cached_batches, query_ids = build_query_cache(
        query_dataset_config=query_dataset_config,
        model_config=reference_model_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_examples=args.max_examples,
    )

    result_rows: List[torch.Tensor] = []
    used_model_indices: List[int] = []
    used_model_dirs: List[str] = []

    for model_index, model_dir in model_pairs:
        checkpoint_path = model_dir / "model.ckpt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model_config_path = model_dir / "resolved_model_config.json"
        if not model_config_path.exists():
            model_config_path = Path(args.model_config).resolve()
        model_config = load_json(str(model_config_path))

        if int(model_config["sample_rate"]) != int(reference_model_config["sample_rate"]):
            raise ValueError(f"sample_rate mismatch for model {model_index}: {model_config_path}")
        if int(model_config["sample_size"]) != int(reference_model_config["sample_size"]):
            raise ValueError(f"sample_size mismatch for model {model_index}: {model_config_path}")

        print(f"[model {model_index}] loading {checkpoint_path}", flush=True)
        diffusion = load_model_for_scoring(
            model_config=model_config,
            checkpoint_path=checkpoint_path,
            pretransform_ckpt_path=args.pretransform_ckpt_path,
            remove_pretransform_weight_norm=args.remove_pretransform_weight_norm,
            device=device,
        )
        pre_encoded = resolve_pre_encoded(args, model_config)

        model_scores = score_queries_for_model(
            diffusion=diffusion,
            cached_batches=cached_batches,
            selected_timesteps=selected_timesteps,
            num_train_steps=args.num_train_steps,
            f_name=args.f,
            cfg_dropout_prob=args.cfg_dropout_prob,
            e_seed=args.e_seed,
            pre_encoded=pre_encoded,
            autocast_pretransform=args.autocast_pretransform,
            device=device,
        )

        result_rows.append(model_scores)
        used_model_indices.append(model_index)
        used_model_dirs.append(str(model_dir))

        del diffusion
        if device.type == "cuda":
            torch.cuda.empty_cache()

    final_result = torch.stack(result_rows, dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_result, output_path)

    query_ids_path.write_text("\n".join(query_ids) + "\n", encoding="utf-8")
    model_ids_path.write_text("\n".join(str(model_index) for model_index in used_model_indices) + "\n", encoding="utf-8")

    save_json(
        output_meta_path,
        {
            "output_path": str(output_path),
            "shape": [int(final_result.shape[0]), int(final_result.shape[1])],
            "models_root": str(models_root),
            "model_indices": used_model_indices,
            "model_dirs": used_model_dirs,
            "query_dataset_config": str(Path(args.query_dataset_config).resolve()),
            "reference_model_config": str(Path(reference_model_config_path).resolve()),
            "query_ids_path": str(query_ids_path),
            "model_ids_path": str(model_ids_path),
            "num_queries": len(query_ids),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "device": str(device),
            "seed": int(args.seed),
            "e_seed": int(args.e_seed),
            "K": int(args.K),
            "t_strategy": args.t_strategy,
            "selected_timesteps": [int(timestep) for timestep in selected_timesteps],
            "num_train_steps": int(args.num_train_steps),
            "f": args.f,
            "cfg_dropout_prob": float(args.cfg_dropout_prob),
            "max_examples": int(args.max_examples) if args.max_examples is not None else None,
            "pretransform_ckpt_path": str(Path(args.pretransform_ckpt_path).resolve()) if args.pretransform_ckpt_path else None,
            "remove_pretransform_weight_norm": args.remove_pretransform_weight_norm,
        },
    )

    print(f"Saved ground truth tensor: {output_path}", flush=True)
    print(f"Saved ground truth metadata: {output_meta_path}", flush=True)
    print(f"Ground truth shape: {tuple(final_result.shape)}", flush=True)


if __name__ == "__main__":
    main()
