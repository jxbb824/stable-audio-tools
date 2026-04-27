#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import huggingface_hub
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIFFUSION_INFLUENCE_ROOT = Path("/ocean/projects/mth240006p/xjiang6/diffusion-influence")
DIFFUSION_INFLUENCE_ROOT = Path(
    os.environ.get("DIFFUSION_INFLUENCE_ROOT", str(DEFAULT_DIFFUSION_INFLUENCE_ROOT))
).resolve()
if str(DIFFUSION_INFLUENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_INFLUENCE_ROOT))

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download

import curvlinops

if not hasattr(curvlinops, "KFACInverseLinearOperator"):
    curvlinops.KFACInverseLinearOperator = curvlinops.CGInverseLinearOperator

from curvlinops import EKFACLinearOperator
from diffusion_influence.compressor import IdentityCompressor, QuantizationCompressor
from diffusion_influence.iter_utils import ChainedIterable, SizedIterable, func_on_enum, reiterable_map
from diffusion_influence.score_calculator import ScoreCalculator
from groundtruth import (
    get_sample_id,
    load_model_for_scoring,
    maybe_encode_input,
    resolve_pre_encoded,
    score_from_prediction_per_example,
    set_seeds,
)
from groundtruth_dtrak_attribution import (
    candidate_keys,
    discover_model_dirs,
    load_json,
    parse_ensemble_model_indices,
    read_csv_rows,
    save_json,
    write_csv,
)
from stable_audio_tools.data.dataset import collation_fn, create_dataloader_from_config
from stable_audio_tools.inference.sampling import get_alphas_sigmas


def move_cached_gradient_value_to_device(value: Any, device: torch.device) -> Any:
    non_blocking = device.type == "cuda"
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=non_blocking)
    if isinstance(value, list):
        return [move_cached_gradient_value_to_device(item, device) for item in value]
    if isinstance(value, tuple) and not hasattr(value, "_replace"):
        return tuple(move_cached_gradient_value_to_device(item, device) for item in value)
    if hasattr(value, "_replace"):
        updates: Dict[str, Any] = {}
        for field_name in value._fields:
            field_value = getattr(value, field_name)
            moved_value = move_cached_gradient_value_to_device(field_value, device)
            if moved_value is not field_value:
                updates[field_name] = moved_value
        if updates:
            return value._replace(**updates)
    return value


def move_cached_gradient_sequence_to_device(
    values: Sequence[Any], device: torch.device
) -> List[Any]:
    return [move_cached_gradient_value_to_device(value, device) for value in values]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the groundtruth-experiment attribution pipeline using EKFAC influence."
    )
    parser.add_argument("--model-config", type=str, default="model_config_freeze_vae.json")
    parser.add_argument(
        "--train-dataset-config",
        type=str,
        default="sao_ip_pilot/configs/train_dataset.json",
    )
    parser.add_argument(
        "--query-dataset-config",
        type=str,
        default="sao_ip_pilot/configs/query_dataset.json",
    )
    parser.add_argument("--models-root", type=str, default="sao_ip_pilot/outputs/models")
    parser.add_argument("--ensemble-model-indices", type=str, default="")
    parser.add_argument("--full-model-dir", type=str, default="sao_ip_pilot/outputs/models/0")
    parser.add_argument("--unwrapped-ckpt", type=str, default=None)
    parser.add_argument("--pretransform-ckpt-path", type=str, default=None)
    parser.add_argument(
        "--remove-pretransform-weight-norm",
        type=str,
        default="none",
        choices=["none", "pre_load", "post_load"],
    )
    parser.add_argument("--output-dir", type=str, default="sao_ip_pilot/outputs/ekfac_attribution")
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--query-count", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--measurement-f",
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
    parser.add_argument("--measurement-cfg-dropout-prob", type=float, default=0.0)
    parser.add_argument("--loss-cfg-dropout-prob", type=float, default=None)
    parser.add_argument("--damping", type=float, default=1e-8)
    parser.add_argument("--kfac-num-samples-for-loss", type=int, default=63)
    parser.add_argument("--kfac-batch-size", type=int, default=2)
    parser.add_argument("--kfac-approx", type=str, default="expand", choices=["expand", "reduce"])
    parser.add_argument("--num-samples-for-loss-scoring", type=int, default=125)
    parser.add_argument("--num-samples-for-measurement-scoring", type=int, default=125)
    parser.add_argument("--num-loss-batch-aggregations", type=int, default=1)
    parser.add_argument("--num-measurement-batch-aggregations", type=int, default=1)
    parser.add_argument("--query-batch-size", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=10)
    parser.add_argument("--query-gradient-compression-bits", type=int, default=8, choices=[4, 8])
    parser.add_argument("--linear-layer-start-fraction", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autocast-pretransform", action="store_true")
    parser.add_argument("--pre-encoded", action="store_true")
    parser.add_argument("--no-pre-encoded", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.linear_layer_start_fraction < 1.0:
        raise ValueError("--linear-layer-start-fraction must be in [0.0, 1.0).")
    return args


class OrderedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, ordered_indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.ordered_indices = list(ordered_indices)

    def __len__(self) -> int:
        return len(self.ordered_indices)

    def __getitem__(self, index: int):
        return self.base_dataset[self.ordered_indices[index]]


class FrozenMetadataDataset(Dataset):
    def __init__(self, base_dataset: Dataset, metadata_overrides: Sequence[Dict[str, Any]]) -> None:
        self.base_dataset = base_dataset
        self.metadata_overrides = list(metadata_overrides)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        audio, metadata = self.base_dataset[index]
        merged_metadata = dict(metadata)
        merged_metadata.update(self.metadata_overrides[index])
        return audio, merged_metadata


class CachedLatentDataset(Dataset):
    def __init__(self, latents: Sequence[torch.Tensor], metadata_items: Sequence[Dict[str, Any]]) -> None:
        self.latents = list(latents)
        self.metadata_items = [dict(item) for item in metadata_items]

    def __len__(self) -> int:
        return len(self.latents)

    def __getitem__(self, index: int):
        return self.latents[index], dict(self.metadata_items[index])


class CurvlinopsInverseWrapper:
    def __init__(self, inverse_operator, params: Sequence[nn.Parameter]) -> None:
        self.inverse_operator = inverse_operator
        self._params = list(params)

    @property
    def params(self) -> List[nn.Parameter]:
        return self._params

    def matvec(self, x: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        return list(self.inverse_operator @ list(x))


def prepare_batch(batch) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    reals, metadata = batch
    metadata_list = list(metadata)
    if isinstance(reals, np.ndarray):
        reals = torch.from_numpy(reals)
    if reals.ndim == 4 and reals.shape[0] == 1:
        reals = reals[0]
    return reals, metadata_list


def repeat_metadata(metadata_list: Sequence[Dict[str, Any]], repeats: int) -> List[Dict[str, Any]]:
    if repeats <= 1:
        return list(metadata_list)
    return [metadata for metadata in metadata_list for _ in range(repeats)]


def sample_noised_targets(
    diffusion,
    diffusion_input: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.rand(diffusion_input.shape[0], device=device)
    if diffusion.dist_shift is not None:
        t = diffusion.dist_shift.time_shift(t, diffusion_input.shape[2])

    diffusion_objective = diffusion.diffusion_objective
    if diffusion_objective == "v":
        alphas, sigmas = get_alphas_sigmas(t)
        noise = torch.randn_like(diffusion_input)
        alphas = alphas[:, None, None]
        sigmas = sigmas[:, None, None]
        noised_inputs = diffusion_input * alphas + noise * sigmas
        targets = noise * alphas - diffusion_input * sigmas
    elif diffusion_objective in ["rectified_flow", "rf_denoiser"]:
        noise = torch.randn_like(diffusion_input)
        alphas = (1 - t)[:, None, None]
        sigmas = t[:, None, None]
        noised_inputs = diffusion_input * alphas + noise * sigmas
        targets = noise - diffusion_input
    else:
        raise ValueError(f"Unsupported diffusion objective: {diffusion_objective}")

    return noised_inputs, t, targets


def resolve_loss_cfg_dropout_prob(args: argparse.Namespace, model_config: Dict[str, Any]) -> float:
    if args.loss_cfg_dropout_prob is not None:
        return float(args.loss_cfg_dropout_prob)
    return float(model_config.get("training", {}).get("cfg_dropout_prob", 0.1))


def build_dataset_from_config(
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Dataset:
    loader = create_dataloader_from_config(
        dataset_config,
        batch_size=1,
        num_workers=1,
        sample_rate=int(model_config["sample_rate"]),
        sample_size=int(model_config["sample_size"]),
        audio_channels=int(model_config.get("audio_channels", 2)),
        shuffle=False,
    )
    return loader.dataset


def get_dataset_roots(dataset_config: Dict[str, Any]) -> List[str]:
    roots: List[str] = []
    for dataset_entry in dataset_config.get("datasets", []):
        dataset_path = dataset_entry.get("path")
        if dataset_path:
            roots.append(str((REPO_ROOT / dataset_path).resolve()))
            roots.append(os.path.normpath(dataset_path))
    return roots


def build_train_dataset(
    base_dataset: Dataset,
    candidate_pool_rows: Sequence[Dict[str, str]],
    dataset_roots: Sequence[str],
) -> OrderedDataset:
    filenames = getattr(base_dataset, "filenames", None)
    if filenames is None:
        raise TypeError("The train dataset must expose a .filenames attribute for candidate-pool alignment.")

    index_by_key: Dict[str, int] = {}
    for index, filename in enumerate(filenames):
        normalized = os.path.normpath(os.path.abspath(filename))
        index_by_key.setdefault(normalized, index)
        index_by_key.setdefault(os.path.normpath(filename), index)
        index_by_key.setdefault(Path(filename).name, index)
        index_by_key.setdefault(Path(filename).stem, index)
        for root in dataset_roots:
            try:
                relpath = os.path.normpath(os.path.relpath(filename, root))
            except ValueError:
                continue
            index_by_key.setdefault(relpath, index)
            index_by_key.setdefault(os.path.splitext(relpath)[0], index)

    ordered_indices: List[int] = []
    for candidate_row in candidate_pool_rows:
        matched_index = None
        for key in candidate_keys(candidate_row):
            if key in index_by_key:
                matched_index = index_by_key[key]
                break
        if matched_index is None:
            raise KeyError(f"Could not match candidate pool row to dataset filenames: {candidate_row}")
        ordered_indices.append(matched_index)

    return OrderedDataset(base_dataset, ordered_indices)


def build_train_axis_rows(candidate_pool_rows: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for train_axis_index, row in enumerate(candidate_pool_rows):
        rows.append(
            {
                "train_axis_index": train_axis_index,
                "raw_train_index": train_axis_index,
                "pool_index": int(row["pool_index"]),
                "global_index": int(row["global_index"]),
                "dataset_id": row["dataset_id"],
                "dataset_root": row["dataset_root"],
                "track_id": row["track_id"],
                "path": row["path"],
                "relpath": row["relpath"],
                "train_id": row["path"],
            }
        )
    return rows


def build_query_axis_rows(
    query_dataset: Dataset,
    batch_size: int = 8,
) -> Tuple[FrozenMetadataDataset, List[Dict[str, Any]]]:
    loader = DataLoader(
        query_dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collation_fn,
    )

    metadata_overrides: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    saved = 0
    for batch in loader:
        _, metadata_list = prepare_batch(batch)
        for local_index, metadata_item in enumerate(metadata_list):
            metadata_overrides.append({key: value for key, value in metadata_item.items() if key != "audio"})
            query_id = get_sample_id(metadata_item, fallback_index=saved + local_index)
            rows.append(
                {
                    "query_axis_index": saved + local_index,
                    "query_id": query_id,
                    "filename": Path(query_id).name,
                    "prompt_id": metadata_item.get("prompt_id"),
                    "prompt": metadata_item.get("prompt", ""),
                    "seconds_start": metadata_item.get("seconds_start"),
                    "seconds_total": metadata_item.get("seconds_total"),
                    "audio_path": metadata_item.get("path", query_id),
                }
            )
        saved += len(metadata_list)
    return FrozenMetadataDataset(query_dataset, metadata_overrides), rows


def materialize_latent_dataset(
    dataset: Dataset,
    diffusion,
    pre_encoded: bool,
    autocast_pretransform: bool,
    device: torch.device,
    batch_size: int,
) -> CachedLatentDataset:
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collation_fn,
    )

    cached_latents: List[torch.Tensor] = []
    cached_metadata: List[Dict[str, Any]] = []

    for batch in loader:
        reals, metadata_list = prepare_batch(batch)
        reals = reals.to(device, non_blocking=True)
        with torch.inference_mode():
            diffusion_inputs = maybe_encode_input(
                diffusion=diffusion,
                reals=reals,
                pre_encoded=pre_encoded,
                use_autocast=autocast_pretransform,
                device=device,
            )
        diffusion_inputs = diffusion_inputs.detach().cpu()
        for sample_index, metadata_item in enumerate(metadata_list):
            cached_latents.append(diffusion_inputs[sample_index].clone())
            cached_metadata.append(dict(metadata_item))

        del reals
        del diffusion_inputs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return CachedLatentDataset(cached_latents, cached_metadata)


class StableAudioInfluenceTask:
    def __init__(
        self,
        diffusion,
        pre_encoded: bool,
        autocast_pretransform: bool,
        device: torch.device,
        measurement_f: str,
        measurement_cfg_dropout_prob: float,
        loss_cfg_dropout_prob: float,
        num_samples_for_loss_scoring: int,
        num_samples_for_measurement_scoring: int,
    ) -> None:
        self.diffusion = diffusion
        self.pre_encoded = pre_encoded
        self.autocast_pretransform = autocast_pretransform
        self.device = device
        self.measurement_f = measurement_f
        self.measurement_cfg_dropout_prob = measurement_cfg_dropout_prob
        self.loss_cfg_dropout_prob = loss_cfg_dropout_prob
        self.num_samples_for_loss_scoring = num_samples_for_loss_scoring
        self.num_samples_for_measurement_scoring = num_samples_for_measurement_scoring

    def _prepare_inputs(
        self,
        batch,
        num_repeats: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        reals, metadata_list = prepare_batch(batch)
        if reals.shape[0] != 1:
            raise ValueError("ScoreCalculator should pass one example per batch.")

        reals = reals.to(self.device, non_blocking=True)
        diffusion_input = maybe_encode_input(
            diffusion=self.diffusion,
            reals=reals,
            pre_encoded=self.pre_encoded,
            use_autocast=self.autocast_pretransform,
            device=self.device,
        )
        if num_repeats > 1:
            diffusion_input = diffusion_input.repeat_interleave(num_repeats, dim=0)
            metadata_list = repeat_metadata(metadata_list, num_repeats)

        conditioning = (
            self.diffusion.conditioner(metadata_list, self.device)
            if self.diffusion.conditioner is not None
            else {}
        )
        noised_inputs, timesteps, targets = sample_noised_targets(
            diffusion=self.diffusion,
            diffusion_input=diffusion_input,
            device=self.device,
        )
        return noised_inputs, timesteps, targets, conditioning

    def compute_train_loss(self, batch, model) -> torch.Tensor:
        noised_inputs, timesteps, targets, conditioning = self._prepare_inputs(
            batch,
            num_repeats=self.num_samples_for_loss_scoring,
        )
        prediction = model(
            noised_inputs,
            timesteps,
            cond=conditioning,
            cfg_dropout_prob=self.loss_cfg_dropout_prob,
        )
        if hasattr(prediction, "sample"):
            prediction = prediction.sample
        return F.mse_loss(prediction, targets, reduction="mean")

    def compute_measurement(self, batch, model) -> torch.Tensor:
        noised_inputs, timesteps, targets, conditioning = self._prepare_inputs(
            batch,
            num_repeats=self.num_samples_for_measurement_scoring,
        )
        prediction = model(
            noised_inputs,
            timesteps,
            cond=conditioning,
            cfg_dropout_prob=self.measurement_cfg_dropout_prob,
        )
        if hasattr(prediction, "sample"):
            prediction = prediction.sample
        scores = score_from_prediction_per_example(prediction, targets, self.measurement_f)
        return scores.mean()


class StableAudioKFACWrapper(nn.Module):
    def __init__(
        self,
        diffusion,
        cfg_dropout_prob: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.diffusion = diffusion
        self.cfg_dropout_prob = cfg_dropout_prob
        self.device = device

    def forward(self, inputs):
        noised_inputs, timesteps, conditioning = inputs
        prediction = self.diffusion(
            noised_inputs.to(self.device),
            timesteps.to(self.device),
            cond=conditioning,
            cfg_dropout_prob=self.cfg_dropout_prob,
        )
        if hasattr(prediction, "sample"):
            prediction = prediction.sample
        return prediction.reshape(prediction.shape[0], -1)


def batch_to_audio_regression_batch(
    batch,
    diffusion,
    pre_encoded: bool,
    autocast_pretransform: bool,
    device: torch.device,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]], torch.Tensor]:
    reals, metadata_list = prepare_batch(batch)
    reals = reals.to(device, non_blocking=True)
    diffusion_input = maybe_encode_input(
        diffusion=diffusion,
        reals=reals,
        pre_encoded=pre_encoded,
        use_autocast=autocast_pretransform,
        device=device,
    )
    conditioning = diffusion.conditioner(metadata_list, device) if diffusion.conditioner is not None else {}
    noised_inputs, timesteps, targets = sample_noised_targets(
        diffusion=diffusion,
        diffusion_input=diffusion_input,
        device=device,
    )
    return (noised_inputs, timesteps, conditioning), targets.reshape(targets.shape[0], -1)


def select_linear_params(
    model,
    start_fraction: float = 0.0,
) -> Tuple[Dict[str, nn.Parameter], List[str]]:
    module_entries: List[Tuple[str, nn.Linear]] = [
        (module_name, module)
        for module_name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]

    start_index = int(len(module_entries) * start_fraction)
    selected_entries = module_entries[start_index:]

    params: Dict[str, nn.Parameter] = {}
    module_names: List[str] = []

    for module_name, module in selected_entries:
        added_module = False
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            params[full_name] = param
            added_module = True
        if added_module:
            module_names.append(module_name)

    if not params:
        raise RuntimeError("No trainable nn.Linear parameters were found in diffusion.model for EKFAC.")
    return params, module_names


def build_kfac_preconditioner(
    diffusion,
    train_dataset: Dataset,
    named_params: Dict[str, nn.Parameter],
    args: argparse.Namespace,
    pre_encoded: bool,
    loss_cfg_dropout_prob: float,
    device: torch.device,
    num_workers: int,
) -> CurvlinopsInverseWrapper:
    data_loader = DataLoader(
        train_dataset,
        shuffle=False,
        batch_size=args.kfac_batch_size,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=False,
        collate_fn=collation_fn,
    )
    data_loader = ChainedIterable(*(data_loader for _ in range(args.kfac_num_samples_for_loss)))
    data_loader = reiterable_map(
        lambda batch: batch_to_audio_regression_batch(
            batch=batch,
            diffusion=diffusion,
            pre_encoded=pre_encoded,
            autocast_pretransform=args.autocast_pretransform,
            device=device,
        ),
        data_loader,
    )
    data_loader = SizedIterable(
        iterable=data_loader,
        size=math.ceil(len(train_dataset) / args.kfac_batch_size) * args.kfac_num_samples_for_loss,
    )

    wrapped_model = StableAudioKFACWrapper(
        diffusion=diffusion,
        cfg_dropout_prob=loss_cfg_dropout_prob,
        device=device,
    )
    kfac_ggn = EKFACLinearOperator(
        wrapped_model,
        nn.MSELoss(reduction="mean"),
        named_params,
        data_loader,
        progressbar=True,
        fisher_type="mc",
        mc_samples=1,
        kfac_approx=args.kfac_approx,
        num_data=args.kfac_num_samples_for_loss * len(train_dataset),
        num_per_example_loss_terms=1,
        check_deterministic=False,
    )
    return CurvlinopsInverseWrapper(
        inverse_operator=kfac_ggn.inverse(damping=args.damping),
        params=list(named_params.values()),
    )


def build_gradient_iterables(
    score_calculator: ScoreCalculator,
    train_dataset: Dataset,
    query_dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
    num_workers: int,
):
    dataloader_kwargs = {
        "collate_fn": collation_fn,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }

    query_gradients = score_calculator.get_query_gradients_iterable(
        query_dataset=query_dataset,
        dataloader_kwargs=dataloader_kwargs,
    )
    query_compressor = QuantizationCompressor(bits=args.query_gradient_compression_bits)
    query_gradients = map(func_on_enum(query_compressor.compress), query_gradients)
    query_gradients = list(
        map(
            func_on_enum(
                lambda compressed: move_cached_gradient_sequence_to_device(
                    compressed, torch.device("cpu")
                )
            ),
            query_gradients,
        )
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
    query_gradients = reiterable_map(
        func_on_enum(
            lambda compressed: move_cached_gradient_sequence_to_device(
                query_compressor.decompress(compressed), device
            )
        ),
        query_gradients,
    )

    train_gradients = score_calculator.get_train_gradients_iterable(
        train_dataset=train_dataset,
        dataloader_kwargs=dataloader_kwargs,
    )
    train_gradients = map(func_on_enum(IdentityCompressor().compress), train_gradients)
    train_gradients = reiterable_map(func_on_enum(IdentityCompressor().decompress), train_gradients)
    return train_gradients, query_gradients


def save_query_x_train_memmap(path: Path, values: torch.Tensor) -> None:
    values = values.detach().cpu().to(torch.float32).contiguous()
    output_scores = np.memmap(
        str(path),
        dtype=np.float32,
        mode="w+",
        shape=(int(values.shape[0]), int(values.shape[1])),
    )
    output_scores[:] = values.numpy()
    output_scores.flush()


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parse_args()

    model_config_path = (REPO_ROOT / args.model_config).resolve()
    train_dataset_config_path = (REPO_ROOT / args.train_dataset_config).resolve()
    query_dataset_config_path = (REPO_ROOT / args.query_dataset_config).resolve()
    models_root = (REPO_ROOT / args.models_root).resolve()
    ensemble_model_indices = parse_ensemble_model_indices(args.ensemble_model_indices)
    ensemble_pairs = discover_model_dirs(models_root, ensemble_model_indices)
    if ensemble_pairs and args.unwrapped_ckpt is not None:
        raise ValueError("Use either --ensemble-model-indices or --unwrapped-ckpt, not both.")
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if ensemble_pairs:
        reference_model_dir = ensemble_pairs[0][1]
    else:
        reference_model_dir = (REPO_ROOT / args.full_model_dir).resolve()

    candidate_pool_path = reference_model_dir / "candidate_pool.csv"
    if not candidate_pool_path.exists():
        raise FileNotFoundError(f"candidate_pool.csv not found: {candidate_pool_path}")
    candidate_pool_rows = read_csv_rows(candidate_pool_path)
    if not candidate_pool_rows:
        raise RuntimeError(f"No candidate pool rows found in {candidate_pool_path}")

    max_train_count = len(candidate_pool_rows)
    train_count = args.train_count if args.train_count is not None else max_train_count
    if train_count <= 0 or train_count > max_train_count:
        raise ValueError(
            f"train_count={train_count} must be in [1, {max_train_count}] for candidate_pool.csv."
        )
    candidate_pool_rows = candidate_pool_rows[:train_count]

    if ensemble_pairs:
        member_pairs = ensemble_pairs
    else:
        checkpoint_path = (
            Path(args.unwrapped_ckpt).resolve()
            if args.unwrapped_ckpt is not None
            else (reference_model_dir / "model.ckpt").resolve()
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        member_pairs = [(-1, reference_model_dir)]

    reference_model_config_path = reference_model_dir / "resolved_model_config.json"
    if not reference_model_config_path.exists():
        reference_model_config_path = model_config_path
    reference_model_config = load_json(reference_model_config_path)

    train_dataset_config = deepcopy(load_json(train_dataset_config_path))
    train_dataset_config["drop_last"] = False
    train_dataset_config["random_crop"] = False
    resolved_train_dataset_config_path = output_dir / "resolved_train_dataset_config.json"
    save_json(resolved_train_dataset_config_path, train_dataset_config)

    query_dataset_config = deepcopy(load_json(query_dataset_config_path))
    query_dataset_config["drop_last"] = False
    query_dataset_config["random_crop"] = False
    resolved_query_dataset_config_path = output_dir / "resolved_query_dataset_config.json"
    save_json(resolved_query_dataset_config_path, query_dataset_config)

    train_base_dataset = build_dataset_from_config(train_dataset_config, reference_model_config)
    train_dataset_roots = get_dataset_roots(train_dataset_config)
    train_dataset = build_train_dataset(train_base_dataset, candidate_pool_rows, train_dataset_roots)

    query_base_dataset = build_dataset_from_config(query_dataset_config, reference_model_config)
    query_count = args.query_count if args.query_count is not None else len(query_base_dataset)
    if query_count <= 0:
        raise ValueError("Query dataset must contain at least one example.")
    if query_count > len(query_base_dataset):
        raise ValueError(f"query_count={query_count} exceeds dataset size={len(query_base_dataset)}.")
    query_dataset = Subset(query_base_dataset, list(range(query_count)))

    train_axis_rows = build_train_axis_rows(candidate_pool_rows)
    set_seeds(args.seed)
    query_dataset, query_axis_rows = build_query_axis_rows(query_dataset)

    device = torch.device(args.device)

    final_score_memmap_path = output_dir / "scores_query_x_train.memmap"
    final_score_memmap_meta_path = Path(str(final_score_memmap_path) + ".meta.json")
    final_score_pt_path = output_dir / "scores_train_x_query.pt"
    train_axis_manifest_path = output_dir / "train_axis_manifest.csv"
    query_axis_manifest_path = output_dir / "query_axis_manifest.csv"

    ensemble_sum_query_x_train: Optional[torch.Tensor] = None
    member_metadata: List[Dict[str, Any]] = []
    cached_train_dataset: Optional[Dataset] = None
    cached_query_dataset: Optional[Dataset] = None

    for member_position, (member_index, member_dir) in enumerate(member_pairs):
        checkpoint_path = (
            (member_dir / "model.ckpt").resolve()
            if member_index >= 0 and args.unwrapped_ckpt is None
            else (
                Path(args.unwrapped_ckpt).resolve()
                if args.unwrapped_ckpt is not None
                else (member_dir / "model.ckpt").resolve()
            )
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        model_config_for_member_path = member_dir / "resolved_model_config.json"
        if not model_config_for_member_path.exists():
            model_config_for_member_path = model_config_path
        model_config = load_json(model_config_for_member_path)

        if int(model_config["sample_rate"]) != int(reference_model_config["sample_rate"]):
            raise ValueError(f"sample_rate mismatch for member {member_index}: {model_config_for_member_path}")
        if int(model_config["sample_size"]) != int(reference_model_config["sample_size"]):
            raise ValueError(f"sample_size mismatch for member {member_index}: {model_config_for_member_path}")

        set_seeds(args.seed)
        diffusion = load_model_for_scoring(
            model_config=model_config,
            checkpoint_path=checkpoint_path,
            pretransform_ckpt_path=args.pretransform_ckpt_path,
            remove_pretransform_weight_norm=args.remove_pretransform_weight_norm,
            device=device,
        )
        pre_encoded = resolve_pre_encoded(args, model_config)
        loss_cfg_dropout_prob = resolve_loss_cfg_dropout_prob(args, model_config)
        using_cached_latents = False
        if diffusion.pretransform is not None:
            if cached_train_dataset is None or cached_query_dataset is None:
                cached_train_dataset = materialize_latent_dataset(
                    dataset=train_dataset,
                    diffusion=diffusion,
                    pre_encoded=pre_encoded,
                    autocast_pretransform=args.autocast_pretransform,
                    device=device,
                    batch_size=1,
                )
                cached_query_dataset = materialize_latent_dataset(
                    dataset=query_dataset,
                    diffusion=diffusion,
                    pre_encoded=pre_encoded,
                    autocast_pretransform=args.autocast_pretransform,
                    device=device,
                    batch_size=1,
                )
            member_train_dataset = cached_train_dataset
            member_query_dataset = cached_query_dataset
            pre_encoded = True
            using_cached_latents = True
        else:
            member_train_dataset = train_dataset
            member_query_dataset = query_dataset

        effective_num_workers = 0 if using_cached_latents else args.num_workers
        param_probe = StableAudioKFACWrapper(
            diffusion=diffusion,
            cfg_dropout_prob=loss_cfg_dropout_prob,
            device=device,
        )
        named_params, linear_module_names = select_linear_params(
            param_probe,
            start_fraction=args.linear_layer_start_fraction,
        )
        print(
            f"Selected {len(linear_module_names)} nn.Linear modules "
            f"({sum(param.numel() for param in named_params.values()):,} parameters) "
            f"with kfac_approx={args.kfac_approx} "
            f"and linear_layer_start_fraction={args.linear_layer_start_fraction:.3f}"
        )

        preconditioner = build_kfac_preconditioner(
            diffusion=diffusion,
            train_dataset=member_train_dataset,
            named_params=named_params,
            args=args,
            pre_encoded=pre_encoded,
            loss_cfg_dropout_prob=loss_cfg_dropout_prob,
            device=device,
            num_workers=effective_num_workers,
        )
        task = StableAudioInfluenceTask(
            diffusion=diffusion,
            pre_encoded=pre_encoded,
            autocast_pretransform=args.autocast_pretransform,
            device=device,
            measurement_f=args.measurement_f,
            measurement_cfg_dropout_prob=args.measurement_cfg_dropout_prob,
            loss_cfg_dropout_prob=loss_cfg_dropout_prob,
            num_samples_for_loss_scoring=args.num_samples_for_loss_scoring,
            num_samples_for_measurement_scoring=args.num_samples_for_measurement_scoring,
        )
        score_calculator = ScoreCalculator(
            model=diffusion,
            task=task,
            preconditioner=preconditioner,
            num_loss_batch_aggregations=args.num_loss_batch_aggregations,
            num_measurement_batch_aggregations=args.num_measurement_batch_aggregations,
            precondition_query_gradients=True,
        )
        train_gradients, query_gradients = build_gradient_iterables(
            score_calculator=score_calculator,
            train_dataset=member_train_dataset,
            query_dataset=member_query_dataset,
            args=args,
            device=device,
            num_workers=effective_num_workers,
        )
        member_scores_query_x_train = score_calculator.compute_pairwise_scores_from_gradients(
            query_gradients=query_gradients,
            train_gradients=train_gradients,
            num_query_examples=len(member_query_dataset),
            num_train_examples=len(member_train_dataset),
            query_batch_size=args.query_batch_size,
            train_batch_size=args.train_batch_size,
            outer_is_query=False,
        ).to(torch.float32)

        member_output_dir = output_dir / "members" / (str(member_index) if member_index >= 0 else "single")
        member_output_dir.mkdir(parents=True, exist_ok=True)
        save_query_x_train_memmap(member_output_dir / "scores_query_x_train.memmap", member_scores_query_x_train)

        if ensemble_sum_query_x_train is None:
            ensemble_sum_query_x_train = member_scores_query_x_train.to(torch.float64)
        else:
            ensemble_sum_query_x_train.add_(member_scores_query_x_train.to(torch.float64))

        member_metadata.append(
            {
                "member_position": member_position,
                "model_index": None if member_index < 0 else int(member_index),
                "model_dir": str(member_dir),
                "checkpoint_path": str(checkpoint_path),
                "output_dir": str(member_output_dir),
                "num_linear_modules": len(linear_module_names),
                "num_parameters": int(sum(param.numel() for param in preconditioner.params)),
                "loss_cfg_dropout_prob": float(loss_cfg_dropout_prob),
                "pre_encoded": bool(pre_encoded),
                "cached_latents": bool(using_cached_latents),
                "kfac_approx": args.kfac_approx,
                "linear_layer_start_fraction": float(args.linear_layer_start_fraction),
            }
        )

        del diffusion
        del preconditioner
        del score_calculator
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if ensemble_sum_query_x_train is None:
        raise RuntimeError("No attribution scores were computed.")

    ordered_scores_query_x_train = (ensemble_sum_query_x_train / float(len(member_pairs))).to(torch.float32)
    save_query_x_train_memmap(final_score_memmap_path, ordered_scores_query_x_train)
    torch.save(ordered_scores_query_x_train.T.contiguous(), final_score_pt_path)

    write_csv(
        train_axis_manifest_path,
        [
            "train_axis_index",
            "raw_train_index",
            "pool_index",
            "global_index",
            "dataset_id",
            "dataset_root",
            "track_id",
            "path",
            "relpath",
            "train_id",
        ],
        train_axis_rows,
    )
    write_csv(
        query_axis_manifest_path,
        [
            "query_axis_index",
            "query_id",
            "filename",
            "prompt_id",
            "prompt",
            "seconds_start",
            "seconds_total",
            "audio_path",
        ],
        query_axis_rows,
    )

    save_json(
        final_score_memmap_meta_path,
        {
            "output_score_path": str(final_score_memmap_path),
            "shape": [
                int(ordered_scores_query_x_train.shape[0]),
                int(ordered_scores_query_x_train.shape[1]),
            ],
            "layout": "query_x_train",
            "train_axis_order": "candidate_pool_pool_index",
            "query_axis_manifest": str(query_axis_manifest_path),
            "train_axis_manifest": str(train_axis_manifest_path),
            "method": "ekfac_influence",
        },
    )

    save_json(
        output_dir / "attribution_metadata.json",
        {
            "method": "ekfac_influence",
            "diffusion_influence_root": str(DIFFUSION_INFLUENCE_ROOT),
            "model_config": str(model_config_path),
            "train_dataset_config": str(train_dataset_config_path),
            "query_dataset_config": str(query_dataset_config_path),
            "resolved_train_dataset_config": str(resolved_train_dataset_config_path),
            "resolved_query_dataset_config": str(resolved_query_dataset_config_path),
            "models_root": str(models_root),
            "full_model_dir": str((REPO_ROOT / args.full_model_dir).resolve()),
            "output_dir": str(output_dir),
            "train_count": int(len(train_dataset)),
            "query_count": int(len(query_dataset)),
            "device": str(device),
            "seed": int(args.seed),
            "measurement_f": args.measurement_f,
            "measurement_cfg_dropout_prob": float(args.measurement_cfg_dropout_prob),
            "loss_cfg_dropout_prob": float(resolve_loss_cfg_dropout_prob(args, reference_model_config)),
            "kfac_num_samples_for_loss": int(args.kfac_num_samples_for_loss),
            "kfac_batch_size": int(args.kfac_batch_size),
            "num_samples_for_loss_scoring": int(args.num_samples_for_loss_scoring),
            "num_samples_for_measurement_scoring": int(args.num_samples_for_measurement_scoring),
            "num_loss_batch_aggregations": int(args.num_loss_batch_aggregations),
            "num_measurement_batch_aggregations": int(args.num_measurement_batch_aggregations),
            "query_batch_size": int(args.query_batch_size),
            "train_batch_size": int(args.train_batch_size),
            "query_gradient_compression": {
                "type": "quantization",
                "bits": int(args.query_gradient_compression_bits),
            },
            "kfac": {
                "correct_eigenvalues": True,
                "kfac_approx": args.kfac_approx,
                "fisher_type": "mc",
                "mc_samples": 1,
                "damping": float(args.damping),
                "use_exact_damping": True,
                "use_heuristic_damping": False,
            },
            "parameter_selection": {
                "type": "diffusion.model nn.Linear parameters",
                "linear_layer_start_fraction": float(args.linear_layer_start_fraction),
            },
            "train_axis_manifest": str(train_axis_manifest_path),
            "query_axis_manifest": str(query_axis_manifest_path),
            "scores_query_x_train_memmap": str(final_score_memmap_path),
            "scores_train_x_query_pt": str(final_score_pt_path),
            "members": member_metadata,
        },
    )


if __name__ == "__main__":
    main()
