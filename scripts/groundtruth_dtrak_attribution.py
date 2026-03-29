#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the groundtruth-experiment attribution pipeline using D-TRAK."
    )
    parser.add_argument("--model-config", type=str, default="model_config_freeze_vae.json")
    parser.add_argument(
        "--train-dataset-config",
        type=str,
        default="stable_audio_tools/configs/dataset_configs/local_training_custom.json",
    )
    parser.add_argument(
        "--query-dataset-config",
        type=str,
        default="stable_audio_tools/configs/dataset_configs/dtrak_generated_queries.json",
    )
    parser.add_argument("--models-root", type=str, default="outputs/groundtruth_models")
    parser.add_argument("--ensemble-model-indices", type=str, default="")
    parser.add_argument("--full-model-dir", type=str, default="outputs/groundtruth_models/full")
    parser.add_argument("--unwrapped-ckpt", type=str, default=None)
    parser.add_argument("--pretransform-ckpt-path", type=str, default=None)
    parser.add_argument(
        "--remove-pretransform-weight-norm",
        type=str,
        default="none",
        choices=["none", "pre_load", "post_load"],
    )
    parser.add_argument("--output-dir", type=str, default="outputs/groundtruth_models/dtrak_attribution")
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--query-count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--proj-dim", type=int, default=16384)
    parser.add_argument("--used-dim", type=int, default=8192)
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
        default="l2-norm",
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
    parser.add_argument("--lambda-reg", type=float, default=1e2)
    parser.add_argument("--param-regex", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--e-seed", type=int, default=0)
    parser.add_argument("--autocast-pretransform", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_text_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def normalize_path(value: str) -> str:
    return os.path.normpath(os.path.abspath(value))


def candidate_keys(row: Dict[str, str]) -> List[str]:
    path_value = row.get("path", "")
    relpath_value = row.get("relpath", "")
    keys = []
    if path_value:
        keys.append(normalize_path(path_value))
        keys.append(os.path.normpath(path_value))
        keys.append(Path(path_value).name)
        keys.append(Path(path_value).stem)
    if relpath_value:
        keys.append(os.path.normpath(relpath_value))
        keys.append(Path(relpath_value).name)
        keys.append(Path(relpath_value).stem)
    return keys


def read_prompt_manifest(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    return list(payload.get("items", []))


def infer_query_dir(query_dataset_config: Dict[str, Any]) -> Path:
    datasets = query_dataset_config.get("datasets", [])
    if not datasets:
        raise ValueError("Query dataset config must include at least one dataset entry.")
    return (REPO_ROOT / datasets[0]["path"]).resolve()


def infer_query_count(query_dir: Path, explicit_query_count: int | None) -> int:
    if explicit_query_count is not None:
        return explicit_query_count
    manifest_items = read_prompt_manifest(query_dir / "prompt_manifest.json")
    if manifest_items:
        return len(manifest_items)
    count = len(list(query_dir.glob("*.mp3"))) + len(list(query_dir.glob("*.wav")))
    if count <= 0:
        raise ValueError(f"No query audio files found in {query_dir}")
    return count


def infer_train_count(candidate_pool_rows: List[Dict[str, str]], explicit_train_count: int | None) -> int:
    if explicit_train_count is not None:
        return explicit_train_count
    return len(candidate_pool_rows)


def parse_ensemble_model_indices(value: str) -> List[int]:
    if value.strip() == "":
        return []
    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        indices.append(int(item))
    return indices


def discover_model_dirs(models_root: Path, model_indices: Sequence[int]) -> List[Tuple[int, Path]]:
    pairs = []
    for model_index in model_indices:
        model_dir = models_root / str(model_index)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        pairs.append((int(model_index), model_dir))
    return pairs


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_paths_file(path: Path, values: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def build_axis_rows(
    candidate_pool_rows: List[Dict[str, str]],
    raw_train_ids: List[str],
) -> List[Dict[str, Any]]:
    raw_index_by_key: Dict[str, int] = {}
    for raw_index, raw_id in enumerate(raw_train_ids):
        normalized = normalize_path(raw_id)
        raw_index_by_key[normalized] = raw_index
        raw_index_by_key[os.path.normpath(raw_id)] = raw_index
        raw_index_by_key[Path(raw_id).name] = raw_index
        raw_index_by_key[Path(raw_id).stem] = raw_index

    axis_rows: List[Dict[str, Any]] = []
    seen_raw_indices = set()
    for pool_index, candidate_row in enumerate(candidate_pool_rows):
        match_index = None
        for key in candidate_keys(candidate_row):
            if key in raw_index_by_key:
                match_index = raw_index_by_key[key]
                break
        if match_index is None:
            raise KeyError(f"Could not match candidate pool row to train ids: {candidate_row}")
        seen_raw_indices.add(match_index)
        axis_rows.append(
            {
                "train_axis_index": pool_index,
                "raw_train_index": match_index,
                "pool_index": int(candidate_row["pool_index"]),
                "global_index": int(candidate_row["global_index"]),
                "dataset_id": candidate_row["dataset_id"],
                "dataset_root": candidate_row["dataset_root"],
                "track_id": candidate_row["track_id"],
                "path": candidate_row["path"],
                "relpath": candidate_row["relpath"],
                "train_id": raw_train_ids[match_index],
            }
        )
    if len(seen_raw_indices) != len(raw_train_ids):
        raise ValueError(
            f"Matched {len(seen_raw_indices)} unique train ids, but raw train feature axis has {len(raw_train_ids)} entries."
        )
    return axis_rows


def reorder_score_matrix(
    raw_score_path: Path,
    raw_shape: List[int],
    axis_rows: List[Dict[str, Any]],
    output_path: Path,
) -> torch.Tensor:
    query_count, raw_train_count = int(raw_shape[0]), int(raw_shape[1])
    raw_scores = np.memmap(str(raw_score_path), dtype=np.float32, mode="r", shape=(query_count, raw_train_count))
    output_scores = np.memmap(
        str(output_path),
        dtype=np.float32,
        mode="w+",
        shape=(query_count, len(axis_rows)),
    )
    for train_axis_index, row in enumerate(axis_rows):
        output_scores[:, train_axis_index] = raw_scores[:, int(row["raw_train_index"])]
    output_scores.flush()
    return torch.from_numpy(np.asarray(output_scores)).clone()


def build_query_axis_rows(query_ids: List[str], prompt_manifest_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prompt_by_filename = {item.get("filename", ""): item for item in prompt_manifest_items}
    rows: List[Dict[str, Any]] = []
    for query_axis_index, query_id in enumerate(query_ids):
        filename = Path(query_id).name
        prompt_item = prompt_by_filename.get(filename, {})
        rows.append(
            {
                "query_axis_index": query_axis_index,
                "query_id": query_id,
                "filename": filename,
                "prompt_id": prompt_item.get("prompt_id"),
                "prompt": prompt_item.get("prompt", ""),
                "seconds_start": prompt_item.get("seconds_start"),
                "seconds_total": prompt_item.get("seconds_total"),
                "audio_path": prompt_item.get("audio_path", query_id),
            }
        )
    return rows


def run_command(command: List[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=str(REPO_ROOT))


def run_member_pipeline(
    member_output_dir: Path,
    common_args: List[str],
    resolved_train_dataset_config_path: Path,
    resolved_query_dataset_config_path: Path,
    train_count: int,
    query_count: int,
    used_dim: int,
    lambda_reg: float,
    device: str,
) -> Tuple[Path, dict, dict, dict]:
    member_output_dir.mkdir(parents=True, exist_ok=True)

    train_feature_path = member_output_dir / "train_features_raw.memmap"
    query_feature_path = member_output_dir / "query_features.memmap"
    raw_score_path = member_output_dir / "scores_query_x_train_raw.memmap"

    run_command(
        common_args
        + [
            "--dataset-config",
            str(resolved_train_dataset_config_path),
            "--feature-path",
            str(train_feature_path),
            "--max-examples",
            str(train_count),
        ]
    )
    run_command(
        common_args
        + [
            "--dataset-config",
            str(resolved_query_dataset_config_path),
            "--feature-path",
            str(query_feature_path),
            "--max-examples",
            str(query_count),
        ]
    )
    run_command(
        [
            sys.executable,
            "scripts/dtrak_score.py",
            "--train-feature-paths",
            str(train_feature_path),
            "--train-meta-paths",
            str(train_feature_path) + ".meta.json",
            "--query-feature-path",
            str(query_feature_path),
            "--query-meta-path",
            str(query_feature_path) + ".meta.json",
            "--output-score-path",
            str(raw_score_path),
            "--lambda-reg",
            str(lambda_reg),
            "--used-dim",
            str(used_dim),
            "--device",
            device,
        ]
    )

    raw_score_meta = load_json(Path(str(raw_score_path) + ".meta.json"))
    train_feature_meta = load_json(Path(str(train_feature_path) + ".meta.json"))
    query_feature_meta = load_json(Path(str(query_feature_path) + ".meta.json"))
    return raw_score_path, raw_score_meta, train_feature_meta, query_feature_meta


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
    args = parse_args()

    model_config_path = (REPO_ROOT / args.model_config).resolve()
    train_dataset_config_path = (REPO_ROOT / args.train_dataset_config).resolve()
    query_dataset_config_path = (REPO_ROOT / args.query_dataset_config).resolve()
    models_root = (REPO_ROOT / args.models_root).resolve()
    ensemble_model_indices = parse_ensemble_model_indices(args.ensemble_model_indices)
    ensemble_pairs = discover_model_dirs(models_root, ensemble_model_indices)
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if ensemble_pairs:
        reference_model_index, reference_model_dir = ensemble_pairs[0]
        candidate_pool_path = reference_model_dir / "candidate_pool.csv"
        if not candidate_pool_path.exists():
            raise FileNotFoundError(f"candidate_pool.csv not found: {candidate_pool_path}")
        candidate_pool_rows = read_csv_rows(candidate_pool_path)
        if not candidate_pool_rows:
            raise RuntimeError(f"No candidate pool rows found in {candidate_pool_path}")
        train_include_paths_file = output_dir / "candidate_pool_paths.txt"
        write_paths_file(train_include_paths_file, [row["path"] for row in candidate_pool_rows])
    else:
        reference_model_index = None
        reference_model_dir = (REPO_ROOT / args.full_model_dir).resolve()
        candidate_pool_path = reference_model_dir / "candidate_pool.csv"
        train_include_paths_file = reference_model_dir / "selected_paths.txt"
        if not candidate_pool_path.exists():
            raise FileNotFoundError(f"candidate_pool.csv not found: {candidate_pool_path}")
        if not train_include_paths_file.exists():
            raise FileNotFoundError(f"selected_paths.txt not found: {train_include_paths_file}")
        candidate_pool_rows = read_csv_rows(candidate_pool_path)
        if not candidate_pool_rows:
            raise RuntimeError(f"No candidate pool rows found in {candidate_pool_path}")

    if ensemble_pairs:
        member_pairs = ensemble_pairs
    else:
        unwrapped_ckpt_path = (
            Path(args.unwrapped_ckpt).resolve()
            if args.unwrapped_ckpt is not None
            else (reference_model_dir / "model.ckpt").resolve()
        )
        if not unwrapped_ckpt_path.exists():
            raise FileNotFoundError(f"Unwrapped checkpoint not found: {unwrapped_ckpt_path}")
        member_pairs = [(-1, reference_model_dir)]

    train_count = infer_train_count(candidate_pool_rows, args.train_count)
    if train_count != len(candidate_pool_rows):
        raise ValueError(
            f"train_count={train_count} does not match candidate_pool.csv row count={len(candidate_pool_rows)}."
        )

    train_dataset_config = deepcopy(load_json(train_dataset_config_path))
    train_dataset_config["include_paths_file"] = str(train_include_paths_file)
    train_dataset_config["drop_last"] = False
    train_dataset_config["random_crop"] = False
    resolved_train_dataset_config_path = output_dir / "resolved_train_dataset_config.json"
    save_json(resolved_train_dataset_config_path, train_dataset_config)

    query_dataset_config = deepcopy(load_json(query_dataset_config_path))
    query_dataset_config["drop_last"] = False
    query_dataset_config["random_crop"] = False
    resolved_query_dataset_config_path = output_dir / "resolved_query_dataset_config.json"
    save_json(resolved_query_dataset_config_path, query_dataset_config)

    query_dir = infer_query_dir(query_dataset_config)
    prompt_manifest_items = read_prompt_manifest(query_dir / "prompt_manifest.json")
    query_count = infer_query_count(query_dir, args.query_count)

    final_score_memmap_path = output_dir / "scores_query_x_train.memmap"
    final_score_memmap_meta_path = Path(str(final_score_memmap_path) + ".meta.json")
    final_score_pt_path = output_dir / "scores_train_x_query.pt"
    train_axis_manifest_path = output_dir / "train_axis_manifest.csv"
    query_axis_manifest_path = output_dir / "query_axis_manifest.csv"

    ensemble_sum_query_x_train: torch.Tensor | None = None
    axis_rows: List[Dict[str, Any]] | None = None
    query_axis_rows: List[Dict[str, Any]] | None = None
    reference_raw_train_ids: List[str] | None = None
    reference_query_ids: List[str] | None = None
    member_metadata: List[Dict[str, Any]] = []
    raw_score_shape: List[int] | None = None

    for member_position, (member_index, member_dir) in enumerate(member_pairs):
        if member_index >= 0:
            checkpoint_path = member_dir / "model.ckpt"
        else:
            checkpoint_path = (
                Path(args.unwrapped_ckpt).resolve()
                if args.unwrapped_ckpt is not None
                else (member_dir / "model.ckpt").resolve()
            )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        member_output_dir = output_dir / "members" / (str(member_index) if member_index >= 0 else "single")
        common_args = [
            sys.executable,
            "scripts/dtrak_extract_features.py",
            "--model-config",
            str(model_config_path),
            "--pretrained-ckpt-path",
            str(checkpoint_path),
            "--remove-pretransform-weight-norm",
            args.remove_pretransform_weight_norm,
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--proj-dim",
            str(args.proj_dim),
            "--used-dim",
            str(args.used_dim),
            "--seed",
            str(args.seed),
            "--e-seed",
            str(args.e_seed),
            "--K",
            str(args.K),
            "--t-strategy",
            args.t_strategy,
            "--num-train-steps",
            str(args.num_train_steps),
            "--f",
            args.f,
            "--cfg-dropout-prob",
            str(args.cfg_dropout_prob),
            "--disable-random-crop",
        ]
        if args.param_regex:
            common_args.extend(["--param-regex", args.param_regex])
        if args.pretransform_ckpt_path:
            common_args.extend(["--pretransform-ckpt-path", str(Path(args.pretransform_ckpt_path).resolve())])
        if args.autocast_pretransform:
            common_args.append("--autocast-pretransform")

        raw_score_path, raw_score_meta, train_feature_meta, query_feature_meta = run_member_pipeline(
            member_output_dir=member_output_dir,
            common_args=common_args,
            resolved_train_dataset_config_path=resolved_train_dataset_config_path,
            resolved_query_dataset_config_path=resolved_query_dataset_config_path,
            train_count=train_count,
            query_count=query_count,
            used_dim=args.used_dim,
            lambda_reg=args.lambda_reg,
            device=args.device,
        )

        raw_train_ids = read_text_lines(Path(train_feature_meta["ids_path"]))
        query_ids = read_text_lines(Path(query_feature_meta["ids_path"]))
        if len(raw_train_ids) != train_count:
            raise ValueError(f"Expected {train_count} train ids, found {len(raw_train_ids)} for member {member_index}.")
        if len(query_ids) != query_count:
            raise ValueError(f"Expected {query_count} query ids, found {len(query_ids)} for member {member_index}.")

        if axis_rows is None:
            axis_rows = build_axis_rows(candidate_pool_rows=candidate_pool_rows, raw_train_ids=raw_train_ids)
            query_axis_rows = build_query_axis_rows(query_ids=query_ids, prompt_manifest_items=prompt_manifest_items)
            reference_raw_train_ids = list(raw_train_ids)
            reference_query_ids = list(query_ids)
            raw_score_shape = raw_score_meta["shape"]
        else:
            if raw_train_ids != reference_raw_train_ids:
                raise ValueError(f"Train ids mismatch for member {member_index}.")
            if query_ids != reference_query_ids:
                raise ValueError(f"Query ids mismatch for member {member_index}.")

        member_reordered_score_path = member_output_dir / "scores_query_x_train.memmap"
        ordered_scores_query_x_train = reorder_score_matrix(
            raw_score_path=raw_score_path,
            raw_shape=raw_score_meta["shape"],
            axis_rows=axis_rows,
            output_path=member_reordered_score_path,
        )

        ordered_scores_query_x_train = ordered_scores_query_x_train.to(torch.float64)
        if ensemble_sum_query_x_train is None:
            ensemble_sum_query_x_train = ordered_scores_query_x_train
        else:
            ensemble_sum_query_x_train.add_(ordered_scores_query_x_train)

        member_metadata.append(
            {
                "member_position": member_position,
                "model_index": None if member_index < 0 else int(member_index),
                "model_dir": str(member_dir),
                "checkpoint_path": str(checkpoint_path),
                "output_dir": str(member_output_dir),
                "raw_score_path": str(raw_score_path),
                "raw_score_shape": raw_score_meta["shape"],
            }
        )

    assert ensemble_sum_query_x_train is not None
    assert axis_rows is not None
    assert query_axis_rows is not None
    assert raw_score_shape is not None

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
        axis_rows,
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
            "source_raw_score_path": "member-specific",
            "source_raw_score_meta_path": "member-specific",
        },
    )

    save_json(
        output_dir / "attribution_metadata.json",
        {
            "model_config": str(model_config_path),
            "train_dataset_config": str(train_dataset_config_path),
            "query_dataset_config": str(query_dataset_config_path),
            "resolved_train_dataset_config": str(resolved_train_dataset_config_path),
            "resolved_query_dataset_config": str(resolved_query_dataset_config_path),
            "models_root": str(models_root),
            "reference_model_index": reference_model_index,
            "reference_model_dir": str(reference_model_dir),
            "ensemble_model_indices": [None if model_id < 0 else int(model_id) for model_id, _ in member_pairs],
            "ensemble_member_count": len(member_pairs),
            "ensemble_reduction": "mean",
            "full_model_dir": None if ensemble_pairs else str(reference_model_dir),
            "unwrapped_ckpt": None if ensemble_pairs else str(checkpoint_path),
            "candidate_pool_csv": str(candidate_pool_path),
            "train_include_paths_file": str(train_include_paths_file),
            "query_dir": str(query_dir),
            "train_count": train_count,
            "query_count": query_count,
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "device": args.device,
            "proj_dim": int(args.proj_dim),
            "used_dim": int(args.used_dim),
            "K": int(args.K),
            "t_strategy": args.t_strategy,
            "num_train_steps": int(args.num_train_steps),
            "f": args.f,
            "cfg_dropout_prob": float(args.cfg_dropout_prob),
            "lambda_reg": float(args.lambda_reg),
            "param_regex": args.param_regex,
            "member_metadata": member_metadata,
            "train_feature_path_raw": None,
            "train_feature_meta_path_raw": None,
            "query_feature_path": None,
            "query_feature_meta_path": None,
            "score_path_raw": None,
            "score_meta_path_raw": None,
            "score_path_query_x_train": str(final_score_memmap_path),
            "score_meta_path_query_x_train": str(final_score_memmap_meta_path),
            "score_path_train_x_query_pt": str(final_score_pt_path),
            "train_axis_manifest": str(train_axis_manifest_path),
            "query_axis_manifest": str(query_axis_manifest_path),
            "raw_score_shape": raw_score_shape,
            "final_score_shape_query_x_train": [
                int(ordered_scores_query_x_train.shape[0]),
                int(ordered_scores_query_x_train.shape[1]),
            ],
            "final_score_shape_train_x_query": [
                int(ordered_scores_query_x_train.shape[1]),
                int(ordered_scores_query_x_train.shape[0]),
            ],
        },
    )

    print(f"Saved canonical score matrix: {final_score_memmap_path}", flush=True)
    print(f"Saved anticipation-style score tensor: {final_score_pt_path}", flush=True)
    print(f"Saved train axis manifest: {train_axis_manifest_path}", flush=True)
    print(f"Saved query axis manifest: {query_axis_manifest_path}", flush=True)


if __name__ == "__main__":
    main()
