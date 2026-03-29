#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a prompt JSON file for full-model query generation."
    )
    parser.add_argument(
        "--dataset-config",
        type=str,
        default="stable_audio_tools/configs/dataset_configs/local_training_custom.json",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="model_config_freeze_vae.json",
    )
    parser.add_argument(
        "--candidate-manifest",
        type=str,
        default="outputs/groundtruth_models/full/candidate_pool.csv",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/groundtruth_models/full/prompts_200.json",
    )
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--candidate-pool-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def load_candidate_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    candidate_manifest = Path(args.candidate_manifest)
    if candidate_manifest.exists():
        with candidate_manifest.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            return list(reader)

    from stable_audio_tools.data.subsets import resolve_audio_subset_records

    dataset_config = load_json(args.dataset_config)
    records = resolve_audio_subset_records(dataset_config)[: args.candidate_pool_size]
    rows = []
    for idx, record in enumerate(records):
        rows.append(
            {
                "pool_index": str(idx),
                "global_index": str(record.global_index),
                "dataset_id": record.dataset_id,
                "dataset_root": record.dataset_root,
                "track_id": "" if record.track_id is None else str(record.track_id),
                "path": record.path,
                "relpath": record.relpath,
            }
        )
    return rows


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def load_custom_md_fma():
    module_path = REPO_ROOT / "stable_audio_tools/configs/dataset_configs/custom_metadata/custom_md_fma.py"
    spec = importlib.util.spec_from_file_location("custom_md_fma", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    custom_md_fma = load_custom_md_fma()

    model_config = load_json(args.model_config)
    sample_size = int(model_config["sample_size"])
    sample_rate = int(model_config["sample_rate"])
    seconds_total = float(sample_size / sample_rate)

    candidate_rows = load_candidate_rows(args)
    if len(candidate_rows) < args.num_prompts:
        raise ValueError(
            f"Requested num_prompts={args.num_prompts}, but only {len(candidate_rows)} candidate rows are available."
        )

    prompts = []
    for prompt_id, row in enumerate(candidate_rows[: args.num_prompts]):
        info = {
            "path": row["path"],
            "relpath": row.get("relpath") or row["path"],
        }
        prompt = custom_md_fma.get_custom_metadata(info, None).get("prompt", "")
        prompts.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "seconds_start": 0.0,
                "seconds_total": seconds_total,
                "source_path": row["path"],
                "source_relpath": row.get("relpath") or row["path"],
                "source_pool_index": int(row["pool_index"]),
                "source_global_index": int(row["global_index"]),
                "source_track_id": int(row["track_id"]) if row.get("track_id") else None,
            }
        )

    output_path = Path(args.output_path)
    save_json(
        output_path,
        {
            "seed": args.seed,
            "num_prompts": args.num_prompts,
            "candidate_pool_size": args.candidate_pool_size,
            "sample_rate": sample_rate,
            "sample_size": sample_size,
            "seconds_total": seconds_total,
            "prompts": prompts,
        },
    )
    print(f"Saved prompt JSON to {output_path}")


if __name__ == "__main__":
    main()
