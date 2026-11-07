#!/usr/bin/env python3

"""Merge contiguous D-TRAK feature shards with strict metadata validation."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


INVARIANT_FIELDS = (
    "proj_dim",
    "used_dim",
    "K",
    "t_strategy",
    "num_train_steps",
    "f",
    "cfg_dropout_prob",
    "model_id",
    "param_regex",
    "grad_dim",
    "pre_encoded",
    "projection_mode",
    "projection_chunk_numel",
    "projection_kernel_batch_size",
    "projection_chunk_sizes",
    "projection_chunk_seeds",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-paths", nargs="+", required=True)
    parser.add_argument("--meta-paths", nargs="+", required=True)
    parser.add_argument("--output-feature-path", required=True)
    parser.add_argument("--output-ids-path", default=None)
    parser.add_argument("--output-meta-path", default=None)
    return parser.parse_args()


def load_meta(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def read_ids(path: str) -> List[str]:
    return Path(path).read_text().splitlines()


def main() -> None:
    args = parse_args()
    if len(args.feature_paths) != len(args.meta_paths):
        raise ValueError("--feature-paths and --meta-paths must have equal length.")

    records: List[Tuple[int, str, str, Dict[str, Any]]] = []
    for feature_path, meta_path in zip(args.feature_paths, args.meta_paths):
        meta = load_meta(meta_path)
        shard = meta.get("shard", {})
        if "shard_index" not in shard:
            raise ValueError(f"Missing shard metadata in {meta_path}.")
        records.append((int(shard["shard_index"]), feature_path, meta_path, meta))
    records.sort(key=lambda record: record[0])

    reference = records[0][3]
    num_shards = int(reference["shard"]["num_shards"])
    if len(records) != num_shards:
        raise ValueError(f"Expected {num_shards} shards, received {len(records)}.")
    if [record[0] for record in records] != list(range(num_shards)):
        raise ValueError("Shard indices must be exactly 0..num_shards-1.")

    global_num_examples = int(reference["shard"]["global_num_examples"])
    proj_dim = int(reference["proj_dim"])
    expected_start = 0
    all_ids: List[str] = []

    output_feature_path = Path(args.output_feature_path)
    output_feature_path.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(
        str(output_feature_path),
        dtype=np.float32,
        mode="w+",
        shape=(global_num_examples, proj_dim),
    )

    source_shards = []
    for shard_index, feature_path, meta_path, meta in records:
        for field in INVARIANT_FIELDS:
            if meta.get(field) != reference.get(field):
                raise ValueError(
                    f"Shard {shard_index} differs in {field}: "
                    f"{meta.get(field)!r} != {reference.get(field)!r}."
                )

        shard = meta["shard"]
        start = int(shard["global_start_index"])
        end = int(shard["global_end_index"])
        valid_rows = int(meta["valid_rows"])
        allocated_rows = int(meta["num_rows_allocated"])
        if start != expected_start:
            raise ValueError(
                f"Non-contiguous shard {shard_index}: starts at {start}, expected {expected_start}."
            )
        if end - start != valid_rows or allocated_rows != valid_rows:
            raise ValueError(
                f"Shard {shard_index} row mismatch: range={end-start}, "
                f"valid={valid_rows}, allocated={allocated_rows}."
            )

        block = np.memmap(
            feature_path,
            dtype=np.float32,
            mode="r",
            shape=(allocated_rows, proj_dim),
        )
        output[start:end] = block[:valid_rows]
        ids = read_ids(meta["ids_path"])
        if len(ids) != valid_rows:
            raise ValueError(
                f"Shard {shard_index} has {len(ids)} IDs for {valid_rows} feature rows."
            )
        all_ids.extend(ids)
        source_shards.append(
            {
                "shard_index": shard_index,
                "feature_path": feature_path,
                "meta_path": meta_path,
                "global_start_index": start,
                "global_end_index": end,
            }
        )
        expected_start = end

    if expected_start != global_num_examples:
        raise ValueError(
            f"Merged range ends at {expected_start}, expected {global_num_examples}."
        )
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Merged sample IDs are not unique; refusing ambiguous feature axes.")

    output.flush()
    output_ids_path = Path(
        args.output_ids_path
        if args.output_ids_path is not None
        else str(output_feature_path) + ".ids.txt"
    )
    output_meta_path = Path(
        args.output_meta_path
        if args.output_meta_path is not None
        else str(output_feature_path) + ".meta.json"
    )
    output_ids_path.parent.mkdir(parents=True, exist_ok=True)
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    output_ids_path.write_text("".join(f"{sample_id}\n" for sample_id in all_ids))

    merged_meta = {
        key: value
        for key, value in reference.items()
        if key not in {"feature_path", "ids_path", "num_rows_allocated", "valid_rows", "shard"}
    }
    merged_meta.update(
        {
            "feature_path": str(output_feature_path),
            "ids_path": str(output_ids_path),
            "num_rows_allocated": global_num_examples,
            "valid_rows": global_num_examples,
            "shard": {
                "merged": True,
                "num_shards": num_shards,
                "global_num_examples": global_num_examples,
            },
            "source_shards": source_shards,
        }
    )
    output_meta_path.write_text(json.dumps(merged_meta, indent=2) + "\n")
    print(
        f"Merged {num_shards} shards and {global_num_examples} rows -> {output_feature_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
