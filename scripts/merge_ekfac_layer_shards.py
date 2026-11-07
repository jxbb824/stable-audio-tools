#!/usr/bin/env python3
"""Validate and sum additive EK-FAC score shards over disjoint layer blocks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-linear-start", type=int, required=True)
    parser.add_argument("--expected-linear-end", type=int, required=True)
    parser.add_argument("--row-chunk-size", type=int, default=32)
    args = parser.parse_args()

    if len(args.shard_dirs) < 2:
        raise ValueError("At least two layer shards are required.")
    metadata = [json.loads((path / "attribution_metadata.json").read_text()) for path in args.shard_dirs]
    ranges = [item["members"][0]["linear_layer_range"] for item in metadata]
    ranges.sort(key=lambda item: item["start"])
    cursor = args.expected_linear_start
    totals = {int(item["total"]) for item in ranges}
    for item in ranges:
        if int(item["start"]) != cursor:
            raise RuntimeError(f"Layer shards have a gap or overlap at {cursor}: {ranges}")
        cursor = int(item["end"])
    if cursor != args.expected_linear_end or len(totals) != 1:
        raise RuntimeError(f"Layer shards do not cover the expected interval: {ranges}")

    train_manifests = [(path / "train_axis_manifest.csv").read_bytes() for path in args.shard_dirs]
    query_manifests = [(path / "query_axis_manifest.csv").read_bytes() for path in args.shard_dirs]
    if len(set(train_manifests)) != 1 or len(set(query_manifests)) != 1:
        raise RuntimeError("Layer shards do not have identical train/query axes.")

    score_meta = [
        json.loads((path / "scores_query_x_train.memmap.meta.json").read_text())
        for path in args.shard_dirs
    ]
    shapes = {tuple(item["shape"]) for item in score_meta}
    if len(shapes) != 1:
        raise RuntimeError(f"Layer-shard score shapes differ: {shapes}")
    shape = next(iter(shapes))
    sources = [
        np.memmap(path / "scores_query_x_train.memmap", dtype=np.float32, mode="r", shape=shape)
        for path in args.shard_dirs
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "scores_query_x_train.memmap"
    output = np.memmap(output_path, dtype=np.float32, mode="w+", shape=shape)
    for start in range(0, shape[0], args.row_chunk_size):
        end = min(shape[0], start + args.row_chunk_size)
        block = np.zeros((end - start, shape[1]), dtype=np.float64)
        for source in sources:
            block += source[start:end]
        output[start:end] = block.astype(np.float32)
    output.flush()
    del output

    merged = np.memmap(output_path, dtype=np.float32, mode="r", shape=shape)
    torch.save(torch.from_numpy(np.asarray(merged).copy()).T.contiguous(), args.output_dir / "scores_train_x_query.pt")
    shutil.copy2(args.shard_dirs[0] / "train_axis_manifest.csv", args.output_dir / "train_axis_manifest.csv")
    shutil.copy2(args.shard_dirs[0] / "query_axis_manifest.csv", args.output_dir / "query_axis_manifest.csv")
    (args.output_dir / "scores_query_x_train.memmap.meta.json").write_text(
        json.dumps(
            {
                "output_score_path": str(output_path.resolve()),
                "shape": list(shape),
                "layout": "query_x_train",
                "method": "ekfac_influence_layer_sharded",
                "layer_ranges": ranges,
                "source_shards": [str(path.resolve()) for path in args.shard_dirs],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"merged={len(sources)} shape={shape} ranges={ranges} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
