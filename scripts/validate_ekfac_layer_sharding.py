#!/usr/bin/env python3
"""Compare EK-FAC layer shards against repeat-run numerical variation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(order.size, dtype=np.float64)
    return ranks


def _comparison(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    difference = second.astype(np.float64) - first.astype(np.float64)
    scale = max(float(np.linalg.norm(first)), np.finfo(np.float64).eps)
    max_scale = max(float(np.max(np.abs(first))), np.finfo(np.float64).eps)
    first_flat = first.astype(np.float64).ravel()
    second_flat = second.astype(np.float64).ravel()
    spearman = float(np.corrcoef(_rankdata(first_flat), _rankdata(second_flat))[0, 1])
    return {
        "relative_frobenius_error": float(np.linalg.norm(difference) / scale),
        "normalized_max_absolute_error": float(np.max(np.abs(difference)) / max_scale),
        "spearman": spearman,
        "sign_agreement": float(np.mean(np.signbit(first_flat) == np.signbit(second_flat))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-a", type=Path, required=True)
    parser.add_argument("--reference-b", type=Path, required=True)
    parser.add_argument("--merged", type=Path, required=True)
    parser.add_argument("--shape", type=int, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shape = tuple(args.shape)
    arrays = {
        "reference_a": np.memmap(args.reference_a, dtype=np.float32, mode="r", shape=shape),
        "reference_b": np.memmap(args.reference_b, dtype=np.float32, mode="r", shape=shape),
        "merged": np.memmap(args.merged, dtype=np.float32, mode="r", shape=shape),
    }
    repeat = _comparison(arrays["reference_a"], arrays["reference_b"])
    shard = _comparison(arrays["reference_a"], arrays["merged"])
    # A one-percent matrix-level discrepancy is conservative for stochastic,
    # four-bit-compressed scores; ranking and signs must remain essentially intact.
    thresholds = {
        "relative_frobenius_error": max(0.01, 3.0 * repeat["relative_frobenius_error"]),
        "minimum_spearman": 0.999,
        "minimum_sign_agreement": 0.97,
    }
    passed = (
        shard["relative_frobenius_error"] <= thresholds["relative_frobenius_error"]
        and shard["spearman"] >= thresholds["minimum_spearman"]
        and shard["sign_agreement"] >= thresholds["minimum_sign_agreement"]
    )
    summary = {
        "status": "pass" if passed else "fail",
        "shape": list(shape),
        "repeat_run": repeat,
        "layer_sharded": shard,
        "thresholds": thresholds,
    }
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not passed:
        raise RuntimeError(f"EK-FAC layer-sharding validation failed: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
