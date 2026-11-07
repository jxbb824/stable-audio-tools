#!/usr/bin/env python3
"""Validate score axes and seller-level outputs for an SAO baseline."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import torch


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _prompt_id(row: dict[str, str]) -> str:
    if row.get("prompt_id") not in (None, ""):
        return str(row["prompt_id"])
    filename = row.get("filename") or Path(row.get("query_id", "")).name
    match = re.search(r"query_(\d+)", filename)
    return str(int(match.group(1))) if match else filename


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["random", "clap", "fingerprint"], required=True)
    parser.add_argument("--score-path", type=Path, required=True)
    parser.add_argument("--train-axis-manifest", type=Path, required=True)
    parser.add_argument("--query-axis-manifest", type=Path, required=True)
    parser.add_argument("--seller-manifest", type=Path, required=True)
    parser.add_argument("--a-hat-path", type=Path, required=True)
    parser.add_argument("--i-hat-path", type=Path, required=True)
    args = parser.parse_args()

    train_rows = _rows(args.train_axis_manifest)
    query_rows = _rows(args.query_axis_manifest)
    seller_rows = _rows(args.seller_manifest)
    a_hat_rows = _rows(args.a_hat_path)
    i_hat_rows = _rows(args.i_hat_path)
    seller_ids = {row["seller_id"] for row in seller_rows}
    prompt_ids = {_prompt_id(row) for row in query_rows}

    score = torch.load(args.score_path, map_location="cpu", weights_only=False)
    expected_shape = (len(train_rows), len(query_rows))
    if not torch.is_tensor(score) or tuple(score.shape) != expected_shape:
        raise RuntimeError(f"score shape mismatch: got {getattr(score, 'shape', None)}, expected {expected_shape}")
    if not torch.isfinite(score).all():
        raise RuntimeError("score matrix contains NaN or Inf")
    lower, upper = {
        "random": (0.0, 1.0),
        "clap": (-1.0001, 1.0001),
        "fingerprint": (0.0, 1.0),
    }[args.method]
    if float(score.min()) < lower or float(score.max()) > upper:
        raise RuntimeError(
            f"{args.method} scores outside [{lower}, {upper}]: "
            f"min={float(score.min())}, max={float(score.max())}"
        )

    expected_pairs = {(seller_id, prompt_id) for seller_id in seller_ids for prompt_id in prompt_ids}
    actual_pairs = {(row["seller_id"], str(row["prompt_id"])) for row in a_hat_rows}
    if len(a_hat_rows) != len(expected_pairs) or actual_pairs != expected_pairs:
        raise RuntimeError("a_hat does not contain exactly one row per seller-query pair")
    if len(i_hat_rows) != len(seller_ids) or {row["seller_id"] for row in i_hat_rows} != seller_ids:
        raise RuntimeError("I_hat does not contain exactly one row per seller")
    bad_t = [row for row in i_hat_rows if int(row["T"]) != len(query_rows)]
    if bad_t:
        raise RuntimeError(f"I_hat has incomplete query coverage, e.g. {bad_t[0]}")

    print(
        {
            "status": "pass",
            "method": args.method,
            "score_shape": list(score.shape),
            "score_min": float(score.min()),
            "score_max": float(score.max()),
            "sellers": len(seller_ids),
            "queries": len(query_rows),
            "a_hat_rows": len(a_hat_rows),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
