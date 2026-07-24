#!/usr/bin/env python3
"""Compute LDS (mean per-query Spearman) for subset-model ground truth.

Adapted from ../anticipation/spearman.py, generalized with CLI and pool-index mapping.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import combine_pvalues, spearmanr

from common import resolve_repo_path, write_json


def read_index_rows(path: Path) -> list[int]:
    values: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            for item in row:
                item = item.strip()
                if item:
                    values.append(int(item))
    return values


def load_pool_globals(path: Path) -> list[int]:
    text = path.read_text(encoding="utf-8")
    return [int(x) for x in text.replace("\n", ",").split(",") if x.strip()]


def aggregate_subset_scores(
    score_matrix: torch.Tensor,
    subset_indices: list[list[int]],
    *,
    aggregation: str,
) -> torch.Tensor:
    if aggregation == "included":
        rows = [score_matrix[idx, :].sum(dim=0) for idx in subset_indices]
        return torch.stack(rows, dim=0)
    if aggregation == "excluded":
        total = score_matrix.sum(dim=0)
        rows = [total - score_matrix[idx, :].sum(dim=0) for idx in subset_indices]
        return torch.stack(rows, dim=0)
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def compute_lds(
    approx_matrix: torch.Tensor,
    groundtruth_matrix: torch.Tensor,
) -> tuple[float, float, list[dict]]:
    if approx_matrix.shape != groundtruth_matrix.shape:
        raise ValueError(
            f"Shape mismatch: approx {tuple(approx_matrix.shape)} vs gt {tuple(groundtruth_matrix.shape)}"
        )

    stats: list[float] = []
    pvalues: list[float] = []
    rows: list[dict] = []

    for query_idx in range(groundtruth_matrix.shape[1]):
        x = approx_matrix[:, query_idx].numpy()
        y = groundtruth_matrix[:, query_idx].numpy()
        stat, pval = spearmanr(x, y)
        valid = not (np.isnan(stat) or np.isnan(pval))
        rows.append(
            {
                "query_index": query_idx,
                "spearman": "" if np.isnan(stat) else float(stat),
                "pvalue": "" if np.isnan(pval) else float(pval),
                "valid": int(valid),
            }
        )
        if valid:
            stats.append(float(stat))
            pvalues.append(max(float(pval), np.finfo(float).tiny))

    mean_lds = float(np.mean(stats)) if stats else float("nan")
    fisher_p = (
        float(combine_pvalues(pvalues, method="fisher")[1])
        if pvalues
        else float("nan")
    )
    if not np.isnan(fisher_p):
        fisher_p = max(fisher_p, np.finfo(float).tiny)
    return mean_lds, fisher_p, rows


def build_subset_local_indices(
    subsets_root: Path,
    num_subsets: int,
    global_to_local: dict[int, int],
    *,
    index_filename: str,
) -> list[list[int]]:
    subset_local: list[list[int]] = []
    for subset_id in range(num_subsets):
        index_path = subsets_root / str(subset_id) / index_filename
        if not index_path.exists():
            raise FileNotFoundError(f"Missing subset index file: {index_path}")
        globals_ = read_index_rows(index_path)
        locals_ = [global_to_local[g] for g in globals_ if g in global_to_local]
        if not locals_:
            raise ValueError(f"Subset {subset_id} has no indices in the attribution pool.")
        subset_local.append(locals_)
    return subset_local


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute LDS for a train×query attribution matrix.")
    parser.add_argument("--score-path", required=True)
    parser.add_argument("--groundtruth-path", default="anticipation_lds_pilot/outputs/lds/gt_gen.pt")
    parser.add_argument("--subsets-root", default="anticipation_lds_pilot/outputs/models/subsets")
    parser.add_argument("--pool-indices-file", default="anticipation_ip_pilot/outputs/selection/seller_indices.txt")
    parser.add_argument("--subset-index-filename", default="train_index.csv")
    parser.add_argument("--num-subsets", type=int, default=30)
    parser.add_argument(
        "--subset-aggregation",
        choices=("included", "excluded"),
        default="included",
        help="Sum attribution over included subset rows (default) or over the complement.",
    )
    parser.add_argument("--score-label", default="method")
    parser.add_argument("--output-dir", default="anticipation_lds_pilot/outputs/lds")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    score_path = resolve_repo_path(args.score_path)
    gt_path = resolve_repo_path(args.groundtruth_path)
    subsets_root = resolve_repo_path(args.subsets_root)
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    score_matrix = torch.load(score_path, map_location="cpu", weights_only=False).float()
    groundtruth_matrix = torch.load(gt_path, map_location="cpu", weights_only=False).float()

    pool_globals = load_pool_globals(resolve_repo_path(args.pool_indices_file))
    global_to_local = {g: i for i, g in enumerate(pool_globals)}

    if score_matrix.shape[0] < len(pool_globals) and score_matrix.shape[1] >= len(pool_globals):
        score_matrix = score_matrix.T

    if score_matrix.shape[0] != len(pool_globals):
        raise ValueError(
            f"Score train axis {score_matrix.shape[0]} does not match pool size {len(pool_globals)}."
        )
    if groundtruth_matrix.shape[0] != args.num_subsets:
        raise ValueError(
            f"Ground truth has {groundtruth_matrix.shape[0]} subsets, expected {args.num_subsets}."
        )
    if score_matrix.shape[1] != groundtruth_matrix.shape[1]:
        raise ValueError(
            f"Query axis mismatch: score {score_matrix.shape[1]} vs gt {groundtruth_matrix.shape[1]}."
        )

    subset_local = build_subset_local_indices(
        subsets_root,
        args.num_subsets,
        global_to_local,
        index_filename=args.subset_index_filename,
    )
    approx_matrix = aggregate_subset_scores(
        score_matrix,
        subset_local,
        aggregation=args.subset_aggregation,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.random_seed)
    random_matrix = torch.rand(score_matrix.shape, generator=generator, dtype=score_matrix.dtype)
    random_approx = aggregate_subset_scores(
        random_matrix,
        subset_local,
        aggregation=args.subset_aggregation,
    )

    primary_lds, primary_fisher, per_query = compute_lds(approx_matrix, groundtruth_matrix)
    random_lds, random_fisher, random_rows = compute_lds(random_approx, groundtruth_matrix)

    stem = Path(args.score_path).stem
    torch.save(approx_matrix, output_dir / f"{stem}_subset_query_scores.pt")

    summary = {
        "score_path": str(score_path),
        "groundtruth_path": str(gt_path),
        "subsets_root": str(subsets_root),
        "pool_size": len(pool_globals),
        "num_subsets": args.num_subsets,
        "num_queries": int(groundtruth_matrix.shape[1]),
        "subset_aggregation": args.subset_aggregation,
        "score_label": args.score_label,
        "mean_spearman_lds": primary_lds,
        "fisher_pvalue": primary_fisher,
        "random_mean_spearman_lds": random_lds,
        "random_fisher_pvalue": random_fisher,
    }
    write_json(output_dir / f"{stem}_lds_summary.json", summary)

    per_query_path = output_dir / f"{stem}_lds_per_query.csv"
    with per_query_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["query_index", "spearman", "pvalue", "valid", "random_spearman", "random_pvalue", "random_valid"],
        )
        writer.writeheader()
        for idx, row in enumerate(per_query):
            rnd = random_rows[idx]
            writer.writerow(
                {
                    "query_index": row["query_index"],
                    "spearman": row["spearman"],
                    "pvalue": row["pvalue"],
                    "valid": row["valid"],
                    "random_spearman": rnd["spearman"],
                    "random_pvalue": rnd["pvalue"],
                    "random_valid": rnd["valid"],
                }
            )

    print(f"{args.score_label} LDS={primary_lds:.6f}  Fisher p={primary_fisher:.6e}")
    print(f"Random LDS={random_lds:.6f}  Fisher p={random_fisher:.6e}")
    print(f"[lds] wrote {output_dir / f'{stem}_lds_summary.json'}")


if __name__ == "__main__":
    main()
