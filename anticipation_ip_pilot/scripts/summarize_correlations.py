#!/usr/bin/env python3
"""Print mean/median Pearson and Spearman correlations for each attribution method."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def summarize_csv(csv_path: Path, method: str) -> dict:
    df = pd.read_csv(csv_path).dropna(subset=["pearson", "spearman"])
    return {
        "method": method,
        "num_queries": len(df),
        "mean_pearson": float(np.mean(df["pearson"])),
        "median_pearson": float(np.median(df["pearson"])),
        "mean_spearman": float(np.mean(df["spearman"])),
        "median_spearman": float(np.median(df["spearman"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize query-level correlations.")
    parser.add_argument(
        "--output-dir",
        default="anticipation_ip_pilot/outputs/selected",
        help="Directory containing query_level_corr_*.csv files.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["logra", "trak"],
        help="Method tags to look for (default: logra trak).",
    )
    args = parser.parse_args()

    output_dir = resolve(args.output_dir)
    results: list[dict] = []

    for tag in args.methods:
        csv_path = output_dir / f"query_level_corr_{tag}.csv"
        json_path = output_dir / f"query_level_corr_{tag}_summary.json"

        if json_path.exists():
            with json_path.open() as fh:
                summary = json.load(fh)
            results.append(summary)
        elif csv_path.exists():
            summary = summarize_csv(csv_path, tag)
            results.append(summary)
            with json_path.open("w") as fh:
                json.dump(summary, fh, indent=2)
            print(f"[saved] {json_path}")
        else:
            print(f"[skip] no data found for method '{tag}'")

    if not results:
        print("No results found.")
        return

    # ── print table ──────────────────────────────────────────────────────────
    col_w = 12
    header = f"{'Method':<20} {'Queries':>7}  {'Mean Pearson':>{col_w}}  {'Median Pearson':>{col_w}}  {'Mean Spearman':>{col_w}}  {'Median Spearman':>{col_w}}"
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['method']:<20} {r['num_queries']:>7}  "
            f"{r['mean_pearson']:>{col_w}.4f}  {r['median_pearson']:>{col_w}.4f}  "
            f"{r['mean_spearman']:>{col_w}.4f}  {r['median_spearman']:>{col_w}.4f}"
        )
    print()


if __name__ == "__main__":
    main()
