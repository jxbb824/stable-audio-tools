#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

import pandas as pd

from common import resolve_repo_path


METHODS = {
    "logra_influence": ("anticipation_ip_pilot/outputs/a_hat_logra.csv", "a_hat_logra"),
    "trak_influence": ("anticipation_ip_pilot/outputs/a_hat_trak.csv", "a_hat_trak"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute across-seller query-level correlation.")
    parser.add_argument("--method", choices=sorted(METHODS), required=True)
    parser.add_argument("--a-star-path", default="anticipation_ip_pilot/outputs/a_star.csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    a_path, col = METHODS[args.method]
    a_star = pd.read_csv(resolve_repo_path(args.a_star_path))
    a_hat = pd.read_csv(resolve_repo_path(a_path))
    merged = a_star.merge(a_hat, on=["seller_id", "prompt_id"], how="inner")
    rows = []
    for query_index, sub in merged.groupby("query_index_x"):
        rows.append(
            {
                "query_index": int(query_index),
                "prompt_id": sub["prompt_id"].iloc[0],
                "n_sellers": int(len(sub)),
                "pearson": float(sub["a_star"].corr(sub[col], method="pearson")),
                "spearman": float(sub["a_star"].corr(sub[col], method="spearman")),
            }
        )
    out = pd.DataFrame(rows)
    output_csv = resolve_repo_path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    summary = {
        "method": args.method,
        "num_queries": int(len(out)),
        "mean_pearson": float(out["pearson"].mean()),
        "median_pearson": float(out["pearson"].median()),
        "mean_spearman": float(out["spearman"].mean()),
        "median_spearman": float(out["spearman"].median()),
    }
    summary_path = resolve_repo_path(args.summary_json)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(f"[corr] wrote {output_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
