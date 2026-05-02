#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd
import torch

from common import read_csv_rows, resolve_repo_path


def parse_indices(value: str) -> list[int]:
    return [int(item) for item in value.replace(",", " ").split() if item]


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate train-example attribution scores to artist sellers.")
    parser.add_argument("--score-path", required=True)
    parser.add_argument("--seller-manifest", default="anticipation_ip_pilot/outputs/selection/seller_manifest.csv")
    parser.add_argument("--query-manifest", default="anticipation_ip_pilot/outputs/queries/query_manifest.csv")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--score-column", required=True)
    parser.add_argument("--score-sign", type=float, default=1.0)
    args = parser.parse_args()

    sellers = read_csv_rows(args.seller_manifest)
    queries = read_csv_rows(args.query_manifest)
    score = torch.load(resolve_repo_path(args.score_path), map_location="cpu", weights_only=False).float()
    n_train = max(max(parse_indices(row["train_indices"])) for row in sellers) + 1
    n_query = len(queries)
    if score.shape[0] < n_train and score.shape[1] >= n_train:
        score = score.T
    if score.shape[1] < n_query:
        raise ValueError(f"Score tensor has only {score.shape[1]} query columns; need {n_query}.")

    rows = []
    for seller in sellers:
        indices = parse_indices(seller["train_indices"])
        seller_scores = args.score_sign * score[indices, :n_query].sum(dim=0)
        for query_index, value in enumerate(seller_scores.tolist()):
            rows.append(
                {
                    "seller_id": seller["seller_id"],
                    "seller_index": int(seller["seller_index"]),
                    "prompt_id": queries[query_index]["prompt_id"],
                    "query_index": query_index,
                    args.score_column: float(value),
                    "num_train_examples": len(indices),
                }
            )

    output_path = resolve_repo_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[aggregate] wrote {output_path}")


if __name__ == "__main__":
    main()
