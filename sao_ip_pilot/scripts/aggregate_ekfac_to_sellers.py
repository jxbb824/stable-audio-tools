#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch

from common import candidate_path_keys, read_csv_rows, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate sample-level EKFAC scores to seller-level a_hat.")
    parser.add_argument("--score-path", default="sao_ip_pilot/outputs/ekfac_attribution/scores_train_x_query.pt")
    parser.add_argument("--train-axis-manifest", default="sao_ip_pilot/outputs/ekfac_attribution/train_axis_manifest.csv")
    parser.add_argument("--query-axis-manifest", default="sao_ip_pilot/outputs/ekfac_attribution/query_axis_manifest.csv")
    parser.add_argument("--category-manifest", default="sao_ip_pilot/outputs/selection/category_manifest.csv")
    parser.add_argument("--output-path", default="sao_ip_pilot/outputs/a_hat_ekfac.csv")
    parser.add_argument("--score-column", default="a_hat_ekfac")
    parser.add_argument("--aggregation", choices=["sum", "mean"], default="sum")
    return parser.parse_args()


def prompt_id_from_query_row(row: dict[str, str]) -> int | str:
    value = row.get("prompt_id")
    if value not in (None, ""):
        return int(value) if str(value).isdigit() else value
    filename = row.get("filename") or Path(row.get("query_id", "")).name
    match = re.search(r"query_(\d+)", filename)
    return int(match.group(1)) if match else filename


def build_seller_lookup(category_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for row in category_rows:
        for key in candidate_path_keys(row.get("path", ""), row.get("relpath", "")):
            lookup.setdefault(key, row)
    return lookup


def main() -> None:
    args = parse_args()
    score_path = resolve_repo_path(args.score_path)
    output_path = resolve_repo_path(args.output_path)

    scores = torch.load(score_path, map_location="cpu").float()
    if scores.ndim != 2:
        raise ValueError(f"Expected [train, query] score tensor, got {tuple(scores.shape)}")

    train_rows = read_csv_rows(args.train_axis_manifest)
    query_rows = read_csv_rows(args.query_axis_manifest)
    category_rows = read_csv_rows(args.category_manifest)
    if len(train_rows) != scores.shape[0]:
        raise ValueError("train_axis_manifest length does not match score tensor")
    if len(query_rows) != scores.shape[1]:
        raise ValueError("query_axis_manifest length does not match score tensor")

    seller_lookup = build_seller_lookup(category_rows)
    train_seller_ids: list[str] = []
    for row in train_rows:
        matched = None
        for key in candidate_path_keys(row.get("path", ""), row.get("relpath", "")):
            if key in seller_lookup:
                matched = seller_lookup[key]
                break
        if matched is None:
            raise KeyError(f"Could not map train row to seller: {row}")
        train_seller_ids.append(matched["seller_id"])

    seller_ids = sorted(set(train_seller_ids), key=lambda sid: int(sid.rsplit("_", 1)[1]))
    prompt_ids = [prompt_id_from_query_row(row) for row in query_rows]

    rows = []
    for seller_id in seller_ids:
        indices = [idx for idx, sid in enumerate(train_seller_ids) if sid == seller_id]
        seller_scores = scores[indices, :]
        if args.aggregation == "sum":
            values = seller_scores.sum(dim=0)
        else:
            values = seller_scores.mean(dim=0)
        for query_index, prompt_id in enumerate(prompt_ids):
            rows.append(
                {
                    "seller_id": seller_id,
                    "prompt_id": prompt_id,
                    "query_index": query_index,
                    args.score_column: float(values[query_index]),
                    "num_train_examples": len(indices),
                    "aggregation": args.aggregation,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[aggregate] wrote {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
