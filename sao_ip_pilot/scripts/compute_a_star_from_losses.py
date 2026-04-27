#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch

from common import read_csv_rows, read_json, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert full/LOO query losses into seller-level a_star.csv.")
    parser.add_argument("--loss-path", default="sao_ip_pilot/outputs/losses_model_x_query.pt")
    parser.add_argument("--seller-manifest", default="sao_ip_pilot/outputs/selection/seller_manifest.csv")
    parser.add_argument("--prompt-manifest", default="sao_ip_pilot/outputs/generated_queries/prompt_manifest.json")
    parser.add_argument("--output-path", default="sao_ip_pilot/outputs/a_star.csv")
    parser.add_argument("--full-model-id", default="0")
    return parser.parse_args()


def load_text_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prompt_id_from_query_id(query_id: str, manifest_by_filename: dict[str, dict]) -> int | str:
    filename = Path(query_id).name
    record = manifest_by_filename.get(filename, {})
    if "prompt_id" in record:
        return record["prompt_id"]
    match = re.search(r"query_(\d+)", filename)
    return int(match.group(1)) if match else filename


def main() -> None:
    args = parse_args()
    loss_path = resolve_repo_path(args.loss_path)
    model_ids_path = Path(f"{loss_path}.model_ids.txt")
    query_ids_path = Path(f"{loss_path}.query_ids.txt")
    output_path = resolve_repo_path(args.output_path)

    losses = torch.load(loss_path, map_location="cpu")
    if losses.ndim != 2:
        raise ValueError(f"Expected [model, query] loss tensor, got {tuple(losses.shape)}")
    losses = losses.float()

    model_ids = load_text_lines(model_ids_path)
    query_ids = load_text_lines(query_ids_path)
    if len(model_ids) != losses.shape[0]:
        raise ValueError("model_ids length does not match loss tensor")
    if len(query_ids) != losses.shape[1]:
        raise ValueError("query_ids length does not match loss tensor")

    prompt_manifest = read_json(args.prompt_manifest)
    manifest_by_filename = {
        item.get("filename", ""): item for item in prompt_manifest.get("items", [])
    }
    prompt_ids = [prompt_id_from_query_id(query_id, manifest_by_filename) for query_id in query_ids]

    model_index = {model_id: idx for idx, model_id in enumerate(model_ids)}
    if args.full_model_id not in model_index:
        raise KeyError(f"Full model id {args.full_model_id!r} not found in {model_ids}")
    full_losses = losses[model_index[args.full_model_id]]

    sellers = read_csv_rows(args.seller_manifest)
    rows = []
    for seller in sellers:
        seller_index = str(int(seller["seller_index"]))
        seller_id = seller["seller_id"]
        if seller_index not in model_index:
            raise KeyError(f"LOO model id {seller_index!r} missing for {seller_id}")
        loo_losses = losses[model_index[seller_index]]
        a_star = loo_losses - full_losses
        for query_index, prompt_id in enumerate(prompt_ids):
            rows.append(
                {
                    "seller_id": seller_id,
                    "seller_index": int(seller_index),
                    "prompt_id": prompt_id,
                    "query_index": query_index,
                    "loss_full": float(full_losses[query_index]),
                    "loss_loo": float(loo_losses[query_index]),
                    "a_star": float(a_star[query_index]),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"[a_star] wrote {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
