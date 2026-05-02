#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, default_data_collator

PILOT_ROOT = Path(__file__).resolve().parents[1]
if str(PILOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PILOT_ROOT))

from anticipation.vocab import AUTOREGRESS
from common import ensure_dir, read_csv_rows, resolve_repo_path, set_all_seeds


class TextDataset(Dataset):
    def __init__(self, file_path: str | Path, max_length: int = 1024, num_samples: int | None = None, is_generated: bool = True):
        self.examples: list[str] = []
        self.max_length = max_length
        self.is_generated = is_generated
        with resolve_repo_path(file_path).open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if num_samples is not None and idx >= num_samples:
                    break
                text = line.strip()
                if text:
                    self.examples.append(text)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.is_generated:
            arr = np.fromstring(self.examples[idx], dtype=int, sep=" ")
            input_ids = np.concatenate([np.array([AUTOREGRESS], dtype=int), arr]) if arr.size else np.array([AUTOREGRESS], dtype=int)
        else:
            input_ids = np.fromstring(" ".join(self.examples[idx].split()[:-1]), dtype=int, sep=" ")
        input_ids = input_ids[: self.max_length]
        tensor = torch.tensor(input_ids, dtype=torch.long)
        return {"input_ids": tensor, "labels": tensor, "attention_mask": torch.ones_like(tensor)}


def model_dir(path: str | Path) -> Path:
    path = resolve_repo_path(path)
    nested = path / "full_model"
    return nested if nested.is_dir() else path


def score_model(model_path: Path, dataloader: DataLoader, device: torch.device) -> list[float]:
    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager").to(device)
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch_size = int(batch["input_ids"].shape[0])
            for row_idx in range(batch_size):
                outputs = model(
                    batch["input_ids"][row_idx : row_idx + 1].to(device),
                    attention_mask=batch["attention_mask"][row_idx : row_idx + 1].to(device),
                    labels=batch["labels"][row_idx : row_idx + 1].to(device),
                )
                losses.append(float(outputs.loss.detach().cpu()))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute leave-artist-out NLL ground truth.")
    parser.add_argument("--queries-file", default="anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt")
    parser.add_argument("--seller-manifest", default="anticipation_ip_pilot/outputs/selection/seller_manifest.csv")
    parser.add_argument("--full-model-dir", default="anticipation_ip_pilot/outputs/models/full")
    parser.add_argument("--loo-root", default="anticipation_ip_pilot/outputs/models/loo")
    parser.add_argument("--output-dir", default="anticipation_ip_pilot/outputs")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_all_seeds(args.seed)

    output_dir = ensure_dir(args.output_dir)
    sellers = read_csv_rows(args.seller_manifest)
    dataset = TextDataset(args.queries_file, num_samples=args.num_prompts, is_generated=True)
    dataloader = DataLoader(dataset, collate_fn=default_data_collator, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_losses = score_model(model_dir(args.full_model_dir), dataloader, device)
    full_rows = [
        {"prompt_id": f"query_{idx:04d}", "query_index": idx, "loss_full": loss}
        for idx, loss in enumerate(full_losses)
    ]
    pd.DataFrame(full_rows).to_csv(output_dir / "full_losses.csv", index=False)

    loo_rows = []
    astar_rows = []
    loo_root = resolve_repo_path(args.loo_root)
    for seller in sellers:
        seller_id = seller["seller_id"]
        seller_index = int(seller["seller_index"])
        losses = score_model(model_dir(loo_root / seller_id), dataloader, device)
        for query_index, (loss_loo, loss_full) in enumerate(zip(losses, full_losses)):
            prompt_id = f"query_{query_index:04d}"
            loo_rows.append(
                {
                    "seller_id": seller_id,
                    "seller_index": seller_index,
                    "prompt_id": prompt_id,
                    "query_index": query_index,
                    "loss_loo": loss_loo,
                }
            )
            astar_rows.append(
                {
                    "seller_id": seller_id,
                    "seller_index": seller_index,
                    "prompt_id": prompt_id,
                    "query_index": query_index,
                    "a_star": loss_loo - loss_full,
                    "loss_loo": loss_loo,
                    "loss_full": loss_full,
                }
            )

    pd.DataFrame(loo_rows).to_csv(output_dir / "loo_losses.csv", index=False)
    pd.DataFrame(astar_rows).to_csv(output_dir / "a_star.csv", index=False)
    print(f"[groundtruth] wrote {output_dir / 'a_star.csv'}")


if __name__ == "__main__":
    main()
