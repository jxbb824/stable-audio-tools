#!/usr/bin/env python3
"""Score subset checkpoints on generated queries; save gt_gen.pt for LDS.

Adapted from ../anticipation/groundtruth.py.
Output shape: [num_subsets, num_queries], each entry is log-odds of correctness.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from dattri.func.utils import flatten_func, flatten_params
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, default_data_collator

PILOT_ROOT = Path(__file__).resolve().parents[1]
IP_PILOT_ROOT = PILOT_ROOT.parent / "anticipation_ip_pilot"
for root in (PILOT_ROOT, IP_PILOT_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from anticipation.vocab import AUTOREGRESS
from common import resolve_repo_path, set_all_seeds


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute subset-model ground truth for LDS.")
    parser.add_argument("--queries-file", default="anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt")
    parser.add_argument("--subsets-root", default="anticipation_lds_pilot/outputs/models/subsets")
    parser.add_argument("--output-path", default="anticipation_lds_pilot/outputs/lds/gt_gen.pt")
    parser.add_argument("--num-subsets", type=int, default=30)
    parser.add_argument("--num-queries", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_all_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    eval_dataset = TextDataset(args.queries_file, num_samples=args.num_queries, is_generated=True)
    if len(eval_dataset) == 0:
        raise RuntimeError(f"Empty query dataset: {args.queries_file}")

    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=args.batch_size,
        shuffle=False,
    )

    subsets_root = resolve_repo_path(args.subsets_root)
    result_list: list[torch.Tensor] = []

    for subset_id in range(args.num_subsets):
        checkpoint_path = subsets_root / str(subset_id)
        if not checkpoint_path.is_dir():
            print(f"Warning: missing subset checkpoint {checkpoint_path}, skipping.")
            continue

        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, attn_implementation="eager").to(device)
        model.eval()
        params = {k: p for k, p in model.named_parameters() if p.requires_grad}
        if not params:
            print(f"Warning: no trainable params in {checkpoint_path}, skipping.")
            continue

        @flatten_func(model)
        def f(flat_params, batch):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = torch.func.functional_call(
                model,
                flat_params,
                (input_ids,),
                kwargs={"attention_mask": attention_mask, "labels": labels},
            )
            logp = -outputs.loss
            return logp - torch.log(1 - torch.exp(logp) + 1e-7)

        per_batch = []
        for batch in eval_dataloader:
            per_batch.append(f(flatten_params(params), batch).detach().cpu())
        result_list.append(torch.stack(per_batch))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[gt] scored subset {subset_id}")

    if not result_list:
        raise RuntimeError("No subset checkpoints were scored.")

    final_result = torch.stack(result_list)
    output_path = resolve_repo_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_result, output_path)
    print(f"[gt] wrote {output_path}")
    print(f"[gt] shape={tuple(final_result.shape)}")


if __name__ == "__main__":
    main()
