#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, default_data_collator

PILOT_ROOT = Path(__file__).resolve().parents[1]
if str(PILOT_ROOT) not in sys.path:
    sys.path.insert(0, str(PILOT_ROOT))

from anticipation.vocab import AUTOREGRESS
from dattri.algorithm.trak import TRAKAttributor
from dattri.task import AttributionTask

try:
    from dattri.params.projection import TRAKProjectionParams
except ModuleNotFoundError:
    TRAKProjectionParams = None


class TextDataset(Dataset):
    def __init__(self, file_path: str, max_length: int = 1024, num_samples: int | None = None, is_generated: bool = False):
        self.examples: list[str] = []
        self.max_length = max_length
        self.is_generated = is_generated
        with open(file_path, "r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if num_samples is not None and idx >= num_samples:
                    break
                self.examples.append(line.strip())

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
    parser = argparse.ArgumentParser(description="Calculate symbolic TRAK attribution scores.")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--valid_file", required=True)
    parser.add_argument("--checkpoint_dir", required=True, help="Directory with full_model and 0..K-1 checkpoint dirs.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_checkpoints", type=int, default=5)
    parser.add_argument("--proj_dim", type=int, default=8192)
    parser.add_argument("--proj_max_batch_size", type=int, default=40)
    parser.add_argument("--proj_type", default="normal", choices=["normal", "rademacher", "random_mask", "sjlt"])
    parser.add_argument("--layer_name", action="append", default=None)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--num_prompts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid_is_generated", action="store_true")
    parser.add_argument("--output_filename", default="score_TRAK.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = TextDataset(args.train_file)
    eval_dataset = TextDataset(args.valid_file, num_samples=args.num_prompts, is_generated=args.valid_is_generated)
    train_dataloader = DataLoader(train_dataset, collate_fn=default_data_collator, batch_size=args.batch_size, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size, shuffle=False)

    checkpoints = [os.path.join(args.checkpoint_dir, str(i)) for i in range(args.num_checkpoints)]
    model_path = os.path.join(args.checkpoint_dir, "full_model")
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Missing full model directory: {model_path}")
    for checkpoint in checkpoints:
        if not os.path.isdir(checkpoint):
            raise FileNotFoundError(f"Missing TRAK checkpoint directory: {checkpoint}")

    model = AutoModelForCausalLM.from_pretrained(model_path, attn_implementation="eager").to(device)
    model.eval()

    def batch_inputs(batch):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)
            labels = labels.unsqueeze(0)
        return input_ids, attention_mask, labels

    def f(params, batch):
        input_ids, attention_mask, labels = batch_inputs(batch)
        outputs = torch.func.functional_call(
            model,
            params,
            input_ids,
            kwargs={
                "attention_mask": attention_mask,
                "labels": labels,
            },
        )
        p = torch.exp(-outputs.loss).clamp(max=1 - 1e-7)
        return torch.log(p) - torch.log1p(-p)

    def m(params, batch):
        input_ids, attention_mask, labels = batch_inputs(batch)
        outputs = torch.func.functional_call(
            model,
            params,
            input_ids,
            kwargs={
                "attention_mask": attention_mask,
                "labels": labels,
            },
        )
        return torch.exp(-outputs.loss)

    def checkpoints_load_func(_, checkpoint):
        loaded = AutoModelForCausalLM.from_pretrained(checkpoint, attn_implementation="eager").to(device)
        loaded.eval()
        return loaded

    task = AttributionTask(
        loss_func=f,
        model=model,
        checkpoints=checkpoints,
        checkpoints_load_func=checkpoints_load_func,
    )
    projector_kwargs = {
        "device": device,
        "proj_dim": args.proj_dim,
        "proj_type": args.proj_type,
        "proj_max_batch_size": args.proj_max_batch_size,
    }
    if TRAKProjectionParams is None:
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device=device,
            projector_kwargs=projector_kwargs,
            layer_name=args.layer_name,
            regularization=args.regularization,
        )
    else:
        projector_kwargs.pop("device")
        attributor = TRAKAttributor(
            task=task,
            correct_probability_func=m,
            device=device,
            proj_params=TRAKProjectionParams(**projector_kwargs),
            layer_name=args.layer_name,
            regularization=args.regularization,
        )
    if args.layer_name:
        print(f"Using TRAK layer_name={args.layer_name}")
    print(f"Using TRAK projection type={args.proj_type}, proj_dim={args.proj_dim}")

    print("Caching train dataloader...")
    attributor.cache(train_dataloader)
    print("Attributing scores...")
    with torch.no_grad():
        score = attributor.attribute(eval_dataloader)

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    torch.save(score, output_path)
    print(f"Results saved to {output_path}")
    print(f"Score shape: {score.shape}")


if __name__ == "__main__":
    main()
