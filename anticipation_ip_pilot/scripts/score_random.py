#!/usr/bin/env python3
"""Create a random train×query attribution score matrix matching the selected pool."""
from __future__ import annotations

import argparse

import torch

from common import resolve_repo_path, set_all_seeds


def main() -> None:
    parser = argparse.ArgumentParser(description="Random baseline attribution scores.")
    parser.add_argument(
        "--reference-score-path",
        default="anticipation_ip_pilot/outputs/attribution/logra-selected/score_LoGra.pt",
        help="Existing score tensor used only for shape.",
    )
    parser.add_argument(
        "--output-path",
        default="anticipation_ip_pilot/outputs/baselines/random/score_random.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_all_seeds(args.seed)
    ref = torch.load(resolve_repo_path(args.reference_score_path), map_location="cpu", weights_only=False)
    if not torch.is_tensor(ref):
        raise TypeError(f"Expected tensor at {args.reference_score_path}, got {type(ref)}")
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    score = torch.rand(ref.shape, generator=generator, dtype=torch.float32)

    out = resolve_repo_path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(score, out)
    print(f"[random] wrote {out} shape={tuple(score.shape)} seed={args.seed}")


if __name__ == "__main__":
    main()
