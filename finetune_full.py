#!/usr/bin/env python3

import argparse
import torch

from finetune_subset import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one stable-audio fine-tuning run on the full 1000-song candidate pool."
    )
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True)
    parser.add_argument("--val-dataset-config", type=str, default="")
    parser.add_argument("--pretrained-ckpt-path", type=str, required=True)
    parser.add_argument("--pretransform-ckpt-path", type=str, default="")
    parser.add_argument(
        "--remove-pretransform-weight-norm",
        type=str,
        default="none",
        choices=["none", "pre_load", "post_load"],
    )
    parser.add_argument("--output-dir", type=str, default="outputs/groundtruth_models")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate-pool-size", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--accum-batches", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--strategy", type=str, default="")
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--name", type=str, default="stable_audio_groundtruth_full")
    parser.add_argument("--logger", type=str, default="none", choices=["none", "wandb", "comet"])
    parser.add_argument("--use-safetensors", action="store_true")
    return parser.parse_args()


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = parse_args()
    args.run_id = 0
    args.output_subdir = "full"
    args.train_subset_size = args.candidate_pool_size
    run(args)


if __name__ == "__main__":
    main()
