#!/usr/bin/env python3
"""
Minimal DAC histogram similarity test for two songs.
"""

import argparse
import json
import os
import sys

import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from dac_hist_dedup import cosine_similarity_between_files, load_dac_model


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two songs with DAC histogram similarity.")
    parser.add_argument("--file-a", type=str, required=True, help="First audio file path.")
    parser.add_argument("--file-b", type=str, required=True, help="Second audio file path.")
    parser.add_argument("--model-type", type=str, default="44khz", help="DAC model type.")
    parser.add_argument("--model-path", type=str, default=None, help="Optional local DAC model path.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for DAC encoding.",
    )
    parser.add_argument("--codebook-size", type=int, default=1024, help="Histogram bins.")
    parser.add_argument(
        "--histogram-mode",
        type=str,
        choices=["global", "per_codebook"],
        default="per_codebook",
        help="Histogram construction mode.",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=120.0,
        help="Max seconds from start used for each file. Use <=0 for full file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.985,
        help="Reference threshold to judge near-duplicate.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    max_audio_seconds = args.max_audio_seconds
    if max_audio_seconds is not None and max_audio_seconds <= 0:
        max_audio_seconds = None

    model, resolved_model_path = load_dac_model(
        model_type=args.model_type,
        model_path=args.model_path,
        device=args.device,
    )

    similarity = cosine_similarity_between_files(
        file_a=args.file_a,
        file_b=args.file_b,
        model=model,
        device=args.device,
        codebook_size=args.codebook_size,
        max_audio_seconds=max_audio_seconds,
        histogram_mode=args.histogram_mode,
    )

    result = {
        "file_a": os.path.abspath(args.file_a),
        "file_b": os.path.abspath(args.file_b),
        "similarity": similarity,
        "threshold": args.threshold,
        "is_near_duplicate": similarity >= args.threshold,
        "model_type": args.model_type,
        "model_path": resolved_model_path,
        "histogram_mode": args.histogram_mode,
        "codebook_size": args.codebook_size,
        "max_audio_seconds": max_audio_seconds,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
