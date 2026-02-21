#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from stable_audio_tools.training.songscriber_clap import (
    CLAP_SAMPLE_RATE,
    DEFAULT_CLAP_MODEL_NAME,
    ClapEmbedder,
    load_song_describer_samples,
    save_precomputed_song_describer_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute CLAP embeddings for full song_describer dataset.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("dataset/small-700/song_describer.csv"),
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("dataset/small-700/audio"),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("dataset/small-700/song_describer_clap_embeddings.npz"),
    )
    parser.add_argument(
        "--clap-model-name",
        type=str,
        default=DEFAULT_CLAP_MODEL_NAME,
    )
    parser.add_argument(
        "--clap-sample-rate",
        type=int,
        default=CLAP_SAMPLE_RATE,
    )
    parser.add_argument(
        "--audio-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--text-batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap for debugging.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["auto", "cuda", "cpu"],
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    samples = load_song_describer_samples(args.csv_path, args.audio_dir, limit=args.limit)
    if not samples:
        raise RuntimeError(f"No samples loaded from {args.csv_path}")

    print(f"Loaded {len(samples)} song_describer rows")
    print(f"Using device: {device}")
    print(f"CLAP model: {args.clap_model_name}")

    embedder = ClapEmbedder(
        model_name=args.clap_model_name,
        sample_rate=args.clap_sample_rate,
        device=device,
    )

    audio_paths = [sample.audio_path for sample in samples]
    prompts = [sample.prompt for sample in samples]

    print("Computing audio embeddings...")
    audio_embeddings = embedder.embed_audio_files(audio_paths, batch_size=args.audio_batch_size)
    print("Computing text embeddings...")
    text_embeddings = embedder.embed_texts(prompts, batch_size=args.text_batch_size)

    if audio_embeddings.shape[0] != len(samples) or text_embeddings.shape[0] != len(samples):
        raise RuntimeError(
            f"Embedding size mismatch: rows={len(samples)}, "
            f"audio={audio_embeddings.shape}, text={text_embeddings.shape}"
        )

    save_precomputed_song_describer_embeddings(
        output_path=args.output_path,
        samples=samples,
        audio_embeddings=audio_embeddings,
        text_embeddings=text_embeddings,
        clap_model_name=args.clap_model_name,
        clap_sample_rate=args.clap_sample_rate,
    )
    print(f"Saved precomputed embeddings: {args.output_path}")


if __name__ == "__main__":
    main()
