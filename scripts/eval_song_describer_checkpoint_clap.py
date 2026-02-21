#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.training.songscriber_clap import (
    DEFAULT_CLAP_MODEL_NAME,
    ClapEmbedder,
    clap_alignment,
    clap_fad,
    load_precomputed_song_describer_embeddings,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one checkpoint on full song_describer with CLAP-FAD and CLAP alignment."
    )
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--pretransform-ckpt-path", type=Path, default=None)
    parser.add_argument(
        "--precomputed-embeddings-path",
        type=Path,
        default=Path("dataset/small-700/song_describer_clap_embeddings.npz"),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--gen-batch-size", type=int, default=4)
    parser.add_argument("--clap-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Optional cap for quick debugging.")
    parser.add_argument("--seed", type=int, default=-1, help="Base seed, -1 for random.")
    parser.add_argument("--sampler-type", type=str, default="dpmpp-3m-sde")
    parser.add_argument("--sigma-min", type=float, default=0.3)
    parser.add_argument("--sigma-max", type=float, default=500.0)
    parser.add_argument("--clap-model-name", type=str, default=DEFAULT_CLAP_MODEL_NAME)
    parser.add_argument("--device", type=str, default="cuda", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def build_conditioning(prompts, durations, max_duration):
    conditioning = []
    for prompt, duration in zip(prompts, durations):
        seconds_total = float(duration)
        if seconds_total <= 0:
            seconds_total = max_duration
        seconds_total = min(seconds_total, max_duration)
        conditioning.append(
            {
                "prompt": str(prompt),
                "seconds_start": 0.0,
                "seconds_total": float(seconds_total),
            }
        )
    return conditioning


def main():
    args = parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    with args.model_config.open("r", encoding="utf-8") as f:
        model_config = json.load(f)

    print(f"Loading model config: {args.model_config}")
    model = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(str(args.ckpt_path)), strict=False)

    if args.pretransform_ckpt_path is not None and model.pretransform is not None:
        model.pretransform.load_state_dict(load_ckpt_state_dict(str(args.pretransform_ckpt_path)), strict=False)

    model = model.to(device).eval()

    precomputed = load_precomputed_song_describer_embeddings(args.precomputed_embeddings_path)
    prompts = precomputed["prompts"]
    durations = precomputed["durations"]
    ref_audio_embeddings = precomputed["audio_embeddings"]
    text_embeddings = precomputed["text_embeddings"]

    total = len(prompts) if args.limit is None else min(args.limit, len(prompts))
    if total <= 0:
        raise RuntimeError("No prompts to evaluate.")

    print(f"Using device: {device}")
    print(f"Evaluating prompts: {total}")
    print(f"Loading CLAP model: {args.clap_model_name}")
    clap_embedder = ClapEmbedder(model_name=args.clap_model_name, device=device)

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    max_duration = sample_size / sample_rate
    generated_embeddings = []

    for start in range(0, total, args.gen_batch_size):
        end = min(start + args.gen_batch_size, total)
        cond = build_conditioning(prompts[start:end], durations[start:end], max_duration=max_duration)

        seed = args.seed + start if args.seed >= 0 else -1
        audio = generate_diffusion_cond(
            model,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            conditioning=cond,
            batch_size=len(cond),
            sample_size=sample_size,
            sampler_type=args.sampler_type,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            device=str(device),
            seed=seed,
        )
        emb = clap_embedder.embed_audio_tensor(audio, sample_rate=sample_rate, batch_size=args.clap_batch_size)
        generated_embeddings.append(emb)

        del audio
        del emb
        torch.cuda.empty_cache()

        print(f"Generated/evaluated {end}/{total}")

    generated_embeddings = np.concatenate(generated_embeddings, axis=0)

    ref_subset = ref_audio_embeddings[:total]
    text_subset = text_embeddings[:total]

    fad_value = clap_fad(ref_subset, generated_embeddings)
    align_mean, align_std = clap_alignment(text_subset, generated_embeddings)

    print("========== Song Describer CLAP Metrics ==========")
    print(f"num_prompts: {total}")
    print(f"clap_fad: {fad_value}")
    print(f"clap_alignment_mean: {align_mean}")
    print(f"clap_alignment_std: {align_std}")


if __name__ == "__main__":
    main()
