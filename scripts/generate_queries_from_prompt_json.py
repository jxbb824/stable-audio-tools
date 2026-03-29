#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate query audio from a prompt JSON using the full 1000-song model."
    )
    parser.add_argument("--model-config", type=str, default="model_config_freeze_vae.json")
    parser.add_argument("--ckpt-path", type=str, default="outputs/groundtruth_models/full/model.ckpt")
    parser.add_argument("--prompt-json", type=str, default="outputs/groundtruth_models/full/prompts_200.json")
    parser.add_argument("--output-dir", type=str, default="outputs/generated_queries")
    parser.add_argument("--output-ext", type=str, default="mp3", choices=["mp3", "wav"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cfg-scale", type=float, default=7.0)
    parser.add_argument("--sigma-min", type=float, default=0.3)
    parser.add_argument("--sigma-max", type=float, default=500.0)
    parser.add_argument("--sampler-type", type=str, default="dpmpp-3m-sde")
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--no-clean-output-dir", action="store_true")
    return parser.parse_args()


def save_audio(audio: torch.Tensor, output_path: Path, sample_rate: int) -> None:
    audio = audio.to(torch.float32)
    peak = torch.max(torch.abs(audio)).clamp(min=1e-8)
    audio = (audio / peak).clamp(-1, 1).cpu()

    if output_path.suffix.lower() == ".mp3":
        torchaudio.save(str(output_path), audio, sample_rate, format="mp3")
    else:
        audio_i16 = audio.mul(32767).to(torch.int16)
        torchaudio.save(str(output_path), audio_i16, sample_rate)


def clean_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.mp3", "*.wav", "prompt_manifest.json", "prompts.json"):
        for path in output_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    if not args.no_clean_output_dir:
        clean_output_dir(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.prompt_json, "r", encoding="utf-8") as handle:
        prompt_payload = json.load(handle)
    prompt_rows = prompt_payload["prompts"]

    with open(args.model_config, "r", encoding="utf-8") as handle:
        model_config = json.load(handle)

    model = create_model_from_config(model_config)
    model.load_state_dict(load_ckpt_state_dict(args.ckpt_path))
    model = model.to(device).eval()

    sample_rate = int(model_config["sample_rate"])
    sample_size = int(model_config["sample_size"])
    manifest = []

    for start in range(0, len(prompt_rows), args.batch_size):
        end = min(start + args.batch_size, len(prompt_rows))
        batch_rows = prompt_rows[start:end]
        conditioning = [
            {
                "prompt": row["prompt"],
                "seconds_start": float(row.get("seconds_start", 0.0)),
                "seconds_total": float(row.get("seconds_total", sample_size / sample_rate)),
            }
            for row in batch_rows
        ]

        output = generate_diffusion_cond(
            model,
            steps=args.steps,
            cfg_scale=args.cfg_scale,
            conditioning=conditioning,
            batch_size=len(batch_rows),
            sample_size=sample_size,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
            sampler_type=args.sampler_type,
            device=device,
            seed=args.base_seed + start,
        )

        for local_index, (audio, row) in enumerate(zip(output, batch_rows)):
            prompt_id = int(row["prompt_id"])
            basename = f"query_{prompt_id:04d}.{args.output_ext}"
            output_path = output_dir / basename
            save_audio(audio, output_path, sample_rate)
            manifest.append(
                {
                    **row,
                    "filename": basename,
                    "audio_path": str(output_path.resolve()),
                    "batch_seed": args.base_seed + start,
                    "sample_rate": sample_rate,
                    "sample_size": sample_size,
                }
            )

        if device == "cuda":
            torch.cuda.empty_cache()

    (output_dir / "prompts.json").write_text(
        json.dumps(prompt_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "prompt_manifest.json").write_text(
        json.dumps({"items": manifest}, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(manifest)} generated queries to {output_dir}")


if __name__ == "__main__":
    main()
