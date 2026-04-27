#!/usr/bin/env python3
"""
Phase 4: Generation
===================
Generate one audio clip per (model_tag, prompt_id) pair. Loads the
shared MusicGen base + the given LoRA adapter, generates, saves WAV.

Usage:
  python generate.py --config configs/pilot.yaml --model-tag full
  python generate.py --config configs/pilot.yaml --model-tag loo_1
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from common import (
    checkpoint_dir,
    generated_audio_dir,
    prompts_json_path,
    read_json,
    set_all_seeds,
    set_hf_env,
)


def _load_model_with_adapter(cfg: Dict[str, Any], model_tag: str):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained(cfg["base_model"])
    # bf16 to match training-time dtype; L40S has plenty of VRAM but bf16
    # halves memory and speeds generation notably.
    base = MusicgenForConditionalGeneration.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16
    )

    adapter_dir = checkpoint_dir(cfg, model_tag) / "adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter not found for tag='{model_tag}': {adapter_dir}")

    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()
    return processor, model


def main(cfg: Dict[str, Any], model_tag: str) -> None:
    import torch
    import soundfile as sf

    set_hf_env(cfg)
    set_all_seeds(cfg["seed"])

    out_dir = generated_audio_dir(cfg, model_tag)
    prompts = read_json(prompts_json_path(cfg))

    # Short-circuit if all expected files exist
    existing = [(out_dir / f"{p['prompt_id']}.wav") for p in prompts]
    if all(f.exists() and f.stat().st_size > 0 for f in existing):
        print(f"[generate/{model_tag}] all {len(prompts)} outputs already exist; skip", flush=True)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[generate/{model_tag}] device={device}", flush=True)

    t0 = time.time()
    processor, model = _load_model_with_adapter(cfg, model_tag)
    model = model.to(device)
    print(f"[generate/{model_tag}] model loaded in {time.time()-t0:.1f}s", flush=True)

    sampling_rate = processor.feature_extractor.sampling_rate
    gen_kwargs = {
        "max_new_tokens": int(cfg["gen_max_new_tokens"]),
        "do_sample": bool(cfg.get("gen_do_sample", True)),
        "top_k": int(cfg["gen_top_k"]),
        "temperature": float(cfg["gen_temperature"]),
    }

    for p in prompts:
        dst = out_dir / f"{p['prompt_id']}.wav"
        if dst.exists() and dst.stat().st_size > 0:
            continue
        t1 = time.time()
        inputs = processor(text=[p["text"]], padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        # out shape: (batch, channels=1, samples); cast bf16 -> fp32 BEFORE
        # .numpy() because numpy doesn't support bf16 directly.
        audio = out[0, 0].to(torch.float32).cpu().numpy()
        # clip to [-1, 1] defensively
        audio = np.clip(audio, -1.0, 1.0)
        sf.write(str(dst), audio, sampling_rate, subtype="PCM_16")
        print(f"[generate/{model_tag}] {p['prompt_id']}: '{p['text'][:60]}...' "
              f"-> {dst.name} ({len(audio)/sampling_rate:.1f}s, {time.time()-t1:.1f}s)",
              flush=True)

    print(f"[generate/{model_tag}] done", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--model-tag", required=True,
                        help="checkpoint tag: 'full' or 'loo_<j>'")
    args = parser.parse_args()
    from common import load_config
    main(load_config(args.config), model_tag=args.model_tag)
