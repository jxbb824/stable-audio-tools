#!/usr/bin/env python3
"""
Phase 5a: Ground-truth attribution a*_{j,t}
===========================================

For each seller j and prompt t:
  a*_{j,t}  :=  Q(y_t^full, text_t) - Q(y_t^{loo_j}, text_t)

where Q(y, text) is CLAP cosine similarity between audio and text.
A large positive a*_{j,t} means removing seller j degraded the match
to the prompt -> seller j was influential for prompt t.

Writes outputs/a_star.csv with columns:
  seller_id, prompt_id, q_full, q_loo, a_star
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from common import (
    generated_audio_dir,
    outputs_dir,
    prompts_json_path,
    read_json,
    sellers_json_path,
    set_hf_env,
)


def _load_clap(cfg: Dict[str, Any]):
    import torch
    from transformers import ClapModel, ClapProcessor

    set_hf_env(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ClapModel.from_pretrained(cfg["clap_model"]).to(device).eval()
    processor = ClapProcessor.from_pretrained(cfg["clap_model"])
    return model, processor, device


def _read_wav_mono_48k(path: str | Path) -> np.ndarray:
    """CLAP expects 48kHz mono float32. Resample if needed."""
    import soundfile as sf
    import librosa
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 48000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=48000)
    return audio.astype(np.float32)


def _clap_text_embed(model, processor, texts: List[str], device: str) -> np.ndarray:
    import torch
    inputs = processor(text=texts, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.cpu().numpy()


def _clap_audio_embed(model, processor, audios: List[np.ndarray], device: str) -> np.ndarray:
    import torch
    inputs = processor(audios=audios, sampling_rate=48000, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = model.get_audio_features(**inputs)
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.cpu().numpy()


def main(cfg: Dict[str, Any]) -> None:
    out_path = outputs_dir(cfg) / "a_star.csv"
    if out_path.exists():
        print(f"[a_star] already exists: {out_path}; skip", flush=True)
        return

    sellers = read_json(sellers_json_path(cfg))
    prompts = read_json(prompts_json_path(cfg))
    n = len(sellers)
    print(f"[a_star] {n} sellers x {len(prompts)} prompts", flush=True)

    # Pre-flight: every expected generated WAV must exist; otherwise
    # surface a clear "generate phase incomplete" message instead of a
    # libsndfile error deep in the loop.
    missing = []
    for tag in ["full"] + [f"loo_{j}" for j in range(1, n + 1)]:
        d = generated_audio_dir(cfg, tag)
        for p in prompts:
            f = d / f"{p['prompt_id']}.wav"
            if not f.exists() or f.stat().st_size == 0:
                missing.append(str(f))
    if missing:
        raise FileNotFoundError(
            f"[a_star] generate phase incomplete: {len(missing)} missing WAV(s). "
            f"First few: {missing[:5]}. Rerun the 'generate' phase before attribution."
        )

    model, processor, device = _load_clap(cfg)

    # Text embeddings (one per prompt)
    text_emb = _clap_text_embed(model, processor, [p["text"] for p in prompts], device)
    # shape: (T, D)

    # Full-model audio embeddings (T clips)
    full_dir = generated_audio_dir(cfg, "full")
    full_audios = [_read_wav_mono_48k(full_dir / f"{p['prompt_id']}.wav") for p in prompts]
    full_aud_emb = _clap_audio_embed(model, processor, full_audios, device)  # (T, D)

    # Q(y_full, text_t) = diag of full_aud_emb @ text_emb.T
    q_full = (full_aud_emb * text_emb).sum(axis=1)  # (T,)

    rows = []
    for j, seller in enumerate(sellers, start=1):
        t0 = time.time()
        loo_dir = generated_audio_dir(cfg, f"loo_{j}")
        loo_audios = [_read_wav_mono_48k(loo_dir / f"{p['prompt_id']}.wav") for p in prompts]
        loo_aud_emb = _clap_audio_embed(model, processor, loo_audios, device)  # (T, D)
        q_loo = (loo_aud_emb * text_emb).sum(axis=1)  # (T,)
        a_star = q_full - q_loo  # (T,)
        for ti, p in enumerate(prompts):
            rows.append({
                "seller_id": seller["seller_id"],
                "prompt_id": p["prompt_id"],
                "q_full": float(q_full[ti]),
                "q_loo": float(q_loo[ti]),
                "a_star": float(a_star[ti]),
            })
        print(f"[a_star] seller {j}: mean a*={a_star.mean():+.4f} "
              f"(std {a_star.std():.4f}) in {time.time()-t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[a_star] wrote {len(df)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    from common import load_config
    main(load_config(args.config))
