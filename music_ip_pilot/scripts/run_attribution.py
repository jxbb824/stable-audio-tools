#!/usr/bin/env python3
"""
Phase 5b: Measured attribution a_hat_{j,t}  (embedding similarity)
==================================================================

For each seller j and prompt t:
  a_hat^{embed}_{j,t}  :=  cos(CLAP_audio(y_t^full), c_j)

where c_j is the mean CLAP audio embedding of seller j's catalog D_j.
Higher means "the generated output is closer in style to D_j".

This is our PILOT method. Future additions: influence functions (TracIn),
Data Shapley. All will write to their own columns in a_hat_embed.csv or
a new a_hat_<method>.csv.

Writes outputs/a_hat_embed.csv with columns:
  seller_id, prompt_id, a_hat_embed
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
    seller_audio_dir,
    sellers_json_path,
    set_hf_env,
)

# Re-use helpers from compute_a_star without circular imports
from compute_a_star import _clap_audio_embed, _load_clap, _read_wav_mono_48k


def _catalog_embedding(
    seller: Dict[str, Any],
    cfg: Dict[str, Any],
    model, processor, device: str,
    batch_size: int = 8,
) -> np.ndarray:
    """Mean CLAP audio embedding over seller's catalog (L2-normalised)."""
    audio_dir = seller_audio_dir(cfg, seller["seller_id"])
    wavs = []
    for t in seller["tracks"]:
        fp = audio_dir / f"{t['track_id']}.wav"
        if not fp.exists():
            raise FileNotFoundError(fp)
        wavs.append(_read_wav_mono_48k(fp))

    embs: List[np.ndarray] = []
    for i in range(0, len(wavs), batch_size):
        batch = wavs[i : i + batch_size]
        embs.append(_clap_audio_embed(model, processor, batch, device))
    emb_stack = np.concatenate(embs, axis=0)  # (K, D)
    mean_emb = emb_stack.mean(axis=0)
    # re-normalise
    norm = np.linalg.norm(mean_emb)
    return (mean_emb / norm) if norm > 0 else mean_emb


def main(cfg: Dict[str, Any]) -> None:
    out_path = outputs_dir(cfg) / "a_hat_embed.csv"
    if out_path.exists():
        print(f"[a_hat_embed] already exists: {out_path}; skip", flush=True)
        return

    sellers = read_json(sellers_json_path(cfg))
    prompts = read_json(prompts_json_path(cfg))
    print(f"[a_hat_embed] {len(sellers)} sellers x {len(prompts)} prompts", flush=True)

    # Pre-flight: full-model WAVs and catalog WAVs must exist.
    missing = []
    full_dir = generated_audio_dir(cfg, "full")
    for p in prompts:
        f = full_dir / f"{p['prompt_id']}.wav"
        if not f.exists() or f.stat().st_size == 0:
            missing.append(str(f))
    for s in sellers:
        d = seller_audio_dir(cfg, s["seller_id"])
        for t in s["tracks"]:
            f = d / f"{t['track_id']}.wav"
            if not f.exists() or f.stat().st_size == 0:
                missing.append(str(f))
    if missing:
        raise FileNotFoundError(
            f"[a_hat_embed] required WAV(s) missing: {len(missing)}. "
            f"First few: {missing[:5]}. Rerun 'setup' and/or 'generate' phases."
        )

    model, processor, device = _load_clap(cfg)

    # Catalog centroids
    catalog_embs = {}
    for j, s in enumerate(sellers, start=1):
        t0 = time.time()
        catalog_embs[s["seller_id"]] = _catalog_embedding(s, cfg, model, processor, device)
        print(f"[a_hat_embed] catalog embed for {s['seller_id']}: "
              f"{time.time()-t0:.1f}s", flush=True)

    # Full-model outputs
    full_dir = generated_audio_dir(cfg, "full")
    full_audios = [_read_wav_mono_48k(full_dir / f"{p['prompt_id']}.wav") for p in prompts]
    full_aud_emb = _clap_audio_embed(model, processor, full_audios, device)  # (T, D)

    rows = []
    for s in sellers:
        c = catalog_embs[s["seller_id"]]  # (D,)
        sims = full_aud_emb @ c  # (T,)  (both are L2-normalized)
        for ti, p in enumerate(prompts):
            rows.append({
                "seller_id": s["seller_id"],
                "prompt_id": p["prompt_id"],
                "a_hat_embed": float(sims[ti]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"[a_hat_embed] wrote {len(df)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    from common import load_config
    main(load_config(args.config))
