#!/usr/bin/env python3
"""
Precompute CLAP catalog embeddings + geometry stats
===================================================

For each seller j in sellers.json:
  1. Load every staged catalog WAV from `mtg_jamendo_subset/{seller_id}/`.
  2. Resample to 48 kHz mono (CLAP's native rate) and compute CLAP audio
     embeddings (L2-normalised) using the same model as Phase 5 attribution.
  3. Save per-track embeddings, per-seller centroid, and geometry summaries.

Outputs (both in `outputs/`):
  * `catalog_embeddings.npz` : float32 arrays, keyed by seller_id:
      "{seller_id}/embeddings" -- shape (K, D)  (per-track, L2-normalised)
      "{seller_id}/centroid"   -- shape (D,)     (L2-normalised mean)
      "{seller_id}/track_ids"  -- shape (K,)     (string array, for ordering)
      "grand_centroid"         -- shape (D,)     (L2-normalised mean of centroids)
  * `catalog_stats.json` : per-seller geometry stats:
      distinctiveness_to_grand : 1 - cos(c_j, c_bar)
      distinctiveness_to_nearest : 1 - max_{k != j} cos(c_j, c_k)
      within_catalog_coherence : mean pairwise cosine sim over tracks in D_j
      n_tracks

Why precompute?
  * Phase 5b (`run_attribution.py`) needs exactly these centroids to compute
    a_hat_embed = cos(CLAP(y_t^full), c_j). Without this file it recomputes on
    GPU; with it, Phase 5b only needs the T=10 generation-side embeddings.
  * The stats feed the "cheap I proxy" story: regress I_j on distinctiveness +
    coherence, report R^2. See §Sampling design Q4 in experiments_plan.md.

Runs on: cluster `cpu` partition (or any CPU). ~30 WAVs at 48kHz is a few minutes
of CLAP inference on CPU; no GPU needed. CLAP (~600 MB) loads from HF hub cache
at `cfg["models_hub"]`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import (
    load_config,
    outputs_dir,
    read_json,
    seller_audio_dir,
    sellers_json_path,
    set_hf_env,
)
from compute_a_star import _clap_audio_embed, _load_clap, _read_wav_mono_48k


def _pairwise_cosine_mean(embs: np.ndarray) -> float:
    """Mean of off-diagonal cosine similarities for L2-normalised rows."""
    if embs.shape[0] < 2:
        return 1.0  # degenerate: one track, trivially coherent
    sims = embs @ embs.T  # (K, K); diag == 1 since L2-normalised
    # off-diag mean: (sum - trace) / (K*(K-1))
    k = embs.shape[0]
    off_diag_sum = float(sims.sum() - np.trace(sims))
    return off_diag_sum / (k * (k - 1))


def main(cfg: Dict[str, Any]) -> None:
    out_dir = outputs_dir(cfg)
    emb_path = out_dir / "catalog_embeddings.npz"
    stats_path = out_dir / "catalog_stats.json"

    if emb_path.exists() and stats_path.exists():
        print(f"[catalog-emb] already present: {emb_path}, {stats_path}; skip",
              flush=True)
        return

    sellers = read_json(sellers_json_path(cfg))
    print(f"[catalog-emb] {len(sellers)} sellers", flush=True)

    # Pre-flight: every staged WAV must exist
    missing: List[str] = []
    for s in sellers:
        audio_dir = seller_audio_dir(cfg, s["seller_id"])
        for t in s["tracks"]:
            fp = audio_dir / f"{t['track_id']}.wav"
            if not fp.exists() or fp.stat().st_size == 0:
                missing.append(str(fp))
    if missing:
        raise FileNotFoundError(
            f"[catalog-emb] {len(missing)} WAV(s) missing. First: {missing[:5]}. "
            f"Run `pilot.py --phase setup` first."
        )

    set_hf_env(cfg)
    model, processor, device = _load_clap(cfg)
    print(f"[catalog-emb] CLAP on device={device}", flush=True)

    # Per-seller: embed catalog, compute centroid + coherence
    per_seller: Dict[str, Dict[str, Any]] = {}
    batch_size = 8

    for s in sellers:
        t0 = time.time()
        audio_dir = seller_audio_dir(cfg, s["seller_id"])
        track_ids = [t["track_id"] for t in s["tracks"]]
        wavs = [_read_wav_mono_48k(audio_dir / f"{tid}.wav") for tid in track_ids]

        emb_chunks: List[np.ndarray] = []
        for i in range(0, len(wavs), batch_size):
            chunk = wavs[i : i + batch_size]
            emb_chunks.append(_clap_audio_embed(model, processor, chunk, device))
        embs = np.concatenate(emb_chunks, axis=0)  # (K, D), L2-normalised by _clap_audio_embed

        centroid = embs.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        centroid_n = centroid / norm if norm > 0 else centroid

        coherence = _pairwise_cosine_mean(embs)

        per_seller[s["seller_id"]] = {
            "embeddings": embs.astype(np.float32),
            "centroid": centroid_n.astype(np.float32),
            "track_ids": np.array(track_ids, dtype=object),
            "n_tracks": int(len(track_ids)),
            "coherence": float(coherence),
            "primary_genre": s["primary_genre"],
        }
        print(f"[catalog-emb] {s['seller_id']} ({s['primary_genre']}): "
              f"{len(embs)} tracks, coherence={coherence:+.4f}, "
              f"{time.time()-t0:.1f}s", flush=True)

    # Grand centroid = L2-normalised mean of per-seller centroids
    centroids = np.stack([per_seller[s["seller_id"]]["centroid"] for s in sellers], axis=0)
    grand = centroids.mean(axis=0)
    gnorm = float(np.linalg.norm(grand))
    grand_n = (grand / gnorm) if gnorm > 0 else grand

    # Geometry stats per seller
    stats: Dict[str, Any] = {
        "meta": {
            "clap_model": cfg["clap_model"],
            "sample_rate_used": 48000,
            "n_sellers": len(sellers),
        },
        "grand_centroid_norm": gnorm,
        "sellers": {},
    }
    for idx, s in enumerate(sellers):
        c = centroids[idx]
        # distinctiveness vs. the grand centroid
        d_grand = 1.0 - float(c @ grand_n)
        # distinctiveness vs. the nearest other seller centroid
        others = np.delete(centroids, idx, axis=0)
        nearest_cos = float((others @ c).max()) if others.shape[0] else 0.0
        d_nearest = 1.0 - nearest_cos

        stats["sellers"][s["seller_id"]] = {
            "artist": s["artist"],
            "primary_genre": s["primary_genre"],
            "n_tracks": per_seller[s["seller_id"]]["n_tracks"],
            "within_catalog_coherence": per_seller[s["seller_id"]]["coherence"],
            "distinctiveness_to_grand": d_grand,
            "distinctiveness_to_nearest": d_nearest,
        }

    # Save artefacts
    save_kwargs: Dict[str, Any] = {"grand_centroid": grand_n.astype(np.float32)}
    for sid, payload in per_seller.items():
        save_kwargs[f"{sid}__embeddings"] = payload["embeddings"]
        save_kwargs[f"{sid}__centroid"] = payload["centroid"]
        save_kwargs[f"{sid}__track_ids"] = payload["track_ids"]
    np.savez(emb_path, **save_kwargs)
    print(f"[catalog-emb] wrote {emb_path} ({emb_path.stat().st_size/1e6:.2f} MB)")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[catalog-emb] wrote {stats_path}")

    # Console summary: the key numbers
    print("\n[catalog-emb] summary:")
    hdr = f"{'seller':<11} {'genre':<11} {'K':>3} {'coherence':>10} {'d_grand':>10} {'d_nearest':>10}"
    print(hdr)
    print("-" * len(hdr))
    for sid, info in stats["sellers"].items():
        print(f"  {sid:<9} {info['primary_genre']:<11} "
              f"{info['n_tracks']:>3} "
              f"{info['within_catalog_coherence']:>10.4f} "
              f"{info['distinctiveness_to_grand']:>10.4f} "
              f"{info['distinctiveness_to_nearest']:>10.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
