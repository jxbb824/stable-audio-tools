#!/usr/bin/env python3
"""
Phase 1: Setup
==============
Consume the pre-staged MTG-Jamendo MP3s (downloaded by `download_mtg.py` into
a local directory mirroring MTG's `{prefix}/{track_id}.mp3` layout), decode
each to 32 kHz mono PCM-16 WAV clips, and emit `sellers.json` + `prompts.json`.

Idempotent: if `sellers.json` + `prompts.json` exist and every expected WAV is
present, returns early.

Pipeline pivot note (vs. earlier design)
----------------------------------------
We no longer stream from the HuggingFace dataset `rkstgr/mtg-jamendo`. That
path had three problems: (i) the dataset ships a legacy `.py` loader that
requires `datasets<4.0` and `trust_remote_code=True`; (ii) even with that
pinned, it insists on downloading the full 109 GiB dataset in non-streaming
mode; (iii) the HF subset is only 13 854 tracks, so primary picks at the
long-tail floor may fall outside it. Instead we download specific tracks
directly from the Freesound CDN mirror, which is O(100 MB) for N=3 and gives
us the full 55 609-track space.

Inputs consumed by Phase 1:
  - `cfg["sellers_candidates_json"]` : primaries come from pilot EDA (local).
  - `cfg["mtg_raw_dir"]`             : already-staged MP3s
                                        (from `download_mtg.py`, Phase 0.5).

Outputs:
  - `{data_root}/data/sellers.json`                       (source of truth)
  - `{data_root}/data/prompts.json`                       (cross-stratified 10)
  - `{data_root}/data/mtg_jamendo_subset/{seller_id}/{track_id}.wav`
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import (
    data_subdir,
    ensure_dir,
    prompts_json_path,
    sellers_json_path,
    seller_audio_dir,
    set_all_seeds,
    write_json,
)


# ─────────────────────────────────────────────────────────────────────
# Candidate loading
# ─────────────────────────────────────────────────────────────────────

def load_candidates(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def candidates_to_sellers(candidates: Dict[str, Any], n_sellers: int) -> List[Dict[str, Any]]:
    """Pick primaries across the low/mid/high tiers; assemble sellers.json schema.

    The 3-tier primaries are the canonical pilot selection from the EDA.
    Walk order: all primaries first (low -> mid -> high), then fallbacks
    round-robin across tiers (low.F0, mid.F0, high.F0, low.F1, ...). This
    preserves prolificness diversity when N > 3 and keeps N=3 = [low.P, mid.P,
    high.P] regardless of how many fallbacks exist per tier.
    """
    tiers = [t for t in ("low", "mid", "high") if t in candidates]
    ordered: List[Dict[str, Any]] = [candidates[t]["primary"] for t in tiers]

    # Round-robin fallbacks, tier-by-tier
    max_fb = max(
        (len(candidates[t].get("fallbacks", [])) for t in tiers),
        default=0,
    )
    for i in range(max_fb):
        for t in tiers:
            fb_list = candidates[t].get("fallbacks", [])
            if i < len(fb_list):
                ordered.append(fb_list[i])

    if len(ordered) < n_sellers:
        raise RuntimeError(
            f"candidates JSON has only {len(ordered)} artists total; "
            f"config requests n_sellers={n_sellers}."
        )

    sellers: List[Dict[str, Any]] = []
    for idx, c in enumerate(ordered[:n_sellers], start=1):
        sellers.append({
            "seller_id": f"seller_{idx}",
            "artist": c["artist_id"],
            "primary_genre": c["primary_genre"],
            "tracks": [
                {
                    "track_id": t["track_id"],
                    "hf_key": t["hf_key"],          # e.g., "14/214"
                    "audio_path": t["path"],         # e.g., "14/214.mp3"
                    "duration_s": float(t["duration"]),
                }
                for t in c["tracks"]
            ],
        })
    return sellers


def trim_tracks_per_seller(sellers: List[Dict[str, Any]], tracks_per_seller: int) -> None:
    """Cap each seller's track list to `tracks_per_seller` (in-place)."""
    for s in sellers:
        if len(s["tracks"]) > tracks_per_seller:
            s["tracks"] = s["tracks"][:tracks_per_seller]


# ─────────────────────────────────────────────────────────────────────
# Audio decoding (local MP3 -> WAV clip)
# ─────────────────────────────────────────────────────────────────────

def decode_and_stage(sellers: List[Dict[str, Any]], cfg: Dict[str, Any]) -> None:
    """Decode each tracked MP3 from `mtg_raw_dir`, clip, resample, write WAV.

    Raises RuntimeError if any source MP3 is missing — we want hard failure
    here, not silent synthetic fallback.
    """
    import librosa
    import soundfile as sf

    raw_dir = Path(cfg["mtg_raw_dir"])
    target_rate = int(cfg["sample_rate"])
    clip_samples = int(target_rate * float(cfg["clip_seconds"]))

    n_written = 0
    n_skipped = 0
    missing: List[str] = []

    for seller in sellers:
        out_dir = seller_audio_dir(cfg, seller["seller_id"])
        ensure_dir(out_dir)
        for t in seller["tracks"]:
            dst = out_dir / f"{t['track_id']}.wav"
            if dst.exists() and dst.stat().st_size > 0:
                n_skipped += 1
                continue

            src = raw_dir / t["audio_path"]  # e.g., {raw_dir}/14/214.mp3
            if not src.exists():
                missing.append(str(src))
                continue

            # librosa.load decodes MP3 via audioread or soundfile backends.
            # sr=target_rate triggers the resample automatically; mono=True
            # averages channels.
            audio, _ = librosa.load(str(src), sr=target_rate, mono=True)
            clip = _take_clip(audio.astype(np.float32), clip_samples)
            clip = np.clip(clip, -1.0, 1.0)  # safety vs int16 wrap-around
            sf.write(str(dst), clip, target_rate, subtype="PCM_16")
            n_written += 1

    if missing:
        raise RuntimeError(
            f"[setup] {len(missing)} source MP3(s) missing from {raw_dir}. "
            f"First few: {missing[:5]}. Run `download_mtg.py` first."
        )
    print(f"[setup] decoded {n_written} new WAV(s); {n_skipped} already present",
          flush=True)


def _take_clip(audio: np.ndarray, clip_samples: int) -> np.ndarray:
    """Take a centered clip of exactly `clip_samples`; zero-pad if shorter."""
    if len(audio) >= clip_samples:
        start = max(0, (len(audio) - clip_samples) // 2)
        return audio[start : start + clip_samples]
    out = np.zeros(clip_samples, dtype=np.float32)
    out[: len(audio)] = audio
    return out


# ─────────────────────────────────────────────────────────────────────
# Prompt design: cross-stratified genre grid
# ─────────────────────────────────────────────────────────────────────
#
# Per §Sampling design and §Scale-up sampling strategy in experiments_plan.md,
# the pilot's T prompts should be a stratified grid over seller genres plus
# an off-genre control. For N=3 sellers and T=10: 3 prompts per seller genre
# (on-genre) + 1 off-genre control. The mix generates variance in both a*
# and â_embed which is required for I_j = Cov/Var to be estimable at T=10.
#
# Templates are indexed by genre; we normalise unknown/long-tail genres to a
# generic "music" family so the pipeline never silently fails.

GENRE_TEMPLATES: Dict[str, List[str]] = {
    "pop": [
        "An upbeat pop song with bright synthesizers and a catchy hook",
        "A mellow acoustic pop ballad with soft piano accompaniment",
        "An energetic dance-pop track with driving rhythm and vocal chops",
    ],
    "reggae": [
        "A roots reggae piece with deep bass and offbeat guitar skank",
        "A dub-inspired reggae track with spacey reverb and heavy low end",
        "A ska-influenced reggae tune with upbeat horns and quick tempo",
    ],
    "classical": [
        "An orchestral cinematic piece with sweeping strings and brass",
        "A contemplative solo piano classical melody with gentle phrasing",
        "A string quartet with intricate counterpoint and warm tone",
    ],
    "rock": [
        "A driving rock song with distorted guitars and pounding drums",
        "A mid-tempo rock ballad with clean guitar arpeggios",
        "An aggressive hard rock instrumental with fast double-bass kicks",
    ],
    "electronic": [
        "A minimal techno track with a steady four-on-the-floor kick",
        "An atmospheric electronic piece with evolving synth pads",
        "A glitchy IDM track with syncopated drum programming",
    ],
    "ambient": [
        "A slow-evolving ambient drone with lush reverb",
        "A minimalist ambient piece with field recordings and soft textures",
        "A dark ambient soundscape with sub bass and tension",
    ],
    "soundtrack": [
        "A suspenseful film soundtrack with staccato strings",
        "A heroic orchestral cue with triumphant brass",
        "A melancholic cinematic theme with solo cello",
    ],
    "folk": [
        "A fingerpicked acoustic folk tune with warm vocal harmony feel",
        "A traditional folk ballad with gentle percussion",
        "An uptempo bluegrass-inflected folk track with banjo",
    ],
    "indie": [
        "An indie rock song with jangly guitars and laid-back vocals",
        "A dreamy indie pop track with shimmering guitar effects",
        "A lo-fi indie instrumental with tape saturation",
    ],
    "poprock": [
        "A power-pop rock track with big choruses and driving drums",
        "A melodic pop-rock song with layered guitars",
        "An arena-style pop-rock ballad with soaring dynamics",
    ],
    "hiphop": [
        "A boom-bap hip-hop beat with crisp snares and vinyl samples",
        "A trap-style hip-hop instrumental with rolling hi-hats and 808 bass",
        "A lo-fi hip-hop beat with jazz piano loops",
    ],
    "metal": [
        "A thrash metal instrumental with palm-muted riffs and blast beats",
        "A doom metal piece with slow, crushing riffs and deep distortion",
        "A progressive metal passage with odd time signatures",
    ],
    "jazz": [
        "A cool jazz piece with walking bass and muted trumpet",
        "A bebop jazz track with fast upright bass and brushed drums",
        "A smooth jazz ballad with electric piano and saxophone",
    ],
}

# Used when a seller's primary_genre isn't in the table above.
GENERIC_TEMPLATES = [
    "A melodic instrumental piece with distinctive character",
    "An expressive composition with dynamic shifts",
    "A rhythmically engaging piece with clear groove",
]

# Off-genre controls: exercised by ALL sellers to detect false-positive attribution.
# Choose genres absent from the pilot seller set (pop/reggae/classical) so
# none of the LOO models should differentially affect them.
OFF_GENRE_TEMPLATES = [
    "A boom-bap hip-hop beat with crisp snares and vinyl samples",
    "A trap-style hip-hop instrumental with rolling hi-hats and 808 bass",
    "A thrash metal instrumental with palm-muted riffs and blast beats",
]
OFF_GENRE_LABEL = "off_genre_control"


def build_prompts(sellers: List[Dict[str, Any]], n_prompts: int) -> List[Dict[str, Any]]:
    """Cross-stratified grid: k on-genre prompts per seller + ~10% off-genre controls.

    Allocation: reserve ceil(n_prompts/10) slots for off-genre controls (at least 1),
    then split the remainder evenly across the N sellers. Remainder after floor
    division gets absorbed back into the off-genre slots so total == n_prompts.

    For N=3 sellers and T=10: 1 off-genre + 3 on-genre × 3 sellers = 10. ✓
    For N=4 sellers and T=20: 2 off-genre + 4 on-genre × 4 sellers = 18 → 2 more off = 20.
    """
    n = len(sellers)
    min_off = max(1, n_prompts // 10)
    per_seller = max(1, (n_prompts - min_off) // n)
    off_n = n_prompts - per_seller * n
    assert off_n >= 1, "n_prompts too small to guarantee an off-genre control"

    prompts: List[Dict[str, Any]] = []

    # On-genre prompts per seller (cycle templates if per_seller > len(templates))
    for s in sellers:
        genre = s["primary_genre"]
        templates = GENRE_TEMPLATES.get(genre, GENERIC_TEMPLATES)
        pool = list(templates)
        while len(pool) < per_seller:
            pool = pool + templates
        for i in range(per_seller):
            prompts.append({
                "prompt_id": f"prompt_{len(prompts)+1}",
                "text": pool[i],
                "kind": "on_genre",
                "target_genre": genre,
                "linked_seller": s["seller_id"],
            })

    # Off-genre controls (cycle through OFF_GENRE_TEMPLATES if off_n > len(templates))
    for i in range(off_n):
        text = OFF_GENRE_TEMPLATES[i % len(OFF_GENRE_TEMPLATES)]
        prompts.append({
            "prompt_id": f"prompt_{len(prompts)+1}",
            "text": text,
            "kind": OFF_GENRE_LABEL,
            "target_genre": "hiphop_offgenre" if "hip-hop" in text else "metal_offgenre",
            "linked_seller": None,
        })
    return prompts


# ─────────────────────────────────────────────────────────────────────
# Phase entry point
# ─────────────────────────────────────────────────────────────────────

def main(cfg: Dict[str, Any]) -> None:
    set_all_seeds(cfg["seed"])

    sellers_path = sellers_json_path(cfg)
    prompts_path = prompts_json_path(cfg)

    data_subdir(cfg, "data")
    data_subdir(cfg, "data", "mtg_jamendo_subset")

    # Short-circuit if already built AND all WAVs present
    if sellers_path.exists() and prompts_path.exists():
        sellers = json.load(open(sellers_path))
        all_ok = all(
            (seller_audio_dir(cfg, s["seller_id"]) / f"{t['track_id']}.wav").exists()
            and (seller_audio_dir(cfg, s["seller_id"]) / f"{t['track_id']}.wav").stat().st_size > 0
            for s in sellers for t in s["tracks"]
        )
        if all_ok:
            print(f"[setup] already built: {len(sellers)} sellers, audio present; skip",
                  flush=True)
            return

    # Load candidate picks (from pilot EDA)
    cand_path = Path(cfg["sellers_candidates_json"])
    if not cand_path.exists():
        raise FileNotFoundError(f"sellers_candidates_json not found: {cand_path}")
    candidates = load_candidates(cand_path)
    print(f"[setup] candidates: {cand_path}", flush=True)

    sellers = candidates_to_sellers(candidates, cfg["n_sellers"])
    trim_tracks_per_seller(sellers, cfg["tracks_per_seller"])
    print(f"[setup] {len(sellers)} sellers assembled:", flush=True)
    for s in sellers:
        print(f"  - {s['seller_id']}: {s['artist']} "
              f"({s['primary_genre']}, {len(s['tracks'])} tracks)", flush=True)

    # Decode MP3s -> 32 kHz mono WAV clips
    t0 = time.time()
    decode_and_stage(sellers, cfg)
    print(f"[setup] audio staged in {time.time()-t0:.1f}s", flush=True)

    # Cross-stratified prompt grid
    prompts = build_prompts(sellers, cfg["n_prompts"])
    print(f"[setup] built {len(prompts)} prompts "
          f"(kinds: "
          f"{', '.join(sorted({p['kind'] for p in prompts}))})", flush=True)

    write_json(sellers_path, sellers)
    write_json(prompts_path, prompts)
    print(f"[setup] sellers.json -> {sellers_path}", flush=True)
    print(f"[setup] prompts.json -> {prompts_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    from common import load_config
    main(load_config(args.config))
