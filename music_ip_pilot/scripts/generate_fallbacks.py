#!/usr/bin/env python3
"""
Generate ranked fallback candidates per pilot tier.

For each prolificness tercile (low / mid / high) produce the top-K coherent
artists ordered by proximity to tier median (tie-break: coherence desc,
n_tracks desc). The first entry per tier is the primary pick; the rest are
fallbacks for verify_hf_subset.py to swap in if the primary isn't in the
HF stream (rkstgr/mtg-jamendo ~= 25% of the 55k autotagging split).

Genre diversity across primary picks is enforced by greedy genre exclusion
(same logic as eda_groupings.select_pilot_candidates). Fallbacks within a
tier may share a genre with another tier's fallback.

Inputs:
  data/stats/artists_summary.csv
  data/mtg_jamendo_metadata/autotagging.tsv  (for per-artist track list)

Outputs (overwrites):
  data/stats/pilot_sellers_candidates.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARTISTS_CSV = ROOT / "data" / "stats" / "artists_summary.csv"
META_TSV = ROOT / "data" / "mtg_jamendo_metadata" / "autotagging.tsv"
OUT_JSON = ROOT / "data" / "stats" / "pilot_sellers_candidates.json"

TOP_K_PER_TIER = 5
MIN_TRACKS = 10
COHERENCE_FLOOR = 0.7


def load_artist_tracks(meta_tsv: Path) -> dict[str, list[dict]]:
    """artist_id -> list of {track_id, path_no_ext (HF __key__), duration}."""
    out: dict[str, list[dict]] = defaultdict(list)
    with open(meta_tsv) as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header[:5] == ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION"], header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            track_id, artist_id, album_id, rel_path, duration = parts[:5]
            key = rel_path.rsplit(".", 1)[0]  # strip .mp3 -> "15/979815" for HF __key__
            out[artist_id].append({
                "track_id": track_id,
                "album_id": album_id,
                "path": rel_path,
                "hf_key": key,
                "duration": float(duration),
            })
    return dict(out)


def rank_tier(tier_pool: pd.DataFrame, tier_median: float, k: int) -> pd.DataFrame:
    """Rank artists in a tier by (|n_tracks - median|, -coherence, -n_tracks)."""
    t = tier_pool.copy()
    t["dist"] = (t["n_tracks"] - tier_median).abs()
    t["neg_coh"] = -t["genre_coherence"]
    t["neg_n"] = -t["n_tracks"]
    return t.sort_values(["dist", "neg_coh", "neg_n"]).head(k).reset_index(drop=True)


def pack_candidate(row: pd.Series, tracks_map: dict[str, list[dict]]) -> dict:
    aid = row["artist_id"]
    tracks = tracks_map.get(aid, [])
    return {
        "artist_id": aid,
        "n_tracks": int(row["n_tracks"]),
        "primary_genre": row["primary_genre"],
        "genre_coherence": float(row["genre_coherence"]),
        "total_duration_s": float(row["total_duration_s"]),
        "n_albums": int(row["n_albums"]),
        "tracks": tracks,  # full per-track metadata for Phase 1 staging
    }


def main() -> None:
    artists = pd.read_csv(ARTISTS_CSV)
    pool = artists[
        (artists["n_tracks"] >= MIN_TRACKS)
        & (artists["genre_coherence"] >= COHERENCE_FLOOR)
    ].copy()
    q33, q66 = np.quantile(pool["n_tracks"], [1 / 3, 2 / 3])
    pool["tercile"] = pd.cut(
        pool["n_tracks"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["low", "mid", "high"],
    )

    print(f"[fallbacks] coherent pool size: {len(pool)}")
    print(f"[fallbacks] tercile cuts: q33={q33:.0f}, q66={q66:.0f}")

    tracks_map = load_artist_tracks(META_TSV)
    print(f"[fallbacks] loaded track lists for {len(tracks_map):,} artists")

    # Per-tier ranked list (pre-genre-diversity)
    raw_ranked = {}
    for tier in ["low", "mid", "high"]:
        tier_pool = pool[pool["tercile"] == tier]
        tier_median = tier_pool["n_tracks"].median()
        raw_ranked[tier] = rank_tier(tier_pool, tier_median, TOP_K_PER_TIER)
        print(f"[fallbacks] tier={tier} median={tier_median:.0f} "
              f"top-{TOP_K_PER_TIER} artists: "
              f"{raw_ranked[tier]['artist_id'].tolist()}")

    # Greedy genre-diverse primary selection across tiers, fallbacks unconstrained.
    # Process tiers in fixed order; each tier's primary = first un-used-genre entry;
    # if all top-K share an already-used genre, relax and take rank-1.
    primary = {}
    used_genres: set[str] = set()
    for tier in ["low", "mid", "high"]:
        tier_ranked = raw_ranked[tier]
        chosen_idx = None
        for idx, row in tier_ranked.iterrows():
            if row["primary_genre"] not in used_genres:
                chosen_idx = idx
                break
        if chosen_idx is None:
            chosen_idx = 0  # relax: just take the best-ranked
            print(f"[fallbacks] WARN: tier={tier} genre-diversity relaxed")
        primary[tier] = tier_ranked.iloc[chosen_idx]
        used_genres.add(primary[tier]["primary_genre"])

    # Build output structure
    out = {
        "pool_meta": {
            "n_pool": int(len(pool)),
            "min_tracks": MIN_TRACKS,
            "coherence_floor": COHERENCE_FLOOR,
            "tercile_cuts": {"q33": float(q33), "q66": float(q66)},
            "top_k_per_tier": TOP_K_PER_TIER,
        },
    }
    for tier in ["low", "mid", "high"]:
        primary_row = primary[tier]
        tier_ranked = raw_ranked[tier]
        # Fallbacks = the rest of the top-K in that tier, excluding the primary
        fallback_rows = [
            r for _, r in tier_ranked.iterrows()
            if r["artist_id"] != primary_row["artist_id"]
        ]
        out[tier] = {
            "primary": pack_candidate(primary_row, tracks_map),
            "fallbacks": [pack_candidate(r, tracks_map) for r in fallback_rows],
        }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[fallbacks] wrote {OUT_JSON}")

    # Console summary
    for tier in ["low", "mid", "high"]:
        p = out[tier]["primary"]
        fb = out[tier]["fallbacks"]
        print(f"  {tier}: primary={p['artist_id']} ({p['primary_genre']}, "
              f"{p['n_tracks']} tracks); fallbacks="
              f"{[f['artist_id'] + '(' + f['primary_genre'] + ',' + str(f['n_tracks']) + ')' for f in fb]}")


if __name__ == "__main__":
    main()
