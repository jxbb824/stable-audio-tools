#!/usr/bin/env python3
"""
MTG-Jamendo grouping EDA
========================

Metadata-only analysis to inform pilot seller selection:
  - prolificness distribution (tracks per artist)
  - genre distribution
  - per-artist genre coherence (how mono-genre is a catalog?)
  - prolificness x genre cross-tab

Also picks N=3 candidate pilot sellers (low/mid/high prolificness,
genre-coherent) and writes a draft sellers_candidates.json.

Inputs:
  ../../data/mtg_jamendo_metadata/autotagging.tsv

Outputs:
  ../../data/figures/*.png
  ../../data/stats/grouping_summary.json
  ../../data/stats/pilot_sellers_candidates.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "data" / "mtg_jamendo_metadata" / "autotagging.tsv"
FIG_DIR = ROOT / "data" / "figures"
STATS_DIR = ROOT / "data" / "stats"
FIG_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)


def load_autotagging(path: Path) -> pd.DataFrame:
    """Load autotagging.tsv with TAGS split into genre / mood / instrument lists."""
    rows = []
    with open(path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        assert header[:5] == ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION"], header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            track_id, artist_id, album_id, rel_path, duration = parts[:5]
            tags = parts[5:]  # TAGS column is \t-separated too
            genres = [t.split("---", 1)[1] for t in tags if t.startswith("genre---")]
            moods = [t.split("---", 1)[1] for t in tags if t.startswith("mood/theme---")]
            insts = [t.split("---", 1)[1] for t in tags if t.startswith("instrument---")]
            rows.append({
                "track_id": track_id,
                "artist_id": artist_id,
                "album_id": album_id,
                "path": rel_path,
                "duration": float(duration),
                "genres": genres,
                "moods": moods,
                "instruments": insts,
            })
    return pd.DataFrame(rows)


def per_artist_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tracks to one row per artist."""
    rows = []
    for aid, grp in df.groupby("artist_id"):
        all_genres = [g for gs in grp["genres"] for g in gs]
        if all_genres:
            genre_counter = Counter(all_genres)
            primary_genre, primary_count = genre_counter.most_common(1)[0]
            genre_coherence = primary_count / len(all_genres)
            unique_genres = len(genre_counter)
        else:
            primary_genre, genre_coherence, unique_genres = "unknown", 0.0, 0
        rows.append({
            "artist_id": aid,
            "n_tracks": len(grp),
            "total_duration_s": grp["duration"].sum(),
            "mean_duration_s": grp["duration"].mean(),
            "n_albums": grp["album_id"].nunique(),
            "primary_genre": primary_genre,
            "genre_coherence": genre_coherence,
            "n_unique_genres": unique_genres,
        })
    return pd.DataFrame(rows).sort_values("n_tracks", ascending=False).reset_index(drop=True)


def plot_prolificness(artists: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # (a) log-binned histogram
    ax = axes[0]
    counts = artists["n_tracks"].values
    bins = np.logspace(0, np.log10(counts.max() + 1), 40)
    ax.hist(counts, bins=bins, color="#3f6cd6", edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("tracks per artist (log)")
    ax.set_ylabel("# artists")
    ax.set_title("(a) Prolificness distribution")

    # (b) CDF with tercile lines
    ax = axes[1]
    sorted_c = np.sort(counts)
    cdf = np.arange(1, len(sorted_c) + 1) / len(sorted_c)
    ax.plot(sorted_c, cdf, color="#3f6cd6", lw=1.8)
    ax.set_xscale("log")
    q33, q66 = np.quantile(counts, [1 / 3, 2 / 3])
    for q, label in [(q33, f"33% = {q33:.0f}"), (q66, f"66% = {q66:.0f}")]:
        ax.axvline(q, color="#d64545", ls="--", lw=1)
        ax.text(q * 1.05, 0.08, label, color="#d64545", fontsize=9)
    ax.set_xlabel("tracks per artist (log)")
    ax.set_ylabel("CDF")
    ax.set_title("(b) CDF with prolificness terciles")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "prolificness.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_genre(artists: pd.DataFrame, top_k: int = 15) -> None:
    genre_counts = artists["primary_genre"].value_counts().head(top_k)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # (a) top-K genres by artist count
    ax = axes[0]
    sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax,
                color="#3f6cd6", edgecolor="white")
    ax.set_xlabel("# artists (primary genre)")
    ax.set_ylabel("")
    ax.set_title(f"(a) Top-{top_k} genres by artist count")

    # (b) prolificness distribution within top genres
    ax = axes[1]
    top_artists = artists[artists["primary_genre"].isin(genre_counts.index)]
    order = genre_counts.index.tolist()
    sns.boxplot(data=top_artists, x="n_tracks", y="primary_genre",
                order=order, ax=ax, color="#3f6cd6",
                fliersize=2, linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("tracks per artist (log)")
    ax.set_ylabel("")
    ax.set_title(f"(b) Prolificness by genre (top-{top_k})")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "genre.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_coherence(artists: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.hist(artists["genre_coherence"], bins=30,
            color="#3f6cd6", edgecolor="white")
    ax.set_xlabel("genre coherence (fraction of tags = primary genre)")
    ax.set_ylabel("# artists")
    ax.set_title("(a) Catalog genre coherence")

    ax = axes[1]
    ax.scatter(artists["n_tracks"], artists["genre_coherence"],
               s=6, alpha=0.3, color="#3f6cd6")
    ax.set_xscale("log")
    ax.set_xlabel("tracks per artist (log)")
    ax.set_ylabel("genre coherence")
    ax.set_title("(b) Coherence vs. prolificness")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "coherence.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def select_pilot_candidates(
    artists: pd.DataFrame,
    min_tracks: int = 10,
    coherence_floor: float = 0.70,
) -> dict:
    """Pick N=3 candidates: low / mid / high prolificness, coherent catalog.

    Constraints:
      - n_tracks >= min_tracks (need enough tracks to fine-tune + stage 10)
      - genre_coherence >= coherence_floor (single-genre enough to analyze)
      - 3 distinct primary_genres (diversity across picks)
    """
    pool = artists[
        (artists["n_tracks"] >= min_tracks)
        & (artists["genre_coherence"] >= coherence_floor)
    ].copy()
    q33, q66 = np.quantile(pool["n_tracks"], [1 / 3, 2 / 3])

    pool["tercile"] = pd.cut(pool["n_tracks"],
                             bins=[-np.inf, q33, q66, np.inf],
                             labels=["low", "mid", "high"])

    picks = {}
    used_genres = set()
    # For each tercile pick the artist closest to the median of that tercile
    # among artists whose primary_genre hasn't been used yet.
    for tier in ["low", "mid", "high"]:
        tier_pool = pool[pool["tercile"] == tier].copy()
        tier_pool = tier_pool[~tier_pool["primary_genre"].isin(used_genres)]
        if tier_pool.empty:
            # relax genre-diversity constraint if we must
            tier_pool = pool[pool["tercile"] == tier].copy()
        target = tier_pool["n_tracks"].median()
        tier_pool["dist"] = (tier_pool["n_tracks"] - target).abs()
        # tie-break: higher coherence wins
        chosen = tier_pool.sort_values(
            ["dist", "genre_coherence"], ascending=[True, False]
        ).iloc[0]
        picks[tier] = {
            "artist_id": chosen["artist_id"],
            "n_tracks": int(chosen["n_tracks"]),
            "total_duration_s": float(chosen["total_duration_s"]),
            "primary_genre": chosen["primary_genre"],
            "genre_coherence": float(chosen["genre_coherence"]),
            "n_albums": int(chosen["n_albums"]),
        }
        used_genres.add(chosen["primary_genre"])

    picks["pool_meta"] = {
        "min_tracks": min_tracks,
        "coherence_floor": coherence_floor,
        "pool_size": int(len(pool)),
        "tercile_cuts": {"q33": float(q33), "q66": float(q66)},
    }
    return picks


def plot_picks(artists: pd.DataFrame, picks: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(artists["n_tracks"], artists["genre_coherence"],
               s=5, alpha=0.2, color="#999999", label="all artists")
    for tier, colour in [("low", "#2ca02c"), ("mid", "#ff7f0e"), ("high", "#d62728")]:
        p = picks[tier]
        ax.scatter(p["n_tracks"], p["genre_coherence"],
                   s=130, color=colour, edgecolor="black", linewidth=1.2,
                   label=f"{tier}: {p['artist_id']} ({p['primary_genre']}, {p['n_tracks']} tracks)",
                   zorder=3)
    ax.axhline(picks["pool_meta"]["coherence_floor"], color="k",
               ls="--", lw=0.8, alpha=0.5)
    ax.axvline(picks["pool_meta"]["min_tracks"], color="k",
               ls="--", lw=0.8, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("tracks per artist (log)")
    ax.set_ylabel("genre coherence")
    ax.set_title("Candidate pilot sellers (N=3)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pilot_candidates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    print(f"[eda] reading {META_PATH}")
    df = load_autotagging(META_PATH)
    print(f"[eda] tracks: {len(df):,}")

    artists = per_artist_summary(df)
    print(f"[eda] artists: {len(artists):,}")

    # Summary stats
    counts = artists["n_tracks"].values
    durations = artists["total_duration_s"].values
    coherence = artists["genre_coherence"].values
    primary_genres = artists["primary_genre"].value_counts()

    summary = {
        "n_tracks": int(len(df)),
        "n_artists": int(len(artists)),
        "n_albums": int(df["album_id"].nunique()),
        "n_unique_genres": int(primary_genres.shape[0]),
        "prolificness": {
            "mean": float(counts.mean()),
            "median": float(np.median(counts)),
            "max": int(counts.max()),
            "q25": float(np.quantile(counts, 0.25)),
            "q75": float(np.quantile(counts, 0.75)),
            "q33": float(np.quantile(counts, 1 / 3)),
            "q66": float(np.quantile(counts, 2 / 3)),
            "share_artists_>=10_tracks": float((counts >= 10).mean()),
            "share_artists_>=30_tracks": float((counts >= 30).mean()),
        },
        "duration_hours": {
            "total": float(durations.sum() / 3600),
            "mean_per_artist": float(durations.mean() / 60),  # minutes
        },
        "coherence": {
            "mean": float(coherence.mean()),
            "median": float(np.median(coherence)),
            "share_artists_>=0.7": float((coherence >= 0.7).mean()),
            "share_artists_==1.0": float((coherence == 1.0).mean()),
        },
        "top10_primary_genres": primary_genres.head(10).to_dict(),
    }

    with open(STATS_DIR / "grouping_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eda] wrote {STATS_DIR / 'grouping_summary.json'}")

    # Plots
    plot_prolificness(artists)
    plot_genre(artists)
    plot_coherence(artists)
    print(f"[eda] wrote 3 plots to {FIG_DIR}")

    # Pilot candidates
    picks = select_pilot_candidates(artists)
    with open(STATS_DIR / "pilot_sellers_candidates.json", "w") as f:
        json.dump(picks, f, indent=2)
    plot_picks(artists, picks)
    print(f"[eda] wrote {STATS_DIR / 'pilot_sellers_candidates.json'}")
    print(f"[eda] picks: low={picks['low']['artist_id']} "
          f"mid={picks['mid']['artist_id']} high={picks['high']['artist_id']}")

    # Also save the full per-artist table for downstream use
    artists.to_csv(STATS_DIR / "artists_summary.csv", index=False)
    print(f"[eda] wrote {STATS_DIR / 'artists_summary.csv'}")


if __name__ == "__main__":
    main()
