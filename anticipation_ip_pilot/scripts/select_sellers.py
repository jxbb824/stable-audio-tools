#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict

import numpy as np

from common import ensure_dir, load_config, resolve_repo_path, set_all_seeds, write_csv, write_json


def read_metadata(path: str) -> dict[str, dict[str, str]]:
    import csv

    out: dict[str, dict[str, str]] = {}
    with resolve_repo_path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            track_id = row.get("ID", "")
            if track_id:
                out[track_id] = row
    return out


def load_train_index(train_file: str, metadata_csv: str) -> tuple[list[dict[str, str]], dict[str, list[int]]]:
    metadata = read_metadata(metadata_csv)
    rows: list[dict[str, str]] = []
    artist_to_indices: dict[str, list[int]] = defaultdict(list)
    with resolve_repo_path(train_file).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            parts = line.strip().split()
            if not parts:
                continue
            track_id = parts[-1]
            meta = metadata.get(track_id, {})
            artist = meta.get("artist", "")
            song = meta.get("song", "")
            row = {
                "line_index": str(idx),
                "track_id": track_id,
                "artist": artist,
                "song": song,
            }
            rows.append(row)
            if artist:
                artist_to_indices[artist].append(idx)
    return rows, artist_to_indices


def choose_artists(
    artist_to_indices: dict[str, list[int]],
    n_sellers: int,
    min_count: int,
    seed: int,
    mean: float | None,
    std: float | None,
    count_min: int | None,
    count_max: int | None,
) -> list[tuple[str, list[int]]]:
    eligible = [
        (artist, indices)
        for artist, indices in artist_to_indices.items()
        if len(indices) >= min_count
        and (count_min is None or len(indices) >= count_min)
        and (count_max is None or len(indices) <= count_max)
    ]
    if len(eligible) < n_sellers:
        raise ValueError(
            f"Only {len(eligible)} eligible artists; need {n_sellers}. "
            "Lower min_train_examples_per_seller or relax count bounds."
        )

    counts = np.array([len(indices) for _, indices in eligible], dtype=np.float64)
    target_mean = float(np.median(counts) if mean is None else mean)
    target_std = float(np.std(counts) * 0.5 if std is None else std)
    if target_std <= 0:
        target_std = 1.0

    rng = np.random.default_rng(seed)
    target_counts = rng.normal(target_mean, target_std, size=n_sellers)
    target_counts = np.clip(target_counts, counts.min(), counts.max())

    remaining = {artist: indices for artist, indices in eligible}
    chosen: list[tuple[str, list[int]]] = []
    for target in sorted(target_counts):
        best_artist = min(
            remaining,
            key=lambda artist: (abs(len(remaining[artist]) - target), artist),
        )
        chosen.append((best_artist, remaining.pop(best_artist)))
    return chosen


def band_for_count(count: int, qs: np.ndarray) -> str:
    if count <= qs[0]:
        return "low"
    if count <= qs[1]:
        return "mid"
    if count <= qs[2]:
        return "high"
    return "extra_high"


def main() -> None:
    parser = argparse.ArgumentParser(description="Select artist sellers from TheoryTab train_v2.")
    parser.add_argument("--config", default="anticipation_ip_pilot/scripts/pilot.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_all_seeds(int(cfg.get("seed", 42)))

    selection_dir = ensure_dir(cfg["selection_dir"])
    train_rows, artist_to_indices = load_train_index(cfg["train_file"], cfg["metadata_csv"])
    selected = choose_artists(
        artist_to_indices=artist_to_indices,
        n_sellers=int(cfg["n_sellers"]),
        min_count=int(cfg["min_train_examples_per_seller"]),
        seed=int(cfg.get("seed", 42)),
        mean=cfg.get("seller_count_mean"),
        std=cfg.get("seller_count_std"),
        count_min=cfg.get("seller_count_min"),
        count_max=cfg.get("seller_count_max"),
    )

    selected_counts = np.array([len(indices) for _, indices in selected], dtype=np.float64)
    qs = np.quantile(selected_counts, [0.25, 0.5, 0.75])
    seller_by_artist = {artist: f"seller_{idx + 1:03d}" for idx, (artist, _) in enumerate(selected)}

    seller_rows = []
    for seller_index, (artist, indices) in enumerate(selected, start=1):
        track_ids = [train_rows[i]["track_id"] for i in indices]
        count = len(indices)
        seller_rows.append(
            {
                "seller_id": seller_by_artist[artist],
                "seller_index": seller_index,
                "artist": artist,
                "num_train_examples": count,
                "num_distinct_ids": len(set(track_ids)),
                "prolificness_band": band_for_count(count, qs),
                "train_indices": " ".join(str(i) for i in indices),
                "exclude_indices": ",".join(str(i) for i in indices),
                "track_ids": " ".join(track_ids),
            }
        )

    candidate_rows = []
    for row in train_rows:
        artist = row["artist"]
        if artist not in seller_by_artist:
            continue
        candidate_rows.append(
            {
                "line_index": row["line_index"],
                "seller_id": seller_by_artist[artist],
                "artist": artist,
                "track_id": row["track_id"],
                "song": row["song"],
            }
        )

    train_manifest_rows = []
    missing_metadata = 0
    for row in train_rows:
        artist = row["artist"]
        if not artist:
            missing_metadata += 1
        train_manifest_rows.append(
            {
                "line_index": row["line_index"],
                "track_id": row["track_id"],
                "artist": artist,
                "song": row["song"],
                "seller_id": seller_by_artist.get(artist, ""),
            }
        )

    write_csv(
        selection_dir / "seller_manifest.csv",
        [
            "seller_id",
            "seller_index",
            "artist",
            "num_train_examples",
            "num_distinct_ids",
            "prolificness_band",
            "train_indices",
            "exclude_indices",
            "track_ids",
        ],
        seller_rows,
    )
    write_csv(selection_dir / "candidate_pool.csv", ["line_index", "seller_id", "artist", "track_id", "song"], candidate_rows)
    write_csv(selection_dir / "train_manifest.csv", ["line_index", "track_id", "artist", "song", "seller_id"], train_manifest_rows)
    write_json(
        selection_dir / "selection_summary.json",
        {
            "n_sellers": len(seller_rows),
            "total_train_rows": len(train_rows),
            "selected_train_rows": len(candidate_rows),
            "missing_metadata_rows": missing_metadata,
            "selected_count_min": int(selected_counts.min()),
            "selected_count_median": float(np.median(selected_counts)),
            "selected_count_max": int(selected_counts.max()),
            "artist_count_top10": Counter({a: len(i) for a, i in artist_to_indices.items()}).most_common(10),
        },
    )
    print(f"[select] wrote {selection_dir}")


if __name__ == "__main__":
    main()
