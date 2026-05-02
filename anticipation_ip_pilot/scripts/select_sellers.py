#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from common import ensure_dir, load_config, resolve_repo_path, set_all_seeds, write_csv, write_json

BANDS_ORDER = ("low", "mid", "high", "extra_high")


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


def stratum_for_count(n: int, qs: np.ndarray) -> str:
    if n <= qs[0]:
        return "low"
    if n <= qs[1]:
        return "mid"
    if n <= qs[2]:
        return "high"
    return "extra_high"


def bucket_eligible_by_quartiles(
    eligible: list[tuple[str, list[int]]],
) -> tuple[dict[str, list[tuple[str, list[int]]]], dict[str, str], np.ndarray]:
    counts = np.array([len(indices) for _, indices in eligible], dtype=np.float64)
    qs = np.quantile(counts, [0.25, 0.5, 0.75])
    buckets: dict[str, list[tuple[str, list[int]]]] = {b: [] for b in BANDS_ORDER}
    artist_stratum: dict[str, str] = {}
    for artist, indices in eligible:
        band = stratum_for_count(len(indices), qs)
        buckets[band].append((artist, indices))
        artist_stratum[artist] = band
    return buckets, artist_stratum, qs


def reconcile_quotas_to_caps(
    band_quotas: dict[str, int],
    caps: dict[str, int],
    n_sellers: int,
) -> tuple[dict[str, int], int]:
    """Cap targets by bucket sizes; assign deficit greedily by quota-vs-fill priority."""
    assigned = {b: min(int(band_quotas[b]), caps[b]) for b in BANDS_ORDER}
    deficit = n_sellers - sum(assigned.values())
    if deficit < 0:
        raise ValueError("Bucket caps sum below target assignments; check seller_band_quotas.")
    slack = {b: caps[b] - assigned[b] for b in BANDS_ORDER}
    slack_total = sum(slack.values())
    if deficit > slack_total:
        raise ValueError(
            f"Cannot fill {n_sellers} sellers: deficit={deficit} after capping quotas "
            f"but only slack_total={slack_total} seats remain across strata."
        )
    band_idx = {b: i for i, b in enumerate(BANDS_ORDER)}
    while deficit > 0:
        candidates = [b for b in BANDS_ORDER if slack[b] > 0]
        if not candidates:
            raise RuntimeError("Deficit remains but no stratum slack (internal error).")
        scores = []
        for b in candidates:
            prio = float(band_quotas[b]) / float(assigned[b] + 1)
            scores.append((prio, -band_idx[b], b))
        best = max(scores)[2]
        assigned[best] += 1
        slack[best] -= 1
        deficit -= 1
    return assigned, 0


def choose_artists_stratified(
    artist_to_indices: dict[str, list[int]],
    n_sellers: int,
    min_count: int,
    seed: int,
    band_quotas: dict[str, int],
    count_min: int | None,
    count_max: int | None,
) -> tuple[list[tuple[str, list[int]]], dict[str, str], dict[str, Any]]:
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

    quota_sum = sum(int(band_quotas[b]) for b in BANDS_ORDER)
    if quota_sum != n_sellers:
        raise ValueError(f"seller_band_quotas must sum to n_sellers={n_sellers}; got {quota_sum}.")

    buckets, artist_stratum, qs = bucket_eligible_by_quartiles(eligible)
    caps = {b: len(buckets[b]) for b in BANDS_ORDER}
    adjusted_quotas, _ = reconcile_quotas_to_caps(band_quotas, caps, n_sellers)

    rng = np.random.default_rng(seed)
    chosen: list[tuple[str, list[int]]] = []
    picked_per_band: dict[str, int] = {b: 0 for b in BANDS_ORDER}

    for band in BANDS_ORDER:
        pool = list(buckets[band])
        perm = rng.permutation(len(pool))
        shuffled = [pool[int(i)] for i in perm]
        take = int(adjusted_quotas[band])
        assert take <= len(shuffled)
        for artist, indices in shuffled[:take]:
            chosen.append((artist, indices))
            picked_per_band[band] += 1

    remainder_fill = 0

    meta = {
        "pool_quartiles": {"q25": float(qs[0]), "q50": float(qs[1]), "q75": float(qs[2])},
        "seller_band_quotas_requested": {b: int(band_quotas[b]) for b in BANDS_ORDER},
        "seller_band_quotas_effective": adjusted_quotas,
        "picked_per_band": picked_per_band,
        "remainder_fill": remainder_fill,
        "bucket_sizes": {b: len(buckets[b]) for b in BANDS_ORDER},
    }
    return chosen, artist_stratum, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Select artist sellers from TheoryTab train_v2.")
    parser.add_argument("--config", default="anticipation_ip_pilot/scripts/pilot.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_all_seeds(int(cfg.get("seed", 42)))

    selection_dir = ensure_dir(cfg["selection_dir"])
    train_rows, artist_to_indices = load_train_index(cfg["train_file"], cfg["metadata_csv"])
    n_sellers = int(cfg["n_sellers"])
    raw_quotas = cfg.get("seller_band_quotas") or {}
    band_quotas = {b: int(raw_quotas[b]) for b in BANDS_ORDER}

    selected, artist_stratum, strat_meta = choose_artists_stratified(
        artist_to_indices=artist_to_indices,
        n_sellers=n_sellers,
        min_count=int(cfg["min_train_examples_per_seller"]),
        seed=int(cfg.get("seed", 42)),
        band_quotas=band_quotas,
        count_min=cfg.get("seller_count_min"),
        count_max=cfg.get("seller_count_max"),
    )

    selected_counts = np.array([len(indices) for _, indices in selected], dtype=np.float64)
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
                "prolificness_band": artist_stratum[artist],
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
    eligible_n = sum(
        1
        for a, ix in artist_to_indices.items()
        if len(ix) >= int(cfg["min_train_examples_per_seller"])
        and (cfg.get("seller_count_min") is None or len(ix) >= int(cfg["seller_count_min"]))
        and (cfg.get("seller_count_max") is None or len(ix) <= int(cfg["seller_count_max"]))
    )
    selected_band_hist = Counter(artist_stratum[artist] for artist, _ in selected)

    write_json(
        selection_dir / "selection_summary.json",
        {
            "selection_method": "stratified_quartiles",
            "n_sellers": len(seller_rows),
            "min_train_examples_per_seller": int(cfg["min_train_examples_per_seller"]),
            "seller_band_quotas": band_quotas,
            "eligible_artists": eligible_n,
            "total_train_rows": len(train_rows),
            "selected_train_rows": len(candidate_rows),
            "missing_metadata_rows": missing_metadata,
            "selected_count_min": int(selected_counts.min()),
            "selected_count_median": float(np.median(selected_counts)),
            "selected_count_max": int(selected_counts.max()),
            "selected_band_counts": dict(selected_band_hist),
            **strat_meta,
            "artist_count_top10": Counter({a: len(i) for a, i in artist_to_indices.items()}).most_common(10),
        },
    )
    print(f"[select] wrote {selection_dir}")
    print(
        f"[select] eligible={eligible_n} bands={dict(selected_band_hist)} "
        f"remainder_fill={strat_meta['remainder_fill']} quartiles={strat_meta['pool_quartiles']}"
    )


if __name__ == "__main__":
    main()
