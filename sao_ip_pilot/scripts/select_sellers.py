#!/usr/bin/env python3
"""
Select artist sellers from the configured Jamendo subset and write SAO pilot manifests.

This is the SAO version of the collaborator handoff's
eda_groupings.py / generate_fallbacks.py path: artists are sellers, and each
seller contributes a catalog D_j. The output is intentionally simple and is
consumed by the training, EKFAC aggregation, and a* scripts in this folder.
"""
from __future__ import annotations

import argparse
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import ensure_dir, load_config, resolve_repo_path, write_csv, write_json, write_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build seller/category manifests from the configured dataset.")
    parser.add_argument("--config", default="sao_ip_pilot/scripts/pilot.yaml")
    parser.add_argument("--n-sellers", type=int, default=None)
    parser.add_argument("--tracks-per-seller", type=int, default=None)
    parser.add_argument("--min-tracks", type=int, default=None)
    parser.add_argument("--coherence-floor", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def track_num(track_id: str) -> int | None:
    match = re.search(r"(\d+)$", track_id or "")
    return int(match.group(1)) if match else None


def clean_tag(tag: str) -> str:
    return " ".join(tag.replace("_", " ").split())


def split_tags(parts: list[str]) -> dict[str, list[str]]:
    tags_by_type: dict[str, list[str]] = defaultdict(list)
    for raw_tag in parts:
        raw_tag = raw_tag.strip()
        if not raw_tag:
            continue
        if "---" in raw_tag:
            tag_type, tag_value = raw_tag.split("---", 1)
        else:
            tag_type, tag_value = "tag", raw_tag
        tag_value = clean_tag(tag_value)
        if tag_value:
            tags_by_type[tag_type].append(tag_value)
    return dict(tags_by_type)


def load_meta(meta_tsv: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    with meta_tsv.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: idx for idx, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            track_id = parts[index["TRACK_ID"]]
            release_date = parts[index.get("RELEASEDATE", -1)] if "RELEASEDATE" in index else ""
            rows[track_id] = {
                "track_id": track_id,
                "artist_id": parts[index["ARTIST_ID"]],
                "album_id": parts[index["ALBUM_ID"]],
                "title": parts[index.get("TRACK_NAME", -1)] if "TRACK_NAME" in index else "",
                "artist_name": parts[index.get("ARTIST_NAME", -1)] if "ARTIST_NAME" in index else "",
                "album_name": parts[index.get("ALBUM_NAME", -1)] if "ALBUM_NAME" in index else "",
                "year": release_date.split("-", 1)[0] if release_date else "",
            }
    return rows


def load_tracks(raw_tsv: Path, meta_tsv: Path, audio_dir: Path) -> list[dict[str, Any]]:
    meta_by_track = load_meta(meta_tsv)
    rows: list[dict[str, Any]] = []
    with raw_tsv.open("r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if header[:5] != ["TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION"]:
            raise ValueError(f"Unexpected raw TSV header: {header[:5]}")
        for global_index, line in enumerate(handle):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            track_id, artist_id, album_id, relpath, duration = parts[:5]
            abs_path = audio_dir / relpath
            if not abs_path.exists():
                continue
            tags_by_type = split_tags(parts[5:])
            meta = meta_by_track.get(track_id, {})
            rows.append(
                {
                    "global_index": global_index,
                    "track_id": track_id,
                    "track_num": track_num(track_id),
                    "artist_id": artist_id,
                    "artist_name": meta.get("artist_name", ""),
                    "album_id": album_id,
                    "album_name": meta.get("album_name", ""),
                    "title": meta.get("title", ""),
                    "year": meta.get("year", ""),
                    "relpath": os.path.normpath(relpath),
                    "path": str(abs_path.resolve()),
                    "duration": float(duration),
                    "tags_by_type": tags_by_type,
                    "genres": tags_by_type.get("genre", []),
                    "moods": tags_by_type.get("mood/theme", []),
                    "instruments": tags_by_type.get("instrument", []),
                }
            )
    return rows


def summarize_artists(tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for track in tracks:
        grouped[track["artist_id"]].append(track)

    summaries: list[dict[str, Any]] = []
    for artist_id, artist_tracks in grouped.items():
        all_genres = [genre for track in artist_tracks for genre in track["genres"]]
        if all_genres:
            genre_counts = Counter(all_genres)
            primary_genre, primary_count = genre_counts.most_common(1)[0]
            genre_coherence = primary_count / len(all_genres)
        else:
            primary_genre = "unknown"
            genre_coherence = 0.0
        first = artist_tracks[0]
        summaries.append(
            {
                "artist_id": artist_id,
                "artist_name": first.get("artist_name", ""),
                "n_tracks": len(artist_tracks),
                "total_duration_s": sum(float(t["duration"]) for t in artist_tracks),
                "primary_genre": primary_genre,
                "genre_coherence": genre_coherence,
                "n_albums": len({t["album_id"] for t in artist_tracks}),
                "tracks": sorted(artist_tracks, key=lambda t: (t["relpath"], t["track_id"])),
            }
        )
    return sorted(summaries, key=lambda item: (-item["n_tracks"], item["artist_id"]))


def select_artists(
    artists: list[dict[str, Any]],
    seller_cells: list[dict[str, Any]],
    min_tracks: int,
    coherence_floor: float,
    rng: random.Random,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for cell in seller_cells:
        band = str(cell["band"])
        lo = int(cell["min_tracks"])
        hi = cell.get("max_tracks")
        hi_value = None if hi is None else int(hi)
        count = int(cell["count"])
        if lo < min_tracks:
            lo = min_tracks

        candidates = []
        for artist in artists:
            if artist["artist_id"] in used_ids:
                continue
            if artist["genre_coherence"] < coherence_floor:
                continue
            if artist["n_tracks"] < lo:
                continue
            if hi_value is not None and artist["n_tracks"] > hi_value:
                continue
            candidates.append(artist)

        rng.shuffle(candidates)
        if len(candidates) < count:
            raise ValueError(
                f"Cell {band!r} has only {len(candidates)} eligible artists; need {count}. "
                f"Range={lo}-{hi_value if hi_value is not None else 'inf'}, "
                f"coherence_floor={coherence_floor}."
            )

        chosen = candidates[:count]
        for artist in chosen:
            item = dict(artist)
            item["prolificness_band"] = band
            selected.append(item)
            used_ids.add(artist["artist_id"])

    return selected


def build_prompt(track: dict[str, Any], primary_genre: str) -> str:
    pieces: list[str] = []
    if track.get("title"):
        pieces.append(f"title: {track['title']}")
    if track.get("artist_name"):
        pieces.append(f"artist: {track['artist_name']}")
    if track.get("album_name"):
        pieces.append(f"album: {track['album_name']}")
    if track.get("year"):
        pieces.append(f"year: {track['year']}")
    if primary_genre and primary_genre != "unknown":
        pieces.append(f"genre: {primary_genre}")
    if track.get("moods"):
        pieces.append("mood: " + ", ".join(track["moods"][:3]))
    if track.get("instruments"):
        pieces.append("instrument: " + ", ".join(track["instruments"][:3]))
    return ", ".join(pieces) if pieces else track["relpath"]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    n_sellers = int(args.n_sellers or cfg["n_sellers"])
    tracks_per_seller = int(args.tracks_per_seller or cfg["tracks_per_seller"])
    seller_cells = list(cfg["seller_cells"])
    cell_total = sum(int(cell["count"]) for cell in seller_cells)
    if cell_total != n_sellers:
        raise ValueError(f"seller_cells count total={cell_total} does not match n_sellers={n_sellers}.")
    min_tracks = int(args.min_tracks or cfg["min_tracks_per_artist"])
    coherence_floor = float(args.coherence_floor or cfg["coherence_floor"])
    seed = int(args.seed if args.seed is not None else cfg["seed"])

    rng = random.Random(seed)
    raw_tsv = resolve_repo_path(cfg["raw_tsv"])
    meta_tsv = resolve_repo_path(cfg["meta_tsv"])
    audio_dir = resolve_repo_path(cfg["audio_dir"])
    selection_dir = ensure_dir(cfg["selection_dir"])
    models_dir = ensure_dir(cfg["models_dir"])

    tracks = load_tracks(raw_tsv=raw_tsv, meta_tsv=meta_tsv, audio_dir=audio_dir)
    artists = summarize_artists(tracks)
    selected = select_artists(
        artists=artists,
        seller_cells=seller_cells,
        min_tracks=min_tracks,
        coherence_floor=coherence_floor,
        rng=rng,
    )

    sellers_json: list[dict[str, Any]] = []
    seller_rows: list[dict[str, Any]] = []
    track_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    full_paths: list[str] = []

    train_axis_index = 0
    for seller_index, artist in enumerate(selected, start=1):
        seller_id = f"seller_{seller_index:02d}"
        artist_tracks = list(artist["tracks"])
        rng.shuffle(artist_tracks)
        chosen_tracks = sorted(
            artist_tracks[: min(tracks_per_seller, len(artist_tracks))],
            key=lambda t: (t["relpath"], t["track_id"]),
        )

        sellers_json.append(
            {
                "seller_index": seller_index,
                "seller_id": seller_id,
                "artist_id": artist["artist_id"],
                "artist_name": artist["artist_name"],
                "primary_genre": artist["primary_genre"],
                "genre_coherence": artist["genre_coherence"],
                "prolificness_band": artist["prolificness_band"],
                "available_tracks": artist["n_tracks"],
                "selected_tracks": len(chosen_tracks),
            }
        )
        seller_rows.append(sellers_json[-1])

        for track in chosen_tracks:
            prompt = build_prompt(track, artist["primary_genre"])
            row = {
                "train_axis_index": train_axis_index,
                "seller_index": seller_index,
                "seller_id": seller_id,
                "artist_id": artist["artist_id"],
                "artist_name": artist["artist_name"],
                "primary_genre": artist["primary_genre"],
                "genre_coherence": f"{artist['genre_coherence']:.8f}",
                "prolificness_band": artist["prolificness_band"],
                "pool_index": train_axis_index,
                "global_index": track["global_index"],
                "dataset_id": "small_5000",
                "dataset_root": str(audio_dir.resolve()),
                "track_id": track["track_num"] if track["track_num"] is not None else track["track_id"],
                "track_key": track["track_id"],
                "path": track["path"],
                "relpath": track["relpath"],
                "title": track["title"],
                "album_name": track["album_name"],
                "year": track["year"],
                "prompt": prompt,
            }
            track_rows.append(row)
            category_rows.append(
                {
                    "train_axis_index": train_axis_index,
                    "seller_index": seller_index,
                    "seller_id": seller_id,
                    "artist_id": artist["artist_id"],
                    "path": track["path"],
                    "relpath": track["relpath"],
                    "track_id": row["track_id"],
                }
            )
            candidate_rows.append(
                {
                    "pool_index": train_axis_index,
                    "global_index": track["global_index"],
                    "dataset_id": "small_5000",
                    "dataset_root": str(audio_dir.resolve()),
                    "track_id": row["track_id"],
                    "path": track["path"],
                    "relpath": track["relpath"],
                }
            )
            full_paths.append(track["path"])
            train_axis_index += 1

    write_json(selection_dir / "sellers.json", sellers_json)
    write_csv(
        selection_dir / "seller_manifest.csv",
        [
            "seller_index",
            "seller_id",
            "artist_id",
            "artist_name",
            "primary_genre",
            "genre_coherence",
            "prolificness_band",
            "available_tracks",
            "selected_tracks",
        ],
        seller_rows,
    )
    write_csv(
        selection_dir / "selected_tracks.csv",
        [
            "train_axis_index",
            "seller_index",
            "seller_id",
            "artist_id",
            "artist_name",
            "primary_genre",
            "genre_coherence",
            "prolificness_band",
            "pool_index",
            "global_index",
            "dataset_id",
            "dataset_root",
            "track_id",
            "track_key",
            "path",
            "relpath",
            "title",
            "album_name",
            "year",
            "prompt",
        ],
        track_rows,
    )
    write_csv(
        selection_dir / "category_manifest.csv",
        ["train_axis_index", "seller_index", "seller_id", "artist_id", "path", "relpath", "track_id"],
        category_rows,
    )
    write_csv(
        selection_dir / "candidate_pool.csv",
        ["pool_index", "global_index", "dataset_id", "dataset_root", "track_id", "path", "relpath"],
        candidate_rows,
    )

    # Model id 0 is the full model; ids 1..N are LOO models.
    full_model_dir = ensure_dir(models_dir / "0")
    write_csv(
        full_model_dir / "candidate_pool.csv",
        ["pool_index", "global_index", "dataset_id", "dataset_root", "track_id", "path", "relpath"],
        candidate_rows,
    )
    write_lines(full_model_dir / "selected_paths.txt", full_paths)
    write_lines(selection_dir / "selected_paths_full.txt", full_paths)

    for seller_index in range(1, n_sellers + 1):
        loo_paths = [
            row["path"]
            for row in track_rows
            if int(row["seller_index"]) != seller_index
        ]
        loo_dir = ensure_dir(models_dir / str(seller_index))
        write_lines(loo_dir / "selected_paths.txt", loo_paths)
        write_lines(selection_dir / f"selected_paths_loo_{seller_index}.txt", loo_paths)

    summary = {
        "n_sellers": n_sellers,
        "tracks_per_seller": tracks_per_seller,
        "seller_cells": seller_cells,
        "actual_train_count": len(track_rows),
        "min_tracks": min_tracks,
        "coherence_floor": coherence_floor,
        "seed": seed,
        "selection_dir": str(selection_dir),
        "models_dir": str(models_dir),
    }
    write_json(selection_dir / "selection_summary.json", summary)
    print(f"[select] selected {n_sellers} sellers / {len(track_rows)} tracks")
    print(f"[select] wrote manifests under {selection_dir}")


if __name__ == "__main__":
    main()
