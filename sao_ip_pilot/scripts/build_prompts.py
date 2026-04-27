#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from common import load_config, read_csv_rows, resolve_repo_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fixed prompt JSON for SAO query generation.")
    parser.add_argument("--config", default="sao_ip_pilot/scripts/pilot.yaml")
    parser.add_argument("--selected-tracks-csv", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    selected_tracks_csv = resolve_repo_path(
        args.selected_tracks_csv or Path(cfg["selection_dir"]) / "selected_tracks.csv"
    )
    output_path = resolve_repo_path(args.output_path or cfg["prompts_json"])
    num_prompts = int(args.num_prompts or cfg["num_prompts"])
    seed = int(args.seed if args.seed is not None else cfg["seed"])

    rows = read_csv_rows(selected_tracks_csv)
    if len(rows) < num_prompts:
        raise ValueError(
            f"Requested {num_prompts} prompts but only {len(rows)} selected tracks are available. "
            "Lower num_prompts or increase tracks_per_seller / n_sellers."
        )

    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    rows = rows[:num_prompts]

    sample_rate = 44100
    sample_size = 2097152
    seconds_total = sample_size / sample_rate
    prompts = []
    for prompt_id, row in enumerate(rows):
        prompts.append(
            {
                "prompt_id": prompt_id,
                "prompt": row["prompt"],
                "seconds_start": 0.0,
                "seconds_total": seconds_total,
                "source_path": row["path"],
                "source_relpath": row["relpath"],
                "source_pool_index": int(row["pool_index"]),
                "source_global_index": int(row["global_index"]),
                "source_track_id": int(row["track_id"]) if str(row["track_id"]).isdigit() else row["track_id"],
                "source_seller_id": row["seller_id"],
                "source_artist_id": row["artist_id"],
                "source_primary_genre": row["primary_genre"],
            }
        )

    write_json(
        output_path,
        {
            "seed": seed,
            "num_prompts": len(prompts),
            "sample_rate": sample_rate,
            "sample_size": sample_size,
            "seconds_total": seconds_total,
            "prompts": prompts,
        },
    )
    print(f"[prompts] wrote {len(prompts)} prompts -> {output_path}")


if __name__ == "__main__":
    main()
