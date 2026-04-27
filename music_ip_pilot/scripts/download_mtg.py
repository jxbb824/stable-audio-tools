#!/usr/bin/env python3
"""
MTG-Jamendo direct track downloader
===================================

Pulls specific MTG-Jamendo 30-second MP3 clips from the Freesound CDN mirror,
one file at a time. Used instead of HF's `rkstgr/mtg-jamendo` dataset because:

  1. The HF dataset's custom loader is incompatible with `datasets>=4`.
  2. Even in `streaming=False` mode it insists on downloading the full 109 GiB
     dataset; we only need ~67 MP3s (~330 MB) for the pilot.
  3. The HF subset is 13,854 tracks; direct download gives us the full 55,609.

URL pattern:
  https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/{prefix}/{track_id}.mp3
where {prefix} is the first two digits of the numeric track_id (the autotagging
TSV's `PATH` column is exactly `{prefix}/{track_id}.mp3`).

Input (JSON):
  {"low":  {"primary": {...}, "fallbacks": [...]},
   "mid":  {...},
   "high": {...}}
Each candidate has a `tracks` list whose entries include `hf_key` == "<prefix>/<id>"
(the path stem, no extension).

Output: MP3s mirror the MTG structure on disk:
  {raw_dir}/{prefix}/{track_id}.mp3

Usage:
  python download_mtg.py \\
      --candidates <HOME>/music_ip_pilot/data/stats/pilot_sellers_candidates.json \\
      --raw-dir    <DATA_ROOT>/music_ip_pilot/data/mtg_jamendo_raw \\
      --mode primary                # or "all" to include fallbacks

Runs on: cluster `cpu` partition. Idempotent (skips already-present MP3s).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable

import urllib.request
import urllib.error

CDN_BASE = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
RETRY_BACKOFF_S = 2.0


def collect_tracks(candidates: dict, mode: str) -> list[tuple[str, dict]]:
    """Return [(candidate_artist_id, track_dict), ...] to download."""
    assert mode in ("primary", "all"), mode
    out: list[tuple[str, dict]] = []
    for tier in ("low", "mid", "high"):
        block = candidates[tier]
        cands = [block["primary"]]
        if mode == "all":
            cands += block.get("fallbacks", [])
        for c in cands:
            for t in c["tracks"]:
                out.append((c["artist_id"], t))
    return out


def download_one(hf_key: str, dest: Path, retries: int, timeout: int) -> str:
    """Download one track; returns 'ok', 'skipped', or raises on failure."""
    if dest.exists() and dest.stat().st_size > 0:
        return "skipped"

    url = f"{CDN_BASE}/{hf_key}.mp3"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            # urlopen returns a file-like; stream it in chunks
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status} for {url}")
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
            if tmp.stat().st_size == 0:
                raise RuntimeError(f"empty body for {url}")
            tmp.rename(dest)
            return "ok"
        except (urllib.error.URLError, urllib.error.HTTPError,
                TimeoutError, RuntimeError) as e:
            last_err = e
            if attempt < retries:
                time.sleep(RETRY_BACKOFF_S * attempt)
            else:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"failed after {retries} attempts: {url}: {e}"
                ) from e
    # unreachable
    raise RuntimeError(str(last_err))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, type=Path,
                        help="pilot_sellers_candidates.json")
    parser.add_argument("--raw-dir", required=True, type=Path,
                        help="target dir for MP3s (mirrors MTG layout)")
    parser.add_argument("--mode", choices=("primary", "all"), default="primary",
                        help="primary = primaries only (~67 files for pilot); "
                             "all = primaries + fallbacks (~335 files)")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    if not args.candidates.exists():
        sys.exit(f"ERROR: candidates JSON not found: {args.candidates}")

    with open(args.candidates) as f:
        candidates = json.load(f)

    tasks = collect_tracks(candidates, args.mode)
    print(f"[dl-mtg] mode={args.mode} -> {len(tasks)} tracks queued")
    print(f"[dl-mtg] raw_dir: {args.raw_dir}")
    args.raw_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    counts = {"ok": 0, "skipped": 0, "failed": 0}
    bytes_dl = 0
    per_artist: dict[str, int] = {}

    for i, (aid, t) in enumerate(tasks, start=1):
        hf_key = t["hf_key"]  # "14/214"
        dest = args.raw_dir / f"{hf_key}.mp3"
        try:
            status = download_one(hf_key, dest, args.retries, args.timeout)
        except RuntimeError as e:
            counts["failed"] += 1
            print(f"[dl-mtg] [{i}/{len(tasks)}] {aid} {hf_key}: FAIL ({e})",
                  flush=True)
            continue
        counts[status] += 1
        if status == "ok":
            bytes_dl += dest.stat().st_size
        per_artist[aid] = per_artist.get(aid, 0) + (1 if status != "failed" else 0)
        if i % 10 == 0 or i == len(tasks):
            elapsed = time.time() - t0
            print(f"[dl-mtg] [{i}/{len(tasks)}] ok={counts['ok']} "
                  f"skipped={counts['skipped']} failed={counts['failed']} "
                  f"({bytes_dl/1e6:.1f} MB in {elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"[dl-mtg] DONE in {elapsed:.1f}s | "
          f"ok={counts['ok']} skipped={counts['skipped']} "
          f"failed={counts['failed']} | {bytes_dl/1e6:.1f} MB downloaded")
    print(f"[dl-mtg] per-artist totals (present locally):")
    for aid in sorted(per_artist):
        print(f"  {aid}: {per_artist[aid]}")

    if counts["failed"] > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
