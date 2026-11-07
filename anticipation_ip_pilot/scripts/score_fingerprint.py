#!/usr/bin/env python3
"""Chromaprint (fpcalc / pyacoustid) similarity as train×query attribution scores (CPU)."""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import acoustid

from common import resolve_repo_path

# Prefer a user-local fpcalc if present (Clusters often lack system chromaprint).
_DEFAULT_FPCALC = Path.home() / "local" / "chromaprint" / "fpcalc"
if _DEFAULT_FPCALC.is_file():
    os.environ.setdefault("FPCALC", str(_DEFAULT_FPCALC))


def list_wavs(directory: Path, prefix: str) -> list[Path]:
    files = sorted(directory.glob(f"{prefix}*.wav"))
    if not files:
        raise FileNotFoundError(f"No {prefix}*.wav under {directory}")
    return files


def fpcalc_raw(path: Path, length: int) -> list[int]:
    fpcalc = os.environ.get("FPCALC", acoustid.FPCALC_COMMAND)
    try:
        out = subprocess.check_output(
            [fpcalc, "-raw", "-length", str(length), str(path)],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"[fingerprint] failed {path}: {exc}")
        return []
    for line in out.splitlines():
        if not line.startswith(b"FINGERPRINT="):
            continue
        payload = line.split(b"=", 1)[1]
        if not payload:
            return []
        return [int(x) for x in payload.split(b",") if x]
    return []


def _compare_one(args: tuple[int, list[int]]) -> tuple[int, np.ndarray]:
    q_idx, q_fp = args
    train_fps = _WORKER_TRAIN_FPS
    if not q_fp:
        return q_idx, np.zeros(len(train_fps), dtype=np.float32)
    col = np.empty(len(train_fps), dtype=np.float32)
    for i, t_fp in enumerate(train_fps):
        if not t_fp:
            col[i] = 0.0
        else:
            col[i] = float(acoustid._match_fingerprints(t_fp, q_fp))
    return q_idx, col


_WORKER_TRAIN_FPS: list[list[int]] = []


def _init_worker(train_fps: list[list[int]]) -> None:
    global _WORKER_TRAIN_FPS
    _WORKER_TRAIN_FPS = train_fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Chromaprint fingerprint similarity scores.")
    parser.add_argument(
        "--audio-root",
        default="anticipation_ip_pilot/outputs/baselines/clap/audio",
        help="Reuse rendered MIDI audio from the CLAP baseline.",
    )
    parser.add_argument(
        "--output-path",
        default="anticipation_ip_pilot/outputs/baselines/fingerprint/score_fingerprint.pt",
    )
    parser.add_argument("--length", type=int, default=30, help="Seconds fed to fpcalc.")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    audio_root = resolve_repo_path(args.audio_root)
    train_paths = list_wavs(audio_root / "train", "train_")
    query_paths = list_wavs(audio_root / "queries", "query_")
    print(
        f"[fingerprint] train={len(train_paths)} queries={len(query_paths)} "
        f"length={args.length}s workers={args.workers} fpcalc={os.environ.get('FPCALC', 'fpcalc')}"
    )

    train_fps = [fpcalc_raw(p, args.length) for p in tqdm(train_paths, desc="fp-train")]
    query_fps = [fpcalc_raw(p, args.length) for p in tqdm(query_paths, desc="fp-query")]

    score = np.zeros((len(train_fps), len(query_fps)), dtype=np.float32)
    jobs = [(j, query_fps[j]) for j in range(len(query_fps))]
    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(train_fps,),
    ) as pool:
        futures = [pool.submit(_compare_one, job) for job in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="compare"):
            j, col = fut.result()
            score[:, j] = col

    out = resolve_repo_path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(score)
    torch.save(tensor, out)
    # Lightweight cache for debugging / reuse.
    np.savez_compressed(
        out.parent / "fingerprints_raw.npz",
        train_lengths=np.array([len(x) for x in train_fps], dtype=np.int32),
        query_lengths=np.array([len(x) for x in query_fps], dtype=np.int32),
    )
    print(
        f"[fingerprint] wrote {out} shape={tuple(tensor.shape)} "
        f"min={float(tensor.min()):.4f} max={float(tensor.max()):.4f} mean={float(tensor.mean()):.4f}"
    )


if __name__ == "__main__":
    main()
