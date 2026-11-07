#!/usr/bin/env python3
"""Chromaprint (fpcalc / pyacoustid) similarity for SAO IP baselines (CPU)."""
from __future__ import annotations

import argparse
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import acoustid

from common import resolve_repo_path

_DEFAULT_FPCALC = Path.home() / "local" / "chromaprint" / "fpcalc"
if _DEFAULT_FPCALC.is_file():
    os.environ.setdefault("FPCALC", str(_DEFAULT_FPCALC))


def resolve_audio_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return resolve_repo_path(raw)


def resolve_query_audio_paths(query_manifest: pd.DataFrame) -> list[Path]:
    """Resolve each query, falling back from stale absolute paths to query_id."""
    paths: list[Path] = []
    for row_index, row in query_manifest.iterrows():
        # query_id is repository-relative and therefore portable across project moves.
        candidates = [row.get("query_id"), row.get("audio_path")]
        resolved = [resolve_audio_path(str(raw)) for raw in candidates if pd.notna(raw) and str(raw)]
        match = next((path for path in resolved if path.is_file()), None)
        if match is None:
            raise FileNotFoundError(
                f"query row {row_index} has no existing audio path; tried "
                + ", ".join(str(path) for path in resolved)
            )
        paths.append(match)
    return paths


def fpcalc_raw(path: Path, length: int) -> list[int]:
    fpcalc = os.environ.get("FPCALC", acoustid.FPCALC_COMMAND)
    try:
        out = subprocess.check_output(
            [fpcalc, "-raw", "-length", str(length), str(path)],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"fpcalc failed for {path}: {exc}") from exc
    for line in out.splitlines():
        if not line.startswith(b"FINGERPRINT="):
            continue
        payload = line.split(b"=", 1)[1]
        if not payload:
            raise RuntimeError(f"fpcalc returned an empty fingerprint for {path}")
        return [int(x) for x in payload.split(b",") if x]
    raise RuntimeError(f"fpcalc returned no FINGERPRINT record for {path}")


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
    parser = argparse.ArgumentParser(description="SAO Chromaprint fingerprint similarity scores.")
    parser.add_argument(
        "--train-axis-manifest",
        default="sao_ip_pilot/outputs/ekfac_attribution/train_axis_manifest.csv",
    )
    parser.add_argument(
        "--query-axis-manifest",
        default="sao_ip_pilot/outputs/ekfac_attribution/query_axis_manifest.csv",
    )
    parser.add_argument(
        "--output-path",
        default="sao_ip_pilot/outputs/baselines/fingerprint/score_fingerprint.pt",
    )
    parser.add_argument("--length", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    train_manifest = pd.read_csv(resolve_repo_path(args.train_axis_manifest))
    query_manifest = pd.read_csv(resolve_repo_path(args.query_axis_manifest))
    train_paths = [resolve_audio_path(p) for p in train_manifest["path"].tolist()]
    query_paths = resolve_query_audio_paths(query_manifest)

    missing_train = [str(p) for p in train_paths if not p.exists()]
    missing_query = [str(p) for p in query_paths if not p.exists()]
    if missing_train:
        raise FileNotFoundError(f"{len(missing_train)} train audio missing, e.g. {missing_train[0]}")
    if missing_query:
        raise FileNotFoundError(f"{len(missing_query)} query audio missing, e.g. {missing_query[0]}")

    print(
        f"[sao-fingerprint] train={len(train_paths)} queries={len(query_paths)} "
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
    np.savez_compressed(
        out.parent / "fingerprints_raw.npz",
        train_lengths=np.array([len(x) for x in train_fps], dtype=np.int32),
        query_lengths=np.array([len(x) for x in query_fps], dtype=np.int32),
    )
    print(
        f"[sao-fingerprint] wrote {out} shape={tuple(tensor.shape)} "
        f"min={float(tensor.min()):.4f} max={float(tensor.max()):.4f} mean={float(tensor.mean()):.4f}"
    )


if __name__ == "__main__":
    main()
