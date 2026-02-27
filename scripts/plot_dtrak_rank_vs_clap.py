#!/usr/bin/env python3

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

# Ensure local repo package is importable when executed as a file path, e.g.:
# `python3 /path/to/repo/scripts/plot_dtrak_rank_vs_clap.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.training.songscriber_clap import (
    CLAP_SAMPLE_RATE,
    DEFAULT_CLAP_MODEL_NAME,
    ClapEmbedder,
    load_audio_mono_resampled,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot relation between D-TRAK influence rank and CLAP cosine similarity. "
            "For each query sample, training samples are sorted by influence score, "
            "then CLAP cosine(query, ranked-train) is computed and averaged across queries."
        )
    )
    parser.add_argument(
        "--attribution-dir",
        type=Path,
        default=Path("/home/xiruij/stable-audio-tools/outputs/dtrak_attribution_20260225_124956"),
        help="Directory containing D-TRAK score/features outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <attribution-dir>/rank_vs_clap).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="influence_rank_vs_clap",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=None,
        help="Max rank to plot (default: all train samples).",
    )
    parser.add_argument(
        "--clap-model-name",
        type=str,
        default=DEFAULT_CLAP_MODEL_NAME,
    )
    parser.add_argument(
        "--clap-device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--clap-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--audio-load-batch-size",
        type=int,
        default=4,
        help="How many audio files to decode into RAM per outer batch (smaller = lower memory).",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=None,
        help="Path to CLAP embedding cache .npz (default: <output-dir>/clap_embeddings_train_query.npz).",
    )
    parser.add_argument(
        "--force-recompute-clap",
        action="store_true",
        help="Ignore existing CLAP cache and recompute embeddings.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_ids(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return x / norms


def all_files_exist(paths: Sequence[str]) -> tuple[bool, list[str]]:
    missing = [p for p in paths if not Path(p).exists()]
    return len(missing) == 0, missing


def same_path_list(a: np.ndarray, b: Sequence[str]) -> bool:
    if a.shape[0] != len(b):
        return False
    return bool(np.all(a.astype(str) == np.array(b, dtype=str)))


def embed_audio_files_streaming(
    clap: ClapEmbedder,
    audio_paths: Sequence[str],
    clap_batch_size: int,
    audio_load_batch_size: int,
    desc: str,
) -> np.ndarray:
    n = len(audio_paths)
    if n == 0:
        return np.empty((0, 512), dtype=np.float32)

    load_bs = max(1, int(audio_load_batch_size))
    iterator = range(0, n, load_bs)
    if tqdm is not None:
        total_batches = (n + load_bs - 1) // load_bs
        iterator = tqdm(iterator, total=total_batches, desc=desc, unit="batch")

    emb_chunks: list[np.ndarray] = []
    for batch_idx, start in enumerate(iterator):
        end = min(start + load_bs, n)
        paths_chunk = audio_paths[start:end]
        audios = [load_audio_mono_resampled(Path(p), target_sr=CLAP_SAMPLE_RATE) for p in paths_chunk]
        emb = clap.embed_audio_arrays(
            audios,
            batch_size=max(1, min(clap_batch_size, len(audios))),
            show_progress=False,
        )
        emb_chunks.append(emb.astype(np.float32, copy=False))
        del audios
        del emb
        gc.collect()
        if tqdm is None and ((batch_idx + 1) % 10 == 0 or end == n):
            print(f"[progress] {desc}: {end}/{n}", flush=True)

    return np.concatenate(emb_chunks, axis=0)


def load_or_compute_clap_embeddings(
    train_paths: list[str],
    query_paths: list[str],
    cache_path: Path,
    clap_model_name: str,
    clap_device: str,
    clap_batch_size: int,
    audio_load_batch_size: int,
    force_recompute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists() and not force_recompute:
        try:
            data = np.load(cache_path, allow_pickle=False)
            cache_ok = (
                str(data["clap_model_name"][0]) == clap_model_name
                and int(data["clap_sample_rate"][0]) == CLAP_SAMPLE_RATE
                and same_path_list(data["train_paths"], train_paths)
                and same_path_list(data["query_paths"], query_paths)
            )
            if cache_ok:
                print(f"[cache] using cached CLAP embeddings: {cache_path}")
                return data["train_embeddings"].astype(np.float32), data["query_embeddings"].astype(np.float32)
            print(f"[cache] mismatch detected, recomputing CLAP embeddings: {cache_path}", flush=True)
        except Exception as exc:
            print(f"[cache] failed to read cache ({cache_path}): {exc}; recomputing.", flush=True)

    device = resolve_device(clap_device)
    print(f"[clap] loading model={clap_model_name} on device={device}", flush=True)
    clap = ClapEmbedder(model_name=clap_model_name, device=device)

    print(
        f"[clap] embedding train audio ({len(train_paths)}) "
        f"with audio_load_batch_size={audio_load_batch_size}, clap_batch_size={clap_batch_size}",
        flush=True,
    )
    train_embeddings = embed_audio_files_streaming(
        clap=clap,
        audio_paths=train_paths,
        clap_batch_size=clap_batch_size,
        audio_load_batch_size=audio_load_batch_size,
        desc="CLAP train",
    ).astype(np.float32)

    print(
        f"[clap] embedding query audio ({len(query_paths)}) "
        f"with audio_load_batch_size={audio_load_batch_size}, clap_batch_size={clap_batch_size}",
        flush=True,
    )
    query_embeddings = embed_audio_files_streaming(
        clap=clap,
        audio_paths=query_paths,
        clap_batch_size=clap_batch_size,
        audio_load_batch_size=audio_load_batch_size,
        desc="CLAP query",
    ).astype(np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        created_at=np.array([datetime.now(timezone.utc).isoformat()]),
        clap_model_name=np.array([clap_model_name], dtype=np.str_),
        clap_sample_rate=np.array([CLAP_SAMPLE_RATE], dtype=np.int32),
        train_paths=np.array(train_paths, dtype=np.str_),
        query_paths=np.array(query_paths, dtype=np.str_),
        train_embeddings=train_embeddings,
        query_embeddings=query_embeddings,
    )
    print(f"[cache] saved CLAP embeddings: {cache_path}")
    return train_embeddings, query_embeddings


def save_plot(
    ranks: np.ndarray,
    mean_cosine_by_rank: np.ndarray,
    std_cosine_by_rank: np.ndarray,
    n_query: int,
    n_train: int,
    plot_path: Path,
    dpi: int,
) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to generate the plot. "
            "Install it with `pip install matplotlib`."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(ranks, mean_cosine_by_rank, linewidth=2.0, label="Mean CLAP cosine")
    plt.fill_between(
        ranks,
        mean_cosine_by_rank - std_cosine_by_rank,
        mean_cosine_by_rank + std_cosine_by_rank,
        alpha=0.2,
        label="±1 std across queries",
    )
    plt.xlabel("Influence rank (1 = highest influence)")
    plt.ylabel("CLAP cosine (train sample vs its query sample)")
    plt.title(f"D-TRAK Influence Rank vs CLAP Cosine (queries={n_query}, train={n_train})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=dpi)
    plt.close()


def main() -> None:
    args = parse_args()
    print("[1/6] preparing paths and metadata", flush=True)

    attribution_dir = args.attribution_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir is not None else attribution_dir / "rank_vs_clap"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_path.resolve() if args.cache_path is not None else output_dir / "clap_embeddings_train_query.npz"

    score_meta_path = attribution_dir / "scores_query_x_train.memmap.meta.json"
    query_meta_path = attribution_dir / "query_features.memmap.meta.json"
    train_meta_path = attribution_dir / "train_features.memmap.meta.json"
    score_path = attribution_dir / "scores_query_x_train.memmap"

    for required in [score_meta_path, query_meta_path, train_meta_path, score_path]:
        if not required.exists():
            raise FileNotFoundError(f"Missing required file: {required}")

    score_meta = read_json(score_meta_path)
    query_meta = read_json(query_meta_path)
    train_meta = read_json(train_meta_path)

    n_query, n_train = score_meta["shape"]
    print(f"[2/6] loading score matrix memmap (query={n_query}, train={n_train})", flush=True)
    scores = np.memmap(score_path, dtype=np.float32, mode="r", shape=(n_query, n_train))

    query_ids_path = Path(query_meta["ids_path"])
    train_ids_path = Path(train_meta["ids_path"])
    query_paths = read_ids(query_ids_path)
    train_paths = read_ids(train_ids_path)

    if len(query_paths) != n_query:
        raise ValueError(f"Query count mismatch: meta={n_query}, ids={len(query_paths)}")
    if len(train_paths) != n_train:
        raise ValueError(f"Train count mismatch: meta={n_train}, ids={len(train_paths)}")

    ok_query, missing_query = all_files_exist(query_paths)
    ok_train, missing_train = all_files_exist(train_paths)
    if not ok_query:
        raise FileNotFoundError(f"Missing query audio files: {missing_query[:10]}")
    if not ok_train:
        raise FileNotFoundError(f"Missing train audio files: {missing_train[:10]}")

    max_rank = n_train if args.max_rank is None else min(int(args.max_rank), n_train)
    if max_rank <= 0:
        raise ValueError("--max-rank must be positive.")

    print("[3/6] loading or computing CLAP embeddings", flush=True)
    train_embeddings, query_embeddings = load_or_compute_clap_embeddings(
        train_paths=train_paths,
        query_paths=query_paths,
        cache_path=cache_path,
        clap_model_name=args.clap_model_name,
        clap_device=args.clap_device,
        clap_batch_size=args.clap_batch_size,
        audio_load_batch_size=args.audio_load_batch_size,
        force_recompute=args.force_recompute_clap,
    )

    print("[4/6] normalizing CLAP embeddings", flush=True)
    train_embeddings = normalize_rows(train_embeddings.astype(np.float32))
    query_embeddings = normalize_rows(query_embeddings.astype(np.float32))

    print("[5/6] computing rank-aligned cosine statistics", flush=True)
    print("[compute] cosine matrix query x train", flush=True)
    cosine_query_train = query_embeddings @ train_embeddings.T

    print("[compute] sorting train indices by influence score for each query", flush=True)
    ranked_train_indices = np.argsort(scores, axis=1)[:, ::-1]
    ranked_train_indices = ranked_train_indices[:, :max_rank]

    ranked_cosines = np.take_along_axis(cosine_query_train, ranked_train_indices, axis=1)
    mean_cosine_by_rank = ranked_cosines.mean(axis=0)
    std_cosine_by_rank = ranked_cosines.std(axis=0)

    ranks = np.arange(1, max_rank + 1, dtype=np.int32)

    csv_path = output_dir / f"{args.output_prefix}.csv"
    npy_path = output_dir / f"{args.output_prefix}_per_query.npy"
    plot_path = output_dir / f"{args.output_prefix}.png"
    meta_out_path = output_dir / f"{args.output_prefix}.meta.json"

    csv_data = np.column_stack([ranks, mean_cosine_by_rank, std_cosine_by_rank])
    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",
        header="rank,mean_clap_cosine,std_clap_cosine",
        comments="",
        fmt=["%d", "%.8f", "%.8f"],
    )
    np.save(npy_path, ranked_cosines.astype(np.float32))

    print("[6/6] saving plot and outputs", flush=True)
    save_plot(
        ranks=ranks,
        mean_cosine_by_rank=mean_cosine_by_rank,
        std_cosine_by_rank=std_cosine_by_rank,
        n_query=n_query,
        n_train=n_train,
        plot_path=plot_path,
        dpi=args.dpi,
    )

    out_meta = {
        "attribution_dir": str(attribution_dir),
        "score_path": str(score_path),
        "score_shape": [int(n_query), int(n_train)],
        "train_ids_path": str(train_ids_path),
        "query_ids_path": str(query_ids_path),
        "clap_model_name": args.clap_model_name,
        "clap_sample_rate": int(CLAP_SAMPLE_RATE),
        "clap_device": resolve_device(args.clap_device),
        "clap_batch_size": int(args.clap_batch_size),
        "max_rank": int(max_rank),
        "cache_path": str(cache_path),
        "csv_path": str(csv_path),
        "npy_path": str(npy_path),
        "plot_path": str(plot_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with meta_out_path.open("w", encoding="utf-8") as f:
        json.dump(out_meta, f, indent=2, ensure_ascii=True)

    print(f"[done] csv: {csv_path}", flush=True)
    print(f"[done] per-query cosine matrix: {npy_path}", flush=True)
    print(f"[done] plot: {plot_path}", flush=True)
    print(f"[done] meta: {meta_out_path}", flush=True)


if __name__ == "__main__":
    main()
