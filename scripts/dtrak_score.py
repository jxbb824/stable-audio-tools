#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute D-TRAK / TRAK scores from projected gradients."
    )
    parser.add_argument("--train-feature-paths", type=str, nargs="+", required=True)
    parser.add_argument("--train-meta-paths", type=str, nargs="+", required=True)
    parser.add_argument("--query-feature-path", type=str, required=True)
    parser.add_argument("--query-meta-path", type=str, required=True)
    parser.add_argument("--output-score-path", type=str, required=True)
    parser.add_argument("--output-meta-path", type=str, default=None)
    parser.add_argument("--lambda-reg", type=float, default=1e-3)
    parser.add_argument("--used-dim", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--negate", action="store_true")
    return parser.parse_args()


def load_meta(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_feature_from_meta(feature_path: str, meta: dict) -> np.ndarray:
    rows = int(meta["num_rows_allocated"])
    cols = int(meta["proj_dim"])
    valid_rows = int(meta.get("valid_rows", rows))
    arr = np.memmap(feature_path, dtype=np.float32, mode="r", shape=(rows, cols))
    return np.asarray(arr[:valid_rows])


def maybe_load_ids(meta: dict) -> Optional[List[str]]:
    ids_path = meta.get("ids_path", None)
    if ids_path is None:
        return None
    path = Path(ids_path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def concat_train_features(paths: Sequence[str], metas: Sequence[dict]) -> Tuple[np.ndarray, Optional[List[str]]]:
    blocks = []
    all_ids: List[str] = []
    has_all_ids = True
    for path, meta in zip(paths, metas):
        blocks.append(load_feature_from_meta(path, meta))
        ids = maybe_load_ids(meta)
        if ids is None:
            has_all_ids = False
        else:
            all_ids.extend(ids)
    train_features = np.vstack(blocks)
    return train_features, all_ids if has_all_ids else None


def main() -> None:
    args = parse_args()
    if len(args.train_feature_paths) != len(args.train_meta_paths):
        raise ValueError("--train-feature-paths and --train-meta-paths must have the same length.")

    train_metas = [load_meta(path) for path in args.train_meta_paths]
    query_meta = load_meta(args.query_meta_path)

    train_features, train_ids = concat_train_features(args.train_feature_paths, train_metas)
    query_features = load_feature_from_meta(args.query_feature_path, query_meta)
    query_ids = maybe_load_ids(query_meta)

    used_dim = args.used_dim if args.used_dim is not None else train_features.shape[1]
    if used_dim <= 0:
        raise ValueError("--used-dim must be positive.")
    used_dim = min(used_dim, train_features.shape[1], query_features.shape[1])

    device = torch.device(args.device)
    x_train = torch.from_numpy(train_features[:, :used_dim]).to(device)
    x_query = torch.from_numpy(query_features[:, :used_dim]).to(device)

    kernel = x_train.T @ x_train
    kernel = kernel + args.lambda_reg * torch.eye(kernel.shape[0], device=device, dtype=kernel.dtype)
    kernel_inv = torch.linalg.inv(kernel)
    x_train_kernel = x_train @ kernel_inv

    output_score_path = Path(args.output_score_path)
    output_score_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.memmap(
        str(output_score_path),
        dtype=np.float32,
        mode="w+",
        shape=(x_query.shape[0], x_train.shape[0]),
    )

    batch_size = max(1, int(args.query_batch_size))
    for start in range(0, x_query.shape[0], batch_size):
        end = min(start + batch_size, x_query.shape[0])
        block = x_query[start:end] @ x_train_kernel.T
        if args.negate:
            block = -block
        scores[start:end] = block.detach().cpu().numpy().astype(np.float32)
        print(f"scored={end}/{x_query.shape[0]}")
    scores.flush()

    output_meta_path = (
        Path(args.output_meta_path)
        if args.output_meta_path is not None
        else output_score_path.with_suffix(output_score_path.suffix + ".meta.json")
    )
    output_meta_path.parent.mkdir(parents=True, exist_ok=True)
    out_meta = {
        "output_score_path": str(output_score_path),
        "shape": [int(x_query.shape[0]), int(x_train.shape[0])],
        "lambda_reg": float(args.lambda_reg),
        "used_dim": int(used_dim),
        "negate": bool(args.negate),
        "train_num_examples": int(x_train.shape[0]),
        "query_num_examples": int(x_query.shape[0]),
        "train_ids_path": train_metas[0].get("ids_path") if train_ids is not None else None,
        "query_ids_path": query_meta.get("ids_path") if query_ids is not None else None,
    }
    with open(output_meta_path, "w", encoding="utf-8") as handle:
        json.dump(out_meta, handle, indent=2, ensure_ascii=True)

    print(f"output_score_path={output_score_path}")
    print(f"output_meta_path={output_meta_path}")


if __name__ == "__main__":
    main()
