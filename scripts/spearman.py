#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from scipy.stats import combine_pvalues, spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Spearman correlation between D-TRAK scores and subset-model ground truth."
    )
    parser.add_argument("--models-root", type=str, default="outputs/groundtruth_models")
    parser.add_argument(
        "--score-path",
        type=str,
        default="outputs/groundtruth_models/dtrak_attribution/scores_train_x_query.pt",
    )
    parser.add_argument("--groundtruth-path", type=str, default="outputs/groundtruth_models/gt_gen.pt")
    parser.add_argument("--model-ids-path", type=str, default=None)
    parser.add_argument("--query-ids-path", type=str, default=None)
    parser.add_argument("--subset-index-filename", type=str, default="train_index.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/groundtruth_models/spearman")
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def load_tensor(path: Path) -> torch.Tensor:
    if not path.exists():
        raise FileNotFoundError(f"Tensor file not found: {path}")
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor in {path}, found {type(tensor)!r}")
    return tensor.detach().cpu()


def load_text_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def read_int_csv(path: Path) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    values: List[int] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            for item in row:
                item = item.strip()
                if not item:
                    continue
                values.append(int(item))
    if not values:
        raise ValueError(f"No indices found in {path}")
    return values


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_model_ids(models_root: Path, groundtruth_path: Path, explicit_path: str | None) -> List[str]:
    if explicit_path:
        return load_text_lines(Path(explicit_path).resolve())

    default_path = Path(f"{groundtruth_path}.model_ids.txt")
    if default_path.exists():
        return load_text_lines(default_path)

    model_ids = [child.name for child in models_root.iterdir() if child.is_dir() and child.name.isdigit()]
    model_ids.sort(key=int)
    if not model_ids:
        raise FileNotFoundError(f"No numeric model directories found under {models_root}")
    return model_ids


def infer_query_ids(groundtruth_path: Path, explicit_path: str | None, query_count: int) -> List[str]:
    if explicit_path:
        query_ids = load_text_lines(Path(explicit_path).resolve())
    else:
        default_path = Path(f"{groundtruth_path}.query_ids.txt")
        if default_path.exists():
            query_ids = load_text_lines(default_path)
        else:
            query_ids = [f"query_{index:04d}" for index in range(query_count)]

    if len(query_ids) != query_count:
        raise ValueError(
            f"Expected {query_count} query ids, but found {len(query_ids)} in the provided query id file."
        )
    return query_ids


def load_subset_indices(models_root: Path, model_ids: Sequence[str], filename: str, train_count: int) -> List[List[int]]:
    subset_indices: List[List[int]] = []
    for model_id in model_ids:
        indices = read_int_csv(models_root / model_id / filename)
        for value in indices:
            if value < 0 or value >= train_count:
                raise IndexError(
                    f"Subset index {value} for model {model_id} is out of bounds for train axis size {train_count}."
                )
        subset_indices.append(indices)
    return subset_indices


def aggregate_subset_scores(score_matrix: torch.Tensor, subset_indices: Sequence[Sequence[int]]) -> torch.Tensor:
    approx_rows = []
    for indices in subset_indices:
        approx_rows.append(score_matrix[indices, :].sum(dim=0))
    return torch.stack(approx_rows, dim=0)


def compute_query_stats(
    approx_matrix: torch.Tensor,
    groundtruth_matrix: torch.Tensor,
    query_ids: Sequence[str],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if approx_matrix.shape != groundtruth_matrix.shape:
        raise ValueError(
            f"Approx matrix shape {tuple(approx_matrix.shape)} does not match ground truth shape {tuple(groundtruth_matrix.shape)}."
        )

    valid_stats: List[float] = []
    valid_pvalues: List[float] = []
    rows: List[Dict[str, Any]] = []

    for query_index in range(groundtruth_matrix.shape[1]):
        approx_column = approx_matrix[:, query_index].numpy()
        groundtruth_column = groundtruth_matrix[:, query_index].numpy()
        statistic, pvalue = spearmanr(approx_column, groundtruth_column)

        valid = not (np.isnan(statistic) or np.isnan(pvalue))
        row = {
            "query_index": query_index,
            "query_id": query_ids[query_index],
            "spearman": "" if np.isnan(statistic) else float(statistic),
            "pvalue": "" if np.isnan(pvalue) else float(pvalue),
            "valid": int(valid),
        }
        rows.append(row)

        if valid:
            valid_stats.append(float(statistic))
            valid_pvalues.append(max(float(pvalue), np.finfo(float).tiny))

    average_statistic = float(np.mean(valid_stats)) if valid_stats else float("nan")
    fisher_pvalue = (
        float(combine_pvalues(valid_pvalues, method="fisher")[1])
        if valid_pvalues
        else float("nan")
    )
    if not np.isnan(fisher_pvalue):
        fisher_pvalue = max(fisher_pvalue, np.finfo(float).tiny)

    summary = {
        "average_spearman": average_statistic,
        "fisher_pvalue": fisher_pvalue,
        "valid_query_count": len(valid_stats),
        "total_query_count": int(groundtruth_matrix.shape[1]),
    }
    return summary, rows


def merge_query_rows(
    query_ids: Sequence[str],
    dtrak_rows: Sequence[Dict[str, Any]],
    random_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged_rows: List[Dict[str, Any]] = []
    for query_index, query_id in enumerate(query_ids):
        dtrak_row = dtrak_rows[query_index]
        random_row = random_rows[query_index]
        merged_rows.append(
            {
                "query_index": query_index,
                "query_id": query_id,
                "dtrak_spearman": dtrak_row["spearman"],
                "dtrak_pvalue": dtrak_row["pvalue"],
                "dtrak_valid": dtrak_row["valid"],
                "random_spearman": random_row["spearman"],
                "random_pvalue": random_row["pvalue"],
                "random_valid": random_row["valid"],
            }
        )
    return merged_rows


def main() -> None:
    args = parse_args()

    models_root = Path(args.models_root).resolve()
    score_path = Path(args.score_path).resolve()
    groundtruth_path = Path(args.groundtruth_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    score_matrix = load_tensor(score_path).float()
    groundtruth_matrix = load_tensor(groundtruth_path).float()

    if score_matrix.ndim != 2 or groundtruth_matrix.ndim != 2:
        raise ValueError("Both score and ground truth tensors must be 2D matrices.")
    if score_matrix.shape[1] != groundtruth_matrix.shape[1]:
        raise ValueError(
            f"Score query dimension {score_matrix.shape[1]} does not match ground truth query dimension {groundtruth_matrix.shape[1]}."
        )

    model_ids = infer_model_ids(models_root, groundtruth_path, args.model_ids_path)
    if len(model_ids) != groundtruth_matrix.shape[0]:
        raise ValueError(
            f"Model id count {len(model_ids)} does not match ground truth model dimension {groundtruth_matrix.shape[0]}."
        )
    query_ids = infer_query_ids(groundtruth_path, args.query_ids_path, groundtruth_matrix.shape[1])

    subset_indices = load_subset_indices(
        models_root=models_root,
        model_ids=model_ids,
        filename=args.subset_index_filename,
        train_count=int(score_matrix.shape[0]),
    )

    dtrak_subset_scores = aggregate_subset_scores(score_matrix, subset_indices)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.random_seed)
    random_score_matrix = torch.rand(
        size=tuple(score_matrix.shape),
        generator=generator,
        dtype=score_matrix.dtype,
    )
    random_subset_scores = aggregate_subset_scores(random_score_matrix, subset_indices)

    dtrak_summary, dtrak_rows = compute_query_stats(dtrak_subset_scores, groundtruth_matrix, query_ids)
    random_summary, random_rows = compute_query_stats(random_subset_scores, groundtruth_matrix, query_ids)

    torch.save(dtrak_subset_scores, output_dir / "dtrak_subset_query_scores.pt")
    torch.save(random_score_matrix, output_dir / "random_scores_train_x_query.pt")
    torch.save(random_subset_scores, output_dir / "random_subset_query_scores.pt")

    merged_rows = merge_query_rows(query_ids, dtrak_rows, random_rows)
    write_csv(
        output_dir / "spearman_per_query.csv",
        fieldnames=[
            "query_index",
            "query_id",
            "dtrak_spearman",
            "dtrak_pvalue",
            "dtrak_valid",
            "random_spearman",
            "random_pvalue",
            "random_valid",
        ],
        rows=merged_rows,
    )

    summary = {
        "score_path": str(score_path),
        "groundtruth_path": str(groundtruth_path),
        "models_root": str(models_root),
        "subset_index_filename": args.subset_index_filename,
        "random_seed": args.random_seed,
        "model_count": len(model_ids),
        "train_count": int(score_matrix.shape[0]),
        "query_count": int(score_matrix.shape[1]),
        "model_ids": list(model_ids),
        "dtrak": dtrak_summary,
        "random": random_summary,
    }
    save_json(output_dir / "spearman_summary.json", summary)

    print(f"D-TRAK average spearman: {dtrak_summary['average_spearman']:.6f}")
    print(f"D-TRAK Fisher p-value: {dtrak_summary['fisher_pvalue']:.6e}")
    print(f"Random average spearman: {random_summary['average_spearman']:.6f}")
    print(f"Random Fisher p-value: {random_summary['fisher_pvalue']:.6e}")


if __name__ == "__main__":
    main()
