#!/usr/bin/env python3
"""Verify that the 50-seller scale-up attribution pipeline is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("dtrak", "ekfac")
FIGURE_STEMS = (
    "I_hat_scatter",
    "ranking_comparison",
    "snr_spearman_summary",
    "sample_count_diagnostics",
    "marginal_welfare_vs_snr",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="sao_ip_100/outputs",
        help="Scale-up output directory, relative to the repository root.",
    )
    parser.add_argument(
        "--json-output",
        default="sao_ip_100/outputs/verification_summary.json",
        help="Where to write the machine-readable verification summary.",
    )
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty required output: {path}")
    return path


def require_finite(frame: pd.DataFrame, columns: tuple[str, ...], label: str) -> None:
    values = frame.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values in {columns}")


def verify_method(output_root: Path, a_star: pd.DataFrame, method: str) -> dict[str, object]:
    attribution_root = output_root / f"{method}_attribution"
    score_path = require_file(attribution_root / "scores_train_x_query.pt")
    scores = torch.load(score_path, map_location="cpu").float()
    if tuple(scores.shape) != (1120, 200):
        raise ValueError(f"{method}: expected score shape (1120, 200), got {tuple(scores.shape)}")
    if not torch.isfinite(scores).all():
        raise ValueError(f"{method}: attribution scores contain non-finite values")

    train_axis = pd.read_csv(require_file(attribution_root / "train_axis_manifest.csv"))
    query_axis = pd.read_csv(require_file(attribution_root / "query_axis_manifest.csv"))
    if len(train_axis) != 1120 or len(query_axis) != 200:
        raise ValueError(
            f"{method}: axis sizes are train={len(train_axis)}, query={len(query_axis)}; "
            "expected 1120 and 200"
        )
    require_file(attribution_root / "attribution_metadata.json")

    a_hat_column = f"a_hat_{method}"
    a_hat = pd.read_csv(require_file(output_root / f"a_hat_{method}.csv"))
    required_a_hat = {"seller_id", "prompt_id", "query_index", a_hat_column, "num_train_examples"}
    if not required_a_hat.issubset(a_hat.columns):
        raise ValueError(f"{method}: a_hat is missing columns {sorted(required_a_hat - set(a_hat.columns))}")
    if len(a_hat) != 10_000 or a_hat["seller_id"].nunique() != 50 or a_hat["query_index"].nunique() != 200:
        raise ValueError(f"{method}: a_hat does not cover exactly 50 sellers x 200 queries")
    if a_hat.duplicated(["seller_id", "query_index"]).any():
        raise ValueError(f"{method}: a_hat has duplicate seller/query pairs")
    if set(map(tuple, a_hat[["seller_id", "query_index"]].to_numpy())) != set(
        map(tuple, a_star[["seller_id", "query_index"]].to_numpy())
    ):
        raise ValueError(f"{method}: a_hat seller/query keys do not match a_star")
    require_finite(a_hat, (a_hat_column, "num_train_examples"), f"{method} a_hat")
    seller_counts = a_hat.groupby("seller_id")["num_train_examples"].nunique()
    if not (seller_counts == 1).all():
        raise ValueError(f"{method}: num_train_examples changes across queries for a seller")
    if int(a_hat.groupby("seller_id")["num_train_examples"].first().sum()) != 1120:
        raise ValueError(f"{method}: seller training counts do not sum to 1120")

    i_hat = pd.read_csv(require_file(output_root / f"I_hat_{method}.csv"))
    if len(i_hat) != 50 or i_hat["seller_id"].nunique() != 50 or set(i_hat["seller_id"]) != set(a_hat["seller_id"]):
        raise ValueError(f"{method}: I_hat does not cover the same 50 sellers as a_hat")
    if not (i_hat["T"] == 200).all():
        raise ValueError(f"{method}: I_hat contains a seller with T != 200")
    require_finite(
        i_hat,
        ("I_hat", "I_hat_lo", "I_hat_hi", "spearman", "K_hat", "SNR_hat_calibrated", "var_astar"),
        f"{method} I_hat",
    )

    welfare = pd.read_csv(require_file(output_root / f"marginal_welfare_numbers_{method}.csv"))
    if len(welfare) != 50 or set(welfare["seller_id"]) != set(i_hat["seller_id"]):
        raise ValueError(f"{method}: welfare table does not cover the same 50 sellers as I_hat")
    require_finite(welfare, ("I_hat", "SNR_hat_calibrated", "var_astar", "marg_welfare_at_alpha_ref"), f"{method} welfare")

    figure_root = output_root / f"figures_{method}"
    figures = [require_file(figure_root / f"{stem}.{suffix}") for stem in FIGURE_STEMS for suffix in ("png", "pdf")]
    return {
        "score_shape": list(scores.shape),
        "a_hat_rows": len(a_hat),
        "i_hat_rows": len(i_hat),
        "welfare_rows": len(welfare),
        "figures": [str(path.relative_to(REPO_ROOT)) for path in figures],
    }


def main() -> None:
    args = parse_args()
    output_root = resolve(args.output_root)
    a_star = pd.read_csv(require_file(output_root / "a_star.csv"))
    if len(a_star) != 10_000 or a_star["seller_id"].nunique() != 50 or a_star["query_index"].nunique() != 200:
        raise ValueError("a_star does not cover exactly 50 sellers x 200 queries")
    require_finite(a_star, ("loss_full", "loss_loo", "a_star"), "a_star")

    summary = {
        "status": "complete",
        "expected_sellers": 50,
        "expected_queries": 200,
        "expected_train_examples": 1120,
        "methods": {method: verify_method(output_root, a_star, method) for method in METHODS},
    }
    json_output = resolve(args.json_output)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
