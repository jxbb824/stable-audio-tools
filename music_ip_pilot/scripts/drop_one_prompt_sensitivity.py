#!/usr/bin/env python3
"""
Phase 6b: Drop-one-prompt jackknife sensitivity
================================================

For each seller, recompute I_hat omitting one prompt at a time (T jackknife
replicates per seller). Reports:
  - Range of I_hat over jackknife (min, max)
  - Std over jackknife
  - Which dropped prompt has the largest influence (max |delta|)

Addresses Jiaqi's per-seller-estimator-robustness concern: "is I_hat stable,
or is it driven by one outlier prompt?"

Input:
  data/outputs/results/a_star.csv
  data/outputs/results/a_hat_embed.csv

Output:
  data/outputs/results/I_hat_sensitivity.csv
    columns: seller_id, method, I_hat_full,
             I_hat_min_jk, I_hat_max_jk, I_hat_std_jk,
             most_influential_prompt_id, delta_at_most_influential
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
# Can run from repo-root or from the data folder directly; pick the
# outputs/results folder as the authoritative input.
DEFAULT_RES = ROOT / "data" / "outputs" / "results"


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    if vx == 0:
        return float("nan")
    return float(np.cov(x, y, ddof=1)[0, 1] / vx)


def jackknife(
    astar: pd.DataFrame,
    ahat: pd.DataFrame,
    a_hat_col: str = "a_hat_embed",
) -> pd.DataFrame:
    """Per-seller, drop-one-prompt I_hat sensitivity."""
    merged = astar.merge(ahat, on=["seller_id", "prompt_id"])
    rows: List[Dict[str, Any]] = []

    for sid, sub in merged.groupby("seller_id", sort=True):
        sub = sub.sort_values("prompt_id").reset_index(drop=True)
        x_full = sub[a_hat_col].to_numpy(float)
        y_full = sub["a_star"].to_numpy(float)
        pids = sub["prompt_id"].tolist()

        I_full = ols_slope(x_full, y_full)

        jk_I: List[float] = []
        deltas: List[float] = []
        for i in range(len(sub)):
            x_jk = np.delete(x_full, i)
            y_jk = np.delete(y_full, i)
            I_i = ols_slope(x_jk, y_jk)
            jk_I.append(I_i)
            deltas.append(I_i - I_full if not np.isnan(I_full) and not np.isnan(I_i) else float("nan"))

        jk_arr = np.array(jk_I, dtype=float)
        d_arr = np.array(deltas, dtype=float)
        abs_d = np.abs(d_arr)
        if np.all(np.isnan(abs_d)):
            most_idx = 0
        else:
            most_idx = int(np.nanargmax(abs_d))

        rows.append({
            "seller_id": sid,
            "method": "embedding_similarity",
            "I_hat_full": I_full,
            "I_hat_min_jk": float(np.nanmin(jk_arr)) if not np.all(np.isnan(jk_arr)) else float("nan"),
            "I_hat_max_jk": float(np.nanmax(jk_arr)) if not np.all(np.isnan(jk_arr)) else float("nan"),
            "I_hat_std_jk": float(np.nanstd(jk_arr, ddof=1)) if np.sum(np.isfinite(jk_arr)) >= 2 else float("nan"),
            "most_influential_prompt_id": pids[most_idx],
            "delta_at_most_influential": float(d_arr[most_idx]),
            "T": len(sub),
        })

    return pd.DataFrame(rows)


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RES)
    args = parser.parse_args()

    astar_path = args.results_dir / "a_star.csv"
    ahat_path = args.results_dir / "a_hat_embed.csv"
    if not astar_path.exists() or not ahat_path.exists():
        raise FileNotFoundError(
            f"need {astar_path} and {ahat_path}; run Phase 5 first"
        )

    astar = pd.read_csv(astar_path)
    ahat = pd.read_csv(ahat_path)

    df = jackknife(astar, ahat)
    out = args.results_dir / "I_hat_sensitivity.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\n[sens] wrote -> {out}")


if __name__ == "__main__":
    main()
