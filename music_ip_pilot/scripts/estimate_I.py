#!/usr/bin/env python3
"""
Phase 6a: Estimate per-seller informativeness I_hat_j
======================================================

For each seller j, using the paired observations (a_hat_{j,t}, a*_{j,t})
over t = 1,...,T, compute the OLS regression slope:

  I_hat_j^{(m)}  =  Cov_t(a_hat^{(m)}_{j,t}, a*_{j,t}) / Var_t(a_hat^{(m)}_{j,t})

This corresponds exactly to the framework's informativeness measure
I_j (Eq. in paper), without the pi_t multiplier (set to 1 in pilot).

We also report Spearman rank correlation as a cross-check against the
ML-standard attribution-quality metric.

Bootstrap CIs are obtained by resampling over t with replacement.

Output: outputs/I_hat.csv with columns:
  seller_id, method, I_hat, I_hat_lo, I_hat_hi,
  spearman, spearman_lo, spearman_hi, mse, T

The `mse = mean((a_hat - a_star)**2)` column is a calibration check per
Jiaqi's review: I_hat is scale-invariant (Cov/Var is a regression slope),
so a method can have high I_hat while a_hat is numerically miscalibrated
vs a_star in absolute terms. MSE surfaces that miscalibration.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from common import outputs_dir, read_json, sellers_json_path


METHOD_COLS = {
    "embedding_similarity": "a_hat_embed",
    # future methods register here
}


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Cov(x, y) / Var(x). Returns NaN if Var(x) is 0."""
    if len(x) < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    if vx == 0:
        return float("nan")
    cxy = np.cov(x, y, ddof=1)[0, 1]
    return float(cxy / vx)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Uses pandas to avoid scipy dep."""
    if len(x) < 2:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def _bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn,
    n_boot: int,
    ci: float,
    seed: int,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI of stat_fn(x, y) over paired resamples.
    Returns (nan, nan) when the point estimate or all resamples are NaN.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2:
        return (float("nan"), float("nan"))
    # If the full-sample estimate is already NaN (e.g., Var(x)=0),
    # every bootstrap replicate will also be NaN — short-circuit.
    if np.isnan(stat_fn(x, y)):
        return (float("nan"), float("nan"))
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = stat_fn(x[idx], y[idx])
    n_valid = int(np.isfinite(vals).sum())
    if n_valid == 0:
        return (float("nan"), float("nan"))
    if n_valid < int(0.9 * n_boot):
        print(f"    [warn] bootstrap: only {n_valid}/{n_boot} replicates finite; "
              f"CI may be unreliable", flush=True)
    alpha = (1.0 - ci) / 2.0
    return (float(np.nanquantile(vals, alpha)),
            float(np.nanquantile(vals, 1.0 - alpha)))


def main(cfg: Dict[str, Any]) -> None:
    out_path = outputs_dir(cfg) / "I_hat.csv"
    if out_path.exists():
        print(f"[I_hat] already exists: {out_path}; skip", flush=True)
        return

    a_star_path = outputs_dir(cfg) / "a_star.csv"
    if not a_star_path.exists():
        raise FileNotFoundError(f"a_star.csv missing: {a_star_path}")
    a_star_df = pd.read_csv(a_star_path)

    sellers = read_json(sellers_json_path(cfg))
    seed = int(cfg["seed"])
    n_boot = int(cfg.get("bootstrap_samples", 1000))
    ci = float(cfg.get("ci_quantile", 0.95))
    methods = list(cfg.get("attribution_methods", ["embedding_similarity"]))

    rows: List[Dict[str, Any]] = []
    for method in methods:
        col = METHOD_COLS.get(method)
        if col is None:
            raise ValueError(f"unknown attribution method: {method}")
        a_hat_path = outputs_dir(cfg) / f"a_hat_{method.split('_')[0]}.csv"
        # our pilot writes a_hat_embed.csv; map method->file explicitly
        if method == "embedding_similarity":
            a_hat_path = outputs_dir(cfg) / "a_hat_embed.csv"
        if not a_hat_path.exists():
            raise FileNotFoundError(f"missing {a_hat_path} for method {method}")
        a_hat_df = pd.read_csv(a_hat_path)

        merged = a_star_df.merge(a_hat_df, on=["seller_id", "prompt_id"])
        for s in sellers:
            sub = merged[merged["seller_id"] == s["seller_id"]]
            x = sub[col].to_numpy(dtype=np.float64)
            y = sub["a_star"].to_numpy(dtype=np.float64)

            I_hat = _ols_slope(x, y)
            sp = _spearman(x, y)
            # MSE(a_hat, a_star) - calibration diagnostic (scale-aware, unlike I_hat)
            mse = float(np.mean((x - y) ** 2)) if len(x) >= 1 else float("nan")

            I_lo, I_hi = _bootstrap_ci(x, y, _ols_slope, n_boot, ci, seed)
            sp_lo, sp_hi = _bootstrap_ci(x, y, _spearman, n_boot, ci, seed + 1)

            rows.append({
                "seller_id": s["seller_id"],
                "method": method,
                "I_hat": I_hat,
                "I_hat_lo": I_lo,
                "I_hat_hi": I_hi,
                "spearman": sp,
                "spearman_lo": sp_lo,
                "spearman_hi": sp_hi,
                "mse": mse,
                "T": len(sub),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False), flush=True)
    print(f"[I_hat] wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    from common import load_config
    main(load_config(args.config))
