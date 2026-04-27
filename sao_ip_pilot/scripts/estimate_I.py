#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Callable

import numpy as np
import pandas as pd

from common import read_csv_rows, resolve_repo_path


METHODS = {
    "ekfac_influence": {
        "path": "sao_ip_pilot/outputs/a_hat_ekfac.csv",
        "column": "a_hat_ekfac",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate I_j, Spearman, multiplicative K, and calibrated SNR.")
    parser.add_argument("--a-star-path", default="sao_ip_pilot/outputs/a_star.csv")
    parser.add_argument("--seller-manifest", default="sao_ip_pilot/outputs/selection/seller_manifest.csv")
    parser.add_argument("--method", action="append", choices=sorted(METHODS), default=None)
    parser.add_argument("--output-path", default="sao_ip_pilot/outputs/I_hat.csv")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    var_x = np.var(x, ddof=1)
    if var_x <= 0:
        return float("nan")
    return float(np.cov(x, y, ddof=1)[0, 1] / var_x)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def k_hat(x: np.ndarray, y: np.ndarray) -> float:
    var_y = np.var(y, ddof=1)
    if len(x) < 2 or var_y <= 0:
        return float("nan")
    return float(np.cov(x, y, ddof=1)[0, 1] / var_y)


def snr_calibrated(x: np.ndarray, y: np.ndarray) -> float:
    k = k_hat(x, y)
    if not np.isfinite(k) or abs(k) < 1e-12:
        return float("nan")
    x_cal = x / k
    signal = np.var(y, ddof=1)
    noise = np.var(x_cal - y, ddof=1)
    if noise <= 0:
        return float("inf")
    return float(signal / noise)


def snr_shortcut_from_I(I: float) -> float:
    if not np.isfinite(I):
        return float("nan")
    if I >= 1.0:
        return float("inf")
    if I <= 0.0:
        return 0.0
    return float(I / (1.0 - I))


def bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    stat_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int,
    ci: float,
    seed: int,
) -> tuple[float, float]:
    if len(x) < 2:
        return float("nan"), float("nan")
    point = stat_fn(x, y)
    if not np.isfinite(point):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values = np.empty(n_boot, dtype=np.float64)
    for idx in range(n_boot):
        sample_idx = rng.integers(0, len(x), size=len(x))
        values[idx] = stat_fn(x[sample_idx], y[sample_idx])
    if not np.isfinite(values).any():
        return float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    return float(np.nanquantile(values, alpha)), float(np.nanquantile(values, 1.0 - alpha))


def main() -> None:
    args = parse_args()
    methods = args.method or ["ekfac_influence"]
    a_star = pd.read_csv(resolve_repo_path(args.a_star_path))
    a_star["prompt_id"] = a_star["prompt_id"].astype(str)
    sellers = read_csv_rows(args.seller_manifest)

    rows = []
    for method in methods:
        method_cfg = METHODS[method]
        a_hat = pd.read_csv(resolve_repo_path(method_cfg["path"]))
        a_hat["prompt_id"] = a_hat["prompt_id"].astype(str)
        col = method_cfg["column"]
        merged = a_star.merge(a_hat, on=["seller_id", "prompt_id"], how="inner")

        for seller in sellers:
            seller_id = seller["seller_id"]
            sub = merged[merged["seller_id"] == seller_id].sort_values("query_index_x")
            x = sub[col].to_numpy(dtype=np.float64)
            y = sub["a_star"].to_numpy(dtype=np.float64)

            I = ols_slope(x, y)
            sp = spearman(x, y)
            K = k_hat(x, y)
            snr_cal = snr_calibrated(x, y)
            mse = float(np.mean((x - y) ** 2)) if len(x) else float("nan")
            var_astar = float(np.var(y, ddof=1)) if len(y) >= 2 else float("nan")
            I_lo, I_hi = bootstrap_ci(x, y, ols_slope, args.bootstrap_samples, args.ci, args.seed)
            sp_lo, sp_hi = bootstrap_ci(x, y, spearman, args.bootstrap_samples, args.ci, args.seed + 1)

            rows.append(
                {
                    "seller_id": seller_id,
                    "seller_index": int(seller["seller_index"]),
                    "method": method,
                    "I_hat": I,
                    "I_hat_lo": I_lo,
                    "I_hat_hi": I_hi,
                    "spearman": sp,
                    "spearman_lo": sp_lo,
                    "spearman_hi": sp_hi,
                    "K_hat": K,
                    "SNR_hat_calibrated": snr_cal,
                    "SNR_hat_shortcut_unscaled": snr_shortcut_from_I(I),
                    "var_astar": var_astar,
                    "mse_uncalibrated": mse,
                    "T": int(len(sub)),
                }
            )

    output_path = resolve_repo_path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(df.to_string(index=False))
    print(f"[I] wrote -> {output_path}")


if __name__ == "__main__":
    main()
