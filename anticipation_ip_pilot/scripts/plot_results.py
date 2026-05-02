#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import resolve_repo_path


BAND_COLORS = {
    "low": "#4C78A8",
    "mid": "#F58518",
    "high": "#54A24B",
    "extra_high": "#B279A2",
}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=180 if suffix == "png" else None)
        print(f"[plot] wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot symbolic IP pilot diagnostics.")
    parser.add_argument("--i-hat-path", required=True)
    parser.add_argument("--a-hat-path", required=True)
    parser.add_argument("--seller-manifest", default="anticipation_ip_pilot/outputs/selection/seller_manifest.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--marginal-welfare-csv", required=True)
    parser.add_argument("--method-label", required=True)
    parser.add_argument("--alpha-ref", type=float, default=2.0)
    args = parser.parse_args()

    ihat = pd.read_csv(resolve_repo_path(args.i_hat_path))
    sellers = pd.read_csv(resolve_repo_path(args.seller_manifest))
    ahat = pd.read_csv(resolve_repo_path(args.a_hat_path))
    counts = (
        ahat.groupby("seller_id", as_index=False)["num_train_examples"]
        .first()
        .rename(columns={"num_train_examples": "train_examples"})
    )
    df = ihat.merge(sellers, on=["seller_id", "seller_index"], how="left").merge(counts, on="seller_id", how="left")
    df["band_color"] = df["prolificness_band"].map(BAND_COLORS).fillna("#777777")
    output_dir = resolve_repo_path(args.output_dir)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    vals = df["I_hat"].to_numpy(dtype=np.float64)
    ax.hist(vals[np.isfinite(vals)], bins=30, color="#4C78A8", edgecolor="white")
    ax.axvline(0.0, color="#6B7280", lw=0.8, ls="--")
    ax.set_xlabel(r"$\widehat{\mathcal{I}}_j$")
    ax.set_ylabel("Seller count")
    ax.set_title(f"{args.method_label} informativeness across sellers")
    save(fig, output_dir, "I_hat_hist")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(df["spearman"], df["SNR_hat_calibrated"], s=30, c=df["band_color"], edgecolor="black", linewidth=0.3)
    ax.axvline(0.0, color="#6B7280", lw=0.8, ls="--")
    ax.set_xlabel("Seller-level Spearman")
    ax.set_ylabel("Calibrated SNR")
    ax.set_title(f"{args.method_label}: SNR vs Spearman")
    ax.grid(alpha=0.25)
    save(fig, output_dir, "snr_vs_spearman")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x = df["train_examples"].to_numpy(dtype=np.float64)
    for ax, col, label in zip(
        axes,
        ["spearman", "I_hat", "SNR_hat_calibrated"],
        ["Spearman", r"$\widehat{\mathcal{I}}_j$", "Calibrated SNR"],
    ):
        y = df[col].to_numpy(dtype=np.float64)
        ax.scatter(x, y, s=30, c=df["band_color"], edgecolor="black", linewidth=0.3)
        ax.text(0.04, 0.95, f"Pearson r={pearson(x, y):.2f}", transform=ax.transAxes, va="top")
        ax.set_xlabel("Seller train examples")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    save(fig, output_dir, "sample_count_diagnostics")

    welfare = df[["seller_id", "seller_index", "artist", "prolificness_band", "train_examples", "I_hat", "SNR_hat_calibrated", "var_astar"]].copy()
    welfare["alpha_ref"] = args.alpha_ref
    welfare["marg_welfare_at_alpha_ref"] = welfare["var_astar"] / (2.0 * args.alpha_ref * (1.0 + welfare["SNR_hat_calibrated"]) ** 2)
    welfare_path = resolve_repo_path(args.marginal_welfare_csv)
    welfare_path.parent.mkdir(parents=True, exist_ok=True)
    welfare.to_csv(welfare_path, index=False)
    print(f"[plot] wrote {welfare_path}")


if __name__ == "__main__":
    main()
