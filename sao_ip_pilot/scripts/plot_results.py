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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SAO IP pilot attribution diagnostics.")
    parser.add_argument("--i-hat-path", default="sao_ip_pilot/outputs/I_hat.csv")
    parser.add_argument("--a-hat-path", default="sao_ip_pilot/outputs/a_hat_ekfac.csv")
    parser.add_argument("--seller-manifest", default="sao_ip_pilot/outputs/selection/seller_manifest.csv")
    parser.add_argument("--output-dir", default="sao_ip_pilot/outputs/figures")
    parser.add_argument("--marginal-welfare-csv", default="sao_ip_pilot/outputs/marginal_welfare_numbers.csv")
    parser.add_argument("--alpha-ref", type=float, default=2.0)
    parser.add_argument("--method-label", default="EKFAC")
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=180 if suffix == "png" else None)
        print(f"[plot] wrote {path}", flush=True)
    plt.close(fig)


def rank_desc(values: pd.Series) -> pd.Series:
    return values.rank(ascending=False, method="average")


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        out[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denom) if denom > 0 else float("nan")


def spearman_rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(ranks(np.asarray(x, dtype=np.float64)), ranks(np.asarray(y, dtype=np.float64)))


def marginal_welfare(snr: np.ndarray, var_astar: np.ndarray | float, alpha: float) -> np.ndarray:
    return np.asarray(var_astar, dtype=np.float64) / (2.0 * alpha * (1.0 + snr) ** 2)


def load_results(args: argparse.Namespace) -> pd.DataFrame:
    ihat = pd.read_csv(resolve_repo_path(args.i_hat_path))
    if "method" in ihat.columns:
        methods = sorted(str(method) for method in ihat["method"].dropna().unique())
        if len(methods) > 1:
            raise ValueError(
                "plot_results expects one attribution method per run; "
                f"got mixed methods: {', '.join(methods)}"
            )
    sellers = pd.read_csv(resolve_repo_path(args.seller_manifest))
    ahat = pd.read_csv(resolve_repo_path(args.a_hat_path))

    counts = (
        ahat.groupby("seller_id", as_index=False)["num_train_examples"]
        .first()
        .rename(columns={"num_train_examples": "train_examples"})
    )
    df = ihat.merge(sellers, on=["seller_id", "seller_index"], how="left").merge(
        counts, on="seller_id", how="left"
    )
    df["label"] = df["seller_id"].str.replace("seller_", "S", regex=False)
    df["band_color"] = df["prolificness_band"].map(BAND_COLORS).fillna("#777777")
    return df.sort_values("seller_index").reset_index(drop=True)


def plot_i_scatter(df: pd.DataFrame, output_dir: Path, method_label: str) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(df))
    vals = df["I_hat"].to_numpy()
    err_lo = np.clip(vals - df["I_hat_lo"].to_numpy(), 0, None)
    err_hi = np.clip(df["I_hat_hi"].to_numpy() - vals, 0, None)
    ax.errorbar(x, vals, yerr=[err_lo, err_hi], fmt="none", ecolor="#4B5563", capsize=3, lw=1)
    ax.scatter(x, vals, s=56, c=df["band_color"], edgecolor="black", linewidth=0.45, zorder=3)
    ax.axhline(0.0, color="#6B7280", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right")
    ax.set_ylabel(r"$\widehat{\mathcal{I}}_j$")
    ax.set_xlabel("Seller")
    ax.set_title(f"Per-seller {method_label} informativeness with bootstrap CI")
    ax.grid(axis="y", alpha=0.25)
    add_band_legend(ax)
    fig.tight_layout()
    save_figure(fig, output_dir, "I_hat_scatter")


def plot_ranking_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    rank_i = rank_desc(df["I_hat"])
    rank_sp = rank_desc(df["spearman"])
    ax.scatter(rank_i, rank_sp, s=60, c=df["band_color"], edgecolor="black", linewidth=0.45)
    for _, row in df.iterrows():
        ax.annotate(row["label"], (rank_i.loc[row.name], rank_sp.loc[row.name]), xytext=(4, 3),
                    textcoords="offset points", fontsize=8)
    lim = [0.5, len(df) + 0.5]
    ax.plot(lim, lim, color="#6B7280", lw=0.8, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel(r"Rank by $\widehat{\mathcal{I}}_j$")
    ax.set_ylabel("Rank by Spearman")
    ax.set_title("Seller rank comparison")
    ax.grid(alpha=0.25)
    add_band_legend(ax)
    fig.tight_layout()
    save_figure(fig, output_dir, "ranking_comparison")


def plot_snr_spearman(df: pd.DataFrame, output_dir: Path) -> None:
    ordered = df.sort_values("SNR_hat_calibrated", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    x = np.arange(len(ordered))

    axes[0].bar(x, ordered["SNR_hat_calibrated"], color=ordered["band_color"], edgecolor="black", lw=0.35)
    axes[0].set_ylabel("Calibrated SNR")
    axes[0].set_title("Calibrated SNR by seller")
    axes[0].grid(axis="y", alpha=0.25)

    sp = ordered["spearman"].to_numpy()
    err_lo = np.clip(sp - ordered["spearman_lo"].to_numpy(), 0, None)
    err_hi = np.clip(ordered["spearman_hi"].to_numpy() - sp, 0, None)
    axes[1].errorbar(x, sp, yerr=[err_lo, err_hi], fmt="none", ecolor="#4B5563", capsize=3, lw=1)
    axes[1].scatter(x, sp, s=50, c=ordered["band_color"], edgecolor="black", linewidth=0.45, zorder=3)
    axes[1].axhline(0.0, color="#6B7280", lw=0.8, ls="--")
    axes[1].set_ylabel("Spearman")
    axes[1].set_title("Rank correlation with ground truth")
    axes[1].grid(axis="y", alpha=0.25)

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(ordered["label"], rotation=45, ha="right")
        ax.set_xlabel("Seller, sorted by calibrated SNR")
    add_band_legend(axes[0])
    fig.tight_layout()
    save_figure(fig, output_dir, "snr_spearman_summary")


def plot_sample_count_diagnostics(df: pd.DataFrame, output_dir: Path) -> None:
    metrics = [
        ("spearman", "Spearman"),
        ("I_hat", r"$\widehat{\mathcal{I}}_j$"),
        ("SNR_hat_calibrated", "Calibrated SNR"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    x = df["train_examples"].to_numpy(dtype=np.float64)
    for ax, (col, label) in zip(axes, metrics):
        y = df[col].to_numpy(dtype=np.float64)
        ax.scatter(x, y, s=60, c=df["band_color"], edgecolor="black", linewidth=0.45)
        for _, row in df.iterrows():
            ax.annotate(row["label"], (row["train_examples"], row[col]), xytext=(4, 3),
                        textcoords="offset points", fontsize=7)
        if np.isfinite(y).sum() >= 2:
            coef = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, coef[0] * xx + coef[1], color="#111827", lw=1.0, alpha=0.75)
        r = pearson(x, y)
        rho = spearman_rank_corr(x, y)
        ax.text(
            0.03,
            0.95,
            f"Pearson r={r:.2f}\nSpearman rho={rho:.2f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#D1D5DB", "alpha": 0.9},
        )
        ax.set_xlabel("Seller train examples")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].set_title("Sample count vs Spearman")
    axes[1].set_title(r"Sample count vs $\widehat{\mathcal{I}}_j$")
    axes[2].set_title("Sample count vs calibrated SNR")
    add_band_legend(axes[2])
    fig.tight_layout()
    save_figure(fig, output_dir, "sample_count_diagnostics")


def plot_marginal_welfare(df: pd.DataFrame, output_dir: Path, csv_path: Path, alpha_ref: float) -> None:
    table = df[[
        "seller_id",
        "seller_index",
        "prolificness_band",
        "train_examples",
        "I_hat",
        "SNR_hat_calibrated",
        "var_astar",
    ]].copy()
    table["alpha_ref"] = alpha_ref
    table["marg_welfare_at_alpha_ref"] = marginal_welfare(
        table["SNR_hat_calibrated"].to_numpy(dtype=np.float64),
        table["var_astar"].to_numpy(dtype=np.float64),
        alpha_ref,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    print(f"[plot] wrote {csv_path}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    snr_grid = np.linspace(0.0, max(1.0, float(df["SNR_hat_calibrated"].max()) * 1.2), 300)
    median_var = float(df["var_astar"].median())
    for alpha, color in zip([1.0, 2.0, 3.0, 4.0], ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]):
        axes[0].plot(
            snr_grid,
            marginal_welfare(snr_grid, median_var, alpha),
            color=color,
            lw=1.7,
            label=fr"$\alpha={alpha:g}$",
        )
    axes[0].set_xlabel("Calibrated SNR")
    axes[0].set_ylabel(r"$|\partial L / \partial \mathrm{SNR}|$")
    axes[0].set_title(fr"Reference curves at median Var$(a^*)={median_var:.2e}$")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)

    y = table["marg_welfare_at_alpha_ref"].to_numpy(dtype=np.float64)
    axes[1].scatter(
        table["SNR_hat_calibrated"],
        y,
        s=70,
        c=df["band_color"],
        edgecolor="black",
        linewidth=0.45,
    )
    for _, row in table.iterrows():
        axes[1].annotate(
            row["seller_id"].replace("seller_", "S"),
            (row["SNR_hat_calibrated"], row["marg_welfare_at_alpha_ref"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axes[1].set_xlabel("Calibrated SNR")
    axes[1].set_ylabel(r"$|\partial L / \partial \mathrm{SNR}|$")
    axes[1].set_title(fr"Per-seller marginal welfare at $\alpha={alpha_ref:g}$")
    axes[1].grid(alpha=0.25)
    add_band_legend(axes[1])
    fig.tight_layout()
    save_figure(fig, output_dir, "marginal_welfare_vs_snr")


def add_band_legend(ax: plt.Axes) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=band, markerfacecolor=color,
                   markeredgecolor="black", markersize=7)
        for band, color in BAND_COLORS.items()
    ]
    ax.legend(handles=handles, title="Band", frameon=False, fontsize=8, title_fontsize=8)


def main() -> None:
    args = parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(args)

    plot_i_scatter(df, output_dir, args.method_label)
    plot_ranking_comparison(df, output_dir)
    plot_snr_spearman(df, output_dir)
    plot_sample_count_diagnostics(df, output_dir)
    plot_marginal_welfare(
        df,
        output_dir,
        resolve_repo_path(args.marginal_welfare_csv),
        args.alpha_ref,
    )


if __name__ == "__main__":
    main()
