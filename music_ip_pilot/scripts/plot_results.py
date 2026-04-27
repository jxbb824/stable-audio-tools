#!/usr/bin/env python3
"""
Phase 6b: Plot results
======================
Two figures for the pilot:
  (1) I_scatter.pdf : per-seller I_hat with CI bars (per method).
  (2) ranking_comparison.pdf : seller ranking by cardinal I_hat vs Spearman.

These are sanity-check visualizations for the pilot, not publication
figures. Full-scale plots (e.g., |dL/dSNR| curves) are deferred.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

# Matplotlib non-interactive backend for headless compute nodes
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import figures_dir, outputs_dir


def _plot_I_scatter(df: pd.DataFrame, fig_path: Path) -> None:
    sellers = sorted(df["seller_id"].unique())
    methods = sorted(df["method"].unique())

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(sellers))
    width = 0.8 / max(1, len(methods))
    for mi, m in enumerate(methods):
        sub = df[df["method"] == m].set_index("seller_id").reindex(sellers)
        vals = sub["I_hat"].to_numpy()
        lo = sub["I_hat_lo"].to_numpy()
        hi = sub["I_hat_hi"].to_numpy()
        # asymmetric errorbars
        err_lo = np.clip(vals - lo, 0, None)
        err_hi = np.clip(hi - vals, 0, None)
        ax.errorbar(
            x + (mi - (len(methods)-1)/2) * width,
            vals,
            yerr=[err_lo, err_hi],
            fmt="o",
            capsize=4,
            label=m.replace("_", " "),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(sellers, rotation=0)
    ax.set_ylabel(r"$\widehat{\mathcal{I}}_j$")
    ax.set_xlabel("Seller")
    ax.set_title(r"Per-seller informativeness $\widehat{\mathcal{I}}_j$ (pilot)")
    ax.axhline(0.0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[plot] wrote {fig_path}", flush=True)


def _plot_ranking_comparison(df: pd.DataFrame, fig_path: Path) -> None:
    methods = sorted(df["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4), squeeze=False)
    for mi, m in enumerate(methods):
        ax = axes[0, mi]
        sub = df[df["method"] == m].copy()
        rank_I = sub["I_hat"].rank(ascending=False).to_numpy()
        rank_sp = sub["spearman"].rank(ascending=False).to_numpy()
        ax.scatter(rank_I, rank_sp)
        for _, row in sub.reset_index().iterrows():
            ax.annotate(row["seller_id"],
                        (rank_I[row.name], rank_sp[row.name]),
                        fontsize=8, ha="left", va="bottom")
        lim = [0.5, sub.shape[0] + 0.5]
        ax.plot(lim, lim, color="gray", linestyle="--", linewidth=0.5)
        ax.set_xlim(lim); ax.set_ylim(lim)
        ax.set_xlabel(r"Rank by $\widehat{\mathcal{I}}_j$ (cardinal)")
        ax.set_ylabel("Rank by Spearman")
        ax.set_title(m.replace("_", " "))
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"[plot] wrote {fig_path}", flush=True)


def main(cfg: Dict[str, Any]) -> None:
    src = outputs_dir(cfg) / "I_hat.csv"
    if not src.exists():
        raise FileNotFoundError(f"I_hat.csv missing: {src}")
    df = pd.read_csv(src)
    figs = figures_dir(cfg)
    _plot_I_scatter(df, figs / "I_scatter.pdf")
    _plot_ranking_comparison(df, figs / "ranking_comparison.pdf")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    from common import load_config
    main(load_config(args.config))
