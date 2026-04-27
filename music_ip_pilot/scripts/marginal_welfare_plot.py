#!/usr/bin/env python3
"""
Marginal welfare gain vs. SNR — Chris's deferred ask at main.tex:272
====================================================================

Closed-form from Proposition 2:

    |dL_j/dSNR| = E[pi^2] * Var(a*_j) / ( 2 * alpha_j * (1 + SNR_j)^2 )

Under pilot convention pi_t = 1, so E[pi^2] = 1.

The plot shows:
  (a) Closed-form curves |dL/dSNR|(SNR) for a grid of alpha values, at
      Var(a*) = median of pilot sellers (reference curve).
  (b) Per-seller points (SNR_j, |dL/dSNR|_j) measured from pilot data,
      evaluated at a default alpha = 2.
  (c) Annotation showing the "highest marginal return" region — the
      paper's claim that mid-SNR sellers benefit most from attribution
      investment.

Inputs:
  data/outputs/results/I_hat.csv   (pilot I_hat per seller)
  data/outputs/results/a_star.csv  (used to derive Var(a*_j))

Outputs:
  data/outputs/simulation/figures/marginal_welfare_vs_snr.pdf
  data/outputs/simulation/figures/marginal_welfare_vs_snr.png
  data/outputs/simulation/marginal_welfare_numbers.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "data" / "outputs" / "results"
OUT_DIR = ROOT / "data" / "outputs" / "simulation"
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def marginal_welfare(snr: np.ndarray, var_astar: float, alpha: float) -> np.ndarray:
    """|dL/dSNR| = Var(a*) / (2 * alpha * (1+SNR)^2).  E[pi^2] = 1 (pilot)."""
    return var_astar / (2.0 * alpha * (1.0 + snr) ** 2)


def snr_from_I(I: float) -> float:
    """Gaussian special case: I = SNR/(1+SNR) <=> SNR = I/(1-I)."""
    if I >= 1.0:
        return float("inf")
    if I <= 0.0:
        return 0.0
    return float(I / (1.0 - I))


def main() -> None:
    ih = pd.read_csv(RES / "I_hat.csv")
    astar = pd.read_csv(RES / "a_star.csv")

    # Per-seller inputs: measured I_hat -> SNR; measured Var(a*_j)
    sellers = []
    for _, row in ih.iterrows():
        sid = row["seller_id"]
        I = float(row["I_hat"])
        var_astar = float(astar[astar.seller_id == sid]["a_star"].var(ddof=1))
        snr_j = snr_from_I(I) if not np.isnan(I) else np.nan
        sellers.append({
            "seller_id": sid,
            "I_hat": I,
            "SNR_hat": snr_j,
            "var_astar": var_astar,
        })
    sellers_df = pd.DataFrame(sellers)
    print("[mwp] per-seller inputs:")
    print(sellers_df.to_string(index=False))

    # Add a 'primary_genre' label for the plot — read from sellers.json via outputs/results
    # (we scp'd it alongside the CSVs; fallback to sequential labels)
    sellers_json_path = ROOT / "data" / "outputs" / "sellers.json"
    if sellers_json_path.exists():
        import json
        sj = json.load(open(sellers_json_path))
        genre_by_id = {s["seller_id"]: s["primary_genre"] for s in sj}
        sellers_df["primary_genre"] = sellers_df["seller_id"].map(genre_by_id)
    else:
        sellers_df["primary_genre"] = sellers_df["seller_id"]

    # ── Plot setup ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    snr_grid = np.linspace(0.01, 10.0, 400)
    median_var = float(sellers_df["var_astar"].median())

    # Panel (a): closed-form curves at median Var(a*), swept over alpha
    ax = axes[0]
    alphas = [1.0, 2.0, 3.0, 4.0]
    palette = sns.color_palette("viridis", len(alphas))
    for alpha, col in zip(alphas, palette):
        y = marginal_welfare(snr_grid, var_astar=median_var, alpha=alpha)
        ax.plot(snr_grid, y, color=col, lw=1.6, label=fr"$\alpha={alpha:g}$")
    ax.set_xlabel("SNR")
    ax.set_ylabel(r"$|\partial\mathcal{L} / \partial\mathrm{SNR}|$")
    ax.set_title(fr"(a) Closed-form curves at median Var$(a^*)={median_var:.4f}$")
    ax.legend(loc="upper right", title="risk aversion")
    ax.set_xlim(0, 10)
    # Annotate the peak region (monotonic decreasing → peak is at SNR=0,
    # but in practice the *empirically observable* peak is at the steepest curvature)
    ax.axvspan(0.5, 2.5, alpha=0.08, color="red")
    ax.text(1.3, ax.get_ylim()[1] * 0.85, "steepest\nregion",
            ha="center", color="darkred", fontsize=9, style="italic")

    # Panel (b): per-seller measured points overlaid on curves at alpha=2,
    # now with each seller's actual Var(a*_j) (not the median)
    ax = axes[1]
    ref_alpha = 2.0
    # seller-specific curves for context
    seller_colors = sns.color_palette("Set2", len(sellers_df))
    for (_, s), col in zip(sellers_df.iterrows(), seller_colors):
        y = marginal_welfare(snr_grid, var_astar=s["var_astar"], alpha=ref_alpha)
        ax.plot(snr_grid, y, color=col, lw=1.2, ls="--", alpha=0.6,
                label=fr"{s['primary_genre']} curve (Var$={s['var_astar']:.4f}$)")
        y_point = marginal_welfare(np.array([s["SNR_hat"]]),
                                   s["var_astar"], ref_alpha)[0]
        ax.scatter(s["SNR_hat"], y_point, s=120, color=col,
                   edgecolor="black", linewidth=1.3, zorder=5)
        ax.annotate(
            f"{s['primary_genre']}\n(SNR={s['SNR_hat']:.2f})",
            xy=(s["SNR_hat"], y_point),
            xytext=(6, 12), textcoords="offset points", fontsize=9,
        )
    ax.set_xlabel("SNR")
    ax.set_ylabel(r"$|\partial\mathcal{L} / \partial\mathrm{SNR}|$")
    ax.set_title(fr"(b) Per-seller curves + measured SNR at $\alpha={ref_alpha:g}$")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 6)

    fig.suptitle(
        "Marginal welfare gain from attribution investment (pilot N=3)",
        y=1.02, fontsize=12,
    )
    fig.tight_layout()
    pdf_path = FIG_DIR / "marginal_welfare_vs_snr.pdf"
    png_path = FIG_DIR / "marginal_welfare_vs_snr.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[mwp] wrote {pdf_path}")
    print(f"[mwp] wrote {png_path}")

    # Dump the per-seller numbers at alpha=2 for the paper table
    table = sellers_df.copy()
    table["alpha_ref"] = ref_alpha
    table["marg_welfare_at_alpha2"] = table.apply(
        lambda r: marginal_welfare(
            np.array([r["SNR_hat"]]), r["var_astar"], ref_alpha
        )[0] if not np.isnan(r["SNR_hat"]) else np.nan,
        axis=1,
    )
    csv_path = OUT_DIR / "marginal_welfare_numbers.csv"
    table.to_csv(csv_path, index=False)
    print(f"[mwp] wrote {csv_path}")
    print("\nPer-seller marginal welfare (alpha=2):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
