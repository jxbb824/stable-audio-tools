#!/usr/bin/env python3
"""Recompute Tables 2 and 6 from saved catalog/query scores, without a GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = {
    "sao50": ("sao_ip_100/outputs", 50, 200, {
        "EKFAC": "a_hat_ekfac_v100_layer_sharded.csv",
        "D-TRAK": "a_hat_dtrak_v100_chunked.csv",
        "Random": "baselines/random/a_hat_random.csv",
        "CLAP": "baselines/clap/a_hat_clap.csv",
        "Fingerprint": "baselines/fingerprint/a_hat_fingerprint.csv",
    }),
    "sao22": ("sao_ip_pilot/outputs", 22, 200, {
        "EKFAC": "a_hat_ekfac.csv", "D-TRAK": "a_hat_dtrak.csv",
        "Random": "baselines/random/a_hat_random.csv",
        "CLAP": "baselines/clap/a_hat_clap.csv",
        "Fingerprint": "baselines/fingerprint/a_hat_fingerprint.csv",
    }),
    "amt100": ("anticipation_ip_pilot/outputs", 100, 500, {
        "LoGra": "selected/a_hat_logra.csv", "TRAK": "selected/a_hat_trak.csv",
        "Random": "baselines/random/a_hat_random.csv",
        "CLAP": "baselines/clap/a_hat_clap.csv",
        "Fingerprint": "baselines/fingerprint/a_hat_fingerprint.csv",
    }),
}
# Rounded entries in the supplied final manuscript (Tables 2 and 6).
# The four columns are Spearman, Pearson, median calibrated SNR, gain percent.
PAPER = {
    "sao50": {"EKFAC": (.007, .109, .023, 9.36), "D-TRAK": (.031, .125, .009, 9.18),
              "Random": (.011, .053, .003, .53), "CLAP": (.046, .148, .016, 4.96),
              "Fingerprint": (.029, .150, .006, 5.80)},
    "sao22": {"EKFAC": (.050, .188, .092, 10.59), "D-TRAK": (.054, .177, .021, 9.77),
              "Random": (.053, .092, .002, .27), "CLAP": (.127, .289, .133, 16.58),
              "Fingerprint": (.114, .290, .035, 10.94)},
    "amt100": {"LoGra": (.068, .194, .017, 6.48), "TRAK": (.035, .087, .002, 1.38),
               "Random": (.001, .005, .001, .15), "CLAP": (.004, .007, .012, 1.42),
               "Fingerprint": (.012, .109, .006, 9.09)},
}


def column_correlations(x, y):
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    denominator = np.sqrt((x*x).sum(axis=0) * (y*y).sum(axis=0))
    return np.divide((x*y).sum(axis=0), denominator,
                     out=np.full(x.shape[1], np.nan), where=denominator > 0)


def read_scores(path):
    frame = pd.read_csv(path, dtype={"seller_id": str, "prompt_id": str})
    if frame.duplicated(["seller_id", "prompt_id"]).any():
        raise ValueError(f"duplicate catalog/query keys: {path}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "preservation/results")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries, catalog_rows, query_rows, provenance = [], [], [], []
    for experiment, (directory, catalogs, queries, methods) in EXPERIMENTS.items():
        base = ROOT / directory
        gt_path = base / ("selected/a_star.csv" if experiment == "amt100" else "a_star.csv")
        truth = read_scores(gt_path)
        target = truth.pivot(index="seller_id", columns="prompt_id", values="a_star").sort_index().sort_index(axis=1)
        if target.shape != (catalogs, queries) or target.isna().any().any():
            raise ValueError(f"incomplete reference: {experiment}, {target.shape}")
        y = target.to_numpy()
        yc = y - y.mean(axis=1, keepdims=True)
        vy = np.mean(yc**2, axis=1)
        for method, relative in methods.items():
            path = base / relative
            scores = read_scores(path)
            column, = [c for c in scores if c.startswith("a_hat")]
            table = scores.pivot(index="seller_id", columns="prompt_id", values=column)
            if set(table.index) != set(target.index) or set(table.columns) != set(target.columns):
                raise ValueError(f"score/reference key mismatch: {path}")
            x = table.reindex(index=target.index, columns=target.columns).to_numpy()
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"nonfinite scores: {path}")
            xc = x - x.mean(axis=1, keepdims=True)
            vx = np.mean(xc**2, axis=1)
            cov = np.mean(xc*yc, axis=1)
            slope = np.divide(cov, vx, out=np.zeros_like(cov), where=vx > 0)
            explained = np.divide(cov**2, vx, out=np.zeros_like(cov), where=vx > 0)
            r2 = np.divide(explained, vy, out=np.zeros_like(vy), where=vy > 0)
            if np.any(r2 > 1 + 1e-10):
                raise ValueError("invalid variance decomposition")
            r2 = np.clip(r2, 0, 1)
            snr = np.divide(r2, 1-r2, out=np.full_like(r2, np.inf), where=r2 < 1)
            pearson = column_correlations(x, y)
            spearman = column_correlations(pd.DataFrame(x).rank(axis=0).to_numpy(),
                                            pd.DataFrame(y).rank(axis=0).to_numpy())
            measured = (float(np.nanmean(spearman)), float(np.nanmean(pearson)),
                        float(np.median(snr)), float(100*explained.sum()/vy.sum()))
            rounded = tuple(round(v, digits) for v, digits in zip(measured, (3, 3, 3, 2)))
            if rounded != PAPER[experiment][method]:
                raise ValueError(f"paper mismatch {experiment}/{method}: {rounded}")
            summaries.append(dict(experiment=experiment, method=method, catalogs=catalogs,
                                  queries=queries, defined_correlation_queries=int(np.isfinite(pearson).sum()),
                                  mean_spearman=measured[0], mean_pearson=measured[1],
                                  median_calibrated_snr=measured[2], welfare_gap_closed_percent=measured[3],
                                  negative_calibration_slopes=int((slope < 0).sum()),
                                  matches_paper_rounding=True, a_star_path=str(gt_path.relative_to(ROOT)),
                                  a_hat_path=str(path.relative_to(ROOT))))
            for i, seller in enumerate(target.index):
                catalog_rows.append(dict(experiment=experiment, method=method, seller_id=seller,
                                         calibration_slope=slope[i], calibrated_snr=snr[i],
                                         reference_variance=vy[i], explained_variance=explained[i],
                                         gain_share=explained[i]/explained.sum() if explained.sum() else 0))
            for i, prompt in enumerate(target.columns):
                query_rows.append(dict(experiment=experiment, method=method, prompt_id=prompt,
                                       pearson=pearson[i], spearman=spearman[i]))
            for source in (gt_path, path):
                provenance.append(dict(path=str(source.relative_to(ROOT)),
                                       sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
    for name, rows in (("paper_metrics", summaries), ("per_catalog", catalog_rows), ("per_query", query_rows)):
        pd.DataFrame(rows).to_csv(args.output_dir / f"{name}.csv", index=False)
    unique_sources = {row["path"]: row for row in provenance}
    (args.output_dir / "sources.json").write_text(json.dumps(list(unique_sources.values()), indent=2) + "\n")
    print(pd.DataFrame(summaries).drop(columns=["a_star_path", "a_hat_path"]).to_string(index=False))
    print("All 60 reported metric cells match Tables 2 and 6 at their printed precision.")


if __name__ == "__main__":
    main()
