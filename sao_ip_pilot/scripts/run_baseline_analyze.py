#!/usr/bin/env python3
"""Aggregate + analyze one SAO IP baseline method."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import REPO_ROOT, resolve_repo_path


def run(cmd: list[str]) -> None:
    print("[sao-baseline]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["random_influence", "clap_similarity", "fingerprint_similarity"],
    )
    parser.add_argument("--score-path", required=True)
    parser.add_argument(
        "--train-axis-manifest",
        default="sao_ip_pilot/outputs/ekfac_attribution/train_axis_manifest.csv",
    )
    parser.add_argument(
        "--query-axis-manifest",
        default="sao_ip_pilot/outputs/ekfac_attribution/query_axis_manifest.csv",
    )
    parser.add_argument(
        "--category-manifest",
        default="sao_ip_pilot/outputs/selection/category_manifest.csv",
    )
    parser.add_argument("--a-star-path", default="sao_ip_pilot/outputs/a_star.csv")
    parser.add_argument(
        "--seller-manifest",
        default="sao_ip_pilot/outputs/selection/seller_manifest.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to sao_ip_pilot/outputs/baselines/<method_stem>.",
    )
    args = parser.parse_args()

    stem = {
        "random_influence": "random",
        "clap_similarity": "clap",
        "fingerprint_similarity": "fingerprint",
    }[args.method]
    column = {
        "random": "a_hat_random",
        "clap": "a_hat_clap",
        "fingerprint": "a_hat_fingerprint",
    }[stem]
    label = {
        "random": "Random",
        "clap": "CLAP",
        "fingerprint": "Fingerprint",
    }[stem]
    out_dir = (
        resolve_repo_path(args.output_dir)
        if args.output_dir
        else resolve_repo_path(f"sao_ip_pilot/outputs/baselines/{stem}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    a_hat = out_dir / f"a_hat_{stem}.csv"
    i_hat = out_dir / f"I_hat_{stem}.csv"
    figures = out_dir / f"figures_{stem}"
    welfare = out_dir / f"marginal_welfare_numbers_{stem}.csv"

    py = str(Path(sys.executable))
    scripts = resolve_repo_path("sao_ip_pilot/scripts")

    run(
        [
            py,
            str(scripts / "aggregate_ekfac_to_sellers.py"),
            "--score-path",
            str(resolve_repo_path(args.score_path)),
            "--train-axis-manifest",
            args.train_axis_manifest,
            "--query-axis-manifest",
            args.query_axis_manifest,
            "--category-manifest",
            args.category_manifest,
            "--output-path",
            str(a_hat),
            "--score-column",
            column,
        ]
    )
    run(
        [
            py,
            str(scripts / "estimate_I.py"),
            "--method",
            args.method,
            "--a-hat-path",
            str(a_hat),
            "--a-star-path",
            args.a_star_path,
            "--seller-manifest",
            args.seller_manifest,
            "--output-path",
            str(i_hat),
        ]
    )
    run(
        [
            py,
            str(scripts / "plot_results.py"),
            "--i-hat-path",
            str(i_hat),
            "--a-hat-path",
            str(a_hat),
            "--seller-manifest",
            args.seller_manifest,
            "--output-dir",
            str(figures),
            "--marginal-welfare-csv",
            str(welfare),
            "--method-label",
            label,
        ]
    )
    print(f"[sao-baseline] done method={args.method} -> {out_dir}")


if __name__ == "__main__":
    main()
