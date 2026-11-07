#!/usr/bin/env python3
"""Aggregate a baseline score matrix and run the same IP analyze/plot stack."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import REPO_ROOT, resolve_repo_path


def run(cmd: list[str]) -> None:
    print("[baseline]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate + analyze one baseline method.")
    parser.add_argument(
        "--method",
        required=True,
        choices=["random_influence", "clap_similarity", "fingerprint_similarity"],
    )
    parser.add_argument("--score-path", required=True)
    parser.add_argument(
        "--include-indices-file",
        default="anticipation_ip_pilot/outputs/selection/seller_indices.txt",
    )
    parser.add_argument(
        "--a-star-path",
        default="anticipation_ip_pilot/outputs/selected/a_star.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Defaults to anticipation_ip_pilot/outputs/baselines/<method_stem>",
    )
    args = parser.parse_args()

    column = {
        "random_influence": "a_hat_random",
        "clap_similarity": "a_hat_clap",
        "fingerprint_similarity": "a_hat_fingerprint",
    }[args.method]
    label = {
        "random_influence": "Random",
        "clap_similarity": "CLAP",
        "fingerprint_similarity": "Fingerprint",
    }[args.method]
    stem = {
        "random_influence": "random",
        "clap_similarity": "clap",
        "fingerprint_similarity": "fingerprint",
    }[args.method]

    out_dir = resolve_repo_path(args.output_dir) if args.output_dir else resolve_repo_path(
        f"anticipation_ip_pilot/outputs/baselines/{stem}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    a_hat = out_dir / f"a_hat_{stem}.csv"
    i_hat = out_dir / f"I_hat_{stem}.csv"
    corr_csv = out_dir / f"query_level_corr_{stem}.csv"
    corr_json = out_dir / f"query_level_corr_{stem}_summary.json"
    figures = out_dir / f"figures_{stem}"
    welfare = out_dir / f"marginal_welfare_numbers_{stem}.csv"

    py = Path(sys.executable)
    scripts = resolve_repo_path("anticipation_ip_pilot/scripts")

    run(
        [
            str(py),
            str(scripts / "aggregate_scores_to_sellers.py"),
            "--score-path",
            str(resolve_repo_path(args.score_path)),
            "--output-path",
            str(a_hat),
            "--score-column",
            column,
            "--include_indices_file",
            args.include_indices_file,
        ]
    )
    run(
        [
            str(py),
            str(scripts / "estimate_I.py"),
            "--method",
            args.method,
            "--a-star-path",
            args.a_star_path,
            "--a-hat-path",
            str(a_hat),
            "--output-path",
            str(i_hat),
        ]
    )
    run(
        [
            str(py),
            str(scripts / "query_level_corr.py"),
            "--method",
            args.method,
            "--a-star-path",
            args.a_star_path,
            "--a-hat-path",
            str(a_hat),
            "--output-csv",
            str(corr_csv),
            "--summary-json",
            str(corr_json),
        ]
    )
    run(
        [
            str(py),
            str(scripts / "plot_results.py"),
            "--i-hat-path",
            str(i_hat),
            "--a-hat-path",
            str(a_hat),
            "--output-dir",
            str(figures),
            "--marginal-welfare-csv",
            str(welfare),
            "--method-label",
            label,
        ]
    )
    print(f"[baseline] done method={args.method} -> {out_dir}")


if __name__ == "__main__":
    main()
