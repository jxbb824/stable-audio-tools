#!/usr/bin/env python3
"""Serial TRAK hyperparameter grid search; ranks runs by mean Spearman (LDS)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PILOT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PILOT_ROOT.parent
SCRIPTS = PILOT_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a serial TRAK hyperparameter grid search.")
    parser.add_argument(
        "--grid-root",
        default="anticipation_ip_pilot/outputs/grid_search/trak",
        help="Root directory for per-run outputs.",
    )
    parser.add_argument("--train-file", default="anticipation_ip_pilot/data/finetune/train_v2.txt")
    parser.add_argument(
        "--queries-file",
        default="anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="anticipation_ip_pilot/outputs/models/full-selected",
    )
    parser.add_argument(
        "--include-indices-file",
        default="anticipation_ip_pilot/outputs/selection/seller_indices.txt",
    )
    parser.add_argument("--a-star-path", default="anticipation_ip_pilot/outputs/selected/a_star.csv")
    parser.add_argument(
        "--seller-manifest",
        default="anticipation_ip_pilot/outputs/selection/seller_manifest.csv",
    )
    parser.add_argument("--proj-dims", default="4096,8192,16384")
    parser.add_argument("--regularizations", default="0.001,0.01,0.1")
    parser.add_argument("--batch-sizes", default="2,4")
    parser.add_argument("--proj-types", default="random_mask")
    parser.add_argument("--num-checkpoints", type=int, default=5)
    parser.add_argument("--proj-max-batch-size", type=int, default=40)
    parser.add_argument("--num-prompts", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-plots", action="store_true", help="Skip plot_results.py to save time.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"\n[grid] {printable}", flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def run_tag(
    tag: str,
    run_dir: Path,
    args: argparse.Namespace,
    *,
    batch_size: int,
    proj_dim: int,
    regularization: float,
    proj_type: str,
) -> dict:
    attr_dir = run_dir / "attribution"
    metrics_dir = run_dir / "metrics"
    figures_dir = metrics_dir / "figures"
    attr_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "tag": tag,
        "batch_size": batch_size,
        "proj_dim": proj_dim,
        "regularization": regularization,
        "proj_type": proj_type,
        "num_checkpoints": args.num_checkpoints,
        "proj_max_batch_size": args.proj_max_batch_size,
        "num_prompts": args.num_prompts,
        "seed": args.seed,
        "train_file": args.train_file,
        "queries_file": args.queries_file,
        "checkpoint_dir": args.checkpoint_dir,
        "include_indices_file": args.include_indices_file,
        "a_star_path": args.a_star_path,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    score_path = attr_dir / "score_TRAK.pt"
    a_hat_path = run_dir / "a_hat_trak.csv"
    corr_csv = metrics_dir / "query_level_corr_trak.csv"
    corr_json = metrics_dir / "query_level_corr_trak_summary.json"
    i_hat_path = metrics_dir / "I_hat_trak.csv"

    py = sys.executable
    run_cmd(
        [
            py,
            str(SCRIPTS / "score_trak.py"),
            "--train_file",
            args.train_file,
            "--valid_file",
            args.queries_file,
            "--checkpoint_dir",
            args.checkpoint_dir,
            "--output_dir",
            str(attr_dir),
            "--include_indices_file",
            args.include_indices_file,
            "--batch_size",
            str(batch_size),
            "--num_checkpoints",
            str(args.num_checkpoints),
            "--proj_dim",
            str(proj_dim),
            "--proj_type",
            proj_type,
            "--proj_max_batch_size",
            str(args.proj_max_batch_size),
            "--regularization",
            str(regularization),
            "--num_prompts",
            str(args.num_prompts),
            "--seed",
            str(args.seed),
            "--valid_is_generated",
            "--output_filename",
            "score_TRAK.pt",
        ],
        dry_run=args.dry_run,
    )
    run_cmd(
        [
            py,
            str(SCRIPTS / "aggregate_scores_to_sellers.py"),
            "--score-path",
            str(score_path),
            "--seller-manifest",
            args.seller_manifest,
            "--output-path",
            str(a_hat_path),
            "--score-column",
            "a_hat_trak",
            "--include_indices_file",
            args.include_indices_file,
        ],
        dry_run=args.dry_run,
    )
    run_cmd(
        [
            py,
            str(SCRIPTS / "query_level_corr.py"),
            "--method",
            "trak_influence",
            "--a-star-path",
            args.a_star_path,
            "--a-hat-path",
            str(a_hat_path),
            "--output-csv",
            str(corr_csv),
            "--summary-json",
            str(corr_json),
        ],
        dry_run=args.dry_run,
    )
    run_cmd(
        [
            py,
            str(SCRIPTS / "estimate_I.py"),
            "--method",
            "trak_influence",
            "--a-star-path",
            args.a_star_path,
            "--a-hat-path",
            str(a_hat_path),
            "--seller-manifest",
            args.seller_manifest,
            "--output-path",
            str(i_hat_path),
        ],
        dry_run=args.dry_run,
    )

    if not args.skip_plots and not args.dry_run:
        figures_dir.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                py,
                str(SCRIPTS / "plot_results.py"),
                "--i-hat-path",
                str(i_hat_path),
                "--a-hat-path",
                str(a_hat_path),
                "--seller-manifest",
                args.seller_manifest,
                "--output-dir",
                str(figures_dir),
                "--marginal-welfare-csv",
                str(metrics_dir / "marginal_welfare_numbers_trak.csv"),
                "--method-label",
                f"TRAK {tag}",
            ],
            dry_run=False,
        )

    if args.dry_run:
        return {
            "tag": tag,
            "batch_size": batch_size,
            "proj_dim": proj_dim,
            "regularization": regularization,
            "proj_type": proj_type,
            "mean_spearman_lds": float("nan"),
            "median_spearman": float("nan"),
            "mean_pearson": float("nan"),
            "median_pearson": float("nan"),
            "mean_I_hat": float("nan"),
            "median_I_hat": float("nan"),
            "run_dir": str(run_dir),
        }

    with corr_json.open(encoding="utf-8") as handle:
        corr_summary = json.load(handle)
    i_hat = pd.read_csv(i_hat_path)
    return {
        "tag": tag,
        "batch_size": batch_size,
        "proj_dim": proj_dim,
        "regularization": regularization,
        "proj_type": proj_type,
        "mean_spearman_lds": float(corr_summary["mean_spearman"]),
        "median_spearman": float(corr_summary["median_spearman"]),
        "mean_pearson": float(corr_summary["mean_pearson"]),
        "median_pearson": float(corr_summary["median_pearson"]),
        "mean_I_hat": float(i_hat["I_hat"].mean()),
        "median_I_hat": float(i_hat["I_hat"].median()),
        "mean_seller_spearman": float(i_hat["spearman"].mean()),
        "run_dir": str(run_dir),
    }


def main() -> None:
    args = parse_args()
    grid_root = (REPO_ROOT / args.grid_root).resolve()
    grid_root.mkdir(parents=True, exist_ok=True)

    proj_dims = parse_int_list(args.proj_dims)
    regularizations = parse_float_list(args.regularizations)
    batch_sizes = parse_int_list(args.batch_sizes)
    proj_types = parse_str_list(args.proj_types)

    rows: list[dict] = []
    total = len(proj_dims) * len(regularizations) * len(batch_sizes) * len(proj_types)
    run_idx = 0
    for batch_size in batch_sizes:
        for proj_dim in proj_dims:
            for regularization in regularizations:
                for proj_type in proj_types:
                    run_idx += 1
                    tag = f"bs{batch_size}_pd{proj_dim}_r{regularization:g}_pt{proj_type}"
                    run_dir = grid_root / tag
                    print(f"[grid] run {run_idx}/{total}: {tag}", flush=True)
                    rows.append(
                        run_tag(
                            tag,
                            run_dir,
                            args,
                            batch_size=batch_size,
                            proj_dim=proj_dim,
                            regularization=regularization,
                            proj_type=proj_type,
                        )
                    )

    summary = pd.DataFrame(rows).sort_values("mean_spearman_lds", ascending=False, na_position="last")
    summary_path = grid_root / "grid_summary.csv"
    summary.to_csv(summary_path, index=False)

    best = summary.iloc[0].to_dict() if len(summary) else {}
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric": "mean_spearman_lds",
        "num_runs": int(len(summary)),
        "grid": {
            "proj_dims": proj_dims,
            "regularizations": regularizations,
            "batch_sizes": batch_sizes,
            "proj_types": proj_types,
            "num_checkpoints": args.num_checkpoints,
            "proj_max_batch_size": args.proj_max_batch_size,
            "seed": args.seed,
        },
        "best": best,
        "summary_csv": str(summary_path),
    }
    manifest_path = grid_root / "grid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"\n[grid] wrote {summary_path}", flush=True)
    print(f"[grid] wrote {manifest_path}", flush=True)
    if len(summary):
        print(
            "[grid] best LDS="
            f"{best.get('mean_spearman_lds', float('nan')):.6f} "
            f"({best.get('tag', 'n/a')})",
            flush=True,
        )


if __name__ == "__main__":
    main()
