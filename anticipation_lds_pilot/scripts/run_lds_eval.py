#!/usr/bin/env python3
"""Evaluate LoGra/TRAK attribution scores with subset-model LDS."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PILOT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PILOT_ROOT.parent
SCRIPTS = PILOT_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LDS evaluation for one or more score files.")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["logra", "trak"],
        choices=["logra", "trak"],
    )
    parser.add_argument("--attribution-dir", default="anticipation_lds_pilot/outputs/attribution")
    parser.add_argument("--groundtruth-path", default="anticipation_lds_pilot/outputs/lds/gt_gen.pt")
    parser.add_argument("--subsets-root", default="anticipation_lds_pilot/outputs/models/subsets")
    parser.add_argument("--pool-indices-file", default="anticipation_ip_pilot/outputs/selection/seller_indices.txt")
    parser.add_argument("--output-dir", default="anticipation_lds_pilot/outputs/lds")
    parser.add_argument("--num-subsets", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    attr_dir = (REPO_ROOT / args.attribution_dir).resolve()
    score_map = {
        "logra": attr_dir / "score_LoGra.pt",
        "trak": attr_dir / "score_TRAK.pt",
    }

    py = sys.executable
    for method in args.methods:
        score_path = score_map[method]
        if not score_path.exists():
            print(f"[skip] missing score file for {method}: {score_path}")
            continue
        subprocess.run(
            [
                py,
                str(SCRIPTS / "spearman_lds.py"),
                "--score-path",
                str(score_path),
                "--groundtruth-path",
                args.groundtruth_path,
                "--subsets-root",
                args.subsets_root,
                "--pool-indices-file",
                args.pool_indices_file,
                "--num-subsets",
                str(args.num_subsets),
                "--score-label",
                method.upper(),
                "--output-dir",
                args.output_dir,
            ],
            cwd=REPO_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
