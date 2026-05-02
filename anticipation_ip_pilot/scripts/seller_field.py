#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import read_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Print one seller manifest field for SLURM scripts.")
    parser.add_argument("--manifest", default="anticipation_ip_pilot/outputs/selection/seller_manifest.csv")
    parser.add_argument("--seller-index", type=int, required=True, help="Zero-based array index.")
    parser.add_argument("--field", required=True)
    args = parser.parse_args()

    rows = read_csv_rows(args.manifest)
    if args.seller_index < 0 or args.seller_index >= len(rows):
        raise IndexError(f"seller-index {args.seller_index} outside [0, {len(rows)})")
    print(rows[args.seller_index][args.field])


if __name__ == "__main__":
    main()
