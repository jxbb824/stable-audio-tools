#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU dispatcher for anticipation IP pilot setup/analysis.")
    parser.add_argument("--phase", choices=["setup"], required=True)
    parser.add_argument("--config", default="anticipation_ip_pilot/scripts/pilot.yaml")
    args = parser.parse_args()

    if args.phase == "setup":
        run([sys.executable, "anticipation_ip_pilot/scripts/select_sellers.py", "--config", args.config])
        run([sys.executable, "anticipation_ip_pilot/scripts/build_queries.py", "--config", args.config])


if __name__ == "__main__":
    main()
