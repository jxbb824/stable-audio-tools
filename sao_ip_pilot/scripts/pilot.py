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
    parser = argparse.ArgumentParser(description="Small dispatcher for CPU-only SAO IP pilot phases.")
    parser.add_argument("--phase", required=True, choices=["setup", "analyze"])
    parser.add_argument("--config", default="sao_ip_pilot/scripts/pilot.yaml")
    args = parser.parse_args()

    if args.phase == "setup":
        run([sys.executable, "sao_ip_pilot/scripts/select_sellers.py", "--config", args.config])
        run([sys.executable, "sao_ip_pilot/scripts/build_prompts.py", "--config", args.config])
    elif args.phase == "analyze":
        run([sys.executable, "sao_ip_pilot/scripts/estimate_I.py"])


if __name__ == "__main__":
    main()
