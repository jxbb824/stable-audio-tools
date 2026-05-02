#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import ensure_dir, load_config, resolve_repo_path, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Copy the first prompted continuations as fixed query samples.")
    parser.add_argument("--config", default="anticipation_ip_pilot/scripts/pilot.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    source = resolve_repo_path(cfg["generated_file"])
    target = resolve_repo_path(cfg["queries_file"])
    ensure_dir(target.parent)
    n = int(cfg["num_prompts"])

    rows = []
    with source.open("r", encoding="utf-8") as src, target.open("w", encoding="utf-8") as dst:
        for query_index, line in enumerate(src):
            if query_index >= n:
                break
            text = line.strip()
            if not text:
                continue
            dst.write(text + "\n")
            rows.append(
                {
                    "prompt_id": f"query_{query_index:04d}",
                    "query_index": query_index,
                    "source_index": query_index,
                    "num_tokens": len(text.split()),
                }
            )

    write_csv(
        resolve_repo_path(cfg["outputs_dir"]) / "queries" / "query_manifest.csv",
        ["prompt_id", "query_index", "source_index", "num_tokens"],
        rows,
    )
    print(f"[queries] wrote {len(rows)} queries -> {target}")


if __name__ == "__main__":
    main()
