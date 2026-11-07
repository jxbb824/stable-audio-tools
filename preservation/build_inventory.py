#!/usr/bin/env python3
"""Inventory the final-paper archive. Never upload, delete, or copy large files."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("sao_ip_100", "sao_ip_pilot", "anticipation_ip_pilot")
SOURCE_SUFFIXES = {".py", ".sh", ".slurm", ".yaml", ".yml", ".json", ".toml", ".md"}
TABLE_SUFFIXES = {".csv", ".json", ".pdf", ".png", ".svg"}
MODEL_SUFFIXES = {".ckpt", ".safetensors"}
EXCLUDED_NAMES = {"nvidia_smi.csv", "monitor_state.json", "pipeline_jobs.json", "move_models.sh"}
PRIVATE_NAMES = {"candidate_pool.csv", "candidate_pool_paths.txt", "selected_paths.txt",
                 "train_axis_manifest.csv", "query_axis_manifest.csv", "prompts.json",
                 "prompts_artist_only.json", "resolved_dataset_config.json",
                 "resolved_train_dataset_config.json", "resolved_query_dataset_config.json"}


def files_under(path: Path):
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from (p for p in sorted(path.rglob("*")) if p.is_file()
                    and "__pycache__" not in p.parts and ".git" not in p.parts)


def classify(path: Path) -> tuple[str, str, str]:
    """Return destination, purpose, and optional identical-weight alias."""
    rel = path.relative_to(ROOT)
    parts = rel.parts
    if path.suffix in {".log", ".out", ".err", ".lock", ".pyc"} or path.name in EXCLUDED_NAMES:
        return "excluded", "runtime log, telemetry, or temporary state", ""
    if parts[0] == "estimator_diagnosis":
        if path.suffix in SOURCE_SUFFIXES | {".txt"}:
            return "github", "attribution and scoring diagnostic code/results", ""
    if len(parts) > 1 and parts[1] != "outputs":
        if parts[1] == "assets":
            return "redownload", "publicly downloadable rendering soundfont", ""
        if "data" in parts:
            return "hf_private", "symbolic training data and metadata", ""
        if path.suffix in SOURCE_SUFFIXES:
            return "github", "experiment source and configuration", ""
    if "outputs" not in parts:
        return "excluded", "outside selected experiment artifacts", ""
    tail = parts[parts.index("outputs") + 1:]
    if not tail:
        return "excluded", "empty path", ""
    if parts[0] == "anticipation_ip_pilot" and tail[0] == "models":
        if len(tail) < 2 or tail[1] not in {"full-selected", "loo-selected"}:
            return "excluded", "superseded symbolic experiment model", ""
        if (tail[1] == "full-selected" and path.name == "model.safetensors"
                and (len(tail) == 3 or tail[2] == "full_model")):
            return "alias", "identical final weight; SHA-256 checked", (
                "anticipation_ip_pilot/outputs/models/full-selected/4/model.safetensors")
    if tail[0] in {"models", "replicates"}:
        if path.suffix in MODEL_SUFFIXES or path.name in {"config.json", "generation_config.json", "resolved_model_config.json"}:
            return "hf_public", "trained model or model architecture configuration", ""
        return "hf_private", "training subset, seed, and fit provenance", ""
    if (tail[0] in {"selection", "queries"} or tail[0].startswith("generated_queries")
            or path.name in PRIVATE_NAMES or "manifest" in path.name):
        return "hf_private", "data membership, exact evaluation queries, and axis order", ""
    if parts[0] == "anticipation_ip_pilot" and tail[:2] == ("baselines", "clap") and "audio" in tail:
        return "excluded", "regenerable symbolic-to-audio rendering cache", ""
    if any("probe" in p or p == "ekfac_layer_shard_production_validation" for p in tail):
        if path.suffix in {".json", ".csv", ".txt"}:
            return "github", "small validation result", ""
        return "excluded", "temporary validation output", ""
    if path.suffix in TABLE_SUFFIXES:
        return "github", "numerical results, scoring provenance, or figures", ""
    return "hf_private", "raw scores, features, losses, or numerical-array metadata", ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "preservation")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}

    def add(path: Path, destination: str, purpose: str, alias: str = "", remote_path: str = ""):
        path = path.absolute()
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            rel = path.relative_to(ROOT).as_posix()
        except ValueError:
            rel = "../" + path.relative_to(ROOT.parent).as_posix()
        record = {"source": rel, "destination": destination,
                  "remote_path": remote_path or (rel if destination not in {"excluded", "redownload", "alias"} else ""),
                  "bytes": path.stat().st_size, "purpose": purpose,
                  "alias_of": alias, "sha256": ""}
        if destination == "github":
            record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif destination == "alias":
            record["sha256"] = "24e6ac41eba15d32b6958add921627e5b6e9c5997d5b1de6cac01d946e1195da"
        records[rel] = record

    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    for rel in tracked:
        path = ROOT / rel
        if rel and path.is_file() and not rel.startswith("preservation/"):
            add(path, "github", "existing tracked source/documentation")

    for experiment in EXPERIMENTS:
        for path in files_under(ROOT / experiment):
            rel = path.relative_to(ROOT).as_posix()
            # Existing source files remain tracked; do not newly expose training data.
            if rel in records:
                continue
            destination, purpose, alias = classify(path)
            add(path, destination, purpose, alias)
    for path in files_under(ROOT / "estimator_diagnosis"):
        add(path, *classify(path))
    for path in files_under(ROOT / "scripts"):
        if path.suffix in SOURCE_SUFFIXES and not path.name.startswith("tmp_"):
            add(path, "github", "shared training, scoring, and validation source")

    # Only the union actually selected by the two paper experiments, not 20k songs.
    audio = set()
    for experiment in EXPERIMENTS[:2]:
        manifest = ROOT / experiment / "outputs/selection/selected_paths_full.txt"
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            old = Path(line.strip())
            marker = "/dataset/"
            if marker not in str(old):
                raise ValueError(f"unexpected audio path: {old}")
            audio.add(ROOT / "dataset" / str(old).split(marker, 1)[1])
    for path in sorted(audio):
        add(path, "hf_private", "selected SAO training audio; original bytes and relative layout")
    for subset in ("small-5000", "small-20000"):
        for name in ("raw_30s.tsv", "raw.meta.tsv"):
            add(ROOT / "dataset" / subset / name, "hf_private", "original metadata used by prompt construction")
    add(ROOT / "outputs/dac_dedup/exclude_paths.txt", "hf_private", "training/attribution exclusion list")

    # Keep the exact selected symbolic indices valid by retaining the full indexed text files.
    for name in ("train_v2.txt", "test_v2.txt", "generated_samples_prompted.txt"):
        candidates = list((ROOT / "anticipation_ip_pilot/data").rglob(name))
        if len(candidates) != 1:
            raise ValueError(f"expected one {name}; found {candidates}")
        add(candidates[0], "hf_private", "indexed symbolic training/validation/evaluation sequences")
    # Redundant source text can be regenerated from train/test; do not upload it twice.
    for path in (ROOT / "anticipation_ip_pilot/data").rglob("tokenized-events-song_v2_shuf.txt"):
        add(path, "excluded", "redundant concatenated symbolic corpus")

    for path in files_under(ROOT / "dataset/pretrained_model"):
        add(path, "redownload", "public SAO base model / VAE; pinned upstream in README")
    if (ROOT / "exported_model.ckpt").exists():
        add(ROOT / "exported_model.ckpt", "excluded", "redundant exported base weights")
    for dirname in ("outputs/groundtruth_models", "outputs/groundtruth_models_old",
                    "anticipation_lds_pilot/outputs", "anticipation_lds_pilot/outputs_e25",
                    "anticipation_lds_pilot/outputs_logra_epoch_sweep"):
        for path in files_under(ROOT / dirname):
            add(path, "excluded", "LDS/random-subset experiment absent from final paper")

    paper = ROOT.parent / "name-free-attribution/draft/23375_What_s_a_Credit_Worth_A_.pdf"
    if paper.exists():
        add(paper, "hf_private", "exact submitted manuscript used to set archive scope",
            remote_path="paper/submitted_manuscript.pdf")
    rows = sorted(records.values(), key=lambda row: (row["destination"], row["source"]))
    with (out / "inventory.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    totals = defaultdict(lambda: {"files": 0, "bytes": 0})
    for row in rows:
        item = totals[row["destination"]]
        item["files"] += 1
        item["bytes"] += row["bytes"]
    summary = {
        "status": "File allocation inventory; see PRESERVATION.md and transfer_probe.json for publication status",
        "base_commit": "0bc4c4fa51925aae2c99788a2738b71d746aaf13",
        "github_branch": "archive/credit-worth-20261107",
        "planned_hf_public_model_repo": "xirui824/credit-worth-models",
        "planned_hf_private_dataset_repo": "xirui824/credit-worth-artifacts",
        "selected_audio_files": len(audio),
        "selected_audio_bytes": sum(p.stat().st_size for p in audio),
        "totals": dict(totals),
        "notes": ["Excluded totals cover inventoried paths, not every unrelated file in the checkout.",
                  "GitHub totals exclude this inventory and other preservation/ files themselves.",
                  "HF payloads retain original model/data bytes; this script does not repack or convert them.",
                  "SHA-256 is populated for small GitHub files only; large payload hashing belongs to transfer verification."]}
    (out / "storage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "github_paths.txt").write_text("".join(row["source"] + "\n" for row in rows if row["destination"] == "github"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
