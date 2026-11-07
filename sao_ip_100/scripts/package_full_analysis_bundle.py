#!/usr/bin/env python3
"""Package the complete SAO-50 downstream-analysis artifacts into one ZIP."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "sao_ip_100" / "outputs"
BUNDLE_NAME = "sao50_attribution_full_analysis_bundle_20260919"
BUNDLE_PATH = ROOT / f"{BUNDLE_NAME}.zip"


README = """# SAO-50 attribution full analysis bundle

This bundle contains the complete downstream-analysis inputs and outputs for the
50-seller, 200-query Stable Audio Open attribution study. It combines the
causal full-vs-leave-one-seller-out ground truth with D-TRAK, EKFAC, Random,
audio-fingerprint, and CLAP attribution estimates.

## What “raw data” means here

The archive includes the numerical artifacts needed to redo downstream
statistics without rerunning training or attribution:

- the raw 51-model x 200-query loss tensor and the derived 50-seller x
  200-query causal ground truth (`a_star.csv`);
- raw 1,117-training-example x 200-query score tensors for D-TRAK, EKFAC,
  Random, Fingerprint, and CLAP;
- D-TRAK projected train/query features, EKFAC layer-shard outputs, CLAP
  embeddings, and raw fingerprint features;
- train/query axis manifests, candidate-pool and seller manifests, prompts,
  model/subset provenance, and dataset configurations;
- all 200 generated query MP3 files and their prompt manifest;
- seller x query aggregate tables (`a_hat`), seller summaries (`I_hat`),
  confidence intervals, marginal-welfare tables, and figures;
- the analysis and scoring scripts used to produce the included artifacts.

The archive deliberately excludes model checkpoints, the full 20k-track source
corpus, duplicated source training audio, scheduler logs, and GPU telemetry.
The 1,117 selected training recordings are identified exactly by the selection
and train-axis manifests. Those exclusions are not needed for downstream
statistical analysis and would increase the archive by many gigabytes.

## Main directory layout

```text
ground_truth/       Raw full/LOO losses and causal a_star values
selection/          Seller definitions, sampled tracks, and inclusion lists
queries/            200 generated MP3 queries plus prompts and metadata
methods/
  dtrak/raw/        Per-example scores, manifests, and projected features
  dtrak/derived/    a_hat, I_hat, marginal welfare, and figures
  ekfac/raw/        Merged scores plus all three layer shards and manifests
  ekfac/derived/    a_hat, I_hat, marginal welfare, and figures
  baselines/        Random, Fingerprint, and CLAP raw and derived artifacts
model_provenance/   Non-checkpoint metadata for the full and 50 LOO models
configs/            Experiment and dataset configuration
scripts/            Relevant scoring, aggregation, statistics, and plotting code
MANIFEST.csv        SHA-256, byte size, and archive path for every payload file
```

## Core shapes and orientation

- Queries: 200
- Sellers: 50
- Candidate training examples after exclusions: 1,117
- `ground_truth/losses_model_x_query.pt`: `[51, 200]`
- `ground_truth/a_star.csv`: 10,000 rows = 50 sellers x 200 queries
- `methods/*/raw/score*.pt` and `scores_train_x_query.pt`:
  `[1,117, 200]` (training example x query)
- `scores_query_x_train.memmap`: `[200, 1,117]`; dtype and shape are recorded
  in the adjacent `.meta.json`
- each `a_hat*.csv`: 10,000 rows = 50 sellers x 200 queries
- each `I_hat*.csv`: 50 rows = one summary per seller

Use the train/query axis manifests rather than relying on machine-local absolute
paths embedded in older metadata. The archive paths and relative IDs are
portable; some historical absolute paths record the filesystem used at runtime.

## Earlier bundles

The older `sao_ip_dtrak_ekfac_analysis_bundle.zip` already contained raw D-TRAK
and EKFAC matrices, manifests, seller-level tables, and plots for the smaller
pilot. The separate `baseline_results_random_clap_fingerprint.zip` contained raw
baseline scores and CLAP embeddings, but mixed outputs from two experiments.
This archive applies the same useful idea to the completed SAO-50 experiment
and keeps all methods and provenance in one self-contained, non-mixed package.
"""


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def files_under(path: Path, *, exclude_names: set[str] | None = None) -> list[Path]:
    exclude_names = exclude_names or set()
    return sorted(
        item
        for item in require(path).rglob("*")
        if item.is_file()
        and item.name not in exclude_names
        and "__pycache__" not in item.parts
        and item.suffix != ".pyc"
    )


def csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.reader(handle)) - 1


def main() -> int:
    expected_rows = {
        OUTPUTS / "a_star.csv": 10_000,
        OUTPUTS / "a_hat_dtrak_v100_chunked.csv": 10_000,
        OUTPUTS / "I_hat_dtrak_v100_chunked.csv": 50,
        OUTPUTS / "a_hat_ekfac_v100_layer_sharded.csv": 10_000,
        OUTPUTS / "I_hat_ekfac_v100_layer_sharded.csv": 50,
    }
    for method in ("random", "fingerprint", "clap"):
        expected_rows[OUTPUTS / "baselines" / method / f"a_hat_{method}.csv"] = 10_000
        expected_rows[OUTPUTS / "baselines" / method / f"I_hat_{method}.csv"] = 50
    for path, expected in expected_rows.items():
        actual = csv_rows(require(path))
        if actual != expected:
            raise RuntimeError(f"unexpected row count for {path}: {actual} != {expected}")

    query_audio = sorted((OUTPUTS / "generated_queries").glob("query_*.mp3"))
    if len(query_audio) != 200:
        raise RuntimeError(f"expected 200 query MP3s, found {len(query_audio)}")

    entries: dict[str, Path] = {}

    def add(source: Path, archive_path: str) -> None:
        source = require(source)
        if archive_path in entries:
            raise RuntimeError(f"duplicate archive path: {archive_path}")
        entries[archive_path] = source

    def add_tree(source: Path, archive_prefix: str, *, exclude_names: set[str] | None = None) -> None:
        for item in files_under(source, exclude_names=exclude_names):
            add(item, f"{archive_prefix}/{item.relative_to(source).as_posix()}")

    for name in (
        "a_star.csv",
        "losses_model_x_query.pt",
        "losses_model_x_query.pt.meta.json",
        "losses_model_x_query.pt.model_ids.txt",
        "losses_model_x_query.pt.query_ids.txt",
        "prev_sao_full_loss_by_epoch.csv",
    ):
        add(OUTPUTS / name, f"ground_truth/{name}")

    add_tree(OUTPUTS / "selection", "selection")
    add_tree(OUTPUTS / "generated_queries", "queries/generated")
    add(OUTPUTS / "prompts.json", "queries/prompts.json")

    excluded_runtime = {"nvidia_smi.csv", "run.out", "run.err"}
    add_tree(
        OUTPUTS / "dtrak_attribution_v100_chunked",
        "methods/dtrak/raw",
        exclude_names=excluded_runtime,
    )
    for name in (
        "a_hat_dtrak_v100_chunked.csv",
        "I_hat_dtrak_v100_chunked.csv",
        "marginal_welfare_numbers_dtrak_v100_chunked.csv",
    ):
        add(OUTPUTS / name, f"methods/dtrak/derived/{name}")
    add_tree(
        OUTPUTS / "figures_dtrak_v100_chunked",
        "methods/dtrak/derived/figures",
    )

    add_tree(
        OUTPUTS / "ekfac_attribution_v100_layer_sharded",
        "methods/ekfac/raw",
        exclude_names=excluded_runtime,
    )
    for name in (
        "a_hat_ekfac_v100_layer_sharded.csv",
        "I_hat_ekfac_v100_layer_sharded.csv",
        "marginal_welfare_numbers_ekfac_v100_layer_sharded.csv",
    ):
        add(OUTPUTS / name, f"methods/ekfac/derived/{name}")
    add_tree(
        OUTPUTS / "figures_ekfac_v100_layer_sharded",
        "methods/ekfac/derived/figures",
    )

    add_tree(OUTPUTS / "baselines", "methods/baselines")

    models_root = OUTPUTS / "models"
    for model_index in range(51):
        model_dir = require(models_root / str(model_index))
        for item in sorted(model_dir.iterdir()):
            if item.is_file() and item.suffix not in {".ckpt", ".safetensors"}:
                add(item, f"model_provenance/{model_index}/{item.name}")

    add_tree(ROOT / "sao_ip_100" / "configs", "configs/datasets")
    add(ROOT / "sao_ip_100" / "scripts" / "pilot.yaml", "configs/pilot.yaml")
    add(ROOT / "model_config_freeze_vae.json", "configs/model_config_freeze_vae.json")

    script_paths = (
        ROOT / "sao_ip_pilot" / "scripts" / "estimate_I.py",
        ROOT / "sao_ip_pilot" / "scripts" / "plot_results.py",
        ROOT / "sao_ip_pilot" / "scripts" / "compute_a_star_from_losses.py",
        ROOT / "sao_ip_pilot" / "scripts" / "aggregate_ekfac_to_sellers.py",
        ROOT / "sao_ip_pilot" / "scripts" / "run_baseline_analyze.py",
        ROOT / "sao_ip_pilot" / "scripts" / "score_baselines.py",
        ROOT / "sao_ip_pilot" / "scripts" / "groundtruth_ekfac_attribution.py",
        ROOT / "scripts" / "groundtruth_dtrak_attribution.py",
        ROOT / "scripts" / "merge_ekfac_layer_shards.py",
    )
    for path in script_paths:
        add(path, f"scripts/{path.name}")
    add_tree(ROOT / "sao_ip_100" / "scripts", "scripts/slurm", exclude_names={"package_full_analysis_bundle.py"})

    manifest_buffer = io.StringIO()
    writer = csv.writer(manifest_buffer)
    writer.writerow(("archive_path", "bytes", "sha256"))
    hashes: dict[str, str] = {}
    for archive_path, source in sorted(entries.items()):
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[archive_path] = digest.hexdigest()
        writer.writerow((archive_path, source.stat().st_size, hashes[archive_path]))

    with zipfile.ZipFile(
        BUNDLE_PATH,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        archive.writestr(f"{BUNDLE_NAME}/README.md", README)
        archive.writestr(f"{BUNDLE_NAME}/MANIFEST.csv", manifest_buffer.getvalue())
        for archive_path, source in sorted(entries.items()):
            archive.write(source, f"{BUNDLE_NAME}/{archive_path}")

    with zipfile.ZipFile(BUNDLE_PATH) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure: {bad}")

    bundle_digest = hashlib.sha256()
    with BUNDLE_PATH.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            bundle_digest.update(chunk)
    summary = {
        "bundle": str(BUNDLE_PATH),
        "payload_files": len(entries),
        "query_mp3s": len(query_audio),
        "zip_bytes": BUNDLE_PATH.stat().st_size,
        "sha256": bundle_digest.hexdigest(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
