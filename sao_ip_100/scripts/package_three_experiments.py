#!/usr/bin/env python3
"""Create one complete downstream-analysis bundle for three attribution studies."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BUNDLE_NAME = "music_attribution_three_experiments_full_bundle_20260919"
BUNDLE_PATH = ROOT / f"{BUNDLE_NAME}.zip"
SCALEUP_ZIP = ROOT / "sao50_attribution_full_analysis_bundle_20260919.zip"

README = """# Music attribution: three complete downstream-analysis datasets

This archive combines the three completed attribution studies in this project.
Each experiment has its own directory and keeps raw numerical artifacts separate
from derived statistics and figures.

## Experiments

### `experiments/original_sao_pilot/`

The original Stable Audio Open pilot: 22 sellers, 200 generated audio queries,
and 4,400 seller-query causal ground-truth rows. Included methods are D-TRAK,
EKFAC, Random, audio fingerprinting, and CLAP. The directory contains raw
per-training-example score tensors/memmaps, projected features and embeddings,
full-vs-LOO losses, selection/axis manifests, generated query MP3s, derived
tables, figures, scripts, configurations, and non-checkpoint model provenance.

### `experiments/anticipation_pilot/`

The Anticipation symbolic-music study: 100 sellers, 500 symbolic queries, and
50,000 seller-query causal ground-truth rows. Included methods are LoGra, TRAK,
Random, audio fingerprinting, and CLAP. The selected primary experiment,
original/identity attribution runs, all later LoGra/TRAK grid-search artifacts,
raw full/LOO losses, raw scores/embeddings/fingerprints, seller and train
manifests, symbolic query data, derived tables, figures, scripts, source modules,
and non-checkpoint model provenance are included.

### `experiments/sao50_scaleup/`

The completed SAO scale-up: 50 sellers, 200 generated audio queries, 1,117
attribution-axis training examples after exclusions, and 10,000 seller-query
causal ground-truth rows. Included methods are D-TRAK, layer-sharded EKFAC,
Random, audio fingerprinting, and CLAP. This is the full verified bundle created
on 2026-09-19, including raw losses/scores/features/embeddings, all three EKFAC
shards, manifests, 200 query MP3s, model provenance, scripts, tables, and figures.

## Included versus excluded data

“Raw data” here means the artifacts needed to redo downstream attribution and
statistical analysis without rerunning model training or attribution: raw loss
and score tensors, embeddings/features, manifests, prompts/query data, and
method metadata. Generated audio queries are included for both SAO studies; the
Anticipation study includes its native symbolic query sequences.

The archive excludes fine-tuned model checkpoints, full source training
corpora, duplicated source training audio, scheduler logs, GPU telemetry, and
the 4.1 GB temporary WAV cache created only while computing Anticipation CLAP
features. The corresponding embeddings and exact manifests are included.

`MANIFEST.csv` records byte size and SHA-256 for every payload file. Historical
absolute paths inside metadata describe the original cluster filesystem; use
the included relative IDs and axis manifests for portable analysis.
"""


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def include_file(path: Path) -> bool:
    return (
        path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix != ".pyc"
    )


def is_checkpoint(path: Path) -> bool:
    return (
        path.suffix in {".ckpt", ".safetensors"}
        or path.name in {
            "pytorch_model.bin",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "scaler.pt",
        }
        or any(part.startswith("checkpoint-") for part in path.parts)
    )


def main() -> int:
    require(SCALEUP_ZIP)
    entries: dict[str, tuple[str, object, str | None]] = {}

    def add_fs(source: Path, archive_path: str) -> None:
        source = require(source)
        if archive_path in entries:
            raise RuntimeError(f"duplicate archive path: {archive_path}")
        entries[archive_path] = ("fs", source, None)

    def add_tree(
        source: Path,
        archive_prefix: str,
        *,
        predicate=None,
    ) -> None:
        source = require(source)
        for item in sorted(source.rglob("*")):
            if not include_file(item):
                continue
            if predicate is not None and not predicate(item):
                continue
            add_fs(item, f"{archive_prefix}/{item.relative_to(source).as_posix()}")

    # Original SAO pilot.
    sao = ROOT / "sao_ip_pilot"
    sao_out = sao / "outputs"
    sao_prefix = "experiments/original_sao_pilot"
    add_fs(sao / "README.md", f"{sao_prefix}/ORIGINAL_README.md")
    add_tree(sao / "configs", f"{sao_prefix}/configs")
    add_tree(sao / "scripts", f"{sao_prefix}/scripts")
    for item in sorted(sao_out.iterdir()):
        if item.is_file():
            add_fs(item, f"{sao_prefix}/outputs/tables/{item.name}")
    for dirname in (
        "selection",
        "generated_queries",
        "dtrak_attribution",
        "ekfac_attribution",
        "baselines",
        "figures",
        "figures_dtrak",
    ):
        add_tree(sao_out / dirname, f"{sao_prefix}/outputs/{dirname}")
    add_tree(
        sao_out / "models",
        f"{sao_prefix}/model_provenance",
        predicate=lambda path: not is_checkpoint(path),
    )

    # Anticipation symbolic-music pilot, including later parameter sweeps.
    ant = ROOT / "anticipation_ip_pilot"
    ant_out = ant / "outputs"
    ant_prefix = "experiments/anticipation_pilot"
    add_fs(ant / "README.md", f"{ant_prefix}/ORIGINAL_README.md")
    add_tree(ant / "scripts", f"{ant_prefix}/scripts")
    add_tree(ant / "anticipation", f"{ant_prefix}/source/anticipation")
    add_fs(
        ant / "data" / "metadata" / "tt_data_hashed.csv",
        f"{ant_prefix}/data_metadata/tt_data_hashed.csv",
    )
    for item in sorted(ant_out.iterdir()):
        if item.is_file():
            add_fs(item, f"{ant_prefix}/outputs/tables/{item.name}")
    for dirname in (
        "selection",
        "queries",
        "selected",
        "attribution",
        "figures_logra",
        "figures_logra_identity",
        "grid_search",
    ):
        add_tree(ant_out / dirname, f"{ant_prefix}/outputs/{dirname}")
    add_tree(
        ant_out / "baselines",
        f"{ant_prefix}/outputs/baselines",
        predicate=lambda path: "audio" not in path.relative_to(ant_out / "baselines").parts,
    )
    for model_subdir in ("full-selected", "loo-selected"):
        add_tree(
            ant_out / "models" / model_subdir,
            f"{ant_prefix}/model_provenance/{model_subdir}",
            predicate=lambda path: not is_checkpoint(path),
        )

    # Import the already validated SAO-50 full bundle, stripping its outer root.
    scaleup_archive = zipfile.ZipFile(SCALEUP_ZIP)
    scaleup_names = [name for name in scaleup_archive.namelist() if not name.endswith("/")]
    scaleup_roots = {name.split("/", 1)[0] for name in scaleup_names}
    if len(scaleup_roots) != 1:
        raise RuntimeError(f"unexpected SAO-50 ZIP roots: {scaleup_roots}")
    scaleup_root = next(iter(scaleup_roots))
    for source_name in scaleup_names:
        relative = source_name[len(scaleup_root) + 1 :]
        if relative == "MANIFEST.csv":
            continue
        archive_path = f"experiments/sao50_scaleup/{relative}"
        if archive_path in entries:
            raise RuntimeError(f"duplicate archive path: {archive_path}")
        entries[archive_path] = ("zip", scaleup_archive, source_name)

    # Essential completeness checks before spending time writing the archive.
    required_suffixes = (
        "experiments/original_sao_pilot/outputs/dtrak_attribution/scores_train_x_query.pt",
        "experiments/original_sao_pilot/outputs/ekfac_attribution/scores_train_x_query.pt",
        "experiments/original_sao_pilot/outputs/baselines/clap/score_clap.pt",
        "experiments/original_sao_pilot/outputs/tables/a_star.csv",
        "experiments/anticipation_pilot/outputs/attribution/logra-selected/score_LoGra.pt",
        "experiments/anticipation_pilot/outputs/attribution/trak-selected/score_TRAK.pt",
        "experiments/anticipation_pilot/outputs/baselines/clap/score_clap.pt",
        "experiments/anticipation_pilot/outputs/selected/a_star.csv",
        "experiments/sao50_scaleup/methods/dtrak/raw/scores_train_x_query.pt",
        "experiments/sao50_scaleup/methods/ekfac/raw/scores_train_x_query.pt",
        "experiments/sao50_scaleup/methods/baselines/clap/score_clap.pt",
        "experiments/sao50_scaleup/ground_truth/a_star.csv",
    )
    missing = [name for name in required_suffixes if name not in entries]
    if missing:
        raise RuntimeError(f"missing required payloads: {missing}")

    compression_stored = {".mp3", ".wav", ".png", ".pdf", ".pt", ".npz", ".bin"}
    manifest_rows: list[tuple[str, int, str]] = []

    def write_stream(archive: zipfile.ZipFile, archive_path: str, chunks, size: int) -> None:
        digest = hashlib.sha256()
        info = zipfile.ZipInfo(f"{BUNDLE_NAME}/{archive_path}")
        info.compress_type = (
            zipfile.ZIP_STORED
            if Path(archive_path).suffix.lower() in compression_stored
            else zipfile.ZIP_DEFLATED
        )
        with archive.open(info, "w", force_zip64=True) as output:
            for chunk in chunks:
                digest.update(chunk)
                output.write(chunk)
        manifest_rows.append((archive_path, size, digest.hexdigest()))

    with zipfile.ZipFile(BUNDLE_PATH, mode="w", allowZip64=True, compresslevel=6) as archive:
        readme_data = README.encode("utf-8")
        write_stream(archive, "README.md", (readme_data,), len(readme_data))
        for archive_path, (kind, source, source_name) in sorted(entries.items()):
            if kind == "fs":
                source_path = source
                assert isinstance(source_path, Path)
                size = source_path.stat().st_size
                with source_path.open("rb") as handle:
                    write_stream(
                        archive,
                        archive_path,
                        iter(lambda: handle.read(1024 * 1024), b""),
                        size,
                    )
            else:
                source_zip = source
                assert isinstance(source_zip, zipfile.ZipFile)
                assert source_name is not None
                size = source_zip.getinfo(source_name).file_size
                with source_zip.open(source_name) as handle:
                    write_stream(
                        archive,
                        archive_path,
                        iter(lambda: handle.read(1024 * 1024), b""),
                        size,
                    )

        manifest_buffer = io.StringIO()
        writer = csv.writer(manifest_buffer)
        writer.writerow(("archive_path", "bytes", "sha256"))
        writer.writerows(sorted(manifest_rows))
        archive.writestr(
            f"{BUNDLE_NAME}/MANIFEST.csv",
            manifest_buffer.getvalue(),
            compress_type=zipfile.ZIP_DEFLATED,
        )

    scaleup_archive.close()
    with zipfile.ZipFile(BUNDLE_PATH) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP integrity failure: {bad}")

    bundle_digest = hashlib.sha256()
    with BUNDLE_PATH.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            bundle_digest.update(chunk)

    counts = {name: 0 for name in ("original_sao_pilot", "anticipation_pilot", "sao50_scaleup")}
    for archive_path in entries:
        for experiment in counts:
            if archive_path.startswith(f"experiments/{experiment}/"):
                counts[experiment] += 1
    print(
        json.dumps(
            {
                "bundle": str(BUNDLE_PATH),
                "payload_files": len(manifest_rows),
                "files_by_experiment": counts,
                "zip_bytes": BUNDLE_PATH.stat().st_size,
                "sha256": bundle_digest.hexdigest(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
