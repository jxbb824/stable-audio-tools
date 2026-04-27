#!/bin/bash
#SBATCH --job-name=plot_ekfac_rank_vs_clap
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/plot_ekfac_rank_vs_clap_%j.out
#SBATCH --error=logs/plot_ekfac_rank_vs_clap_%j.err
#SBATCH --time=5:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

ATTRIBUTION_DIR="${ATTRIBUTION_DIR:-outputs/groundtruth_models/ekfac_attribution}"
NEGATED_ATTRIBUTION_DIR="${NEGATED_ATTRIBUTION_DIR:-${ATTRIBUTION_DIR}_negated_for_clap}"
OUTPUT_DIR="${OUTPUT_DIR:-${NEGATED_ATTRIBUTION_DIR}/rank_vs_clap}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-ekfac_rank_vs_clap}"
CLAP_DEVICE="${CLAP_DEVICE:-cuda}"
CLAP_BATCH_SIZE="${CLAP_BATCH_SIZE:-8}"
AUDIO_LOAD_BATCH_SIZE="${AUDIO_LOAD_BATCH_SIZE:-4}"
MAX_RANK="${MAX_RANK:-1000}"
FORCE_REBUILD_NEGATED="${FORCE_REBUILD_NEGATED:-0}"
CACHE_PATH="${CACHE_PATH:-}"
FORCE_RECOMPUTE_CLAP="${FORCE_RECOMPUTE_CLAP:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

export ATTRIBUTION_DIR
export NEGATED_ATTRIBUTION_DIR
export FORCE_REBUILD_NEGATED

python3 - <<'PY'
import json
import os
import shutil
from pathlib import Path

import numpy as np

src = Path(os.environ["ATTRIBUTION_DIR"]).resolve()
dst = Path(os.environ["NEGATED_ATTRIBUTION_DIR"]).resolve()
force = os.environ.get("FORCE_REBUILD_NEGATED", "0") == "1"

required = [
    "scores_query_x_train.memmap",
    "scores_query_x_train.memmap.meta.json",
    "attribution_metadata.json",
    "train_axis_manifest.csv",
    "query_axis_manifest.csv",
]
for name in required:
    path = src / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required source file: {path}")

dst.mkdir(parents=True, exist_ok=True)
meta_path = src / "scores_query_x_train.memmap.meta.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
shape = tuple(meta["shape"])

dst_score_path = dst / "scores_query_x_train.memmap"
needs_rebuild = force or not dst_score_path.exists()
if not needs_rebuild:
    dst_meta = dst / "scores_query_x_train.memmap.meta.json"
    if not dst_meta.exists():
        needs_rebuild = True
    else:
        try:
            cached_meta = json.loads(dst_meta.read_text(encoding="utf-8"))
            needs_rebuild = tuple(cached_meta["shape"]) != shape
        except Exception:
            needs_rebuild = True

for name in [
    "scores_query_x_train.memmap.meta.json",
    "attribution_metadata.json",
    "train_axis_manifest.csv",
    "query_axis_manifest.csv",
]:
    shutil.copy2(src / name, dst / name)

if needs_rebuild:
    src_mm = np.memmap(src / "scores_query_x_train.memmap", dtype=np.float32, mode="r", shape=shape)
    dst_mm = np.memmap(dst_score_path, dtype=np.float32, mode="w+", shape=shape)
    block_rows = 16
    for start in range(0, shape[0], block_rows):
        end = min(start + block_rows, shape[0])
        dst_mm[start:end] = -src_mm[start:end]
    dst_mm.flush()
    print(f"[negate] wrote negated score memmap to {dst_score_path}", flush=True)
else:
    print(f"[negate] reusing existing negated score memmap at {dst_score_path}", flush=True)
PY

CMD=(
  python3 -u scripts/plot_dtrak_rank_vs_clap.py
  --attribution-dir "${NEGATED_ATTRIBUTION_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --output-prefix "${OUTPUT_PREFIX}"
  --clap-device "${CLAP_DEVICE}"
  --clap-batch-size "${CLAP_BATCH_SIZE}"
  --audio-load-batch-size "${AUDIO_LOAD_BATCH_SIZE}"
  --max-rank "${MAX_RANK}"
)

if [[ -n "${CACHE_PATH}" ]]; then
  CMD+=(--cache-path "${CACHE_PATH}")
fi

if [[ "${FORCE_RECOMPUTE_CLAP}" == "1" ]]; then
  CMD+=(--force-recompute-clap)
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${EXTRA_ARGS})
  CMD+=("${EXTRA_ARR[@]}")
fi

"${CMD[@]}"
