#!/bin/bash
#SBATCH --job-name=sao50_ekv_probe
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=sao_ip_100/logs/ekfac_layer_shard_v100_probe_%j.out
#SBATCH --error=sao_ip_100/logs/ekfac_layer_shard_v100_probe_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
cd "$REPO_ROOT"

PROBE_DIR="sao_ip_100/outputs/ekfac_layer_shard_v100_probe/${SLURM_JOB_ID}"
mkdir -p sao_ip_100/logs "$PROBE_DIR"

source /jet/home/xjiang6/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo
export DIFFUSION_INFLUENCE_ROOT="${REPO_ROOT}/../diffusion-influence"
export PYTHONPATH="${DIFFUSION_INFLUENCE_ROOT}:${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,nounits --loop=2 > "$PROBE_DIR/nvidia_smi.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

run_range() {
  local name="$1"
  local start="$2"
  local end="$3"
  python3 -u sao_ip_pilot/scripts/groundtruth_ekfac_attribution.py \
    --model-config model_config_freeze_vae.json \
    --train-dataset-config sao_ip_100/configs/train_dataset.json \
    --query-dataset-config sao_ip_100/configs/query_dataset.json \
    --models-root sao_ip_100/outputs/models \
    --full-model-dir sao_ip_100/outputs/models/0 \
    --output-dir "$PROBE_DIR/$name" \
    --train-count 2 --query-count 2 --num-workers 0 --device cuda:0 \
    --measurement-f loss --measurement-cfg-dropout-prob 0.0 --damping 1e-8 \
    --kfac-num-samples-for-loss 1 --kfac-batch-size 1 \
    --num-samples-for-loss-scoring 1 --num-samples-for-measurement-scoring 1 \
    --num-loss-batch-aggregations 1 --num-measurement-batch-aggregations 1 \
    --query-batch-size 1 --train-batch-size 1 --query-gradient-compression-bits 4 \
    --linear-layer-start-index "$start" --linear-layer-end-index "$end" --seed 42
}

# A small exactness test: the three one-layer contributions must sum to the
# score from the same three layers evaluated together.
run_range reference_175_178 175 178
run_range shard_175_176 175 176
run_range shard_176_177 176 177
run_range shard_177_178 177 178
python3 -u scripts/merge_ekfac_layer_shards.py \
  --shard-dirs "$PROBE_DIR/shard_175_176" "$PROBE_DIR/shard_176_177" "$PROBE_DIR/shard_177_178" \
  --output-dir "$PROBE_DIR/merged_175_178" \
  --expected-linear-start 175 --expected-linear-end 178

# The largest intended production block is the actual 32-GiB fit test.
run_range fit_89_119 89 119

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT

python3 - "$PROBE_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1])
shape = (2, 2)
reference = np.memmap(root / "reference_175_178/scores_query_x_train.memmap", dtype=np.float32, mode="r", shape=shape)
merged = np.memmap(root / "merged_175_178/scores_query_x_train.memmap", dtype=np.float32, mode="r", shape=shape)
absolute = np.abs(np.asarray(reference) - np.asarray(merged))
scale = np.maximum(np.abs(np.asarray(reference)), 1e-12)
rows = list(csv.DictReader((root / "nvidia_smi.csv").open()))
peak_mib = max(
    float(value)
    for row in rows
    for key, value in row.items()
    if key.strip().startswith("memory.used")
)
summary = {
    "status": "pass",
    "reference": np.asarray(reference).tolist(),
    "merged": np.asarray(merged).tolist(),
    "max_abs_error": float(absolute.max()),
    "max_rel_error": float((absolute / scale).max()),
    "nvidia_smi_peak_memory_mib": peak_mib,
    "production_ranges": [[89, 119], [119, 149], [149, 178]],
}
if not np.allclose(reference, merged, rtol=2e-4, atol=2e-6):
    raise RuntimeError(f"Layer-shard additivity check failed: {summary}")
(root / "probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "[ekfac-layer-probe] PASS -> $PROBE_DIR/probe_summary.json"
