#!/bin/bash
#SBATCH --job-name=sao50_dtck_probe
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=sao_ip_100/logs/dtrak_chunked_v100_probe_%j.out
#SBATCH --error=sao_ip_100/logs/dtrak_chunked_v100_probe_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
cd "$REPO_ROOT"

PROBE_DIR="sao_ip_100/outputs/dtrak_chunked_v100_probe/${SLURM_JOB_ID}"
mkdir -p sao_ip_100/logs "$PROBE_DIR"

source /jet/home/xjiang6/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,nounits \
  --loop=2 > "$PROBE_DIR/nvidia_smi.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

python3 -u scripts/validate_dtrak_chunked_projection.py \
  --device cuda:0 \
  --output-path "$PROBE_DIR/projection_validation.json"

python3 -u scripts/dtrak_extract_features.py \
  --model-config model_config_freeze_vae.json \
  --dataset-config sao_ip_100/configs/query_dataset.json \
  --pretrained-ckpt-path sao_ip_100/outputs/models/0/model.ckpt \
  --batch-size 1 \
  --num-workers 1 \
  --device cuda:0 \
  --proj-dim 2048 \
  --used-dim 2048 \
  --K 1 \
  --t-strategy uniform \
  --num-train-steps 1000 \
  --f loss \
  --cfg-dropout-prob 0.0 \
  --seed 42 \
  --e-seed 0 \
  --disable-random-crop \
  --projection-mode chunked \
  --projection-chunk-numel 33554432 \
  --projection-kernel-batch-size 8 \
  --feature-path "$PROBE_DIR/full_model_chunked.memmap" \
  --max-examples 1

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT

python3 - "$PROBE_DIR" <<'PY'
import csv
import json
import sys
from pathlib import Path

probe_dir = Path(sys.argv[1])
meta = json.loads((probe_dir / "full_model_chunked.memmap.meta.json").read_text())
rows = list(csv.DictReader((probe_dir / "nvidia_smi.csv").open()))
observed_mib = max(
    float(value)
    for row in rows
    for key, value in row.items()
    if key.strip().startswith("memory.used")
)
summary = {
    "status": "pass",
    "valid_rows": meta["valid_rows"],
    "grad_dim": meta["grad_dim"],
    "projection_mode": meta["projection_mode"],
    "num_projection_chunks": len(meta["projection_chunk_sizes"]),
    "torch_peak_cuda_memory_gib": meta["peak_cuda_memory_gib"],
    "nvidia_smi_peak_memory_mib": observed_mib,
}
if summary["valid_rows"] != 1:
    raise RuntimeError(f"Expected one feature row, got {summary['valid_rows']}")
(probe_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "[probe] PASS -> $PROBE_DIR/probe_summary.json"
