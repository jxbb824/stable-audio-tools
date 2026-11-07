#!/bin/bash
# Executed inside a 1-GPU interactive-style allocation (srun/salloc, same as `interact`).
set -euo pipefail
cd /ocean/projects/mth260005p/xjiang6/stable-audio-tools
source /jet/home/xjiang6/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "host=$(hostname) start=$(date -Is) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=name,memory.total --format=csv
python3 -u estimator_diagnosis/gpu_validation.py 2700
echo "end=$(date -Is)"
