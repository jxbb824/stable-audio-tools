#!/bin/bash
#SBATCH --job-name=sao50_dtv3
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:3
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --output=sao_ip_100/logs/dtrak_score_3v100_chunked_%j.out
#SBATCH --error=sao_ip_100/logs/dtrak_score_3v100_chunked_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-sao_ip_100/outputs/dtrak_attribution_v100_chunked}"
A_HAT_PATH="${A_HAT_PATH:-sao_ip_100/outputs/a_hat_dtrak_v100_chunked.csv}"
mkdir -p sao_ip_100/logs "$OUTPUT_DIR" "$(dirname "$A_HAT_PATH")"

source /jet/home/xjiang6/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export PYTHONPATH="${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,nounits \
  --loop=10 > "$OUTPUT_DIR/nvidia_smi.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

python3 -u scripts/groundtruth_dtrak_attribution.py \
  --model-config model_config_freeze_vae.json \
  --train-dataset-config sao_ip_100/configs/train_dataset.json \
  --query-dataset-config sao_ip_100/configs/query_dataset.json \
  --models-root sao_ip_100/outputs/models \
  --full-model-dir sao_ip_100/outputs/models/0 \
  --output-dir "$OUTPUT_DIR" \
  --batch-size 2 \
  --num-workers 2 \
  --device cuda:0 \
  --proj-dim 2048 \
  --used-dim 2048 \
  --K 10 \
  --t-strategy uniform \
  --num-train-steps 1000 \
  --f loss \
  --cfg-dropout-prob 0.0 \
  --lambda-reg 0.1 \
  --seed 42 \
  --e-seed 0 \
  --query-count 200 \
  --projection-mode chunked \
  --projection-chunk-numel 33554432 \
  --projection-kernel-batch-size 8 \
  --num-shards 3

python3 -u sao_ip_pilot/scripts/aggregate_ekfac_to_sellers.py \
  --score-path "$OUTPUT_DIR/scores_train_x_query.pt" \
  --train-axis-manifest "$OUTPUT_DIR/train_axis_manifest.csv" \
  --query-axis-manifest "$OUTPUT_DIR/query_axis_manifest.csv" \
  --category-manifest sao_ip_100/outputs/selection/category_manifest.csv \
  --output-path "$A_HAT_PATH" \
  --score-column a_hat_dtrak

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT

echo "[dtrak-score-3v100-chunked] done -> $A_HAT_PATH"
