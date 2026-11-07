#!/bin/bash
#SBATCH --job-name=sao50_dtrak
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --output=sao_ip_100/logs/dtrak_score_%j.out
#SBATCH --error=sao_ip_100/logs/dtrak_score_%j.err

# D-TRAK scoring + seller aggregation only (no a* needed; can run parallel with gt).
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-sao_ip_100/outputs/dtrak_attribution}"
A_HAT_PATH="${A_HAT_PATH:-sao_ip_100/outputs/a_hat_dtrak.csv}"
mkdir -p sao_ip_100/logs "$OUTPUT_DIR" "$(dirname "$A_HAT_PATH")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export PYTHONPATH="${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python3 -u scripts/groundtruth_dtrak_attribution.py \
  --model-config model_config_freeze_vae.json \
  --train-dataset-config sao_ip_100/configs/train_dataset.json \
  --query-dataset-config sao_ip_100/configs/query_dataset.json \
  --models-root sao_ip_100/outputs/models \
  --full-model-dir sao_ip_100/outputs/models/0 \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "${BATCH_SIZE:-2}" \
  --num-workers 4 \
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
  --query-count 200

python3 -u sao_ip_pilot/scripts/aggregate_ekfac_to_sellers.py \
  --score-path "$OUTPUT_DIR/scores_train_x_query.pt" \
  --train-axis-manifest "$OUTPUT_DIR/train_axis_manifest.csv" \
  --query-axis-manifest "$OUTPUT_DIR/query_axis_manifest.csv" \
  --category-manifest sao_ip_100/outputs/selection/category_manifest.csv \
  --output-path "$A_HAT_PATH" \
  --score-column a_hat_dtrak

echo "[dtrak-score] done -> $A_HAT_PATH"
