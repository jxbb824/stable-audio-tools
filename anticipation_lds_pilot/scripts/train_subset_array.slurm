#!/bin/bash
#SBATCH --job-name=lds_subset
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --array=0-29
#SBATCH --gpus=v100-32:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=5
#SBATCH --time=01:30:00
#SBATCH --output=anticipation_lds_pilot/logs/subset_%A_%a.out
#SBATCH --error=anticipation_lds_pilot/logs/subset_%A_%a.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p anticipation_lds_pilot/logs anticipation_lds_pilot/outputs/models/subsets

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:${PYTHONPATH:-}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

python anticipation_lds_pilot/scripts/finetune_subset.py \
  --pretrained_model_path "${PRETRAINED_MODEL_PATH:-/jet/home/xjiang6/.cache/huggingface/hub/models--stanford-crfm--music-small-800k/snapshots/fa800530aa1126dd6b58b38f3bdb8fcec9c9d4f5}" \
  --train_file "${TRAIN_FILE:-anticipation_ip_pilot/data/finetune/train_v2.txt}" \
  --valid_file "${VALID_FILE:-anticipation_ip_pilot/data/finetune/test_v2.txt}" \
  --output_dir "${SUBSETS_ROOT:-anticipation_lds_pilot/outputs/models/subsets}/${TASK_ID}" \
  --include_indices_file "${POOL_INDICES_FILE:-anticipation_ip_pilot/outputs/selection/seller_indices.txt}" \
  --subset_ratio "${SUBSET_RATIO:-0.5}" \
  --seed "${TASK_ID}" \
  --epochs "${EPOCHS:-5}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --learning_rate "${LEARNING_RATE:-2e-5}"
