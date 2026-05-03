#!/bin/bash
#SBATCH --job-name=anticip_ip_full
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=anticipation_ip_pilot/logs/full_%j.out
#SBATCH --error=anticipation_ip_pilot/logs/full_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p anticipation_ip_pilot/logs anticipation_ip_pilot/outputs/models/full

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python anticipation_ip_pilot/scripts/finetune_trak_checkpoints.py \
  --pretrained_model_path "${PRETRAINED_MODEL_PATH:-stanford-crfm/music-small-800k}" \
  --train_file "${TRAIN_FILE:-anticipation_ip_pilot/data/finetune/train_v2.txt}" \
  --valid_file "${VALID_FILE:-anticipation_ip_pilot/data/finetune/test_v2.txt}" \
  --output_dir "${OUTPUT_DIR:-anticipation_ip_pilot/outputs/models/full}" \
  --epochs "${EPOCHS:-20}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --learning_rate "${LEARNING_RATE:-2e-5}" \
  --seed "${SEED:-42}"
