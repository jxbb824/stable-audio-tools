#!/bin/bash
#SBATCH --job-name=sa_full_finetune
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/finetune_full_%j.out
#SBATCH --error=logs/finetune_full_%j.err
#SBATCH --time=40:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODEL_CONFIG="${MODEL_CONFIG:-model_config_freeze_vae.json}"
DATASET_CONFIG="${DATASET_CONFIG:-stable_audio_tools/configs/dataset_configs/local_training_custom.json}"
MODEL_CKPT="${MODEL_CKPT:-dataset/pretrained_model/model.ckpt}"
VAE_CKPT="${VAE_CKPT:-dataset/pretrained_model/vae_model.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/groundtruth_models}"
RUN_NAME="${RUN_NAME:-stable_audio_groundtruth_full}"
BASE_SEED="${BASE_SEED:-0}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-1000}"
MAX_EPOCHS="${MAX_EPOCHS:-600}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

python3 finetune_full.py \
  --model-config "$MODEL_CONFIG" \
  --dataset-config "$DATASET_CONFIG" \
  --pretrained-ckpt-path "$MODEL_CKPT" \
  --pretransform-ckpt-path "$VAE_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --seed "$BASE_SEED" \
  --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
  --batch-size 8 \
  --num-workers 8 \
  --precision "16-mixed" \
  --max-epochs "$MAX_EPOCHS" \
  --name "$RUN_NAME" \
  --logger "none"
