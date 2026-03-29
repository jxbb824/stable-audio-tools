#!/bin/bash
#SBATCH --job-name=sa_gt_finetune
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --output=logs/finetune_subset_%A_%a.out
#SBATCH --error=logs/finetune_subset_%A_%a.err
#SBATCH --time=40:00:00
#SBATCH --array=0-29

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODEL_CONFIG="${MODEL_CONFIG:-model_config_freeze_vae.json}"
DATASET_CONFIG="${DATASET_CONFIG:-stable_audio_tools/configs/dataset_configs/local_training_custom.json}"
MODEL_CKPT="${MODEL_CKPT:-dataset/pretrained_model/model.ckpt}"
VAE_CKPT="${VAE_CKPT:-dataset/pretrained_model/vae_model.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/groundtruth_models}"
RUN_NAME="${RUN_NAME:-stable_audio_groundtruth_subset}"
LOGGER="${LOGGER:-wandb}"
BASE_SEED="${BASE_SEED:-0}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-1000}"
TRAIN_SUBSET_SIZE="${TRAIN_SUBSET_SIZE:-500}"
MAX_EPOCHS="${MAX_EPOCHS:-600}"

RUN_ID="${SLURM_ARRAY_TASK_ID}"
SEED="$((BASE_SEED + RUN_ID))"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo
export WANDB_API_KEY=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
mkdir -p "$OUTPUT_DIR"

python3 finetune_subset.py \
  --model-config "$MODEL_CONFIG" \
  --dataset-config "$DATASET_CONFIG" \
  --pretrained-ckpt-path "$MODEL_CKPT" \
  --pretransform-ckpt-path "$VAE_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --run-id "$RUN_ID" \
  --seed "$SEED" \
  --candidate-pool-size "$CANDIDATE_POOL_SIZE" \
  --train-subset-size "$TRAIN_SUBSET_SIZE" \
  --batch-size 8 \
  --num-workers 8 \
  --precision "16-mixed" \
  --max-epochs "$MAX_EPOCHS" \
  --name "$RUN_NAME" \
  --logger "$LOGGER"
