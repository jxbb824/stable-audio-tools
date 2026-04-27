#!/bin/bash
#SBATCH --job-name=sao_ip_full
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=12
#SBATCH --time=40:00:00
#SBATCH --output=logs/sao_ip_full_%j.out
#SBATCH --error=logs/sao_ip_full_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

MODEL_CONFIG="${MODEL_CONFIG:-model_config_freeze_vae.json}"
DATASET_CONFIG="${DATASET_CONFIG:-sao_ip_pilot/configs/train_dataset.json}"
MODEL_CKPT="${MODEL_CKPT:-dataset/pretrained_model/model.ckpt}"
VAE_CKPT="${VAE_CKPT:-dataset/pretrained_model/vae_model.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-sao_ip_pilot/outputs/models}"
INCLUDE_PATHS="${INCLUDE_PATHS:-sao_ip_pilot/outputs/selection/selected_paths_full.txt}"
SEED="${SEED:-42}"
MAX_EPOCHS="${MAX_EPOCHS:-600}"

python3 -u sao_ip_pilot/scripts/finetune_sao.py \
  --model-config "$MODEL_CONFIG" \
  --dataset-config "$DATASET_CONFIG" \
  --pretrained-ckpt-path "$MODEL_CKPT" \
  --pretransform-ckpt-path "$VAE_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --tag 0 \
  --include-paths-file "$INCLUDE_PATHS" \
  --seed "$SEED" \
  --batch-size "${BATCH_SIZE:-8}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --precision "${PRECISION:-16-mixed}" \
  --accum-batches "${ACCUM_BATCHES:-1}" \
  --max-epochs "$MAX_EPOCHS" \
  --logger "${LOGGER:-none}"
