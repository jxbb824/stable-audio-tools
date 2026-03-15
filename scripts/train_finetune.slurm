#!/bin/bash
#SBATCH --job-name=sa_finetune
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:2
#SBATCH --mem=256G
#SBATCH --cpus-per-task=24
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=40:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODEL_CONFIG="model_config_freeze_vae.json"
DATASET_CONFIG="stable_audio_tools/configs/dataset_configs/local_training_custom.json"
MODEL_CKPT="dataset/pretrained_model/model.ckpt"
VAE_CKPT="dataset/pretrained_model/vae_model.ckpt"
SAVE_DIR="outputs"


source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo


export WANDB_API_KEY=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
python3 train.py \
  --model-config "$MODEL_CONFIG" \
  --dataset-config "$DATASET_CONFIG" \
  --pretrained-ckpt-path "$MODEL_CKPT" \
  --pretransform-ckpt-path "$VAE_CKPT" \
  --batch-size 4 \
  --num-workers 4 \
  --precision "16-mixed" \
  --name "stable_audio_open_finetune" \
  --save-dir "$SAVE_DIR" \
  # --logger "none" \
