#!/bin/bash
#SBATCH --job-name=sa_finetune
#SBATCH --partition=gpuA100x8
#SBATCH --account=bgek-delta-gpu
#SBATCH --gres=gpu:nvidia_a100:2
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=30:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODEL_CONFIG="/u/xjiang6/stable-audio-tools/model_config_freeze_vae.json"
DATASET_CONFIG="/u/xjiang6/stable-audio-tools/stable_audio_tools/configs/dataset_configs/local_training_custom.json"
MODEL_CKPT="/projects/bffl/xirui/stable-audio-tools/dataset/pretrained_model/model.ckpt"
VAE_CKPT="/projects/bffl/xirui/stable-audio-tools/dataset/pretrained_model/vae_model.ckpt"
SAVE_DIR="/u/xjiang6/stable-audio-tools/outputs"

export WANDB_API_KEY=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
python3 train.py \
  --model-config "$MODEL_CONFIG" \
  --dataset-config "$DATASET_CONFIG" \
  --pretrained-ckpt-path "$MODEL_CKPT" \
  --pretransform-ckpt-path "$VAE_CKPT" \
  --batch-size 16 \
  --num-workers 16 \
  --precision "16-mixed" \
  --name "stable_audio_open_finetune" \
  --save-dir "$SAVE_DIR" \
  # --logger "none" \
