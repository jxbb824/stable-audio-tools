#!/bin/bash
#SBATCH --job-name=dac_dedup
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dac_dedup_%j.out
#SBATCH --error=logs/dac_dedup_%j.err
#SBATCH --time=48:00:00

set -euo pipefail

cd /home/xiruij/stable-audio-tools
mkdir -p logs

DATASET_CONFIG="/home/xiruij/stable-audio-tools/stable_audio_tools/configs/dataset_configs/local_training_custom.json"
OUTPUT_DIR="/home/xiruij/stable-audio-tools/outputs/dac_dedup"

python3 scripts/dac_hist_dedup.py \
  --dataset-config "$DATASET_CONFIG" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --similarity-device cuda \
  --similarity-threshold 0.95 \
  --histogram-mode per_codebook \
  --max-audio-seconds 120
