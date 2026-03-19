#!/bin/bash
#SBATCH --job-name=gemini-wavespeed-clap
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

PROMPT_WORKERS="${PROMPT_WORKERS:-8}"
MUSIC_WORKERS="${MUSIC_WORKERS:-4}"
LIMIT="${LIMIT:-100}"
CLAP_DEVICE="${CLAP_DEVICE:-cuda}"
CLAP_BATCH_SIZE="${CLAP_BATCH_SIZE:-4}"
AUDIO_LOAD_BATCH_SIZE="${AUDIO_LOAD_BATCH_SIZE:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"

python3 -u scripts/gemini_wavespeed_clap_eval.py \
  --output-root "${OUTPUT_ROOT}" \
  --limit "${LIMIT}" \
  --prompt-workers "${PROMPT_WORKERS}" \
  --music-workers "${MUSIC_WORKERS}" \
  --clap-device "${CLAP_DEVICE}" \
  --clap-batch-size "${CLAP_BATCH_SIZE}" \
  --audio-load-batch-size "${AUDIO_LOAD_BATCH_SIZE}"
