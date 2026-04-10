#!/bin/bash
#SBATCH --job-name=acestep-wavespeed-clap
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=l40s-48:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

: "${WAVESPEED_API_KEY:?Please export WAVESPEED_API_KEY before submitting the job.}"

AUDIO_DIR="${AUDIO_DIR:-dataset/small-5000/audio/00}"
LIMIT="${LIMIT:-100}"
MUSIC_WORKERS="${MUSIC_WORKERS:-4}"
DURATION="${DURATION:-60}"
POLL_INTERVAL="${POLL_INTERVAL:-1.0}"
CLAP_DEVICE="${CLAP_DEVICE:-cuda}"
CLAP_BATCH_SIZE="${CLAP_BATCH_SIZE:-4}"
AUDIO_LOAD_BATCH_SIZE="${AUDIO_LOAD_BATCH_SIZE:-4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs}"
HF_HOME="${HF_HOME:-/ocean/projects/mth240006p/xjiang6/huggingface}"
CAPTIONER_MODEL_ID="${CAPTIONER_MODEL_ID:-ACE-Step/acestep-captioner}"
WAVESPEED_MODEL="${WAVESPEED_MODEL:-wavespeed-ai/ace-step-1.5}"

python3 -u scripts/acestep_captioner_wavespeed_clap_eval.py \
  --audio-dir "${AUDIO_DIR}" \
  --output-root "${OUTPUT_ROOT}" \
  --limit "${LIMIT}" \
  --hf-home "${HF_HOME}" \
  --captioner-model-id "${CAPTIONER_MODEL_ID}" \
  --wavespeed-model "${WAVESPEED_MODEL}" \
  --music-workers "${MUSIC_WORKERS}" \
  --duration "${DURATION}" \
  --poll-interval "${POLL_INTERVAL}" \
  --clap-device "${CLAP_DEVICE}" \
  --clap-batch-size "${CLAP_BATCH_SIZE}" \
  --audio-load-batch-size "${AUDIO_LOAD_BATCH_SIZE}"
