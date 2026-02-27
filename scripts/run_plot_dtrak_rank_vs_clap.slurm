#!/bin/bash
#SBATCH --job-name=dtrak-rank-clap
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dtrak_rank_clap_%j.out
#SBATCH --error=logs/dtrak_rank_clap_%j.err
#SBATCH --time=24:00:00

set -euo pipefail

cd /home/xiruij/stable-audio-tools
mkdir -p logs
export PYTHONPATH="/home/xiruij/stable-audio-tools:${PYTHONPATH:-}"

ATTRIBUTION_DIR="${ATTRIBUTION_DIR:-/home/xiruij/stable-audio-tools/outputs/dtrak_attribution_20260225_124956}"
OUTPUT_DIR="${OUTPUT_DIR:-${ATTRIBUTION_DIR}/rank_vs_clap}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-influence_rank_vs_clap}"
CLAP_DEVICE="${CLAP_DEVICE:-cuda}"
CLAP_BATCH_SIZE="${CLAP_BATCH_SIZE:-4}"
AUDIO_LOAD_BATCH_SIZE="${AUDIO_LOAD_BATCH_SIZE:-2}"
MAX_RANK="${MAX_RANK:-5000}"

python3 -u scripts/plot_dtrak_rank_vs_clap.py \
  --attribution-dir "${ATTRIBUTION_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --clap-device "${CLAP_DEVICE}" \
  --clap-batch-size "${CLAP_BATCH_SIZE}" \
  --audio-load-batch-size "${AUDIO_LOAD_BATCH_SIZE}" \
  --max-rank "${MAX_RANK}"
