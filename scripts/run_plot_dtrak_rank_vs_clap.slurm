#!/bin/bash
#SBATCH --job-name=plot_dtrak_rank_vs_clap
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/plot_dtrak_rank_vs_clap_%j.out
#SBATCH --error=logs/plot_dtrak_rank_vs_clap_%j.err
#SBATCH --time=48:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

ATTRIBUTION_DIR="${ATTRIBUTION_DIR:-outputs/groundtruth_models/dtrak_attribution}"
OUTPUT_DIR="${OUTPUT_DIR:-${ATTRIBUTION_DIR}/rank_vs_clap}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-influence_rank_vs_clap}"
CLAP_DEVICE="${CLAP_DEVICE:-cuda}"
CLAP_BATCH_SIZE="${CLAP_BATCH_SIZE:-8}"
AUDIO_LOAD_BATCH_SIZE="${AUDIO_LOAD_BATCH_SIZE:-4}"
MAX_RANK="${MAX_RANK:-1000}"

python3 -u scripts/plot_dtrak_rank_vs_clap.py \
  --attribution-dir "${ATTRIBUTION_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --clap-device "${CLAP_DEVICE}" \
  --clap-batch-size "${CLAP_BATCH_SIZE}" \
  --audio-load-batch-size "${AUDIO_LOAD_BATCH_SIZE}" \
  --max-rank "${MAX_RANK}"
