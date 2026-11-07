#!/bin/bash
#SBATCH --job-name=anticip_ip_clap
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=5
#SBATCH --time=03:00:00
#SBATCH --chdir=/ocean/projects/mth240006p/xjiang6/stable-audio-tools
#SBATCH --output=anticipation_ip_pilot/logs/clap_%j.out
#SBATCH --error=anticipation_ip_pilot/logs/clap_%j.err

set -euo pipefail

cd /ocean/projects/mth240006p/xjiang6/stable-audio-tools
mkdir -p anticipation_ip_pilot/logs anticipation_ip_pilot/outputs/baselines/clap

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:${PYTHONPATH:-}"

AUDIO_ROOT="${AUDIO_ROOT:-anticipation_ip_pilot/outputs/baselines/clap/audio}"
SCORE_PATH="${SCORE_PATH:-anticipation_ip_pilot/outputs/baselines/clap/score_clap.pt}"

python anticipation_ip_pilot/scripts/score_clap.py \
  --audio-root "${AUDIO_ROOT}" \
  --output-path "${SCORE_PATH}" \
  --batch-size "${BATCH_SIZE:-16}"

python anticipation_ip_pilot/scripts/run_baseline_analyze.py \
  --method clap_similarity \
  --score-path "${SCORE_PATH}"
