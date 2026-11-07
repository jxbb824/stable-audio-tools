#!/bin/bash
#SBATCH --job-name=anticip_ip_render
#SBATCH -p RM-shared
#SBATCH -A mth260005p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=02:00:00
#SBATCH --chdir=/ocean/projects/mth240006p/xjiang6/stable-audio-tools
#SBATCH --output=anticipation_ip_pilot/logs/render_%j.out
#SBATCH --error=anticipation_ip_pilot/logs/render_%j.err

set -euo pipefail

cd /ocean/projects/mth240006p/xjiang6/stable-audio-tools
mkdir -p anticipation_ip_pilot/logs anticipation_ip_pilot/outputs/baselines/clap/audio

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:${PYTHONPATH:-}"

python anticipation_ip_pilot/scripts/render_midi_audio.py \
  --workers "${WORKERS:-8}" \
  --sample-rate "${SAMPLE_RATE:-22050}"
