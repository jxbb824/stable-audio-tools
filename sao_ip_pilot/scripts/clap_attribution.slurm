#!/bin/bash
#SBATCH --job-name=sao_ip_clap
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --chdir=/ocean/projects/mth240006p/xjiang6/stable-audio-tools
#SBATCH --output=sao_ip_pilot/logs/clap_%j.out
#SBATCH --error=sao_ip_pilot/logs/clap_%j.err

set -euo pipefail

cd /ocean/projects/mth240006p/xjiang6/stable-audio-tools
mkdir -p sao_ip_pilot/logs sao_ip_pilot/outputs/baselines/clap

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/sao_ip_pilot:${PYTHONPATH:-}"

python sao_ip_pilot/scripts/score_baselines.py \
  --mode clap \
  --output-path sao_ip_pilot/outputs/baselines/clap/score_clap.pt \
  --batch-size "${BATCH_SIZE:-8}"

python sao_ip_pilot/scripts/run_baseline_analyze.py \
  --method clap_similarity \
  --score-path sao_ip_pilot/outputs/baselines/clap/score_clap.pt
