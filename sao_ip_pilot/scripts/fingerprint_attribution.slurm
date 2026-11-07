#!/bin/bash
#SBATCH --job-name=sao_ip_fp
#SBATCH -p RM-shared
#SBATCH -A mth260005p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=02:00:00
#SBATCH --chdir=/ocean/projects/mth240006p/xjiang6/stable-audio-tools
#SBATCH --output=sao_ip_pilot/logs/fingerprint_%j.out
#SBATCH --error=sao_ip_pilot/logs/fingerprint_%j.err

set -euo pipefail

cd /ocean/projects/mth240006p/xjiang6/stable-audio-tools
mkdir -p sao_ip_pilot/logs sao_ip_pilot/outputs/baselines/fingerprint

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PATH="${HOME}/local/bin:${PATH}"
export FPCALC="${FPCALC:-${HOME}/local/chromaprint/fpcalc}"
export PYTHONPATH="$PWD/sao_ip_pilot:${PYTHONPATH:-}"

SCORE_PATH="${SCORE_PATH:-sao_ip_pilot/outputs/baselines/fingerprint/score_fingerprint.pt}"

python sao_ip_pilot/scripts/score_fingerprint.py \
  --output-path "${SCORE_PATH}" \
  --length "${FP_LENGTH:-30}" \
  --workers "${SLURM_CPUS_PER_TASK:-8}"

python sao_ip_pilot/scripts/run_baseline_analyze.py \
  --method fingerprint_similarity \
  --score-path "${SCORE_PATH}"
