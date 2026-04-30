#!/bin/bash
#SBATCH --job-name=sao_ip_analyze
#SBATCH -A mth240006p
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=logs/sao_ip_analyze_%j.out
#SBATCH --error=logs/sao_ip_analyze_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

python3 -u sao_ip_pilot/scripts/aggregate_ekfac_to_sellers.py
python3 -u sao_ip_pilot/scripts/estimate_I.py
python3 -u sao_ip_pilot/scripts/plot_results.py
