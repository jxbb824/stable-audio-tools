#!/bin/bash
#SBATCH --job-name=anticip_ip_analyze
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=anticipation_ip_pilot/logs/analyze_%j.out
#SBATCH --error=anticipation_ip_pilot/logs/analyze_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p anticipation_ip_pilot/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:${PYTHONPATH:-}"

if [[ -f anticipation_ip_pilot/outputs/a_hat_logra.csv ]]; then
  python anticipation_ip_pilot/scripts/estimate_I.py \
    --method logra_influence \
    --output-path anticipation_ip_pilot/outputs/I_hat_logra.csv
  python anticipation_ip_pilot/scripts/query_level_corr.py \
    --method logra_influence \
    --output-csv anticipation_ip_pilot/outputs/query_level_corr_logra.csv \
    --summary-json anticipation_ip_pilot/outputs/query_level_corr_logra_summary.json
  python anticipation_ip_pilot/scripts/plot_results.py \
    --i-hat-path anticipation_ip_pilot/outputs/I_hat_logra.csv \
    --a-hat-path anticipation_ip_pilot/outputs/a_hat_logra.csv \
    --output-dir anticipation_ip_pilot/outputs/figures_logra \
    --marginal-welfare-csv anticipation_ip_pilot/outputs/marginal_welfare_numbers_logra.csv \
    --method-label LoGra
fi

if [[ -f anticipation_ip_pilot/outputs/a_hat_trak.csv ]]; then
  python anticipation_ip_pilot/scripts/estimate_I.py \
    --method trak_influence \
    --output-path anticipation_ip_pilot/outputs/I_hat_trak.csv
  python anticipation_ip_pilot/scripts/query_level_corr.py \
    --method trak_influence \
    --output-csv anticipation_ip_pilot/outputs/query_level_corr_trak.csv \
    --summary-json anticipation_ip_pilot/outputs/query_level_corr_trak_summary.json
  python anticipation_ip_pilot/scripts/plot_results.py \
    --i-hat-path anticipation_ip_pilot/outputs/I_hat_trak.csv \
    --a-hat-path anticipation_ip_pilot/outputs/a_hat_trak.csv \
    --output-dir anticipation_ip_pilot/outputs/figures_trak \
    --marginal-welfare-csv anticipation_ip_pilot/outputs/marginal_welfare_numbers_trak.csv \
    --method-label TRAK
fi
