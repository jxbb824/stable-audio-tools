#!/bin/bash
#SBATCH --job-name=sao50_dtrak_an
#SBATCH -p RM-shared
#SBATCH -A mth260005p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=01:00:00
#SBATCH --output=sao_ip_100/logs/dtrak_analyze_%j.out
#SBATCH --error=sao_ip_100/logs/dtrak_analyze_%j.err

# Needs both a_star.csv (from gt) and a_hat_dtrak.csv (from dtrak_score).
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
cd "$REPO_ROOT"

A_HAT_PATH="${A_HAT_PATH:-sao_ip_100/outputs/a_hat_dtrak.csv}"
I_HAT_PATH="${I_HAT_PATH:-sao_ip_100/outputs/I_hat_dtrak.csv}"
FIGURE_DIR="${FIGURE_DIR:-sao_ip_100/outputs/figures_dtrak}"
WELFARE_PATH="${WELFARE_PATH:-sao_ip_100/outputs/marginal_welfare_numbers_dtrak.csv}"
mkdir -p sao_ip_100/logs "$FIGURE_DIR" "$(dirname "$I_HAT_PATH")" "$(dirname "$WELFARE_PATH")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export PYTHONPATH="${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"

test -f sao_ip_100/outputs/a_star.csv
test -f "$A_HAT_PATH"

python3 -u sao_ip_pilot/scripts/estimate_I.py \
  --method dtrak_influence \
  --a-star-path sao_ip_100/outputs/a_star.csv \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --a-hat-path "$A_HAT_PATH" \
  --output-path "$I_HAT_PATH"

python3 -u sao_ip_pilot/scripts/plot_results.py \
  --i-hat-path "$I_HAT_PATH" \
  --a-hat-path "$A_HAT_PATH" \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --output-dir "$FIGURE_DIR" \
  --marginal-welfare-csv "$WELFARE_PATH" \
  --method-label D-TRAK

echo "[dtrak-analyze] done -> $I_HAT_PATH"
