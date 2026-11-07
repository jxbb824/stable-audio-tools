#!/bin/bash
#SBATCH --job-name=sao50_ekfac_an
#SBATCH -p RM-shared
#SBATCH -A mth260005p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=01:00:00
#SBATCH --output=sao_ip_100/logs/ekfac_analyze_%j.out
#SBATCH --error=sao_ip_100/logs/ekfac_analyze_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
cd "$REPO_ROOT"
mkdir -p sao_ip_100/logs sao_ip_100/outputs/figures_ekfac

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export PYTHONPATH="${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"

test -f sao_ip_100/outputs/a_star.csv
test -f sao_ip_100/outputs/a_hat_ekfac.csv

python3 -u sao_ip_pilot/scripts/estimate_I.py \
  --method ekfac_influence \
  --a-star-path sao_ip_100/outputs/a_star.csv \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --a-hat-path sao_ip_100/outputs/a_hat_ekfac.csv \
  --output-path sao_ip_100/outputs/I_hat_ekfac.csv

python3 -u sao_ip_pilot/scripts/plot_results.py \
  --i-hat-path sao_ip_100/outputs/I_hat_ekfac.csv \
  --a-hat-path sao_ip_100/outputs/a_hat_ekfac.csv \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --output-dir sao_ip_100/outputs/figures_ekfac \
  --marginal-welfare-csv sao_ip_100/outputs/marginal_welfare_numbers_ekfac.csv \
  --method-label EKFAC

echo "[ekfac-analyze] done -> sao_ip_100/outputs/I_hat_ekfac.csv"
