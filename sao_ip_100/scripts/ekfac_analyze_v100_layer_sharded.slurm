#!/bin/bash
#SBATCH --job-name=sao50_ekv3_an
#SBATCH -p RM-small
#SBATCH -A mth260005p
#SBATCH --mem=3800M
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=sao_ip_100/logs/ekfac_analyze_v100_layer_sharded_%j.out
#SBATCH --error=sao_ip_100/logs/ekfac_analyze_v100_layer_sharded_%j.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
mkdir -p sao_ip_100/logs sao_ip_100/outputs/figures_ekfac_v100_layer_sharded
source /jet/home/xjiang6/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

python3 -u sao_ip_pilot/scripts/estimate_I.py \
  --method ekfac_influence \
  --a-star-path sao_ip_100/outputs/a_star.csv \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --a-hat-path sao_ip_100/outputs/a_hat_ekfac_v100_layer_sharded.csv \
  --output-path sao_ip_100/outputs/I_hat_ekfac_v100_layer_sharded.csv

python3 -u sao_ip_pilot/scripts/plot_results.py \
  --i-hat-path sao_ip_100/outputs/I_hat_ekfac_v100_layer_sharded.csv \
  --a-hat-path sao_ip_100/outputs/a_hat_ekfac_v100_layer_sharded.csv \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --output-dir sao_ip_100/outputs/figures_ekfac_v100_layer_sharded \
  --marginal-welfare-csv sao_ip_100/outputs/marginal_welfare_numbers_ekfac_v100_layer_sharded.csv \
  --method-label EKFAC

echo "[ekfac-analyze-v100-layer-sharded] done"
