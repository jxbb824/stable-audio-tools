#!/bin/bash
#SBATCH --job-name=sao50_random
#SBATCH -p RM-small
#SBATCH -A mth260005p
#SBATCH --mem=3800M
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00
#SBATCH --output=sao_ip_100/logs/baseline_random_%j.out
#SBATCH --error=sao_ip_100/logs/baseline_random_%j.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
mkdir -p sao_ip_100/logs sao_ip_100/outputs/baselines/random
source /jet/home/xjiang6/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo
export PYTHONPATH="${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"

TRAIN_AXIS=sao_ip_100/outputs/dtrak_attribution_v100_chunked/train_axis_manifest.csv
QUERY_AXIS=sao_ip_100/outputs/dtrak_attribution_v100_chunked/query_axis_manifest.csv
OUTPUT_DIR=sao_ip_100/outputs/baselines/random

python3 -u sao_ip_pilot/scripts/score_baselines.py \
  --mode random --seed 42 \
  --reference-score-path sao_ip_100/outputs/dtrak_attribution_v100_chunked/scores_train_x_query.pt \
  --output-path "$OUTPUT_DIR/score_random.pt"

python3 -u sao_ip_pilot/scripts/run_baseline_analyze.py \
  --method random_influence --score-path "$OUTPUT_DIR/score_random.pt" \
  --train-axis-manifest "$TRAIN_AXIS" --query-axis-manifest "$QUERY_AXIS" \
  --category-manifest sao_ip_100/outputs/selection/category_manifest.csv \
  --a-star-path sao_ip_100/outputs/a_star.csv \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --output-dir "$OUTPUT_DIR"

python3 -u scripts/validate_sao_baseline_outputs.py \
  --method random --score-path "$OUTPUT_DIR/score_random.pt" \
  --train-axis-manifest "$TRAIN_AXIS" --query-axis-manifest "$QUERY_AXIS" \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --a-hat-path "$OUTPUT_DIR/a_hat_random.csv" --i-hat-path "$OUTPUT_DIR/I_hat_random.csv"
