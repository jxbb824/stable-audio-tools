#!/bin/bash
#SBATCH --job-name=sao50_fp
#SBATCH -p RM-shared
#SBATCH -A mth260005p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15G
#SBATCH --time=02:00:00
#SBATCH --output=sao_ip_100/logs/baseline_fingerprint_%j.out
#SBATCH --error=sao_ip_100/logs/baseline_fingerprint_%j.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
mkdir -p sao_ip_100/logs sao_ip_100/outputs/baselines/fingerprint
source /jet/home/xjiang6/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo
export PYTHONPATH="${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"
export FPCALC="${FPCALC:-/jet/home/xjiang6/local/chromaprint/fpcalc}"

TRAIN_AXIS=sao_ip_100/outputs/dtrak_attribution_v100_chunked/train_axis_manifest.csv
QUERY_AXIS=sao_ip_100/outputs/dtrak_attribution_v100_chunked/query_axis_manifest.csv
OUTPUT_DIR=sao_ip_100/outputs/baselines/fingerprint

python3 -u sao_ip_pilot/scripts/score_fingerprint.py \
  --train-axis-manifest "$TRAIN_AXIS" --query-axis-manifest "$QUERY_AXIS" \
  --output-path "$OUTPUT_DIR/score_fingerprint.pt" \
  --length 30 --workers "${SLURM_CPUS_PER_TASK:-8}"

python3 -u sao_ip_pilot/scripts/run_baseline_analyze.py \
  --method fingerprint_similarity --score-path "$OUTPUT_DIR/score_fingerprint.pt" \
  --train-axis-manifest "$TRAIN_AXIS" --query-axis-manifest "$QUERY_AXIS" \
  --category-manifest sao_ip_100/outputs/selection/category_manifest.csv \
  --a-star-path sao_ip_100/outputs/a_star.csv \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --output-dir "$OUTPUT_DIR"

python3 -u scripts/validate_sao_baseline_outputs.py \
  --method fingerprint --score-path "$OUTPUT_DIR/score_fingerprint.pt" \
  --train-axis-manifest "$TRAIN_AXIS" --query-axis-manifest "$QUERY_AXIS" \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --a-hat-path "$OUTPUT_DIR/a_hat_fingerprint.csv" \
  --i-hat-path "$OUTPUT_DIR/I_hat_fingerprint.csv"
