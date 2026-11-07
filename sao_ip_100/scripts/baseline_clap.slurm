#!/bin/bash
#SBATCH --job-name=sao50_clap
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --output=sao_ip_100/logs/baseline_clap_%j.out
#SBATCH --error=sao_ip_100/logs/baseline_clap_%j.err

set -euo pipefail
cd "${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
mkdir -p sao_ip_100/logs sao_ip_100/outputs/baselines/clap
source /jet/home/xjiang6/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo
export PYTHONPATH="${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

TRAIN_AXIS=sao_ip_100/outputs/dtrak_attribution_v100_chunked/train_axis_manifest.csv
QUERY_AXIS=sao_ip_100/outputs/dtrak_attribution_v100_chunked/query_axis_manifest.csv
OUTPUT_DIR=sao_ip_100/outputs/baselines/clap

python3 -u sao_ip_pilot/scripts/score_baselines.py \
  --mode clap --seed 42 --batch-size 8 --device cuda \
  --train-axis-manifest "$TRAIN_AXIS" --query-axis-manifest "$QUERY_AXIS" \
  --output-path "$OUTPUT_DIR/score_clap.pt"

python3 -u sao_ip_pilot/scripts/run_baseline_analyze.py \
  --method clap_similarity --score-path "$OUTPUT_DIR/score_clap.pt" \
  --train-axis-manifest "$TRAIN_AXIS" --query-axis-manifest "$QUERY_AXIS" \
  --category-manifest sao_ip_100/outputs/selection/category_manifest.csv \
  --a-star-path sao_ip_100/outputs/a_star.csv \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --output-dir "$OUTPUT_DIR"

python3 -u scripts/validate_sao_baseline_outputs.py \
  --method clap --score-path "$OUTPUT_DIR/score_clap.pt" \
  --train-axis-manifest "$TRAIN_AXIS" --query-axis-manifest "$QUERY_AXIS" \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --a-hat-path "$OUTPUT_DIR/a_hat_clap.csv" --i-hat-path "$OUTPUT_DIR/I_hat_clap.csv"
