#!/bin/bash
#SBATCH --job-name=anticip_ip_logra
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=anticipation_ip_pilot/logs/logra_%j.out
#SBATCH --error=anticipation_ip_pilot/logs/logra_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p anticipation_ip_pilot/logs anticipation_ip_pilot/outputs/attribution/logra

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python anticipation_ip_pilot/scripts/score_logra.py \
  --train_file "${TRAIN_FILE:-anticipation_ip_pilot/data/finetune/train_v2.txt}" \
  --valid_file "${QUERIES_FILE:-anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt}" \
  --model_path "${MODEL_PATH:-anticipation_ip_pilot/outputs/models/full/full_model}" \
  --output_dir "${ATTR_DIR:-anticipation_ip_pilot/outputs/attribution/logra}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --proj_dim "${PROJ_DIM:-4096}" \
  --damping "${DAMPING:-0.01}" \
  --valid_is_generated \
  --output_filename score_LoGra.pt

python anticipation_ip_pilot/scripts/aggregate_scores_to_sellers.py \
  --score-path "${ATTR_DIR:-anticipation_ip_pilot/outputs/attribution/logra}/score_LoGra.pt" \
  --output-path anticipation_ip_pilot/outputs/a_hat_logra.csv \
  --score-column a_hat_logra

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
