#!/bin/bash
#SBATCH --job-name=anticip_ip_trak
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=7:00:00
#SBATCH --output=anticipation_ip_pilot/logs/trak_%j.out
#SBATCH --error=anticipation_ip_pilot/logs/trak_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p anticipation_ip_pilot/logs anticipation_ip_pilot/outputs/attribution/trak

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python anticipation_ip_pilot/scripts/score_trak.py \
  --train_file "${TRAIN_FILE:-anticipation_ip_pilot/data/finetune/train_v2.txt}" \
  --valid_file "${QUERIES_FILE:-anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt}" \
  --checkpoint_dir "${CHECKPOINT_DIR:-anticipation_ip_pilot/outputs/models/full}" \
  --output_dir "${ATTR_DIR:-anticipation_ip_pilot/outputs/attribution/trak}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --num_checkpoints "${NUM_CHECKPOINTS:-5}" \
  --proj_dim "${PROJ_DIM:-8192}" \
  --proj_type "${PROJ_TYPE:-random_mask}" \
  --proj_max_batch_size "${PROJ_MAX_BATCH_SIZE:-40}" \
  --regularization "${REGULARIZATION:-0.01}" \
  --num_prompts "${NUM_PROMPTS:-500}" \
  --seed "${SEED:-42}" \
  --valid_is_generated \
  --layer_name "${LAYER_NAME:-transformer.h.11.mlp.c_proj.weight}" \
  --output_filename score_TRAK.pt

python anticipation_ip_pilot/scripts/aggregate_scores_to_sellers.py \
  --score-path "${ATTR_DIR:-anticipation_ip_pilot/outputs/attribution/trak}/score_TRAK.pt" \
  --output-path anticipation_ip_pilot/outputs/a_hat_trak.csv \
  --score-column a_hat_trak

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
