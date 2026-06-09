#!/bin/bash
#SBATCH --job-name=anticip_logra_grid
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=5
#SBATCH --time=12:00:00
#SBATCH --output=anticipation_ip_pilot/logs/logra_grid_%j.out
#SBATCH --error=anticipation_ip_pilot/logs/logra_grid_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p anticipation_ip_pilot/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Serial grid search over LoGra knobs. Primary metric: mean_spearman (LDS).
# Per-run outputs:
#   anticipation_ip_pilot/outputs/grid_search/logra/<tag>/
#     config.json
#     attribution/score_LoGra.pt
#     a_hat_logra.csv
#     metrics/query_level_corr_logra.csv
#     metrics/query_level_corr_logra_summary.json
#     metrics/I_hat_logra.csv
#     metrics/figures/            (optional)
# Aggregate:
#   grid_summary.csv
#   grid_manifest.json

python anticipation_ip_pilot/scripts/run_logra_grid_search.py \
  --grid-root "${GRID_ROOT:-anticipation_ip_pilot/outputs/grid_search/logra}" \
  --train-file "${TRAIN_FILE:-anticipation_ip_pilot/data/finetune/train_v2.txt}" \
  --queries-file "${QUERIES_FILE:-anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt}" \
  --model-path "${MODEL_PATH:-anticipation_ip_pilot/outputs/models/full-selected/full_model}" \
  --include-indices-file "${INCLUDE_INDICES_FILE:-anticipation_ip_pilot/outputs/selection/seller_indices.txt}" \
  --a-star-path "${A_STAR_PATH:-anticipation_ip_pilot/outputs/selected/a_star.csv}" \
  --seller-manifest "${SELLER_MANIFEST:-anticipation_ip_pilot/outputs/selection/seller_manifest.csv}" \
  --proj-dims "${PROJ_DIMS:-1024,4096}" \
  --dampings "${DAMPINGS:-0.001,0.1,1,10,100}" \
  --batch-sizes "${BATCH_SIZES:-4}" \
  --seed "${SEED:-42}" \
  ${SKIP_PLOTS:+--skip-plots}
