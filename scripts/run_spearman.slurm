#!/bin/bash
#SBATCH --job-name=sa_spearman
#SBATCH -p RM-shared
#SBATCH -A mth240006p
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/spearman_%j.out
#SBATCH --error=logs/spearman_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODELS_ROOT="${MODELS_ROOT:-outputs/groundtruth_models}"
SCORE_PATH="${SCORE_PATH:-outputs/groundtruth_models/dtrak_attribution/scores_train_x_query.pt}"
GROUNDTRUTH_PATH="${GROUNDTRUTH_PATH:-outputs/groundtruth_models/gt_gen.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/groundtruth_models/spearman}"
SUBSET_INDEX_FILENAME="${SUBSET_INDEX_FILENAME:-train_index.csv}"
RANDOM_SEED="${RANDOM_SEED:-42}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

python3 -u scripts/spearman.py \
  --models-root "$MODELS_ROOT" \
  --score-path "$SCORE_PATH" \
  --groundtruth-path "$GROUNDTRUTH_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --subset-index-filename "$SUBSET_INDEX_FILENAME" \
  --random-seed "$RANDOM_SEED" \
  $EXTRA_ARGS
