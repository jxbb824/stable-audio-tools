#!/bin/bash
#SBATCH --job-name=lds_trak
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=5
#SBATCH --time=08:00:00
#SBATCH --output=anticipation_lds_pilot/logs/trak_%j.out
#SBATCH --error=anticipation_lds_pilot/logs/trak_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p anticipation_lds_pilot/logs anticipation_lds_pilot/outputs/attribution

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:$PWD/anticipation_lds_pilot:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python anticipation_lds_pilot/scripts/score_trak.py \
  --train_file "${TRAIN_FILE:-anticipation_ip_pilot/data/finetune/train_v2.txt}" \
  --valid_file "${QUERIES_FILE:-anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt}" \
  --checkpoint_dir "${CHECKPOINT_DIR:-anticipation_lds_pilot/outputs/models/subsets}" \
  --model_path "${MODEL_PATH:-anticipation_lds_pilot/outputs/models/full_logra}" \
  --output_dir "${ATTR_DIR:-anticipation_lds_pilot/outputs/attribution}" \
  --include_indices_file "${POOL_INDICES_FILE:-anticipation_ip_pilot/outputs/selection/seller_indices.txt}" \
  --batch_size "${BATCH_SIZE:-2}" \
  --num_checkpoints "${NUM_CHECKPOINTS:-5}" \
  --proj_dim "${PROJ_DIM:-8192}" \
  --proj_type "${PROJ_TYPE:-random_mask}" \
  --proj_max_batch_size "${PROJ_MAX_BATCH_SIZE:-40}" \
  --regularization "${REGULARIZATION:-0.01}" \
  --num_prompts "${NUM_PROMPTS:-500}" \
  --seed "${SEED:-42}" \
  --valid_is_generated \
  --output_filename score_TRAK.pt
