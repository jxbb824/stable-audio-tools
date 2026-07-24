#!/bin/bash
#SBATCH --job-name=lds_logra_epoch_sweep
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --output=anticipation_lds_pilot/logs/logra_epoch_sweep_%j.out
#SBATCH --error=anticipation_lds_pilot/logs/logra_epoch_sweep_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"

SWEEP_ROOT="${SWEEP_ROOT:-anticipation_lds_pilot/outputs_logra_epoch_sweep}"
EPOCH_LIST="${EPOCH_LIST:-0 1 3 5 10}"

PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-/jet/home/xjiang6/.cache/huggingface/hub/models--stanford-crfm--music-small-800k/snapshots/fa800530aa1126dd6b58b38f3bdb8fcec9c9d4f5}"
TRAIN_FILE="${TRAIN_FILE:-anticipation_ip_pilot/data/finetune/train_v2.txt}"
VALID_FILE="${VALID_FILE:-anticipation_ip_pilot/data/finetune/test_v2.txt}"
QUERIES_FILE="${QUERIES_FILE:-anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt}"
POOL_INDICES_FILE="${POOL_INDICES_FILE:-anticipation_ip_pilot/outputs/selection/seller_indices.txt}"
GROUNDTRUTH_PATH="${GROUNDTRUTH_PATH:-anticipation_lds_pilot/outputs_e5/lds/gt_gen.pt}"
SUBSETS_ROOT="${SUBSETS_ROOT:-anticipation_lds_pilot/outputs/models/subsets}"

mkdir -p anticipation_lds_pilot/logs "${SWEEP_ROOT}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:$PWD/anticipation_lds_pilot:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[sweep] root=${SWEEP_ROOT}"
echo "[sweep] epochs=${EPOCH_LIST}"
echo "[sweep] LDS groundtruth=${GROUNDTRUTH_PATH}"
echo "[sweep] LDS subsets_root=${SUBSETS_ROOT}"

for EPOCH in ${EPOCH_LIST}; do
  RUN_DIR="${SWEEP_ROOT}/epoch_${EPOCH}"
  MODEL_DIR="${RUN_DIR}/models/full_logra"
  ATTR_DIR="${RUN_DIR}/attribution"
  LDS_DIR="${RUN_DIR}/lds"

  mkdir -p "${RUN_DIR}/models" "${ATTR_DIR}" "${LDS_DIR}"

  echo "[sweep][epoch=${EPOCH}] start"

  if [[ "${EPOCH}" == "0" ]]; then
    if [[ ! -e "${MODEL_DIR}" ]]; then
      ln -s "${PRETRAINED_MODEL_PATH}" "${MODEL_DIR}"
    fi
    MODEL_PATH="${MODEL_DIR}"
    echo "[sweep][epoch=0] skipping fine-tune; using pretrained model via ${MODEL_PATH}"
  else
    mkdir -p "${MODEL_DIR}"
    echo "[sweep][epoch=${EPOCH}] training full LoGra model -> ${MODEL_DIR}"
    python anticipation_lds_pilot/scripts/finetune_subset.py \
      --pretrained_model_path "${PRETRAINED_MODEL_PATH}" \
      --train_file "${TRAIN_FILE}" \
      --valid_file "${VALID_FILE}" \
      --output_dir "${MODEL_DIR}" \
      --include_indices_file "${POOL_INDICES_FILE}" \
      --subset_ratio 1.0 \
      --seed "${SEED:-42}" \
      --epochs "${EPOCH}" \
      --batch_size "${TRAIN_BATCH_SIZE:-8}" \
      --learning_rate "${LEARNING_RATE:-2e-5}"
    MODEL_PATH="${MODEL_DIR}"
  fi

  echo "[sweep][epoch=${EPOCH}] scoring LoGra -> ${ATTR_DIR}/score_LoGra.pt"
  python anticipation_lds_pilot/scripts/score_logra.py \
    --train_file "${TRAIN_FILE}" \
    --valid_file "${QUERIES_FILE}" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${ATTR_DIR}" \
    --include_indices_file "${POOL_INDICES_FILE}" \
    --batch_size "${SCORE_BATCH_SIZE:-8}" \
    --proj_dim "${PROJ_DIM:-4096}" \
    --damping "${DAMPING:-0.01}" \
    --seed "${SEED:-42}" \
    --valid_is_generated \
    --output_filename score_LoGra.pt

  echo "[sweep][epoch=${EPOCH}] evaluating LDS with fixed 5epoch groundtruth -> ${LDS_DIR}"
  python anticipation_lds_pilot/scripts/run_lds_eval.py \
    --methods logra \
    --attribution-dir "${ATTR_DIR}" \
    --groundtruth-path "${GROUNDTRUTH_PATH}" \
    --subsets-root "${SUBSETS_ROOT}" \
    --pool-indices-file "${POOL_INDICES_FILE}" \
    --output-dir "${LDS_DIR}" \
    --num-subsets "${NUM_SUBSETS:-30}"

  echo "[sweep][epoch=${EPOCH}] done"
done

echo "[sweep] complete"
