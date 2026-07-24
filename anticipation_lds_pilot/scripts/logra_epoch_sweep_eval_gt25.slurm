#!/bin/bash
#SBATCH --job-name=lds_logra_epoch_sweep_gt25
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --output=anticipation_lds_pilot/logs/logra_epoch_sweep_gt25_%j.out
#SBATCH --error=anticipation_lds_pilot/logs/logra_epoch_sweep_gt25_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"

SWEEP_ROOT="${SWEEP_ROOT:-anticipation_lds_pilot/outputs_logra_epoch_sweep}"
EPOCH_LIST="${EPOCH_LIST:-0 1 3 5 10}"
GROUNDTRUTH_PATH="${GROUNDTRUTH_PATH:-anticipation_lds_pilot/outputs_e25/lds/gt_gen.pt}"
SUBSETS_ROOT="${SUBSETS_ROOT:-anticipation_lds_pilot/outputs_e25/models/subsets}"
POOL_INDICES_FILE="${POOL_INDICES_FILE:-anticipation_ip_pilot/outputs/selection/seller_indices.txt}"

mkdir -p anticipation_lds_pilot/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:$PWD/anticipation_lds_pilot:${PYTHONPATH:-}"

echo "[eval-gt25] root=${SWEEP_ROOT}"
echo "[eval-gt25] epochs=${EPOCH_LIST}"
echo "[eval-gt25] LDS groundtruth=${GROUNDTRUTH_PATH}"
echo "[eval-gt25] LDS subsets_root=${SUBSETS_ROOT}"

for EPOCH in ${EPOCH_LIST}; do
  ATTR_DIR="${SWEEP_ROOT}/epoch_${EPOCH}/attribution"
  LDS_DIR="${SWEEP_ROOT}/epoch_${EPOCH}/lds_gt25"

  if [[ ! -f "${ATTR_DIR}/score_LoGra.pt" ]]; then
    echo "[eval-gt25][epoch=${EPOCH}] missing ${ATTR_DIR}/score_LoGra.pt" >&2
    exit 1
  fi

  mkdir -p "${LDS_DIR}"
  echo "[eval-gt25][epoch=${EPOCH}] evaluating LDS -> ${LDS_DIR}"
  python anticipation_lds_pilot/scripts/run_lds_eval.py \
    --methods logra \
    --attribution-dir "${ATTR_DIR}" \
    --groundtruth-path "${GROUNDTRUTH_PATH}" \
    --subsets-root "${SUBSETS_ROOT}" \
    --pool-indices-file "${POOL_INDICES_FILE}" \
    --output-dir "${LDS_DIR}" \
    --num-subsets "${NUM_SUBSETS:-30}"
done

echo "[eval-gt25] complete"
