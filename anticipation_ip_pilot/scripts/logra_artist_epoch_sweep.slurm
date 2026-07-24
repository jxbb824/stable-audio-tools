#!/bin/bash
#SBATCH --job-name=artist_logra_epoch
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --output=anticipation_ip_pilot/logs/logra_artist_epoch_%j.out
#SBATCH --error=anticipation_ip_pilot/logs/logra_artist_epoch_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p anticipation_ip_pilot/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EPOCH_LIST="${EPOCH_LIST:-0 1 3 5 10}"
MODEL_ROOT="${MODEL_ROOT:-anticipation_lds_pilot/outputs_logra_epoch_sweep}"
SWEEP_ROOT="${SWEEP_ROOT:-anticipation_ip_pilot/outputs/grid_search/logra_artist_epoch_sweep_pd4096_d0p1}"

PROJ_DIM="${PROJ_DIM:-4096}"
DAMPING="${DAMPING:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-4}"

TRAIN_FILE="${TRAIN_FILE:-anticipation_ip_pilot/data/finetune/train_v2.txt}"
QUERIES_FILE="${QUERIES_FILE:-anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt}"
INCLUDE_INDICES_FILE="${INCLUDE_INDICES_FILE:-anticipation_ip_pilot/outputs/selection/seller_indices.txt}"
A_STAR_PATH="${A_STAR_PATH:-anticipation_ip_pilot/outputs/selected/a_star.csv}"
SELLER_MANIFEST="${SELLER_MANIFEST:-anticipation_ip_pilot/outputs/selection/seller_manifest.csv}"

mkdir -p "${SWEEP_ROOT}"
STATUS_PATH="${SWEEP_ROOT}/status.tsv"
SUMMARY_PATH="${SWEEP_ROOT}/epoch_summary.csv"

printf "epoch\tstatus\texit_code\tmodel_path\trun_root\n" > "${STATUS_PATH}"
printf "epoch,tag,batch_size,proj_dim,damping,mean_spearman_lds,median_spearman,mean_pearson,median_pearson,mean_I_hat,median_I_hat,mean_seller_spearman,run_dir\n" > "${SUMMARY_PATH}"

echo "[epoch-sweep] root=${SWEEP_ROOT}"
echo "[epoch-sweep] epochs=${EPOCH_LIST}"
echo "[epoch-sweep] proj_dim=${PROJ_DIM} damping=${DAMPING} batch_size=${BATCH_SIZE}"

for EPOCH in ${EPOCH_LIST}; do
  MODEL_PATH="${MODEL_ROOT}/epoch_${EPOCH}/models/full_logra"
  RUN_ROOT="${SWEEP_ROOT}/epoch_${EPOCH}"

  echo "[epoch-sweep][epoch=${EPOCH}] model=${MODEL_PATH}"
  if [[ ! -e "${MODEL_PATH}" ]]; then
    echo "[epoch-sweep][epoch=${EPOCH}] missing model path; skipping"
    printf "%s\tmissing_model\t127\t%s\t%s\n" "${EPOCH}" "${MODEL_PATH}" "${RUN_ROOT}" >> "${STATUS_PATH}"
    continue
  fi

  set +e
  python anticipation_ip_pilot/scripts/run_logra_grid_search.py \
    --grid-root "${RUN_ROOT}" \
    --train-file "${TRAIN_FILE}" \
    --queries-file "${QUERIES_FILE}" \
    --model-path "${MODEL_PATH}" \
    --include-indices-file "${INCLUDE_INDICES_FILE}" \
    --a-star-path "${A_STAR_PATH}" \
    --seller-manifest "${SELLER_MANIFEST}" \
    --proj-dims "${PROJ_DIM}" \
    --dampings "${DAMPING}" \
    --batch-sizes "${BATCH_SIZE}" \
    --seed "${SEED:-42}" \
    ${SKIP_PLOTS:+--skip-plots}
  EXIT_CODE=$?
  set -e

  if [[ "${EXIT_CODE}" -ne 0 ]]; then
    echo "[epoch-sweep][epoch=${EPOCH}] failed with exit_code=${EXIT_CODE}; continuing"
    printf "%s\tfailed\t%s\t%s\t%s\n" "${EPOCH}" "${EXIT_CODE}" "${MODEL_PATH}" "${RUN_ROOT}" >> "${STATUS_PATH}"
    continue
  fi

  printf "%s\tsucceeded\t0\t%s\t%s\n" "${EPOCH}" "${MODEL_PATH}" "${RUN_ROOT}" >> "${STATUS_PATH}"
  awk -v epoch="${EPOCH}" 'NR > 1 {print epoch "," $0}' "${RUN_ROOT}/grid_summary.csv" >> "${SUMMARY_PATH}"
  echo "[epoch-sweep][epoch=${EPOCH}] done"
done

echo "[epoch-sweep] wrote ${STATUS_PATH}"
echo "[epoch-sweep] wrote ${SUMMARY_PATH}"
