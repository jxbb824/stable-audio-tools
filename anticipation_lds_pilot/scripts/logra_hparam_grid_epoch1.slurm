#!/bin/bash
#SBATCH --job-name=lds_logra_hparam_grid_e1
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=5
#SBATCH --time=18:00:00
#SBATCH --output=anticipation_lds_pilot/logs/logra_hparam_grid_e1_%j.out
#SBATCH --error=anticipation_lds_pilot/logs/logra_hparam_grid_e1_%j.err

set -uo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"

GRID_ROOT="${GRID_ROOT:-anticipation_lds_pilot/outputs_logra_hparam_grid_epoch1_gt5}"
MODEL_PATH="${MODEL_PATH:-anticipation_lds_pilot/outputs_logra_epoch_sweep/epoch_1/models/full_logra}"
TRAIN_FILE="${TRAIN_FILE:-anticipation_ip_pilot/data/finetune/train_v2.txt}"
QUERIES_FILE="${QUERIES_FILE:-anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt}"
POOL_INDICES_FILE="${POOL_INDICES_FILE:-anticipation_ip_pilot/outputs/selection/seller_indices.txt}"
GROUNDTRUTH_PATH="${GROUNDTRUTH_PATH:-anticipation_lds_pilot/outputs_e5/lds/gt_gen.pt}"
SUBSETS_ROOT="${SUBSETS_ROOT:-anticipation_lds_pilot/outputs/models/subsets}"

PROJ_DIMS="${PROJ_DIMS:-1024 2048 4096 8192}"
DAMPINGS="${DAMPINGS:-0.001 0.003 0.01 0.03 0.1 1 10}"

mkdir -p anticipation_lds_pilot/logs "${GRID_ROOT}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:$PWD/anticipation_lds_pilot:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SUMMARY_TSV="${GRID_ROOT}/summary.tsv"
STATUS_TSV="${GRID_ROOT}/status.tsv"
printf "proj_dim\tdamping\tmean_spearman_lds\tfisher_pvalue\trandom_mean_spearman_lds\tscore_path\n" > "${SUMMARY_TSV}"
printf "proj_dim\tdamping\tstatus\telapsed_sec\tmean_spearman_lds\tfisher_pvalue\trandom_mean_spearman_lds\tscore_path\n" > "${STATUS_TSV}"

echo "[grid] root=${GRID_ROOT}"
echo "[grid] model=${MODEL_PATH}"
echo "[grid] gt=${GROUNDTRUTH_PATH}"
echo "[grid] subsets=${SUBSETS_ROOT}"
echo "[grid] proj_dims=${PROJ_DIMS}"
echo "[grid] dampings=${DAMPINGS}"

for PROJ_DIM in ${PROJ_DIMS}; do
  for DAMPING in ${DAMPINGS}; do
    DAMP_TAG="${DAMPING//./p}"
    DAMP_TAG="${DAMP_TAG//-/m}"
    RUN_DIR="${GRID_ROOT}/proj_${PROJ_DIM}_damping_${DAMP_TAG}"
    ATTR_DIR="${RUN_DIR}/attribution"
    LDS_DIR="${RUN_DIR}/lds"
    SCORE_LOG="${RUN_DIR}/score.log"
    LDS_LOG="${RUN_DIR}/lds.log"
    START_SEC="$(date +%s)"

    mkdir -p "${ATTR_DIR}" "${LDS_DIR}"

    echo "[grid][proj_dim=${PROJ_DIM}][damping=${DAMPING}] scoring LoGra"
    if ! python anticipation_lds_pilot/scripts/score_logra.py \
      --train_file "${TRAIN_FILE}" \
      --valid_file "${QUERIES_FILE}" \
      --model_path "${MODEL_PATH}" \
      --output_dir "${ATTR_DIR}" \
      --include_indices_file "${POOL_INDICES_FILE}" \
      --batch_size "${SCORE_BATCH_SIZE:-8}" \
      --proj_dim "${PROJ_DIM}" \
      --damping "${DAMPING}" \
      --seed "${SEED:-42}" \
      --valid_is_generated \
      --output_filename score_LoGra.pt \
      > "${SCORE_LOG}" 2>&1; then
      ELAPSED_SEC="$(( $(date +%s) - START_SEC ))"
      printf "%s\t%s\tscore_failed\t%s\t\t\t\t\n" "${PROJ_DIM}" "${DAMPING}" "${ELAPSED_SEC}" >> "${STATUS_TSV}"
      echo "[grid][proj_dim=${PROJ_DIM}][damping=${DAMPING}] score failed; see ${SCORE_LOG}"
      continue
    fi

    echo "[grid][proj_dim=${PROJ_DIM}][damping=${DAMPING}] evaluating LDS"
    if ! python anticipation_lds_pilot/scripts/run_lds_eval.py \
      --methods logra \
      --attribution-dir "${ATTR_DIR}" \
      --groundtruth-path "${GROUNDTRUTH_PATH}" \
      --subsets-root "${SUBSETS_ROOT}" \
      --pool-indices-file "${POOL_INDICES_FILE}" \
      --output-dir "${LDS_DIR}" \
      --num-subsets "${NUM_SUBSETS:-30}" \
      > "${LDS_LOG}" 2>&1; then
      ELAPSED_SEC="$(( $(date +%s) - START_SEC ))"
      printf "%s\t%s\tlds_failed\t%s\t\t\t\t\n" "${PROJ_DIM}" "${DAMPING}" "${ELAPSED_SEC}" >> "${STATUS_TSV}"
      echo "[grid][proj_dim=${PROJ_DIM}][damping=${DAMPING}] LDS failed; see ${LDS_LOG}"
      continue
    fi

    if ! python - "${LDS_DIR}/score_LoGra_lds_summary.json" "${PROJ_DIM}" "${DAMPING}" "${SUMMARY_TSV}" "${STATUS_TSV}" "$(( $(date +%s) - START_SEC ))" <<'PY'
import json
import sys

summary_path, proj_dim, damping, summary_out, status_out, elapsed_sec = sys.argv[1:]
with open(summary_path, "r", encoding="utf-8") as handle:
    row = json.load(handle)

fields = [
    proj_dim,
    damping,
    str(row["mean_spearman_lds"]),
    str(row["fisher_pvalue"]),
    str(row["random_mean_spearman_lds"]),
    row["score_path"],
]

with open(summary_out, "a", encoding="utf-8") as handle:
    handle.write("\t".join(fields) + "\n")

with open(status_out, "a", encoding="utf-8") as handle:
    handle.write(
        "\t".join([proj_dim, damping, "ok", elapsed_sec] + fields[2:]) + "\n"
    )
PY
    then
      ELAPSED_SEC="$(( $(date +%s) - START_SEC ))"
      printf "%s\t%s\tsummary_failed\t%s\t\t\t\t\n" "${PROJ_DIM}" "${DAMPING}" "${ELAPSED_SEC}" >> "${STATUS_TSV}"
      echo "[grid][proj_dim=${PROJ_DIM}][damping=${DAMPING}] summary failed"
      continue
    fi

    echo "[grid][proj_dim=${PROJ_DIM}][damping=${DAMPING}] done in $(( $(date +%s) - START_SEC ))s"
  done
done

{ head -n 1 "${SUMMARY_TSV}"; tail -n +2 "${SUMMARY_TSV}" | sort -t $'\t' -k3,3gr; } > "${GRID_ROOT}/summary_sorted.tsv"
echo "[grid] complete; sorted summary: ${GRID_ROOT}/summary_sorted.tsv"
