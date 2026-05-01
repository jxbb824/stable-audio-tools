#!/bin/bash
#SBATCH --job-name=sao_ip_dtrak
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=logs/sao_ip_dtrak_%j.out
#SBATCH --error=logs/sao_ip_dtrak_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR="${OUTPUT_DIR:-sao_ip_pilot/outputs/dtrak_attribution}"

CMD=(
python3 -u scripts/groundtruth_dtrak_attribution.py
  --model-config "${MODEL_CONFIG:-model_config_freeze_vae.json}"
  --train-dataset-config "${TRAIN_DATASET_CONFIG:-sao_ip_pilot/configs/train_dataset.json}"
  --query-dataset-config "${QUERY_DATASET_CONFIG:-sao_ip_pilot/configs/query_dataset.json}"
  --models-root "${MODELS_ROOT:-sao_ip_pilot/outputs/models}"
  --full-model-dir "${FULL_MODEL_DIR:-sao_ip_pilot/outputs/models/0}"
  --output-dir "$OUTPUT_DIR"
  --batch-size "${BATCH_SIZE:-4}"
  --num-workers "${NUM_WORKERS:-8}"
  --device "${DEVICE:-cuda:0}"
  --proj-dim "${PROJ_DIM:-2048}"
  --used-dim "${USED_DIM:-2048}"
  --K "${K:-10}"
  --t-strategy "${T_STRATEGY:-uniform}"
  --num-train-steps "${NUM_TRAIN_STEPS:-1000}"
  --f "${F_OBJECTIVE:-loss}"
  --cfg-dropout-prob "${CFG_DROPOUT_PROB:-0.0}"
  --lambda-reg "${LAMBDA_REG:-0.1}"
  --param-regex "${PARAM_REGEX:-}"
  --seed "${SEED:-42}"
  --e-seed "${E_SEED:-0}"
  --query-count "${QUERY_COUNT:-200}"
)

if [[ -n "${UNWRAPPED_CKPT:-}" ]]; then
  CMD+=(--unwrapped-ckpt "$UNWRAPPED_CKPT")
fi
if [[ -n "${TRAIN_COUNT:-}" ]]; then
  CMD+=(--train-count "$TRAIN_COUNT")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

"${CMD[@]}"

python3 -u sao_ip_pilot/scripts/aggregate_ekfac_to_sellers.py \
  --score-path "$OUTPUT_DIR/scores_train_x_query.pt" \
  --train-axis-manifest "$OUTPUT_DIR/train_axis_manifest.csv" \
  --query-axis-manifest "$OUTPUT_DIR/query_axis_manifest.csv" \
  --output-path sao_ip_pilot/outputs/a_hat_dtrak.csv \
  --score-column a_hat_dtrak

python3 -u sao_ip_pilot/scripts/estimate_I.py \
  --method dtrak_influence \
  --output-path sao_ip_pilot/outputs/I_hat_dtrak.csv

python3 -u sao_ip_pilot/scripts/plot_results.py \
  --i-hat-path sao_ip_pilot/outputs/I_hat_dtrak.csv \
  --a-hat-path sao_ip_pilot/outputs/a_hat_dtrak.csv \
  --output-dir sao_ip_pilot/outputs/figures_dtrak \
  --marginal-welfare-csv sao_ip_pilot/outputs/marginal_welfare_numbers_dtrak.csv \
  --method-label D-TRAK
