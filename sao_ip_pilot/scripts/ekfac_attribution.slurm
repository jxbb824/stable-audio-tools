#!/bin/bash
#SBATCH --job-name=sao_ip_ekfac
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=220G
#SBATCH --cpus-per-task=8
#SBATCH --time=40:00:00
#SBATCH --output=logs/sao_ip_ekfac_%j.out
#SBATCH --error=logs/sao_ip_ekfac_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export DIFFUSION_INFLUENCE_ROOT="${DIFFUSION_INFLUENCE_ROOT:-/ocean/projects/mth240006p/xjiang6/diffusion-influence}"
export PYTHONPATH="${DIFFUSION_INFLUENCE_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(
python3 -u sao_ip_pilot/scripts/groundtruth_ekfac_attribution.py \
  --model-config "${MODEL_CONFIG:-model_config_freeze_vae.json}" \
  --train-dataset-config "${TRAIN_DATASET_CONFIG:-sao_ip_pilot/configs/train_dataset.json}" \
  --query-dataset-config "${QUERY_DATASET_CONFIG:-sao_ip_pilot/configs/query_dataset.json}" \
  --models-root "${MODELS_ROOT:-sao_ip_pilot/outputs/models}" \
  --full-model-dir "${FULL_MODEL_DIR:-sao_ip_pilot/outputs/models/0}" \
  --output-dir "${OUTPUT_DIR:-sao_ip_pilot/outputs/ekfac_attribution}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --device "${DEVICE:-cuda:0}" \
  --measurement-f loss \
  --measurement-cfg-dropout-prob "${MEASUREMENT_CFG_DROPOUT_PROB:-0.0}" \
  --damping "${DAMPING:-1e-8}" \
  --kfac-num-samples-for-loss "${KFAC_NUM_SAMPLES_FOR_LOSS:-16}" \
  --kfac-batch-size "${KFAC_BATCH_SIZE:-2}" \
  --num-samples-for-loss-scoring "${NUM_SAMPLES_FOR_LOSS_SCORING:-8}" \
  --num-samples-for-measurement-scoring "${NUM_SAMPLES_FOR_MEASUREMENT_SCORING:-8}" \
  --num-loss-batch-aggregations "${NUM_LOSS_BATCH_AGGREGATIONS:-1}" \
  --num-measurement-batch-aggregations "${NUM_MEASUREMENT_BATCH_AGGREGATIONS:-1}" \
  --query-batch-size "${QUERY_BATCH_SIZE:-4}" \
  --train-batch-size "${TRAIN_BATCH_SIZE:-4}" \
  --query-gradient-compression-bits "${QUERY_GRADIENT_COMPRESSION_BITS:-4}" \
  --linear-layer-start-fraction "${LINEAR_LAYER_START_FRACTION:-0.5}" \
  --seed "${SEED:-42}"
)

if [[ -n "${TRAIN_COUNT:-}" ]]; then
  CMD+=(--train-count "$TRAIN_COUNT")
fi
if [[ -n "${QUERY_COUNT:-}" ]]; then
  CMD+=(--query-count "$QUERY_COUNT")
fi
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

"${CMD[@]}"

python3 -u sao_ip_pilot/scripts/aggregate_ekfac_to_sellers.py
