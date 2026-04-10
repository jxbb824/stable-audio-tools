#!/bin/bash
#SBATCH --job-name=ekfac-groundtruth
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/ekfac_groundtruth_%j.out
#SBATCH --error=logs/ekfac_groundtruth_%j.err
#SBATCH --time=40:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODEL_CONFIG="${MODEL_CONFIG:-model_config_freeze_vae.json}"
TRAIN_DATASET_CONFIG="${TRAIN_DATASET_CONFIG:-stable_audio_tools/configs/dataset_configs/local_training_custom.json}"
QUERY_DATASET_CONFIG="${QUERY_DATASET_CONFIG:-stable_audio_tools/configs/dataset_configs/dtrak_generated_queries.json}"
MODELS_ROOT="${MODELS_ROOT:-outputs/groundtruth_models}"
ENSEMBLE_MODEL_INDICES="${ENSEMBLE_MODEL_INDICES:-}"
FULL_MODEL_DIR="${FULL_MODEL_DIR:-outputs/groundtruth_models/full}"
UNWRAPPED_CKPT="${UNWRAPPED_CKPT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/groundtruth_models/ekfac_attribution}"
TRAIN_COUNT="${TRAIN_COUNT:-}"
QUERY_COUNT="${QUERY_COUNT:-}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda:0}"
MEASUREMENT_F="${MEASUREMENT_F:-l2-norm}"
MEASUREMENT_CFG_DROPOUT_PROB="${MEASUREMENT_CFG_DROPOUT_PROB:-0.0}"
LOSS_CFG_DROPOUT_PROB="${LOSS_CFG_DROPOUT_PROB:-}"
DAMPING="${DAMPING:-1e-8}"
KFAC_NUM_SAMPLES_FOR_LOSS="${KFAC_NUM_SAMPLES_FOR_LOSS:-63}"
KFAC_BATCH_SIZE="${KFAC_BATCH_SIZE:-2}"
NUM_SAMPLES_FOR_LOSS_SCORING="${NUM_SAMPLES_FOR_LOSS_SCORING:-8}"
NUM_SAMPLES_FOR_MEASUREMENT_SCORING="${NUM_SAMPLES_FOR_MEASUREMENT_SCORING:-8}"
NUM_LOSS_BATCH_AGGREGATIONS="${NUM_LOSS_BATCH_AGGREGATIONS:-2}"
NUM_MEASUREMENT_BATCH_AGGREGATIONS="${NUM_MEASUREMENT_BATCH_AGGREGATIONS:-8}"
QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
QUERY_GRADIENT_COMPRESSION_BITS="${QUERY_GRADIENT_COMPRESSION_BITS:-4}"
LINEAR_LAYER_START_FRACTION="${LINEAR_LAYER_START_FRACTION:-0.5}"
SEED="${SEED:-42}"
DIFFUSION_INFLUENCE_ROOT="${DIFFUSION_INFLUENCE_ROOT:-/ocean/projects/mth240006p/xjiang6/diffusion-influence}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export PYTHONPATH="${DIFFUSION_INFLUENCE_ROOT}:${PYTHONPATH:-}"

CMD=(
  python3 -u scripts/groundtruth_ekfac_attribution.py
  --model-config "$MODEL_CONFIG"
  --train-dataset-config "$TRAIN_DATASET_CONFIG"
  --query-dataset-config "$QUERY_DATASET_CONFIG"
  --models-root "$MODELS_ROOT"
  --ensemble-model-indices "$ENSEMBLE_MODEL_INDICES"
  --full-model-dir "$FULL_MODEL_DIR"
  --output-dir "$OUTPUT_DIR"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --measurement-f "$MEASUREMENT_F"
  --measurement-cfg-dropout-prob "$MEASUREMENT_CFG_DROPOUT_PROB"
  --damping "$DAMPING"
  --kfac-num-samples-for-loss "$KFAC_NUM_SAMPLES_FOR_LOSS"
  --kfac-batch-size "$KFAC_BATCH_SIZE"
  --num-samples-for-loss-scoring "$NUM_SAMPLES_FOR_LOSS_SCORING"
  --num-samples-for-measurement-scoring "$NUM_SAMPLES_FOR_MEASUREMENT_SCORING"
  --num-loss-batch-aggregations "$NUM_LOSS_BATCH_AGGREGATIONS"
  --num-measurement-batch-aggregations "$NUM_MEASUREMENT_BATCH_AGGREGATIONS"
  --query-batch-size "$QUERY_BATCH_SIZE"
  --train-batch-size "$TRAIN_BATCH_SIZE"
  --query-gradient-compression-bits "$QUERY_GRADIENT_COMPRESSION_BITS"
  --linear-layer-start-fraction "$LINEAR_LAYER_START_FRACTION"
  --seed "$SEED"
)

if [[ -n "$UNWRAPPED_CKPT" ]]; then
  CMD+=(--unwrapped-ckpt "$UNWRAPPED_CKPT")
fi

if [[ -n "$TRAIN_COUNT" ]]; then
  CMD+=(--train-count "$TRAIN_COUNT")
fi

if [[ -n "$QUERY_COUNT" ]]; then
  CMD+=(--query-count "$QUERY_COUNT")
fi

if [[ -n "$LOSS_CFG_DROPOUT_PROB" ]]; then
  CMD+=(--loss-cfg-dropout-prob "$LOSS_CFG_DROPOUT_PROB")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

"${CMD[@]}"
