#!/bin/bash
#SBATCH --job-name=dtrak-groundtruth
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dtrak_groundtruth_%j.out
#SBATCH --error=logs/dtrak_groundtruth_%j.err
#SBATCH --time=40:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODEL_CONFIG="${MODEL_CONFIG:-model_config_freeze_vae.json}"
TRAIN_DATASET_CONFIG="${TRAIN_DATASET_CONFIG:-stable_audio_tools/configs/dataset_configs/local_training_custom.json}"
QUERY_DATASET_CONFIG="${QUERY_DATASET_CONFIG:-stable_audio_tools/configs/dataset_configs/dtrak_generated_queries.json}"
MODELS_ROOT="${MODELS_ROOT:-outputs/groundtruth_models}"
ENSEMBLE_MODEL_INDICES="${ENSEMBLE_MODEL_INDICES:-0, 1, 2, 3, 4}"
FULL_MODEL_DIR="${FULL_MODEL_DIR:-outputs/groundtruth_models/full}"
UNWRAPPED_CKPT="${UNWRAPPED_CKPT:-}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/groundtruth_models/dtrak_attribution}"
TRAIN_COUNT="${TRAIN_COUNT:-}"
QUERY_COUNT="${QUERY_COUNT:-}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda:0}"
PROJ_DIM="${PROJ_DIM:-16384}"
USED_DIM="${USED_DIM:-8192}"
K="${K:-10}"
T_STRATEGY="${T_STRATEGY:-uniform}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000}"
F_OBJECTIVE="${F_OBJECTIVE:-l2-norm}"
CFG_DROPOUT_PROB="${CFG_DROPOUT_PROB:-0.0}"
LAMBDA_REG="${LAMBDA_REG:-1e2}"
PARAM_REGEX=""
SEED="${SEED:-42}"
E_SEED="${E_SEED:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

CMD=(
  python3 -u scripts/groundtruth_dtrak_attribution.py
  --model-config "$MODEL_CONFIG"
  --train-dataset-config "$TRAIN_DATASET_CONFIG"
  --query-dataset-config "$QUERY_DATASET_CONFIG"
  --models-root "$MODELS_ROOT"
  --ensemble-model-indices "$ENSEMBLE_MODEL_INDICES"
  --output-dir "$OUTPUT_DIR"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --device "$DEVICE"
  --proj-dim "$PROJ_DIM"
  --used-dim "$USED_DIM"
  --K "$K"
  --t-strategy "$T_STRATEGY"
  --num-train-steps "$NUM_TRAIN_STEPS"
  --f "$F_OBJECTIVE"
  --cfg-dropout-prob "$CFG_DROPOUT_PROB"
  --lambda-reg "$LAMBDA_REG"
  --param-regex "$PARAM_REGEX"
  --seed "$SEED"
  --e-seed "$E_SEED"
)

if [[ -n "$FULL_MODEL_DIR" ]]; then
  CMD+=(--full-model-dir "$FULL_MODEL_DIR")
fi

if [[ -n "$UNWRAPPED_CKPT" ]]; then
  CMD+=(--unwrapped-ckpt "$UNWRAPPED_CKPT")
fi

if [[ -n "$TRAIN_COUNT" ]]; then
  CMD+=(--train-count "$TRAIN_COUNT")
fi

if [[ -n "$QUERY_COUNT" ]]; then
  CMD+=(--query-count "$QUERY_COUNT")
fi

if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=($EXTRA_ARGS)
  CMD+=("${EXTRA_ARR[@]}")
fi

"${CMD[@]}"
