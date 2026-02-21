#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/xiruij/stable-audio-tools"
cd "${ROOT_DIR}"

MODEL_CONFIG="${MODEL_CONFIG:-${ROOT_DIR}/model_config.json}"
TRAIN_DATASET_CONFIG="${TRAIN_DATASET_CONFIG:-${ROOT_DIR}/stable_audio_tools/configs/dataset_configs/local_training_custom.json}"
QUERY_DATASET_CONFIG="${QUERY_DATASET_CONFIG:-${ROOT_DIR}/stable_audio_tools/configs/dataset_configs/dtrak_generated_queries.json}"

UNWRAPPED_CKPT="${UNWRAPPED_CKPT:-${ROOT_DIR}/model_unwrap.ckpt}"
LIGHTNING_CKPT="${LIGHTNING_CKPT:-${ROOT_DIR}/outputs/stable_audio_open_finetune/kklqsk68/checkpoints/epoch=127-step=40000.ckpt}"
PRETRANSFORM_CKPT="${PRETRANSFORM_CKPT:-}"
REMOVE_PRETRANSFORM_WEIGHT_NORM="${REMOVE_PRETRANSFORM_WEIGHT_NORM:-none}"

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/outputs/dtrak_attribution_$(date +%Y%m%d_%H%M%S)}"
TRAIN_COUNT="${TRAIN_COUNT:-5000}"
QUERY_COUNT="${QUERY_COUNT:-10}"

PROJ_DIM="${PROJ_DIM:-16384}"
USED_DIM="${USED_DIM:-8192}"
K="${K:-10}"
T_STRATEGY="${T_STRATEGY:-uniform}"
F_OBJECTIVE="${F_OBJECTIVE:-l2-norm}"
LAMBDA_REG="${LAMBDA_REG:-1e2}"

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DEVICE="${DEVICE:-cuda:0}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000}"
CFG_DROPOUT_PROB="${CFG_DROPOUT_PROB:-0.0}"
# Only use the last 8 transformer layers (16-23) for D-TRAK gradients to save GPU memory.
# Full model has 24 layers; last-8 gives ~1.30 GiB grad_dim vs ~3.94 GiB for all layers.
PARAM_REGEX="${PARAM_REGEX:-layers\.(1[6-9]|2[0-3])\.}"

mkdir -p "${OUT_DIR}" logs

echo "[check] querying generated samples in ${ROOT_DIR}/outputs/generated"
GEN_COUNT="$(find "${ROOT_DIR}/outputs/generated" -maxdepth 1 -type f -name '*.wav' | wc -l | tr -d ' ')"
if [[ "${GEN_COUNT}" -lt "${QUERY_COUNT}" ]]; then
  echo "Expected at least ${QUERY_COUNT} wavs in outputs/generated, found ${GEN_COUNT}."
  exit 1
fi

CKPT_TO_USE=""
if [[ -f "${UNWRAPPED_CKPT}" ]]; then
  CKPT_TO_USE="${UNWRAPPED_CKPT}"
  echo "[ckpt] using existing unwrapped checkpoint: ${CKPT_TO_USE}"
elif [[ -f "${LIGHTNING_CKPT}" ]]; then
  CKPT_TO_USE="${OUT_DIR}/model_unwrap_auto.ckpt"
  echo "[ckpt] unwrapped checkpoint not found, exporting from Lightning checkpoint..."
  python3 unwrap_model.py \
    --model-config "${MODEL_CONFIG}" \
    --ckpt-path "${LIGHTNING_CKPT}" \
    --name "${OUT_DIR}/model_unwrap_auto"
  echo "[ckpt] exported: ${CKPT_TO_USE}"
else
  echo "Neither UNWRAPPED_CKPT nor LIGHTNING_CKPT exists."
  echo "UNWRAPPED_CKPT=${UNWRAPPED_CKPT}"
  echo "LIGHTNING_CKPT=${LIGHTNING_CKPT}"
  exit 1
fi

TRAIN_FEATURE="${OUT_DIR}/train_features.memmap"
QUERY_FEATURE="${OUT_DIR}/query_features.memmap"
SCORE_MATRIX="${OUT_DIR}/scores_query_x_train.memmap"

COMMON_ARGS=(
  --model-config "${MODEL_CONFIG}"
  --pretrained-ckpt-path "${CKPT_TO_USE}"
  --remove-pretransform-weight-norm "${REMOVE_PRETRANSFORM_WEIGHT_NORM}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --proj-dim "${PROJ_DIM}"
  --used-dim "${USED_DIM}"
  --K "${K}"
  --t-strategy "${T_STRATEGY}"
  --num-train-steps "${NUM_TRAIN_STEPS}"
  --f "${F_OBJECTIVE}"
  --cfg-dropout-prob "${CFG_DROPOUT_PROB}"
  --disable-random-crop
)

if [[ -n "${PARAM_REGEX}" ]]; then
  COMMON_ARGS+=(--param-regex "${PARAM_REGEX}")
fi

if [[ -n "${PRETRANSFORM_CKPT}" ]]; then
  if [[ ! -f "${PRETRANSFORM_CKPT}" ]]; then
    echo "PRETRANSFORM_CKPT set but not found: ${PRETRANSFORM_CKPT}"
    exit 1
  fi
  COMMON_ARGS+=(--pretransform-ckpt-path "${PRETRANSFORM_CKPT}")
fi

echo "[1/3] extracting train features (${TRAIN_COUNT})"
python3 scripts/dtrak_extract_features.py \
  "${COMMON_ARGS[@]}" \
  --dataset-config "${TRAIN_DATASET_CONFIG}" \
  --feature-path "${TRAIN_FEATURE}" \
  --max-examples "${TRAIN_COUNT}"

echo "[2/3] extracting query features (${QUERY_COUNT})"
python3 scripts/dtrak_extract_features.py \
  "${COMMON_ARGS[@]}" \
  --dataset-config "${QUERY_DATASET_CONFIG}" \
  --feature-path "${QUERY_FEATURE}" \
  --max-examples "${QUERY_COUNT}"

echo "[3/3] computing score matrix"
python3 scripts/dtrak_score.py \
  --train-feature-paths "${TRAIN_FEATURE}" \
  --train-meta-paths "${TRAIN_FEATURE}.meta.json" \
  --query-feature-path "${QUERY_FEATURE}" \
  --query-meta-path "${QUERY_FEATURE}.meta.json" \
  --output-score-path "${SCORE_MATRIX}" \
  --lambda-reg "${LAMBDA_REG}" \
  --used-dim "${USED_DIM}" \
  --device "${DEVICE}"

echo "[done] all outputs are in: ${OUT_DIR}"
ls -lah "${OUT_DIR}"
