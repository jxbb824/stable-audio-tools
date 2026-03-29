#!/bin/bash
#SBATCH --job-name=sa_groundtruth
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/groundtruth_%j.out
#SBATCH --error=logs/groundtruth_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODELS_ROOT="${MODELS_ROOT:-outputs/groundtruth_models}"
MODEL_CONFIG="${MODEL_CONFIG:-model_config_freeze_vae.json}"
QUERY_DATASET_CONFIG="${QUERY_DATASET_CONFIG:-stable_audio_tools/configs/dataset_configs/dtrak_generated_queries.json}"
OUTPUT_PATH="${OUTPUT_PATH:-outputs/groundtruth_models/gt_gen.pt}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
E_SEED="${E_SEED:-0}"
K="${K:-10}"
T_STRATEGY="${T_STRATEGY:-uniform}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-1000}"
F_OBJECTIVE="${F_OBJECTIVE:-l2-norm}"
CFG_DROPOUT_PROB="${CFG_DROPOUT_PROB:-0.0}"
MAX_EXAMPLES="${MAX_EXAMPLES:-200}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

python3 -u scripts/groundtruth.py \
  --models-root "$MODELS_ROOT" \
  --model-config "$MODEL_CONFIG" \
  --query-dataset-config "$QUERY_DATASET_CONFIG" \
  --output-path "$OUTPUT_PATH" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --device "$DEVICE" \
  --seed "$SEED" \
  --e-seed "$E_SEED" \
  --K "$K" \
  --t-strategy "$T_STRATEGY" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --f "$F_OBJECTIVE" \
  --cfg-dropout-prob "$CFG_DROPOUT_PROB" \
  --max-examples "$MAX_EXAMPLES" \
  $EXTRA_ARGS
