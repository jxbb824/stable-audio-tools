#!/bin/bash
#SBATCH --job-name=sa_generate_queries
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/generate_queries_%j.out
#SBATCH --error=logs/generate_queries_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MODEL_CONFIG="${MODEL_CONFIG:-model_config_freeze_vae.json}"
CKPT_PATH="${CKPT_PATH:-outputs/groundtruth_models/full/model.ckpt}"
PROMPT_JSON="${PROMPT_JSON:-outputs/groundtruth_models/full/prompts_200.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/generated_queries}"
OUTPUT_EXT="${OUTPUT_EXT:-mp3}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS="${STEPS:-100}"
CFG_SCALE="${CFG_SCALE:-7.0}"
SIGMA_MIN="${SIGMA_MIN:-0.3}"
SIGMA_MAX="${SIGMA_MAX:-500.0}"
SAMPLER_TYPE="${SAMPLER_TYPE:-dpmpp-3m-sde}"
BASE_SEED="${BASE_SEED:-42}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

python3 -u scripts/generate_queries_from_prompt_json.py \
  --model-config "$MODEL_CONFIG" \
  --ckpt-path "$CKPT_PATH" \
  --prompt-json "$PROMPT_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --output-ext "$OUTPUT_EXT" \
  --batch-size "$BATCH_SIZE" \
  --steps "$STEPS" \
  --cfg-scale "$CFG_SCALE" \
  --sigma-min "$SIGMA_MIN" \
  --sigma-max "$SIGMA_MAX" \
  --sampler-type "$SAMPLER_TYPE" \
  --base-seed "$BASE_SEED" \
  $EXTRA_ARGS
