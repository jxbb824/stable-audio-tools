#!/bin/bash
#SBATCH --job-name=sao_ip_gen
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/sao_ip_generate_%j.out
#SBATCH --error=logs/sao_ip_generate_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

python3 -u sao_ip_pilot/scripts/generate_queries_from_prompt_json.py \
  --model-config "${MODEL_CONFIG:-model_config_freeze_vae.json}" \
  --ckpt-path "${CKPT_PATH:-sao_ip_pilot/outputs/models/0/model.ckpt}" \
  --prompt-json "${PROMPT_JSON:-sao_ip_pilot/outputs/prompts.json}" \
  --output-dir "${OUTPUT_DIR:-sao_ip_pilot/outputs/generated_queries}" \
  --output-ext "${OUTPUT_EXT:-mp3}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --steps "${STEPS:-100}" \
  --cfg-scale "${CFG_SCALE:-7.0}" \
  --sigma-min "${SIGMA_MIN:-0.3}" \
  --sigma-max "${SIGMA_MAX:-500.0}" \
  --sampler-type "${SAMPLER_TYPE:-dpmpp-3m-sde}" \
  --base-seed "${BASE_SEED:-42}" \
  ${EXTRA_ARGS:-}
