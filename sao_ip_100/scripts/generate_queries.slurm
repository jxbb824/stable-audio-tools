#!/bin/bash
#SBATCH --job-name=sao50_gen
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=04:00:00
#SBATCH --chdir=/ocean/projects/mth240006p/xjiang6/stable-audio-tools
#SBATCH --output=sao_ip_100/logs/gen_%j.out
#SBATCH --error=sao_ip_100/logs/gen_%j.err

set -euo pipefail
cd /ocean/projects/mth240006p/xjiang6/stable-audio-tools
mkdir -p sao_ip_100/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python3 -u sao_ip_pilot/scripts/generate_queries_from_prompt_json.py \
  --model-config model_config_freeze_vae.json \
  --ckpt-path sao_ip_100/outputs/models/0/model.ckpt \
  --prompt-json sao_ip_100/outputs/prompts.json \
  --output-dir sao_ip_100/outputs/generated_queries \
  --output-ext mp3 \
  --batch-size "${BATCH_SIZE:-2}" \
  --steps 100 \
  --cfg-scale 7.0 \
  --sigma-min 0.3 \
  --sigma-max 500.0 \
  --sampler-type dpmpp-3m-sde \
  --base-seed 42
