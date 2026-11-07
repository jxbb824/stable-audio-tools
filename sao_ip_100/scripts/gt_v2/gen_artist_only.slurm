#!/bin/bash
#SBATCH --job-name=gtv2_gen
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=03:00:00
#SBATCH --chdir=/ocean/projects/mth260005p/xjiang6/stable-audio-tools
#SBATCH --output=sao_ip_100/logs/gtv2_gen_%j.out
#SBATCH --error=sao_ip_100/logs/gtv2_gen_%j.err
source sao_ip_100/scripts/gt_v2/common.sh
# Same sampler, seeds and batch size as the original queries, so query i gets the same noise.
# Writes to a NEW directory: the generator clears *.mp3 in its output dir.
python3 -u sao_ip_pilot/scripts/generate_queries_from_prompt_json.py \
  --model-config model_config_freeze_vae.json \
  --ckpt-path sao_ip_100/outputs/models/0/model.ckpt \
  --prompt-json sao_ip_100/outputs/prompts_artist_only.json \
  --output-dir sao_ip_100/outputs/generated_queries_artist_only \
  --output-ext mp3 --batch-size 2 --steps 100 --cfg-scale 7.0 \
  --sigma-min 0.3 --sigma-max 500.0 --sampler-type dpmpp-3m-sde --base-seed 42
