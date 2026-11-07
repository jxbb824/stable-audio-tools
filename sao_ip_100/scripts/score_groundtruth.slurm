#!/bin/bash
#SBATCH --job-name=sao50_gt
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=24:00:00
#SBATCH --chdir=/ocean/projects/mth240006p/xjiang6/stable-audio-tools
#SBATCH --output=sao_ip_100/logs/gt_%j.out
#SBATCH --error=sao_ip_100/logs/gt_%j.err

set -euo pipefail
cd /ocean/projects/mth240006p/xjiang6/stable-audio-tools
mkdir -p sao_ip_100/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export PYTHONPATH="${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

N_SELLERS=50
MODEL_INDICES="$(seq -s ' ' 0 "$N_SELLERS")"

python3 -u sao_ip_pilot/scripts/groundtruth.py \
  --models-root sao_ip_100/outputs/models \
  --model-indices $MODEL_INDICES \
  --model-config model_config_freeze_vae.json \
  --query-dataset-config sao_ip_100/configs/query_dataset.json \
  --output-path sao_ip_100/outputs/losses_model_x_query.pt \
  --batch-size "${BATCH_SIZE:-2}" \
  --num-workers 4 \
  --device cuda:0 \
  --seed 42 \
  --e-seed 0 \
  --K 10 \
  --t-strategy uniform \
  --num-train-steps 1000 \
  --f loss \
  --cfg-dropout-prob 0.0 \
  --max-examples 200

python3 -u sao_ip_pilot/scripts/compute_a_star_from_losses.py \
  --loss-path sao_ip_100/outputs/losses_model_x_query.pt \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --prompt-manifest sao_ip_100/outputs/generated_queries/prompt_manifest.json \
  --output-path sao_ip_100/outputs/a_star.csv
