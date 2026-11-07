#!/bin/bash
#SBATCH --job-name=sao50_dtrak
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=16:00:00
#SBATCH --chdir=/ocean/projects/mth240006p/xjiang6/stable-audio-tools
#SBATCH --output=sao_ip_100/logs/dtrak_%j.out
#SBATCH --error=sao_ip_100/logs/dtrak_%j.err

set -euo pipefail
cd /ocean/projects/mth240006p/xjiang6/stable-audio-tools
mkdir -p sao_ip_100/logs sao_ip_100/outputs/dtrak_attribution sao_ip_100/outputs/figures_dtrak

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export PYTHONPATH="${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR=sao_ip_100/outputs/dtrak_attribution

python3 -u scripts/groundtruth_dtrak_attribution.py \
  --model-config model_config_freeze_vae.json \
  --train-dataset-config sao_ip_100/configs/train_dataset.json \
  --query-dataset-config sao_ip_100/configs/query_dataset.json \
  --models-root sao_ip_100/outputs/models \
  --full-model-dir sao_ip_100/outputs/models/0 \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "${BATCH_SIZE:-2}" \
  --num-workers 4 \
  --device cuda:0 \
  --proj-dim 2048 \
  --used-dim 2048 \
  --K 10 \
  --t-strategy uniform \
  --num-train-steps 1000 \
  --f loss \
  --cfg-dropout-prob 0.0 \
  --lambda-reg 0.1 \
  --seed 42 \
  --e-seed 0 \
  --query-count 200

python3 -u sao_ip_pilot/scripts/aggregate_ekfac_to_sellers.py \
  --score-path "$OUTPUT_DIR/scores_train_x_query.pt" \
  --train-axis-manifest "$OUTPUT_DIR/train_axis_manifest.csv" \
  --query-axis-manifest "$OUTPUT_DIR/query_axis_manifest.csv" \
  --category-manifest sao_ip_100/outputs/selection/category_manifest.csv \
  --output-path sao_ip_100/outputs/a_hat_dtrak.csv \
  --score-column a_hat_dtrak

python3 -u sao_ip_pilot/scripts/estimate_I.py \
  --method dtrak_influence \
  --a-star-path sao_ip_100/outputs/a_star.csv \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --a-hat-path sao_ip_100/outputs/a_hat_dtrak.csv \
  --output-path sao_ip_100/outputs/I_hat_dtrak.csv

python3 -u sao_ip_pilot/scripts/plot_results.py \
  --i-hat-path sao_ip_100/outputs/I_hat_dtrak.csv \
  --a-hat-path sao_ip_100/outputs/a_hat_dtrak.csv \
  --seller-manifest sao_ip_100/outputs/selection/seller_manifest.csv \
  --output-dir sao_ip_100/outputs/figures_dtrak \
  --marginal-welfare-csv sao_ip_100/outputs/marginal_welfare_numbers_dtrak.csv \
  --method-label D-TRAK
