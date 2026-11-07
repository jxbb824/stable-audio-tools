#!/bin/bash
#SBATCH --job-name=sao50_ekfac
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=sao_ip_100/logs/ekfac_score_%j.out
#SBATCH --error=sao_ip_100/logs/ekfac_score_%j.err

# EKFAC scoring + seller aggregation (no a* needed; can run parallel with gt).
set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
cd "$REPO_ROOT"
mkdir -p sao_ip_100/logs sao_ip_100/outputs/ekfac_attribution

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export DIFFUSION_INFLUENCE_ROOT="${DIFFUSION_INFLUENCE_ROOT:-${REPO_ROOT}/../diffusion-influence}"
export PYTHONPATH="${DIFFUSION_INFLUENCE_ROOT}:${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUTPUT_DIR=sao_ip_100/outputs/ekfac_attribution

python3 -u sao_ip_pilot/scripts/groundtruth_ekfac_attribution.py \
  --model-config model_config_freeze_vae.json \
  --train-dataset-config sao_ip_100/configs/train_dataset.json \
  --query-dataset-config sao_ip_100/configs/query_dataset.json \
  --models-root sao_ip_100/outputs/models \
  --full-model-dir sao_ip_100/outputs/models/0 \
  --output-dir "$OUTPUT_DIR" \
  --num-workers 8 \
  --device cuda:0 \
  --measurement-f loss \
  --measurement-cfg-dropout-prob 0.0 \
  --damping 1e-8 \
  --kfac-num-samples-for-loss 16 \
  --kfac-batch-size 2 \
  --num-samples-for-loss-scoring 8 \
  --num-samples-for-measurement-scoring 8 \
  --num-loss-batch-aggregations 1 \
  --num-measurement-batch-aggregations 1 \
  --query-batch-size 4 \
  --train-batch-size 4 \
  --query-gradient-compression-bits 4 \
  --linear-layer-start-fraction 0.5 \
  --query-count 200 \
  --seed 42

python3 -u sao_ip_pilot/scripts/aggregate_ekfac_to_sellers.py \
  --score-path "$OUTPUT_DIR/scores_train_x_query.pt" \
  --train-axis-manifest "$OUTPUT_DIR/train_axis_manifest.csv" \
  --query-axis-manifest "$OUTPUT_DIR/query_axis_manifest.csv" \
  --category-manifest sao_ip_100/outputs/selection/category_manifest.csv \
  --output-path sao_ip_100/outputs/a_hat_ekfac.csv \
  --score-column a_hat_ekfac

echo "[ekfac-score] done -> sao_ip_100/outputs/a_hat_ekfac.csv"
