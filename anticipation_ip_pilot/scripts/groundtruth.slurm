#!/bin/bash
#SBATCH --job-name=anticip_ip_gt
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:15:00
#SBATCH --output=anticipation_ip_pilot/logs/groundtruth_%j.out
#SBATCH --error=anticipation_ip_pilot/logs/groundtruth_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p anticipation_ip_pilot/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:${PYTHONPATH:-}"

python anticipation_ip_pilot/scripts/groundtruth.py \
  --queries-file "${QUERIES_FILE:-anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt}" \
  --seller-manifest "${SELLER_MANIFEST:-anticipation_ip_pilot/outputs/selection/seller_manifest.csv}" \
  --full-model-dir "${FULL_MODEL_DIR:-anticipation_ip_pilot/outputs/models/full}" \
  --loo-root "${LOO_ROOT:-anticipation_ip_pilot/outputs/models/loo}" \
  --output-dir "${OUTPUT_DIR:-anticipation_ip_pilot/outputs}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-prompts "${NUM_PROMPTS:-500}" \
  --seed "${SEED:-42}"
