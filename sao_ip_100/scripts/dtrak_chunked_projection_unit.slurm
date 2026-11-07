#!/bin/bash
#SBATCH --job-name=dtck_unit
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-16:1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=sao_ip_100/logs/dtrak_chunked_projection_unit_%j.out
#SBATCH --error=sao_ip_100/logs/dtrak_chunked_projection_unit_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
cd "$REPO_ROOT"

source /jet/home/xjiang6/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

python3 -u scripts/validate_dtrak_chunked_projection.py \
  --device cuda:0 \
  --output-path "sao_ip_100/outputs/dtrak_chunked_projection_unit_${SLURM_JOB_ID}.json"
