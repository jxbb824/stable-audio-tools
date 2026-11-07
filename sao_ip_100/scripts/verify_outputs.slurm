#!/bin/bash
#SBATCH --job-name=sao50_verify
#SBATCH -p RM-shared
#SBATCH -A mth260005p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=sao_ip_100/logs/verify_outputs_%j.out
#SBATCH --error=sao_ip_100/logs/verify_outputs_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
cd "$REPO_ROOT"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

python3 -u sao_ip_100/scripts/verify_outputs.py

echo "[verify] scale-up attribution outputs are complete"
