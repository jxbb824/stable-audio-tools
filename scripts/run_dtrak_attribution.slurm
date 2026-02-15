#!/bin/bash
#SBATCH --job-name=dtrak
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/dtrak_%j.out
#SBATCH --error=logs/dtrak_%j.err
#SBATCH --time=96:00:00

set -euo pipefail

cd /home/xiruij/stable-audio-tools
mkdir -p logs

# Optional overrides:
# export OUT_DIR="/home/xiruij/stable-audio-tools/outputs/dtrak_attribution_run1"
# export PROJ_DIM=16384
# export USED_DIM=8192
# export K=10
# export LAMBDA_REG=1e2
# export DEVICE="cuda:0"


bash scripts/run_dtrak_attribution.sh

