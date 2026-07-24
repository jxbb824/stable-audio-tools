#!/bin/bash
#SBATCH --job-name=lds_gt
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=5
#SBATCH --time=02:00:00
#SBATCH --output=anticipation_lds_pilot/logs/gt_%j.out
#SBATCH --error=anticipation_lds_pilot/logs/gt_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p anticipation_lds_pilot/logs anticipation_lds_pilot/outputs/lds

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:$PWD/anticipation_lds_pilot:${PYTHONPATH:-}"

python anticipation_lds_pilot/scripts/groundtruth.py \
  --queries-file "${QUERIES_FILE:-anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt}" \
  --subsets-root "${SUBSETS_ROOT:-anticipation_lds_pilot/outputs/models/subsets}" \
  --output-path "${OUTPUT_PATH:-anticipation_lds_pilot/outputs/lds/gt_gen.pt}" \
  --num-subsets "${NUM_SUBSETS:-30}" \
  --num-queries "${NUM_PROMPTS:-500}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --seed "${SEED:-42}"
