#!/bin/bash
#SBATCH --job-name=lds_eval
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=anticipation_lds_pilot/logs/eval_%j.out
#SBATCH --error=anticipation_lds_pilot/logs/eval_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p anticipation_lds_pilot/logs anticipation_lds_pilot/outputs/lds

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_lds_pilot:${PYTHONPATH:-}"

python anticipation_lds_pilot/scripts/run_lds_eval.py \
  --methods logra trak \
  --attribution-dir "${ATTR_DIR:-anticipation_lds_pilot/outputs/attribution}" \
  --groundtruth-path "${GT_PATH:-anticipation_lds_pilot/outputs/lds/gt_gen.pt}" \
  --subsets-root "${SUBSETS_ROOT:-anticipation_lds_pilot/outputs/models/subsets}" \
  --pool-indices-file "${POOL_INDICES_FILE:-anticipation_ip_pilot/outputs/selection/seller_indices.txt}" \
  --output-dir "${OUTPUT_DIR:-anticipation_lds_pilot/outputs/lds}" \
  --num-subsets "${NUM_SUBSETS:-30}"
