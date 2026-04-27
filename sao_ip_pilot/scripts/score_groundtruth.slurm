#!/bin/bash
#SBATCH --job-name=sao_ip_gt
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/sao_ip_groundtruth_%j.out
#SBATCH --error=logs/sao_ip_groundtruth_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

N_SELLERS="${N_SELLERS:-22}"
MODEL_INDICES="${MODEL_INDICES:-$(seq -s ' ' 0 "$N_SELLERS")}"

python3 -u sao_ip_pilot/scripts/groundtruth.py \
  --models-root "${MODELS_ROOT:-sao_ip_pilot/outputs/models}" \
  --model-indices $MODEL_INDICES \
  --model-config "${MODEL_CONFIG:-model_config_freeze_vae.json}" \
  --query-dataset-config "${QUERY_DATASET_CONFIG:-sao_ip_pilot/configs/query_dataset.json}" \
  --output-path "${OUTPUT_PATH:-sao_ip_pilot/outputs/losses_model_x_query.pt}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --device "${DEVICE:-cuda:0}" \
  --seed "${SEED:-42}" \
  --e-seed "${E_SEED:-0}" \
  --K "${K:-10}" \
  --t-strategy "${T_STRATEGY:-uniform}" \
  --num-train-steps "${NUM_TRAIN_STEPS:-1000}" \
  --f loss \
  --cfg-dropout-prob "${CFG_DROPOUT_PROB:-0.0}" \
  --max-examples "${MAX_EXAMPLES:-500}" \
  ${EXTRA_ARGS:-}

python3 -u sao_ip_pilot/scripts/compute_a_star_from_losses.py \
  --loss-path "${OUTPUT_PATH:-sao_ip_pilot/outputs/losses_model_x_query.pt}"
