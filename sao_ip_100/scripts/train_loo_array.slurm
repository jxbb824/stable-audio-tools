#!/bin/bash
#SBATCH --job-name=sao50_loo
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=12:00:00
#SBATCH --array=1-50%20
#SBATCH --chdir=/ocean/projects/mth240006p/xjiang6/stable-audio-tools
#SBATCH --output=sao_ip_100/logs/loo_%A_%a.out
#SBATCH --error=sao_ip_100/logs/loo_%A_%a.err

set -euo pipefail
cd /ocean/projects/mth240006p/xjiang6/stable-audio-tools
mkdir -p sao_ip_100/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SELLER_INDEX="${SLURM_ARRAY_TASK_ID}"

python3 -u sao_ip_pilot/scripts/finetune_sao.py \
  --model-config model_config_freeze_vae.json \
  --dataset-config sao_ip_100/configs/train_dataset.json \
  --pretrained-ckpt-path dataset/pretrained_model/model.ckpt \
  --pretransform-ckpt-path dataset/pretrained_model/vae_model.ckpt \
  --output-dir sao_ip_100/outputs/models \
  --tag "$SELLER_INDEX" \
  --include-paths-file "sao_ip_100/outputs/selection/selected_paths_loo_${SELLER_INDEX}.txt" \
  --seed "$((42 + SELLER_INDEX))" \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --precision 16-mixed \
  --accum-batches "${ACCUM_BATCHES:-2}" \
  --max-epochs "${MAX_EPOCHS:-50}" \
  --logger none
