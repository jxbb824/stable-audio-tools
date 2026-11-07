#!/bin/bash
#SBATCH --job-name=sao50_full
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=12:00:00
#SBATCH --chdir=/ocean/projects/mth240006p/xjiang6/stable-audio-tools
#SBATCH --output=sao_ip_100/logs/full_%j.out
#SBATCH --error=sao_ip_100/logs/full_%j.err

set -euo pipefail
cd /ocean/projects/mth240006p/xjiang6/stable-audio-tools
mkdir -p sao_ip_100/logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# v100-32: batch 4 + accum 2 ~= old effective batch 8
python3 -u sao_ip_pilot/scripts/finetune_sao.py \
  --model-config model_config_freeze_vae.json \
  --dataset-config sao_ip_100/configs/train_dataset.json \
  --pretrained-ckpt-path dataset/pretrained_model/model.ckpt \
  --pretransform-ckpt-path dataset/pretrained_model/vae_model.ckpt \
  --output-dir sao_ip_100/outputs/models \
  --tag 0 \
  --include-paths-file sao_ip_100/outputs/selection/selected_paths_full.txt \
  --seed 42 \
  --batch-size "${BATCH_SIZE:-4}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --precision 16-mixed \
  --accum-batches "${ACCUM_BATCHES:-2}" \
  --max-epochs "${MAX_EPOCHS:-50}" \
  --logger none
