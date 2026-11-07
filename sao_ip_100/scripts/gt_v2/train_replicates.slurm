#!/bin/bash
#SBATCH --job-name=gtv2_rep
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=11:00:00
#SBATCH --array=1-3
#SBATCH --chdir=/ocean/projects/mth260005p/xjiang6/stable-audio-tools
#SBATCH --output=sao_ip_100/logs/gtv2_rep_%A_%a.out
#SBATCH --error=sao_ip_100/logs/gtv2_rep_%A_%a.err
source sao_ip_100/scripts/gt_v2/common.sh
# Same data and settings as model 0 (train_full.slurm); only the seed and output location differ.
python3 -u sao_ip_pilot/scripts/finetune_sao.py \
  --model-config model_config_freeze_vae.json \
  --dataset-config sao_ip_100/configs/train_dataset.json \
  --pretrained-ckpt-path dataset/pretrained_model/model.ckpt \
  --pretransform-ckpt-path dataset/pretrained_model/vae_model.ckpt \
  --output-dir sao_ip_100/outputs/replicates \
  --tag "${SLURM_ARRAY_TASK_ID}" \
  --include-paths-file sao_ip_100/outputs/selection/selected_paths_full_newroot.txt \
  --seed "$((1000 + SLURM_ARRAY_TASK_ID))" \
  --batch-size 4 --num-workers 4 --precision 16-mixed --accum-batches 2 \
  --max-epochs 50 --logger none
