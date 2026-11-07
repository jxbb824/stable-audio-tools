#!/bin/bash
#SBATCH --job-name=gtv2_repgt
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=02:30:00
#SBATCH --chdir=/ocean/projects/mth260005p/xjiang6/stable-audio-tools
#SBATCH --output=sao_ip_100/logs/gtv2_repgt_%j.out
#SBATCH --error=sao_ip_100/logs/gtv2_repgt_%j.err
source sao_ip_100/scripts/gt_v2/common.sh
R=(--models-root sao_ip_100/outputs/replicates --model-indices 1 2 3 --max-examples 200)
# (a) original queries, OLD timesteps: combines directly with the saved 51-model losses.
python3 -u sao_ip_pilot/scripts/groundtruth.py "${GT_COMMON[@]}" "${R[@]}" --t-strategy uniform \
  --query-dataset-config sao_ip_100/configs/query_dataset.json --output-path $OUT/losses_original_uniform_replicates.pt
# (b) original queries, midpoint timesteps.
python3 -u sao_ip_pilot/scripts/groundtruth.py "${GT_COMMON[@]}" "${R[@]}" --t-strategy midpoint \
  --query-dataset-config sao_ip_100/configs/query_dataset.json --output-path $OUT/losses_original_midpoint_replicates.pt
# (c) artist-only queries, midpoint timesteps.
python3 -u sao_ip_pilot/scripts/groundtruth.py "${GT_COMMON[@]}" "${R[@]}" --t-strategy midpoint \
  --query-dataset-config sao_ip_100/configs/query_dataset_artist_only.json --output-path $OUT/losses_artist_only_midpoint_replicates.pt
