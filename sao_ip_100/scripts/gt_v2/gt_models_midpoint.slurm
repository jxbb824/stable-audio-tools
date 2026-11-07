#!/bin/bash
#SBATCH --job-name=gtv2_mid
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=03:30:00
#SBATCH --array=0-2
#SBATCH --chdir=/ocean/projects/mth260005p/xjiang6/stable-audio-tools
#SBATCH --output=sao_ip_100/logs/gtv2_mid_%A_%a.out
#SBATCH --error=sao_ip_100/logs/gtv2_mid_%A_%a.err
# Usage: sbatch --export=ALL,QUERY_SET=original|artist_only gt_models_midpoint.slurm
source sao_ip_100/scripts/gt_v2/common.sh
QUERY_SET="${QUERY_SET:?set QUERY_SET=original or artist_only}"
case "$QUERY_SET" in
  original)    QCFG=sao_ip_100/configs/query_dataset.json ;;
  artist_only) QCFG=sao_ip_100/configs/query_dataset_artist_only.json ;;
  *) echo "bad QUERY_SET"; exit 1 ;;
esac
START=$((SLURM_ARRAY_TASK_ID * 17)); END=$((START + 16))   # shards 0-16, 17-33, 34-50
python3 -u sao_ip_pilot/scripts/groundtruth.py "${GT_COMMON[@]}" --t-strategy midpoint \
  --models-root sao_ip_100/outputs/models --model-indices $(seq -s ' ' $START $END) \
  --query-dataset-config "$QCFG" --max-examples 200 \
  --output-path "$OUT/losses_${QUERY_SET}_midpoint_models_${START}_${END}.pt"
