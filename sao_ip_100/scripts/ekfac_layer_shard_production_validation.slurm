#!/bin/bash
#SBATCH --job-name=sao50_ekv_val
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=sao_ip_100/logs/ekfac_layer_shard_production_validation_%j.out
#SBATCH --error=sao_ip_100/logs/ekfac_layer_shard_production_validation_%j.err

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
cd "$REPO_ROOT"

VALIDATION_DIR="sao_ip_100/outputs/ekfac_layer_shard_production_validation/${SLURM_JOB_ID}"
mkdir -p sao_ip_100/logs "$VALIDATION_DIR"

source /jet/home/xjiang6/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo
export DIFFUSION_INFLUENCE_ROOT="${REPO_ROOT}/../diffusion-influence"
export PYTHONPATH="${DIFFUSION_INFLUENCE_ROOT}:${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,nounits --loop=5 > "$VALIDATION_DIR/nvidia_smi.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

run_range() {
  local name="$1"
  local start="$2"
  local end="$3"
  python3 -u sao_ip_pilot/scripts/groundtruth_ekfac_attribution.py \
    --model-config model_config_freeze_vae.json \
    --train-dataset-config sao_ip_100/configs/train_dataset.json \
    --query-dataset-config sao_ip_100/configs/query_dataset.json \
    --models-root sao_ip_100/outputs/models \
    --full-model-dir sao_ip_100/outputs/models/0 \
    --output-dir "$VALIDATION_DIR/$name" \
    --train-count 6 --query-count 6 --num-workers 0 --device cuda:0 \
    --measurement-f loss --measurement-cfg-dropout-prob 0.0 --damping 1e-8 \
    --kfac-num-samples-for-loss 16 --kfac-batch-size 2 \
    --num-samples-for-loss-scoring 8 --num-samples-for-measurement-scoring 8 \
    --num-loss-batch-aggregations 1 --num-measurement-batch-aggregations 1 \
    --query-batch-size 4 --train-batch-size 4 --query-gradient-compression-bits 4 \
    --linear-layer-start-index "$start" --linear-layer-end-index "$end" --seed 42
}

# Two identical unsharded runs measure the irreducible repeat-run variation.
# The merged result then isolates any additional variation caused by sharding.
run_range reference_a_089_095 89 95
run_range reference_b_089_095 89 95
run_range shard_089_092 89 92
run_range shard_092_095 92 95

python3 -u scripts/merge_ekfac_layer_shards.py \
  --shard-dirs "$VALIDATION_DIR/shard_089_092" "$VALIDATION_DIR/shard_092_095" \
  --output-dir "$VALIDATION_DIR/merged_089_095" \
  --expected-linear-start 89 --expected-linear-end 95

python3 -u scripts/validate_ekfac_layer_sharding.py \
  --reference-a "$VALIDATION_DIR/reference_a_089_095/scores_query_x_train.memmap" \
  --reference-b "$VALIDATION_DIR/reference_b_089_095/scores_query_x_train.memmap" \
  --merged "$VALIDATION_DIR/merged_089_095/scores_query_x_train.memmap" \
  --shape 6 6 --output "$VALIDATION_DIR/validation_summary.json"

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT
echo "[ekfac-layer-production-validation] PASS -> $VALIDATION_DIR/validation_summary.json"
