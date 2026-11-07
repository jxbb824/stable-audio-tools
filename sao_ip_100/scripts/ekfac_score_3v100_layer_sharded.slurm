#!/bin/bash
#SBATCH --job-name=sao50_ekv3
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:3
#SBATCH --mem=180G
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=sao_ip_100/logs/ekfac_score_3v100_layer_sharded_%j.out
#SBATCH --error=sao_ip_100/logs/ekfac_score_3v100_layer_sharded_%j.err

set -euo pipefail

run_shard_task() {
  local start end
  case "${SLURM_PROCID:?SLURM_PROCID is required}" in
    0) start=89;  end=119 ;;
    1) start=119; end=149 ;;
    2) start=149; end=178 ;;
    *) echo "unexpected SLURM_PROCID=${SLURM_PROCID}" >&2; return 2 ;;
  esac

  local shard_dir="$OUTPUT_DIR/shards/layers_$(printf '%03d' "$start")_$(printf '%03d' "$end")"
  mkdir -p "$shard_dir"
  exec python3 -u sao_ip_pilot/scripts/groundtruth_ekfac_attribution.py \
    --model-config model_config_freeze_vae.json \
    --train-dataset-config sao_ip_100/configs/train_dataset.json \
    --query-dataset-config sao_ip_100/configs/query_dataset.json \
    --models-root sao_ip_100/outputs/models \
    --full-model-dir sao_ip_100/outputs/models/0 \
    --output-dir "$shard_dir" \
    --num-workers 2 --device cuda:0 \
    --measurement-f loss --measurement-cfg-dropout-prob 0.0 --damping 1e-8 \
    --kfac-num-samples-for-loss 16 --kfac-batch-size 2 \
    --num-samples-for-loss-scoring 8 --num-samples-for-measurement-scoring 8 \
    --num-loss-batch-aggregations 1 --num-measurement-batch-aggregations 1 \
    --query-batch-size 4 --train-batch-size 4 --query-gradient-compression-bits 4 \
    --linear-layer-start-index "$start" --linear-layer-end-index "$end" \
    --query-count 200 --seed 42 \
    > "$shard_dir/run.out" 2> "$shard_dir/run.err"
}

# The same file is launched by one srun task per GPU.  Using a single
# three-task job step avoids inheriting the outer three-GPU request into three
# separate srun steps, which made those steps serialize on this cluster.
if [[ "${1:-}" == "__run_shard_task" ]]; then
  run_shard_task
  exit $?
fi

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit this job from the repository root}}"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-sao_ip_100/outputs/ekfac_attribution_v100_layer_sharded}"
A_HAT_PATH="${A_HAT_PATH:-sao_ip_100/outputs/a_hat_ekfac_v100_layer_sharded.csv}"
export REPO_ROOT OUTPUT_DIR A_HAT_PATH
mkdir -p \
  sao_ip_100/logs \
  "$OUTPUT_DIR/shards/layers_089_119" \
  "$OUTPUT_DIR/shards/layers_119_149" \
  "$OUTPUT_DIR/shards/layers_149_178" \
  "$(dirname "$A_HAT_PATH")"

source /jet/home/xjiang6/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo
export DIFFUSION_INFLUENCE_ROOT="${REPO_ROOT}/../diffusion-influence"
export PYTHONPATH="${DIFFUSION_INFLUENCE_ROOT}:${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

nvidia-smi --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,nounits --loop=10 > "$OUTPUT_DIR/nvidia_smi.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

srun --exact --kill-on-bad-exit=1 \
  --nodes=1 --ntasks=3 --ntasks-per-node=3 --cpus-per-task=4 \
  --gpus=v100-32:3 --gpus-per-task=v100-32:1 --gpu-bind=single:1 \
  bash "$0" __run_shard_task

python3 -u scripts/merge_ekfac_layer_shards.py \
  --shard-dirs \
    "$OUTPUT_DIR/shards/layers_089_119" \
    "$OUTPUT_DIR/shards/layers_119_149" \
    "$OUTPUT_DIR/shards/layers_149_178" \
  --output-dir "$OUTPUT_DIR" \
  --expected-linear-start 89 --expected-linear-end 178

python3 -u sao_ip_pilot/scripts/aggregate_ekfac_to_sellers.py \
  --score-path "$OUTPUT_DIR/scores_train_x_query.pt" \
  --train-axis-manifest "$OUTPUT_DIR/train_axis_manifest.csv" \
  --query-axis-manifest "$OUTPUT_DIR/query_axis_manifest.csv" \
  --category-manifest sao_ip_100/outputs/selection/category_manifest.csv \
  --output-path "$A_HAT_PATH" \
  --score-column a_hat_ekfac

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT
echo "[ekfac-score-3v100-layer-sharded] done -> $A_HAT_PATH"
