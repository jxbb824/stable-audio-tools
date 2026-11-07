# Shared setup for the gt_v2 jobs (replicate baseline + midpoint timesteps + artist-only queries).
set -euo pipefail
cd /ocean/projects/mth260005p/xjiang6/stable-audio-tools
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo
export PYTHONPATH="${PWD}/sao_ip_pilot/scripts:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
OUT=sao_ip_100/outputs/gt_v2
ORDER=$OUT/query_order.txt
mkdir -p "$OUT"
# Identical to the settings of the saved ground truth, except --t-strategy.
GT_COMMON=(--model-config model_config_freeze_vae.json --batch-size 2 --num-workers 4
           --device cuda:0 --seed 42 --e-seed 0 --K 10 --num-train-steps 1000
           --f loss --cfg-dropout-prob 0.0 --query-order-file "$ORDER")
