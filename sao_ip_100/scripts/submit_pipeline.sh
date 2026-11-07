#!/bin/bash
# Submit sao_ip_100 pipeline (50 artists, 50 epochs) as a dependency chain:
#   full -> LOO array -> gen queries -> groundtruth a* -> D-TRAK + analyze
set -euo pipefail
ROOT=/ocean/projects/mth240006p/xjiang6/stable-audio-tools
cd "$ROOT"
mkdir -p sao_ip_100/logs sao_ip_100/outputs

echo "[submit] select 50 sellers..."
PYTHONPATH=sao_ip_pilot/scripts \
  /jet/home/xjiang6/miniconda3/envs/sat-jamendo/bin/python \
  sao_ip_pilot/scripts/select_sellers.py \
  --config sao_ip_100/scripts/pilot.yaml

echo "[submit] building prompts.json..."
PYTHONPATH=sao_ip_pilot/scripts \
  /jet/home/xjiang6/miniconda3/envs/sat-jamendo/bin/python \
  sao_ip_pilot/scripts/build_prompts.py \
  --config sao_ip_100/scripts/pilot.yaml

# Optional: GPU type override for OOM recovery
#   GPU_TYPE=l40s-48  or  GPU_TYPE=h100-80
GPU_TYPE="${GPU_TYPE:-v100-32}"
MEM="${MEM:-48G}"
export GPU_TYPE MEM

rewrite_gpu() {
  local src="$1" tmp
  tmp=$(mktemp)
  sed -E \
    -e "s/--gpus=[^ ]+/--gpus=${GPU_TYPE}:1/" \
    -e "s/--mem=[^ ]+/--mem=${MEM}/" \
    "$src" > "$tmp"
  echo "$tmp"
}

FULL_SCRIPT=$(rewrite_gpu sao_ip_100/scripts/train_full.slurm)
LOO_SCRIPT=$(rewrite_gpu sao_ip_100/scripts/train_loo_array.slurm)
GEN_SCRIPT=$(rewrite_gpu sao_ip_100/scripts/generate_queries.slurm)
GT_SCRIPT=$(rewrite_gpu sao_ip_100/scripts/score_groundtruth.slurm)
DTRAK_SCRIPT=$(rewrite_gpu sao_ip_100/scripts/dtrak_attribution.slurm)

FULL=$(sbatch --parsable "$FULL_SCRIPT")
echo "[submit] full=$FULL gpu=$GPU_TYPE"

LOO=$(sbatch --parsable --dependency=afterok:${FULL} "$LOO_SCRIPT")
echo "[submit] loo_array=$LOO (afterok:$FULL)"

GEN=$(sbatch --parsable --dependency=afterok:${LOO} "$GEN_SCRIPT")
echo "[submit] gen=$GEN (afterok:$LOO)"

GT=$(sbatch --parsable --dependency=afterok:${GEN} "$GT_SCRIPT")
echo "[submit] gt=$GT (afterok:$GEN)"

DTRAK=$(sbatch --parsable --dependency=afterok:${GT} "$DTRAK_SCRIPT")
echo "[submit] dtrak=$DTRAK (afterok:$GT)"

rm -f "$FULL_SCRIPT" "$LOO_SCRIPT" "$GEN_SCRIPT" "$GT_SCRIPT" "$DTRAK_SCRIPT"

cat > sao_ip_100/outputs/pipeline_jobs.json <<EOF
{
  "n_sellers": 50,
  "max_epochs": 50,
  "gpu_type": "$GPU_TYPE",
  "full": "$FULL",
  "loo": "$LOO",
  "gen": "$GEN",
  "gt": "$GT",
  "dtrak": "$DTRAK",
  "submitted_at": "$(date -Is)"
}
EOF

echo
echo "Chain: full($FULL) -> loo($LOO) -> gen($GEN) -> gt($GT) -> dtrak($DTRAK)"
squeue -u "$USER" -o '%.18i %.12P %.16j %.8T %.10M %.6D %R' | head -40
