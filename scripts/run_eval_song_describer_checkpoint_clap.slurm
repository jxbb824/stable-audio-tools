#!/bin/bash
#SBATCH --job-name=eval-song-describer-clap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

MODEL_CONFIG="model_config.json"
MODEL_CKPT="model_unwrap.ckpt"
PRECOMPUTED="dataset/small-700/song_describer_clap_embeddings.npz"

python3 scripts/eval_song_describer_checkpoint_clap.py \
  --model-config "$MODEL_CONFIG" \
  --ckpt-path "$MODEL_CKPT" \
  --precomputed-embeddings-path "$PRECOMPUTED" \
  --steps 100 \
  --cfg-scale 7 \
  --gen-batch-size 4 \
  --clap-batch-size 8 \
  --device cuda
