#!/bin/bash
#SBATCH --job-name=eval-song-describer-clap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/xiruij/stable-audio-tools
mkdir -p logs
export PYTHONPATH="/home/xiruij/stable-audio-tools:${PYTHONPATH:-}"

MODEL_CONFIG="/home/xiruij/stable-audio-tools/model_config.json"
MODEL_CKPT="/home/xiruij/stable-audio-tools/model_unwrap.ckpt"
PRECOMPUTED="/home/xiruij/stable-audio-tools/dataset/small-700/song_describer_clap_embeddings.npz"

python3 scripts/eval_song_describer_checkpoint_clap.py \
  --model-config "$MODEL_CONFIG" \
  --ckpt-path "$MODEL_CKPT" \
  --precomputed-embeddings-path "$PRECOMPUTED" \
  --steps 100 \
  --cfg-scale 7 \
  --gen-batch-size 4 \
  --clap-batch-size 8 \
  --device cuda
