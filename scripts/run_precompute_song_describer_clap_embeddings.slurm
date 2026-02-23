#!/bin/bash
#SBATCH --job-name=precompute-song-describer-clap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd /home/xiruij/stable-audio-tools
mkdir -p logs
export PYTHONPATH="/home/xiruij/stable-audio-tools:${PYTHONPATH:-}"

python3 scripts/precompute_song_describer_clap_embeddings.py \
  --csv-path dataset/small-700/song_describer.csv \
  --audio-dir dataset/small-700/audio \
  --output-path dataset/small-700/song_describer_clap_embeddings.npz \
  --prompt-source training_style \
  --prompt-seed 0 \
  --device cuda
