#!/bin/bash
#SBATCH --job-name=precompute-song-describer-clap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONPATH="/u/xjiang6/stable-audio-tools:${PYTHONPATH:-}"

python3 scripts/precompute_song_describer_clap_embeddings.py \
  --csv-path /projects/bffl/xirui/stable-audio-tools/dataset/small-700/song_describer.csv \
  --audio-dir /projects/bffl/xirui/stable-audio-tools/dataset/small-700/audio \
  --output-path /projects/bffl/xirui/stable-audio-tools/dataset/small-700/song_describer_clap_embeddings.npz \
  --prompt-source training_style \
  --prompt-seed 0 \
  --device cuda
