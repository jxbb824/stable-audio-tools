#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --job-name=songscriber-metrics-test
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

cd /home/xiruij/stable-audio-tools
echo "job is starting on $(hostname)"

python3 test_songscriber_metrics.py
