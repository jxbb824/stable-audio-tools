#!/bin/bash
#SBATCH --job-name=mip_base_ref
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=<HOME>/music_ip_pilot/logs/base_ref_%j.out
#SBATCH --error=<HOME>/music_ip_pilot/logs/base_ref_%j.err

# Post-pilot sanity: add Q(y^base, text) reference column to a_star.csv.
# Generates 10 clips from unfinetuned MusicGen + CLAP-scores vs prompts.
# Resolves the LOO-asymmetry diagnostic uncertainty before handover.
# Expected runtime: ~3-5 min once GPU slot available.

set -euo pipefail
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=<DATA_ROOT>/models/hub
export HF_HUB_CACHE=<DATA_ROOT>/models/hub

source <VENV>/bin/activate

echo "=== base_ref start $(date) ==="
echo "node: $(hostname)"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

cd <HOME>/music_ip_pilot
python add_base_reference.py --config <HOME>/music_ip_pilot/pilot.yaml

echo "=== base_ref done $(date) ==="
