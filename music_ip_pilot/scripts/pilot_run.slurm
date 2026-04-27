#!/bin/bash
#SBATCH --job-name=music_ip_pilot
#SBATCH --partition=general
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=04:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --output=<HOME>/music_ip_pilot/logs/pilot_%j.out
#SBATCH --error=<HOME>/music_ip_pilot/logs/pilot_%j.err

# Music IP Pilot — one-job end-to-end run
# Runs all 6 phases: setup -> baseline -> loo -> generate -> attribution -> analyze
# Scope: N=3 sellers, T=10 prompts, LoRA fine-tunes. Expected ~75-120 min.
# Time limit is 4h for safety (streaming MTG-Jamendo can be slow).
#
# RESUMABILITY: every phase is idempotent — if this job times out or fails
# mid-pipeline, simply resubmit the same script. Phases that already wrote
# their outputs (checkpoints, CSVs, WAVs) are automatically skipped.

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

# Point HF caches at the shared models hub on data disk
export HF_HOME=<DATA_ROOT>/models/hub
export HF_HUB_CACHE=<DATA_ROOT>/models/hub

# Activate the isolated music-IP venv
source <VENV>/bin/activate

echo "=== Music IP Pilot ==="
echo "Start: $(date)"
echo "Node:  $(hostname)"
echo "GPU:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Python: $(python --version)"
python -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"

cd <HOME>/music_ip_pilot
CFG=<HOME>/music_ip_pilot/pilot.yaml

echo ""
echo "--- Phase 1/6: setup ---"
python pilot.py --phase setup --config "$CFG"

echo ""
echo "--- Phase 2/6: baseline (full fine-tune) ---"
python pilot.py --phase baseline --config "$CFG"

echo ""
echo "--- Phase 3/6: LOO fine-tunes ---"
python pilot.py --phase loo --config "$CFG"

echo ""
echo "--- Phase 4/6: generation ---"
python pilot.py --phase generate --config "$CFG"

echo ""
echo "--- Phase 5/6: attribution (a*, a_hat) ---"
python pilot.py --phase attribution --config "$CFG"

echo ""
echo "--- Phase 6/6: analyze (I_hat + plots) ---"
python pilot.py --phase analyze --config "$CFG"

echo ""
echo "=== Pilot complete: $(date) ==="
