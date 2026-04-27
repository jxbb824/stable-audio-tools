#!/bin/bash
#SBATCH --job-name=mip_phase1
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=<HOME>/music_ip_pilot/logs/phase1_%j.out
#SBATCH --error=<HOME>/music_ip_pilot/logs/phase1_%j.err

# Phase 1 smoke test: read pilot_sellers_candidates.json, decode pre-staged
# MP3s to 32kHz mono WAV clips, emit sellers.json + prompts.json.
# CPU-only; no GPU needed.

set -euo pipefail

export PYTHONUNBUFFERED=1

source <VENV>/bin/activate

echo "=== phase1 setup start $(date) ==="
echo "node: $(hostname)"
python --version

cd <HOME>/music_ip_pilot
CFG=<HOME>/music_ip_pilot/pilot.yaml

python pilot.py --phase setup --config "$CFG"

echo
echo "=== expected outputs ==="
ls -la <DATA_ROOT>/music_ip_pilot/data/sellers.json \
       <DATA_ROOT>/music_ip_pilot/data/prompts.json
echo
echo "=== per-seller WAV counts ==="
for d in <DATA_ROOT>/music_ip_pilot/data/mtg_jamendo_subset/*/; do
    count=$(find "$d" -name '*.wav' | wc -l)
    size=$(du -sh "$d" | cut -f1)
    echo "  $(basename $d): $count WAVs, $size"
done

echo "=== phase1 setup done $(date) ==="
