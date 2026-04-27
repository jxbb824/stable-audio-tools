#!/bin/bash
#SBATCH --job-name=mip_dl_mtg
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --output=<HOME>/music_ip_pilot/logs/dl_mtg_%j.out
#SBATCH --error=<HOME>/music_ip_pilot/logs/dl_mtg_%j.err

# Stage pilot MP3s from MTG-Jamendo Freesound CDN to /data disk.
# Primary picks only (~67 files, ~330 MB). Idempotent.

set -euo pipefail

export PYTHONUNBUFFERED=1

source <VENV>/bin/activate

echo "=== dl_mtg start $(date) ==="
echo "node: $(hostname)"

python <HOME>/music_ip_pilot/download_mtg.py \
    --candidates <HOME>/music_ip_pilot/data/stats/pilot_sellers_candidates.json \
    --raw-dir    <DATA_ROOT>/music_ip_pilot/data/mtg_jamendo_raw \
    --mode       primary

echo "=== dl_mtg done $(date) ==="
du -sh <DATA_ROOT>/music_ip_pilot/data/mtg_jamendo_raw
find <DATA_ROOT>/music_ip_pilot/data/mtg_jamendo_raw -name '*.mp3' | wc -l
