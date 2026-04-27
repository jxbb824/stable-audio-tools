#!/bin/bash
#SBATCH --job-name=mip_catalog_emb
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=<HOME>/music_ip_pilot/logs/catalog_emb_%j.out
#SBATCH --error=<HOME>/music_ip_pilot/logs/catalog_emb_%j.err

# CPU-only precompute of seller catalog CLAP embeddings + geometry stats
# (distinctiveness_to_grand, distinctiveness_to_nearest, within-catalog coherence).
#
# These feed (a) Phase 5b (run_attribution.py reuses catalog centroids), and
# (b) the "cheap I proxy" paper story (regress I_j on distinctiveness + coherence).
#
# Expected runtime: 2-5 minutes on CPU for N=3 sellers, 30 WAVs, CLAP inference.

set -euo pipefail

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME=<DATA_ROOT>/models/hub
export HF_HUB_CACHE=<DATA_ROOT>/models/hub

source <VENV>/bin/activate

echo "=== catalog_emb start $(date) ==="
echo "node: $(hostname)"
python --version

cd <HOME>/music_ip_pilot
CFG=<HOME>/music_ip_pilot/pilot.yaml

python precompute_catalog_embeddings.py --config "$CFG"

echo "=== catalog_emb done $(date) ==="
