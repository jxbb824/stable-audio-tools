#!/bin/bash
#SBATCH --job-name=anticip_ip_logra_identity
#SBATCH -p GPU-shared
#SBATCH -A mth240006p
#SBATCH --gpus=h100-80:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=anticipation_ip_pilot/logs/logra_identity_%j.out
#SBATCH --error=anticipation_ip_pilot/logs/logra_identity_%j.err

set -euo pipefail

cd "${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
mkdir -p anticipation_ip_pilot/logs anticipation_ip_pilot/outputs/attribution/logra

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-sat-jamendo}"

export PYTHONPATH="$PWD/anticipation_ip_pilot:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TMP_SCRIPT="$(mktemp /tmp/score_logra_identity.XXXXXX.py)"
TMP_ESTIMATE_SCRIPT=""
trap 'rm -f "$TMP_SCRIPT" "$TMP_ESTIMATE_SCRIPT"' EXIT

python - <<'PY' "$TMP_SCRIPT"
import sys
from pathlib import Path

src = Path("anticipation_ip_pilot/scripts/score_logra.py").read_text()
src = src.replace('hessian="eFIM"', 'hessian="Identity"')
Path(sys.argv[1]).write_text(src)
PY

python "$TMP_SCRIPT" \
  --train_file "${TRAIN_FILE:-anticipation_ip_pilot/data/finetune/train_v2.txt}" \
  --valid_file "${QUERIES_FILE:-anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt}" \
  --model_path "${MODEL_PATH:-anticipation_ip_pilot/outputs/models/full/full_model}" \
  --output_dir "${ATTR_DIR:-anticipation_ip_pilot/outputs/attribution/logra}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --proj_dim "${PROJ_DIM:-4096}" \
  --damping "${DAMPING:-0.01}" \
  --valid_is_generated \
  --output_filename score_LoGra_identity.pt

python anticipation_ip_pilot/scripts/aggregate_scores_to_sellers.py \
  --score-path "${ATTR_DIR:-anticipation_ip_pilot/outputs/attribution/logra}/score_LoGra_identity.pt" \
  --output-path anticipation_ip_pilot/outputs/a_hat_logra_identity.csv \
  --score-column a_hat_logra_identity

python - <<'PY'
import json
from pathlib import Path

import pandas as pd

a_star = pd.read_csv("anticipation_ip_pilot/outputs/a_star.csv")
a_hat = pd.read_csv("anticipation_ip_pilot/outputs/a_hat_logra_identity.csv")
merged = a_star.merge(a_hat, on=["seller_id", "prompt_id"], how="inner")

rows = []
for query_index, sub in merged.groupby("query_index_x"):
    rows.append(
        {
            "query_index": int(query_index),
            "prompt_id": sub["prompt_id"].iloc[0],
            "n_sellers": int(len(sub)),
            "pearson": float(sub["a_star"].corr(sub["a_hat_logra_identity"], method="pearson")),
            "spearman": float(sub["a_star"].corr(sub["a_hat_logra_identity"], method="spearman")),
        }
    )

out = pd.DataFrame(rows)
out.to_csv("anticipation_ip_pilot/outputs/query_level_corr_logra_identity.csv", index=False)

summary = {
    "method": "logra_identity",
    "num_queries": int(len(out)),
    "mean_pearson": float(out["pearson"].mean()),
    "median_pearson": float(out["pearson"].median()),
    "mean_spearman": float(out["spearman"].mean()),
    "median_spearman": float(out["spearman"].median()),
}

Path("anticipation_ip_pilot/outputs/query_level_corr_logra_identity_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n",
)
print(json.dumps(summary, indent=2))
PY

TMP_ESTIMATE_SCRIPT="$(mktemp /tmp/estimate_I_logra_identity.XXXXXX.py)"
python - <<'PY' "$TMP_ESTIMATE_SCRIPT"
import sys
from pathlib import Path

src = Path("anticipation_ip_pilot/scripts/estimate_I.py").read_text()
src = src.replace(
    "METHODS = {\n",
    'METHODS = {\n    "logra_identity_influence": {\n        "path": "anticipation_ip_pilot/outputs/a_hat_logra_identity.csv",\n        "column": "a_hat_logra_identity",\n    },\n',
)
Path(sys.argv[1]).write_text(src)
PY

python "$TMP_ESTIMATE_SCRIPT" \
  --method logra_identity_influence \
  --output-path anticipation_ip_pilot/outputs/I_hat_logra_identity.csv

python anticipation_ip_pilot/scripts/plot_results.py \
  --i-hat-path anticipation_ip_pilot/outputs/I_hat_logra_identity.csv \
  --a-hat-path anticipation_ip_pilot/outputs/a_hat_logra_identity.csv \
  --output-dir anticipation_ip_pilot/outputs/figures_logra_identity \
  --marginal-welfare-csv anticipation_ip_pilot/outputs/marginal_welfare_numbers_logra_identity.csv \
  --method-label "LoGra Identity"
