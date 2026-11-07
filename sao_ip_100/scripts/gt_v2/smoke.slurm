#!/bin/bash
#SBATCH --job-name=gtv2_smoke
#SBATCH -p GPU-shared
#SBATCH -A mth260005p
#SBATCH --gpus=v100-32:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=5
#SBATCH --time=00:20:00
#SBATCH --chdir=/ocean/projects/mth260005p/xjiang6/stable-audio-tools
#SBATCH --output=sao_ip_100/logs/gtv2_smoke_%j.out
#SBATCH --error=sao_ip_100/logs/gtv2_smoke_%j.err
source sao_ip_100/scripts/gt_v2/common.sh
# 1) Must reproduce the saved losses of model 0 on the first 4 queries exactly.
python3 -u sao_ip_pilot/scripts/groundtruth.py "${GT_COMMON[@]}" --t-strategy uniform \
  --models-root sao_ip_100/outputs/models --model-indices 0 \
  --query-dataset-config sao_ip_100/configs/query_dataset.json --max-examples 4 \
  --output-path $OUT/smoke_uniform.pt
# 2) Exercises the new midpoint code path.
python3 -u sao_ip_pilot/scripts/groundtruth.py "${GT_COMMON[@]}" --t-strategy midpoint \
  --models-root sao_ip_100/outputs/models --model-indices 0 \
  --query-dataset-config sao_ip_100/configs/query_dataset.json --max-examples 4 \
  --output-path $OUT/smoke_midpoint.pt
python3 - <<'PY'
import torch
saved = torch.load('sao_ip_100/outputs/losses_model_x_query.pt', map_location='cpu')[0, :4]
new = torch.load('sao_ip_100/outputs/gt_v2/smoke_uniform.pt', map_location='cpu').reshape(-1)[:4]
mid = torch.load('sao_ip_100/outputs/gt_v2/smoke_midpoint.pt', map_location='cpu').reshape(-1)[:4]
print('saved   :', saved.tolist()); print('new     :', new.tolist()); print('midpoint:', mid.tolist())
d = (saved - new).abs().max().item()
print(f'SMOKE max abs diff vs saved = {d:.3e} ->', 'PASS' if d < 1e-4 else 'FAIL')
PY
