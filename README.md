# Stable Audio Open EKFAC LDS

For the final **What's a Credit Worth?** experiments, retained results, model/data
locations, and recovery instructions, start with [PRESERVATION.md](PRESERVATION.md).
The instructions below describe the older LDS workflow, not the final-paper
50-catalog audio experiment.

Use only these entry points for this workflow:

- `download_mtg_jamendo.py`
- `finetune_full.py`
- `finetune_subset.py`
- `scripts/train_finetune_full.slurm`
- `scripts/train_finetune_subset.slurm`
- `scripts/generate_groundtruth_prompt_json.py`
- `scripts/generate_queries_from_prompt_json.slurm`
- `scripts/run_groundtruth.slurm`
- `scripts/run_groundtruth_ekfac_attribution_fast.slurm`
- `scripts/spearman.py`

## 1. Install Environment

Run from the repository root:

```bash
conda create -n sat-jamendo python=3.10 -y
conda activate sat-jamendo

pip install --upgrade pip

# Change this PyTorch wheel if your CUDA version is different.
pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124

pip install -e .

# Extra packages used by EKFAC/LDS.
pip install scipy diffusers==0.27.2 curvlinops-for-pytorch

# diffusion-influence provides ScoreCalculator. Put it next to this repo.
git clone https://github.com/BrunoKM/diffusion-influence.git ../diffusion-influence
pip install -e ../diffusion-influence --no-deps
```

Check the install:

```bash
python - <<'PY'
import torch
import scipy
import curvlinops
import diffusion_influence
print("torch", torch.__version__)
PY
```

## 2. Download MTG-Jamendo Data

The training config expects this layout:

```text
dataset/small-5000/
  audio/
  raw_30s.tsv
  raw.meta.tsv
```

Download metadata:

```bash
mkdir -p dataset/small-5000

curl -L -o dataset/small-5000/raw_30s.tsv \
  https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw_30s.tsv

curl -L -o dataset/small-5000/raw.meta.tsv \
  https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw.meta.tsv
```

Download audio:

```bash
python download_mtg_jamendo.py \
  --target 5000 \
  --audio-dir dataset/small-5000/audio
```

Check:

```bash
test -f dataset/small-5000/raw_30s.tsv
test -f dataset/small-5000/raw.meta.tsv
find dataset/small-5000/audio -type f -name '*.mp3' | wc -l
```

## 3. Deduplication

Deduplication can be skipped at first. If skipping it, create an empty exclude file because the training dataset config reads this path:

```bash
mkdir -p outputs/dac_dedup
: > outputs/dac_dedup/exclude_paths.txt
```

To rerun deduplication:

```bash
sbatch scripts/run_dac_hist_dedup.slurm
```

Main outputs:

```text
outputs/dac_dedup/exclude_paths.txt
outputs/dac_dedup/similar_pairs.jsonl
outputs/dac_dedup/summary.json
```

## 4. Download Stable Audio Open Weights

Accept the terms for `stabilityai/stable-audio-open-1.0` on Hugging Face, then log in:

```bash
huggingface-cli login
```

Download the two checkpoints used by this workflow:

```bash
mkdir -p dataset/pretrained_model

python - <<'PY'
from huggingface_hub import hf_hub_download

for filename in ("model.ckpt", "vae_model.ckpt"):
    path = hf_hub_download(
        repo_id="stabilityai/stable-audio-open-1.0",
        filename=filename,
        local_dir="dataset/pretrained_model",
    )
    print(path)
PY
```


## 5. Slurm Settings

If running on a different cluster or allocation, edit these lines in the Slurm scripts you use:

```bash
#SBATCH -A mth240006p
#SBATCH -p GPU-shared
#SBATCH --gpus=h100-80:1
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo
```

The EKFAC scripts look for `diffusion-influence` at `../diffusion-influence` by default. If it is somewhere else, set:

```bash
export DIFFUSION_INFLUENCE_ROOT="/path/to/diffusion-influence"
```

## 6. Train Full and Random Half-Subset Models

This LDS setup uses the first 1000 tracks as the candidate pool. The full model trains on all 1000 tracks. Each subset model trains on a random 500-track half. The default subset array trains 30 models, with IDs `0-29`.

Train the full model:

```bash
CANDIDATE_POOL_SIZE=1000 \
MAX_EPOCHS=600 \
sbatch scripts/train_finetune_full.slurm
```

Outputs:

```text
outputs/groundtruth_models/full/model.ckpt
outputs/groundtruth_models/full/candidate_pool.csv
```

Train 30 random half-subset models:

```bash
CANDIDATE_POOL_SIZE=1000 \
TRAIN_SUBSET_SIZE=500 \
MAX_EPOCHS=600 \
LOGGER=none \
sbatch scripts/train_finetune_subset.slurm
```

Outputs:

```text
outputs/groundtruth_models/0/model.ckpt
outputs/groundtruth_models/0/train_index.csv
...
outputs/groundtruth_models/29/model.ckpt
outputs/groundtruth_models/29/train_index.csv
```

Each `train_index.csv` records the 500 candidate-pool indices included in that subset model.

## 7. Generate Query Audio

First generate 200 prompts from the full model candidate pool:

```bash
python scripts/generate_groundtruth_prompt_json.py \
  --candidate-manifest outputs/groundtruth_models/full/candidate_pool.csv \
  --output-path outputs/groundtruth_models/full/prompts_200.json \
  --num-prompts 200 \
  --candidate-pool-size 1000 \
  --seed 0
```

Then generate query audio from the full model:

```bash
sbatch scripts/generate_queries_from_prompt_json.slurm
```

Outputs:

```text
outputs/generated_queries/query_0000.mp3
...
outputs/generated_queries/query_0199.mp3
outputs/generated_queries/prompt_manifest.json
```

If MP3 writing fails on the cluster, generate WAV instead:

```bash
OUTPUT_EXT=wav sbatch scripts/generate_queries_from_prompt_json.slurm
```

## 8. Compute Retraining Ground Truth

Use loss as the measurement:

```bash
F_OBJECTIVE=loss \
OUTPUT_PATH=outputs/groundtruth_models/gt_loss.pt \
MAX_EXAMPLES=200 \
sbatch scripts/run_groundtruth.slurm
```

Outputs:

```text
outputs/groundtruth_models/gt_loss.pt
outputs/groundtruth_models/gt_loss.pt.meta.json
outputs/groundtruth_models/gt_loss.pt.model_ids.txt
outputs/groundtruth_models/gt_loss.pt.query_ids.txt
```

`gt_loss.pt` has shape:

```text
[30 subset models, 200 queries]
```

## 9. Run EKFAC Influence

Start with the fast EKFAC script:

```bash
export DIFFUSION_INFLUENCE_ROOT="$PWD/../diffusion-influence"

MEASUREMENT_F=loss \
OUTPUT_DIR=outputs/groundtruth_models/ekfac_attribution_fast_loss \
TRAIN_COUNT=1000 \
QUERY_COUNT=200 \
sbatch scripts/run_groundtruth_ekfac_attribution_fast.slurm
```

Outputs:

```text
outputs/groundtruth_models/ekfac_attribution_fast_loss/scores_train_x_query.pt
outputs/groundtruth_models/ekfac_attribution_fast_loss/attribution_metadata.json
outputs/groundtruth_models/ekfac_attribution_fast_loss/train_axis_manifest.csv
outputs/groundtruth_models/ekfac_attribution_fast_loss/query_axis_manifest.csv
```

`scores_train_x_query.pt` has shape:

```text
[1000 train candidates, 200 queries]
```

## 10. Compute LDS/Spearman

Each subset model was trained on the included 500 tracks. For loss measurement, aggregate scores over the excluded half:

```bash
python scripts/spearman.py \
  --models-root outputs/groundtruth_models \
  --score-path outputs/groundtruth_models/ekfac_attribution_fast_loss/scores_train_x_query.pt \
  --groundtruth-path outputs/groundtruth_models/gt_loss.pt \
  --output-dir outputs/groundtruth_models/spearman_ekfac_fast_loss \
  --subset-index-filename train_index.csv \
  --subset-aggregation excluded \
  --score-label EKFAC
```

Final outputs:

```text
outputs/groundtruth_models/spearman_ekfac_fast_loss/spearman_summary.json
outputs/groundtruth_models/spearman_ekfac_fast_loss/spearman_per_query.csv
outputs/groundtruth_models/spearman_ekfac_fast_loss/score_subset_query_scores.pt
```
