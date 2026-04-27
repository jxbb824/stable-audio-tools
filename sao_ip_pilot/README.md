# SAO IP Pilot

Self-contained SAO version of the music-IP pilot. This keeps the collaborator's
seller-level logic, but replaces:

- primary ground truth with fixed-target diffusion loss / NLL delta
- measured attribution with EKFAC influence aggregated from tracks to sellers

The original `music_ip_pilot` CLAP quality path is intentionally not copied here
except as background. This folder keeps only the pieces needed for:

```text
dataset/small-5000 artist selection
-> full and leave-one-seller-out SAO fine-tunes
-> full-model query generation
-> a* from LOO loss deltas
-> a_hat from EKFAC influence
-> I_hat, Spearman, K calibration, calibrated SNR
```

## Model id convention

To avoid changing the existing SAO groundtruth scorer, model folders are numeric:

| Directory | Meaning |
|---|---|
| `outputs/models/0` | full model trained on all selected tracks |
| `outputs/models/1` | LOO model leaving out `seller_01` |
| `outputs/models/2` | LOO model leaving out `seller_02` |
| `outputs/models/22` | LOO model leaving out `seller_22` |

## Main outputs

| File | Meaning |
|---|---|
| `outputs/selection/seller_manifest.csv` | 22 selected artist sellers from `dataset/small-5000` |
| `outputs/selection/category_manifest.csv` | training track to seller/category map |
| `outputs/prompts.json` | fixed prompt set for full-model query generation |
| `outputs/losses_model_x_query.pt` | loss matrix for full + LOO models |
| `outputs/a_star.csv` | `a* = loss_loo - loss_full` |
| `outputs/ekfac_attribution/scores_train_x_query.pt` | sample-level EKFAC influence |
| `outputs/a_hat_ekfac.csv` | seller-level EKFAC attribution |
| `outputs/I_hat.csv` | `I_hat`, Spearman, `K_hat`, calibrated SNR |

## Runbook

Run everything from the repo root with the `sat-jamendo` environment.

### 1. Select artists and prompts

CPU only. This finds 22 artist sellers from `dataset/small-5000`, targeting about
500 selected tracks total, and writes the include-path files used by training.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate sat-jamendo

python3 sao_ip_pilot/scripts/pilot.py --phase setup
```

Check:

```bash
head sao_ip_pilot/outputs/selection/seller_manifest.csv
wc -l sao_ip_pilot/outputs/selection/selected_paths_full.txt
```

### 2. Train full model

```bash
sbatch sao_ip_pilot/scripts/train_full.slurm
```

Output checkpoint:

```text
sao_ip_pilot/outputs/models/0/model.ckpt
```

### 3. Train LOO models

Default array is `1-22`, one job per seller.

```bash
sbatch sao_ip_pilot/scripts/train_loo_array.slurm
```

Output checkpoints:

```text
sao_ip_pilot/outputs/models/{1..22}/model.ckpt
```

### 4. Generate fixed query audio from the full model

```bash
sbatch sao_ip_pilot/scripts/generate_queries.slurm
```

Output:

```text
sao_ip_pilot/outputs/generated_queries/query_*.mp3
sao_ip_pilot/outputs/generated_queries/prompt_manifest.json
```

### 5. Compute NLL/loss ground truth a*

This scores the same full-model generated queries under model `0` and all LOO
models, then writes `a_star.csv`.

```bash
sbatch sao_ip_pilot/scripts/score_groundtruth.slurm
```

Definition:

```text
a*_{j,t} = loss(y_full_t | theta^{-j}) - loss(y_full_t | theta_full)
```

Positive means removing seller `j` makes the full output harder for the model to
explain, so seller `j` contributed positively to that output.

### 6. Compute EKFAC attribution and aggregate to sellers

```bash
sbatch sao_ip_pilot/scripts/ekfac_attribution.slurm
```

This runs the copied SAO EKFAC scorer with `--measurement-f loss`, then sums
sample-level scores over each seller catalog:

```text
a_hat_ekfac[j,t] = sum_{i in D_j} EKFAC_score[i,t]
```

### 7. Estimate I, Spearman, K, calibrated SNR

If Step 6 finished aggregation, this can run on CPU:

```bash
python3 sao_ip_pilot/scripts/estimate_I.py
```

Or as a small job:

```bash
sbatch sao_ip_pilot/scripts/analyze.slurm
```

The SNR column to use is:

```text
SNR_hat_calibrated
```

`SNR_hat_shortcut_unscaled = I/(1-I)` is retained only as the old shortcut for
comparison; it is not safe under multiplicative bias.

## Important defaults

The default config is [scripts/pilot.yaml](scripts/pilot.yaml):

```yaml
n_sellers: 22
tracks_per_seller: 50
num_prompts: 500
```

Seller selection follows the handoff's prolificness-band allocation: low
10-15 tracks gets 8 artists, mid 16-25 gets 8, high 26-50 gets 5, and
extra-high 50+ gets 1. Genres are recorded but not used as constraints here.
If compute is tight, reduce `num_prompts` first.

## What was intentionally omitted

The old MusicGen handoff had CLAP/FAD visualization, base-reference generation,
drop-one-prompt sensitivity, and marginal welfare plots. Those are left out here
to keep the SAO pilot focused. The retained analysis is only:

- `I_hat = Cov(a_hat, a*) / Var(a_hat)`
- Spearman rank correlation
- `K_hat = Cov(a_hat, a*) / Var(a*)`
- calibrated SNR after dividing `a_hat` by `K_hat`
