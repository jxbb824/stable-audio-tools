# Anticipation IP Pilot

This is the SAO-style IP pilot for the symbolic Anticipation/TheoryTab setting, runnable from the `stable-audio-tools` repository root.

## Design

- Training corpus: `anticipation_ip_pilot/data/finetune/train_v2.txt` with 28,000 tokenized songs.
- Query set: first 500 rows of `anticipation_ip_pilot/data/finetune/generated_samples_prompted.txt`.
- Sellers: 200 artists selected from `train_v2` via `anticipation_ip_pilot/data/metadata/tt_data_hashed.csv`.
- Full model: trained on all 28,000 rows.
- LOO model for seller `j`: trained on all rows except the rows belonging to that artist.
- Ground truth: per-seller, per-query NLL difference:

```text
a_star[j, t] = loss(query_t; theta_without_j) - loss(query_t; theta_full)
```

- Attribution methods:
  - `LoGra`: one full model.
  - `TRAK`: optional; currently skipped for the pilot smoke test.

The pilot includes a local copy of the small `anticipation` package used by the scoring scripts. Slurm scripts set `PYTHONPATH=$PWD/anticipation_ip_pilot` so the local copy is used first.

## Run Order

From the repository root:

```bash
python anticipation_ip_pilot/scripts/pilot.py --phase setup
```

Then submit GPU jobs:

```bash
sbatch anticipation_ip_pilot/scripts/train_full_trak.slurm
sbatch anticipation_ip_pilot/scripts/train_loo_array.slurm
sbatch anticipation_ip_pilot/scripts/groundtruth.slurm
sbatch anticipation_ip_pilot/scripts/logra_attribution.slurm
```

If attribution and ground truth already exist, rerun only analysis:

```bash
sbatch anticipation_ip_pilot/scripts/analyze.slurm
```

Set `CONDA_ENV=<env>` if your installed environment is not named `sat-jamendo`.

## Outputs

- `outputs/selection/seller_manifest.csv`
- `outputs/queries/generated_samples_prompted_500.txt`
- `outputs/a_star.csv`
- `outputs/a_hat_logra.csv`, `outputs/I_hat_logra.csv`, `outputs/figures_logra/`
- `outputs/query_level_corr_logra.csv`

## Smoke-Test Time Estimate

On an H100-80GB, 5-epoch full/LOO training at batch size 4 ran at about 14.1 steps/s, using about 13GB GPU memory. A full 35k-step run should take about 42 minutes plus checkpoint saving; the slurm limit is set to 1:15:00.

Ground-truth scoring measured about 18 seconds for one seller over 500 queries using temporary smoke-test models. The full 200-seller run should fit within the 1:15:00 slurm limit.

LoGra batch size 8 entered train-gradient caching stably at about 20GB GPU memory and near-full GPU utilization. The train cache alone extrapolates to about 12 minutes; the slurm limit is set to 2:00:00 to cover IFVP and test attribution.
