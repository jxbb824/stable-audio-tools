# What's a Credit Worth? — experiment archive and recovery guide

This archive covers **What's a Credit Worth? A Market Framework for
Attribution-Aware Compensation in Generative Music**, using the supplied final
ICLR 2027 manuscript `23375_What_s_a_Credit_Worth_A_.pdf` as the scope reference.
It is separate from the NameDrop/SA3 Shapley experiments in the neighboring
`name-free-attribution` repository.

**Status, November 7, 2026:** the preservation branch is published on GitHub.
The public HF model upload is complete: all 473 planned files are present with
matching sizes, and all 182 weight files match their local upload SHA-256 records.
See [the final verification](preservation/public_upload_verification.json).
The private HF dataset is also complete: all 2,901 payload files (10.65 GiB)
passed size and hash verification, and anonymous access is denied. See
[the private-upload verification](preservation/private_upload_verification.json).
The original files remain in place.

## Where to find things

| Location | Contents | Visibility |
| --- | --- | --- |
| [GitHub archive branch](https://github.com/jxbb824/stable-audio-tools/tree/archive/credit-worth-20261107) | Source, experiment configs, scoring/analysis code, lightweight numerical results and figures, this guide | Public; pushed |
| [HF **model** repository `xirui824/credit-worth-models`](https://huggingface.co/xirui824/credit-worth-models) | Trained full and LOCO weights, the five symbolic training-stage checkpoints, three audio replicate weights, model architecture configs | Public; complete and verified |
| [HF **dataset** repository `xirui824/credit-worth-artifacts`](https://huggingface.co/datasets/xirui824/credit-worth-artifacts) | Selected training data, exact generated evaluation examples, selection/axis manifests, raw numerical arrays, fit provenance, submitted manuscript | Private; complete and verified |
| Upstream Hugging Face / GitHub repositories | Downloadable base weights, tokenizers, encoders, source dependencies | Retrieved again when restoring |

Both HF repositories use paths **relative to this repository root**.
Downloading them into the same checkout will reconstruct the expected layout.
The manuscript is the one exception: its archive path is
`paper/submitted_manuscript.pdf`.

The precise file allocation is in [preservation/inventory.csv](preservation/inventory.csv).
Its columns record source path, destination, future repository-relative path,
byte size, purpose, duplicate alias, and SHA-256 where measured.
[storage_summary.json](preservation/storage_summary.json) totals the categories;
[github_paths.txt](preservation/github_paths.txt) lists the files selected for Git.
The inventory records the original allocation and sizes. The final upload
reports record remote integrity checks; every retained public weight and private
payload was checked against its upload hash. The private repository additionally
contains `archive_manifest.jsonl` with per-file sizes and SHA-256 checksums.

The first real upload measurement is saved in
[preservation/transfer_probe.json](preservation/transfer_probe.json):
4,854,060,002 bytes, 29.05 seconds to hash, 49.83 seconds for upload/commit,
79.62 seconds including remote verification. That corresponds to 92.91 MiB/s
for upload/commit and 58.14 MiB/s end to end. Linear extrapolation to the public
inventory gives about 1.95 hours; deduplication, file preparation, and service
variation can change this. Git transferred a 69.15 MiB pack at about 8.48 MiB/s.

## Experiments actually used in the final paper

| Experiment | Directory | Data and outputs | Trained weights to retain |
| --- | --- | --- | --- |
| Main audio experiment, Tables 2–5 and 7, Figures 2–7 | `sao_ip_100/` | **50** catalogs, 1,120 selected tracks, 1,117 scored training examples after exclusions, 200 generated queries | `outputs/models/0/model.ckpt` is full, seed 42; directories `1`–`50` are LOCO, seeds 43–92 |
| Original audio pilot, Table 6 and Appendix G.3 | `sao_ip_pilot/` | 22 catalogs, 475 selected tracks, 200 generated queries | Full model `outputs/models/0/model.ckpt` plus 22 LOCO models |
| Symbolic experiment, Tables 2–4 | `anticipation_ip_pilot/` | 100 artists, 3,633 selected training examples, 500 generated sequences | `outputs/models/loo-selected/seller_*/model.safetensors` and `outputs/models/full-selected/{0..4}/model.safetensors` |
| Audio robustness diagnostics, relevant to the seed rebuttal | `sao_ip_100/outputs/gt_v2/`, `outputs/replicates/`, `estimator_diagnosis/` | Alternate timing grid, artist-only query variant, existing replicate analyses | Three existing replicate `model.ckpt` files |

Despite its name, **`sao_ip_100` contains 50 artists, not 100**. The 22-catalog
pilot is still used in the final appendix and must not be discarded as an
obsolete experiment. It also used a different training duration: 200 epochs
versus 50 for the main audio experiment.

The symbolic full model's root `model.safetensors` and
`full_model/model.safetensors` are byte-identical to checkpoint `4/model.safetensors`.
Their SHA-256 was checked:
`24e6ac41eba15d32b6958add921627e5b6e9c5997d5b1de6cac01d946e1195da`.
Only one copy is retained on HF; restore the two aliases before using scripts
that expect the final model in those other locations. Keep epochs 0–4 because
the attribution workflow may need the checkpoint ensemble, not just the final fit.

## Storage budget

All sizes below are logical bytes; GiB means 2^30 bytes, not decimal GB.

| Payload | Approximate size |
| --- | ---: |
| 51 main-audio checkpoints | 230.56 GiB |
| 23 audio-pilot checkpoints | 103.98 GiB |
| 100 symbolic LOCO plus 5 distinct full-model stage checkpoints | 50.11 GiB |
| 3 existing audio replicate checkpoints | 13.56 GiB |
| **Public trained-model payload** | **398.20 GiB** |
| Selected training audio: 1,137 distinct files across both audio experiments | 9.60 GiB |
| Private data, examples, manifests, and numerical arrays combined | About 10.65 GiB, including that audio |
| Git source and lightweight result payload | About 165 MiB, plus the small preservation files |

The 20,000-track and 5,000-track source corpora are not being copied wholesale.
Their selected audio overlaps: retain the union once, preserving the original
relative paths. Keep the original symbolic `train_v2.txt` and `test_v2.txt`
because saved selection files address their original row indices. Replacing
them with a renumbered 3,633-line subset would silently break those indices.

GitHub warns above 50 MiB per file and blocks ordinary Git files above 100 MiB.
The selected Git payload has no model weights, training recordings, or large
binary feature arrays. HF's free private allowance is currently 100 GB, enough
for this private payload if the account has sufficient unused quota. Public
storage on a free HF account is best-effort. The current public and private
payloads have both been uploaded and verified.
Sources: [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github),
[HF storage limits](https://huggingface.co/docs/hub/storage-limits).

## What is preserved and what is deliberately excluded

Retain:

- The current working source, including the previously untracked `sao_ip_100`
  scripts/configs, diagnostic scripts, and fixes to shared scoring code.
- The original full/LOCO loss measurements, `a_star.csv`, per-method `a_hat*.csv`,
  per-artist `I_hat*.csv`, baseline scores, uncertainty tables, and existing plots.
- Raw score tensors/memmaps, projected features, and array sidecars. Their axis
  manifests are essential; array position alone is not a stable artist/query ID.
- Both original and later D-TRAK configurations, the symbolic grid-search results,
  and `gt_v2` diagnostics, keeping their experiment identities separate.
- Exact generated audio/symbolic evaluation examples and their prompts. New
  generations with the same text cannot substitute for the paper's examples.
- Dataset membership, exclusions, training seeds, resolved model configs, and
  package/source revisions. Full scheduler logs are unnecessary for this purpose.

Do not include:

- The unrelated LDS/random-subset models under `outputs/groundtruth_models*`
  or `anticipation_lds_pilot/outputs*`.
- Superseded symbolic `outputs/models/full` and `outputs/models/loo`; the paper
  uses the **selected** experiment.
- Duplicate final symbolic weights, old ZIP bundles duplicating the new layout,
  the symbolic CLAP rendering cache, GPU telemetry, runtime logs, and temporary
  probe outputs beyond their small diagnostic summaries.
- Unselected source music, installed conda environments, HF download caches,
  authentication files, or `.claude` histories.
- Downloadable SAO base/VAE weights, T5 weights/tokenizer, and the symbolic base
  model. Their sources/revisions are recorded instead. This does **not** remove
  frozen components already inside an existing trained checkpoint: those files
  are retained byte-for-byte, without weight conversion or repacking.

## How the experiment pipeline fits together

```text
Raw music + metadata                     Symbolic training text + artist metadata
         |                                                  |
  catalog selection                              selected artist/row indices
         |                                                  |
full fine-tune + one LOCO fit per catalog, each from the original pretrained model
         |
fixed examples generated by the full model
         |
score every fixed example with full and LOCO models
         |
a_star = loss(LOCO model, example) - loss(full model, same example)
         |
compare with EKFAC / D-TRAK (audio), LoGra / TRAK (symbolic), and baselines
         |
per-query correlations + per-catalog calibration/SNR + contract calculations
```

This paper uses **leave-one-catalog-out loss differences**, not Shapley values.
They are an operational contribution reference, not measured listener utility
or realized revenue.

Key entry points:

| Task | Files |
| --- | --- |
| Main audio configuration and orchestration | `sao_ip_100/scripts/pilot.yaml`, `submit_pipeline.sh`, `train_full.slurm`, `train_loo_array.slurm` |
| Shared audio training | `sao_ip_pilot/scripts/finetune_sao.py` |
| Audio reference losses | `sao_ip_pilot/scripts/groundtruth.py`, `scripts/dtrak_extract_features.py` |
| Audio D-TRAK | `scripts/groundtruth_dtrak_attribution.py`, `scripts/merge_dtrak_feature_shards.py` |
| Audio EKFAC | `sao_ip_pilot/scripts/groundtruth_ekfac_attribution.py`, `scripts/merge_ekfac_layer_shards.py` |
| Calibration and plots | `sao_ip_pilot/scripts/estimate_I.py`, `sao_ip_pilot/scripts/plot_results.py` |
| Symbolic training | `anticipation_ip_pilot/scripts/finetune_trak_checkpoints.py` |
| Symbolic attribution/reference | `anticipation_ip_pilot/scripts/score_logra.py`, `score_trak.py`, `groundtruth.py` |
| Seed/time-grid diagnostics | `sao_ip_100/scripts/gt_v2/`, `estimator_diagnosis/` |
| Lightweight independent re-analysis | `preservation/summarize_results.py` |

## Verified numerical results and their provenance

`preservation/summarize_results.py` re-reads the saved catalog/query tables and
recomputes the four metrics in **Tables 2 and 6**, joining by catalog and prompt
ID rather than assuming row order. Its outputs are:

- [results/paper_metrics.csv](preservation/results/paper_metrics.csv): all 15
  experiment/method rows, with exact source paths and paper-rounding checks.
- [results/per_catalog.csv](preservation/results/per_catalog.csv): calibration
  slope, calibrated SNR, reference variance, and contribution to aggregate gain.
- [results/per_query.csv](preservation/results/per_query.csv): Pearson/Spearman
  values, retaining undefined correlations rather than replacing them with zero.
- [results/sources.json](preservation/results/sources.json): source-table hashes.

Main results, rounded as in the manuscript:

| Experiment | Method | Spearman | Pearson | Median calibrated SNR | Welfare gap closed |
| --- | --- | ---: | ---: | ---: | ---: |
| SAO 50 | EKFAC | .007 | .109 | .023 | 9.36% |
| SAO 50 | D-TRAK | .031 | .125 | .009 | 9.18% |
| SAO 50 | CLAP | .046 | .148 | .016 | 4.96% |
| Symbolic 100 | LoGra | .068 | .194 | .017 | 6.48% |
| Symbolic 100 | TRAK | .035 | .087 | .002 | 1.38% |

The main paper's D-TRAK numbers come from **`dtrak_attribution_v100_chunked`**,
not `dtrak_attribution_6h`. Both are retained, but substituting the latter changes
the numbers. The pilot Fingerprint method has 199 defined per-query correlations
because one query has constant scores across catalogs; its other metrics still
use all 200 examples.

These results support the paper's finding that ranking and calibrated precision
can favor different methods, and that measured attribution precision is low in
these settings. They do not establish a causal effect of increasing catalog
count: the two audio experiments differ in training and queries.

The welfare percentage is the paper's **in-sample model-implied gap closed**,
equivalent here to total explained contribution variance divided by total
reference variance. It is not an actual payment percentage. Calibrated SNR is
computed per catalog; it is not the same operation as correlation across
catalogs for a fixed query.

The final PDF's held-out results (Table 7), concentration calculations, and
bootstrap intervals require additional analysis code. That code was **not found
in the experiment source directories** during this preservation pass. The exact
input tables and submitted manuscript are retained in the private archive;
this guide does not claim those later analyses were independently
reproduced. No new training, GPU scoring, or new held-out experiment was run.

## Restore the repository and data

The repositories above are now populated. With access to the private dataset:

```bash
git clone --branch archive/credit-worth-20261107 --recurse-submodules \
  https://github.com/jxbb824/stable-audio-tools.git
cd stable-audio-tools

# Exclude repository-level metadata so it does not overwrite the Git checkout.
hf download xirui824/credit-worth-models --local-dir . --exclude README.md .gitattributes
hf auth login  # use a read token authorized for the private dataset
hf download xirui824/credit-worth-artifacts --repo-type dataset --local-dir . --exclude README.md .gitattributes

# Restore the two identical symbolic final-model aliases.
cp anticipation_ip_pilot/outputs/models/full-selected/4/model.safetensors \
   anticipation_ip_pilot/outputs/models/full-selected/model.safetensors
mkdir -p anticipation_ip_pilot/outputs/models/full-selected/full_model
cp anticipation_ip_pilot/outputs/models/full-selected/4/model.safetensors \
   anticipation_ip_pilot/outputs/models/full-selected/full_model/model.safetensors
```

The source checkout's `.gitmodules` is repaired on this branch to include the
already-pinned D-TRAK dependency. Package revisions and the one local
`diffusion-influence` compatibility patch are in `preservation/environment/`
and `preservation/dependencies/`. The package list is an observed environment
snapshot, not a promise that one unrestricted `pip install -r` resolves on a new
machine.

In particular, restore `diffusion-influence` as a sibling checkout at the recorded
commit and apply `preservation/dependencies/diffusion-influence.patch` before
installing it. That patch preserves the local Python 3.10 compatibility setting.
The recorded `curvlinops` development commit is also significant for EKFAC;
installing whichever version happens to be latest is not equivalent.

Re-download upstream weights using the revisions in
[preservation/upstream_models.json](preservation/upstream_models.json):

- [`stabilityai/stable-audio-open-1.0`](https://huggingface.co/stabilityai/stable-audio-open-1.0):
  `model.ckpt` and `vae_model.ckpt` into `dataset/pretrained_model/`. Access
  requires accepting that repository's model terms with the downloading account.
- [`t5-base`](https://huggingface.co/google-t5/t5-base): restore its encoder and
  tokenizer through Transformers/HF; existing source refers to the `t5-base` ID.
- [`stanford-crfm/music-small-800k`](https://huggingface.co/stanford-crfm/music-small-800k):
  the symbolic initialization, retrieved through Transformers/HF.
- CLAP weights and the small MIDI rendering soundfont are downloadable dependencies.

The local SAO initialization and VAE files were hashed and matched the upstream
LFS SHA-256 values at the recorded revision. They can therefore be omitted
without losing the identity of those two initialization files. Existing T5 and
symbolic cache revisions are recorded alongside the upstream revisions.

Historical manifests/configs and Slurm files contain machine-specific absolute
paths. Map their old `stable-audio-tools` root to the new checkout and update
Slurm account/partition/working-directory directives before running them. Do not
blindly launch an old Slurm file unchanged. Keep the *relative* data paths,
selection order, query order, model IDs, and tensor-axis manifests intact.

The metadata loader `stable_audio_tools/configs/dataset_configs/custom_metadata/custom_md_fma.py`
prefers `dataset/small-5000/raw_30s.tsv` and `raw.meta.tsv` when present, including
in the later experiment. Both 5k and 20k metadata tables are therefore retained.
Dropping the 5k metadata because the main experiment uses 20k audio paths can
change training prompt construction.

After installing compatible NumPy/Pandas/SciPy, the statistical snapshot needs
only the Git tables:

```bash
python preservation/summarize_results.py
```

## The five-run matched-seed rebuttal check

Use the **50-catalog** experiment. Keep its existing full model at seed 42 and
the existing 200 generated examples fixed. Randomly select five catalogs with a
recorded selection seed before examining their results. Retrain only those five
LOCO models with training seed 42, using the original pretrained initialization
and unchanged 50-epoch training settings (per-device batch 4, accumulation 2,
mixed FP16, original learning-rate/weight-decay settings in the resolved config).
Do not initialize them from the already fine-tuned full model.

Score the same examples with the original loss protocol: ten timesteps
`0, 100, ..., 900`, `seed=42`, `e_seed=0`, batch 2, conditioning dropout zero,
and the saved query order. This is distinct from the later `gt_v2` midpoint-grid
diagnostic. Compare new matched-seed loss differences with the existing
different-seed differences, then recompute attribution correlations and
calibrated SNR using the existing method scores. Identical seeds on different
hardware do not guarantee bitwise-identical training.

The five additional model files would add approximately **22.60 GiB**. They do
not exist yet and are not counted as archived. The existing three full-model
replicates are useful diagnostics but are not replacements for five LOCO fits.
