# Music IP Pilot — Implementation Plan

A pipeline that estimates per-seller attribution informativeness $\widehat{\mathcal{I}}_j$ for music generation models using a leave-one-catalog-out (LOO) protocol on MusicGen-small. The pilot validates the pipeline end-to-end at $N=3$ sellers, $T=10$ prompts; scale-up to $N=22$, $T=200$ is a config change.

## Folder structure (this is the entire deliverable)

```
music_ip_pilot/
├── README.md                              ← you are here
├── scripts/                               ← all .py + .yaml + .slurm, flat
│   ├── pilot.py                           # phase dispatcher
│   ├── build_sellers.py                   # Phase 1
│   ├── finetune_musicgen.py               # Phases 2–3
│   ├── generate.py                        # Phase 4
│   ├── precompute_catalog_embeddings.py   # Phase 4.5
│   ├── compute_a_star.py                  # Phase 5a
│   ├── add_base_reference.py              # Phase 5a-supplement
│   ├── run_attribution.py                 # Phase 5b
│   ├── estimate_I.py                      # Phase 6
│   ├── plot_results.py                    # Phase 6
│   ├── common.py                          # shared helpers
│   ├── download_mtg.py                    # Phase 0.5 (MP3 acquisition)
│   ├── eda_groupings.py                   # local EDA
│   ├── generate_fallbacks.py              # local seller selection
│   ├── drop_one_prompt_sensitivity.py     # post-hoc diagnostic
│   ├── marginal_welfare_plot.py           # post-hoc plot
│   ├── pilot.yaml                         # config (edit paths!)
│   ├── pilot_run.slurm                    # Phases 2–6 wrapper (GPU)
│   ├── phase1_setup.slurm                 # Phase 1 wrapper (CPU)
│   ├── precompute_catalog_embeddings.slurm # Phase 4.5 wrapper (CPU)
│   ├── add_base_reference.slurm           # Phase 5a-supp wrapper (GPU)
│   └── download_mtg.slurm                 # Phase 0.5 wrapper (CPU)
└── data/
    ├── mtg_jamendo_metadata/              ← empty, see README inside
    │   └── README.md                      # one-liner curl to fetch autotagging.tsv
    ├── stats/                             ← EDA outputs (small JSONs/CSVs)
    │   ├── pilot_sellers_candidates.json  # primary + fallback artists per tier
    │   ├── artists_summary.csv            # full per-artist table
    │   └── grouping_summary.json          # pool-level stats
    ├── figures/                           ← EDA plots
    │   ├── prolificness.png
    │   ├── genre.png
    │   ├── coherence.png
    │   └── pilot_candidates.png
    └── outputs/
        ├── results/                       ← pilot artifacts (cluster-fetched)
        │   ├── I_hat.csv                  # headline numbers
        │   ├── I_hat_sensitivity.csv      # drop-one-prompt jackknife
        │   ├── a_star.csv                 # ground-truth attribution (with q_base)
        │   ├── a_hat_embed.csv            # measured attribution
        │   ├── catalog_stats.json         # Phase 4.5 stats
        │   ├── catalog_embeddings.npz     # Phase 4.5 numpy archive
        │   ├── figures/
        │   │   ├── I_scatter.pdf          # per-seller I_hat + bootstrap CIs
        │   │   └── ranking_comparison.pdf # I_hat rank vs Spearman rank
        │   └── audio_samples/             # 5 WAVs for sanity-listening
        │       ├── full_prompt_1.wav
        │       ├── loo_{1,2,3}_prompt_1.wav
        │       └── base_prompt_1.wav
        └── simulation/
            ├── marginal_welfare_numbers.csv
            └── figures/
                └── marginal_welfare_vs_snr.{pdf,png}  # closed-form curve + 3 measured points
```

## What is NOT in this handover (you must fetch / produce these)

To keep the handover small (currently ~3 MB, 47 files), we deliberately exclude bulky artifacts. Most are reproducible by running the pipeline; two need to be fetched once. Subdirectories that are intentionally empty have a `README.md` inside explaining what should land there.

| Artifact | Size | Where it should land | How to get it |
|---|---|---|---|
| **MTG-Jamendo metadata `autotagging.tsv`** | ~10 MB | `data/mtg_jamendo_metadata/autotagging.tsv` | One-time `curl` — see [`data/mtg_jamendo_metadata/README.md`](data/mtg_jamendo_metadata/README.md). Only needed if rerunning EDA (`eda_groupings.py` / `generate_fallbacks.py`). |
| **HuggingFace base models** (MusicGen-small ~1.5 GB, CLAP-htsat-unfused ~600 MB) | ~2.1 GB | `<DATA_ROOT>/models/hub/` on the cluster | Pre-fetch once with the Python snippet under [Before you run anything](#before-you-run-anything). |
| **Python venv** with pinned dependencies | ~5 GB | `<VENV>` on the cluster | `python3 -m venv` + `pip install` — see [Before you run anything](#before-you-run-anything). |
| **Raw MP3s** (pilot: 67 tracks; scale-up: ~500) | ~570 MB pilot / ~2.5 GB scale-up | `<DATA_ROOT>/music_ip_pilot/data/mtg_jamendo_raw/{prefix}/{track_id}.mp3` | Run `download_mtg.slurm` on the cluster (Step 3 of the cookbook). |
| **Phase 1 staged WAVs** (32 kHz mono clips; pilot: 30) | ~150 MB | `<DATA_ROOT>/music_ip_pilot/data/mtg_jamendo_subset/{seller_id}/{track_id}.wav` | Run `phase1_setup.slurm` (Step 4). |
| **LoRA adapters** (1 full + N LOO + 0 base) | ~50 MB × (N+1) | `<DATA_ROOT>/music_ip_pilot/checkpoints/{tag}/adapter/` | Run `pilot_run.slurm` (Step 6). Idempotent; resumable. |
| **Full generated audio** (T × (N+1) WAVs; pilot: 40 + base: 10) | ~30 MB pilot | `<DATA_ROOT>/music_ip_pilot/outputs/generated_audio/{tag}/{prompt_id}.wav` | Produced by `pilot_run.slurm` (Step 6) and `add_base_reference.slurm` (Step 7). **The handover ships 5 sample WAVs (1 per tag) under [`data/outputs/results/audio_samples/`](data/outputs/results/audio_samples/) for the manual sanity check (Step 10); the full 40+10 set is on the cluster only.** |
| **Stage / scp helper scripts** for fetching cluster results back to local | tiny | wherever you keep ad-hoc scripts | Pattern in [Step 8 of the cookbook](#step-8--fetch-results-back-to-local). |

**Already in the handover** (no action needed): EDA outputs (`data/stats/`, `data/figures/`), pilot result CSVs (`data/outputs/results/`), catalog embeddings + stats, 5 sample WAVs, the marginal welfare plot, all scripts/configs/SLURM wrappers, and this README. **You only need the cluster artifacts above if you intend to re-run any pipeline phase.** If you're just reviewing the pilot's findings, the result CSVs and figures already in the handover are sufficient.

---

## Before you run anything

1. **Edit cluster paths.** All `*.slurm` files and `pilot.yaml` use placeholders `<HOME>`, `<DATA_ROOT>`, `<VENV>`. Replace them with your cluster paths via:
   ```bash
   cd scripts/
   for f in *.slurm pilot.yaml; do
       sed -i '' \
           -e "s|<HOME>|/your/cluster/home|g" \
           -e "s|<DATA_ROOT>|/your/cluster/data|g" \
           -e "s|<VENV>|/your/cluster/home/music_ip_venv|g" \
           "$f"
   done
   ```
2. **Set up the Python environment** on the cluster:
   ```bash
   python3 -m venv $VENV
   source $VENV/bin/activate
   pip install 'torch>=2.10' 'transformers>=4.50' 'peft>=0.10' 'datasets<4' \
               'librosa' 'soundfile' 'pandas' 'numpy' 'matplotlib' 'seaborn' \
               'pyyaml' 'tqdm' 'accelerate'
   ```
3. **Pre-fetch base models** to your `models_hub` (saves repeated downloads):
   ```bash
   python -c "from transformers import AutoModel, AutoProcessor; \
              AutoModel.from_pretrained('facebook/musicgen-small'); \
              AutoModel.from_pretrained('laion/clap-htsat-unfused')"
   ```

## Table of contents

- [Conceptual overview](#conceptual-overview)
- [What we built](#what-we-built)
- [How to run it (cookbook)](#how-to-run-it-cookbook)
- [Scaling to N=22, T=200](#scaling-to-n22-t200)
- [Config reference](#config-reference)
- [Output artifact reference](#output-artifact-reference)
- [Notation & symbols](#notation--symbols)
- [Quality metric specification](#quality-metric-specification)
- [Pilot results](#pilot-results)
- [Sampling design](#sampling-design)

---

## Conceptual overview

### The quantity we measure

A generative-music platform is trained on the union of $N$ sellers' catalogs $\{D_1, \dots, D_N\}$. For each generated output at prompt $r_t$, we want to assign a **per-seller, per-output attribution score** $a^*_{j,t}$ that captures *how much seller $j$'s catalog contributed* to that output. The platform doesn't have access to the true $a^*_{j,t}$ at deployment time — it only has a noisy estimator $\hat{a}_{j,t}$ from some attribution method. The framework's central informational primitive is the **per-seller informativeness**:

$$\mathcal{I}_j \;=\; \frac{\mathrm{Cov}_t\bigl(\hat{a}_{j,t}\,\pi_t,\; a^*_{j,t}\,\pi_t\bigr)}{\mathrm{Var}_t\bigl(\hat{a}_{j,t}\,\pi_t\bigr)}$$

where $\pi_t$ is the platform's gross revenue at period $t$. $\mathcal{I}_j$ is the population regression slope of $a^* \pi$ on $\hat{a} \pi$ — exactly the signal-extraction quantity that appears in Holmström's informativeness principle. A method whose $\hat{a}$ tracks $a^*$ well (in covariance, not just rank) yields high $\mathcal{I}_j$. **Pilot convention**: $\pi_t \equiv 1$ (uniform demand); scale-up calibrates $\pi_t$ from a demand model.

Under a Gaussian-noise special case, $\mathcal{I} = \mathrm{SNR} / (1 + \mathrm{SNR})$, so $\mathrm{SNR}_j = \mathcal{I}_j / (1 - \mathcal{I}_j)$.

### Operationalizing each piece

**Ground truth $a^*_{j,t}$.** Define as the quality drop from removing seller $j$'s catalog:

$$a^*_{j,t} \;=\; Q\bigl(y^{\text{full}}_t,\, r_t\bigr) \;-\; Q\bigl(y^{-j}_t,\, r_t\bigr)$$

where $y^{\text{full}}_t$ is generated by the model trained on all $N$ catalogs, $y^{-j}_t$ by the model trained on all-but-$D_j$, and $Q(y, r)$ is a cardinal quality metric. Cost: $N+1$ retrainings (1 full + $N$ leave-one-out). We do leave-one-**catalog**-out, not leave-one-**song**-out, which reduces the retraining count from $O(\text{songs})$ to $O(N)$ — feasible at $N \leq 30$ even on a single GPU.

**Quality metric $Q$.** Pilot uses CLAP audio-text cosine similarity:

$$Q(y, r) \;=\; \cos\bigl(\mathrm{CLAP}_{\text{audio}}(y),\; \mathrm{CLAP}_{\text{text}}(r)\bigr)$$

with L2-normalised embeddings. Scale-up alternatives: FAD (Fréchet Audio Distance), or human-listener ratings on a small subset.

**Measured attribution $\hat{a}^{(m)}_{j,t}$.** A cheap proxy that doesn't require the $N+1$ retrainings. The pilot uses **catalog-centroid CLAP cosine**:

$$\hat{a}^{\text{embed}}_{j,t} \;=\; \cos\bigl(\mathrm{CLAP}_{\text{audio}}(y^{\text{full}}_t),\; c_j\bigr), \quad c_j = \mathrm{normalize}\Bigl(\textstyle\frac{1}{|D_j|} \sum_{x \in D_j} \mathrm{CLAP}_{\text{audio}}(x)\Bigr)$$

Other methods to plug in: influence functions (Koh & Liang 2017), TRAK (Park et al. 2023), Data Shapley (Ghorbani & Zou 2019), Data-OOB (Kwon & Zou 2023). Each writes its own $\hat{a}^{(m)}_{j,t}$ column under the same schema.

**Per-seller estimator $\widehat{\mathcal{I}}_j$.** Sample analog of $\mathcal{I}_j$ over the $T$ prompts:

$$\widehat{\mathcal{I}}_j \;=\; \frac{\widehat{\mathrm{Cov}}_t(\hat{a}_{j,t},\, a^*_{j,t})}{\widehat{\mathrm{Var}}_t(\hat{a}_{j,t})}$$

Reported with 95% percentile bootstrap CIs over the $T$ prompts. We also report **Spearman rank correlation** (ordinal cross-check) and **MSE** $= \frac{1}{T}\sum_t (\hat{a}_{j,t} - a^*_{j,t})^2$ (absolute calibration check — high $\widehat{\mathcal{I}}_j$ can coexist with poorly-calibrated $\hat{a}$ because the slope is scale-invariant).

### How sellers are sampled

Pool: artists in MTG-Jamendo's autotagging split, filtered to **coherent catalogs** ($\geq 70\%$ of tags match the artist's primary genre) with $\geq 10$ tracks. The filter yields 133 eligible artists out of 3,565.

- **Pilot ($N=3$): stratified-purposive.** One artist per prolificness tercile (low / mid / high). Within tercile: pick the artist closest to tercile-median tracks; tie-break by coherence; require distinct primary genres across the three picks. Justified only at $N=3$ where hard constraints saturate the cells.
- **Scale-up ($N \geq 20$): stratified random within cell, seeded by `cfg.seed`.** The same rule scales to any $N$ — bump `n_sellers` in `pilot.yaml` and re-run. Cell allocation in [Sampling design](#sampling-design) below shows one instantiation at $N=22$ (paper-grade target chosen for compute fit + regression degrees-of-freedom); cell counts scale proportionally for larger $N$ up to the coherent-pool ceiling (~133 artists from MTG-Jamendo under our $\geq 10$-track / $\geq 0.7$-coherence filter). Within each cell, draw without replacement; seed from `pilot.yaml` makes selection reproducible.

  **Concrete scaling tiers** (set `n_sellers` to one of these, scale the cell-count table proportionally, and adjust `pilot_run.slurm`'s `--time`):

  | `n_sellers` | Pool used | GPU-time per pipeline run | When to use |
  |---:|---|---|---|
  | 22 | 17% of coherent pool | ~6 GPU-hours | Paper-grade scale-up; tight CIs, full Strategy A coverage |
  | 50 | 38% | ~15 GPU-hours | Large empirical study; rich heterogeneity sweeps |
  | 100 | 75% | ~30 GPU-hours | Near-full pool; weak cells (e.g., 50+ tracks) tap fallbacks |
  | 133 | 100% | ~40 GPU-hours | Whole coherent pool; ceiling without changing the corpus |
  | 200+ | requires expansion | 80+ GPU-hours | Either relax coherence filter (lets in eclectic artists — changes interpretation) or add another corpus (FMA, MUSDB, MagnaTagATune) |

### How prompts are sampled

We use a **cross-stratified grid**. For seller set $\{S_1, \dots, S_N\}$ and prompt budget $T$:

- $T_{\text{off}} = \max(1, \lfloor T/10 \rfloor)$ off-genre **control** prompts. These probe false-positive attribution.
- $T_{\text{on}} = \lfloor (T - T_{\text{off}}) / N \rfloor$ on-genre prompts per seller. These drive variance in $\hat{a}_{j,t}$ and $a^*_{j,t}$ across $t$, which $\widehat{\mathcal{I}}_j$ requires.

For $N=3$, $T=10$: 3 on-genre × 3 sellers + 1 off-genre = 10. On-genre prompts come from genre-specific natural-language templates (`build_sellers.GENRE_TEMPLATES` — 3 per genre, hand-authored for the pilot).

### How audio is generated

For each model tag $\tau \in \{\text{full}, \text{loo}_1, \dots, \text{loo}_N, \text{base}\}$ and each prompt $t$:

1. Load MusicGen-small base in bf16.
2. For non-`base` tags, attach the LoRA adapter via `PeftModel.from_pretrained(base, adapter_dir)`.
3. Tokenize the prompt text via MusicGen's processor (T5 text encoder).
4. Sample with `do_sample=True`, `temperature=1.0`, `top_k=250`, generating 256 codebook tokens (~5 s at MusicGen's 50 Hz codebook rate).
5. Decode through MusicGen's audio decoder to a 32 kHz mono PCM_16 WAV.

**Prompts and seeds are held constant across tags:**

- The same `prompts.json` (T prompts) is read by every `generate.main()` invocation. The text strings $r_t$ are byte-identical across `full`, all `loo_j`, and `base`.
- `set_all_seeds(cfg["seed"])` is called at the top of every per-tag invocation. So the RNG state at the start of `prompt_t` is identical across all tags.
- Despite identical prompts and identical starting RNG, the generated audio differs across tags **because each tag's LoRA adapter produces different logits at each codebook step**, so the categorical samples diverge from token 1 onward.

This is the intended design: $a^*_{j,t} = Q(y^\text{full}_t, r_t) - Q(y^{-j}_t, r_t)$ isolates the **training-data effect** because everything else (prompt text, generation hyperparameters, sampling RNG state) is held constant. The only difference between $y^\text{full}_t$ and $y^{-j}_t$ is which catalog the LoRA was trained on.

**Scale-up note**: if you adopt Commitment #2's recommendation to generate $k \geq 3$ samples per prompt and average $\hat{a}_{j,t}$ before bootstrap, you'll want to vary the seed *across the $k$ samples* (e.g., `seed = base_seed + sample_idx`) so the $k$ samples are independent draws — but keep the seed-by-prompt-index aligned across tags so the comparison stays clean.

### Design choices and alternatives

| Choice | Pilot value | Alternatives |
|---|---|---|
| Base model | MusicGen-small (300M params) | MusicGen-medium / -large / -melody, Stable Audio, AudioLDM, Suno-class proprietary |
| Fine-tune method | LoRA, rank 16, 500 steps | LoRA at rank 4–8 with early stopping (recommended after pilot's overfit finding); full fine-tune; prefix tuning |
| Quality metric $Q$ | CLAP cosine | FAD vs. reference corpus; MuQ embedding cosine; human listener rating |
| Attribution method $\hat{a}$ | Catalog-centroid CLAP cosine | Influence functions, TRAK, Data Shapley, Data-OOB, TracIn |
| Ground-truth method | Leave-one-catalog-out | Leave-one-song-out (infeasible); 10-subset Banzhaf (more robust to substitutability) |
| Estimator | Per-seller Cov/Var (cardinal slope) | Spearman (rank cross-check); MSE (calibration cross-check) |
| CI method | Percentile bootstrap, 1000 reps | Analytical Gaussian CIs; jackknife (used as sensitivity diagnostic) |
| Prompt set | Hand-authored cross-stratified grid | LLM-generated from tag vocabulary; subsample of MusicCaps |
| Within-cell sampling at scale-up | Stratified random, seeded | Purposive (rejected); CLAP-clustering-based |
| Catalog-geometry "cheap proxy" for $\mathcal{I}$ | Distinctiveness $1 - \cos(c_j, \bar{c})$, coherence | Datamodels $R^2$; LDS from TRAK |

### Two data flows

The system has two clearly separated data flows:

1. **Local (your laptop):** `eda_groupings.py` → `generate_fallbacks.py` → produces the seller-candidates JSON + EDA plots from MTG-Jamendo metadata TSVs. No GPU, no cluster.
2. **Cluster (SLURM):** Phase 0.5 (download MP3s) → Phase 1 (decode + sellers/prompts JSONs) → Phase 4.5 (catalog CLAP embeddings, CPU partition) → Phases 2–6 (LoRA fine-tunes + generation + attribution + analysis, GPU partition) → Phase 5a-supplement (base-model reference, GPU). Result CSVs and PDFs are then copied back to local.

Post-hoc diagnostics (`drop_one_prompt_sensitivity.py`, `marginal_welfare_plot.py`) run locally on the fetched results — no further cluster compute.

---

## What we built

A 7-step pipeline (Phase 0.5 → Phase 6, plus a base-model reference supplement) that produces $\widehat{\mathcal{I}}_j$ per seller with bootstrap CIs.

**Path conventions for the table below:** outputs from cluster phases (0.5–6) are paths relative to `cfg["data_root"]` — e.g., on the cluster, `data/sellers.json` means `<DATA_ROOT>/music_ip_pilot/data/sellers.json`. After fetching back to local (Step 8), they live under `<handover>/data/...` (mirroring the same relative tree). Outputs from local-only scripts are relative to the handover folder root.

| Phase | Script | What it does | Output (relative to `data_root` on cluster, or to handover root locally) |
|---|---|---|---|
| 0.5 | `download_mtg.py` | Download specific MTG-Jamendo MP3s from the Freesound CDN | `data/mtg_jamendo_raw/{prefix}/{track_id}.mp3` |
| 1 | `build_sellers.py` | Decode MP3 → 32 kHz mono WAV; build `sellers.json` + `prompts.json` | `data/sellers.json`, `data/prompts.json`, `data/mtg_jamendo_subset/{seller_id}/{track_id}.wav` |
| 2 | `finetune_musicgen.py` (full) | LoRA fine-tune MusicGen on union of all sellers | `checkpoints/full/adapter/` |
| 3 | `finetune_musicgen.py` (×$N$ LOO) | LoRA fine-tune leaving out each seller | `checkpoints/loo_{j}/adapter/` |
| 4 | `generate.py` | Generate $T \times (N{+}1)$ clips: (full + LOO) × prompts | `outputs/generated_audio/{tag}/{prompt_id}.wav` |
| 4.5 | `precompute_catalog_embeddings.py` | CLAP-embed each seller's catalog tracks; compute centroid + within-catalog coherence + distinctiveness | `outputs/catalog_embeddings.npz`, `outputs/catalog_stats.json` |
| 5a | `compute_a_star.py` | $a^*_{j,t} = Q(y^\text{full}_t, r_t) - Q(y^{-j}_t, r_t)$ via CLAP cosine | `outputs/a_star.csv` |
| 5a-supp | `add_base_reference.py` | Generate clips from unfinetuned base; CLAP-score; append `q_base` column to `a_star.csv` | updated `outputs/a_star.csv`; new `outputs/generated_audio/base/{prompt_id}.wav` |
| 5b | `run_attribution.py` | $\hat{a}^\text{embed}_{j,t} = \cos(\text{CLAP}(y^\text{full}_t), c_j)$ | `outputs/a_hat_embed.csv` |
| 6 | `estimate_I.py` + `plot_results.py` | $\widehat{\mathcal{I}}_j = \text{Cov}(\hat{a}, a^*) / \text{Var}(\hat{a})$, Spearman, MSE, bootstrap CIs, plots | `outputs/I_hat.csv`, `outputs/figures/I_scatter.pdf`, `outputs/figures/ranking_comparison.pdf` |

Standalone diagnostic scripts (run **locally** after the main pipeline finishes; paths relative to handover root):

| Script | What it does | Output |
|---|---|---|
| `drop_one_prompt_sensitivity.py` | Jackknife $\widehat{\mathcal{I}}_j$ across $T$, report per-seller range + most-influential prompt | `data/outputs/results/I_hat_sensitivity.csv` |
| `marginal_welfare_plot.py` | Closed-form $\|\partial\mathcal{L}/\partial\text{SNR}\|$ vs SNR plot, per-seller points overlaid | `data/outputs/simulation/marginal_welfare_numbers.csv`, `data/outputs/simulation/figures/marginal_welfare_vs_snr.{pdf,png}` |

Local-only (pre-pipeline) scripts (paths relative to handover root):

| Script | What it does | Output |
|---|---|---|
| `eda_groupings.py` | Read MTG-Jamendo metadata TSV; per-artist stats; pilot pool plots | `data/stats/grouping_summary.json`, `data/stats/artists_summary.csv`, `data/figures/{prolificness,genre,coherence,pilot_candidates}.png` |
| `generate_fallbacks.py` | From the EDA artist table, pick primary + ranked fallback sellers per prolificness tier | `data/stats/pilot_sellers_candidates.json` |

---

## How to run it (cookbook)

Each step has *command*, *output*, and *success check*. Run sequentially.

### Step 1 — Local EDA (CPU, ~30 s)

```bash
cd scripts
python eda_groupings.py
python generate_fallbacks.py
```

*Output*: `data/stats/pilot_sellers_candidates.json`, `data/stats/artists_summary.csv`, `data/figures/{prolificness,genre,coherence,pilot_candidates}.png`.

*Success check*: open the candidates JSON. Three tiers (low/mid/high), each with a `primary` and 4 `fallbacks`; primary picks should span 3 distinct genres at coherence = 1.0.

### Step 2 — Sync to cluster (CPU, ~30 s)

After replacing the placeholder paths in `*.slurm` and `pilot.yaml` (see [Before you run anything](#before-you-run-anything) above):

```bash
ssh CLUSTER 'mkdir -p <HOME>/music_ip_pilot/{data/stats,logs}'
scp scripts/* CLUSTER:<HOME>/music_ip_pilot/
scp data/stats/pilot_sellers_candidates.json CLUSTER:<HOME>/music_ip_pilot/data/stats/
```

### Step 3 — Stage MP3s (cluster `cpu` partition, ~1.5 min)

```bash
ssh CLUSTER 'cd <HOME>/music_ip_pilot && sbatch download_mtg.slurm'
```

*Output*: `<DATA_ROOT>/music_ip_pilot/data/mtg_jamendo_raw/{prefix}/{track_id}.mp3` — 67 files for primary picks.

*Success check*: log ends with `DONE in ~100s | ok=67 skipped=0 failed=0`.

### Step 4 — Phase 1 setup (cluster `cpu`, ~20 s)

```bash
ssh CLUSTER 'cd <HOME>/music_ip_pilot && sbatch phase1_setup.slurm'
```

*Output*: `sellers.json`, `prompts.json`, 30 WAVs under `mtg_jamendo_subset/seller_{1,2,3}/`.

*Success check*: log ends with 3 lines like `seller_1: 10 WAVs, 5.1M`.

### Step 5 — Phase 4.5 catalog embeddings (cluster `cpu`, ~1 min)

```bash
ssh CLUSTER 'cd <HOME>/music_ip_pilot && sbatch precompute_catalog_embeddings.slurm'
```

*Output*: `outputs/catalog_stats.json`, `outputs/catalog_embeddings.npz`.

### Step 6 — Phases 2–6 full pipeline (cluster `general`, GPU, ~60–90 min)

```bash
ssh CLUSTER 'cd <HOME>/music_ip_pilot && sbatch pilot_run.slurm'
```

*Output*: `checkpoints/{full,loo_1,loo_2,loo_3}/adapter/`, 40 generated WAVs, `outputs/{a_star.csv, a_hat_embed.csv, I_hat.csv}`, `outputs/figures/{I_scatter, ranking_comparison}.pdf`.

*Success check*:
- Each Phase 2/3 LoRA run prints `training done in ~190s`, `adapter saved to ...`.
- Phase 4 prints `[generate/<tag>]` lines for all 10 prompts × 4 models.
- Phase 5a prints `[a_star] wrote 30 rows`.
- Phase 6 produces `[I_hat] wrote` with $N$ rows, all with finite `I_hat`.

If it times out, the pipeline is idempotent — just resubmit, it picks up at the last completed adapter / generated clip.

### Step 7 — Base-model reference (cluster `general`, GPU, ~3–5 min)

```bash
ssh CLUSTER 'cd <HOME>/music_ip_pilot && sbatch add_base_reference.slurm'
```

*Output*: `q_base` column appended to `outputs/a_star.csv` (now 6 cols); `outputs/generated_audio/base/*.wav`.

*Success check*: log ends with a per-seller summary table. Positive `dFull-Base` → fine-tuning improved prompt alignment; strongly negative → LoRA overfit (lower the rank for the next run).

### Step 8 — Fetch results back to local

Stage the cluster's `<DATA_ROOT>` outputs to a path the login node can read, then `scp` back to `data/outputs/results/` in this folder.

### Step 9 — Post-hoc diagnostics (local CPU, < 30 s)

```bash
cd scripts
python drop_one_prompt_sensitivity.py
python marginal_welfare_plot.py
```

*Output*: `data/outputs/results/I_hat_sensitivity.csv`, `data/outputs/simulation/figures/marginal_welfare_vs_snr.pdf`.

### Step 10 — Auditory spot-check (local, ~2 min)

```bash
open data/outputs/results/audio_samples/
# Play full_prompt_1.wav vs base_prompt_1.wav. Both should be coherent music.
```

If either sounds like static, the quality metric is unreliable — investigate the MP3 → WAV decode and model sampling parameters before trusting any number.

---

## Scaling to N=22, T=200

Same pipeline, three changes:

**1. Generate a larger candidate pool.** Edit `generate_fallbacks.py`: set `TOP_K_PER_TIER = 8`. Re-run.

**2. Edit `pilot.yaml`:**
```yaml
n_sellers: 22              # was 3
tracks_per_seller: 50      # was 10
n_prompts: 200             # was 10
finetune_steps: 2000       # was 500
lora_rank: 4               # was 16 (avoids overfit observed at rank 16)
```

**3. Expand MP3 download.** Run `download_mtg.py` with `--mode all` (pulls primaries + fallbacks).

Then repeat Steps 3–9. Bump `pilot_run.slurm`'s `--time` to 8h ($N{=}22$ × 4 LoRA runs × ~10 min each + gen + attribution).

---

## Config reference

Every field in `scripts/pilot.yaml`. Change any of these and re-run; no code edits needed.

### Dataset
| Field | Pilot value | Meaning |
|---|---|---|
| `sellers_candidates_json` | `<HOME>/music_ip_pilot/data/stats/pilot_sellers_candidates.json` | Path to candidates JSON from `generate_fallbacks.py`. |
| `mtg_raw_dir` | `<DATA_ROOT>/music_ip_pilot/data/mtg_jamendo_raw` | Where staged MP3s live. |
| `n_sellers` | 3 | $N$. |
| `tracks_per_seller` | 10 | $\|D_j\|$. |
| `sample_rate` | 32000 | Hz. **Don't change** — MusicGen's native rate. |
| `clip_seconds` | 8 | Window length cut from each MP3. |

### Prompts
| Field | Pilot value | Meaning |
|---|---|---|
| `n_prompts` | 10 | $T$. Cross-stratified grid: $\lfloor T/10 \rfloor$ off-genre + $\lfloor (T - \text{off})/N \rfloor$ on-genre per seller. |

### Models
| Field | Pilot value | Meaning |
|---|---|---|
| `base_model` | `facebook/musicgen-small` | HF ID. ~300M-param text-conditioned decoder. bf16 throughout. |
| `clap_model` | `laion/clap-htsat-unfused` | HF ID. CLAP audio-text joint embedding model. |
| `models_hub` | `<DATA_ROOT>/models/hub` | HF cache (set as `HF_HOME` in slurm scripts). |

### LoRA fine-tuning
| Field | Pilot | Meaning |
|---|---|---|
| `use_lora` | true | Always true. |
| `lora_rank` | 16 | PEFT rank. Lower for less overfit. |
| `lora_alpha` | 32 | PEFT scaling. Convention: `2 × rank`. |
| `lora_dropout` | 0.05 | Light regularization. |
| `lora_target_modules` | `["decoder\\..*\\.(q_proj|v_proj)$"]` | Decoder Q/V attention only. |
| `finetune_steps` | 500 | Max optimization steps. |
| `finetune_lr` | 1.0e-4 | AdamW LR. |
| `finetune_batch_size` | 2 | Per-device. |
| `finetune_grad_accum` | 4 | Effective batch = 8. |
| `finetune_warmup_steps` | 20 | Linear LR warmup. |
| `finetune_logging_steps` | 20 | Trainer log frequency. |

### Generation
| Field | Pilot | Meaning |
|---|---|---|
| `gen_max_new_tokens` | 256 | ~5 sec at MusicGen's 50 Hz codebook rate. |
| `gen_top_k` | 250 | Standard. |
| `gen_temperature` | 1.0 | Standard. |
| `gen_do_sample` | true | Stochastic sampling. |

### Attribution & analysis
| Field | Pilot | Meaning |
|---|---|---|
| `attribution_methods` | `[embedding_similarity]` | Each method writes `outputs/a_hat_{name}.csv` with the same schema. |
| `bootstrap_samples` | 1000 | Resamples for $\widehat{\mathcal{I}}_j$ CI over $t$. |
| `ci_quantile` | 0.95 | Two-sided percentile-CI level. |

### Paths
| Field | Pilot | Meaning |
|---|---|---|
| `home_root` | `<HOME>/music_ip_pilot` | Code, configs, logs. |
| `data_root` | `<DATA_ROOT>/music_ip_pilot` | MP3s, WAVs, adapters, generated audio, results. |
| `seed` | 42 | RNG seed (random / numpy / torch CPU+CUDA). |

---

## Output artifact reference

All paths relative to `data_root` (default placeholder: `<DATA_ROOT>/music_ip_pilot/`).

| File | Written by | Schema |
|---|---|---|
| `data/sellers.json` | `build_sellers.py` (Phase 1) | List of $N$ sellers: `{seller_id, artist, primary_genre, tracks: [...]}` |
| `data/prompts.json` | `build_sellers.py` (Phase 1) | List of $T$ prompts: `{prompt_id, text, kind, target_genre, linked_seller}`. `kind ∈ {on_genre, off_genre_control}`. |
| `data/mtg_jamendo_subset/{seller_id}/{track_id}.wav` | `build_sellers.py` (Phase 1) | 32 kHz mono PCM_16, `clip_seconds` long. |
| `checkpoints/{tag}/adapter/` | `finetune_musicgen.py` (Phases 2–3) | PEFT LoRA adapter. `tag ∈ {full, loo_1, ..., loo_N}`. |
| `outputs/generated_audio/{tag}/{prompt_id}.wav` | `generate.py` (Phase 4); `add_base_reference.py` (5a-supp) | 32 kHz mono PCM_16, ~5 sec. `tag ∈ {full, loo_j, base}`. |
| `outputs/catalog_embeddings.npz` | `precompute_catalog_embeddings.py` (4.5) | numpy `.npz`: `grand_centroid` + per-seller `{sid}__embeddings` (K, D), `{sid}__centroid` (D), `{sid}__track_ids`. D = 512 (CLAP). All L2-normalised. |
| `outputs/catalog_stats.json` | `precompute_catalog_embeddings.py` (4.5) | JSON: per-seller `within_catalog_coherence`, `distinctiveness_to_grand`, `distinctiveness_to_nearest`. |
| `outputs/a_star.csv` | `compute_a_star.py` (5a) + `add_base_reference.py` (5a-supp) | Cols: `seller_id, prompt_id, q_full, q_loo, a_star, q_base`. |
| `outputs/a_hat_embed.csv` | `run_attribution.py` (5b) | Cols: `seller_id, prompt_id, a_hat_embed`. |
| `outputs/I_hat.csv` | `estimate_I.py` (6) | Cols: `seller_id, method, I_hat, I_hat_lo, I_hat_hi, spearman, spearman_lo, spearman_hi, mse, T`. |
| `outputs/I_hat_sensitivity.csv` | `drop_one_prompt_sensitivity.py` (post-hoc) | Cols: `seller_id, method, I_hat_full, I_hat_min_jk, I_hat_max_jk, I_hat_std_jk, most_influential_prompt_id, delta_at_most_influential, T`. |
| `outputs/figures/I_scatter.pdf` | `plot_results.py` (6) | Per-seller $\widehat{\mathcal{I}}_j$ + bootstrap CI. |
| `outputs/figures/ranking_comparison.pdf` | `plot_results.py` (6) | Rank by $\widehat{\mathcal{I}}_j$ vs rank by Spearman. |
| `outputs/simulation/figures/marginal_welfare_vs_snr.pdf` | `marginal_welfare_plot.py` (post-hoc) | Two panels: closed-form curves at varying $\alpha$, per-seller measured points. |
| `outputs/simulation/marginal_welfare_numbers.csv` | `marginal_welfare_plot.py` | Per-seller `(I_hat, SNR_hat, var_astar, marg_welfare_at_alpha2)`. |

Local-only:

| File | Written by | Schema |
|---|---|---|
| `data/stats/artists_summary.csv` | `eda_groupings.py` | Per-artist `n_tracks, total_duration_s, n_albums, primary_genre, genre_coherence, n_unique_genres`. |
| `data/stats/grouping_summary.json` | `eda_groupings.py` | Pool stats: prolificness quantiles, coherence distribution, top genres. |
| `data/stats/pilot_sellers_candidates.json` | `generate_fallbacks.py` | `{pool_meta, low: {primary, fallbacks: [...]}, mid: {...}, high: {...}}`. |

---

## Notation & symbols

| Symbol | Meaning |
|---|---|
| $N$ | Number of sellers. Pilot: 3. |
| $T$ | Number of test prompts. Pilot: 10. |
| $D_j$ | Seller $j$'s catalog (set of audio files). |
| $\|D_j\|$ | Tracks per catalog. Pilot: 10. |
| $r_t$ | Text prompt at index $t$. |
| $y^\text{full}_t$ | Generated audio from full-data LoRA at prompt $t$. |
| $y^{-j}_t$ | Generated audio from leave-$D_j$-out LoRA at prompt $t$. |
| $y^\text{base}_t$ | Generated audio from unfinetuned MusicGen base. |
| $Q(y, r) = \cos(\text{CLAP}_\text{audio}(y), \text{CLAP}_\text{text}(r))$ | Quality metric. |
| $a^*_{j,t} = Q(y^\text{full}_t, r_t) - Q(y^{-j}_t, r_t)$ | Ground-truth attribution. |
| $c_j$ | Seller $j$'s CLAP centroid (L2-normalised mean of catalog embeddings). |
| $\bar{c}$ | Grand centroid. |
| $\hat{a}^\text{embed}_{j,t} = \cos(\text{CLAP}(y^\text{full}_t), c_j)$ | Measured attribution (embedding-similarity method). |
| $\widehat{\mathcal{I}}_j = \text{Cov}(\hat{a}_{j,t}, a^*_{j,t}) / \text{Var}(\hat{a}_{j,t})$ | Sample informativeness (per seller). |
| $\text{SNR}_j = \mathcal{I}_j / (1 - \mathcal{I}_j)$ | Gaussian-special-case SNR conversion. |
| $\text{coherence}_j$ | Mean off-diagonal of within-catalog cosine sim matrix. |
| $d(c_j, \bar{c}) = 1 - \cos(c_j, \bar{c})$ | Distinctiveness to grand centroid. |
| $d(c_j, \text{nearest}) = 1 - \max_{k \neq j} \cos(c_j, c_k)$ | Distinctiveness to nearest other centroid. |

---

## Quality metric specification

CLAP cosine, with L2-normalised audio and text embeddings:

$$Q(y, r) = \cos(\text{CLAP}_\text{audio}(y), \text{CLAP}_\text{text}(r))$$

- **CLAP**: `laion/clap-htsat-unfused` via `transformers.ClapModel`.
- **Audio**: 48 kHz mono float32 (resampled from the 32 kHz pipeline WAVs).
- **Text**: passed through `ClapProcessor` with padding.
- **Embeddings**: L2-normalised via `torch.nn.functional.normalize(..., dim=-1)`.

Implementation: `compute_a_star._clap_audio_embed`, `_clap_text_embed`. Embeddings are 512-d float32.

Catalog centroid: $c_j$ = L2-normalised mean of `CLAP_audio` over the $\|D_j\|$ tracks in seller $j$'s catalog.

---

## Pilot results

Catalog geometry (Phase 4.5):

| Seller | Genre | $K$ | Coherence | $d(c_j, \bar{c})$ | $d(c_j, \text{nearest})$ |
|---|---|---:|---:|---:|---:|
| seller_1 | pop | 10 | 0.711 | 0.146 | 0.206 |
| seller_2 | reggae | 10 | 0.752 | 0.119 | 0.206 |
| seller_3 | classical | 10 | **0.894** | **0.350** | **0.692** |

Per-seller informativeness with 95% bootstrap CIs:

| Seller | Genre | $\widehat{\mathcal{I}}_j$ | 95% CI | Spearman | MSE |
|---|---|---:|---|---:|---:|
| seller_1 | pop | 0.617 | [-0.34, 1.02] | 0.333 | 0.153 |
| seller_2 | reggae | 0.755 | [-0.54, 1.63] | 0.467 | 0.235 |
| seller_3 | classical | 0.209 | [-0.26, 0.76] | 0.018 | 0.045 |

All CIs cross zero at $T=10$. Scale-up to $T \geq 200$ for tight intervals.

Drop-one-prompt jackknife (sensitivity):

| Seller | $\widehat{\mathcal{I}}_j$ full | jackknife range | most-influential prompt | delta |
|---|---:|---|---|---:|
| seller_1 | 0.617 | [0.39, 0.77] | prompt_1 (pop on-genre) | −0.23 |
| seller_2 | 0.755 | **[0.24, 1.16]** | prompt_3 (pop on-genre) | **−0.51** |
| seller_3 | 0.209 | [−0.03, 0.40] | prompt_7 (classical on-genre) | −0.24 |

Base-model reference (Phase 5a-supplement):

| Seller | $Q(y^\text{full})$ | $Q(y^\text{loo})$ | $Q(y^\text{base})$ | $\Delta$Full−Base | $\Delta$Loo−Base |
|---|---:|---:|---:|---:|---:|
| seller_1 | +0.109 | +0.047 | +0.467 | **−0.358** | −0.420 |
| seller_2 | +0.109 | +0.142 | +0.467 | **−0.358** | −0.325 |
| seller_3 | +0.109 | +0.152 | +0.467 | **−0.358** | −0.315 |

**LoRA at rank 16 / 500 steps hurt prompt alignment by ~0.36 (CLAP cosine units).** Outputs drifted toward catalog style, away from prompt semantics. Lower the LoRA rank (e.g., 4) and add early stopping for the next run.

---

## Sampling design

Cell allocation for scale-up Strategy A (proportional to what the coherent pool supports):

| Prolificness band | Target | Genres (2 each unless noted) |
|---|---|---|
| Low (10–15 tracks) | 8 | electronic, pop, rock, hiphop |
| Mid (16–25 tracks) | 8 | classical, electronic, pop, hiphop |
| High (26–50 tracks) | 5 | classical (2), electronic (1), hiphop (1), ambient (1) |
| Extra-high (50+ tracks) | 1 | classical |

Why this layout: the coherent pool's genre × prolificness cross-tab is heavily skewed (only classical has artists at 50+ tracks; pop and rock don't reach 26+). A balanced grid is infeasible. Within each cell, draw without replacement using `cfg.seed`.
