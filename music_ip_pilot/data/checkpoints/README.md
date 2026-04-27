# LoRA adapters — NOT in the handover

This folder is intentionally empty in the handover. **LoRA adapters live on the
cluster only**, at `<DATA_ROOT>/music_ip_pilot/checkpoints/{tag}/adapter/`
where `tag ∈ {full, loo_1, loo_2, ..., loo_N}`.

## How they get here

Phases 2–3 (`scripts/finetune_musicgen.py`, called via `pilot_run.slurm` —
Step 6 of the cookbook) produce one adapter per tag:

- `full/` — LoRA fine-tuned on the union of all $N$ sellers' catalogs.
- `loo_j/` — LoRA fine-tuned with seller $j$'s catalog excluded (one per seller).

Each adapter is a PEFT-format directory (`adapter_model.safetensors` +
`adapter_config.json`), ~50 MB. For the pilot: 4 adapters, ~200 MB.

These are loaded by Phase 4 (`generate.py`) for each tag via
`PeftModel.from_pretrained(base, adapter_dir)`.

## Why not in the handover

Adapter weights are heavy and have no value without the base model
(`facebook/musicgen-small`). Re-training on a different cluster reproduces them
deterministically given the same seed + same staged WAVs.
