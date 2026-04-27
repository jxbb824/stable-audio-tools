# Phase 1 staged WAVs — NOT in the handover

This folder is intentionally empty in the handover. **Staged WAVs live on the
cluster only**, at `<DATA_ROOT>/music_ip_pilot/data/mtg_jamendo_subset/{seller_id}/{track_id}.wav`.

## How they get here

Phase 1 (`scripts/build_sellers.py`, called via `phase1_setup.slurm` — Step 4
of the cookbook) decodes each raw MP3 from `mtg_jamendo_raw/`, takes a centered
8-second clip, resamples to 32 kHz mono, and writes one PCM_16 WAV per track.

For the pilot ($N=3$, `tracks_per_seller=10`): **30 files, ~150 MB**.

## Schema

`{seller_id}/{track_id}.wav` — one subdirectory per seller (`seller_1`,
`seller_2`, `seller_3` for the pilot), one WAV per track.

These are the files MusicGen LoRA training reads. They are deterministic given
`sellers.json` + the raw MP3s.
