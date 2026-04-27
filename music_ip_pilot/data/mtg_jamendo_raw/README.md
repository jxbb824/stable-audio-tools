# Raw MTG-Jamendo MP3s — NOT in the handover

This folder is intentionally empty in the handover. **Raw MP3s live on the
cluster only**, at `<DATA_ROOT>/music_ip_pilot/data/mtg_jamendo_raw/{prefix}/{track_id}.mp3`.

## How they get here

Run `scripts/download_mtg.slurm` on the cluster (Step 3 of the cookbook in the
top-level README). It pulls **67 files (~570 MB)** for the pilot's 3 primary
sellers from the Freesound CDN mirror at `cdn.freesound.org/mtg-jamendo/...`.
For scale-up, run with `--mode all` to also fetch the fallback artists' tracks
(~500 files, ~2.5 GB total).

The download is idempotent — re-running skips files that already exist.

## Why not in the handover

Raw audio is bulky (>500 MB even for the pilot) and easily reproducible by the
above one-step download. There's nothing artist-specific about how we acquired
it — `download_mtg.py` is deterministic given `pilot_sellers_candidates.json`.
