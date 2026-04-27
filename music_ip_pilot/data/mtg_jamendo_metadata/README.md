# MTG-Jamendo metadata (download stub)

This folder is intentionally empty in the handover. The pre-pipeline EDA needs
**`autotagging.tsv`** (~10 MB) to be present here. Pull it once with:

```bash
curl -sSL -o autotagging.tsv \
  "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data/raw_30s_cleantags_50artists.tsv"
```

You only need this if you re-run `eda_groupings.py` or `generate_fallbacks.py`.
The handover already ships the EDA outputs (in `../stats/` and `../figures/`),
so the metadata is only needed if you want to redo seller selection.

For the audio MP3s themselves (~570 MB for the pilot's 67 tracks), use the
cluster-side `download_mtg.slurm` — they are **not** shipped in the handover.
