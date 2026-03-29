from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


def _default_seconds_total() -> float:
    return 2097152 / 44100


@lru_cache(maxsize=32)
def _load_manifest(manifest_path: str) -> dict[str, dict]:
    path = Path(manifest_path)
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    records = {}
    for item in items:
        filename = item.get("filename")
        if filename:
            records[filename] = item
    return records


def get_custom_metadata(info, audio):
    audio_path = Path(info.get("path", ""))
    manifest_path = audio_path.parent / "prompt_manifest.json"
    records = _load_manifest(str(manifest_path))
    record = records.get(audio_path.name, {})

    seconds_total = record.get("seconds_total", info.get("seconds_total", _default_seconds_total()))
    return {
        "prompt": record.get("prompt", ""),
        "seconds_start": float(record.get("seconds_start", 0.0)),
        "seconds_total": float(seconds_total),
    }
