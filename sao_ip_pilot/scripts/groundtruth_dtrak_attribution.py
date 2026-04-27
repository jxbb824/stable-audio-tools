from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_path(value: str) -> str:
    return os.path.normpath(os.path.abspath(value))


def candidate_keys(row: Dict[str, str]) -> List[str]:
    path_value = row.get("path", "")
    relpath_value = row.get("relpath", "")
    keys = []
    if path_value:
        keys.append(normalize_path(path_value))
        keys.append(os.path.normpath(path_value))
        keys.append(Path(path_value).name)
        keys.append(Path(path_value).stem)
    if relpath_value:
        keys.append(os.path.normpath(relpath_value))
        keys.append(Path(relpath_value).name)
        keys.append(Path(relpath_value).stem)
    return keys


def parse_ensemble_model_indices(value: str) -> List[int]:
    if value.strip() == "":
        return []
    indices = []
    for item in value.split(","):
        item = item.strip()
        if item:
            indices.append(int(item))
    return indices


def discover_model_dirs(models_root: Path, model_indices: Sequence[int]) -> List[Tuple[int, Path]]:
    pairs = []
    for model_index in model_indices:
        model_dir = models_root / str(model_index)
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        pairs.append((int(model_index), model_dir))
    return pairs


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
