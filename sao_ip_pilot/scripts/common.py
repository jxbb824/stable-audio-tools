from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def load_config(path: str | Path) -> dict[str, Any]:
    path = resolve_repo_path(path)
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    return cfg


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def ensure_dir(path: str | Path) -> Path:
    path = resolve_repo_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: str | Path) -> Any:
    path = resolve_repo_path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    path = resolve_repo_path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_lines(path: str | Path, values: Iterable[str]) -> None:
    path = resolve_repo_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{value}\n")


def normalize_path_key(value: str) -> str:
    return os.path.normpath(value)


def candidate_path_keys(path_value: str, relpath_value: str = "") -> list[str]:
    keys: list[str] = []
    for value in (path_value, relpath_value):
        if not value:
            continue
        p = Path(value)
        keys.extend(
            [
                os.path.normpath(value),
                os.path.normpath(os.path.abspath(value)),
                p.name,
                p.stem,
            ]
        )
    return list(dict.fromkeys(keys))
