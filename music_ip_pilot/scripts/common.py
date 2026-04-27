"""
Shared utilities for the Music IP pilot pipeline.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load pilot config YAML."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # sanity checks
    for required in ("n_sellers", "tracks_per_seller", "n_prompts",
                     "base_model", "clap_model", "models_hub",
                     "seed", "home_root", "data_root",
                     "sellers_candidates_json", "mtg_raw_dir",
                     "sample_rate", "clip_seconds"):
        if required not in cfg:
            raise ValueError(f"config missing required key: {required}")
    return cfg


def set_all_seeds(seed: int) -> None:
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_subdir(cfg: Dict[str, Any], *parts: str) -> Path:
    return ensure_dir(Path(cfg["data_root"]).joinpath(*parts))


def read_json(p: str | Path) -> Any:
    with open(p, "r") as f:
        return json.load(f)


def write_json(p: str | Path, obj: Any) -> None:
    ensure_dir(Path(p).parent)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def sellers_json_path(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["data_root"]) / "data" / "sellers.json"


def prompts_json_path(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["data_root"]) / "data" / "prompts.json"


def seller_audio_dir(cfg: Dict[str, Any], seller_id: str) -> Path:
    return Path(cfg["data_root"]) / "data" / "mtg_jamendo_subset" / seller_id


def checkpoint_dir(cfg: Dict[str, Any], tag: str) -> Path:
    """tag in {'full', 'loo_1', 'loo_2', ...}"""
    return ensure_dir(Path(cfg["data_root"]) / "checkpoints" / tag)


def generated_audio_dir(cfg: Dict[str, Any], tag: str) -> Path:
    return ensure_dir(Path(cfg["data_root"]) / "outputs" / "generated_audio" / tag)


def outputs_dir(cfg: Dict[str, Any]) -> Path:
    return ensure_dir(Path(cfg["data_root"]) / "outputs")


def figures_dir(cfg: Dict[str, Any]) -> Path:
    return ensure_dir(Path(cfg["data_root"]) / "outputs" / "figures")


def set_hf_env(cfg: Dict[str, Any]) -> None:
    """Point HF caches at the shared models hub on data disk."""
    hub = cfg["models_hub"]
    os.environ["HF_HOME"] = hub
    os.environ["HF_HUB_CACHE"] = hub
    # TRANSFORMERS_OFFLINE=1 requires first load to have succeeded;
    # we leave it off so missing files surface as a clear error.
