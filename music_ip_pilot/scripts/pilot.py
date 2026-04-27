#!/usr/bin/env python3
"""
Music IP Pilot — Phase Dispatcher
=================================

Runs one or more phases of the Path A attribution pipeline on tiny scale
(N=3 sellers, T=10 prompts). Each phase is idempotent; outputs go to
the data disk. Scale up by editing configs/pilot.yaml; no code changes.

Phases:
  setup        Build sellers.json + prompts.json, stage audio.
  baseline     Fine-tune LoRA adapter on the full dataset.
  loo          Fine-tune N LoRA adapters, each leaving out seller j.
  generate     Generate audio for T prompts from each (full + LOO) model.
  attribution  Compute ground-truth a* (quality delta) + measured a_hat (CLAP sim).
  analyze      Compute per-seller I_hat with bootstrap CIs; plot.
  all          Run phases in order: setup, baseline, loo, generate, attribution, analyze.

Usage:
  python pilot.py --phase setup     --config configs/pilot.yaml
  python pilot.py --phase baseline  --config configs/pilot.yaml
  python pilot.py --phase loo       --seller-idx 1 --config configs/pilot.yaml
  python pilot.py --phase all       --config configs/pilot.yaml
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sibling modules importable regardless of where python is invoked from.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from common import load_config


VALID_PHASES = ["setup", "baseline", "loo", "generate", "attribution", "analyze", "all"]


def log(msg: str) -> None:
    print(f"[pilot {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_setup(cfg: dict) -> None:
    log("=== Phase: setup (build_sellers.py) ===")
    import build_sellers
    build_sellers.main(cfg)


def run_baseline(cfg: dict) -> None:
    log("=== Phase: baseline (finetune full) ===")
    import finetune_musicgen
    finetune_musicgen.main(cfg, exclude_seller=None, tag="full")


def run_loo(cfg: dict, seller_idx: int | None) -> None:
    import finetune_musicgen
    n = cfg["n_sellers"]
    if seller_idx is None:
        indices = list(range(1, n + 1))
    else:
        indices = [seller_idx]
    for j in indices:
        log(f"=== Phase: loo (finetune leave-out seller {j}) ===")
        finetune_musicgen.main(cfg, exclude_seller=j, tag=f"loo_{j}")


def run_generate(cfg: dict) -> None:
    log("=== Phase: generate ===")
    import generate as gen_mod
    # full model
    gen_mod.main(cfg, model_tag="full")
    # each LOO model
    for j in range(1, cfg["n_sellers"] + 1):
        gen_mod.main(cfg, model_tag=f"loo_{j}")


def run_attribution(cfg: dict) -> None:
    log("=== Phase: attribution (a*, a_hat) ===")
    import compute_a_star
    import run_attribution
    compute_a_star.main(cfg)
    run_attribution.main(cfg)


def run_analyze(cfg: dict) -> None:
    log("=== Phase: analyze (I_hat + plots) ===")
    import estimate_I
    import plot_results
    estimate_I.main(cfg)
    plot_results.main(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Music IP pilot dispatcher")
    parser.add_argument("--phase", required=True, choices=VALID_PHASES)
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--seller-idx", type=int, default=None,
                        help="for loo phase: run only one seller index (1..N)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        # resolve relative to this script's grandparent (repo root)
        root = Path(__file__).resolve().parent.parent
        cfg_path = (root / cfg_path).resolve()
    if not cfg_path.exists():
        log(f"ERROR: config not found: {cfg_path}")
        sys.exit(1)

    cfg = load_config(cfg_path)
    log(f"Config: {cfg_path}")
    log(f"  N={cfg['n_sellers']} sellers, T={cfg['n_prompts']} prompts")
    log(f"  base_model={cfg['base_model']}")
    log(f"  data_root={cfg['data_root']}")

    phase = args.phase
    t0 = time.time()

    if phase == "setup":
        run_setup(cfg)
    elif phase == "baseline":
        run_baseline(cfg)
    elif phase == "loo":
        run_loo(cfg, args.seller_idx)
    elif phase == "generate":
        run_generate(cfg)
    elif phase == "attribution":
        run_attribution(cfg)
    elif phase == "analyze":
        run_analyze(cfg)
    elif phase == "all":
        run_setup(cfg)
        run_baseline(cfg)
        run_loo(cfg, seller_idx=None)
        run_generate(cfg)
        run_attribution(cfg)
        run_analyze(cfg)
    else:
        raise ValueError(f"unknown phase: {phase}")

    log(f"Phase '{phase}' complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
