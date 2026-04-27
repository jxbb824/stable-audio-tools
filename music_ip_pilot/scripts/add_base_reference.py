#!/usr/bin/env python3
"""
Phase 5a-supplement: Add $Q(y^base, \text{text})$ reference column to a_star.csv
================================================================================

Generates 10 clips from the UNFINETUNED MusicGen base model on the pilot's
10 prompts, CLAP-scores each against its prompt text, and appends a
`q_base` column to outputs/a_star.csv.

Purpose (per Scale-up Commitment #8 in experiments_plan.md):
disambiguate the LOO-asymmetry concern. The pilot's a_star signs can be
driven by either
  (a) "LoRA helped, but LOO-j overfits less than full"  OR
  (b) "LoRA had no effect; observed a* is training-stochasticity noise".

Comparing Q(y^full, text) and Q(y^loo, text) against Q(y^base, text) tells us
which mechanism is at play per prompt:
  - Q(y^full) >> Q(y^base)  : fine-tuning helped
  - Q(y^full) ~= Q(y^base)  : LoRA was effectively a no-op
  - Q(y^full) <  Q(y^base)  : LoRA hurt (overfit to catalog away from prompt)

Idempotent: skips generation if base clips exist; skips CSV merge if `q_base`
already present in a_star.csv.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from common import (
    generated_audio_dir,
    load_config,
    outputs_dir,
    prompts_json_path,
    read_json,
    set_all_seeds,
    set_hf_env,
)
from compute_a_star import (
    _clap_audio_embed,
    _clap_text_embed,
    _load_clap,
    _read_wav_mono_48k,
)


def _generate_base_clips(cfg: Dict[str, Any], prompts, out_dir: Path) -> None:
    """Generate clips with unfinetuned MusicGen base. Idempotent per-prompt."""
    import soundfile as sf
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    processor = AutoProcessor.from_pretrained(cfg["base_model"])
    model = MusicgenForConditionalGeneration.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16
    )
    # Same decoder_start_token_id fix as in finetune_musicgen.py: MusicGen's
    # nested config path has it None by default, breaking labels-only forward
    # (though we only call .generate() here, we set both to be consistent with
    # training-time config).
    dec_cfg = model.config.decoder
    if dec_cfg.decoder_start_token_id is None:
        start_id = (getattr(dec_cfg, "pad_token_id", None)
                    or getattr(dec_cfg, "bos_token_id", None))
        if start_id is None:
            raise RuntimeError("Cannot derive decoder_start_token_id on base model")
        dec_cfg.decoder_start_token_id = int(start_id)
        model.config.decoder_start_token_id = int(start_id)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"[base-ref] base model loaded on {device}", flush=True)

    gen_kwargs = dict(
        max_new_tokens=int(cfg["gen_max_new_tokens"]),
        do_sample=bool(cfg["gen_do_sample"]),
        top_k=int(cfg["gen_top_k"]),
        temperature=float(cfg["gen_temperature"]),
    )
    sampling_rate = model.config.audio_encoder.sampling_rate

    for p in prompts:
        dst = out_dir / f"{p['prompt_id']}.wav"
        if dst.exists() and dst.stat().st_size > 0:
            continue
        t0 = time.time()
        inputs = processor(
            text=[p["text"]], padding=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        # bf16 -> fp32 before .numpy() (same fix as in generate.py)
        audio = out[0, 0].to(torch.float32).cpu().numpy()
        audio = np.clip(audio, -1.0, 1.0)
        sf.write(str(dst), audio, sampling_rate, subtype="PCM_16")
        print(f"[base-ref] {p['prompt_id']}: '{p['text'][:60]}...' "
              f"-> {dst.name} ({len(audio)/sampling_rate:.1f}s, {time.time()-t0:.1f}s)",
              flush=True)

    # free model memory before loading CLAP
    del model, processor
    try:
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
    except Exception:
        pass


def main(cfg: Dict[str, Any]) -> None:
    set_hf_env(cfg)
    set_all_seeds(cfg["seed"])

    a_star_path = outputs_dir(cfg) / "a_star.csv"
    if not a_star_path.exists():
        raise FileNotFoundError(f"{a_star_path} missing; run Phase 5a first")
    astar = pd.read_csv(a_star_path)

    if "q_base" in astar.columns:
        print(f"[base-ref] q_base already in {a_star_path}; skip", flush=True)
        # still report summary
        _report_summary(astar)
        return

    prompts = read_json(prompts_json_path(cfg))
    base_dir = generated_audio_dir(cfg, "base")
    print(f"[base-ref] generating/loading base clips in {base_dir}", flush=True)

    _generate_base_clips(cfg, prompts, base_dir)

    # CLAP embedding for the base clips + prompt text
    model, processor_clap, clap_device = _load_clap(cfg)
    print(f"[base-ref] CLAP on {clap_device}", flush=True)

    base_audios = [_read_wav_mono_48k(base_dir / f"{p['prompt_id']}.wav")
                   for p in prompts]
    base_aud_emb = _clap_audio_embed(model, processor_clap, base_audios, clap_device)
    text_emb = _clap_text_embed(
        model, processor_clap, [p["text"] for p in prompts], clap_device
    )
    q_base = (base_aud_emb * text_emb).sum(axis=1)  # (T,)

    # Merge into a_star.csv — q_base is per prompt, broadcast across sellers
    q_base_df = pd.DataFrame({
        "prompt_id": [p["prompt_id"] for p in prompts],
        "q_base": q_base.astype(np.float64),
    })
    astar_new = astar.merge(q_base_df, on="prompt_id", how="left")
    astar_new.to_csv(a_star_path, index=False)
    print(f"[base-ref] appended q_base to {a_star_path} "
          f"(now {len(astar_new.columns)} columns, {len(astar_new)} rows)",
          flush=True)

    _report_summary(astar_new)


def _report_summary(astar: pd.DataFrame) -> None:
    """Paper-friendly diagnostic summary: per-seller means of q_{full,loo,base}."""
    print("\n[base-ref] summary (per-seller Q means; pilot pi_t = 1):")
    cols = ["q_full", "q_loo", "q_base"]
    header = f"{'seller':<11} " + " ".join(f"{c:>10}" for c in cols) + \
             f" {'dFull-Base':>12} {'dLoo-Base':>10}"
    print(header)
    print("-" * len(header))
    for sid in sorted(astar["seller_id"].unique()):
        sub = astar[astar["seller_id"] == sid]
        q_full = sub["q_full"].mean()
        q_loo = sub["q_loo"].mean()
        q_base = sub["q_base"].mean()
        print(f"{sid:<11} {q_full:>+10.4f} {q_loo:>+10.4f} {q_base:>+10.4f} "
              f"{q_full - q_base:>+12.4f} {q_loo - q_base:>+10.4f}")

    print("\nInterpretation key:")
    print("  dFull-Base > 0 : LoRA full-finetune improved prompt-alignment (expected)")
    print("  dFull-Base ~ 0 : LoRA had no effect; pilot a* signal is likely noise")
    print("  dFull-Base < 0 : LoRA hurt prompt-alignment (overfit to catalog)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
