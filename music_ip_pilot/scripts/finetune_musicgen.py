#!/usr/bin/env python3
"""
Phases 2 & 3: Fine-tune MusicGen with a LoRA adapter
====================================================

Fine-tunes a LoRA adapter on a subset of seller catalogs. The "full"
run trains on all sellers; each "loo_j" run trains on everything
except seller j's catalog.

Training data format:
  (text_prompt, audio_clip)
  - text_prompt: a simple caption derived from seller genre + track title
    suffix. We do NOT use artist names (CLAP won't recognize them).
  - audio_clip: 32kHz mono, cfg.clip_seconds long.

LoRA: applied to `q_proj`, `v_proj` modules of MusicGen's decoder (see
config `lora_target_modules`). The EnCodec and T5 text encoder are
frozen. Only the adapter is saved.

Usage:
  python finetune_musicgen.py --config configs/pilot.yaml            # full
  python finetune_musicgen.py --config configs/pilot.yaml --loo-j 2  # leave out seller 2
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common import (
    checkpoint_dir,
    read_json,
    sellers_json_path,
    seller_audio_dir,
    set_all_seeds,
    set_hf_env,
)


def _build_caption(seller: Dict[str, Any], track: Dict[str, Any]) -> str:
    """
    Derive a short text caption for a training pair.
    Uses primary_genre + a neutral descriptor; avoids the artist name
    so the CLAP text encoder isn't required to know proper nouns.
    """
    genre = seller.get("primary_genre", "instrumental")
    return f"{genre} music, short instrumental clip"


def _load_training_pairs(
    cfg: Dict[str, Any],
    exclude_seller: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Build (text, audio_path) pairs from all sellers except the excluded one.

    Returns a list of dicts: {"text": ..., "audio_path": ..., "seller_id": ...}
    """
    sellers = read_json(sellers_json_path(cfg))
    pairs: List[Dict[str, Any]] = []
    for idx, s in enumerate(sellers, start=1):
        if exclude_seller is not None and idx == exclude_seller:
            continue
        audio_dir = seller_audio_dir(cfg, s["seller_id"])
        for t in s["tracks"]:
            p = audio_dir / f"{t['track_id']}.wav"
            if not p.exists():
                raise FileNotFoundError(f"expected staged audio: {p}")
            pairs.append({
                "text": _build_caption(s, t),
                "audio_path": str(p),
                "seller_id": s["seller_id"],
            })
    return pairs


def main(
    cfg: Dict[str, Any],
    exclude_seller: Optional[int] = None,
    tag: str = "full",
) -> None:
    """
    Fine-tune a LoRA adapter. Leaves the base MusicGen weights untouched;
    only adapter weights are saved to <data_root>/checkpoints/<tag>/adapter/.
    """
    import soundfile as sf
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoProcessor,
        MusicgenForConditionalGeneration,
        Trainer,
        TrainingArguments,
    )

    set_hf_env(cfg)
    set_all_seeds(cfg["seed"])
    torch.set_float32_matmul_precision("high")

    ckpt_root = checkpoint_dir(cfg, tag)
    adapter_dst = ckpt_root / "adapter"
    if (adapter_dst / "adapter_config.json").exists():
        print(f"[finetune/{tag}] adapter already exists at {adapter_dst}; skip", flush=True)
        return

    pairs = _load_training_pairs(cfg, exclude_seller)
    print(f"[finetune/{tag}] training pairs: {len(pairs)} "
          f"(exclude_seller={exclude_seller})", flush=True)
    if len(pairs) == 0:
        raise RuntimeError(f"no training pairs for tag={tag}; did setup run?")

    # Load model + processor
    print(f"[finetune/{tag}] loading {cfg['base_model']}", flush=True)
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(cfg["base_model"])
    # Keep base in bf16 to halve static weight memory on L40S; LoRA adapters
    # trained in fp32 per PEFT default is still numerically stable.
    model = MusicgenForConditionalGeneration.from_pretrained(
        cfg["base_model"], torch_dtype=torch.bfloat16
    )
    print(f"[finetune/{tag}] base loaded in {time.time()-t0:.1f}s", flush=True)

    # MusicGen's HF config ships with `decoder.decoder_start_token_id=None`.
    # MusicgenForConditionalGeneration.forward() reads it from the NESTED path
    # (`self.config.decoder.decoder_start_token_id`) — NOT the top-level — and
    # passes it to shift_tokens_right(), which raises "Make sure to set the
    # decoder_start_token_id attribute" when None. We set both locations to be
    # safe. MusicGen uses pad-as-start by design (pad_token_id is the end-of-
    # sequence / start token for the codebook delay pattern).
    dec_cfg = model.config.decoder
    if dec_cfg.decoder_start_token_id is None:
        start_id = getattr(dec_cfg, "pad_token_id", None) \
                   or getattr(dec_cfg, "bos_token_id", None)
        if start_id is None:
            raise RuntimeError(
                "Cannot derive decoder_start_token_id: neither pad_token_id "
                "nor bos_token_id set on model.config.decoder."
            )
        dec_cfg.decoder_start_token_id = int(start_id)
        model.config.decoder_start_token_id = int(start_id)  # mirror top-level too
        print(f"[finetune/{tag}] set decoder.decoder_start_token_id = {start_id}",
              flush=True)

    # Freeze base; attach LoRA to decoder attention projections.
    # target_modules accepts either a list of suffixes or a regex string.
    # Single regex keeps us safe against future model refactors.
    if cfg.get("use_lora", True):
        tm = cfg["lora_target_modules"]
        if isinstance(tm, list) and len(tm) == 1 and tm[0].startswith("decoder"):
            tm_arg = tm[0]  # treat as regex string
        else:
            tm_arg = list(tm) if isinstance(tm, list) else str(tm)
        lora_cfg = LoraConfig(
            r=int(cfg["lora_rank"]),
            lora_alpha=int(cfg["lora_alpha"]),
            lora_dropout=float(cfg["lora_dropout"]),
            target_modules=tm_arg,
            bias="none",
            task_type=None,  # MusicGen doesn't fit standard PEFT task types
        )
        # Freeze everything except LoRA-injected layers (PEFT re-enables LoRA grads)
        for p in model.parameters():
            p.requires_grad = False
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        raise NotImplementedError("pilot assumes LoRA fine-tuning")

    # Dataset wrapper
    ds = _AudioTextDataset(
        pairs=pairs,
        processor=processor,
        sample_rate=int(cfg["sample_rate"]),
        clip_samples=int(cfg["sample_rate"]) * int(cfg["clip_seconds"]),
    )

    # Collator uses the PEFT-wrapped model to (a) encode audio via EnCodec
    # and (b) build the delay-pattern mask for MusicGen's interleaved
    # codebooks. We pass the model by reference so the collator can reach
    # model.audio_encoder and model.decoder.build_delay_pattern_mask.
    collator = _MusicgenCollator(processor=processor, model_ref=model)

    # Trainer
    args = TrainingArguments(
        output_dir=str(ckpt_root / "trainer_state"),
        per_device_train_batch_size=int(cfg["finetune_batch_size"]),
        gradient_accumulation_steps=int(cfg["finetune_grad_accum"]),
        max_steps=int(cfg["finetune_steps"]),
        learning_rate=float(cfg["finetune_lr"]),
        warmup_steps=int(cfg["finetune_warmup_steps"]),
        logging_steps=int(cfg["finetune_logging_steps"]),
        save_strategy="no",
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        fp16=False,
        bf16=True,  # L40S supports BF16
        # num_workers must be 0: the collator calls back into the CUDA-resident
        # PEFT model (for EnCodec + delay pattern) and cannot be forked.
        dataloader_num_workers=0,
        # pin_memory must be False: the collator's EnCodec+delay-pattern pass
        # lives on GPU, so it returns CUDA tensors. Trainer's default
        # pin_memory=True then fails with:
        #   RuntimeError: cannot pin 'torch.cuda.LongTensor' only dense CPU
        #   tensors can be pinned.
        # We skip the pin step since the tensors are already on GPU anyway.
        dataloader_pin_memory=False,
        seed=int(cfg["seed"]),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        data_collator=collator,
    )

    print(f"[finetune/{tag}] starting {cfg['finetune_steps']} steps", flush=True)
    t0 = time.time()
    trainer.train()
    print(f"[finetune/{tag}] training done in {time.time()-t0:.1f}s", flush=True)

    # Save only the adapter
    adapter_dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dst))
    print(f"[finetune/{tag}] adapter saved to {adapter_dst}", flush=True)


class _AudioTextDataset:
    """Minimal torch-compatible dataset returning pre-tokenized dicts."""
    def __init__(self, pairs, processor, sample_rate, clip_samples):
        self.pairs = pairs
        self.processor = processor
        self.sr = int(sample_rate)
        self.clip_samples = int(clip_samples)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        import soundfile as sf
        import numpy as np
        p = self.pairs[i]
        audio, sr = sf.read(p["audio_path"], dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self.sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        # pad/crop to clip_samples
        if len(audio) >= self.clip_samples:
            audio = audio[: self.clip_samples]
        else:
            pad = np.zeros(self.clip_samples - len(audio), dtype=np.float32)
            audio = np.concatenate([audio, pad])
        return {"text": p["text"], "audio": audio}


class _MusicgenCollator:
    """
    Processes a batch of {text, audio} into MusicGen training inputs.

    Critically, MusicGen's `labels` argument expects *integer codebook
    indices* of shape (B, seq_len, num_codebooks) with the delay-pattern
    mask already applied --- NOT raw audio waveforms. So we:
      1. Tokenize text as conditioning (input_ids, attention_mask).
      2. Run audio through the model's EnCodec (audio_encoder) to get
         discrete codebook tokens.
      3. Apply the delayed-interleaving mask via
         decoder.build_delay_pattern_mask + apply_delay_pattern_mask.
      4. Return labels as LongTensor.
    """
    def __init__(self, processor, model_ref):
        self.processor = processor
        self.model = model_ref  # PEFT-wrapped MusicGen; we use .audio_encoder and .decoder

    def __call__(self, batch):
        import torch
        import numpy as np
        texts = [b["text"] for b in batch]
        audios = [np.asarray(b["audio"], dtype=np.float32) for b in batch]

        # Tokenize text (conditioning)
        text_inputs = self.processor.tokenizer(
            texts, padding=True, return_tensors="pt"
        )

        # Run processor to get EnCodec-compatible input_values (B, 1, T_samples)
        audio_inputs = self.processor(
            audio=audios,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        input_values = audio_inputs["input_values"]
        padding_mask = audio_inputs.get("padding_mask")

        # Walk PEFT wrapping to reach the underlying MusicGen composite model.
        # PEFT: `PeftModel.get_base_model()` returns the wrapped HF model.
        try:
            from peft import PeftModel
            if isinstance(self.model, PeftModel):
                base_model = self.model.get_base_model()
            else:
                base_model = self.model
        except Exception:
            base_model = self.model
        # Sanity check: must have MusicGen components.
        if not (hasattr(base_model, "audio_encoder") and hasattr(base_model, "decoder")):
            raise RuntimeError(
                "expected MusicGen composite model with .audio_encoder and .decoder; "
                f"got {type(base_model).__name__}"
            )

        device = next(base_model.parameters()).device
        audio_enc_dtype = next(base_model.audio_encoder.parameters()).dtype
        iv = input_values.to(device=device, dtype=audio_enc_dtype)
        pm = padding_mask.to(device=device) if padding_mask is not None else None

        with torch.no_grad():
            enc_out = base_model.audio_encoder.encode(iv, padding_mask=pm)
            # audio_codes shape: (num_chunks=1, B, num_codebooks, T_codes)
            audio_codes = enc_out.audio_codes
        audio_codes = audio_codes[0]  # (B, K, T_codes)

        # Build delay-pattern mask and convert to labels
        decoder = base_model.decoder
        pad_token_id = int(getattr(decoder.config, "pad_token_id", 2048))

        B, K, T_codes = audio_codes.shape
        # Prepend one pad column so build_delay_pattern_mask is given shape (B, K, T+1)
        pad_col = torch.full((B, K, 1), pad_token_id, dtype=audio_codes.dtype,
                             device=audio_codes.device)
        codes_with_bos = torch.cat([pad_col, audio_codes], dim=-1)  # (B, K, T+1)

        # Flatten batch*codebooks for HF's builder (expects (bsz*K, T))
        codes_flat = codes_with_bos.reshape(B * K, -1)
        # v4.57 signature: build_delay_pattern_mask(input_ids, pad_token_id, max_length)
        delayed_ids, delay_mask = decoder.build_delay_pattern_mask(
            codes_flat, pad_token_id=pad_token_id, max_length=codes_flat.shape[-1] + K,
        )
        delayed_ids = decoder.apply_delay_pattern_mask(delayed_ids, delay_mask)
        # reshape back to (B, K, T_delayed), drop leading pad column
        delayed_ids = delayed_ids.reshape(B, K, -1)
        # MusicgenForCausalLM expects labels of shape (B, seq_len, num_codebooks).
        # We have (B, K, T-1); transpose last two dims.
        labels = delayed_ids[:, :, 1:].transpose(1, 2).contiguous().long()

        # Move text inputs to the same device so Trainer doesn't trip
        batch_out = {
            "input_ids": text_inputs["input_ids"].to(device),
            "attention_mask": text_inputs["attention_mask"].to(device),
            "labels": labels,
        }
        # MusicGen's forward can also take padding_mask for the audio; not required when
        # `labels` is provided (decoder builds its own causal mask internally).
        return batch_out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pilot.yaml")
    parser.add_argument("--loo-j", type=int, default=None,
                        help="exclude this seller index (1..N); omit for 'full' run")
    args = parser.parse_args()
    from common import load_config
    cfg = load_config(args.config)
    tag = f"loo_{args.loo_j}" if args.loo_j is not None else "full"
    main(cfg, exclude_seller=args.loo_j, tag=tag)
