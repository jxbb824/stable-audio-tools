#!/usr/bin/env python3
"""
Minimal ACE-Step captioner test script.

This follows the official Hugging Face model card:
- ACE-Step/acestep-captioner uses the same API as Qwen2.5-Omni-7B
- prompt format: "*Task* Describe this audio in detail" + <audio>
- text output only
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_MODEL_ID = "ACE-Step/acestep-captioner"
DEFAULT_HF_HOME = "/ocean/projects/mth240006p/xjiang6/huggingface"
PROMPT = "*Task* Describe this audio in detail"
USE_AUDIO_IN_VIDEO = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_path", help="Path to an input audio file")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hf-home", default=DEFAULT_HF_HOME)
    return parser.parse_args()


def configure_hf_cache(hf_home: str) -> None:
    os.environ["HF_HOME"] = hf_home
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_home, "transformers")

    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HUB_CACHE"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)


def _skip_load_speakers_for_text_only(self, path):
    del path
    self.speaker_map = {"Chelsie": {}, "Ethan": {}}


def load_model_text_only(model_id: str, cache_dir: str):
    from transformers import Qwen2_5OmniForConditionalGeneration

    Qwen2_5OmniForConditionalGeneration.load_speakers = _skip_load_speakers_for_text_only
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype="auto",
        device_map="auto",
        use_safetensors=True,
    )
    model.disable_talker()
    return model


def main() -> int:
    args = parse_args()
    audio_path = Path(args.audio_path).expanduser().resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    configure_hf_cache(args.hf_home)
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen2_5OmniProcessor

    model = load_model_text_only(args.model_id, args.hf_home)
    processor = Qwen2_5OmniProcessor.from_pretrained(
        args.model_id,
        cache_dir=args.hf_home,
    )

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "audio", "audio": str(audio_path)},
            ],
        }
    ]

    text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False,
    )
    audios, images, videos = process_mm_info(
        conversation,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    text_ids = model.generate(
        **inputs,
        return_audio=False,
        use_audio_in_video=USE_AUDIO_IN_VIDEO,
    )
    captions = processor.batch_decode(
        text_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(captions[0] if captions else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
