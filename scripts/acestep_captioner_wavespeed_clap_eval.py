#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_audio_tools.training.songscriber_clap import (  # noqa: E402
    CLAP_SAMPLE_RATE,
    DEFAULT_CLAP_MODEL_NAME,
    ClapEmbedder,
    load_audio_mono_resampled,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DEFAULT_CAPTIONER_MODEL_ID = "ACE-Step/acestep-captioner"
DEFAULT_HF_HOME = "/ocean/projects/mth240006p/xjiang6/huggingface"
CAPTION_PROMPT = "*Task* Describe this audio in detail"
USE_AUDIO_IN_VIDEO = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption audio with ACE-Step/acestep-captioner, generate music with WaveSpeed, then compute CLAP similarities."
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("dataset/small-5000/audio/00"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs"),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--captioner-model-id",
        type=str,
        default=DEFAULT_CAPTIONER_MODEL_ID,
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default=DEFAULT_HF_HOME,
    )
    parser.add_argument(
        "--wavespeed-model",
        type=str,
        default="wavespeed-ai/ace-step-1.5",
    )
    parser.add_argument(
        "--music-workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--clap-model-name",
        type=str,
        default=DEFAULT_CLAP_MODEL_NAME,
    )
    parser.add_argument(
        "--clap-device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--clap-batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--audio-load-batch-size",
        type=int,
        default=4,
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {k: json_ready(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_ready(payload), f, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_ready(row), ensure_ascii=False) + "\n")


def list_source_audio_files(audio_dir: Path, limit: int) -> list[dict]:
    paths = sorted(p for p in audio_dir.iterdir() if p.is_file())[:limit]
    rows: list[dict] = []
    for index, path in enumerate(paths):
        size_bytes = path.stat().st_size
        rows.append(
            {
                "index": index,
                "track_stem": path.stem,
                "source_audio_path": str(path.resolve()),
                "source_audio_size_bytes": size_bytes,
                "source_audio_size_mb": size_bytes / (1024.0 * 1024.0),
                "over_20mb": bool(size_bytes > 20 * 1024 * 1024),
            }
        )
    return rows


def split_source_audio_rows(source_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    used_rows: list[dict] = []
    skipped_rows: list[dict] = []
    for row in source_rows:
        if row["over_20mb"]:
            skipped_row = dict(row)
            skipped_row["skip_reason"] = "source_audio_size_exceeds_20mb"
            skipped_rows.append(skipped_row)
        else:
            used_rows.append(dict(row))
    return used_rows, skipped_rows


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


def load_captioner(model_id: str, cache_dir: str):
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    Qwen2_5OmniForConditionalGeneration.load_speakers = _skip_load_speakers_for_text_only
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype="auto",
        device_map="auto",
        use_safetensors=True,
    )
    model.disable_talker()
    processor = Qwen2_5OmniProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )
    return model, processor


def caption_audio_rows(source_rows: list[dict], model_id: str, hf_home: str) -> list[dict]:
    configure_hf_cache(hf_home)
    from qwen_omni_utils import process_mm_info

    model, processor = load_captioner(model_id=model_id, cache_dir=hf_home)
    rows: list[dict] = []
    iterator = source_rows
    if tqdm is not None:
        iterator = tqdm(source_rows, desc="caption", unit="track")

    for row in iterator:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {"type": "audio", "audio": row["source_audio_path"]},
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
            audios=audios,
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
        result = dict(row)
        result["generated_prompt"] = captions[0].strip()
        rows.append(result)
        if tqdm is None:
            print(f"[caption] {len(rows)}/{len(source_rows)}: {Path(row['source_audio_path']).name}", flush=True)

    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


def run_parallel_jobs(items: list[dict], worker_fn, max_workers: int, desc: str) -> list[dict]:
    if not items:
        return []

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_item = {executor.submit(worker_fn, item): item for item in items}
        completed = 0
        total = len(future_to_item)
        for future in concurrent.futures.as_completed(future_to_item):
            result = future.result()
            results.append(result)
            completed += 1
            print(f"[{desc}] {completed}/{total}: {Path(result['source_audio_path']).name}", flush=True)
    results.sort(key=lambda row: row["index"])
    return results


def get_wavespeed_api_key() -> str:
    api_key = os.environ.get("WAVESPEED_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("WAVESPEED_API_KEY is not set")
    return api_key


def build_wavespeed_payload(tags_text: str, duration: int) -> dict:
    return {
        "tags": tags_text or "",
        "lyrics": "[vocals]",
        "duration": duration,
        "seed": -1,
    }


def extract_output_suffix(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix
    return suffix if suffix else ".mp3"


def download_to_path(url: str, output_path: Path) -> None:
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(response.content)


def poll_wavespeed_result(request_id: str, poll_interval: float, api_key: str) -> dict:
    result_url = f"https://api.wavespeed.ai/api/v3/predictions/{request_id}/result"
    headers = {"Authorization": f"Bearer {api_key}"}

    while True:
        response = requests.get(result_url, headers=headers, timeout=120)
        response.raise_for_status()
        data = response.json()["data"]
        status = data["status"]
        if status == "completed":
            return data
        if status == "failed":
            raise RuntimeError(f"WaveSpeed task failed for request_id={request_id}: {data.get('error')}")
        time.sleep(poll_interval)


def generate_wavespeed_audio(
    sample: dict,
    prompt_text: str,
    output_dir: Path,
    wavespeed_model: str,
    duration: int,
    poll_interval: float,
    generation_type: str,
    api_key: str,
) -> dict:
    submit_url = f"https://api.wavespeed.ai/api/v3/{wavespeed_model}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = build_wavespeed_payload(tags_text=prompt_text, duration=duration)

    submit_response = requests.post(
        submit_url,
        headers=headers,
        data=json.dumps(payload),
        timeout=120,
    )
    if submit_response.status_code != 200:
        raise RuntimeError(
            "WaveSpeed submit failed: "
            f"status_code={submit_response.status_code}, body={submit_response.text}"
        )
    request_id = submit_response.json()["data"]["id"]

    result = poll_wavespeed_result(request_id=request_id, poll_interval=poll_interval, api_key=api_key)
    output_url = result["outputs"][0]
    file_suffix = extract_output_suffix(output_url)
    output_path = output_dir / f"{sample['index']:03d}_{sample['track_stem']}{file_suffix}"
    download_to_path(output_url, output_path)

    row = dict(sample)
    row["generation_type"] = generation_type
    row["wavespeed_request_id"] = request_id
    row["wavespeed_output_url"] = output_url
    row["generated_audio_path"] = str(output_path.resolve())
    row["wavespeed_payload"] = payload
    row["generation_tags"] = payload["tags"]
    return row


def generate_prompt_conditioned_music(
    prompt_rows: list[dict],
    output_dir: Path,
    wavespeed_model: str,
    duration: int,
    poll_interval: float,
    music_workers: int,
    api_key: str,
) -> list[dict]:
    def worker(row: dict) -> dict:
        return generate_wavespeed_audio(
            sample=row,
            prompt_text=row["generated_prompt"],
            output_dir=output_dir,
            wavespeed_model=wavespeed_model,
            duration=duration,
            poll_interval=poll_interval,
            generation_type="prompt_conditioned",
            api_key=api_key,
        )

    return run_parallel_jobs(
        items=prompt_rows,
        worker_fn=worker,
        max_workers=music_workers,
        desc="wavespeed-prompt",
    )


def generate_unconditional_music(
    source_rows: list[dict],
    output_dir: Path,
    wavespeed_model: str,
    duration: int,
    poll_interval: float,
    music_workers: int,
    api_key: str,
) -> list[dict]:
    def worker(row: dict) -> dict:
        return generate_wavespeed_audio(
            sample=row,
            prompt_text="",
            output_dir=output_dir,
            wavespeed_model=wavespeed_model,
            duration=duration,
            poll_interval=poll_interval,
            generation_type="unconditional",
            api_key=api_key,
        )

    return run_parallel_jobs(
        items=source_rows,
        worker_fn=worker,
        max_workers=music_workers,
        desc="wavespeed-uncond",
    )


def embed_audio_files_streaming(
    clap: ClapEmbedder,
    audio_paths: list[str],
    clap_batch_size: int,
    audio_load_batch_size: int,
    desc: str,
) -> np.ndarray:
    if not audio_paths:
        return np.empty((0, 512), dtype=np.float32)

    iterator = range(0, len(audio_paths), audio_load_batch_size)
    if tqdm is not None:
        total = (len(audio_paths) + audio_load_batch_size - 1) // audio_load_batch_size
        iterator = tqdm(iterator, total=total, desc=desc, unit="batch")

    chunks: list[np.ndarray] = []
    for batch_index, start in enumerate(iterator):
        end = min(start + audio_load_batch_size, len(audio_paths))
        batch_paths = audio_paths[start:end]
        audio_arrays = [load_audio_mono_resampled(Path(path), target_sr=CLAP_SAMPLE_RATE) for path in batch_paths]
        batch_embeddings = clap.embed_audio_arrays(
            audio_arrays,
            batch_size=max(1, min(clap_batch_size, len(audio_arrays))),
            show_progress=False,
        ).astype(np.float32, copy=False)
        chunks.append(batch_embeddings)
        del audio_arrays
        del batch_embeddings
        gc.collect()
        if tqdm is None and ((batch_index + 1) % 10 == 0 or end == len(audio_paths)):
            print(f"[clap] {desc}: {end}/{len(audio_paths)}", flush=True)

    return np.concatenate(chunks, axis=0)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


def paired_cosine_similarity(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    n = min(left.shape[0], right.shape[0])
    if n == 0:
        return np.empty((0,), dtype=np.float32)
    left_norm = normalize_rows(left[:n])
    right_norm = normalize_rows(right[:n])
    return np.sum(left_norm * right_norm, axis=1).astype(np.float32, copy=False)


def summarize_scores(scores: np.ndarray) -> dict:
    if scores.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p25": float("nan"),
            "median": float("nan"),
            "p75": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(scores.size),
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "p25": float(np.percentile(scores, 25)),
        "median": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "max": float(scores.max()),
    }


def write_pairwise_csv(
    output_path: Path,
    source_rows: list[dict],
    prompt_rows: list[dict],
    unconditional_rows: list[dict],
    prompt_scores: np.ndarray,
    unconditional_scores: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "track_stem",
        "source_audio_path",
        "generated_prompt",
        "prompt_conditioned_audio_path",
        "unconditional_audio_path",
        "prompt_conditioned_clap_cosine",
        "unconditional_clap_cosine",
        "source_audio_size_bytes",
        "over_20mb",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(source_rows)):
            source_row = source_rows[index]
            prompt_row = prompt_rows[index]
            unconditional_row = unconditional_rows[index]
            writer.writerow(
                {
                    "index": source_row["index"],
                    "track_stem": source_row["track_stem"],
                    "source_audio_path": source_row["source_audio_path"],
                    "generated_prompt": prompt_row["generated_prompt"],
                    "prompt_conditioned_audio_path": prompt_row["generated_audio_path"],
                    "unconditional_audio_path": unconditional_row["generated_audio_path"],
                    "prompt_conditioned_clap_cosine": float(prompt_scores[index]),
                    "unconditional_clap_cosine": float(unconditional_scores[index]),
                    "source_audio_size_bytes": source_row["source_audio_size_bytes"],
                    "over_20mb": source_row["over_20mb"],
                }
            )


def main() -> None:
    args = parse_args()
    wavespeed_api_key = get_wavespeed_api_key()

    run_name = args.run_name or f"acestep_captioner_wavespeed_clap_eval_{get_timestamp()}"
    output_dir = (args.output_root / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    caption_output_path = output_dir / "captions.jsonl"
    prompt_music_output_path = output_dir / "prompt_conditioned_generations.jsonl"
    unconditional_output_path = output_dir / "unconditional_generations.jsonl"
    pairwise_csv_path = output_dir / "pairwise_clap_similarity.csv"
    summary_json_path = output_dir / "summary.json"
    embeddings_path = output_dir / "clap_embeddings.npz"
    selected_source_audio_path = output_dir / "selected_source_audio.jsonl"
    used_source_audio_path = output_dir / "used_source_audio.jsonl"
    skipped_source_audio_path = output_dir / "skipped_source_audio.jsonl"
    prompt_music_dir = output_dir / "prompt_conditioned_audio"
    unconditional_music_dir = output_dir / "unconditional_audio"

    candidate_source_rows = list_source_audio_files(audio_dir=args.audio_dir.resolve(), limit=args.limit)
    if not candidate_source_rows:
        raise RuntimeError(f"No audio files found in {args.audio_dir}")

    source_rows, skipped_source_rows = split_source_audio_rows(candidate_source_rows)
    if not source_rows:
        raise RuntimeError(
            f"All {len(candidate_source_rows)} candidate audio files were skipped because they exceeded 20MB."
        )

    write_jsonl(selected_source_audio_path, candidate_source_rows)
    write_jsonl(used_source_audio_path, source_rows)
    write_jsonl(skipped_source_audio_path, skipped_source_rows)
    print(
        f"[source] candidate={len(candidate_source_rows)} used={len(source_rows)} "
        f"skipped_over_20mb={len(skipped_source_rows)} from {args.audio_dir}",
        flush=True,
    )

    print("[stage 1/4] generating ACE-Step captions", flush=True)
    caption_rows = caption_audio_rows(
        source_rows=source_rows,
        model_id=args.captioner_model_id,
        hf_home=args.hf_home,
    )
    write_jsonl(caption_output_path, caption_rows)

    print("[stage 2/4] generating WaveSpeed prompt-conditioned audio", flush=True)
    prompt_music_rows = generate_prompt_conditioned_music(
        prompt_rows=caption_rows,
        output_dir=prompt_music_dir,
        wavespeed_model=args.wavespeed_model,
        duration=args.duration,
        poll_interval=args.poll_interval,
        music_workers=args.music_workers,
        api_key=wavespeed_api_key,
    )
    write_jsonl(prompt_music_output_path, prompt_music_rows)

    print("[stage 3/4] generating WaveSpeed unconditional audio", flush=True)
    unconditional_rows = generate_unconditional_music(
        source_rows=source_rows,
        output_dir=unconditional_music_dir,
        wavespeed_model=args.wavespeed_model,
        duration=args.duration,
        poll_interval=args.poll_interval,
        music_workers=args.music_workers,
        api_key=wavespeed_api_key,
    )
    write_jsonl(unconditional_output_path, unconditional_rows)

    print("[stage 4/4] computing CLAP similarities", flush=True)
    clap_device = resolve_device(args.clap_device)
    print(f"[clap] loading model={args.clap_model_name} on device={clap_device}", flush=True)
    clap = ClapEmbedder(model_name=args.clap_model_name, device=clap_device)

    source_embeddings = embed_audio_files_streaming(
        clap=clap,
        audio_paths=[row["source_audio_path"] for row in source_rows],
        clap_batch_size=args.clap_batch_size,
        audio_load_batch_size=args.audio_load_batch_size,
        desc="source",
    )
    prompt_gen_embeddings = embed_audio_files_streaming(
        clap=clap,
        audio_paths=[row["generated_audio_path"] for row in prompt_music_rows],
        clap_batch_size=args.clap_batch_size,
        audio_load_batch_size=args.audio_load_batch_size,
        desc="prompt-conditioned",
    )
    unconditional_embeddings = embed_audio_files_streaming(
        clap=clap,
        audio_paths=[row["generated_audio_path"] for row in unconditional_rows],
        clap_batch_size=args.clap_batch_size,
        audio_load_batch_size=args.audio_load_batch_size,
        desc="unconditional",
    )

    np.savez_compressed(
        embeddings_path,
        source_audio_paths=np.array([row["source_audio_path"] for row in source_rows], dtype=np.str_),
        prompt_conditioned_audio_paths=np.array([row["generated_audio_path"] for row in prompt_music_rows], dtype=np.str_),
        unconditional_audio_paths=np.array([row["generated_audio_path"] for row in unconditional_rows], dtype=np.str_),
        source_embeddings=source_embeddings.astype(np.float32, copy=False),
        prompt_conditioned_embeddings=prompt_gen_embeddings.astype(np.float32, copy=False),
        unconditional_embeddings=unconditional_embeddings.astype(np.float32, copy=False),
    )

    prompt_scores = paired_cosine_similarity(source_embeddings, prompt_gen_embeddings)
    unconditional_scores = paired_cosine_similarity(source_embeddings, unconditional_embeddings)
    write_pairwise_csv(
        output_path=pairwise_csv_path,
        source_rows=source_rows,
        prompt_rows=prompt_music_rows,
        unconditional_rows=unconditional_rows,
        prompt_scores=prompt_scores,
        unconditional_scores=unconditional_scores,
    )

    summary = {
        "run_name": run_name,
        "created_at": datetime.now().isoformat(),
        "audio_dir": str(args.audio_dir.resolve()),
        "num_candidate_tracks": len(candidate_source_rows),
        "num_used_tracks": len(source_rows),
        "num_skipped_tracks": len(skipped_source_rows),
        "used_source_audio": source_rows,
        "skipped_source_audio": skipped_source_rows,
        "captioner_model_id": args.captioner_model_id,
        "caption_prompt": CAPTION_PROMPT,
        "hf_home": str(Path(args.hf_home).resolve()),
        "wavespeed_model": args.wavespeed_model,
        "wavespeed_duration": args.duration,
        "wavespeed_unconditional_pairing": "by_index",
        "music_workers": args.music_workers,
        "clap_model_name": args.clap_model_name,
        "clap_device": clap_device,
        "clap_sample_rate": CLAP_SAMPLE_RATE,
        "prompt_conditioned_similarity": summarize_scores(prompt_scores),
        "unconditional_similarity": summarize_scores(unconditional_scores),
        "mean_gap_prompt_minus_unconditional": float(prompt_scores.mean() - unconditional_scores.mean()),
        "output_files": {
            "selected_source_audio": str(selected_source_audio_path.resolve()),
            "used_source_audio": str(used_source_audio_path.resolve()),
            "skipped_source_audio": str(skipped_source_audio_path.resolve()),
            "captions": str(caption_output_path.resolve()),
            "prompt_conditioned_generations": str(prompt_music_output_path.resolve()),
            "unconditional_generations": str(unconditional_output_path.resolve()),
            "pairwise_clap_similarity_csv": str(pairwise_csv_path.resolve()),
            "clap_embeddings": str(embeddings_path.resolve()),
        },
    }
    write_json(summary_json_path, summary)

    print("[done] outputs written to:", output_dir, flush=True)
    print(json.dumps(json_ready(summary), indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
