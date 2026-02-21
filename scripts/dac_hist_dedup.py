#!/usr/bin/env python3
"""
Deduplicate audio files with DAC-code histogram similarity.
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple

import dac
import torch
import torchaudio

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DEFAULT_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus", ".aif", ".aiff"}
DEFAULT_EXTENSIONS_ARG = ",".join(sorted(ext.lstrip(".") for ext in DEFAULT_EXTENSIONS))


def log_error(message: str) -> None:
    if tqdm is not None:
        tqdm.write(message, file=sys.stderr)
    else:
        print(message, file=sys.stderr)


@dataclass(frozen=True)
class AudioEntry:
    path: str
    root: str
    relpath: str


def parse_extensions(extensions_str: str) -> Set[str]:
    exts = set()
    for item in extensions_str.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = "." + item
        exts.add(item)
    if not exts:
        raise ValueError("No valid file extensions were provided.")
    return exts


def load_audio_roots_from_dataset_config(dataset_config_path: str) -> List[str]:
    with open(dataset_config_path, "r") as f:
        dataset_config = json.load(f)

    dataset_type = dataset_config.get("dataset_type")
    if dataset_type != "audio_dir":
        raise ValueError(
            f"Only dataset_type='audio_dir' is supported. Got: {dataset_type!r}"
        )

    roots = []
    for dataset in dataset_config.get("datasets", []):
        path = dataset.get("path")
        if path is None:
            continue
        roots.append(os.path.abspath(path))

    if not roots:
        raise ValueError(f"No dataset paths found in {dataset_config_path}")

    return roots


def discover_audio_files(roots: Sequence[str], extensions: Set[str]) -> List[AudioEntry]:
    entries: List[AudioEntry] = []
    for root in roots:
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise NotADirectoryError(f"Dataset root does not exist: {root}")

        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                if filename.startswith("."):
                    continue
                ext = os.path.splitext(filename)[1].lower()
                if ext not in extensions:
                    continue
                full_path = os.path.normpath(os.path.join(dirpath, filename))
                relpath = os.path.normpath(os.path.relpath(full_path, root))
                entries.append(AudioEntry(path=full_path, root=root, relpath=relpath))

    entries.sort(key=lambda x: x.path)
    return entries


def load_dac_model(model_type: str, model_path: Optional[str], device: str):
    resolved_model_path = model_path
    if resolved_model_path is None:
        resolved_model_path = dac.utils.download(model_type=model_type)

    resolved_model_path = str(resolved_model_path)

    model = dac.DAC.load(resolved_model_path)
    model = model.to(device)
    model.eval()
    return model, resolved_model_path


def _normalize_codes_shape(codes: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(codes):
        codes = torch.as_tensor(codes)

    codes = codes.detach().long().cpu()

    while codes.ndim > 2:
        squeezed = False
        for dim, size in enumerate(codes.shape):
            if size == 1:
                codes = codes.squeeze(dim)
                squeezed = True
                break
        if not squeezed:
            codes = codes[0]

    if codes.ndim != 2:
        raise ValueError(f"Unexpected DAC code shape: {tuple(codes.shape)}")

    # Prefer [num_codebooks, num_frames].
    if codes.shape[0] > codes.shape[1]:
        codes = codes.transpose(0, 1)

    return codes.contiguous()


def _get_model_input_channels(model) -> int:
    for attr in ("audio_channels", "in_channels", "input_channels"):
        value = getattr(model, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        for module in encoder.modules():
            if isinstance(module, torch.nn.Conv1d):
                return int(module.in_channels)

    for module in model.modules():
        if isinstance(module, torch.nn.Conv1d):
            return int(module.in_channels)

    return 1


def _adapt_waveform_channels(waveform: torch.Tensor, target_channels: int) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform shape [channels, time], got: {tuple(waveform.shape)}")

    current_channels = int(waveform.shape[0])
    if current_channels == target_channels:
        return waveform

    if target_channels == 1:
        # Downmix to mono for mono DAC models.
        return waveform.mean(dim=0, keepdim=True)

    if current_channels == 1 and target_channels > 1:
        # Expand mono to multi-channel if model expects more channels.
        return waveform.repeat(target_channels, 1)

    if current_channels > target_channels:
        # Trim extra channels.
        return waveform[:target_channels]

    repeats = target_channels // current_channels
    remainder = target_channels % current_channels
    expanded = waveform.repeat(repeats, 1)
    if remainder > 0:
        expanded = torch.cat([expanded, waveform[:remainder]], dim=0)
    return expanded


def build_dac_histogram_feature(
    audio_path: str,
    model,
    device: str,
    codebook_size: int = 1024,
    max_audio_seconds: Optional[float] = None,
    histogram_mode: str = "per_codebook",
) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.numel() == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")

    model_sample_rate = getattr(model, "sample_rate", None)
    if isinstance(model_sample_rate, int) and model_sample_rate > 0 and sample_rate != model_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, model_sample_rate)
        sample_rate = model_sample_rate

    if max_audio_seconds is not None:
        max_samples = int(max_audio_seconds * sample_rate)
        waveform = waveform[:, :max_samples]
        if waveform.numel() == 0:
            raise ValueError(f"Audio is empty after cropping: {audio_path}")

    target_channels = _get_model_input_channels(model)
    waveform = _adapt_waveform_channels(waveform, target_channels)

    waveform = waveform.unsqueeze(0).to(device)

    try:
        with torch.inference_mode():
            x = model.preprocess(waveform, sample_rate)
            _, codes, _, _, _ = model.encode(x)
    except Exception as exc:
        raise RuntimeError(
            "DAC encode failed "
            f"(path={audio_path}, sample_rate={sample_rate}, "
            f"target_channels={target_channels}, input_shape={tuple(waveform.shape)})"
        ) from exc

    codes = _normalize_codes_shape(codes)
    max_code = int(codes.max().item())
    if max_code >= codebook_size:
        raise ValueError(
            f"Observed code {max_code} but codebook_size is {codebook_size}. Increase --codebook-size."
        )

    if histogram_mode == "global":
        feature = torch.bincount(codes.flatten(), minlength=codebook_size).float()
        feature = feature / feature.sum().clamp_min(1.0)
    elif histogram_mode == "per_codebook":
        histograms = []
        for i in range(codes.shape[0]):
            hist = torch.bincount(codes[i], minlength=codebook_size).float()
            hist = hist / hist.sum().clamp_min(1.0)
            histograms.append(hist)
        feature = torch.cat(histograms, dim=0)
    else:
        raise ValueError(f"Unknown histogram mode: {histogram_mode}")

    feature = torch.nn.functional.normalize(feature, p=2, dim=0)
    return feature.cpu()


def cosine_similarity_between_files(
    file_a: str,
    file_b: str,
    model,
    device: str,
    codebook_size: int = 1024,
    max_audio_seconds: Optional[float] = None,
    histogram_mode: str = "per_codebook",
) -> float:
    feat_a = build_dac_histogram_feature(
        file_a,
        model=model,
        device=device,
        codebook_size=codebook_size,
        max_audio_seconds=max_audio_seconds,
        histogram_mode=histogram_mode,
    )
    feat_b = build_dac_histogram_feature(
        file_b,
        model=model,
        device=device,
        codebook_size=codebook_size,
        max_audio_seconds=max_audio_seconds,
        histogram_mode=histogram_mode,
    )
    return float(torch.dot(feat_a, feat_b).item())


def extract_feature_matrix(
    entries: Sequence[AudioEntry],
    model,
    device: str,
    codebook_size: int,
    max_audio_seconds: Optional[float],
    histogram_mode: str,
) -> Tuple[List[AudioEntry], torch.Tensor, List[dict]]:
    kept_entries: List[AudioEntry] = []
    features: List[torch.Tensor] = []
    failed_items: List[dict] = []

    start_time = time.time()
    iterator = enumerate(entries, start=1)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=len(entries),
            desc="encode",
            dynamic_ncols=True,
        )

    for idx, entry in iterator:
        if tqdm is None and (idx == 1 or idx % 25 == 0 or idx == len(entries)):
            elapsed = time.time() - start_time
            print(f"[encode] {idx}/{len(entries)} (elapsed: {elapsed:.1f}s)")

        try:
            feature = build_dac_histogram_feature(
                audio_path=entry.path,
                model=model,
                device=device,
                codebook_size=codebook_size,
                max_audio_seconds=max_audio_seconds,
                histogram_mode=histogram_mode,
            )
            kept_entries.append(entry)
            features.append(feature)
        except Exception as exc:
            failed_items.append(
                {
                    "path": entry.path,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            log_error(
                f"[encode-failed] {idx}/{len(entries)} "
                f"path={entry.path} "
                f"error={type(exc).__name__}: {exc}"
            )

        if tqdm is not None and idx % 10 == 0:
            iterator.set_postfix(ok=len(kept_entries), failed=len(failed_items), refresh=False)

    if tqdm is not None:
        iterator.set_postfix(ok=len(kept_entries), failed=len(failed_items), refresh=False)

    if not features:
        raise RuntimeError("No features were extracted successfully.")

    return kept_entries, torch.stack(features, dim=0), failed_items


def find_duplicate_pairs(
    features: torch.Tensor,
    threshold: float,
    similarity_device: str,
    similarity_chunk_size: int,
) -> Tuple[List[Tuple[int, int, float]], torch.Tensor]:
    num_items = int(features.shape[0])
    duplicate_mask = torch.zeros(num_items, dtype=torch.bool)
    duplicate_pairs: List[Tuple[int, int, float]] = []

    if similarity_device == "auto":
        similarity_device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(similarity_device)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    feature_matrix = features.to(device=device, dtype=dtype)

    start_time = time.time()
    iterator = range(num_items)
    if tqdm is not None:
        iterator = tqdm(
            iterator,
            total=num_items,
            desc="match",
            dynamic_ncols=True,
        )

    for i in iterator:
        if duplicate_mask[i]:
            continue
        if i + 1 >= num_items:
            break

        ref = feature_matrix[i]

        for start in range(i + 1, num_items, similarity_chunk_size):
            end = min(num_items, start + similarity_chunk_size)
            sims = torch.mv(feature_matrix[start:end], ref).float().cpu()
            hit_rel_idxs = torch.where(sims >= threshold)[0]

            for rel_idx in hit_rel_idxs.tolist():
                j = start + rel_idx
                if duplicate_mask[j]:
                    continue
                duplicate_mask[j] = True
                duplicate_pairs.append((i, j, float(sims[rel_idx].item())))

        if tqdm is None and ((i + 1) % 100 == 0 or (i + 1) == num_items):
            elapsed = time.time() - start_time
            print(
                f"[match] {i + 1}/{num_items} processed, "
                f"{len(duplicate_pairs)} duplicates found (elapsed: {elapsed:.1f}s)"
            )
        elif tqdm is not None and (i + 1) % 25 == 0:
            iterator.set_postfix(duplicates=len(duplicate_pairs), refresh=False)

    if tqdm is not None:
        iterator.set_postfix(duplicates=len(duplicate_pairs), refresh=False)

    return duplicate_pairs, duplicate_mask


def _ensure_parent_dir(file_path: str) -> None:
    parent_dir = os.path.dirname(os.path.abspath(file_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def write_outputs(
    entries: Sequence[AudioEntry],
    duplicate_pairs: Sequence[Tuple[int, int, float]],
    failed_items: Sequence[dict],
    summary_payload: dict,
    exclude_output: str,
    pairs_output: str,
    failed_output: str,
    summary_output: str,
    exclude_path_format: str,
) -> None:
    _ensure_parent_dir(exclude_output)
    _ensure_parent_dir(pairs_output)
    _ensure_parent_dir(failed_output)
    _ensure_parent_dir(summary_output)

    drop_indices = sorted({j for _, j, _ in duplicate_pairs})

    with open(exclude_output, "w") as f:
        f.write(f"# Paths to exclude from training (format={exclude_path_format})\n")
        for idx in drop_indices:
            if exclude_path_format == "relative":
                item = entries[idx].relpath
            elif exclude_path_format == "absolute":
                item = entries[idx].path
            else:
                raise ValueError(f"Unknown exclude path format: {exclude_path_format}")
            f.write(f"{item}\n")

    with open(pairs_output, "w") as f:
        for keep_idx, drop_idx, sim in duplicate_pairs:
            record = {
                "keep_path": entries[keep_idx].path,
                "keep_relpath": entries[keep_idx].relpath,
                "drop_path": entries[drop_idx].path,
                "drop_relpath": entries[drop_idx].relpath,
                "similarity": sim,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    with open(failed_output, "w") as f:
        json.dump(list(failed_items), f, indent=2)

    with open(summary_output, "w") as f:
        json.dump(summary_payload, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="DAC histogram-based audio deduplication.")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--dataset-config",
        type=str,
        help="Path to dataset config JSON. Must be dataset_type='audio_dir'.",
    )
    source_group.add_argument(
        "--dataset-dir",
        type=str,
        nargs="+",
        help="One or more audio directories to scan recursively.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/dac_dedup",
        help="Directory for generated output files.",
    )
    parser.add_argument(
        "--exclude-output",
        type=str,
        default=None,
        help="Path to output exclusion list text file.",
    )
    parser.add_argument(
        "--pairs-output",
        type=str,
        default=None,
        help="Path to output JSONL file with similar pairs.",
    )
    parser.add_argument(
        "--failed-output",
        type=str,
        default=None,
        help="Path to output JSON file listing files that failed to encode.",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Path to output JSON summary file.",
    )

    parser.add_argument(
        "--extensions",
        type=str,
        default=DEFAULT_EXTENSIONS_ARG,
        help="Comma-separated audio extensions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of files to process after sorting.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="44khz",
        help="DAC model type used by dac.utils.download (e.g., 44khz).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional local DAC model path. If omitted, downloads by --model-type.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for DAC encoding.",
    )
    parser.add_argument(
        "--similarity-device",
        type=str,
        default="auto",
        help="Device for similarity matching: auto/cpu/cuda.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.985,
        help="Cosine similarity threshold to mark duplicates.",
    )
    parser.add_argument(
        "--similarity-chunk-size",
        type=int,
        default=2048,
        help="Chunk size for pairwise matching.",
    )
    parser.add_argument(
        "--codebook-size",
        type=int,
        default=1024,
        help="DAC codebook size for histogram bins.",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=120.0,
        help="Max seconds from start of each file used for matching. Use <=0 for full file.",
    )
    parser.add_argument(
        "--histogram-mode",
        type=str,
        choices=["global", "per_codebook"],
        default="per_codebook",
        help="Histogram construction mode.",
    )
    parser.add_argument(
        "--exclude-path-format",
        type=str,
        choices=["relative", "absolute"],
        default="relative",
        help="Path format used in exclude output file.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.exclude_output is None:
        args.exclude_output = os.path.join(args.output_dir, "exclude_paths.txt")
    if args.pairs_output is None:
        args.pairs_output = os.path.join(args.output_dir, "similar_pairs.jsonl")
    if args.failed_output is None:
        args.failed_output = os.path.join(args.output_dir, "failed_files.json")
    if args.summary_output is None:
        args.summary_output = os.path.join(args.output_dir, "summary.json")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset_config is not None:
        roots = load_audio_roots_from_dataset_config(args.dataset_config)
    else:
        roots = [os.path.abspath(x) for x in args.dataset_dir]

    extensions = parse_extensions(args.extensions)
    entries = discover_audio_files(roots, extensions)
    total_scanned = len(entries)

    if args.limit is not None:
        entries = entries[: args.limit]

    if not entries:
        raise RuntimeError("No audio files found.")

    max_audio_seconds = args.max_audio_seconds
    if max_audio_seconds is not None and max_audio_seconds <= 0:
        max_audio_seconds = None

    print(f"Found {total_scanned} audio files, processing {len(entries)} files.")
    print(f"Using roots: {roots}")
    print(f"Using DAC device={args.device}, similarity_device={args.similarity_device}.")

    model, resolved_model_path = load_dac_model(
        model_type=args.model_type,
        model_path=args.model_path,
        device=args.device,
    )
    print(f"Loaded DAC model from: {resolved_model_path}")

    kept_entries, features, failed_items = extract_feature_matrix(
        entries=entries,
        model=model,
        device=args.device,
        codebook_size=args.codebook_size,
        max_audio_seconds=max_audio_seconds,
        histogram_mode=args.histogram_mode,
    )

    print(f"Successfully encoded {len(kept_entries)} files, failed {len(failed_items)} files.")

    duplicate_pairs, duplicate_mask = find_duplicate_pairs(
        features=features,
        threshold=args.similarity_threshold,
        similarity_device=args.similarity_device,
        similarity_chunk_size=args.similarity_chunk_size,
    )

    num_duplicates = int(duplicate_mask.sum().item())
    num_kept = len(kept_entries) - num_duplicates

    summary_payload = {
        "dataset_config": args.dataset_config,
        "dataset_dirs": roots if args.dataset_config is None else None,
        "total_scanned_files": total_scanned,
        "processed_files": len(entries),
        "encoded_files": len(kept_entries),
        "failed_files": len(failed_items),
        "duplicate_files": num_duplicates,
        "kept_files": num_kept,
        "similarity_threshold": args.similarity_threshold,
        "histogram_mode": args.histogram_mode,
        "codebook_size": args.codebook_size,
        "max_audio_seconds": max_audio_seconds,
        "model_type": args.model_type,
        "model_path": resolved_model_path,
        "exclude_output": os.path.abspath(args.exclude_output),
        "pairs_output": os.path.abspath(args.pairs_output),
        "failed_output": os.path.abspath(args.failed_output),
        "summary_output": os.path.abspath(args.summary_output),
        "exclude_path_format": args.exclude_path_format,
    }

    write_outputs(
        entries=kept_entries,
        duplicate_pairs=duplicate_pairs,
        failed_items=failed_items,
        summary_payload=summary_payload,
        exclude_output=args.exclude_output,
        pairs_output=args.pairs_output,
        failed_output=args.failed_output,
        summary_output=args.summary_output,
        exclude_path_format=args.exclude_path_format,
    )

    print("Dedup finished.")
    print(json.dumps(summary_payload, indent=2))


if __name__ == "__main__":
    main()
