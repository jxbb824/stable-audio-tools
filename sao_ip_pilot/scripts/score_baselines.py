#!/usr/bin/env python3
"""SAO IP baselines: random attribution and CLAP audio-audio similarity."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, ClapModel

from common import resolve_repo_path


def load_mono_resample(path: Path, target_sr: int = 48000) -> np.ndarray:
    waveform, sample_rate = torchaudio.load(str(path))
    if waveform.ndim > 1 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    return waveform.squeeze(0).numpy().astype(np.float32)


def embed_paths(
    paths: list[Path],
    model: ClapModel,
    feature_extractor: AutoFeatureExtractor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    feats: list[torch.Tensor] = []
    for start in tqdm(range(0, len(paths), batch_size), desc=f"embed[{device}]"):
        batch_paths = paths[start : start + batch_size]
        # Fail closed: substituting silence would silently corrupt the baseline.
        audios = [load_mono_resample(path) for path in batch_paths]
        inputs = feature_extractor(audios, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feat = model.get_audio_features(**inputs)
        feats.append(feat.detach().cpu())
    return torch.cat(feats, dim=0)


def resolve_audio_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return resolve_repo_path(raw)


def resolve_query_audio_paths(query_manifest: pd.DataFrame) -> list[Path]:
    """Resolve each query, falling back from stale absolute paths to query_id."""
    paths: list[Path] = []
    for row_index, row in query_manifest.iterrows():
        # query_id is repository-relative and therefore portable across project moves.
        candidates = [row.get("query_id"), row.get("audio_path")]
        resolved = [resolve_audio_path(str(raw)) for raw in candidates if pd.notna(raw) and str(raw)]
        match = next((path for path in resolved if path.is_file()), None)
        if match is None:
            raise FileNotFoundError(
                f"query row {row_index} has no existing audio path; tried "
                + ", ".join(str(path) for path in resolved)
            )
        paths.append(match)
    return paths


def write_random(args: argparse.Namespace) -> Path:
    ref = torch.load(resolve_repo_path(args.reference_score_path), map_location="cpu", weights_only=False)
    if not torch.is_tensor(ref):
        raise TypeError(f"Expected tensor at {args.reference_score_path}")
    generator = torch.Generator().manual_seed(args.seed)
    score = torch.rand(ref.shape, generator=generator, dtype=torch.float32)
    out = resolve_repo_path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(score, out)
    print(f"[sao-random] wrote {out} shape={tuple(score.shape)}")
    return out


def write_clap(args: argparse.Namespace) -> Path:
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("SAO CLAP baseline expects CUDA. Submit via clap_attribution.slurm.")

    train_manifest = pd.read_csv(resolve_repo_path(args.train_axis_manifest))
    query_manifest = pd.read_csv(resolve_repo_path(args.query_axis_manifest))
    train_paths = [resolve_audio_path(p) for p in train_manifest["path"].tolist()]
    query_paths = resolve_query_audio_paths(query_manifest)

    missing_train = [str(p) for p in train_paths if not p.exists()]
    missing_query = [str(p) for p in query_paths if not p.exists()]
    if missing_train:
        raise FileNotFoundError(f"{len(missing_train)} train audio missing, e.g. {missing_train[0]}")
    if missing_query:
        raise FileNotFoundError(f"{len(missing_query)} query audio missing, e.g. {missing_query[0]}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(
        f"[sao-clap] train={len(train_paths)} queries={len(query_paths)} "
        f"device={device} seed={args.seed}"
    )
    model = ClapModel.from_pretrained(args.model_id).to(device).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)
    train_feat = embed_paths(train_paths, model, feature_extractor, device, args.batch_size)
    query_feat = embed_paths(query_paths, model, feature_extractor, device, args.batch_size)
    score = F.normalize(train_feat.float(), p=2, dim=1) @ F.normalize(query_feat.float(), p=2, dim=1).T

    out = resolve_repo_path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(score, out)
    torch.save(train_feat, out.parent / "train_clap_embeddings.pt")
    torch.save(query_feat, out.parent / "query_clap_embeddings.pt")
    print(f"[sao-clap] wrote {out} shape={tuple(score.shape)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="SAO IP random / CLAP baseline scores.")
    parser.add_argument("--mode", choices=["random", "clap"], required=True)
    parser.add_argument(
        "--reference-score-path",
        default="sao_ip_pilot/outputs/ekfac_attribution/scores_train_x_query.pt",
    )
    parser.add_argument(
        "--train-axis-manifest",
        default="sao_ip_pilot/outputs/ekfac_attribution/train_axis_manifest.csv",
    )
    parser.add_argument(
        "--query-axis-manifest",
        default="sao_ip_pilot/outputs/ekfac_attribution/query_axis_manifest.csv",
    )
    parser.add_argument("--output-path", default="")
    parser.add_argument("--model-id", default="laion/clap-htsat-unfused")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.output_path:
        args.output_path = (
            "sao_ip_pilot/outputs/baselines/random/score_random.pt"
            if args.mode == "random"
            else "sao_ip_pilot/outputs/baselines/clap/score_clap.pt"
        )

    if args.mode == "random":
        write_random(args)
    else:
        write_clap(args)


if __name__ == "__main__":
    main()
