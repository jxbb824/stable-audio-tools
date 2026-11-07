#!/usr/bin/env python3
"""CLAP audio-audio cosine similarity as train×query attribution scores (GPU)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from transformers import AutoFeatureExtractor, ClapModel

from common import resolve_repo_path


def list_wavs(directory: Path, prefix: str) -> list[Path]:
    files = sorted(directory.glob(f"{prefix}*.wav"))
    if not files:
        raise FileNotFoundError(f"No {prefix}*.wav under {directory}")
    return files


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
        audios = []
        for path in batch_paths:
            try:
                audios.append(load_mono_resample(path))
            except Exception as exc:  # noqa: BLE001
                print(f"[clap] failed {path}: {exc}; using silence")
                audios.append(np.zeros(48000, dtype=np.float32))
        inputs = feature_extractor(audios, sampling_rate=48000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            feat = model.get_audio_features(**inputs)
        feats.append(feat.detach().cpu())
    return torch.cat(feats, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLAP similarity attribution scores.")
    parser.add_argument(
        "--audio-root",
        default="anticipation_ip_pilot/outputs/baselines/clap/audio",
    )
    parser.add_argument(
        "--output-path",
        default="anticipation_ip_pilot/outputs/baselines/clap/score_clap.pt",
    )
    parser.add_argument("--model-id", default="laion/clap-htsat-unfused")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="", help="cuda / cpu; default auto")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("score_clap.py expects a CUDA GPU. Submit via clap_attribution.slurm.")

    audio_root = resolve_repo_path(args.audio_root)
    train_paths = list_wavs(audio_root / "train", "train_")
    query_paths = list_wavs(audio_root / "queries", "query_")
    print(f"[clap] train={len(train_paths)} queries={len(query_paths)} device={device}")

    model = ClapModel.from_pretrained(args.model_id).to(device).eval()
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_id)

    train_feat = embed_paths(train_paths, model, feature_extractor, device, args.batch_size)
    query_feat = embed_paths(query_paths, model, feature_extractor, device, args.batch_size)
    train_n = F.normalize(train_feat.float(), p=2, dim=1)
    query_n = F.normalize(query_feat.float(), p=2, dim=1)
    score = train_n @ query_n.T  # [n_train, n_query]

    out = resolve_repo_path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(score, out)
    emb_dir = out.parent
    torch.save(train_feat, emb_dir / "train_clap_embeddings.pt")
    torch.save(query_feat, emb_dir / "query_clap_embeddings.pt")
    print(f"[clap] wrote {out} shape={tuple(score.shape)}")


if __name__ == "__main__":
    main()
