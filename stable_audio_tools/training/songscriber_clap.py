import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy import linalg

DEFAULT_CLAP_MODEL_NAME = "laion/clap-htsat-fused"
CLAP_SAMPLE_RATE = 48000


@dataclass
class SongDescriberSample:
    caption_id: int
    track_id: int
    prompt: str
    duration: float
    audio_path: str


def resolve_song_describer_audio_path(audio_dir: Path, row_path: str, track_id: int) -> Path:
    p1 = audio_dir / row_path
    if p1.exists():
        return p1

    p2 = audio_dir / f"{track_id % 100:02d}" / f"{track_id}.2min.mp3"
    if p2.exists():
        return p2

    p3 = audio_dir / f"{track_id % 100:02d}" / f"{track_id}.mp3"
    if p3.exists():
        return p3

    raise FileNotFoundError(
        f"Reference audio not found for track_id={track_id}: tried {p1}, {p2}, {p3}"
    )


def load_song_describer_samples(csv_path: Path, audio_dir: Path, limit: int | None = None) -> list[SongDescriberSample]:
    rows: list[SongDescriberSample] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = int(row["track_id"])
            resolved_path = resolve_song_describer_audio_path(audio_dir, row["path"], track_id)
            rows.append(
                SongDescriberSample(
                    caption_id=int(row["caption_id"]),
                    track_id=track_id,
                    prompt=row["caption"],
                    duration=float(row["duration"]) if row.get("duration") else 0.0,
                    audio_path=str(resolved_path),
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def load_audio_mono_resampled(path: Path, target_sr: int = CLAP_SAMPLE_RATE) -> np.ndarray:
    try:
        wav, sr = torchaudio.load(str(path))
        if wav.ndim > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        return wav.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav).unsqueeze(0), sr, target_sr
            ).squeeze(0).numpy()
        return wav.astype(np.float32)


class ClapEmbedder:
    def __init__(
        self,
        model_name: str = DEFAULT_CLAP_MODEL_NAME,
        sample_rate: int = CLAP_SAMPLE_RATE,
        device: str | torch.device = "cuda",
    ) -> None:
        from transformers import ClapModel, ClapProcessor

        self.model_name = model_name
        self.sample_rate = sample_rate
        self.device = torch.device(device)
        self.model = ClapModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(model_name)

    def embed_audio_arrays(self, audios: Sequence[np.ndarray], batch_size: int = 8) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for i in range(0, len(audios), batch_size):
            chunk = list(audios[i : i + batch_size])
            inputs = self.processor(
                audios=chunk,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.model.get_audio_features(**inputs)
            outputs.append(emb.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 512), dtype=np.float32)

    def embed_audio_files(self, paths: Sequence[Path | str], batch_size: int = 8) -> np.ndarray:
        audios = [load_audio_mono_resampled(Path(p), self.sample_rate) for p in paths]
        return self.embed_audio_arrays(audios, batch_size=batch_size)

    def embed_audio_tensor(self, audio: torch.Tensor, sample_rate: int, batch_size: int = 8) -> np.ndarray:
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        if audio.ndim != 3:
            raise ValueError(f"Expected audio tensor [B, C, T] or [C, T], got {audio.shape}")

        audio = audio.detach().to(torch.float32)
        if audio.shape[1] > 1:
            audio = audio.mean(dim=1, keepdim=True)

        if sample_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.sample_rate)

        audios = [audio[idx, 0].cpu().numpy().astype(np.float32, copy=False) for idx in range(audio.shape[0])]
        return self.embed_audio_arrays(audios, batch_size=batch_size)

    def embed_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = list(texts[i : i + batch_size])
            inputs = self.processor(
                text=chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.model.get_text_features(**inputs)
            outputs.append(emb.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 512), dtype=np.float32)


def clap_fad(ref_emb: np.ndarray, eval_emb: np.ndarray, eps: float = 1e-6) -> float:
    if ref_emb.shape[0] < 2 or eval_emb.shape[0] < 2:
        return float("nan")

    mu1, sigma1 = np.mean(ref_emb, axis=0), np.cov(ref_emb, rowvar=False)
    mu2, sigma2 = np.mean(eval_emb, axis=0), np.cov(eval_emb, rowvar=False)
    diff = np.atleast_1d(mu1) - np.atleast_1d(mu2)

    covmean, _ = linalg.sqrtm(
        np.atleast_2d(sigma1).dot(np.atleast_2d(sigma2)).astype(complex),
        disp=False,
    )
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset).astype(complex))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(
        diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    )


def clap_alignment(text_emb: np.ndarray, audio_emb: np.ndarray) -> tuple[float, float]:
    if text_emb.shape[0] == 0 or audio_emb.shape[0] == 0:
        return float("nan"), float("nan")

    n = min(text_emb.shape[0], audio_emb.shape[0])
    text_emb = text_emb[:n]
    audio_emb = audio_emb[:n]

    t = text_emb / (np.linalg.norm(text_emb, axis=-1, keepdims=True) + 1e-8)
    a = audio_emb / (np.linalg.norm(audio_emb, axis=-1, keepdims=True) + 1e-8)
    scores = (a @ t.T).diagonal()
    return float(scores.mean()), float(scores.std())


def save_precomputed_song_describer_embeddings(
    output_path: Path,
    samples: Sequence[SongDescriberSample],
    audio_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    clap_model_name: str = DEFAULT_CLAP_MODEL_NAME,
    clap_sample_rate: int = CLAP_SAMPLE_RATE,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        clap_model_name=np.array([clap_model_name]),
        clap_sample_rate=np.array([clap_sample_rate], dtype=np.int32),
        caption_ids=np.array([s.caption_id for s in samples], dtype=np.int64),
        track_ids=np.array([s.track_id for s in samples], dtype=np.int64),
        durations=np.array([s.duration for s in samples], dtype=np.float32),
        prompts=np.array([s.prompt for s in samples], dtype=np.str_),
        audio_paths=np.array([s.audio_path for s in samples], dtype=np.str_),
        audio_embeddings=audio_embeddings.astype(np.float32),
        text_embeddings=text_embeddings.astype(np.float32),
    )


def load_precomputed_song_describer_embeddings(precomputed_path: Path) -> dict[str, np.ndarray]:
    data = np.load(precomputed_path, allow_pickle=False)
    required_keys = {
        "caption_ids",
        "track_ids",
        "durations",
        "prompts",
        "audio_paths",
        "audio_embeddings",
        "text_embeddings",
    }
    missing = sorted(list(required_keys - set(data.files)))
    if missing:
        raise KeyError(f"Missing keys in {precomputed_path}: {missing}")

    return {
        "caption_ids": data["caption_ids"],
        "track_ids": data["track_ids"],
        "durations": data["durations"],
        "prompts": data["prompts"],
        "audio_paths": data["audio_paths"],
        "audio_embeddings": data["audio_embeddings"].astype(np.float32),
        "text_embeddings": data["text_embeddings"].astype(np.float32),
    }
