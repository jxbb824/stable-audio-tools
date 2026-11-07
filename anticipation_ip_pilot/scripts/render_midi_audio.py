#!/usr/bin/env python3
"""Render Anticipation token lines to WAV via tinysoundfont (CPU, no system FluidSynth)."""
from __future__ import annotations

import argparse
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
import tinysoundfont
from tqdm import tqdm

from common import resolve_repo_path


def parse_tokens(line: str, *, is_generated: bool) -> list[int]:
    parts = line.strip().split()
    if not parts:
        return []
    if is_generated:
        return [int(x) for x in parts]
    if len(parts) < 3:
        return []
    return [int(x) for x in parts[1:-1]]


def improve_midi(midi_path: str) -> None:
    """Optional TheoryTab-style velocity / drum cleanup (from anticipation/midi_to_wav_converter)."""
    midi = pretty_midi.PrettyMIDI(midi_path)
    for instrument in midi.instruments:
        if instrument.program == 10:
            instrument.program = 0
            instrument.is_drum = True
            for note in instrument.notes:
                note.pitch = 34 if note.pitch == 36 else 33
                note.velocity = 70
        elif instrument.program == 0:
            for note in instrument.notes:
                note.velocity = 90
        elif instrument.program == 66:
            for note in instrument.notes:
                note.velocity = 60
    midi.write(midi_path)


def render_one(
    tokens: list[int],
    output_path: str,
    soundfont: str,
    sample_rate: int,
    max_seconds: float,
    improve: bool,
) -> str:
    # Import inside worker so each process gets a clean synth.
    from anticipation.convert import events_to_midi

    if Path(output_path).exists() and Path(output_path).stat().st_size > 0:
        return output_path

    mid = events_to_midi(tokens)
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        mid_path = tmp.name
    try:
        mid.save(mid_path)
        if improve:
            improve_midi(mid_path)

        synth = tinysoundfont.Synth(gain=-8.0, samplerate=int(sample_rate))
        synth.sfload(soundfont)
        seq = tinysoundfont.Sequencer(synth)
        seq.midi_load(mid_path)

        block = 1024
        chunks: list[np.ndarray] = []
        extra = int(0.25 * sample_rate)
        max_samples = int(max_seconds * sample_rate) * 2  # stereo interleaved floats
        n_samples = 0
        while ((not seq.is_empty()) or extra > 0) and n_samples < max_samples:
            seq.process(block / float(sample_rate))
            mv = synth.generate(block)
            arr = np.frombuffer(mv, dtype=np.float32).copy()
            chunks.append(arr)
            n_samples += arr.size
            if seq.is_empty():
                extra -= block

        if not chunks:
            audio = np.zeros(int(0.1 * sample_rate), dtype=np.float32)
        else:
            audio = np.concatenate(chunks).reshape(-1, 2).mean(axis=1)
            peak = float(np.max(np.abs(audio))) + 1e-8
            audio = np.clip(audio / peak * 0.9, -1.0, 1.0).astype(np.float32)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, audio, sample_rate)
    finally:
        try:
            os.remove(mid_path)
        except OSError:
            pass
    return output_path


def _worker(payload: tuple) -> tuple[int, str]:
    index, tokens, output_path, soundfont, sample_rate, max_seconds, improve = payload
    path = render_one(tokens, output_path, soundfont, sample_rate, max_seconds, improve)
    return index, path


def load_jobs(
    token_file: Path,
    output_dir: Path,
    indices: list[int] | None,
    *,
    is_generated: bool,
    name_prefix: str,
) -> list[tuple[int, list[int], str]]:
    lines = token_file.read_text(encoding="utf-8").splitlines()
    if indices is None:
        indices = list(range(len(lines)))
    jobs = []
    for local_i, global_i in enumerate(indices):
        if global_i < 0 or global_i >= len(lines):
            raise IndexError(f"index {global_i} out of range for {token_file}")
        tokens = parse_tokens(lines[global_i], is_generated=is_generated)
        out = output_dir / f"{name_prefix}{local_i:06d}.wav"
        jobs.append((local_i, tokens, str(out)))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Anticipation tokens to WAV.")
    parser.add_argument("--train-file", default="anticipation_ip_pilot/data/finetune/train_v2.txt")
    parser.add_argument(
        "--queries-file",
        default="anticipation_ip_pilot/outputs/queries/generated_samples_prompted_500.txt",
    )
    parser.add_argument(
        "--pool-indices-file",
        default="anticipation_ip_pilot/outputs/selection/seller_indices.txt",
    )
    parser.add_argument(
        "--output-root",
        default="anticipation_ip_pilot/outputs/baselines/clap/audio",
    )
    parser.add_argument(
        "--soundfont",
        default="anticipation_ip_pilot/assets/TimGM6mb.sf2",
    )
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--improve-midi", action="store_true", default=True)
    parser.add_argument("--no-improve-midi", action="store_false", dest="improve_midi")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-queries", action="store_true")
    args = parser.parse_args()

    soundfont = str(resolve_repo_path(args.soundfont))
    if not Path(soundfont).exists():
        # Fall back to pretty_midi bundled TimGM6mb.
        import pretty_midi as _pm

        bundled = Path(_pm.__file__).resolve().parent / "TimGM6mb.sf2"
        if bundled.exists():
            soundfont = str(bundled)
        else:
            raise FileNotFoundError(f"SoundFont not found: {args.soundfont}")

    output_root = resolve_repo_path(args.output_root)
    train_dir = output_root / "train"
    query_dir = output_root / "queries"
    train_dir.mkdir(parents=True, exist_ok=True)
    query_dir.mkdir(parents=True, exist_ok=True)

    all_payloads: list[tuple] = []

    if not args.skip_train:
        pool_text = resolve_repo_path(args.pool_indices_file).read_text(encoding="utf-8")
        pool_indices = [int(x) for x in pool_text.replace("\n", ",").split(",") if x.strip()]
        train_jobs = load_jobs(
            resolve_repo_path(args.train_file),
            train_dir,
            pool_indices,
            is_generated=False,
            name_prefix="train_",
        )
        for local_i, tokens, out in train_jobs:
            all_payloads.append(
                (local_i, tokens, out, soundfont, args.sample_rate, args.max_seconds, args.improve_midi)
            )
        print(f"[render] train jobs: {len(train_jobs)} -> {train_dir}")

    if not args.skip_queries:
        query_jobs = load_jobs(
            resolve_repo_path(args.queries_file),
            query_dir,
            None,
            is_generated=True,
            name_prefix="query_",
        )
        # Offset local indices in progress only; filenames already unique.
        for local_i, tokens, out in query_jobs:
            all_payloads.append(
                (local_i, tokens, out, soundfont, args.sample_rate, args.max_seconds, args.improve_midi)
            )
        print(f"[render] query jobs: {len(query_jobs)} -> {query_dir}")

    # Filter already-rendered.
    pending = [p for p in all_payloads if not (Path(p[2]).exists() and Path(p[2]).stat().st_size > 0)]
    print(f"[render] total={len(all_payloads)} pending={len(pending)} workers={args.workers}")

    if not pending:
        print("[render] nothing to do")
        return

    if args.workers <= 1:
        for payload in tqdm(pending, desc="render"):
            _worker(payload)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_worker, payload) for payload in pending]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="render"):
                fut.result()

    print("[render] done")


if __name__ == "__main__":
    main()
