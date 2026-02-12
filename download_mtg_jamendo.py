#!/usr/bin/env python3

import argparse
import tarfile
import urllib.request
from pathlib import Path


AUDIO_BASE_URL = "https://cdn.freesound.org/mtg-jamendo/raw_30s/audio/"
TOTAL_TARS = 100
CHUNK_SIZE = 1024 * 1024
REPORT_EVERY = 100 * 1024 * 1024


def download_file(url, dst):
    print(f"Downloading file: {url}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dst.open("wb") as out:
        total = resp.getheader("Content-Length")
        total = int(total) if total else None
        downloaded = 0
        next_report = REPORT_EVERY
        while True:
            chunk = resp.read(CHUNK_SIZE)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if downloaded >= next_report:
                if total:
                    print(
                        f"  {downloaded / (1024**2):.1f} MB / {total / (1024**2):.1f} MB"
                    )
                else:
                    print(f"  {downloaded / (1024**2):.1f} MB")
                next_report += REPORT_EVERY
    print(f"Saved tar: {dst}")


def safe_member_path(name):
    path = Path(name).as_posix()
    path = Path(path)
    if path.is_absolute() or ".." in path.parts:
        return None
    return path


def extract_tar(tar_path, audio_dir):
    added = 0
    with tarfile.open(tar_path, "r") as tf:
        for member in tf.getmembers():
            if not member.isfile() or not member.name.lower().endswith(".mp3"):
                continue
            safe_path = safe_member_path(member.name)
            if safe_path is None:
                continue

            output_path = audio_dir / safe_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

            src = tf.extractfile(member)
            if src is None:
                continue

            with src, output_path.open("wb") as dst:
                while True:
                    chunk = src.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    dst.write(chunk)
            added += 1
    return added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--audio-dir", default="dataset/small-5000/audio")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    tar_dir = audio_dir.parent / "_tar_cache"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tar_dir.mkdir(parents=True, exist_ok=True)

    print(f"Audio output: {audio_dir}")
    print(f"Target tracks: ~{args.target}")

    total_tracks = 0
    for i in range(TOTAL_TARS):
        if total_tracks >= args.target:
            break

        tar_name = f"raw_30s_audio-{i:02d}.tar"
        print(f"\n[{i + 1}/{TOTAL_TARS}] {tar_name}")
        tar_path = tar_dir / tar_name
        download_file(AUDIO_BASE_URL + tar_name, tar_path)

        added = extract_tar(tar_path, audio_dir)
        total_tracks += added
        print(f"Extracted from {tar_name}: {added} tracks")
        print(f"Running total: {total_tracks} tracks")

        if tar_path.exists():
            tar_path.unlink()

    print(f"\nDone. Downloaded about {total_tracks} tracks.")


if __name__ == "__main__":
    main()
