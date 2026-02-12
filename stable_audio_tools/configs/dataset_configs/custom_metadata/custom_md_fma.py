from pathlib import Path
import random
import re


_DATASET_DIR = Path(__file__).resolve().parents[4] / "dataset"
_RAW_TSV = _DATASET_DIR / "raw.tsv"
_AUDIO_METADATA_TSV = _DATASET_DIR / "audio_metadata.tsv"


def _track_id_to_int(track_id):
    match = re.search(r"(\d+)$", track_id or "")
    return int(match.group(1)) if match else None


def _extract_track_num_from_path(audio_path):
    match = re.search(r"(\d+)", Path(audio_path or "").name)
    return int(match.group(1)) if match else None


def _clean_text(text):
    if text is None:
        return None
    text = str(text).strip()
    return text if text else None


def _clean_tag(tag):
    tag = _clean_text(tag)
    if tag is None:
        return None
    tag = tag.replace("_", " ")
    return " ".join(tag.split())


def _load_raw_metadata():
    by_num = {}
    if not _RAW_TSV.exists():
        return by_num
    with _RAW_TSV.open("r", encoding="utf-8", errors="replace") as handle:
        _ = handle.readline()
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            track_id = parts[0]
            track_num = _track_id_to_int(track_id)
            if track_num is None:
                continue
            tags_by_type = {}
            for raw_tag in parts[5:]:
                raw_tag = _clean_text(raw_tag)
                if not raw_tag:
                    continue
                if "---" in raw_tag:
                    tag_type, tag_value = raw_tag.split("---", 1)
                else:
                    tag_type, tag_value = "tag", raw_tag
                tag_type = tag_type.strip().lower()
                tag_value = _clean_tag(tag_value)
                if tag_value:
                    tags_by_type.setdefault(tag_type, []).append(tag_value)
            by_num[track_num] = {
                "track_id": track_id,
                "duration": parts[4],
                "tags_by_type": tags_by_type,
            }
    return by_num


def _load_audio_metadata():
    by_num = {}
    if not _AUDIO_METADATA_TSV.exists():
        return by_num
    with _AUDIO_METADATA_TSV.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            track_id = parts[0]
            track_num = _track_id_to_int(track_id)
            if track_num is None:
                continue
            title = _clean_text(parts[3])
            artist = _clean_text(parts[4])
            album = _clean_text(parts[5])
            release_date = _clean_text(parts[6])
            year = release_date.split("-")[0] if release_date else None
            by_num[track_num] = {
                "track_id": track_id,
                "title": title,
                "artist": artist,
                "album": album,
                "year": year,
            }
    return by_num


_RAW_BY_NUM = _load_raw_metadata()
_META_BY_NUM = _load_audio_metadata()


def _pick_subset(values, max_count=4):
    values = [value for value in values if value]
    if not values:
        return []
    random.shuffle(values)
    count = random.randint(1, min(max_count, len(values)))
    return values[:count]


def _random_case(text):
    roll = random.random()
    if roll < 0.2:
        return text.lower()
    if roll < 0.25:
        return text.upper()
    return text


def _build_prompt(meta, raw):
    fields = []

    for key in ("title", "artist", "album", "year"):
        value = meta.get(key) if meta else None
        if value:
            fields.append((key, value))

    tags_by_type = raw.get("tags_by_type", {}) if raw else {}
    tag_sources = [
        ("genre", "genre"),
        ("mood/theme", "mood"),
        ("instrument", "instrument"),
    ]
    for tag_key, label in tag_sources:
        values = _pick_subset(tags_by_type.get(tag_key, []))
        if values:
            fields.append((label, values))

    other_tags = []
    for tag_key, values in tags_by_type.items():
        if tag_key in {"genre", "mood/theme", "instrument"}:
            continue
        other_tags.extend(values)
    other_tags = _pick_subset(other_tags)
    if other_tags:
        fields.append(("tag", other_tags))

    selected = [field for field in fields if random.random() < 0.7]
    if not selected and fields:
        selected = [random.choice(fields)]
    if not selected:
        return None

    random.shuffle(selected)
    use_labels = random.random() < 0.5

    parts = []
    for label, value in selected:
        if isinstance(value, (list, tuple)):
            text_value = ", ".join(value)
        else:
            text_value = value
        text_value = _random_case(text_value)
        if use_labels:
            label_text = label
            if isinstance(value, (list, tuple)) and len(value) > 1:
                label_text = f"{label}s"
            parts.append(f"{label_text}: {text_value}")
        else:
            parts.append(text_value)

    return ", ".join(parts)


def get_custom_metadata(info, audio):
    track_num = _extract_track_num_from_path(info.get("path"))
    meta = _META_BY_NUM.get(track_num, {}) if track_num is not None else {}
    raw = _RAW_BY_NUM.get(track_num, {}) if track_num is not None else {}

    prompt = _build_prompt(meta, raw)
    if not prompt:
        prompt = info.get("relpath") or info.get("path", "")

    return {"prompt": prompt}


def _resolve_audio_dir():
    return _DATASET_DIR / "audio"


def _iter_audio_paths(audio_dir, limit):
    audio_dir = Path(audio_dir)
    if not audio_dir.exists():
        return []
    paths = sorted(audio_dir.rglob("*.mp3"))
    if limit is not None:
        paths = paths[:limit]
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preview prompts from custom metadata.")
    parser.add_argument("--audio-dir", default=str(_resolve_audio_dir()))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    audio_paths = _iter_audio_paths(args.audio_dir, args.limit)
    if not audio_paths:
        print(f"No audio files found in: {args.audio_dir}")
        raise SystemExit(1)

    for audio_path in audio_paths:
        info = {"path": str(audio_path)}
        try:
            info["relpath"] = str(Path(audio_path).relative_to(args.audio_dir))
        except ValueError:
            pass
        prompt = get_custom_metadata(info, None).get("prompt", "")
        print(f"{info.get('relpath', info['path'])}\t{prompt}")
