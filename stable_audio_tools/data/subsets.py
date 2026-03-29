import copy
import csv
import os
import random
import re

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from .dataset import filter_excluded_filenames, get_audio_filenames, load_exclusion_entries


@dataclass(frozen=True)
class AudioSubsetRecord:
    dataset_id: str
    dataset_root: str
    global_index: int
    track_id: int | None
    path: str
    relpath: str


def _extract_track_id(audio_path: str) -> int | None:
    match = re.search(r"(\d+)", Path(audio_path).name)
    return int(match.group(1)) if match else None


def resolve_audio_subset_records(dataset_config: Dict[str, Any]) -> List[AudioSubsetRecord]:
    dataset_type = dataset_config.get("dataset_type", None)
    if dataset_type != "audio_dir":
        raise ValueError("Subset selection currently supports only dataset_type='audio_dir'.")

    exclusion_entries = load_exclusion_entries(dataset_config.get("exclude_paths_file", None))
    dataset_entries = dataset_config.get("datasets", None)
    if not dataset_entries:
        raise ValueError("Dataset config must include at least one dataset entry.")

    records: List[AudioSubsetRecord] = []
    global_index = 0

    for dataset_entry in dataset_entries:
        dataset_id = dataset_entry["id"]
        dataset_root = Path(dataset_entry["path"]).resolve()
        filenames = get_audio_filenames(str(dataset_root))
        filenames = filter_excluded_filenames(
            filenames,
            exclusion_entries,
            roots=[str(dataset_root)],
        )

        for filename in sorted(Path(path).resolve() for path in filenames):
            records.append(
                AudioSubsetRecord(
                    dataset_id=dataset_id,
                    dataset_root=str(dataset_root),
                    global_index=global_index,
                    track_id=_extract_track_id(str(filename)),
                    path=str(filename),
                    relpath=os.path.normpath(os.path.relpath(filename, dataset_root)),
                )
            )
            global_index += 1

    return records


def select_audio_subset(
    records: Sequence[AudioSubsetRecord],
    candidate_pool_size: int,
    train_subset_size: int,
    seed: int,
) -> Tuple[List[AudioSubsetRecord], List[int], List[AudioSubsetRecord]]:
    if candidate_pool_size <= 0:
        raise ValueError("candidate_pool_size must be positive.")
    if train_subset_size <= 0:
        raise ValueError("train_subset_size must be positive.")
    if len(records) < candidate_pool_size:
        raise ValueError(
            f"Requested candidate_pool_size={candidate_pool_size}, but only {len(records)} records are available."
        )

    candidate_records = list(records[:candidate_pool_size])
    if len(candidate_records) < train_subset_size:
        raise ValueError(
            f"Requested train_subset_size={train_subset_size}, but only {len(candidate_records)} candidate records are available."
        )

    rng = random.Random(seed)
    selected_pool_indices = sorted(rng.sample(range(len(candidate_records)), train_subset_size))
    selected_records = [candidate_records[idx] for idx in selected_pool_indices]
    return candidate_records, selected_pool_indices, selected_records


def build_subset_dataset_config(dataset_config: Dict[str, Any], include_paths_file: str) -> Dict[str, Any]:
    resolved_config = copy.deepcopy(dataset_config)
    resolved_config["include_paths_file"] = include_paths_file
    return resolved_config


def write_subset_records_csv(
    path: str | Path,
    records: Sequence[AudioSubsetRecord],
    pool_indices: Sequence[int] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["pool_index", "global_index", "dataset_id", "dataset_root", "track_id", "path", "relpath"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, record in enumerate(records):
            row = asdict(record)
            row["pool_index"] = pool_indices[idx] if pool_indices is not None else idx
            writer.writerow(row)
