from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[4]
_TEST_PY = _REPO_ROOT / "test.py"
_DEFAULT_SECONDS_TOTAL = 2097152 / 44100


def _load_prompts_from_test_py(path: Path) -> List[str]:
    if not path.exists():
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PROMPTS":
                    try:
                        value = ast.literal_eval(node.value)
                    except Exception:
                        return []
                    if isinstance(value, list):
                        return [str(x) for x in value]
    return []


PROMPTS = _load_prompts_from_test_py(_TEST_PY)


def _extract_index_from_path(audio_path: str) -> Optional[int]:
    filename = Path(audio_path or "").name
    match = re.search(r"(\d+)(?=\.[^.]+$)", filename)
    if not match:
        return None
    return int(match.group(1))


def get_custom_metadata(info, audio):
    idx = _extract_index_from_path(info.get("path", ""))
    prompt = ""
    if idx is not None and idx < len(PROMPTS):
        prompt = PROMPTS[idx]

    seconds_total = info.get("seconds_total", _DEFAULT_SECONDS_TOTAL)
    return {
        "prompt": prompt,
        "seconds_start": 0.0,
        "seconds_total": float(seconds_total),
    }

