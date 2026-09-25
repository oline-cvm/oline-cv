"""Load/save analysis JSON and benchmark profiles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{p} is not a JSON object")
    return data


def save_json(path: str | Path, payload: dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load_analysis_json(path: str | Path) -> dict[str, Any]:
    return load_json(path)


def save_benchmark(path: str | Path, benchmark: dict[str, Any]) -> Path:
    return save_json(path, benchmark)


def load_benchmark(path: str | Path) -> dict[str, Any]:
    data = load_json(path)
    if "features" not in data or "context" not in data:
        raise ValueError(f"{path} is not a benchmark profile (missing features/context)")
    return data


def expand_input_paths(patterns: Iterable[str | Path]) -> list[Path]:
    """Expand globs / directories. PowerShell does not always expand *."""
    found: list[Path] = []
    seen: set[Path] = set()
    for raw in patterns:
        p = Path(raw)
        hits: list[Path]
        if p.is_dir():
            hits = sorted(p.glob("*.json"))
        elif any(ch in str(raw) for ch in "*?[]"):
            hits = sorted(Path().glob(str(raw)))
            if not hits:
                hits = sorted(p.parent.glob(p.name)) if p.parent.exists() else []
        else:
            hits = [p]
        for hit in hits:
            resolved = hit.resolve()
            if resolved in seen:
                continue
            if not hit.is_file():
                continue
            seen.add(resolved)
            found.append(hit)
    return found
