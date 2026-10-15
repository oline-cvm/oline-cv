"""Extract a flat, versioned numeric feature record from analyze_video() JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oline_cv.benchmark.schema import (
    FEATURE_DEFS,
    FEATURE_KEYS,
    SCHEMA_VERSION,
    normalize_play_type,
    normalize_position,
    normalize_side,
    normalize_technique,
)


def _dig(obj: Any, path: tuple[str, ...]) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def as_optional_float(value: Any) -> float | None:
    """Return a finite float, or None. Never coerce missing/bool to 0."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
        return None
    return out


def trust_overall_score(result: dict[str, Any]) -> float | None:
    overall = (result.get("trust") or {}).get("overall")
    if isinstance(overall, dict):
        score = as_optional_float(overall.get("score"))
        if score is not None:
            return score
    packed = (result.get("rep_summary") or {}).get("trust_overall")
    if isinstance(packed, dict):
        return as_optional_float(packed.get("score"))
    return as_optional_float(packed)


def defender_tracked(result: dict[str, Any]) -> bool | None:
    s = result.get("rep_summary") or {}
    if "defender_tracked" in s:
        return bool(s.get("defender_tracked"))
    return None


def infer_rep_id(result: dict[str, Any], source_path: str | None = None) -> str:
    explicit = result.get("rep_id") or (result.get("rep_summary") or {}).get("rep_id")
    if explicit:
        return str(explicit)
    if source_path:
        return Path(source_path).stem.replace("_analysis", "")
    video = (result.get("video") or {}).get("path")
    if video:
        return Path(str(video)).stem
    return "unknown_rep"


def extract_numeric_features(result: dict[str, Any]) -> dict[str, float | None]:
    """Map every registered feature to a float or None (missing stays None)."""
    out: dict[str, float | None] = {}
    for spec in FEATURE_DEFS:
        found: float | None = None
        for path in spec.sources:
            found = as_optional_float(_dig(result, path))
            if found is not None:
                break
        out[spec.key] = found
    return out


def extract_features(
    result: dict[str, Any],
    *,
    source_analysis_path: str | None = None,
    position: str | None = None,
    technique: str | None = None,
    side: str | None = None,
    play_type: str | None = None,
    rep_id: str | None = None,
) -> dict[str, Any]:
    """Build a versioned record from an existing analysis dict. No pose recompute."""
    s = result.get("rep_summary") or {}
    features = extract_numeric_features(result)
    missing = [k for k in FEATURE_KEYS if features[k] is None]
    resolved_play = play_type or result.get("play_type") or s.get("play_type")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_analysis_path": source_analysis_path,
        "rep_id": rep_id or infer_rep_id(result, source_analysis_path),
        "play_type": normalize_play_type(resolved_play),
        "position": normalize_position(position or result.get("position") or s.get("position")),
        "technique": normalize_technique(technique or result.get("technique") or s.get("technique")),
        "side": normalize_side(side or result.get("side") or s.get("side")),
        "trust_overall": trust_overall_score(result),
        "defender_tracked": defender_tracked(result),
        "features": features,
        "missing_features": missing,
    }
