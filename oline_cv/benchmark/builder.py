"""Build a statistical reference profile from approved analysis JSON files."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from oline_cv.benchmark.features import extract_features, trust_overall_score
from oline_cv.benchmark.schema import (
    DEFAULT_MIN_TRUST_SCORE,
    FEATURE_KEYS,
    SCHEMA_VERSION,
    STAT_KEYS,
    normalize_play_type,
    normalize_position,
    normalize_side,
    normalize_technique,
)


def _native(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def feature_stats(values: list[float]) -> dict[str, float | int] | None:
    """Distribution for one feature. None if there are no valid observations."""
    if not values:
        return None
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    std = float(np.std(arr, ddof=1)) if n >= 2 else 0.0
    percentiles = np.percentile(arr, [10, 25, 50, 75, 90])
    return {
        "count": n,
        "mean": float(np.mean(arr)),
        "median": float(percentiles[2]),
        "std": std,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p75": float(percentiles[3]),
        "p90": float(percentiles[4]),
    }


def _context(
    position: str | None,
    play_type: str | None,
    technique: str | None,
    side: str | None,
) -> dict[str, str]:
    return {
        "position": normalize_position(position),
        "play_type": normalize_play_type(play_type),
        "technique": normalize_technique(technique),
        "side": normalize_side(side),
    }


def build_benchmark(
    analyses: list[tuple[dict[str, Any], str | None]],
    *,
    name: str,
    position: str | None = None,
    play_type: str | None = None,
    technique: str | None = None,
    side: str | None = None,
    min_trust_score: float = DEFAULT_MIN_TRUST_SCORE,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate extracted records into a reference profile.

    ``analyses`` is a list of (analysis_dict, source_path).
    Reps below ``min_trust_score`` or missing trust are skipped and recorded.
    """
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for analysis, source in analyses:
        record = extract_features(analysis, source_analysis_path=source)
        trust = record.get("trust_overall")
        if trust is None:
            trust = trust_overall_score(analysis)
        if trust is None:
            skipped.append(
                {
                    "rep_id": record["rep_id"],
                    "source_analysis_path": source,
                    "reason": "missing_trust",
                    "trust_overall": None,
                }
            )
            continue
        if float(trust) < min_trust_score:
            skipped.append(
                {
                    "rep_id": record["rep_id"],
                    "source_analysis_path": source,
                    "reason": "below_trust_threshold",
                    "trust_overall": float(trust),
                }
            )
            continue
        accepted.append(record)

    columns: dict[str, list[float]] = {k: [] for k in FEATURE_KEYS}
    for rec in accepted:
        feats = rec.get("features") or {}
        for key in FEATURE_KEYS:
            val = feats.get(key)
            if val is not None:
                columns[key].append(float(val))

    features: dict[str, dict[str, float | int]] = {}
    coverage: dict[str, int] = {}
    for key in FEATURE_KEYS:
        stats = feature_stats(columns[key])
        if stats is None:
            coverage[key] = 0
            continue
        features[key] = {k: _native(stats[k]) for k in STAT_KEYS}
        coverage[key] = int(stats["count"])

    context = _context(position, play_type, technique, side)
    if extra_context:
        context = {**context, **extra_context}

    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "context": context,
        "min_trust_score": float(min_trust_score),
        "n_reps": len(accepted),
        "n_skipped": len(skipped),
        "input_rep_ids": [r["rep_id"] for r in accepted],
        "skipped": skipped,
        "feature_coverage": coverage,
        "features": features,
        "accepted_records": [
            {
                "rep_id": r["rep_id"],
                "source_analysis_path": r.get("source_analysis_path"),
                "trust_overall": r.get("trust_overall"),
                "missing_features": r.get("missing_features"),
            }
            for r in accepted
        ],
    }
