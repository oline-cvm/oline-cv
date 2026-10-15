"""Compare one extracted rep to a statistical benchmark (deviation, not grade)."""

from __future__ import annotations

from typing import Any

from oline_cv.benchmark.features import extract_features
from oline_cv.benchmark.schema import (
    FEATURE_KEYS,
    FEATURE_REGISTRY,
    MIN_COMPARE_N,
    SCHEMA_VERSION,
    STD_EPS,
)
from oline_cv.benchmark.storage import load_benchmark


def _z_score(player: float, mean: float, std: float, count: int) -> float | None:
    if count < MIN_COMPARE_N or std is None or abs(float(std)) <= STD_EPS:
        return None
    return (player - mean) / std


def _difference_pct(player: float, mean: float) -> float | None:
    if abs(mean) <= STD_EPS:
        return None
    return 100.0 * (player - mean) / abs(mean)


def _status(player: float | None, stats: dict[str, Any] | None) -> str:
    if player is None:
        return "missing_player_data"
    if not stats:
        return "insufficient_reference_data"
    count = int(stats.get("count") or 0)
    if count < MIN_COMPARE_N:
        return "insufficient_reference_data"
    lo = stats.get("p25")
    hi = stats.get("p75")
    if lo is None or hi is None:
        return "insufficient_reference_data"
    if lo <= player <= hi:
        return "within_reference_range"
    if player < lo:
        return "below_reference_range"
    return "above_reference_range"


def compare_feature(feature: str, player_value: float | None, stats: dict[str, Any] | None) -> dict[str, Any]:
    spec = FEATURE_REGISTRY.get(feature)
    status = _status(player_value, stats)
    mean = float(stats["mean"]) if stats and stats.get("mean") is not None else None
    median = float(stats["median"]) if stats and stats.get("median") is not None else None
    std = float(stats["std"]) if stats and stats.get("std") is not None else None
    count = int(stats["count"]) if stats and stats.get("count") is not None else 0
    z = _z_score(player_value, mean, std, count) if player_value is not None and mean is not None and std is not None else None
    diff = (player_value - mean) if player_value is not None and mean is not None else None
    return {
        "feature": feature,
        "display_name": spec.display_name if spec else feature,
        "unit": spec.unit if spec else "",
        "category": spec.category if spec else "unknown",
        "player_value": player_value,
        "benchmark_mean": mean,
        "benchmark_median": median,
        "benchmark_std": std,
        "benchmark_count": count,
        "percentile_range": (
            {
                "p10": stats.get("p10"),
                "p25": stats.get("p25"),
                "p75": stats.get("p75"),
                "p90": stats.get("p90"),
            }
            if stats
            else None
        ),
        "z_score": z,
        "difference": diff,
        "difference_pct": (
            _difference_pct(player_value, mean) if player_value is not None and mean is not None else None
        ),
        "status": status,
    }


def _distance(row: dict[str, Any]) -> float:
    """Statistical distance for ranking. Prefer |z|; never treat higher as better."""
    z = row.get("z_score")
    if z is not None:
        return abs(float(z))
    pct = row.get("difference_pct")
    if pct is not None:
        return abs(float(pct)) / 100.0
    diff = row.get("difference")
    if diff is not None:
        return abs(float(diff))
    return 0.0


def _context_warnings(player: dict[str, Any], bench_ctx: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for key in ("play_type", "position", "technique", "side"):
        pv = player.get(key)
        bv = bench_ctx.get(key)
        if not pv or not bv or pv == "unknown" or bv == "unknown":
            continue
        if str(pv) != str(bv):
            warnings.append(f"{key} mismatch: player={pv} benchmark={bv}")
    return warnings


def summarize_comparison(
    rows: list[dict[str, Any]],
    *,
    benchmark_name: str,
    bench_context: dict[str, Any],
    player_context: dict[str, Any],
    top_n: int = 5,
) -> dict[str, Any]:
    usable = [
        r
        for r in rows
        if r["status"]
        in ("within_reference_range", "above_reference_range", "below_reference_range")
    ]
    ranked = sorted(usable, key=_distance, reverse=True)
    closest = sorted(usable, key=_distance)
    categories: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        categories.setdefault(row["category"], []).append(row)

    def _slim(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "feature": row["feature"],
            "display_name": row["display_name"],
            "player_value": row["player_value"],
            "benchmark_mean": row["benchmark_mean"],
            "z_score": row["z_score"],
            "difference": row["difference"],
            "difference_pct": row["difference_pct"],
            "status": row["status"],
            "category": row["category"],
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark_name,
        "context": bench_context,
        "player_context": player_context,
        "context_warnings": _context_warnings(player_context, bench_context),
        "valid_comparisons": len(usable),
        "largest_deviations": [_slim(r) for r in ranked[:top_n]],
        "closest_matches": [_slim(r) for r in closest[:top_n]],
        "categories": {k: [_slim(r) for r in v] for k, v in categories.items()},
        "features": rows,
    }


def compare_rep(record: dict[str, Any], benchmark: dict[str, Any]) -> dict[str, Any]:
    """Compare an extracted feature record to a built benchmark."""
    player_feats = record.get("features") or {}
    bench_feats = benchmark.get("features") or {}
    rows = [
        compare_feature(key, player_feats.get(key), bench_feats.get(key)) for key in FEATURE_KEYS
    ]
    player_context = {
        "play_type": record.get("play_type"),
        "position": record.get("position"),
        "technique": record.get("technique"),
        "side": record.get("side"),
        "rep_id": record.get("rep_id"),
        "trust_overall": record.get("trust_overall"),
        "defender_tracked": record.get("defender_tracked"),
    }
    return summarize_comparison(
        rows,
        benchmark_name=str(benchmark.get("name") or "unnamed"),
        bench_context=dict(benchmark.get("context") or {}),
        player_context=player_context,
    )


def attach_benchmark_comparison(
    result: dict[str, Any],
    benchmark_path: str,
    *,
    position: str | None = None,
    technique: str | None = None,
    side: str | None = None,
    source_analysis_path: str | None = None,
) -> dict[str, Any]:
    """In-place: result['benchmark_comparison'] = compare(extract(result), load(path))."""
    try:
        benchmark = load_benchmark(benchmark_path)
        record = extract_features(
            result,
            source_analysis_path=source_analysis_path,
            position=position,
            technique=technique,
            side=side,
        )
        comparison = compare_rep(record, benchmark)
        comparison["player_record"] = record
        result["benchmark_comparison"] = comparison
        return comparison
    except Exception as exc:
        payload = {"error": str(exc), "benchmark_path": benchmark_path}
        result["benchmark_comparison"] = payload
        return payload
