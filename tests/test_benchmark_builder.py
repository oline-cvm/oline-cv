"""Benchmark builder stats, tiny datasets, and trust filtering."""

from __future__ import annotations

from oline_cv.benchmark.builder import build_benchmark, feature_stats
from oline_cv.benchmark.schema import DEFAULT_MIN_TRUST_SCORE, SCHEMA_VERSION


def _rep(rep_id: str, trust: float | None, **metrics):
    summary = {
        "play_type": "pass",
        "defender_tracked": True,
        **metrics,
    }
    trust_block = {"overall": {"score": trust, "level": "high"}} if trust is not None else {}
    return (
        {
            "rep_id": rep_id,
            "play_type": "pass",
            "trust": trust_block,
            "rep_summary": summary,
            "modules": {},
        },
        f"{rep_id}_analysis.json",
    )


def test_feature_stats_none_when_empty():
    assert feature_stats([]) is None


def test_feature_stats_single_value_has_zero_std():
    stats = feature_stats([0.48])
    assert stats is not None
    assert stats["count"] == 1
    assert stats["mean"] == 0.48
    assert stats["median"] == 0.48
    assert stats["std"] == 0.0
    assert stats["p10"] == 0.48
    assert stats["p90"] == 0.48


def test_feature_stats_std_zero_when_identical():
    stats = feature_stats([0.5, 0.5, 0.5, 0.5])
    assert stats is not None
    assert stats["std"] == 0.0
    assert stats["min"] == stats["max"] == 0.5


def test_build_skips_low_and_missing_trust():
    analyses = [
        _rep("good_a", 0.80, mean_base_width=0.46, reaction_time_ms=150),
        _rep("good_b", 0.70, mean_base_width=0.50, reaction_time_ms=170),
        _rep("low", 0.20, mean_base_width=0.99, reaction_time_ms=400),
        _rep("no_trust", None, mean_base_width=0.10, reaction_time_ms=50),
    ]
    bench = build_benchmark(
        analyses,
        name="elite_lt_vertical_set_v1",
        position="LT",
        play_type="pass",
        technique="vertical_set",
        side="left",
    )
    assert bench["schema_version"] == SCHEMA_VERSION
    assert bench["name"] == "elite_lt_vertical_set_v1"
    assert bench["context"] == {
        "position": "LT",
        "play_type": "pass",
        "technique": "vertical_set",
        "side": "left",
    }
    assert bench["min_trust_score"] == DEFAULT_MIN_TRUST_SCORE
    assert bench["n_reps"] == 2
    assert bench["n_skipped"] == 2
    assert bench["input_rep_ids"] == ["good_a", "good_b"]
    reasons = {row["reason"] for row in bench["skipped"]}
    assert reasons == {"below_trust_threshold", "missing_trust"}
    # Contaminating 0.99 / 0.10 values must not enter the distribution.
    base = bench["features"]["mean_base_width"]
    assert base["count"] == 2
    assert abs(base["mean"] - 0.48) < 1e-9
    assert "created_at" in bench


def test_no_distribution_when_feature_never_observed():
    analyses = [
        _rep("a", 0.9, mean_base_width=0.45),
        _rep("b", 0.9, mean_base_width=0.47),
    ]
    bench = build_benchmark(analyses, name="tiny", play_type="pass")
    assert "punch_ms" not in bench["features"]
    assert bench["feature_coverage"]["punch_ms"] == 0
    assert bench["feature_coverage"]["mean_base_width"] == 2


def test_custom_trust_threshold():
    analyses = [
        _rep("mid", 0.50, mean_base_width=0.44),
        _rep("high", 0.90, mean_base_width=0.46),
    ]
    strict = build_benchmark(analyses, name="strict", min_trust_score=0.72)
    assert strict["n_reps"] == 1
    assert strict["input_rep_ids"] == ["high"]
