"""Comparison statuses, z-scores, context, and optional pipeline hook."""

from __future__ import annotations

from pathlib import Path

from oline_cv.benchmark.builder import build_benchmark
from oline_cv.benchmark.compare import (
    attach_benchmark_comparison,
    compare_feature,
    compare_rep,
)
from oline_cv.benchmark.features import extract_features
from oline_cv.benchmark.storage import save_benchmark
from oline_cv.config import AnalysisConfig


def _analysis(rep_id: str, trust: float, **metrics):
    return {
        "rep_id": rep_id,
        "play_type": "pass",
        "trust": {"overall": {"score": trust, "level": "high"}},
        "rep_summary": {"play_type": "pass", "defender_tracked": True, **metrics},
        "modules": {},
    }


def _bench_from_bases(values: list[float], **extra_metrics_per_rep):
    analyses = []
    for i, v in enumerate(values):
        metrics = {"mean_base_width": v, **{k: extra[i] for k, extra in extra_metrics_per_rep.items()}}
        analyses.append((_analysis(f"r{i}", 0.85, **metrics), f"r{i}.json"))
    return build_benchmark(
        analyses,
        name="elite_lt_vertical_set_v1",
        position="LT",
        play_type="pass",
        technique="vertical_set",
        side="left",
    )


def test_within_and_outside_reference_range():
    # 0.40, 0.42, 0.44, 0.46, 0.48 → p25≈0.42, p75≈0.46
    bench = _bench_from_bases([0.40, 0.42, 0.44, 0.46, 0.48])
    mid = extract_features(_analysis("p", 0.8, mean_base_width=0.44), position="LT", technique="vertical_set")
    high = extract_features(_analysis("p", 0.8, mean_base_width=0.70), position="LT", technique="vertical_set")
    low = extract_features(_analysis("p", 0.8, mean_base_width=0.20), position="LT", technique="vertical_set")
    mid_row = next(r for r in compare_rep(mid, bench)["features"] if r["feature"] == "mean_base_width")
    high_row = next(r for r in compare_rep(high, bench)["features"] if r["feature"] == "mean_base_width")
    low_row = next(r for r in compare_rep(low, bench)["features"] if r["feature"] == "mean_base_width")
    assert mid_row["status"] == "within_reference_range"
    assert high_row["status"] == "above_reference_range"
    assert low_row["status"] == "below_reference_range"
    assert high_row["z_score"] is not None and high_row["z_score"] > 0
    assert low_row["z_score"] is not None and low_row["z_score"] < 0


def test_missing_player_data_status():
    bench = _bench_from_bases([0.40, 0.42, 0.44])
    rec = extract_features(_analysis("p", 0.8))  # no mean_base_width
    row = next(r for r in compare_rep(rec, bench)["features"] if r["feature"] == "mean_base_width")
    assert rec["features"]["mean_base_width"] is None
    assert row["status"] == "missing_player_data"
    assert row["z_score"] is None


def test_tiny_dataset_and_zero_std_do_not_crash():
    one = _bench_from_bases([0.45])
    same = _bench_from_bases([0.45, 0.45, 0.45])
    player = extract_features(_analysis("p", 0.8, mean_base_width=0.60))
    tiny = next(r for r in compare_rep(player, one)["features"] if r["feature"] == "mean_base_width")
    assert tiny["status"] == "insufficient_reference_data"
    assert tiny["z_score"] is None
    assert abs(tiny["difference"] - 0.15) < 1e-9

    zrow = compare_feature("mean_base_width", 0.60, same["features"]["mean_base_width"])
    assert zrow["z_score"] is None  # std == 0
    assert zrow["status"] == "above_reference_range"
    assert abs(zrow["difference"] - 0.15) < 1e-9


def test_summary_has_no_quality_labels():
    bench = _bench_from_bases([0.40, 0.42, 0.44, 0.46, 0.48], reaction_time_ms=[150, 155, 160, 165, 170])
    rec = extract_features(
        _analysis("p", 0.8, mean_base_width=0.80, reaction_time_ms=158),
        position="LT",
        play_type="pass",
        technique="vertical_set",
    )
    out = compare_rep(rec, bench)
    cue_blob = str(out["largest_deviations"] + out["closest_matches"]).lower()
    for banned in ("good", "bad", "poor", "worse", "better"):
        assert banned not in cue_blob
    assert out["benchmark"] == "elite_lt_vertical_set_v1"
    assert out["valid_comparisons"] >= 1
    assert "body_position" in out["categories"] or "footwork" in out["categories"]
    assert out["largest_deviations"]
    assert out["closest_matches"]
    assert out["context_warnings"] == []


def test_context_mismatch_is_warning_not_a_grade():
    bench = _bench_from_bases([0.40, 0.42, 0.44])
    rec = extract_features(
        _analysis("p", 0.8, mean_base_width=0.43),
        position="RT",
        play_type="run",
        technique="inside_zone",
    )
    out = compare_rep(rec, bench)
    assert any("play_type mismatch" in w for w in out["context_warnings"])
    assert any("position mismatch" in w for w in out["context_warnings"])


def test_config_benchmark_path_defaults_none():
    cfg = AnalysisConfig()
    assert cfg.benchmark_path is None
    assert "benchmark_path" in cfg.to_dict()


def test_attach_hook_writes_comparison(tmp_path: Path):
    bench = _bench_from_bases([0.40, 0.42, 0.44, 0.46, 0.48])
    path = save_benchmark(tmp_path / "ref.json", bench)
    result = _analysis("new", 0.8, mean_base_width=0.43)
    assert "benchmark_comparison" not in result
    attach_benchmark_comparison(result, str(path), position="LT", technique="vertical_set")
    cmp_ = result["benchmark_comparison"]
    assert cmp_["benchmark"] == "elite_lt_vertical_set_v1"
    assert "features" in cmp_
    assert "error" not in cmp_


def test_attach_missing_file_does_not_raise():
    result = _analysis("new", 0.8, mean_base_width=0.43)
    attach_benchmark_comparison(result, "definitely_missing_benchmark.json")
    assert "error" in result["benchmark_comparison"]


def test_backward_compatible_pipeline_shape():
    """A current-style analyze_video() payload still extracts without new fields."""
    legacy = {
        "play_type": "pass",
        "video": {"fps": 30},
        "trust": {"overall": {"score": 0.77, "level": "high"}},
        "rep_summary": {
            "reaction_time_ms": 180,
            "mean_knee_flexion_deg": 142,
            "mean_base_width": 0.44,
            "defender_tracked": True,
        },
        "modules": {"footwork": {"set_depth": 0.2, "set_width": 0.3}},
    }
    rec = extract_features(legacy)
    assert rec["features"]["reaction_time_ms"] == 180
    assert rec["features"]["set_depth"] == 0.2
    assert rec["features"]["punch_ms"] is None
