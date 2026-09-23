"""Feature extraction from synthetic analyze_video() JSON — no YOLO / video."""

from __future__ import annotations

import math

from oline_cv.benchmark.features import (
    as_optional_float,
    extract_features,
    extract_numeric_features,
    trust_overall_score,
)
from oline_cv.benchmark.schema import FEATURE_KEYS, SCHEMA_VERSION


def _analysis(**overrides):
    base = {
        "play_type": "pass",
        "video": {"path": "clips/lt_pass_01.mp4", "fps": 30},
        "trust": {"overall": {"score": 0.81, "level": "high"}},
        "rep_summary": {
            "play_type": "pass",
            "reaction_time_ms": 160.0,
            "reaction_time_frames": 5,
            "mean_knee_flexion_deg": 138.4,
            "min_knee_flexion_deg": 129.1,
            "mean_torso_angle_deg": 12.2,
            "mean_hip_height": 0.58,
            "hip_height_at_lowest": 0.51,
            "step_cadence_hz": 3.4,
            "set_depth": 0.22,
            "set_width": 0.31,
            "mean_base_width": 0.48,
            "lateral_match": 0.62,
            "anchor_give": 0.08,
            "punch_ms": 260.0,
            "engagement_ms": 820.0,
            "defender_tracked": True,
            "posture_confidence": 0.72,
            "trust_overall": {"score": 0.81, "level": "high"},
        },
        "modules": {
            "initial_quicks": {
                "reaction_time_ms": 160.0,
                "first_step_acceleration": 1.2,
                "first_step_direction_deg": 12.0,
            },
            "body_position": {"max_torso_angle_deg": 18.0},
            "footwork": {
                "mean_step_length": 0.19,
                "min_base_width": 0.41,
                "mean_lateral_velocity": 0.4,
            },
            "mirror_redirect": {"mean_separation": 0.35, "lateral_match_correlation": 0.62},
            "anchor": {"max_hip_displacement_after_contact": 0.08},
            "hands": {"time_to_first_contact_ms": 260.0},
            "sustain": {"engagement_ms": 820.0},
            "balance": {"com_jitter": 0.04},
        },
    }
    base.update(overrides)
    return base


def test_extracts_rep_summary_metrics():
    rec = extract_features(_analysis(), source_analysis_path="clips/lt_pass_01_analysis.json")
    assert rec["schema_version"] == SCHEMA_VERSION
    assert rec["rep_id"] == "lt_pass_01"
    assert rec["play_type"] == "pass"
    assert rec["trust_overall"] == 0.81
    assert rec["defender_tracked"] is True
    feats = rec["features"]
    assert feats["reaction_time_ms"] == 160.0
    assert feats["mean_knee_flexion_deg"] == 138.4
    assert feats["mean_base_width"] == 0.48
    assert feats["lateral_match"] == 0.62
    assert feats["anchor_give"] == 0.08
    assert feats["com_jitter"] == 0.04


def test_missing_values_stay_none_not_zero():
    result = _analysis()
    result["rep_summary"]["punch_ms"] = None
    result["modules"]["hands"]["time_to_first_contact_ms"] = None
    result["rep_summary"].pop("engagement_ms")
    result["modules"]["sustain"].pop("engagement_ms")
    feats = extract_numeric_features(result)
    assert feats["punch_ms"] is None
    assert feats["engagement_ms"] is None
    assert feats["reaction_time_ms"] == 160.0
    rec = extract_features(result)
    assert "punch_ms" in rec["missing_features"]
    assert "engagement_ms" in rec["missing_features"]
    assert 0.0 not in (feats["punch_ms"], feats["engagement_ms"])


def test_nan_and_bool_are_missing():
    assert as_optional_float(None) is None
    assert as_optional_float(True) is None
    assert as_optional_float(float("nan")) is None
    assert as_optional_float(math.inf) is None
    assert as_optional_float("12.5") == 12.5


def test_falls_back_to_module_aliases():
    result = {
        "play_type": "run",
        "trust": {"overall": {"score": 0.6, "level": "medium"}},
        "rep_summary": {"defender_tracked": False},
        "modules": {
            "mirror_redirect": {"lateral_match_correlation": 0.41},
            "anchor": {"max_hip_displacement_after_contact": 0.14},
            "point_of_attack": {"max_defender_displacement": 0.21},
            "movement_in_space": {"path_efficiency": 0.8, "travel_distance": 1.1},
        },
    }
    feats = extract_numeric_features(result)
    assert feats["lateral_match"] == 0.41
    assert feats["anchor_give"] == 0.14
    assert feats["max_defender_displacement"] == 0.21
    assert feats["path_efficiency"] == 0.8
    rec = extract_features(result, position="LG", technique="inside_zone", side="interior")
    assert rec["play_type"] == "run"
    assert rec["position"] == "LG"
    assert rec["technique"] == "inside_zone"
    assert rec["side"] == "interior"


def test_trust_reads_nested_or_scalar():
    assert trust_overall_score({"trust": {"overall": {"score": 0.5}}}) == 0.5
    assert trust_overall_score({"rep_summary": {"trust_overall": {"score": 0.4}}}) == 0.4
    assert trust_overall_score({"rep_summary": {"trust_overall": 0.33}}) == 0.33
    assert trust_overall_score({}) is None


def test_every_registered_key_is_present():
    feats = extract_numeric_features(_analysis())
    assert tuple(feats) == FEATURE_KEYS
