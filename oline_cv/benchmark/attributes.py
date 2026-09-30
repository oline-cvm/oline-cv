"""Calibrate five good/bad technique calls from labeled reference analyses.

Fits a midpoint threshold per metric from expert good vs bad means. Not a neural net.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from oline_cv.benchmark.features import as_optional_float, extract_numeric_features
from oline_cv.reference.schema import ATTRIBUTE_KEYS, annotation_attributes

# Metrics already produced by analyze_video() for each expert column.
ATTRIBUTE_FEATURES: dict[str, tuple[str, ...]] = {
    "stance": (
        "mean_knee_flexion_deg",
        "mean_hip_height",
        "hip_height_at_lowest",
        "mean_torso_angle_deg",
        "posture_confidence",
    ),
    "first_step": (
        "reaction_time_ms",
        "first_step_acceleration",
        "initiated_by_foot",
    ),
    "feet": (
        "mean_base_width",
        "min_base_width",
        "step_cadence_hz",
        "feet_close_to_ground_score",
        "set_width",
    ),
    "hands": (
        "punch_ms",
        "approximate_hand_placement_score",
    ),
    "hips": (
        "mean_hip_height",
        "hip_height_at_lowest",
        "anchor_give",
        "mean_torso_angle_deg",
        "initiated_by_foot",
    ),
}

SEPARATION_EPS = 1e-6


def extra_numeric_features(result: dict[str, Any]) -> dict[str, float | None]:
    """Tiny derived flags that are still pipeline outputs, not new pose math."""
    initiated = (result.get("rep_summary") or {}).get("initiated_by")
    if initiated is None:
        initiated = ((result.get("modules") or {}).get("initial_quicks") or {}).get("initiated_by")
    foot = None
    if initiated == "foot":
        foot = 1.0
    elif initiated == "hip":
        foot = 0.0
    return {"initiated_by_foot": foot}


def player_feature_vector(result: dict[str, Any]) -> dict[str, float | None]:
    feats = extract_numeric_features(result)
    feats.update(extra_numeric_features(result))
    return feats


def _midpoint_rule(good: list[float], bad: list[float]) -> dict[str, Any] | None:
    if not good or not bad:
        return None
    g = sum(good) / len(good)
    b = sum(bad) / len(bad)
    if abs(g - b) <= SEPARATION_EPS:
        return None
    higher_is_good = g > b
    threshold = (g + b) / 2.0
    return {
        "threshold": threshold,
        "higher_is_good": higher_is_good,
        "good_mean": g,
        "bad_mean": b,
        "n_good": len(good),
        "n_bad": len(bad),
    }


def fit_attribute_model(
    labeled: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    """labeled: list of (analysis_json, annotation_json)."""
    buckets: dict[str, dict[str, dict[str, list[float]]]] = {
        attr: defaultdict(lambda: {"good": [], "bad": []}) for attr in ATTRIBUTE_KEYS
    }
    used_reps: list[str] = []

    for analysis, annotation in labeled:
        attrs = annotation_attributes(annotation)
        feats = player_feature_vector(analysis)
        used_reps.append(str(analysis.get("rep_id") or annotation.get("rep_id") or "unknown"))
        for attr in ATTRIBUTE_KEYS:
            label = attrs.get(attr, "n/a")
            if label not in ("good", "bad"):
                continue
            for feat in ATTRIBUTE_FEATURES[attr]:
                val = as_optional_float(feats.get(feat))
                if val is None:
                    continue
                buckets[attr][feat][label].append(val)

    attributes: dict[str, Any] = {}
    for attr in ATTRIBUTE_KEYS:
        rules = []
        for feat in ATTRIBUTE_FEATURES[attr]:
            rule = _midpoint_rule(buckets[attr][feat]["good"], buckets[attr][feat]["bad"])
            if rule is None:
                continue
            rules.append({"feature": feat, **rule})
        attributes[attr] = {"rules": rules, "n_rules": len(rules)}

    return {
        "schema_version": "1.0",
        "kind": "attribute_midpoint_v1",
        "input_rep_ids": used_reps,
        "attributes": attributes,
    }


def _vote(value: float, rule: dict[str, Any]) -> str:
    higher = bool(rule["higher_is_good"])
    thresh = float(rule["threshold"])
    if higher:
        return "good" if value >= thresh else "bad"
    return "good" if value <= thresh else "bad"


def score_attributes(
    result: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    feats = player_feature_vector(result)
    out: dict[str, Any] = {}
    for attr in ATTRIBUTE_KEYS:
        spec = (model.get("attributes") or {}).get(attr) or {}
        rules = spec.get("rules") or []
        votes: list[str] = []
        evidence: list[dict[str, Any]] = []
        for rule in rules:
            feat = rule["feature"]
            val = as_optional_float(feats.get(feat))
            if val is None:
                continue
            verdict = _vote(val, rule)
            votes.append(verdict)
            evidence.append(
                {
                    "feature": feat,
                    "player_value": val,
                    "threshold": rule["threshold"],
                    "higher_is_good": rule["higher_is_good"],
                    "vote": verdict,
                }
            )
        if not votes:
            label = "uncertain"
        else:
            n_good = votes.count("good")
            n_bad = votes.count("bad")
            if n_good == n_bad:
                label = "uncertain"
            else:
                label = "good" if n_good > n_bad else "bad"
        out[attr] = {
            "verdict": label,
            "n_votes": len(votes),
            "n_good_votes": votes.count("good"),
            "n_bad_votes": votes.count("bad"),
            "evidence": evidence,
        }
    return {
        "schema_version": "1.0",
        "attributes": out,
    }


def attach_attribute_scores(result: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    payload = score_attributes(result, model)
    result["attribute_scores"] = payload
    return payload
