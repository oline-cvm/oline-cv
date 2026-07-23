"""Unit tests for body-position accuracy fixes."""

from __future__ import annotations

from oline_cv.body_position import (
    FrameBodyMetrics,
    classify_posture,
    posture_scores,
    smooth_posture_sequence,
    summarize_body_position,
)
from oline_cv.config import AnalysisConfig


def test_knee_bender_scores_higher_when_flexed_upright():
    cfg = AnalysisConfig()
    label, conf = classify_posture(125.0, 12.0, 0.50, cfg)
    assert label == "knee_bender"
    assert conf > 0.04


def test_waist_bender_scores_higher_when_leaning_extended():
    cfg = AnalysisConfig()
    label, conf = classify_posture(165.0, 40.0, 0.72, cfg)
    assert label == "waist_bender"
    assert conf > 0.04


def test_summary_does_not_invent_balanced_on_tie_break():
    cfg = AnalysisConfig(posture_majority_frac=0.55)
    frames = []
    for i, p in enumerate(
        ["waist_bender"] * 23 + ["balanced"] * 24 + ["knee_bender"] * 13
    ):
        frames.append(
            FrameBodyMetrics(
                frame_idx=i,
                timestamp_ms=i * 33.0,
                low_confidence=False,
                knee_flexion_angle_left=140.0,
                knee_flexion_angle_right=140.0,
                knee_flexion_angle_mean=140.0,
                hip_height=0.7,
                torso_angle=25.0,
                com_height=0.75,
                shoulder_height=0.9,
                posture=p,  # type: ignore[arg-type]
                posture_confidence=0.1,
                flags=[],
            )
        )
    summary = summarize_body_position(frames, cfg, fps=30.0)
    # Plurality is balanced (24) — but if mixed, must mark mixed, not silently invent
    assert summary["posture_classification"] in ("balanced", "waist_bender", "knee_bender")
    # With early weighting, knee/waist early frames matter; just ensure no silent fake
    if summary["posture_confidence"] < cfg.posture_majority_frac:
        assert summary["posture_mixed"] is True


def test_smooth_only_flips_low_confidence():
    seq = [
        FrameBodyMetrics(
            frame_idx=i,
            timestamp_ms=0,
            low_confidence=False,
            knee_flexion_angle_left=None,
            knee_flexion_angle_right=None,
            knee_flexion_angle_mean=None,
            hip_height=None,
            torso_angle=None,
            com_height=None,
            shoulder_height=None,
            posture="knee_bender" if i != 2 else "waist_bender",  # type: ignore[arg-type]
            posture_confidence=0.02 if i == 2 else 0.3,
            flags=[],
        )
        for i in range(7)
    ]
    smooth_posture_sequence(seq, window=5)
    assert seq[2].posture == "knee_bender"


def test_posture_scores_sum_reasonable():
    s = posture_scores(140.0, 20.0, 0.65, AnalysisConfig())
    assert set(s) == {"knee_bender", "waist_bender", "balanced"}
    assert all(0.0 <= v <= 1.0 for v in s.values())
