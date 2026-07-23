"""Module 2 — Body position (pad level) and posture classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from oline_cv.config import AnalysisConfig, L_ANKLE, R_ANKLE
from oline_cv.geometry import (
    ankle_mid,
    estimate_com_xy,
    hip_mid,
    knee_flexion,
    normalized_height_from_feet,
    shoulder_mid,
    vector_angle_from_vertical_deg,
)
from oline_cv.pose_tracker import FramePose


PostureLabel = Literal["knee_bender", "waist_bender", "balanced", "unknown"]


@dataclass
class FrameBodyMetrics:
    frame_idx: int
    timestamp_ms: float
    low_confidence: bool
    knee_flexion_angle_left: float | None
    knee_flexion_angle_right: float | None
    knee_flexion_angle_mean: float | None
    hip_height: float | None
    torso_angle: float | None
    com_height: float | None
    shoulder_height: float | None
    posture: PostureLabel
    posture_confidence: float
    flags: list[str]


def posture_scores(
    knee_flexion_mean: float,
    torso_angle: float,
    hip_height: float,
    config: AnalysisConfig,
) -> dict[str, float]:
    """Continuous scores in [0,1]. Higher = stronger evidence for that class."""
    # Flexion: 180 straight → 90 deep. Map around knee_bender threshold.
    flex_center = config.knee_bender_flexion_deg  # ~145
    knee_flex_amt = float(np.clip((flex_center + 15.0 - knee_flexion_mean) / 40.0, 0.0, 1.0))
    knee_ext_amt = float(np.clip((knee_flexion_mean - (flex_center - 10.0)) / 40.0, 0.0, 1.0))

    upright = float(np.clip(1.0 - torso_angle / max(config.waist_bender_torso_deg * 2.0, 1.0), 0.0, 1.0))
    lean = float(np.clip(torso_angle / max(config.waist_bender_torso_deg * 2.0, 1.0), 0.0, 1.0))

    # Relative pad level vs standing — film hips often ~0.55–0.80 of standing height.
    low_hips = float(
        np.clip((config.low_hip_height_frac + 0.12 - hip_height) / 0.25, 0.0, 1.0)
    )
    high_hips = 1.0 - low_hips

    knee = 0.45 * knee_flex_amt + 0.35 * upright + 0.20 * low_hips
    waist = 0.50 * lean + 0.30 * knee_ext_amt + 0.20 * high_hips
    # Balanced: moderate flex, moderate torso, not extreme lean
    mid_knee = 1.0 - abs(knee_flexion_mean - (flex_center + 5.0)) / 35.0
    mid_torso = 1.0 - abs(torso_angle - config.waist_bender_torso_deg * 0.7) / 30.0
    balanced = 0.5 * float(np.clip(mid_knee, 0.0, 1.0)) + 0.5 * float(np.clip(mid_torso, 0.0, 1.0))
    # Suppress balanced when a cue is strong
    if lean > 0.65 and knee_ext_amt > 0.4:
        balanced *= 0.4
    if knee_flex_amt > 0.7 and upright > 0.7:
        balanced *= 0.4

    return {"knee_bender": knee, "waist_bender": waist, "balanced": balanced}


def classify_posture(
    knee_flexion_mean: float | None,
    torso_angle: float | None,
    hip_height: float | None,
    config: AnalysisConfig,
) -> tuple[PostureLabel, float]:
    """Return (label, confidence margin). Confidence = top − second score."""
    if knee_flexion_mean is None or torso_angle is None or hip_height is None:
        return "unknown", 0.0

    scores = posture_scores(knee_flexion_mean, torso_angle, hip_height, config)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_s = ordered[0]
    second_s = ordered[1][1]
    conf = float(top_s - second_s)
    # Near-ties stay unknown rather than inventing a clean label
    if conf < 0.04 and top_s < 0.55:
        return "unknown", conf
    return top_name, conf  # type: ignore[return-value]


def compute_frame_body_metrics(
    pose: FramePose,
    standing_height_px: float,
    config: AnalysisConfig,
) -> FrameBodyMetrics:
    flags: list[str] = []
    if pose.low_confidence:
        flags.append("low_keypoint_confidence")

    left_angle, left_low = knee_flexion(
        pose.keypoints_xy, pose.keypoints_conf, "left", config.min_keypoint_confidence
    )
    right_angle, right_low = knee_flexion(
        pose.keypoints_xy, pose.keypoints_conf, "right", config.min_keypoint_confidence
    )
    # Sideline occlusion invents ~30° / ~179° knees — drop physical impossibles.
    if left_angle is not None and not (70.0 <= left_angle <= 175.0):
        left_angle, left_low = None, True
        flags.append("left_knee_implausible")
    if right_angle is not None and not (70.0 <= right_angle <= 175.0):
        right_angle, right_low = None, True
        flags.append("right_knee_implausible")
    if left_low:
        flags.append("left_knee_low_confidence")
    if right_low:
        flags.append("right_knee_low_confidence")

    angles = [a for a in (left_angle, right_angle) if a is not None]
    mean_knee = float(np.mean(angles)) if angles else None

    conf = pose.keypoints_conf
    ankles_ok = (
        float(conf[L_ANKLE]) >= config.min_keypoint_confidence
        and float(conf[R_ANKLE]) >= config.min_keypoint_confidence
    )
    hip_h = None
    shoulder_h = None
    com_h = None
    torso = None

    if ankles_ok and not pose.low_confidence:
        feet_y = float(ankle_mid(pose.keypoints_xy)[1])
        hips = hip_mid(pose.keypoints_xy)
        shoulders = shoulder_mid(pose.keypoints_xy)
        if not np.any(np.isnan(hips)):
            hip_h = normalized_height_from_feet(float(hips[1]), feet_y, standing_height_px)
        if not np.any(np.isnan(shoulders)) and not np.any(np.isnan(hips)):
            torso_vec = shoulders - hips
            torso = vector_angle_from_vertical_deg(torso_vec)
            shoulder_h = normalized_height_from_feet(
                float(shoulders[1]), feet_y, standing_height_px
            )
        com = estimate_com_xy(pose.keypoints_xy, pose.keypoints_conf, config.min_keypoint_confidence)
        if com is not None:
            com_h = normalized_height_from_feet(float(com[1]), feet_y, standing_height_px)
    else:
        flags.append("ankle_reference_unavailable")

    # Hip height must be physically plausible vs standing height.
    if hip_h is not None and not (0.25 <= hip_h <= 1.05):
        flags.append("hip_height_implausible")
        hip_h = None
    if com_h is not None and not (0.25 <= com_h <= 1.15):
        com_h = None

    posture, pconf = classify_posture(mean_knee, torso, hip_h, config)

    return FrameBodyMetrics(
        frame_idx=pose.frame_idx,
        timestamp_ms=pose.timestamp_ms,
        low_confidence=pose.low_confidence,
        knee_flexion_angle_left=left_angle,
        knee_flexion_angle_right=right_angle,
        knee_flexion_angle_mean=mean_knee,
        hip_height=hip_h,
        torso_angle=torso,
        com_height=com_h,
        shoulder_height=shoulder_h,
        posture=posture,
        posture_confidence=pconf,
        flags=flags,
    )


def smooth_posture_sequence(
    metrics: list[FrameBodyMetrics],
    window: int = 5,
) -> None:
    """In-place majority filter. Skips unknown; requires clear plurality to flip."""
    if len(metrics) < 3:
        return
    labels = [m.posture for m in metrics]
    half = window // 2
    for i in range(len(metrics)):
        lo = max(0, i - half)
        hi = min(len(metrics), i + half + 1)
        neigh = [labels[j] for j in range(lo, hi) if labels[j] != "unknown"]
        if len(neigh) < 3:
            continue
        vals, counts = np.unique(np.asarray(neigh), return_counts=True)
        top = str(vals[int(np.argmax(counts))])
        if int(counts.max()) >= (len(neigh) + 1) // 2 + 1 and top != metrics[i].posture:
            if metrics[i].posture_confidence < 0.12 or metrics[i].posture == "unknown":
                metrics[i].flags.append(f"temporal_smooth:{metrics[i].posture}->{top}")
                metrics[i].posture = top  # type: ignore[assignment]


def summarize_body_position(
    frame_metrics: list[FrameBodyMetrics],
    config: AnalysisConfig,
    fps: float = 30.0,
) -> dict[str, Any]:
    valid = [m for m in frame_metrics if not m.low_confidence]
    knees = [m.knee_flexion_angle_mean for m in valid if m.knee_flexion_angle_mean is not None]
    torsos = [m.torso_angle for m in valid if m.torso_angle is not None]
    hips = [m.hip_height for m in valid if m.hip_height is not None]

    # Weight early set harder — pad level at get-off matters most for coaching.
    early_n = max(5, int(fps * 0.5))
    weights: list[float] = []
    labels: list[str] = []
    for i, m in enumerate(valid):
        if m.posture == "unknown":
            continue
        w = 2.0 if i < early_n else 1.0
        # Prefer clearer frames
        w *= 0.5 + min(1.0, m.posture_confidence * 4.0)
        labels.append(m.posture)
        weights.append(w)

    summary_posture: PostureLabel = "unknown"
    posture_confidence = 0.0
    posture_mixed = False
    if labels:
        classes = sorted(set(labels))
        score = {c: 0.0 for c in classes}
        for lab, w in zip(labels, weights):
            score[lab] += w
        top = max(score, key=score.get)
        total = sum(score.values())
        frac = score[top] / total if total else 0.0
        posture_confidence = float(frac)
        if frac >= config.posture_majority_frac:
            summary_posture = top  # type: ignore[assignment]
        else:
            # Honest: report plurality winner, mark mixed — do NOT invent "balanced"
            summary_posture = top  # type: ignore[assignment]
            posture_mixed = True

    min_hip = min(hips) if hips else None
    min_hip_frame = None
    if min_hip is not None:
        for m in valid:
            if m.hip_height is not None and abs(m.hip_height - min_hip) < 1e-9:
                min_hip_frame = m.frame_idx
                break

    coach_flags: list[str] = []
    if summary_posture in ("knee_bender", "waist_bender") and not posture_mixed:
        coach_flags.append(summary_posture)
    elif posture_mixed and summary_posture != "unknown":
        coach_flags.append("mixed_posture")

    return {
        "mean_knee_flexion_deg": float(np.mean(knees)) if knees else None,
        "min_knee_flexion_deg": float(np.min(knees)) if knees else None,
        "mean_torso_angle_deg": float(np.mean(torsos)) if torsos else None,
        "max_torso_angle_deg": float(np.max(torsos)) if torsos else None,
        "hip_height_at_lowest": min_hip,
        "hip_height_lowest_frame": min_hip_frame,
        "mean_hip_height": float(np.mean(hips)) if hips else None,
        "posture_classification": summary_posture,
        "posture_confidence": posture_confidence,
        "posture_mixed": posture_mixed,
        "posture_frame_counts": {
            str(u): int(c) for u, c in zip(*np.unique(np.asarray(labels), return_counts=True))
        }
        if labels
        else {},
        "valid_frame_count": len(valid),
        "flagged_frame_count": sum(1 for m in frame_metrics if m.low_confidence),
        "coach_flags": coach_flags,
    }
