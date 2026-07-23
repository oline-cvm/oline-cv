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
    flags: list[str]


def classify_posture(
    knee_flexion_mean: float | None,
    torso_angle: float | None,
    hip_height: float | None,
    config: AnalysisConfig,
) -> PostureLabel:
    """Distinguish knee bender vs waist bender.

    Knee bender: low hip height via high knee flexion; torso near-vertical.
    Waist bender: large torso lean; knees relatively extended.
    """
    if knee_flexion_mean is None or torso_angle is None or hip_height is None:
        return "unknown"

    flexed = knee_flexion_mean <= config.knee_bender_flexion_deg
    extended = knee_flexion_mean > config.knee_bender_flexion_deg
    upright = torso_angle <= config.waist_bender_torso_deg
    leaning = torso_angle > config.waist_bender_torso_deg
    low_hips = hip_height <= config.low_hip_height_frac

    if low_hips and flexed and upright:
        return "knee_bender"
    if leaning and extended:
        return "waist_bender"
    if low_hips and flexed and leaning:
        # Both cues — lean dominates diagnostically as waist bend tendency
        return "waist_bender"
    if upright and flexed:
        return "knee_bender"
    if leaning:
        return "waist_bender"
    return "balanced"


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
    if left_low:
        flags.append("left_knee_low_confidence")
    if right_low:
        flags.append("right_knee_low_confidence")

    angles = [a for a in (left_angle, right_angle) if a is not None]
    mean_knee = float(np.mean(angles)) if angles else None

    # Foot reference for normalized heights
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
            torso_vec = shoulders - hips  # pointing toward shoulders from hips
            torso = vector_angle_from_vertical_deg(torso_vec)
            shoulder_h = normalized_height_from_feet(
                float(shoulders[1]), feet_y, standing_height_px
            )
        com = estimate_com_xy(pose.keypoints_xy, pose.keypoints_conf, config.min_keypoint_confidence)
        if com is not None:
            com_h = normalized_height_from_feet(float(com[1]), feet_y, standing_height_px)
    else:
        flags.append("ankle_reference_unavailable")

    posture = classify_posture(mean_knee, torso, hip_h, config)

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
        flags=flags,
    )


def summarize_body_position(
    frame_metrics: list[FrameBodyMetrics],
    config: AnalysisConfig,
) -> dict[str, Any]:
    valid = [m for m in frame_metrics if not m.low_confidence]
    knees = [m.knee_flexion_angle_mean for m in valid if m.knee_flexion_angle_mean is not None]
    torsos = [m.torso_angle for m in valid if m.torso_angle is not None]
    hips = [m.hip_height for m in valid if m.hip_height is not None]

    labels = [m.posture for m in valid if m.posture != "unknown"]
    summary_posture: PostureLabel = "unknown"
    if labels:
        # Majority vote
        unique, counts = np.unique(np.asarray(labels), return_counts=True)
        top = str(unique[int(np.argmax(counts))])
        frac = float(counts.max()) / len(labels)
        if frac >= config.posture_majority_frac:
            summary_posture = top  # type: ignore[assignment]
        else:
            summary_posture = "balanced" if "balanced" in labels else top  # type: ignore[assignment]

    min_hip = min(hips) if hips else None
    min_hip_frame = None
    if min_hip is not None:
        for m in valid:
            if m.hip_height is not None and abs(m.hip_height - min_hip) < 1e-9:
                min_hip_frame = m.frame_idx
                break

    return {
        "mean_knee_flexion_deg": float(np.mean(knees)) if knees else None,
        "min_knee_flexion_deg": float(np.min(knees)) if knees else None,
        "mean_torso_angle_deg": float(np.mean(torsos)) if torsos else None,
        "max_torso_angle_deg": float(np.max(torsos)) if torsos else None,
        "hip_height_at_lowest": min_hip,
        "hip_height_lowest_frame": min_hip_frame,
        "mean_hip_height": float(np.mean(hips)) if hips else None,
        "posture_classification": summary_posture,
        "posture_frame_counts": {
            str(u): int(c) for u, c in zip(*np.unique(np.asarray(labels), return_counts=True))
        }
        if labels
        else {},
        "valid_frame_count": len(valid),
        "flagged_frame_count": sum(1 for m in frame_metrics if m.low_confidence),
    }
