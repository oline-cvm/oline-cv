"""Module 1b — Initial quicks (snap reaction timing)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from oline_cv.config import AnalysisConfig, L_ANKLE, L_HIP, R_ANKLE, R_HIP
from oline_cv.geometry import estimate_standing_height_px, hip_mid
from oline_cv.pose_tracker import FramePose


BodyPart = Literal["foot", "hip", "unknown"]


@dataclass
class InitialQuicksResult:
    snap_frame: int
    first_foot_movement_frame: int | None
    first_hip_movement_frame: int | None
    reaction_time_frames: int | None
    reaction_time_ms: float | None
    initiated_by: BodyPart
    standing_height_px: float
    notes: list[str]


def estimate_rep_standing_height(
    poses: list[FramePose],
    snap_frame: int,
    config: AnalysisConfig,
) -> float:
    """Median standing height from pre-snap usable frames."""
    start = max(0, snap_frame - config.standing_height_pre_snap_frames)
    heights: list[float] = []
    for pose in poses[start:snap_frame]:
        if not pose.usable:
            continue
        h = estimate_standing_height_px(
            pose.keypoints_xy,
            pose.keypoints_conf,
            config.min_keypoint_confidence,
            config.nose_to_hip_height_factor,
        )
        if h is not None and h > 1.0:
            heights.append(h)
    if not heights:
        # Fall back to any usable frame near snap
        for pose in poses[max(0, snap_frame - 5) : snap_frame + 15]:
            h = estimate_standing_height_px(
                pose.keypoints_xy,
                pose.keypoints_conf,
                config.min_keypoint_confidence,
                config.nose_to_hip_height_factor,
            )
            if h is not None and h > 1.0:
                heights.append(h)
    if not heights:
        return 200.0  # last-resort pixel default; flagged in notes by caller
    return float(np.median(heights))


def _baseline_point(
    poses: list[FramePose],
    snap_frame: int,
    extractor,
    config: AnalysisConfig,
) -> np.ndarray | None:
    """Median (x,y) over a pre-snap window, ending a few frames before snap."""
    end = max(0, snap_frame - config.baseline_gap_frames)
    start = max(0, snap_frame - config.baseline_lookback_frames)
    if end <= start:
        end = snap_frame
        start = max(0, snap_frame - 5)
    pts = []
    for pose in poses[start:end]:
        if pose.low_confidence:
            continue
        pt = extractor(pose, config)
        if pt is not None and not np.any(np.isnan(pt)):
            pts.append(pt)
    if not pts:
        return None
    return np.median(np.asarray(pts), axis=0)


def _ankle_point(pose: FramePose, config: AnalysisConfig) -> np.ndarray | None:
    conf = pose.keypoints_conf
    xy = pose.keypoints_xy
    pts = []
    for idx in (L_ANKLE, R_ANKLE):
        if float(conf[idx]) >= config.min_keypoint_confidence and not np.any(np.isnan(xy[idx])):
            pts.append(xy[idx])
    if not pts:
        return None
    return np.mean(np.asarray(pts), axis=0)


def _hip_point(pose: FramePose, config: AnalysisConfig) -> np.ndarray | None:
    conf = pose.keypoints_conf
    if (
        float(conf[L_HIP]) < config.min_keypoint_confidence
        or float(conf[R_HIP]) < config.min_keypoint_confidence
    ):
        return None
    mid = hip_mid(pose.keypoints_xy)
    if np.any(np.isnan(mid)):
        return None
    return mid


def _first_monotonic_movement(
    poses: list[FramePose],
    snap_frame: int,
    baseline: np.ndarray,
    extractor,
    standing_height_px: float,
    config: AnalysisConfig,
) -> int | None:
    """First frame where displacement exceeds threshold with monotonic sustain.

    Rejects single-frame spikes from pre-snap stance adjustments by requiring:
      1) displacement > movement_threshold_frac * standing_height
      2) sustained for movement_sustain_frames
      3) cumulative displacement non-decreasing over movement_monotonic_window
    """
    thresh = max(
        config.movement_threshold_frac * standing_height_px,
        config.movement_min_px,
    )
    # Search starts the frame AFTER snap (snap is frame 0; reaction ≥ 1 frame).
    search_start = snap_frame + 1
    end = min(len(poses), snap_frame + config.reaction_search_max_frames + 1)
    displacements: list[float | None] = []

    for pose in poses[search_start:end]:
        if pose.low_confidence:
            displacements.append(None)
            continue
        pt = extractor(pose, config)
        if pt is None:
            displacements.append(None)
            continue
        displacements.append(float(np.linalg.norm(pt - baseline)))

    n = len(displacements)
    sustain = config.movement_sustain_frames
    mono_w = config.movement_monotonic_window

    for i in range(n):
        window_end = i + max(sustain, mono_w)
        if window_end > n:
            break
        window = displacements[i:window_end]
        if any(v is None for v in window):
            continue
        vals = [float(v) for v in window]  # type: ignore[arg-type]
        if not all(v >= thresh for v in vals[:sustain]):
            continue
        mono = vals[:mono_w]
        if all(mono[j] <= mono[j + 1] + 1e-3 for j in range(len(mono) - 1)):
            return search_start + i

    return None


def analyze_initial_quicks(
    poses: list[FramePose],
    snap_frame: int,
    fps: float,
    config: AnalysisConfig,
) -> InitialQuicksResult:
    notes: list[str] = []
    if fps < 59:
        notes.append(
            f"source_fps_{fps:.1f}_below_60_reaction_quantized_to_{1000.0/fps:.1f}ms_steps"
        )
    standing = estimate_rep_standing_height(poses, snap_frame, config)
    if standing <= 200.5 and standing >= 199.5:
        notes.append("standing_height_fallback_default_used")

    foot_base = _baseline_point(poses, snap_frame, _ankle_point, config)
    hip_base = _baseline_point(poses, snap_frame, _hip_point, config)

    foot_frame = None
    hip_frame = None
    if foot_base is not None:
        foot_frame = _first_monotonic_movement(
            poses, snap_frame, foot_base, _ankle_point, standing, config
        )
    else:
        notes.append("foot_baseline_unavailable")

    if hip_base is not None:
        hip_frame = _first_monotonic_movement(
            poses, snap_frame, hip_base, _hip_point, standing, config
        )
    else:
        notes.append("hip_baseline_unavailable")

    candidates = [(foot_frame, "foot"), (hip_frame, "hip")]
    valid = [(f, p) for f, p in candidates if f is not None]
    if not valid:
        return InitialQuicksResult(
            snap_frame=snap_frame,
            first_foot_movement_frame=foot_frame,
            first_hip_movement_frame=hip_frame,
            reaction_time_frames=None,
            reaction_time_ms=None,
            initiated_by="unknown",
            standing_height_px=standing,
            notes=notes + ["no_movement_detected_in_search_window"],
        )

    # Earlier frame wins; if same frame, prefer the part with larger displacement delta
    first_frame = min(f for f, _ in valid)
    initiators = [p for f, p in valid if f == first_frame]
    initiated: BodyPart
    if len(initiators) == 1:
        initiated = initiators[0]  # type: ignore[assignment]
    else:
        initiated = "foot"  # tie → foot (both moved same frame; still note)
        notes.append("foot_and_hip_same_frame")

    rt_frames = first_frame - snap_frame
    rt_ms = (rt_frames / fps) * 1000.0 if fps > 0 else None

    return InitialQuicksResult(
        snap_frame=snap_frame,
        first_foot_movement_frame=foot_frame,
        first_hip_movement_frame=hip_frame,
        reaction_time_frames=rt_frames,
        reaction_time_ms=rt_ms,
        initiated_by=initiated,
        standing_height_px=standing,
        notes=notes,
    )
