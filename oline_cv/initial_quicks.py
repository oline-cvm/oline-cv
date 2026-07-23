"""Initial quicks / get-off (Yeager pass §1, run §1)."""

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
    coach_flags: list[str]
    late_off_the_ball: bool
    first_step_acceleration: float | None
    first_step_direction_deg: float | None


def estimate_rep_standing_height(
    poses: list[FramePose],
    snap_frame: int,
    config: AnalysisConfig,
) -> float:
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
        return 200.0
    return float(np.median(heights))


def _baseline_point(poses, snap_frame, extractor, config):
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


def _ankle_point(pose, config):
    conf, xy = pose.keypoints_conf, pose.keypoints_xy
    pts = []
    for idx in (L_ANKLE, R_ANKLE):
        if float(conf[idx]) >= config.min_keypoint_confidence and not np.any(np.isnan(xy[idx])):
            pts.append(xy[idx])
    if not pts:
        return None
    return np.mean(np.asarray(pts), axis=0)


def _hip_point(pose, config):
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


def _first_monotonic_movement(poses, snap_frame, baseline, extractor, standing_height_px, config):
    thresh = max(config.movement_threshold_frac * standing_height_px, config.movement_min_px)
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
        vals = [float(v) for v in window]
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
            f"source_fps_{fps:.1f}_below_60_reaction_quantized_to_{1000.0 / fps:.1f}ms_steps"
        )
    standing = estimate_rep_standing_height(poses, snap_frame, config)
    if 199.5 <= standing <= 200.5:
        notes.append("standing_height_fallback_default_used")

    foot_base = _baseline_point(poses, snap_frame, _ankle_point, config)
    hip_base = _baseline_point(poses, snap_frame, _hip_point, config)
    foot_frame = (
        _first_monotonic_movement(poses, snap_frame, foot_base, _ankle_point, standing, config)
        if foot_base is not None
        else None
    )
    hip_frame = (
        _first_monotonic_movement(poses, snap_frame, hip_base, _hip_point, standing, config)
        if hip_base is not None
        else None
    )
    if foot_base is None:
        notes.append("foot_baseline_unavailable")
    if hip_base is None:
        notes.append("hip_baseline_unavailable")

    valid = [(f, p) for f, p in ((foot_frame, "foot"), (hip_frame, "hip")) if f is not None]
    coach_flags: list[str] = []
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
            coach_flags=["late_off_the_ball"],
            late_off_the_ball=True,
            first_step_acceleration=None,
            first_step_direction_deg=None,
        )

    first_frame = min(f for f, _ in valid)
    initiators = [p for f, p in valid if f == first_frame]
    initiated: BodyPart = initiators[0] if len(initiators) == 1 else "foot"  # type: ignore
    if len(initiators) > 1:
        notes.append("foot_and_hip_same_frame")

    rt_frames = first_frame - snap_frame
    rt_ms = (rt_frames / fps) * 1000.0 if fps > 0 else None
    # At 30fps, 0–1 frame is quantization noise — don't crown "initial_quicks".
    frame_ms = 1000.0 / fps if fps > 0 else 33.0
    reaction_reliable = rt_frames is not None and rt_frames >= 2
    if not reaction_reliable:
        notes.append(
            f"reaction_{rt_frames}f_unreliable_at_{fps:.0f}fps_bin_{frame_ms:.0f}ms"
        )

    late = rt_ms is not None and rt_ms > config.late_off_ball_ms
    if late:
        coach_flags.append("late_off_the_ball")
    elif reaction_reliable:
        coach_flags.append("initial_quicks")
        if config.play_type == "run":
            coach_flags.append("get_off")

    accel = None
    direction = None
    if foot_base is not None and foot_frame is not None:
        pts = []
        for pose in poses[foot_frame : min(len(poses), foot_frame + 5)]:
            pt = _ankle_point(pose, config)
            if pt is not None:
                pts.append(pt)
        if len(pts) >= 3 and fps > 0:
            # Signed speed along first-step displacement (not absolute-norm delta,
            # which was producing nonsense negative "accel" from noise).
            step_vec = pts[-1] - pts[0]
            step_n = float(np.linalg.norm(step_vec))
            if step_n > 1e-3:
                u = step_vec / step_n
                speeds = []
                for a, b in zip(pts[:-1], pts[1:]):
                    speeds.append(float(np.dot(b - a, u)) * fps)
                if len(speeds) >= 2:
                    accel = ((speeds[1] - speeds[0]) * fps) / standing
                direction = float(np.degrees(np.arctan2(step_vec[0], -step_vec[1])))

    return InitialQuicksResult(
        snap_frame=snap_frame,
        first_foot_movement_frame=foot_frame,
        first_hip_movement_frame=hip_frame,
        reaction_time_frames=rt_frames,
        reaction_time_ms=rt_ms,
        initiated_by=initiated,
        standing_height_px=standing,
        notes=notes,
        coach_flags=coach_flags,
        late_off_the_ball=bool(late),
        first_step_acceleration=accel,
        first_step_direction_deg=direction,
    )
