"""End-to-end OL pass-set analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oline_cv.body_position import (
    compute_frame_body_metrics,
    summarize_body_position,
)
from oline_cv.config import AnalysisConfig
from oline_cv.initial_quicks import analyze_initial_quicks
from oline_cv.pose_tracker import PoseTracker, keypoints_as_dict
from oline_cv.snap_detection import detect_snap
from oline_cv.visualize import write_overlay_video


def determine_set_end(
    poses_len: int,
    snap_frame: int,
    poses_usable_flags: list[bool],
    config: AnalysisConfig,
) -> int:
    if config.set_end_frame_override is not None:
        return min(config.set_end_frame_override, poses_len - 1)

    end = min(poses_len - 1, snap_frame + config.set_max_frames)
    # Only trip "pose lost" after we have seen at least one usable post-snap frame.
    seen_usable = False
    lost = 0
    for i in range(snap_frame, end + 1):
        if poses_usable_flags[i]:
            seen_usable = True
            lost = 0
            continue
        if not seen_usable:
            continue
        lost += 1
        if lost >= config.pose_lost_end_frames:
            return max(snap_frame, i - config.pose_lost_end_frames)
    return end


def analyze_video(
    video_path: str,
    config: AnalysisConfig | None = None,
    output_json: str | None = None,
    overlay_path: str | None = None,
) -> dict[str, Any]:
    config = config or AnalysisConfig()
    video_path = str(video_path)
    tracker = PoseTracker(config)
    fps, n_frames, width, height, poses, frames = tracker.extract_all(video_path)

    snap = detect_snap(frames, config)
    quicks = analyze_initial_quicks(poses, snap.snap_frame, fps, config)

    usable = [p.usable for p in poses]
    set_end = determine_set_end(len(poses), snap.snap_frame, usable, config)

    body_metrics = [
        compute_frame_body_metrics(poses[i], quicks.standing_height_px, config)
        for i in range(snap.snap_frame, set_end + 1)
    ]
    body_summary = summarize_body_position(body_metrics, config)

    per_frame: list[dict[str, Any]] = []
    body_by_idx = {m.frame_idx: m for m in body_metrics}
    for pose in poses[snap.snap_frame : set_end + 1]:
        m = body_by_idx[pose.frame_idx]
        per_frame.append(
            {
                "frame_idx": pose.frame_idx,
                "timestamp_ms": round(pose.timestamp_ms, 3),
                "low_confidence": m.low_confidence,
                "flags": m.flags,
                "keypoints": keypoints_as_dict(pose),
                "knee_flexion_angle": {
                    "left": m.knee_flexion_angle_left,
                    "right": m.knee_flexion_angle_right,
                    "mean": m.knee_flexion_angle_mean,
                },
                "hip_height": m.hip_height,
                "shoulder_height": m.shoulder_height,
                "torso_angle": m.torso_angle,
                "com_height": m.com_height,
                "posture": m.posture,
            }
        )

    result: dict[str, Any] = {
        "video": {
            "path": video_path,
            "fps": fps,
            "frame_count": n_frames,
            "width": width,
            "height": height,
        },
        "target_jersey": config.target_jersey,
        "config": config.to_dict(),
        "snap": {
            "snap_frame": snap.snap_frame,
            "method": snap.method,
            "motion_score": snap.motion_score,
            "confidence": snap.confidence,
        },
        "initial_quicks": {
            "snap_frame": quicks.snap_frame,
            "first_foot_movement_frame": quicks.first_foot_movement_frame,
            "first_hip_movement_frame": quicks.first_hip_movement_frame,
            "reaction_time_frames": quicks.reaction_time_frames,
            "reaction_time_ms": quicks.reaction_time_ms,
            "initiated_by": quicks.initiated_by,
            "standing_height_px": quicks.standing_height_px,
            "notes": quicks.notes,
        },
        "set": {
            "start_frame": snap.snap_frame,
            "end_frame": set_end,
        },
        "rep_summary": {
            "target_jersey": config.target_jersey,
            "reaction_time_frames": quicks.reaction_time_frames,
            "reaction_time_ms": quicks.reaction_time_ms,
            "initiated_by": quicks.initiated_by,
            **body_summary,
        },
        "frames": per_frame,
    }

    if output_json is None:
        output_json = str(Path(video_path).with_suffix("")) + "_analysis.json"
    Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["output_json"] = output_json

    if config.write_overlay_video:
        if overlay_path is None:
            overlay_path = str(Path(video_path).with_suffix("")) + config.overlay_suffix
        write_overlay_video(
            frames, poses, body_metrics, snap, quicks, fps, overlay_path, config=config
        )
        result["overlay_video"] = overlay_path

    return result


def result_brief(result: dict[str, Any]) -> str:
    s = result["rep_summary"]
    q = result["initial_quicks"]
    return (
        f"snap_frame={result['snap']['snap_frame']} "
        f"reaction={q['reaction_time_ms']}ms ({q['initiated_by']}-first) "
        f"posture={s['posture_classification']} "
        f"mean_knee={s['mean_knee_flexion_deg']} "
        f"mean_torso={s['mean_torso_angle_deg']} "
        f"min_hip_h={s['hip_height_at_lowest']}"
    )
