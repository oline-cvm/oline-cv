"""End-to-end OL analysis — Yeager pass-protection & run-blocking framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from oline_cv.body_position import (
    compute_frame_body_metrics,
    smooth_posture_sequence,
    summarize_body_position,
)
from oline_cv.config import AnalysisConfig
from oline_cv.engagement import (
    analyze_anchor,
    analyze_hands,
    analyze_mirror_redirect,
    analyze_sustain,
)
from oline_cv.footwork import analyze_footwork
from oline_cv.initial_quicks import analyze_initial_quicks
from oline_cv.pose_tracker import PoseTracker, keypoints_as_dict
from oline_cv.run_game import analyze_com_balance, analyze_movement_in_space, analyze_point_of_attack
from oline_cv.series import build_series
from oline_cv.snap_detection import detect_snap
from oline_cv.visualize import write_overlay_video


def determine_set_end(poses_len, snap_frame, poses_usable_flags, config):
    if config.set_end_frame_override is not None:
        return min(config.set_end_frame_override, poses_len - 1)
    end = min(poses_len - 1, snap_frame + config.set_max_frames)
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


def _empty_dl_poses(ol_poses):
    return [None] * len(ol_poses)


def analyze_video(
    video_path: str,
    config: AnalysisConfig | None = None,
    output_json: str | None = None,
    overlay_path: str | None = None,
    progress_cb=None,
) -> dict[str, Any]:
    config = config or AnalysisConfig()
    video_path = str(video_path)

    def _prog(pct: float, msg: str, stage: str = "analyze") -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(float(pct), str(msg), str(stage))
        except Exception:
            pass

    _prog(2, "Starting analysis…", "ingest")
    tracker = PoseTracker(config)
    fps, n_frames, width, height, ol_poses, dl_poses, frames = tracker.extract_all(
        video_path, progress_cb=progress_cb
    )
    ol_lock = getattr(tracker, "lock_meta", {}) or {}

    _prog(74, "Detecting snap…", "metrics")
    snap = detect_snap(frames, config)
    _prog(76, "Measuring get-off / reaction…", "metrics")
    quicks = analyze_initial_quicks(ol_poses, snap.snap_frame, fps, config)
    usable = [p.usable for p in ol_poses]
    set_end = determine_set_end(len(ol_poses), snap.snap_frame, usable, config)

    ol_series = build_series(
        ol_poses, snap.snap_frame, set_end, quicks.standing_height_px, fps, config
    )
    # Align DL poses list (may contain None frames)
    dl_filled = []
    for i, p in enumerate(dl_poses):
        if p is None:
            # placeholder empty pose with same index
            from oline_cv.pose_tracker import FramePose
            import numpy as np

            dl_filled.append(
                FramePose(
                    frame_idx=i,
                    timestamp_ms=(i / fps) * 1000 if fps else 0,
                    keypoints_xy=np.full((17, 2), np.nan),
                    keypoints_conf=np.zeros(17),
                    bbox_xyxy=None,
                    person_confidence=0.0,
                    low_confidence=True,
                    usable=False,
                )
            )
        else:
            dl_filled.append(p)
    dl_series = build_series(
        dl_filled, snap.snap_frame, set_end, quicks.standing_height_px, fps, config
    )
    # If defender never usable, treat as missing
    dl_ok = bool(dl_series.usable.any())
    dl_arg = dl_series if dl_ok else None

    mode = config.play_type if config.play_type != "auto" else "pass"

    _prog(78, "Computing posture & footwork…", "metrics")
    body_metrics = [
        compute_frame_body_metrics(ol_poses[i], quicks.standing_height_px, config)
        for i in range(snap.snap_frame, set_end + 1)
    ]
    smooth_posture_sequence(body_metrics, window=5)

    # NN only on borderline frames — not a circular override of clear rule labels.
    if config.use_nn_posture:
        try:
            from oline_cv.nn.infer import classify_window

            nn_hits = 0
            # NN only fills unknowns — never overrides a clear geometric label.
            for m in body_metrics:
                if m.posture != "unknown":
                    continue
                pred = classify_window(ol_poses, quicks.standing_height_px, m.frame_idx)
                if pred is None:
                    break
                label, conf = pred
                if label == "unknown" or conf < config.nn_posture_min_confidence:
                    continue
                m.flags.append(f"nn_fill:{label}:{conf:.2f}")
                m.posture = label  # type: ignore[assignment]
                m.posture_confidence = conf
                nn_hits += 1
            if nn_hits:
                print(f"  NN filled {nn_hits} unknown frames", flush=True)
                smooth_posture_sequence(body_metrics, window=3)
        except Exception as exc:
            print(f"  NN posture skipped: {exc}", flush=True)

    body_summary = summarize_body_position(body_metrics, config, fps=fps)
    _prog(82, "Running Yeager modules…", "metrics")
    balance = analyze_com_balance(ol_series, config)
    footwork = analyze_footwork(ol_series, config, mode=mode)
    mirror = analyze_mirror_redirect(ol_series, dl_arg, config)
    anchor = analyze_anchor(ol_series, dl_arg, config)
    hands = analyze_hands(ol_series, dl_arg, config)
    sustain = analyze_sustain(ol_series, dl_arg, config)
    # Run modules only on run plays — DL travel on pass is not a drive block.
    if mode == "run":
        poa = analyze_point_of_attack(ol_series, dl_arg, config)
        space = analyze_movement_in_space(ol_series, config)
    else:
        poa = {"available": False, "notes": ["skipped_pass_play"], "coach_flags": []}
        space = {"available": False, "notes": ["skipped_pass_play"], "coach_flags": []}

    # Collect coach language — posture via body_summary.coach_flags only
    coach_flags = []
    for block in (
        quicks.coach_flags,
        footwork.get("coach_flags", []),
        mirror.get("coach_flags", []),
        anchor.get("coach_flags", []),
        hands.get("coach_flags", []),
        sustain.get("coach_flags", []),
        balance.get("coach_flags", []),
        poa.get("coach_flags", []),
        space.get("coach_flags", []),
        body_summary.get("coach_flags", []),
    ):
        for f in block or []:
            if f and f not in coach_flags and f != "unknown":
                coach_flags.append(f)

    per_frame: list[dict[str, Any]] = []
    body_by_idx = {m.frame_idx: m for m in body_metrics}
    for pose in ol_poses[snap.snap_frame : set_end + 1]:
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
                "posture_confidence": round(m.posture_confidence, 4),
            }
        )

    modules = {
        "initial_quicks": {
            "snap_frame": quicks.snap_frame,
            "first_foot_movement_frame": quicks.first_foot_movement_frame,
            "first_hip_movement_frame": quicks.first_hip_movement_frame,
            "reaction_time_frames": quicks.reaction_time_frames,
            "reaction_time_ms": quicks.reaction_time_ms,
            "initiated_by": quicks.initiated_by,
            "late_off_the_ball": quicks.late_off_the_ball,
            "first_step_acceleration": quicks.first_step_acceleration,
            "first_step_direction_deg": quicks.first_step_direction_deg,
            "standing_height_px": quicks.standing_height_px,
            "coach_flags": quicks.coach_flags,
            "notes": quicks.notes,
        },
        "footwork": footwork,
        "mirror_redirect": mirror,
        "anchor": anchor,
        "body_position": body_summary,
        "balance": balance,
        "hands": hands,
        "sustain": sustain,
        "point_of_attack": poa,
        "movement_in_space": space,
    }

    result: dict[str, Any] = {
        "video": {
            "path": video_path,
            "fps": fps,
            "frame_count": n_frames,
            "width": width,
            "height": height,
        },
        "play_type": mode,
        "target_jersey": config.target_jersey,
        "ol_lock": ol_lock,
        "config": config.to_dict(),
        "snap": {
            "snap_frame": snap.snap_frame,
            "method": snap.method,
            "motion_score": snap.motion_score,
            "confidence": snap.confidence,
        },
        "set": {"start_frame": snap.snap_frame, "end_frame": set_end},
        "modules": modules,
        # Back-compat aliases
        "initial_quicks": modules["initial_quicks"],
        "coach_language": coach_flags,
        "rep_summary": {
            "target_jersey": config.target_jersey,
            "ol_lock": ol_lock,
            "play_type": mode,
            "reaction_time_ms": quicks.reaction_time_ms,
            "reaction_time_frames": quicks.reaction_time_frames,
            "initiated_by": quicks.initiated_by,
            "late_off_the_ball": quicks.late_off_the_ball,
            "posture_classification": body_summary.get("posture_classification"),
            "posture_confidence": body_summary.get("posture_confidence"),
            "posture_mixed": body_summary.get("posture_mixed"),
            "mean_knee_flexion_deg": body_summary.get("mean_knee_flexion_deg"),
            "min_knee_flexion_deg": body_summary.get("min_knee_flexion_deg"),
            "mean_torso_angle_deg": body_summary.get("mean_torso_angle_deg"),
            "hip_height_at_lowest": body_summary.get("hip_height_at_lowest"),
            "mean_hip_height": body_summary.get("mean_hip_height"),
            "step_cadence_hz": footwork.get("step_cadence_hz"),
            "set_depth": footwork.get("set_depth"),
            "set_width": footwork.get("set_width"),
            "mean_base_width": footwork.get("mean_base_width"),
            "overset": footwork.get("overset"),
            "lateral_match": mirror.get("lateral_match_correlation"),
            "anchor_give": anchor.get("max_hip_displacement_after_contact"),
            "punch_ms": hands.get("time_to_first_contact_ms"),
            "engagement_ms": sustain.get("engagement_ms"),
            "contacted": sustain.get("contacted"),
            "coach_language": coach_flags,
            "defender_tracked": dl_ok,
            **{
                k: body_summary[k]
                for k in (
                    "posture_frame_counts",
                    "valid_frame_count",
                    "flagged_frame_count",
                )
                if k in body_summary
            },
        },
        "frames": per_frame,
    }

    from oline_cv.trust import compute_trust

    result["trust"] = compute_trust(result)
    result["rep_summary"]["trust_overall"] = result["trust"]["overall"]

    if output_json is None:
        output_json = str(Path(video_path).with_suffix("")) + "_analysis.json"
    Path(output_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["output_json"] = output_json

    # Phase 1 of the 3D pipeline: serialize the track while frames are still in RAM.
    if config.motion3d_export_dir:
        _prog(86, "Exporting track for 3D reconstruction…", "overlay")
        try:
            from oline_cv.motion3d.track_export import export_tracks

            manifest = export_tracks(
                video_path,
                ol_poses,
                frames,
                fps,
                width,
                height,
                config.motion3d_export_dir,
                target_jersey=config.target_jersey,
                ol_lock=ol_lock,
                snap_frame=snap.snap_frame,
                set_end=set_end,
                crop_size=config.motion3d_crop_size,
                crop_pad=config.motion3d_crop_pad,
                max_interp_gap=config.motion3d_max_interp_gap,
                save_full_frames=config.motion3d_save_full_frames,
            )
            result["motion3d_export"] = {
                "dir": str(config.motion3d_export_dir),
                "tracks_json": str(Path(config.motion3d_export_dir) / "tracks.json"),
                "stats": manifest.stats(),
            }
        except Exception as exc:
            print(f"  motion3d export failed: {exc}", flush=True)
            result["motion3d_export"] = {"error": str(exc)}

    if config.write_overlay_video:
        if overlay_path is None:
            overlay_path = str(Path(video_path).with_suffix("")) + config.overlay_suffix
        _prog(88, "Rendering overlay film…", "overlay")
        write_overlay_video(
            frames, ol_poses, body_metrics, snap, quicks, fps, overlay_path, config=config
        )
        result["overlay_video"] = overlay_path

    _prog(94, "Analysis complete", "overlay")
    return result


def result_brief(result: dict[str, Any]) -> str:
    s = result["rep_summary"]
    flags = ", ".join(s.get("coach_language", [])[:6]) or "—"
    label = f"#{s.get('target_jersey')}" if s.get("target_jersey") is not None else "OL"
    return (
        f"{label} {s.get('play_type')} "
        f"react={s.get('reaction_time_ms')}ms "
        f"posture={s.get('posture_classification')} "
        f"cadence={s.get('step_cadence_hz')} "
        f"flags=[{flags}]"
    )
