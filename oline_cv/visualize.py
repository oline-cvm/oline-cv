"""Overlay video focused on a single athlete (#76) — no other skeletons."""

from __future__ import annotations

import cv2
import numpy as np

from oline_cv.body_position import FrameBodyMetrics
from oline_cv.config import (
    AnalysisConfig,
    L_ANKLE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
)
from oline_cv.initial_quicks import InitialQuicksResult
from oline_cv.pose_tracker import FramePose
from oline_cv.snap_detection import SnapResult


SKELETON = [
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_HIP),
    (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    (L_HIP, L_KNEE),
    (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE),
    (R_KNEE, R_ANKLE),
]


def write_overlay_video(
    frames: list[np.ndarray],
    poses: list[FramePose],
    body: list[FrameBodyMetrics],
    snap: SnapResult,
    quicks: InitialQuicksResult,
    fps: float,
    out_path: str,
    config: AnalysisConfig | None = None,
) -> None:
    if not frames:
        return
    config = config or AnalysisConfig()
    jersey = config.target_jersey
    label = f"#{jersey}" if jersey is not None else "OL"
    zoom = config.overlay_zoom_on_athlete
    out_size = config.overlay_zoom_size if zoom else None

    body_by_idx = {m.frame_idx: m for m in body}
    # Smooth crop center from bboxes
    centers: list[np.ndarray] = []
    sizes: list[float] = []
    for pose in poses:
        if pose.bbox_xyxy is not None:
            b = pose.bbox_xyxy
            centers.append((b[:2] + b[2:]) / 2.0)
            sizes.append(float(max(b[2] - b[0], b[3] - b[1])))
        else:
            centers.append(centers[-1] if centers else np.array([frames[0].shape[1] / 2, frames[0].shape[0] / 2]))
            sizes.append(sizes[-1] if sizes else 200.0)

    # EMA smooth
    sm_c = centers[0].astype(float)
    sm_s = float(sizes[0])
    smooth_c: list[np.ndarray] = []
    smooth_s: list[float] = []
    for c, s in zip(centers, sizes):
        sm_c = 0.85 * sm_c + 0.15 * c
        sm_s = 0.85 * sm_s + 0.15 * s
        smooth_c.append(sm_c.copy())
        smooth_s.append(sm_s)

    if zoom and out_size:
        writer_w = writer_h = out_size
    else:
        writer_h, writer_w = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps if fps > 0 else 30.0, (writer_w, writer_h))

    for i, (frame, pose) in enumerate(zip(frames, poses)):
        if zoom and out_size:
            img = _zoom_crop(frame, smooth_c[i], smooth_s[i], out_size)
            # Remap keypoints into zoom space for drawing
            pose_draw = _remap_pose_to_zoom(pose, frame.shape, smooth_c[i], smooth_s[i], out_size)
        else:
            img = frame.copy()
            pose_draw = pose

        _draw_skeleton(img, pose_draw)
        if pose_draw.bbox_xyxy is not None:
            b = pose_draw.bbox_xyxy.astype(int)
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (40, 200, 120), 2)

        m = body_by_idx.get(pose.frame_idx)
        state = getattr(pose, "track_state", None) or ""
        hud = [label]
        if pose.frame_idx == snap.snap_frame:
            hud.append("SNAP")
        if quicks.first_foot_movement_frame == pose.frame_idx:
            hud.append("FOOT FIRST")
        if quicks.first_hip_movement_frame == pose.frame_idx:
            hud.append("HIP FIRST")
        if quicks.reaction_time_ms is not None and pose.frame_idx >= snap.snap_frame:
            if pose.frame_idx == (quicks.first_foot_movement_frame or -1) or pose.frame_idx == (
                quicks.first_hip_movement_frame or -1
            ):
                hud.append("GET-OFF")
        if state == "LOST":
            hud.append("TRACK LOST")
        if m is not None and m.posture and m.posture != "unknown":
            hud.append(str(m.posture).replace("_", " ").upper())

        _draw_hud(img, hud, jersey)
        writer.write(img)
    writer.release()


def _zoom_crop(
    frame: np.ndarray,
    center: np.ndarray,
    size: float,
    out_size: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    half = max(size * 0.95, 80.0)
    cx, cy = float(center[0]), float(center[1])
    x0 = int(cx - half)
    y0 = int(cy - half)
    x1 = int(cx + half)
    y1 = int(cy + half)
    # Pad if out of bounds
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - w)
    pad_b = max(0, y1 - h)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    crop = frame[y0:y1, x0:x1]
    if pad_l or pad_t or pad_r or pad_b:
        crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(20, 28, 22))
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def _remap_pose_to_zoom(
    pose: FramePose,
    frame_shape: tuple,
    center: np.ndarray,
    size: float,
    out_size: int,
) -> FramePose:
    h, w = frame_shape[:2]
    half = max(size * 0.95, 80.0)
    cx, cy = float(center[0]), float(center[1])
    x0 = cx - half
    y0 = cy - half
    scale = out_size / (2 * half)

    def map_xy(pt: np.ndarray) -> np.ndarray:
        return np.array([(pt[0] - x0) * scale, (pt[1] - y0) * scale], dtype=float)

    kxy = pose.keypoints_xy.copy()
    for i in range(17):
        if not np.any(np.isnan(kxy[i])):
            kxy[i] = map_xy(kxy[i])

    bbox = None
    if pose.bbox_xyxy is not None:
        b = pose.bbox_xyxy
        p0 = map_xy(b[:2])
        p1 = map_xy(b[2:])
        bbox = np.array([p0[0], p0[1], p1[0], p1[1]], dtype=float)

    return FramePose(
        frame_idx=pose.frame_idx,
        timestamp_ms=pose.timestamp_ms,
        keypoints_xy=kxy,
        keypoints_conf=pose.keypoints_conf,
        bbox_xyxy=bbox,
        person_confidence=pose.person_confidence,
        low_confidence=pose.low_confidence,
        usable=pose.usable,
        track_state=getattr(pose, "track_state", "LOST"),
        track_confidence=float(getattr(pose, "track_confidence", 0.0) or 0.0),
        track_id=getattr(pose, "track_id", None),
        target_id=int(getattr(pose, "target_id", 1) or 1),
    )


def _draw_hud(img: np.ndarray, lines: list[str], _jersey: int | None = None) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (12, 12), (280, 28 + 26 * len(lines)), (18, 28, 22), -1)
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    y = 36
    for i, line in enumerate(lines):
        scale = 1.05 if i == 0 else 0.72
        thickness = 2 if i == 0 else 2
        color = (120, 255, 180) if i == 0 else (230, 240, 230)
        if line == "SNAP" or line.startswith("GET-OFF") or "FIRST" in line:
            color = (80, 210, 255)
        if line.startswith("TRACK LOST"):
            color = (40, 40, 255)
        cv2.putText(img, line, (24, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += 26


def _draw_skeleton(img: np.ndarray, pose: FramePose) -> None:
    xy = pose.keypoints_xy
    conf = pose.keypoints_conf
    for a, b in SKELETON:
        if conf[a] < 0.3 or conf[b] < 0.3:
            continue
        pa, pb = xy[a], xy[b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        cv2.line(
            img,
            (int(pa[0]), int(pa[1])),
            (int(pb[0]), int(pb[1])),
            (60, 220, 140),
            3,
            cv2.LINE_AA,
        )
    for i in range(17):
        if conf[i] < 0.3 or np.any(np.isnan(xy[i])):
            continue
        color = (0, 180, 255) if conf[i] < 0.5 else (40, 255, 220)
        cv2.circle(img, (int(xy[i][0]), int(xy[i][1])), 5, color, -1, cv2.LINE_AA)
