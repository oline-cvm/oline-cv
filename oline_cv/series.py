"""Shared pose series helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oline_cv.config import AnalysisConfig, L_ANKLE, L_HIP, L_SHOULDER, L_WRIST, R_ANKLE, R_HIP, R_SHOULDER, R_WRIST
from oline_cv.geometry import ankle_mid, estimate_com_xy, hip_mid, shoulder_mid
from oline_cv.pose_tracker import FramePose


@dataclass
class AthleteSeries:
    """Per-frame kinematics for one tracked athlete, snap→end."""

    frames: list[int]
    hip: np.ndarray          # (N, 2) NaN if missing
    ankle_l: np.ndarray
    ankle_r: np.ndarray
    ankle_mid: np.ndarray
    shoulder: np.ndarray
    wrist_l: np.ndarray
    wrist_r: np.ndarray
    com: np.ndarray
    usable: np.ndarray       # bool
    standing_height_px: float
    fps: float
    snap_frame: int

    @property
    def n(self) -> int:
        return len(self.frames)

    def idx_of(self, frame: int) -> int | None:
        try:
            return self.frames.index(frame)
        except ValueError:
            return None


def build_series(
    poses: list[FramePose],
    snap_frame: int,
    end_frame: int,
    standing_height_px: float,
    fps: float,
    config: AnalysisConfig,
) -> AthleteSeries:
    frames: list[int] = []
    hips, al, ar, am, sh, wl, wr, coms, usable = [], [], [], [], [], [], [], [], []

    for pose in poses[snap_frame : end_frame + 1]:
        frames.append(pose.frame_idx)
        conf = pose.keypoints_conf
        xy = pose.keypoints_xy
        ok = pose.usable and not pose.low_confidence
        usable.append(ok)

        def pt(idxs, mid=False):
            pts = []
            for i in idxs:
                if float(conf[i]) >= config.min_keypoint_confidence and not np.any(np.isnan(xy[i])):
                    pts.append(xy[i])
            if not pts:
                return np.array([np.nan, np.nan])
            arr = np.mean(pts, axis=0) if mid or len(pts) > 1 else pts[0]
            return np.asarray(arr, dtype=float)

        if (
            float(conf[L_HIP]) >= config.min_keypoint_confidence
            and float(conf[R_HIP]) >= config.min_keypoint_confidence
        ):
            hips.append(hip_mid(xy).astype(float))
        else:
            hips.append(np.array([np.nan, np.nan]))

        al.append(pt([L_ANKLE]))
        ar.append(pt([R_ANKLE]))
        if not np.any(np.isnan(al[-1])) and not np.any(np.isnan(ar[-1])):
            am.append(0.5 * (al[-1] + ar[-1]))
        else:
            am.append(np.array([np.nan, np.nan]))

        if (
            float(conf[L_SHOULDER]) >= config.min_keypoint_confidence
            and float(conf[R_SHOULDER]) >= config.min_keypoint_confidence
        ):
            sh.append(shoulder_mid(xy).astype(float))
        else:
            sh.append(np.array([np.nan, np.nan]))

        wl.append(pt([L_WRIST]))
        wr.append(pt([R_WRIST]))
        c = estimate_com_xy(xy, conf, config.min_keypoint_confidence)
        coms.append(c if c is not None else np.array([np.nan, np.nan]))

    return AthleteSeries(
        frames=frames,
        hip=np.asarray(hips),
        ankle_l=np.asarray(al),
        ankle_r=np.asarray(ar),
        ankle_mid=np.asarray(am),
        shoulder=np.asarray(sh),
        wrist_l=np.asarray(wl),
        wrist_r=np.asarray(wr),
        com=np.asarray(coms),
        usable=np.asarray(usable, dtype=bool),
        standing_height_px=standing_height_px,
        fps=fps,
        snap_frame=snap_frame,
    )


def nan_vel(series: np.ndarray, fps: float) -> np.ndarray:
    """Frame-to-frame velocity (px/s); first sample NaN."""
    out = np.full_like(series, np.nan, dtype=float)
    if len(series) < 2:
        return out
    d = series[1:] - series[:-1]
    out[1:] = d * fps
    return out


def first_valid(arr: np.ndarray) -> np.ndarray | None:
    for row in arr:
        if not np.any(np.isnan(row)):
            return row
    return None
