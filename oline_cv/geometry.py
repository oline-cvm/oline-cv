"""Geometry helpers for keypoints, angles, and normalized heights."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from oline_cv.config import (
    L_ANKLE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
)


def midpoint(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
) -> np.ndarray:
    return (np.asarray(a, dtype=float) + np.asarray(b, dtype=float)) / 2.0


def joint_angle_deg(
    proximal: Sequence[float],
    joint: Sequence[float],
    distal: Sequence[float],
) -> float | None:
    """Interior angle at `joint` formed by proximal–joint–distal, in degrees."""
    p = np.asarray(proximal, dtype=float)
    j = np.asarray(joint, dtype=float)
    d = np.asarray(distal, dtype=float)
    v1 = p - j
    v2 = d - j
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def vector_angle_from_vertical_deg(vec: Sequence[float]) -> float | None:
    """Angle between vector and upward vertical (0, -1) in image coords.

    Image y increases downward, so "up" is (0, -1). Returns 0 when the vector
    points straight up (upright torso).
    """
    v = np.asarray(vec, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-6:
        return None
    up = np.array([0.0, -1.0])
    cos_a = float(np.clip(np.dot(v / n, up), -1.0, 1.0))
    return math.degrees(math.acos(cos_a))


def knee_flexion(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    side: str,
    min_conf: float,
) -> tuple[float | None, bool]:
    """Return (angle_deg, low_confidence_flag) for left or right knee."""
    if side == "left":
        idxs = (L_HIP, L_KNEE, L_ANKLE)
    else:
        idxs = (R_HIP, R_KNEE, R_ANKLE)
    low = any(float(kpts_conf[i]) < min_conf for i in idxs)
    if low:
        return None, True
    angle = joint_angle_deg(kpts_xy[idxs[0]], kpts_xy[idxs[1]], kpts_xy[idxs[2]])
    return angle, False


def shoulder_mid(kpts_xy: np.ndarray) -> np.ndarray:
    return midpoint(kpts_xy[L_SHOULDER], kpts_xy[R_SHOULDER])


def hip_mid(kpts_xy: np.ndarray) -> np.ndarray:
    return midpoint(kpts_xy[L_HIP], kpts_xy[R_HIP])


def ankle_mid(kpts_xy: np.ndarray) -> np.ndarray:
    return midpoint(kpts_xy[L_ANKLE], kpts_xy[R_ANKLE])


def estimate_standing_height_px(
    kpts_xy: np.ndarray,
    kpts_conf: np.ndarray,
    min_conf: float,
    nose_to_hip_factor: float,
) -> float | None:
    """Pixel standing height from ankle midpoint to shoulder midpoint.

    Falls back to nose→hip * factor when ankles are unreliable.
    """
    from oline_cv.config import L_ANKLE, NOSE, R_ANKLE  # local clarity

    hips_ok = float(kpts_conf[L_HIP]) >= min_conf and float(kpts_conf[R_HIP]) >= min_conf
    shoulders_ok = (
        float(kpts_conf[L_SHOULDER]) >= min_conf and float(kpts_conf[R_SHOULDER]) >= min_conf
    )
    ankles_ok = float(kpts_conf[L_ANKLE]) >= min_conf and float(kpts_conf[R_ANKLE]) >= min_conf

    if hips_ok and shoulders_ok and ankles_ok:
        return float(np.linalg.norm(shoulder_mid(kpts_xy) - ankle_mid(kpts_xy)))

    if hips_ok and float(kpts_conf[NOSE]) >= min_conf:
        return float(np.linalg.norm(kpts_xy[NOSE] - hip_mid(kpts_xy)) * nose_to_hip_factor)

    return None


def estimate_com_xy(kpts_xy: np.ndarray, kpts_conf: np.ndarray, min_conf: float) -> np.ndarray | None:
    """Rough COM as confidence-weighted average of torso + limb keypoints.

    Uses a simple segmental approximation: shoulders, hips, knees, ankles.
    """
    idxs = [
        L_SHOULDER,
        R_SHOULDER,
        L_HIP,
        R_HIP,
        L_KNEE,
        R_KNEE,
        L_ANKLE,
        R_ANKLE,
    ]
    # Relative masses (normalized later): torso heavier than limbs.
    weights = np.array([0.15, 0.15, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05], dtype=float)
    pts = []
    ws = []
    for i, w in zip(idxs, weights):
        if float(kpts_conf[i]) >= min_conf:
            pts.append(kpts_xy[i])
            ws.append(w)
    if len(pts) < 4:
        return None
    pts_arr = np.asarray(pts, dtype=float)
    ws_arr = np.asarray(ws, dtype=float)
    ws_arr /= ws_arr.sum()
    return (pts_arr * ws_arr[:, None]).sum(axis=0)


def normalized_height_from_feet(
    point_y: float,
    ankle_y: float,
    standing_height_px: float,
) -> float | None:
    """Height of a point above the feet, divided by standing height.

    Image y grows downward, so height_above_feet = ankle_y - point_y.
    Result ≈ 1.0 at shoulder level for a standing player, lower when crouched.
    """
    if standing_height_px < 1e-3:
        return None
    return float((ankle_y - point_y) / standing_height_px)
