"""Feature extraction for the OL posture / technique neural net."""

from __future__ import annotations

import numpy as np

from oline_cv.config import (
    AnalysisConfig,
    L_ANKLE,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
)
from oline_cv.geometry import hip_mid, joint_angle_deg, shoulder_mid, vector_angle_from_vertical_deg
from oline_cv.pose_tracker import FramePose

FEATURE_DIM = 24
POSTURE_CLASSES = ("knee_bender", "waist_bender", "balanced", "unknown")
POSTURE_TO_ID = {c: i for i, c in enumerate(POSTURE_CLASSES)}


def frame_pose_features(pose: FramePose, standing_height_px: float, min_conf: float = 0.35) -> np.ndarray:
    """Normalized geometric features for one frame. Missing → 0 with confidence channel."""
    xy = pose.keypoints_xy
    conf = pose.keypoints_conf
    h = max(float(standing_height_px), 1.0)
    feats = np.zeros(FEATURE_DIM, dtype=np.float32)

    def pt(i):
        if float(conf[i]) < min_conf or np.any(np.isnan(xy[i])):
            return None
        return xy[i]

    hips = None
    if pt(L_HIP) is not None and pt(R_HIP) is not None:
        hips = hip_mid(xy)
    shoulders = None
    if pt(L_SHOULDER) is not None and pt(R_SHOULDER) is not None:
        shoulders = shoulder_mid(xy)

    for idx, (hip, knee, ankle) in enumerate(
        ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE))
    ):
        if pt(hip) is not None and pt(knee) is not None and pt(ankle) is not None:
            ang = joint_angle_deg(xy[hip], xy[knee], xy[ankle])
            if ang is not None:
                feats[idx] = ang / 180.0

    if hips is not None and shoulders is not None:
        torso = vector_angle_from_vertical_deg(shoulders - hips)
        if torso is not None:
            feats[2] = min(torso, 90.0) / 90.0

    ankles = []
    for a in (L_ANKLE, R_ANKLE):
        if pt(a) is not None:
            ankles.append(xy[a])
    if ankles:
        feet_y = float(np.mean([p[1] for p in ankles]))
        if hips is not None:
            feats[3] = float((feet_y - hips[1]) / h)
        if shoulders is not None:
            feats[4] = float((feet_y - shoulders[1]) / h)

    if pt(L_ANKLE) is not None and pt(R_ANKLE) is not None:
        feats[5] = float(np.linalg.norm(xy[L_ANKLE] - xy[R_ANKLE]) / h)

    if pt(L_SHOULDER) is not None and pt(R_SHOULDER) is not None:
        feats[6] = float(np.linalg.norm(xy[L_SHOULDER] - xy[R_SHOULDER]) / h)

    for i, w in enumerate((L_WRIST, R_WRIST)):
        if pt(w) is not None and ankles:
            feats[7 + i] = float((feet_y - xy[w][1]) / h)

    feats[9] = float(np.mean(conf))

    off = 10
    for hip, knee, ankle in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
        if pt(hip) is not None and pt(knee) is not None:
            v = (xy[knee] - xy[hip]) / h
            feats[off : off + 2] = v
        off += 2
        if pt(knee) is not None and pt(ankle) is not None:
            v = (xy[ankle] - xy[knee]) / h
            feats[off : off + 2] = v
        off += 2

    if hips is not None and shoulders is not None:
        v = (shoulders - hips) / h
        feats[18:20] = v

    feats[20] = 1.0 if pose.usable else 0.0
    feats[21] = 1.0 if pose.low_confidence else 0.0
    feats[22] = float(pose.person_confidence)
    feats[23] = float(len(ankles) / 2.0)
    return feats


def window_features(
    poses: list[FramePose],
    standing_height_px: float,
    start: int,
    length: int,
    min_conf: float = 0.35,
) -> np.ndarray:
    """(T, F) feature matrix; pads/truncates to `length`."""
    out = np.zeros((length, FEATURE_DIM), dtype=np.float32)
    for t in range(length):
        i = start + t
        if 0 <= i < len(poses):
            out[t] = frame_pose_features(poses[i], standing_height_px, min_conf)
    return out


def rule_posture_label(
    knee: float | None,
    torso: float | None,
    hip_h: float | None,
    config: AnalysisConfig | None = None,
) -> str:
    """Weak label via the same score-based classifier as body_position."""
    from oline_cv.body_position import classify_posture

    cfg = config or AnalysisConfig()
    label, conf = classify_posture(knee, torso, hip_h, cfg)
    if label == "unknown" or conf < 0.05:
        return "unknown"
    return label


def synthesize_feature_batch(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic (N, T, F) windows with full-ish feature geometry (not just 4 dims)."""
    rng = np.random.default_rng(seed)
    T = 16
    X = np.zeros((n, T, FEATURE_DIM), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    for i in range(n):
        cls = int(rng.integers(0, 3))
        y[i] = cls
        for t in range(T):
            f = np.zeros(FEATURE_DIM, dtype=np.float32)
            noise = rng.normal(0, 0.025, size=FEATURE_DIM).astype(np.float32)
            if cls == 0:  # knee_bender
                f[0] = f[1] = rng.uniform(0.55, 0.72)
                f[2] = rng.uniform(0.05, 0.22)
                f[3] = rng.uniform(0.40, 0.58)
                f[18] = rng.uniform(-0.05, 0.05)
                f[19] = rng.uniform(-0.45, -0.30)
            elif cls == 1:  # waist_bender
                f[0] = f[1] = rng.uniform(0.82, 0.96)
                f[2] = rng.uniform(0.35, 0.80)
                f[3] = rng.uniform(0.55, 0.78)
                f[18] = rng.uniform(-0.15, 0.15)
                f[19] = rng.uniform(-0.35, -0.15)
            else:  # balanced
                f[0] = f[1] = rng.uniform(0.72, 0.86)
                f[2] = rng.uniform(0.12, 0.35)
                f[3] = rng.uniform(0.52, 0.70)
                f[18] = rng.uniform(-0.08, 0.08)
                f[19] = rng.uniform(-0.40, -0.25)
            f[4] = f[3] + rng.uniform(0.15, 0.28)
            f[5] = rng.uniform(0.20, 0.40)
            f[6] = rng.uniform(0.25, 0.40)
            f[7] = f[8] = rng.uniform(0.35, 0.65)
            f[9] = rng.uniform(0.55, 0.95)
            # limb stubs
            f[10:18] = rng.normal(0, 0.05, size=8)
            f[20] = 1.0
            f[22] = rng.uniform(0.6, 0.95)
            f[23] = 1.0
            X[i, t] = np.clip(f + noise, -1.5, 1.5)
    return X, y
