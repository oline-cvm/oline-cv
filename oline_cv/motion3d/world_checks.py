"""Measurements that decide whether world-space output is trustworthy.

Kept in the package rather than in a script so the CLI verifier and the review
page compute gravity alignment the same way and can never report different
verdicts for the same npz.
"""

from __future__ import annotations

import numpy as np

# SMPL's root up axis. In a gravity-aligned frame this should stay near world +Y.
SMPL_UP = np.array([0.0, 1.0, 0.0])

# Above this much variation the trajectory is tumbling with the camera rather
# than being held to gravity.
MAX_WORLD_UP_STD_DEG = 12.0


def axis_angle_to_matrix(aa: np.ndarray) -> np.ndarray:
    """Rodrigues, batched. (N,3) axis-angle -> (N,3,3)."""
    aa = np.asarray(aa, dtype=float)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.divide(aa, theta, out=np.zeros_like(aa), where=theta > 1e-8)
    x, y, z = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = np.zeros_like(x)
    K = np.stack(
        [
            np.stack([zero, -z, y], axis=-1),
            np.stack([z, zero, -x], axis=-1),
            np.stack([-y, x, zero], axis=-1),
        ],
        axis=-2,
    )
    t = theta[..., None]
    eye = np.broadcast_to(np.eye(3), K.shape)
    return eye + np.sin(t) * K + (1.0 - np.cos(t)) * (K @ K)


def up_angles_deg(root_aa: np.ndarray) -> np.ndarray:
    """Angle in degrees between the body's up axis and +Y, per frame."""
    up = axis_angle_to_matrix(root_aa) @ SMPL_UP
    up = up / np.linalg.norm(up, axis=-1, keepdims=True)
    return np.degrees(np.arccos(np.clip(up @ SMPL_UP, -1.0, 1.0)))


def world_report(
    pose_world: np.ndarray,
    pose_cam: np.ndarray,
    trans_world: np.ndarray,
    fps: float = 30.0,
) -> dict:
    """Gravity alignment plus trajectory magnitudes for one reconstruction.

    Camera space acts as the control: the same up axis tumbles there as the camera
    pans, so a world frame that is genuinely gravity-aligned should be markedly
    steadier.
    """
    world_up = up_angles_deg(np.asarray(pose_world)[:, :3])
    cam_up = up_angles_deg(np.asarray(pose_cam)[:, :3])
    tw = np.asarray(trans_world, dtype=float)
    horiz = tw[:, [0, 2]]
    step = np.linalg.norm(np.diff(horiz, axis=0), axis=1) if len(tw) > 1 else np.zeros(0)
    fps = float(fps or 30.0)

    return {
        "frames": int(len(tw)),
        "fps": fps,
        "world_up_mean": round(float(world_up.mean()), 3),
        "world_up_std": round(float(world_up.std()), 3),
        "world_up_range": [round(float(world_up.min()), 2), round(float(world_up.max()), 2)],
        "cam_up_mean": round(float(cam_up.mean()), 3),
        "cam_up_std": round(float(cam_up.std()), 3),
        "cam_up_range": [round(float(cam_up.min()), 2), round(float(cam_up.max()), 2)],
        "span_m": [round(float(v), 3) for v in np.ptp(tw, axis=0)],
        "path_length_m": round(float(step.sum()), 3),
        "net_displacement_m": round(float(np.linalg.norm(horiz[-1] - horiz[0])), 3),
        "peak_speed_ms": round(float((step * fps).max()) if len(step) else 0.0, 3),
        "mean_speed_ms": round(float((step * fps).mean()) if len(step) else 0.0, 3),
        "finite": bool(np.all(np.isfinite(tw))),
        "gravity_ok": bool(
            np.all(np.isfinite(tw)) and world_up.std() <= MAX_WORLD_UP_STD_DEG
        ),
        "steadier_than_camera": bool(world_up.std() < cam_up.std()),
    }
