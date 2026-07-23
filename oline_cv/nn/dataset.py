"""Dataset builders for OLTechniqueNet."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from oline_cv.nn.features import (
    POSTURE_TO_ID,
    rule_posture_label,
    synthesize_feature_batch,
    window_features,
)


class PoseWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 3 and y.ndim == 1
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def build_training_arrays(
    video_path: str | None = None,
    synth_n: int = 4000,
    window: int = 16,
    seed: int = 42,
    pose_model: str = "yolov8n-pose.pt",
) -> tuple[np.ndarray, np.ndarray]:
    """Mix synthetic pretraining with non-overlapping film windows."""
    Xs, ys = synthesize_feature_batch(synth_n, seed=seed)

    real_X = []
    real_y = []
    if video_path and Path(video_path).exists():
        from oline_cv.body_position import compute_frame_body_metrics
        from oline_cv.config import AnalysisConfig
        from oline_cv.initial_quicks import estimate_rep_standing_height
        from oline_cv.pose_tracker import PoseTracker
        from oline_cv.snap_detection import detect_snap

        cfg = AnalysisConfig(pose_model=pose_model, write_overlay_video=False, track_defender=False)
        tracker = PoseTracker(cfg)
        fps, n, w, h, ol_poses, _, frames = tracker.extract_all(video_path)
        snap = detect_snap(frames, cfg)
        standing = estimate_rep_standing_height(ol_poses, snap.snap_frame, cfg)
        end = min(len(ol_poses) - 1, snap.snap_frame + 120)
        # Non-overlapping windows — no train/val leak from adjacent frames
        for start in range(snap.snap_frame, max(snap.snap_frame + 1, end - window), window):
            mid = start + window // 2
            if mid >= len(ol_poses):
                break
            m = compute_frame_body_metrics(ol_poses[mid], standing, cfg)
            label = rule_posture_label(
                m.knee_flexion_angle_mean, m.torso_angle, m.hip_height, cfg
            )
            if label == "unknown" or m.posture_confidence < 0.06:
                continue
            feats = window_features(ol_poses, standing, start, window)
            real_X.append(feats)
            real_y.append(POSTURE_TO_ID[label])
            # One noise aug per clean window
            noise = np.random.default_rng(start).normal(0, 0.015, size=feats.shape).astype(np.float32)
            real_X.append(np.clip(feats + noise, -2, 2))
            real_y.append(POSTURE_TO_ID[label])

    if real_X:
        Xr = np.stack(real_X, axis=0)
        yr = np.asarray(real_y, dtype=np.int64)
        X = np.concatenate([Xs, Xr], axis=0)
        y = np.concatenate([ys, yr], axis=0)
        print(f"  film windows: {len(real_y)} (non-overlapping)", flush=True)
    else:
        X, y = Xs, ys

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def train_val_split(
    X: np.ndarray, y: np.ndarray, val_frac: float = 0.15, seed: int = 42
) -> tuple[PoseWindowDataset, PoseWindowDataset]:
    """Stratified split by class so val isn't all one posture."""
    n = len(y)
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_frac)) if len(idx) > 4 else max(0, len(idx) // 5)
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())
    if not train_idx:
        # Fallback
        n_val = max(1, int(n * val_frac))
        idx = rng.permutation(n)
        val_idx, train_idx = idx[:n_val].tolist(), idx[n_val:].tolist()
    return PoseWindowDataset(X[train_idx], y[train_idx]), PoseWindowDataset(X[val_idx], y[val_idx])
