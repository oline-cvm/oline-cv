"""Neural posture inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from oline_cv.nn.features import POSTURE_CLASSES, window_features
from oline_cv.nn.checkpoint import DEFAULT_WEIGHTS, load_model
from oline_cv.nn.model import OLTechniqueNet
from oline_cv.pose_tracker import FramePose

_MODEL: OLTechniqueNet | None = None
_MODEL_PATH: str | None = None


def get_model(weights: str | Path | None = None) -> OLTechniqueNet | None:
    global _MODEL, _MODEL_PATH
    path = str(Path(weights) if weights else DEFAULT_WEIGHTS)
    if not Path(path).exists():
        return None
    if _MODEL is None or _MODEL_PATH != path:
        _MODEL = load_model(path)
        _MODEL_PATH = path
    return _MODEL


def classify_window(
    poses: list[FramePose],
    standing_height_px: float,
    center_frame: int,
    window: int = 16,
    weights: str | Path | None = None,
) -> tuple[str, float] | None:
    model = get_model(weights)
    if model is None:
        return None
    start = max(0, center_frame - window // 2)
    feats = window_features(poses, standing_height_px, start, window)
    x = torch.from_numpy(feats[None, ...]).to(next(model.parameters()).device)
    with torch.no_grad():
        prob = model.predict_proba(x)[0].cpu().numpy()
    i = int(np.argmax(prob))
    return POSTURE_CLASSES[i], float(prob[i])
