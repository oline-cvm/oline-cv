"""Expansion IoU (Deep-EIoU, Huang et al. 2024).

Both boxes are grown about their own centres before IoU is taken, so a fast
player whose new box no longer overlaps his last observed box can still be
associated. With expansion scale ``E`` a box of size ``w x h`` becomes
``w(2E+1) x h(2E+1)``.
"""

from __future__ import annotations

import numpy as np


def _as_boxes(boxes) -> np.ndarray:
    a = np.asarray(boxes, dtype=np.float64)
    if a.size == 0:
        return np.zeros((0, 4), dtype=np.float64)
    return a.reshape(-1, 4)


def expand_boxes(
    boxes,
    expansion: float,
    image_wh: tuple[float, float] | None = None,
) -> np.ndarray:
    """Grow ``xyxy`` boxes by ``expansion`` about their centres.

    When ``image_wh`` is given the result is clipped to the image. Clipping
    happens after expansion, so the centre of a box away from the border is
    unchanged.
    """
    if expansion < 0:
        raise ValueError(f"expansion must be >= 0, got {expansion}")
    b = _as_boxes(boxes)
    cx = (b[:, 0] + b[:, 2]) * 0.5
    cy = (b[:, 1] + b[:, 3]) * 0.5
    half_w = np.abs(b[:, 2] - b[:, 0]) * 0.5 * (2.0 * expansion + 1.0)
    half_h = np.abs(b[:, 3] - b[:, 1]) * 0.5 * (2.0 * expansion + 1.0)
    out = np.stack([cx - half_w, cy - half_h, cx + half_w, cy + half_h], axis=1)
    if image_wh is not None:
        w, h = float(image_wh[0]), float(image_wh[1])
        out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0.0, w)
        out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0.0, h)
    return out


def iou_matrix(a, b) -> np.ndarray:
    """Pairwise IoU between ``xyxy`` box sets; degenerate boxes give 0."""
    a = _as_boxes(a)
    b = _as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float64)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return np.nan_to_num(iou, nan=0.0, posinf=0.0, neginf=0.0)


def expansion_iou(
    track_boxes,
    det_boxes,
    expansion: float,
    image_wh: tuple[float, float] | None = None,
) -> np.ndarray:
    """IoU between expanded track boxes and expanded detection boxes."""
    return iou_matrix(
        expand_boxes(track_boxes, expansion, image_wh),
        expand_boxes(det_boxes, expansion, image_wh),
    )


def expansion_schedule(initial: float, step: float, iterations: int) -> list[float]:
    """Expansion scales for the iterative first-stage passes."""
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    return [float(initial + step * i) for i in range(int(iterations))]
