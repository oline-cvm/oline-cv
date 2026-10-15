"""Global camera motion between consecutive frames.

Broadcast and sideline film pans and zooms; a tracklet's last observed box
is in the coordinates of the frame it was seen in, so without compensation a
pan moves every player "away" from his own track. A similarity transform
(zoom, rotation, shift) is fitted to sparse optical flow on the background
(player boxes masked out) with RANSAC, after BoT-SORT's global motion
compensation. Not part of Deep HM-SORT / Deep-EIoU.
"""

from __future__ import annotations

import numpy as np

IDENTITY = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def invert(m: np.ndarray | None) -> np.ndarray:
    if m is None:
        return IDENTITY.copy()
    a = np.vstack([m, [0.0, 0.0, 1.0]])
    try:
        return np.linalg.inv(a)[:2]
    except np.linalg.LinAlgError:
        return IDENTITY.copy()


def warp_boxes(boxes, m: np.ndarray | None) -> np.ndarray:
    """Apply a 2x3 similarity transform to ``xyxy`` boxes (centre + scale)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if m is None or len(b) == 0:
        return b.copy()
    a, t = m[:, :2], m[:, 2]
    s = float(np.sqrt(abs(np.linalg.det(a)))) or 1.0
    c = (b[:, :2] + b[:, 2:]) / 2.0
    half = (b[:, 2:] - b[:, :2]) / 2.0 * s
    c2 = c @ a.T + t
    return np.hstack([c2 - half, c2 + half])


class CameraMotionEstimator:
    """``estimate(frame, boxes)`` -> 2x3 transform previous frame -> this frame."""

    def __init__(self, work_width: int = 960, max_corners: int = 800, min_matches: int = 12):
        self.work_width = work_width
        self.max_corners = max_corners
        self.min_matches = min_matches
        self._prev = None
        self._prev_mask = None
        self._scale = 1.0

    def _prep(self, frame: np.ndarray, boxes) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        h, w = frame.shape[:2]
        self._scale = min(1.0, self.work_width / max(w, 1))
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        if self._scale < 1.0:
            g = cv2.resize(g, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_AREA)
        mask = np.full(g.shape, 255, dtype=np.uint8)
        for b in np.asarray(boxes, dtype=float).reshape(-1, 4) * self._scale:
            x0, y0, x1, y1 = (int(v) for v in b)
            mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 0
        return g, mask

    def estimate(self, frame: np.ndarray, boxes) -> np.ndarray:
        import cv2

        g, mask = self._prep(frame, boxes)
        prev, prev_mask = self._prev, self._prev_mask
        self._prev, self._prev_mask = g, mask
        if prev is None or prev.shape != g.shape:
            return IDENTITY.copy()
        pts = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 8, mask=prev_mask)
        if pts is None or len(pts) < self.min_matches:
            return IDENTITY.copy()
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None, winSize=(21, 21), maxLevel=3)
        ok = st.reshape(-1) == 1
        if ok.sum() < self.min_matches:
            return IDENTITY.copy()
        m, inl = cv2.estimateAffinePartial2D(
            pts[ok], nxt[ok], method=cv2.RANSAC, ransacReprojThreshold=2.0
        )
        if m is None or inl is None or int(inl.sum()) < self.min_matches:
            return IDENTITY.copy()
        m = m.astype(np.float64)
        m[:, 2] /= self._scale
        return m
