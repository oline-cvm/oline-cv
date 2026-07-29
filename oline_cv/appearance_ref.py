"""Frozen multi-frame appearance reference — classical filters only (no NN training).

Builds a permanent template from several clean pre-snap torso crops.
The frozen embedding is never overwritten. A separate *recent* embedding may
be refreshed only under strict gates (high conf, not occluded, similar to frozen).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def crop_torso(frame: np.ndarray, bbox: np.ndarray, size: int = 96) -> np.ndarray | None:
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = [int(float(v)) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    bw, bh = x1 - x0, y1 - y0
    if bw < 12 or bh < 20:
        return None
    ry0 = y0 + int(0.08 * bh)
    ry1 = y0 + int(0.58 * bh)
    rx0 = x0 + int(0.10 * bw)
    rx1 = x1 - int(0.10 * bw)
    if rx1 - rx0 < 8 or ry1 - ry0 < 8:
        return None
    crop = frame[ry0:ry1, rx0:rx1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def appearance_vector(bgr: np.ndarray) -> np.ndarray:
    """HSV hist + edge hist + mean Lab color — L2-normalized."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    parts: list[np.ndarray] = []
    for ch, bins, rng in ((0, 18, [0, 180]), (1, 16, [0, 256]), (2, 16, [0, 256])):
        hist = cv2.calcHist([hsv], [ch], None, [bins], rng)
        hist = cv2.normalize(hist, hist).flatten()
        parts.append(hist)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    eh = cv2.calcHist([edges], [0], None, [8], [0, 256])
    eh = cv2.normalize(eh, eh).flatten()
    parts.append(eh)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    parts.append((lab.mean(axis=(0, 1)) / 255.0).astype(np.float32))
    sig = np.concatenate(parts).astype(np.float32)
    n = float(np.linalg.norm(sig) + 1e-6)
    return sig / n


def jersey_color_vector(bgr: np.ndarray) -> np.ndarray:
    """Dominant jersey hue/sat signature from central torso pixels."""
    h, w = bgr.shape[:2]
    core = bgr[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    if core.size == 0:
        core = bgr
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [12, 8], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    n = float(np.linalg.norm(hist) + 1e-6)
    return hist / n


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


@dataclass
class FrozenAppearanceRef:
    """Permanent target appearance. Frozen vector is immutable after build."""

    frozen: np.ndarray
    frozen_jersey: np.ndarray
    reference_crops: list[np.ndarray] = field(default_factory=list)
    recent: np.ndarray | None = None
    self_similarity: float = 1.0  # min pairwise among clean refs (calib floor)
    formation_xy: np.ndarray | None = None  # lock-time center in image coords
    lock_diag: float = 200.0
    target_id: int = 1  # permanent logical ID — never transferred

    def appearance_sim(self, crop: np.ndarray) -> float:
        return cosine(self.frozen, appearance_vector(crop))

    def jersey_sim(self, crop: np.ndarray) -> float:
        return cosine(self.frozen_jersey, jersey_color_vector(crop))

    def recent_sim(self, crop: np.ndarray) -> float:
        if self.recent is None:
            return self.appearance_sim(crop)
        return cosine(self.recent, appearance_vector(crop))

    def maybe_update_recent(
        self,
        crop: np.ndarray,
        *,
        det_conf: float,
        occluded: bool,
        min_conf: float,
        min_frozen_sim: float,
    ) -> bool:
        """Update recent only — never touches frozen."""
        if occluded or det_conf < min_conf:
            return False
        sim = self.appearance_sim(crop)
        if sim < min_frozen_sim:
            return False
        self.recent = appearance_vector(crop)
        return True


def build_frozen_appearance(
    frames: list[np.ndarray],
    bboxes: list[np.ndarray],
    *,
    target_id: int = 1,
    formation_xy: np.ndarray | None = None,
    lock_diag: float = 200.0,
) -> FrozenAppearanceRef | None:
    """Build frozen ref from several clean crops. Does not train a network."""
    crops: list[np.ndarray] = []
    vecs: list[np.ndarray] = []
    jerseys: list[np.ndarray] = []
    for frame, box in zip(frames, bboxes):
        if frame is None or box is None:
            continue
        crop = crop_torso(frame, box)
        if crop is None:
            continue
        crops.append(crop.copy())
        vecs.append(appearance_vector(crop))
        jerseys.append(jersey_color_vector(crop))
    if not vecs:
        return None

    frozen = np.mean(np.stack(vecs), axis=0)
    frozen = frozen / float(np.linalg.norm(frozen) + 1e-6)
    frozen_j = np.mean(np.stack(jerseys), axis=0)
    frozen_j = frozen_j / float(np.linalg.norm(frozen_j) + 1e-6)

    # Self-similarity among clean refs — used later to recommend thresholds
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(cosine(vecs[i], vecs[j]))
    self_sim = float(min(sims)) if sims else 1.0

    return FrozenAppearanceRef(
        frozen=frozen,
        frozen_jersey=frozen_j,
        reference_crops=crops,
        recent=frozen.copy(),
        self_similarity=self_sim,
        formation_xy=None if formation_xy is None else formation_xy.astype(float),
        lock_diag=float(lock_diag),
        target_id=int(target_id),
    )
