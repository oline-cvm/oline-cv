"""Jersey number reading via EasyOCR on upper-torso crops."""

from __future__ import annotations

import re
from functools import lru_cache

import cv2
import numpy as np

_READER = None


def _reader():
    global _READER
    if _READER is None:
        import easyocr

        _READER = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _READER


def jersey_roi_from_bbox(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
    """Upper half of the person box — where back/chest numbers usually sit."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = [int(float(v)) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    bw, bh = x1 - x0, y1 - y0
    if bw < 20 or bh < 30:
        return None
    # Prefer mid-upper torso (skip headband / helmet)
    ry0 = y0 + int(0.08 * bh)
    ry1 = y0 + int(0.55 * bh)
    rx0 = x0 + int(0.08 * bw)
    rx1 = x1 - int(0.08 * bw)
    if rx1 - rx0 < 12 or ry1 - ry0 < 12:
        return None
    return frame[ry0:ry1, rx0:rx1].copy()


def _prep(crop: np.ndarray) -> list[np.ndarray]:
    """Upscaled variants for OCR."""
    out = []
    scale = 4.0 if min(crop.shape[:2]) < 80 else 3.0
    big = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    out.append(big)
    lab = cv2.cvtColor(big, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    out.append(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR))
    return out


def read_jersey_number(crop: np.ndarray | None) -> tuple[int | None, float]:
    """Return (jersey, confidence) or (None, 0)."""
    if crop is None or crop.size == 0:
        return None, 0.0
    reader = _reader()
    best_num, best_conf = None, 0.0
    for img in _prep(crop):
        try:
            results = reader.readtext(img, allowlist="0123456789", detail=1, paragraph=False)
        except Exception:
            continue
        for _bbox, text, conf in results:
            text = re.sub(r"\D", "", str(text))
            if not text or len(text) > 2:
                continue
            num = int(text)
            if not (0 <= num <= 99):
                continue
            c = float(conf)
            if c > best_conf:
                best_num, best_conf = num, c
    return best_num, best_conf


def find_jersey_match(
    frame: np.ndarray,
    boxes_xyxy: np.ndarray,
    target: int,
    min_conf: float = 0.35,
) -> tuple[int, float] | None:
    """Return (box_index, conf) for best match to target jersey."""
    hits: list[tuple[int, float]] = []
    for i, box in enumerate(boxes_xyxy):
        crop = jersey_roi_from_bbox(frame, box)
        num, conf = read_jersey_number(crop)
        if num == int(target) and conf >= min_conf:
            hits.append((i, conf))
    if not hits:
        return None
    hits.sort(key=lambda t: t[1], reverse=True)
    return hits[0]
