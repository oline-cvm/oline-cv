"""Module 1a — Snap event detection from ball / center hand motion."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from oline_cv.config import AnalysisConfig


@dataclass
class SnapResult:
    snap_frame: int
    method: str
    motion_score: float
    confidence: str  # high | medium | low


def _frame_energies(
    frames: list[np.ndarray],
    config: AnalysisConfig,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = frames[0].shape[:2]
    x0, y0, x1, y1 = config.snap_roi
    rx0, ry0 = int(x0 * w), int(y0 * h)
    rx1, ry1 = int(x1 * w), int(y1 * h)

    roi_energies: list[float] = []
    full_energies: list[float] = []
    prev_roi = None
    prev_full = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(gray[ry0:ry1, rx0:rx1], (5, 5), 0)
        full = cv2.GaussianBlur(cv2.resize(gray, (w // 4, h // 4)), (5, 5), 0)
        if prev_roi is None:
            roi_energies.append(0.0)
            full_energies.append(0.0)
            prev_roi, prev_full = roi, full
            continue
        roi_energies.append(float(np.mean(cv2.absdiff(roi, prev_roi))))
        full_energies.append(float(np.mean(cv2.absdiff(full, prev_full))))
        prev_roi, prev_full = roi, full
    return np.asarray(roi_energies, dtype=float), np.asarray(full_energies, dtype=float)


def _is_graphic(full_e: float, roi_e: float, config: AnalysisConfig) -> bool:
    if full_e > config.snap_max_full_frame_energy:
        return True
    if roi_e > 1e-3 and (full_e / roi_e) > config.snap_max_full_to_roi_ratio:
        return True
    return False


def detect_snap(
    frames: list[np.ndarray],
    config: AnalysisConfig,
) -> SnapResult:
    """Detect snap as the onset of sustained LOS motion (play start).

    Strategy:
      1) Build ROI (ball/center) and full-frame motion energies.
      2) Mask broadcast graphic flashes / hard cuts.
      3) Find the earliest frame where ROI energy stays elevated for a
         sustained window — that onset is the snap.
    """
    if config.snap_frame_override is not None:
        return SnapResult(
            snap_frame=int(config.snap_frame_override),
            method="manual_override",
            motion_score=0.0,
            confidence="high",
        )

    if len(frames) < config.snap_baseline_frames + config.snap_sustained_frames + 2:
        return SnapResult(0, "fallback_too_short", 0.0, "low")

    energy, full_e = _frame_energies(frames, config)
    n = len(energy)
    margin = int(n * config.snap_search_margin_frac)
    search_start = max(config.snap_baseline_frames, margin)
    search_end = max(search_start + 1, n - max(margin, config.snap_sustained_frames))

    graphic = np.array(
        [_is_graphic(float(full_e[i]), float(energy[i]), config) for i in range(n)],
        dtype=bool,
    )

    # Quiet-period threshold from non-graphic baseline near start of search.
    quiet_vals = [
        float(energy[i])
        for i in range(search_start, min(search_start + 40, search_end))
        if not graphic[i]
    ]
    if len(quiet_vals) < 5:
        quiet_vals = [float(v) for v in energy[search_start:search_end] if v > 0]
    quiet_med = float(np.median(quiet_vals)) if quiet_vals else 1.0
    quiet_p75 = float(np.percentile(quiet_vals, 75)) if quiet_vals else quiet_med
    quiet_mad = float(np.median(np.abs(np.asarray(quiet_vals) - quiet_med))) + 1e-6
    # Elevated = clearly above pre-snap fidget level; require sustained play motion.
    # Use p75-based threshold so a few noisy baseline frames don't inflate elevate.
    elevate = max(quiet_p75 * 2.0, quiet_med + 4.0 * quiet_mad, 8.0)
    sustain = max(config.snap_sustained_frames, 5)

    for i in range(search_start, search_end - sustain):
        if graphic[i]:
            continue
        window = energy[i : i + sustain]
        if np.any(graphic[i : i + sustain]):
            continue
        if not np.all(window >= elevate):
            continue
        # Require the window mean to stay high (play underway), not a brief spike.
        if float(np.mean(window)) < elevate:
            continue
        # Prefer onset out of quieter frames (not mid-play).
        pre = energy[max(search_start, i - 5) : i]
        pre = pre[~graphic[max(search_start, i - 5) : i]]
        pre_mean = float(np.mean(pre)) if len(pre) else quiet_med
        # Only skip if we're already deep into play-level motion.
        if pre_mean >= max(elevate * 1.8, elevate + 5.0):
            continue
        z = (float(window[0]) - quiet_med) / (quiet_mad + 1e-6)
        return SnapResult(
            snap_frame=int(i),
            method="sustained_roi_onset",
            motion_score=float(z),
            confidence="high" if z >= 5 else "medium",
        )

    # Fallback: first frame exceeding elevate for sustain frames (ignore rising check)
    for i in range(search_start, search_end - sustain):
        if graphic[i] or np.any(graphic[i : i + sustain]):
            continue
        if np.all(energy[i : i + sustain] >= elevate):
            return SnapResult(int(i), "fallback_sustained", float(energy[i]), "medium")

    return SnapResult(int(search_start), "fallback_default", 0.0, "low")
