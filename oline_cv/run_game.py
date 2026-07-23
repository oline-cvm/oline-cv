"""Run-blocking point of attack + movement in space (Yeager run §§4–5)."""

from __future__ import annotations

from typing import Any

import numpy as np

from oline_cv.config import AnalysisConfig
from oline_cv.series import AthleteSeries, first_valid


def analyze_point_of_attack(
    ol: AthleteSeries,
    dl: AthleteSeries | None,
    config: AnalysisConfig,
) -> dict[str, Any]:
    if dl is None:
        return {"available": False, "notes": ["defender_not_tracked"]}

    h = max(ol.standing_height_px, 1.0)
    n = min(ol.n, dl.n)
    d0 = first_valid(dl.hip[:n])
    if d0 is None:
        return {"available": False, "notes": ["no_defender_hip"]}

    disp = []
    dirs = []
    for i in range(n):
        if np.any(np.isnan(dl.hip[i])):
            continue
        v = dl.hip[i] - d0
        disp.append(float(np.linalg.norm(v) / h))
        dirs.append(v)

    max_disp = float(np.max(disp)) if disp else None
    mean_disp = float(np.mean(disp)) if disp else None
    direction = None
    if dirs:
        mean_v = np.mean(np.asarray(dirs), axis=0)
        direction = {"x": float(mean_v[0] / h), "y": float(mean_v[1] / h)}

    # Leverage angle: vector OL→DL vs vertical
    lev = []
    for i in range(n):
        if np.any(np.isnan(ol.hip[i])) or np.any(np.isnan(dl.hip[i])):
            continue
        v = dl.hip[i] - ol.hip[i]
        ang = float(np.degrees(np.arctan2(v[0], -v[1])))  # 0 = DL straight "upfield" in image
        lev.append(ang)
    mean_lev = float(np.mean(lev)) if lev else None

    coach_flags = []
    if max_disp is not None and max_disp >= 0.15:
        coach_flags.append("gets_movement")
    if max_disp is not None and max_disp >= 0.28:
        coach_flags.append("drive_block")

    return {
        "available": True,
        "max_defender_displacement": max_disp,
        "mean_defender_displacement": mean_disp,
        "displacement_direction": direction,
        "mean_leverage_angle_deg": mean_lev,
        "coach_flags": coach_flags,
        "notes": [],
    }


def analyze_movement_in_space(ol: AthleteSeries, config: AnalysisConfig) -> dict[str, Any]:
    """Path efficiency before engagement — pull / climb / trap proxy."""
    h = max(ol.standing_height_px, 1.0)
    hip = ol.hip
    valid = hip[~np.any(np.isnan(hip), axis=1)]
    if len(valid) < 3:
        return {"available": False, "notes": ["insufficient_path"]}

    start, end = valid[0], valid[-1]
    straight = float(np.linalg.norm(end - start))
    path = float(np.sum(np.linalg.norm(np.diff(valid, axis=0), axis=1)))
    efficiency = (straight / path) if path > 1e-3 else None
    travel = path / h
    time_s = (ol.n - 1) / ol.fps if ol.fps > 0 else None
    # Arrival angle of final segment
    arrival = None
    if len(valid) >= 2:
        v = valid[-1] - valid[-2]
        arrival = float(np.degrees(np.arctan2(v[0], -v[1])))

    coach_flags = []
    if travel >= 0.8 and efficiency is not None and efficiency >= 0.75:
        coach_flags.append("pull_or_climb")
    if travel >= 1.2:
        coach_flags.append("second_level")

    return {
        "available": True,
        "path_efficiency": efficiency,
        "travel_distance": travel,
        "time_s": time_s,
        "arrival_angle_deg": arrival,
        "coach_flags": coach_flags,
        "notes": [],
    }


def analyze_com_balance(ol: AthleteSeries, config: AnalysisConfig) -> dict[str, Any]:
    h = max(ol.standing_height_px, 1.0)
    com = ol.com
    valid = com[~np.any(np.isnan(com), axis=1)]
    if len(valid) < 4:
        return {"available": False, "jitter": None, "coach_flags": [], "notes": ["no_com"]}
    # Detrended jitter
    t = np.arange(len(valid))
    for axis in range(2):
        coef = np.polyfit(t, valid[:, axis], 1)
        valid[:, axis] = valid[:, axis] - np.polyval(coef, t)
    jitter = float(np.mean(np.linalg.norm(np.diff(valid, axis=0), axis=1)) / h)
    stable = jitter <= config.com_stability_jitter_frac
    return {
        "available": True,
        "com_jitter": jitter,
        "balanced": stable,
        "coach_flags": ["balance"] if stable else ["body_control_issue"],
        "notes": [],
    }
