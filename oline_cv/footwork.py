"""Pass-pro / run footwork metrics (Yeager §2)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import find_peaks

from oline_cv.config import AnalysisConfig
from oline_cv.series import AthleteSeries, first_valid, nan_vel


def analyze_footwork(series: AthleteSeries, config: AnalysisConfig, mode: str = "pass") -> dict[str, Any]:
    h = max(series.standing_height_px, 1.0)
    fps = series.fps if series.fps > 0 else 30.0
    n = series.n
    if n < 3:
        return {"available": False, "notes": ["too_few_frames"]}

    # Local frame axes: +x right on screen, +y down. Depth ≈ retreat from snap hip.
    hip0 = first_valid(series.hip)
    if hip0 is None:
        return {"available": False, "notes": ["no_hip_baseline"]}

    hip = series.hip.copy()
    # Lateral = x; depth for pass set = increase in |Δx| and/or rearward. On elevated
    # endzone/sideline, retreat often mixes x and y. Use principal direction of hip travel.
    valid = ~np.any(np.isnan(hip), axis=1)
    if valid.sum() < 3:
        return {"available": False, "notes": ["insufficient_hip_track"]}

    disp = hip - hip0
    # Set depth: max displacement along dominant travel axis in first 2/3 of set
    mid = max(3, int(n * 0.67))
    early = disp[:mid]
    early_v = early[~np.any(np.isnan(early), axis=1)]
    if len(early_v) == 0:
        return {"available": False, "notes": ["no_early_displacement"]}

    # PCA-ish: direction of max variance
    cov = np.cov(early_v.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, int(np.argmax(eigvals))]
    depth_proj = early_v @ principal
    # Orient so positive = away from snap (increasing |proj| median sign)
    if np.median(depth_proj) < 0:
        principal = -principal
        depth_proj = -depth_proj

    full_valid = disp[~np.any(np.isnan(disp), axis=1)]
    depth_all = full_valid @ principal
    lateral_axis = np.array([-principal[1], principal[0]])
    lat_all = full_valid @ lateral_axis

    # Set depth / width: use first ~1.0s only — late tracking drift inflates overset.
    early_n = max(5, min(n, int(fps * 1.0)))
    early_disp = disp[:early_n]
    early_valid = early_disp[~np.any(np.isnan(early_disp), axis=1)]
    if len(early_valid):
        depth_early = early_valid @ principal
        lat_early = early_valid @ lateral_axis
        set_depth = float(np.nanmax(depth_early) / h)
        set_width = float(np.nanmax(np.abs(lat_early)) / h)
    else:
        set_depth = float(np.nanmax(depth_all) / h) if len(depth_all) else None
        set_width = float(np.nanmax(np.abs(lat_all)) / h) if len(lat_all) else None

    # Lateral velocity (px/s → height/s)
    hip_lat = disp @ lateral_axis
    v_lat = nan_vel(hip_lat, fps)
    mean_lat_speed = float(np.nanmean(np.abs(v_lat)) / h) if np.any(~np.isnan(v_lat)) else None
    peak_lat_speed = float(np.nanmax(np.abs(v_lat)) / h) if np.any(~np.isnan(v_lat)) else None

    # Base width: ankle separation / height
    base = []
    for i in range(n):
        if np.any(np.isnan(series.ankle_l[i])) or np.any(np.isnan(series.ankle_r[i])):
            continue
        base.append(float(np.linalg.norm(series.ankle_l[i] - series.ankle_r[i]) / h))
    mean_base = float(np.mean(base)) if base else None
    min_base = float(np.min(base)) if base else None
    wide_base = mean_base is not None and mean_base >= config.base_width_wide_frac
    narrow_base = mean_base is not None and mean_base <= config.base_width_narrow_frac

    # Steps: peaks on ankle-mid speed
    am = series.ankle_mid
    am_speed = np.full(n, np.nan)
    for i in range(1, n):
        if np.any(np.isnan(am[i])) or np.any(np.isnan(am[i - 1])):
            continue
        am_speed[i] = float(np.linalg.norm(am[i] - am[i - 1]) * fps)
    speed_fill = np.nan_to_num(am_speed, nan=0.0)
    prom = config.step_peak_prominence_frac * h * fps
    peaks, _ = find_peaks(
        speed_fill,
        prominence=max(prom, 1.0),
        distance=config.step_min_separation_frames,
    )
    step_count = int(len(peaks))
    duration_s = (n - 1) / fps if n > 1 else 0.0
    step_cadence_hz = (step_count / duration_s) if duration_s > 0 else None

    step_lengths = []
    for a, b in zip(peaks[:-1], peaks[1:]):
        if np.any(np.isnan(am[a])) or np.any(np.isnan(am[b])):
            continue
        step_lengths.append(float(np.linalg.norm(am[b] - am[a]) / h))
    mean_step_len = float(np.mean(step_lengths)) if step_lengths else None

    # Second-step gain (run emphasis): displacement of 2nd step along principal
    second_step_gain = None
    if len(peaks) >= 2:
        i0, i1 = int(peaks[0]), int(peaks[1])
        if not np.any(np.isnan(am[i0])) and not np.any(np.isnan(am[i1])):
            second_step_gain = float(np.dot(am[i1] - am[i0], principal) / h)

    # Crossover: left/right ankle x order flips sustained
    crosses = 0
    streak = 0
    prev_sign = 0
    for i in range(n):
        if np.any(np.isnan(series.ankle_l[i])) or np.any(np.isnan(series.ankle_r[i])):
            streak = 0
            continue
        # In image coords, "crossed" if left ankle is to the right of right ankle
        sign = 1 if series.ankle_l[i][0] > series.ankle_r[i][0] + 2 else -1
        if prev_sign and sign != prev_sign and sign > 0:
            streak += 1
            if streak >= config.crossover_min_frames:
                crosses += 1
                streak = 0
        else:
            streak = 0 if sign < 0 else streak
        prev_sign = sign

    overset = set_width is not None and set_width >= config.overset_width_frac

    # Feet close to ground proxy: low ankle vertical variance relative to hip sink
    ankle_y = am[:, 1]
    feet_ground_score = None
    if np.any(~np.isnan(ankle_y)):
        feet_ground_score = float(1.0 - min(1.0, np.nanstd(ankle_y) / (0.08 * h)))

    coach_flags = []
    if wide_base:
        coach_flags.append("wide_base")
    if narrow_base:
        coach_flags.append("narrow_base")
    notes: list[str] = []
    if overset:
        coach_flags.append("oversets")
    if crosses > 0:
        coach_flags.append("crossover_detected")
    cad_min = getattr(config, "foot_quickness_cadence_min_hz", 2.8)
    cad_max = getattr(config, "foot_quickness_cadence_max_hz", 4.8)
    if step_cadence_hz is not None and cad_min <= step_cadence_hz <= cad_max:
        coach_flags.append("foot_quickness")
    elif step_cadence_hz is not None and step_cadence_hz > cad_max:
        notes.append(f"cadence_{step_cadence_hz:.1f}hz_implausible_step_noise")

    return {
        "available": True,
        "mode": mode,
        "step_count": step_count,
        "step_cadence_hz": step_cadence_hz,
        "mean_step_length": mean_step_len,
        "second_step_gain": second_step_gain,
        "set_depth": set_depth,
        "set_width": set_width,
        "mean_lateral_velocity": mean_lat_speed,
        "peak_lateral_velocity": peak_lat_speed,
        "mean_base_width": mean_base,
        "min_base_width": min_base,
        "crossover_events": crosses,
        "overset": overset,
        "feet_close_to_ground_score": feet_ground_score,
        "coach_flags": coach_flags,
        "notes": notes,
    }
