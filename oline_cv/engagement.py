"""Mirror / redirect + anchor + hands + sustain (Yeager pass-pro §§3–4,6–7)."""

from __future__ import annotations

from typing import Any

import numpy as np

from oline_cv.config import AnalysisConfig
from oline_cv.series import AthleteSeries, nan_vel


def _pair_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full(len(a), np.nan)
    for i in range(len(a)):
        if np.any(np.isnan(a[i])) or np.any(np.isnan(b[i])):
            continue
        out[i] = float(np.linalg.norm(a[i] - b[i]))
    return out


def analyze_mirror_redirect(
    ol: AthleteSeries,
    dl: AthleteSeries | None,
    config: AnalysisConfig,
) -> dict[str, Any]:
    if dl is None or dl.n == 0:
        return {"available": False, "notes": ["defender_not_tracked"]}

    h = max(ol.standing_height_px, 1.0)
    n = min(ol.n, dl.n)
    sep = _pair_distance(ol.hip[:n], dl.hip[:n]) / h
    mean_sep = float(np.nanmean(sep)) if np.any(~np.isnan(sep)) else None
    min_sep = float(np.nanmin(sep)) if np.any(~np.isnan(sep)) else None

    # Lateral matching: correlation of lateral hip velocities
    ol_x = ol.hip[:n, 0]
    dl_x = dl.hip[:n, 0]
    ol_vx = nan_vel(ol_x, ol.fps)
    dl_vx = nan_vel(dl_x, dl.fps)
    mask = ~np.isnan(ol_vx) & ~np.isnan(dl_vx)
    lateral_match = None
    if mask.sum() >= 5:
        lateral_match = float(np.corrcoef(ol_vx[mask], dl_vx[mask])[0, 1])

    # Inside leverage loss: DL stacked on OL midline while close.
    inside_loss_frames = 0
    widths = []
    for i in range(n):
        if not np.any(np.isnan(ol.ankle_l[i])) and not np.any(np.isnan(ol.ankle_r[i])):
            widths.append(abs(float(ol.ankle_l[i][0] - ol.ankle_r[i][0])))
    mean_sw = float(np.mean(widths)) if widths else h * 0.3

    for i in range(n):
        if np.any(np.isnan(ol.hip[i])) or np.any(np.isnan(dl.hip[i])) or np.isnan(sep[i]):
            continue
        if sep[i] < config.separation_close_frac and abs(dl.hip[i][0] - ol.hip[i][0]) < 0.15 * mean_sw:
            inside_loss_frames += 1

    # Redirect: defender lateral velocity sign flip, then OL recovery time
    recovery_frames = None
    redirect_frame = None
    flip_thresh = config.redirect_vel_flip_frac * h * dl.fps
    for i in range(2, n):
        if np.isnan(dl_vx[i]) or np.isnan(dl_vx[i - 1]):
            continue
        if dl_vx[i - 1] * dl_vx[i] < 0 and abs(dl_vx[i]) > flip_thresh and abs(dl_vx[i - 1]) > flip_thresh * 0.5:
            redirect_frame = dl.frames[i]
            target_sign = np.sign(dl_vx[i])
            for j in range(i, min(i + int(dl.fps * 1.5), n)):
                if np.isnan(ol_vx[j]):
                    continue
                if np.sign(ol_vx[j]) == target_sign and abs(ol_vx[j]) > flip_thresh * 0.4:
                    recovery_frames = j - i
                    break
            break

    coach_flags = []
    notes: list[str] = []
    if lateral_match is not None and lateral_match >= 0.45:
        coach_flags.append("mirror")
    if inside_loss_frames > int(0.15 * n):
        coach_flags.append("inside_leverage_loss")
    # Redirect needs a real recovery, not a one-frame vel flicker with no mirror.
    if recovery_frames is not None and recovery_frames <= 5:
        if lateral_match is None or lateral_match >= 0.20:
            coach_flags.append("redirect")
        else:
            notes.append("redirect_ignored_low_lateral_match")
    if recovery_frames is not None and recovery_frames > 10:
        coach_flags.append("slow_redirect")

    return {
        "available": True,
        "mean_separation": mean_sep,
        "min_separation": min_sep,
        "lateral_match_correlation": lateral_match,
        "inside_leverage_loss_frames": inside_loss_frames,
        "defender_redirect_frame": redirect_frame,
        "recovery_frames_after_redirect": recovery_frames,
        "recovery_ms_after_redirect": (recovery_frames / ol.fps * 1000.0) if recovery_frames is not None else None,
        "coach_flags": coach_flags,
        "notes": notes,
    }


def analyze_anchor(
    ol: AthleteSeries,
    dl: AthleteSeries | None,
    config: AnalysisConfig,
) -> dict[str, Any]:
    if dl is None:
        return {"available": False, "notes": ["defender_not_tracked"]}

    h = max(ol.standing_height_px, 1.0)
    n = min(ol.n, dl.n)
    sep = _pair_distance(ol.hip[:n], dl.hip[:n]) / h
    # Sustained contact (≥2 frames) — single-frame sep flash is not engagement.
    contact_idx = None
    run = 0
    for i in range(n):
        if not np.isnan(sep[i]) and sep[i] <= config.contact_distance_frac:
            run += 1
            if run >= 2:
                contact_idx = i - 1
                break
        else:
            run = 0
    if contact_idx is None:
        return {
            "available": True,
            "contact_detected": False,
            "notes": ["no_contact_in_window"],
            "coach_flags": [],
        }

    hip0 = ol.hip[contact_idx]
    sh0 = ol.shoulder[contact_idx]
    # Only first ~0.8s after contact — late identity swaps invent 2H “give”.
    win = min(n, contact_idx + max(6, int(ol.fps * 0.8)))
    hip_disp = []
    torso_disp = []
    for i in range(contact_idx, win):
        if not np.any(np.isnan(ol.hip[i])) and not np.any(np.isnan(hip0)):
            hip_disp.append(float(np.linalg.norm(ol.hip[i] - hip0) / h))
        if not np.any(np.isnan(ol.shoulder[i])) and not np.any(np.isnan(sh0)):
            torso_disp.append(float(np.linalg.norm(ol.shoulder[i] - sh0) / h))

    max_hip = float(np.max(hip_disp)) if hip_disp else None
    max_torso = float(np.max(torso_disp)) if torso_disp else None
    notes: list[str] = []
    track_break = max_hip is not None and max_hip >= 0.85
    if track_break:
        notes.append("hip_displacement_track_break_suspected")
        max_hip = None  # don't coach off broken track

    pocket = max_hip
    gives_ground = max_hip is not None and max_hip >= config.anchor_give_frac
    anchors = max_hip is not None and max_hip < config.anchor_give_frac * 0.7

    recovery = None
    if len(hip_disp) >= 6 and not track_break:
        peak_i = int(np.argmax(hip_disp))
        after = hip_disp[peak_i:]
        if after and after[-1] < after[0] * 0.85:
            recovery = True

    coach_flags = []
    if anchors:
        coach_flags.append("anchor")
    if gives_ground:
        coach_flags.append("gives_ground")
        # on_skates = substantial pocket collapse, not every inch of give
        if max_hip is not None and max_hip >= max(0.35, config.anchor_give_frac * 2.8):
            coach_flags.append("on_skates")
    if recovery:
        coach_flags.append("recovers_after_bull")

    return {
        "available": True,
        "contact_detected": True,
        "contact_frame": ol.frames[contact_idx],
        "max_hip_displacement_after_contact": max_hip,
        "max_torso_displacement_after_contact": max_torso,
        "pocket_depth_lost": pocket,
        "coach_flags": coach_flags,
        "notes": notes,
    }


def analyze_hands(
    ol: AthleteSeries,
    dl: AthleteSeries | None,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Approximate punch timing / placement — wrists often occluded (V1)."""
    notes = ["hand_placement_approximate_v1"]
    if dl is None:
        return {"available": False, "notes": notes + ["defender_not_tracked"]}

    h = max(ol.standing_height_px, 1.0)
    n = min(ol.n, dl.n)
    first_contact = None
    placement_scores = []
    vis = []

    for i in range(n):
        wl, wr = ol.wrist_l[i], ol.wrist_r[i]
        tgt = dl.shoulder[i] if not np.any(np.isnan(dl.shoulder[i])) else dl.hip[i]
        if np.any(np.isnan(tgt)):
            continue
        for w in (wl, wr):
            if np.any(np.isnan(w)):
                vis.append(0.0)
                continue
            vis.append(1.0)
            d = float(np.linalg.norm(w - tgt) / h)
            if d <= config.hand_reach_frac:
                if first_contact is None:
                    first_contact = ol.frames[i]
                # Inside frame: wrist x near DL torso x
                inside = abs(w[0] - tgt[0]) <= config.hand_inside_shoulder_tol_frac * h
                placement_scores.append(1.0 if inside else 0.35)

    visibility = float(np.mean(vis)) if vis else 0.0
    mean_place = float(np.mean(placement_scores)) if placement_scores else None
    contact_ms = None
    if first_contact is not None:
        contact_ms = (first_contact - ol.snap_frame) / ol.fps * 1000.0

    coach_flags = []
    if first_contact is not None and contact_ms is not None and contact_ms <= 350:
        coach_flags.append("punch_timing")
    if mean_place is not None and mean_place >= 0.7:
        coach_flags.append("hand_placement_inside")
    if visibility < 0.35:
        coach_flags.append("hands_occluded")
        notes.append("low_hand_visibility")

    return {
        "available": True,
        "time_to_first_contact_frame": first_contact,
        "time_to_first_contact_ms": contact_ms,
        "approximate_hand_placement_score": mean_place,
        "hand_visibility_confidence": visibility,
        "coach_flags": coach_flags,
        "notes": notes,
    }


def analyze_sustain(
    ol: AthleteSeries,
    dl: AthleteSeries | None,
    config: AnalysisConfig,
) -> dict[str, Any]:
    if dl is None:
        return {"available": False, "notes": ["defender_not_tracked"]}

    h = max(ol.standing_height_px, 1.0)
    n = min(ol.n, dl.n)
    sep = _pair_distance(ol.hip[:n], dl.hip[:n]) / h
    engaged = (~np.isnan(sep)) & (sep <= config.separation_close_frac)

    # Longest engagement run
    best = cur = 0
    start = best_start = None
    for i, e in enumerate(engaged):
        if e:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best = cur
                best_start = start
        else:
            cur = 0

    engagement_frames = int(best)
    engagement_ms = engagement_frames / ol.fps * 1000.0 if ol.fps else None
    disengage_frame = None
    if best_start is not None and best > 0:
        end_i = best_start + best
        if end_i < n:
            disengage_frame = ol.frames[end_i]

    # Pressure before disengagement: max sep growth in last 5 engaged frames
    pressure = None
    if best_start is not None and best >= 3:
        seg = sep[best_start : best_start + best]
        if np.any(~np.isnan(seg)):
            pressure = float(np.nanmax(seg) - np.nanmin(seg))

    # Contact = hips within contact distance for ≥2 consecutive frames (flash ≠ contact).
    contact_run = 0
    contacted = False
    for i in range(n):
        if not np.isnan(sep[i]) and sep[i] <= config.contact_distance_frac:
            contact_run += 1
            if contact_run >= 2:
                contacted = True
                break
        else:
            contact_run = 0

    coach_flags = []
    notes: list[str] = []
    if not contacted:
        notes.append("no_contact_detected")
    if contacted and engagement_frames >= config.sustain_min_frames:
        coach_flags.append("sustain")
    if contacted and engagement_frames >= int(ol.fps * 1.2):
        coach_flags.append("stay_with_him")
    if contacted and engagement_frames > 0 and engagement_frames < config.sustain_min_frames:
        coach_flags.append("early_disengage")

    return {
        "available": True,
        "contacted": contacted,
        "engagement_frames": engagement_frames,
        "engagement_ms": engagement_ms,
        "disengagement_frame": disengage_frame,
        "pressure_span_before_disengage": pressure,
        "coach_flags": coach_flags,
        "notes": notes,
    }
