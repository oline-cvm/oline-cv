"""Per-module trust / confidence scores — when to believe a metric."""

from __future__ import annotations

from typing import Any


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _level(score: float) -> str:
    if score >= 0.72:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _module_trust(key: str, data: dict[str, Any] | None, ctx: dict[str, Any]) -> dict[str, Any]:
    if not data:
        return {"score": 0.0, "level": "low", "reasons": ["not_computed"]}
    if data.get("available") is False:
        notes = list(data.get("notes") or ["unavailable"])
        if any("skipped_pass" in str(n) for n in notes):
            return {"score": 0.0, "level": "low", "reasons": notes}
        return {
            "score": 0.15,
            "level": "low",
            "reasons": notes,
        }

    reasons: list[str] = []
    score = 0.75
    notes = data.get("notes") or []
    fps = float(ctx.get("fps") or 30.0)
    rejects = int(ctx.get("track_rejects") or 0)
    usable = float(ctx.get("usable_frac") or 1.0)

    # Shared tracking health
    if rejects >= 10:
        score -= 0.18
        reasons.append("many_identity_rejects")
    elif rejects >= 4:
        score -= 0.08
        reasons.append("some_identity_rejects")
    if usable < 0.7:
        score -= 0.2
        reasons.append("sparse_pose")
    elif usable < 0.85:
        score -= 0.08
        reasons.append("partial_pose_gaps")

    if key == "initial_quicks":
        rt = data.get("reaction_time_frames")
        if rt is None:
            score = 0.25
            reasons.append("no_reaction")
        elif int(rt) <= 1 and fps < 55:
            score -= 0.35
            reasons.append("1_frame_quantization")
        elif int(rt) <= 2 and fps < 55:
            score -= 0.15
            reasons.append("coarse_timing_30fps")
        if any("unreliable" in str(n) for n in notes):
            score -= 0.2
            reasons.append("reaction_flagged_unreliable")

    elif key == "body_position":
        conf = data.get("posture_confidence")
        if conf is None:
            score -= 0.1
        else:
            score = 0.35 + 0.55 * _clamp01(float(conf))
        if data.get("posture_mixed"):
            score -= 0.15
            reasons.append("mixed_posture")
        if data.get("posture_classification") == "unknown":
            score = min(score, 0.3)
            reasons.append("unknown_posture")

    elif key == "hands":
        vis = data.get("hand_visibility_confidence")
        if vis is None:
            score -= 0.15
        else:
            score = 0.25 + 0.6 * _clamp01(float(vis))
        if any("occluded" in str(n) or "approximate" in str(n) for n in notes):
            score -= 0.12
            reasons.append("hands_approximate")
        if "hands_occluded" in (data.get("coach_flags") or []):
            score = min(score, 0.35)
            reasons.append("hands_occluded")

    elif key in ("mirror_redirect", "anchor", "sustain"):
        if not ctx.get("defender_tracked"):
            score = 0.2
            reasons.append("no_defender")
        if key == "anchor" and data.get("max_hip_displacement_after_contact") is None:
            if "track_break" in " ".join(str(n) for n in notes):
                score = min(score, 0.3)
                reasons.append("track_break")
        if key == "sustain" and data.get("contacted") is False:
            score -= 0.1
            reasons.append("no_contact")

    elif key == "footwork":
        cad = data.get("step_cadence_hz")
        if cad is not None and float(cad) > 5.0:
            score -= 0.2
            reasons.append("implausible_cadence")
        if any("implausible" in str(n) for n in notes):
            score -= 0.15
            reasons.append("step_noise")

    elif key in ("point_of_attack", "movement_in_space"):
        if ctx.get("play_type") != "run":
            return {"score": 0.0, "level": "low", "reasons": ["pass_play_skipped"]}

    elif key == "balance":
        if data.get("com_jitter") is None:
            score = 0.3
            reasons.append("no_com")

    score = _clamp01(score)
    if not reasons and score >= 0.72:
        reasons.append("stable_signal")
    return {"score": round(score, 3), "level": _level(score), "reasons": reasons[:4]}


def compute_trust(result: dict[str, Any]) -> dict[str, Any]:
    """Attach module + overall trust to an analysis result dict."""
    modules = result.get("modules") or {}
    s = result.get("rep_summary") or {}
    video = result.get("video") or {}
    lock = result.get("ol_lock") or s.get("ol_lock") or {}
    frames = result.get("frames") or []
    usable = 0
    if frames:
        usable = sum(1 for f in frames if not f.get("low_confidence")) / max(len(frames), 1)
    body = modules.get("body_position") or {}
    if body.get("valid_frame_count") and body.get("valid_frame_count"):
        # prefer body valid ratio when present
        flagged = int(body.get("flagged_frame_count") or 0)
        valid = int(body.get("valid_frame_count") or 0)
        if valid + flagged > 0:
            usable = valid / (valid + flagged)

    ctx = {
        "fps": video.get("fps") or 30.0,
        "track_rejects": lock.get("track_switch_rejects") or 0,
        "usable_frac": usable,
        "defender_tracked": bool(s.get("defender_tracked")),
        "play_type": result.get("play_type") or s.get("play_type") or "pass",
    }

    keys = [
        "initial_quicks",
        "footwork",
        "mirror_redirect",
        "anchor",
        "body_position",
        "hands",
        "sustain",
        "balance",
        "point_of_attack",
        "movement_in_space",
    ]
    per: dict[str, Any] = {}
    scores: list[float] = []
    for k in keys:
        t = _module_trust(k, modules.get(k), ctx)
        per[k] = t
        if modules.get(k) and modules[k].get("available") is not False:
            if k in ("point_of_attack", "movement_in_space") and ctx["play_type"] != "run":
                continue
            scores.append(t["score"])

    overall = sum(scores) / len(scores) if scores else 0.0
    trust = {
        "overall": {"score": round(overall, 3), "level": _level(overall)},
        "modules": per,
        "context": {
            "usable_pose_frac": round(usable, 3),
            "track_switch_rejects": ctx["track_rejects"],
            "fps": ctx["fps"],
        },
    }
    return trust
