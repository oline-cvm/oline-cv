"""Two-view fusion (sideline + endzone) for occlusion-robust OL analysis.

The single-view pipeline (`analyze_video`) is authoritative for one camera. It
already exposes a per-frame visibility signal: a set-window frame is "covered"
when ``low_confidence`` is False (the athlete was tracked with enough
keypoints). When a lineman leaves the sideline frame or is buried in a pile,
those frames go low-confidence and the metrics that depend on them lose trust.

This module runs the existing pipeline **once per view**, aligns the two reps by
their detected snap, and fuses the results:

* Coverage — the union of covered frames across both cameras. When the sideline
  loses the player, the endzone typically still has him (and vice-versa), so the
  combined rep coverage is far higher than either camera alone. This is the
  primary occlusion win and is reported explicitly.
* Per-metric selection — each Yeager module is taken from whichever view scored
  it best. "Best" is that view's own trust score for the module (which already
  bakes in pose coverage) plus a small perspective bonus, since some reads are
  geometrically better from one angle (e.g. base width / kick width from the
  endzone, pad level / get-off / punch timing from the sideline). Coverage
  dominates, so an occluded view never wins a metric it could not see.

The fused output is the same schema `analyze_video` returns (plus a `multiview`
block and per-view summaries), so packing, keyframes, the coach PDF, and the
dashboard all consume it unchanged. Cameras are not calibrated to each other, so
we deliberately fuse at the metric/coverage level rather than triangulating a
single 3D pose from mismatched pixel coordinates.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from oline_cv.config import AnalysisConfig
from oline_cv.pipeline import analyze_video
from oline_cv.trust import compute_trust

# Which camera is geometrically better suited to each module. Coverage/trust
# still dominates the choice; this only breaks near-ties toward the view whose
# angle reads that metric more faithfully.
ENDZONE_PREFERRED = {"footwork", "mirror_redirect", "movement_in_space"}
SIDELINE_PREFERRED = {
    "initial_quicks",
    "body_position",
    "anchor",
    "hands",
    "sustain",
    "point_of_attack",
    "balance",
}
_PERSPECTIVE_BONUS = 0.12

# Modules fused independently. Order mirrors the single-view result.
_MODULE_KEYS = (
    "initial_quicks",
    "footwork",
    "mirror_redirect",
    "anchor",
    "body_position",
    "balance",
    "hands",
    "sustain",
    "point_of_attack",
    "movement_in_space",
)

ProgressCB = Callable[[float, str, str], None]


@dataclass
class ViewInput:
    """One camera feeding the fusion."""

    video_path: str
    role: str  # "sideline" | "endzone" (any label; only the two above get a bonus)
    pick_xy: tuple[float, float] | None = None
    jersey: int | None = None
    snap_frame: int | None = None


@dataclass
class ViewArtifacts:
    """Per-view outputs surfaced back to the caller (for overlays / UI)."""

    role: str
    result: dict[str, Any]
    analysis_json: str | None = None
    overlay_path: str | None = None
    coverage: float = 0.0
    covered_frames: int = 0
    total_frames: int = 0
    snap_frame: int | None = None
    frames_lost: int = 0
    trust_overall: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


def _view_coverage(result: dict[str, Any]) -> tuple[float, int, int]:
    """Fraction of the set window where the athlete was actually tracked."""
    frames = result.get("frames") or []
    if not frames:
        return 0.0, 0, 0
    covered = sum(1 for f in frames if not f.get("low_confidence"))
    return covered / len(frames), covered, len(frames)


def _snap_ms(result: dict[str, Any]) -> float:
    fps = float((result.get("video") or {}).get("fps") or 30.0) or 30.0
    snap = int((result.get("snap") or {}).get("snap_frame") or 0)
    return (snap / fps) * 1000.0


def _combined_coverage(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Union coverage across views on a shared, snap-aligned timeline.

    Aligns by time relative to each view's snap (robust to different fps and
    clip start offsets), buckets into common frame steps, then compares the
    union of covered buckets against the union of all observed buckets.
    """
    fps_vals = [
        float((r.get("video") or {}).get("fps") or 30.0) or 30.0 for r in results
    ]
    step_ms = 1000.0 / max(1.0, min(fps_vals)) if fps_vals else 1000.0 / 30.0

    total: set[int] = set()
    covered: set[int] = set()
    per_view_covered: list[set[int]] = []
    for r in results:
        snap_ms = _snap_ms(r)
        view_cov: set[int] = set()
        for f in r.get("frames") or []:
            rel = float(f.get("timestamp_ms") or 0.0) - snap_ms
            if rel < -step_ms:  # small tolerance before snap
                continue
            bucket = int(round(rel / step_ms))
            total.add(bucket)
            if not f.get("low_confidence"):
                covered.add(bucket)
                view_cov.add(bucket)
        per_view_covered.append(view_cov)

    denom = len(total) or 1
    combined = len(covered) / denom
    best_single = max((len(c) / denom for c in per_view_covered), default=0.0)
    return {
        "combined_coverage": round(combined, 4),
        "best_single_coverage": round(best_single, 4),
        "occlusion_reduction": round(max(0.0, combined - best_single), 4),
        "rep_buckets": len(total),
        "covered_buckets": len(covered),
        "step_ms": round(step_ms, 3),
    }


def _module_quality(result: dict[str, Any], key: str) -> float:
    """This view's own trust score for a module (0 if unavailable/missing)."""
    mod = (result.get("modules") or {}).get(key) or {}
    if mod.get("available") is False:
        return 0.0
    tmods = ((result.get("trust") or {}).get("modules") or {}).get(key) or {}
    return float(tmods.get("score") or 0.0)


def _perspective_bonus(role: str, key: str) -> float:
    if role == "endzone" and key in ENDZONE_PREFERRED:
        return _PERSPECTIVE_BONUS
    if role == "sideline" and key in SIDELINE_PREFERRED:
        return _PERSPECTIVE_BONUS
    return 0.0


def _select_module_view(
    key: str, views: list[ViewArtifacts], primary_idx: int
) -> int:
    """Index of the view whose read of `key` we trust most."""
    best_idx = primary_idx
    best_score = -1.0
    for i, v in enumerate(views):
        score = _module_quality(v.result, key) + _perspective_bonus(v.role, key)
        # Tie-break toward the primary view, then higher overall coverage.
        tie = (i == primary_idx, v.coverage)
        if score > best_score or (
            abs(score - best_score) < 1e-9
            and (tie > (best_idx == primary_idx, views[best_idx].coverage))
        ):
            best_score = score
            best_idx = i
    return best_idx


def _build_rep_summary(
    base_summary: dict[str, Any],
    fused_modules: dict[str, dict[str, Any]],
    coach_language: list[str],
    defender_tracked: bool,
) -> dict[str, Any]:
    """Rebuild the flat rep rollup from the chosen per-module data.

    Mirrors the construction in ``pipeline.analyze_video`` so downstream
    packing/PDF see identical keys, but each value comes from the fused module.
    """
    quicks = fused_modules.get("initial_quicks") or {}
    body = fused_modules.get("body_position") or {}
    footwork = fused_modules.get("footwork") or {}
    mirror = fused_modules.get("mirror_redirect") or {}
    anchor = fused_modules.get("anchor") or {}
    hands = fused_modules.get("hands") or {}
    sustain = fused_modules.get("sustain") or {}

    summary = {
        "target_jersey": base_summary.get("target_jersey"),
        "ol_lock": base_summary.get("ol_lock"),
        "play_type": base_summary.get("play_type"),
        "reaction_time_ms": quicks.get("reaction_time_ms"),
        "reaction_time_frames": quicks.get("reaction_time_frames"),
        "initiated_by": quicks.get("initiated_by"),
        "late_off_the_ball": quicks.get("late_off_the_ball"),
        "posture_classification": body.get("posture_classification"),
        "posture_confidence": body.get("posture_confidence"),
        "posture_mixed": body.get("posture_mixed"),
        "mean_knee_flexion_deg": body.get("mean_knee_flexion_deg"),
        "min_knee_flexion_deg": body.get("min_knee_flexion_deg"),
        "mean_torso_angle_deg": body.get("mean_torso_angle_deg"),
        "hip_height_at_lowest": body.get("hip_height_at_lowest"),
        "mean_hip_height": body.get("mean_hip_height"),
        "step_cadence_hz": footwork.get("step_cadence_hz"),
        "set_depth": footwork.get("set_depth"),
        "set_width": footwork.get("set_width"),
        "mean_base_width": footwork.get("mean_base_width"),
        "overset": footwork.get("overset"),
        "lateral_match": mirror.get("lateral_match_correlation"),
        "anchor_give": anchor.get("max_hip_displacement_after_contact"),
        "punch_ms": hands.get("time_to_first_contact_ms"),
        "engagement_ms": sustain.get("engagement_ms"),
        "contacted": sustain.get("contacted"),
        "coach_language": coach_language,
        "defender_tracked": defender_tracked,
    }
    for k in ("posture_frame_counts", "valid_frame_count", "flagged_frame_count"):
        if k in body:
            summary[k] = body[k]
    return summary


def _validate_alignment(views: list[ViewArtifacts]) -> dict[str, Any]:
    """Check whether two views likely depict the same play.

    Always returns results (never blocks fusion), but flags mismatches so the
    UI can warn the user.  Handles clips that start at different times: each
    view independently detects its own snap, so alignment works as long as both
    clips contain the same snap event.
    """
    warnings: list[str] = []
    confidence = "high"

    per_view: list[dict[str, Any]] = []
    for v in views:
        video = v.result.get("video") or {}
        snap_info = v.result.get("snap") or {}
        set_info = v.result.get("set") or {}
        fps = float(video.get("fps") or 30.0) or 30.0
        snap_frame = int(snap_info.get("snap_frame") or 0)
        end_frame = int(set_info.get("end_frame") or snap_frame)
        n_frames = int(video.get("frame_count") or end_frame + 1)
        per_view.append(
            {
                "role": v.role,
                "fps": fps,
                "snap_frame": snap_frame,
                "snap_at_s": round(snap_frame / fps, 3),
                "snap_confidence": snap_info.get("confidence", "unknown"),
                "snap_method": snap_info.get("method", "unknown"),
                "rep_duration_s": round((end_frame - snap_frame) / fps, 3),
                "clip_duration_s": round(n_frames / fps, 3),
            }
        )

    # --- Snap confidence --------------------------------------------------
    low_snap = [p for p in per_view if p["snap_confidence"] == "low"]
    if low_snap:
        roles = ", ".join(p["role"] for p in low_snap)
        warnings.append(
            f"Snap detection uncertain in {roles} — view alignment may be off"
        )
        if confidence == "high":
            confidence = "medium"

    # --- Rep-duration similarity ------------------------------------------
    durations = [p["rep_duration_s"] for p in per_view]
    delta: float = 0.0
    overlap_frac: float = 1.0
    if len(durations) >= 2:
        delta = abs(durations[0] - durations[1])
        shorter = min(durations)
        longer = max(durations)
        overlap_frac = shorter / longer if longer > 0 else 1.0

        if delta > 5.0:
            warnings.append(
                f"Rep durations differ by {delta:.1f}s — these may be different plays"
            )
            confidence = "low"
        elif delta > 2.0:
            warnings.append(
                f"Rep durations differ by {delta:.1f}s — check that both views"
                " show the same snap"
            )
            if confidence == "high":
                confidence = "medium"

        if overlap_frac < 0.3:
            warnings.append(
                "Low rep-timeline overlap between views — possible different plays"
            )
            confidence = "low"
        elif overlap_frac < 0.6 and confidence == "high":
            confidence = "medium"

    # --- FPS agreement ----------------------------------------------------
    fps_set = {p["fps"] for p in per_view}
    if len(fps_set) > 1:
        fps_str = ", ".join(f"{p['role']}={p['fps']}" for p in per_view)
        warnings.append(
            f"Frame rates differ ({fps_str}) — alignment uses snap-relative time"
        )

    return {
        "same_play_confidence": confidence,
        "warnings": warnings,
        "rep_duration_delta_s": round(delta, 3),
        "timeline_overlap_frac": round(overlap_frac, 4),
        "per_view": per_view,
    }


def fuse_view_results(
    views: list[ViewArtifacts], primary_role: str = "sideline"
) -> dict[str, Any]:
    """Fuse per-view analysis dicts into one occlusion-robust result.

    Pure function over already-computed results — the unit-testable core. Picks
    a primary view for structural fields (frames/overlay/snap), selects each
    module from its best view, rebuilds the rollup, and recomputes trust against
    the combined coverage.
    """
    if not views:
        raise ValueError("fuse_view_results requires at least one view")

    # Fill coverage stats on each view.
    for v in views:
        cov, ncov, ntot = _view_coverage(v.result)
        v.coverage, v.covered_frames, v.total_frames = cov, ncov, ntot
        v.snap_frame = (v.result.get("snap") or {}).get("snap_frame")
        v.frames_lost = int((v.result.get("ol_lock") or {}).get("frames_lost") or 0)
        v.trust_overall = float(
            ((v.result.get("trust") or {}).get("overall") or {}).get("score") or 0.0
        )

    if len(views) == 1:
        return views[0].result

    # Primary view drives structural fields (frames, overlay, keyframes).
    primary_idx = next(
        (i for i, v in enumerate(views) if v.role == primary_role), 0
    )
    fused = copy.deepcopy(views[primary_idx].result)

    # Select each module from its best-covered / best-suited view.
    fused_modules: dict[str, Any] = {}
    module_sources: dict[str, str] = {}
    for key in _MODULE_KEYS:
        src = _select_module_view(key, views, primary_idx)
        mod = (views[src].result.get("modules") or {}).get(key)
        if mod is None:
            mod = (views[primary_idx].result.get("modules") or {}).get(key) or {}
            src = primary_idx
        fused_modules[key] = copy.deepcopy(mod)
        module_sources[key] = views[src].role

    fused["modules"] = fused_modules
    fused["initial_quicks"] = fused_modules.get("initial_quicks", {})

    # Coach language: union across the chosen modules (dedup, drop "unknown").
    coach_language: list[str] = []
    for mod in fused_modules.values():
        for flag in mod.get("coach_flags") or []:
            if flag and flag != "unknown" and flag not in coach_language:
                coach_language.append(flag)
    fused["coach_language"] = coach_language

    defender_tracked = any(
        bool((v.result.get("rep_summary") or {}).get("defender_tracked"))
        for v in views
    )
    fused["rep_summary"] = _build_rep_summary(
        views[primary_idx].result.get("rep_summary") or {},
        fused_modules,
        coach_language,
        defender_tracked,
    )

    coverage = _combined_coverage([v.result for v in views])
    alignment = _validate_alignment(views)

    # Recompute trust against the *combined* coverage so the shared sparse-pose
    # penalty reflects the fused rep, not the primary camera alone.
    fused["trust"] = compute_trust(
        fused, usable_frac_override=coverage["combined_coverage"]
    )
    fused["rep_summary"]["trust_overall"] = fused["trust"]["overall"]

    # Merge per-view timing from alignment into each view summary.
    timing_by_role = {p["role"]: p for p in alignment["per_view"]}
    view_summaries: list[dict[str, Any]] = []
    for v in views:
        entry: dict[str, Any] = {
            "role": v.role,
            "coverage": round(v.coverage, 4),
            "covered_frames": v.covered_frames,
            "total_frames": v.total_frames,
            "snap_frame": v.snap_frame,
            "frames_lost": v.frames_lost,
            "trust_overall": round(v.trust_overall, 4),
        }
        timing = timing_by_role.get(v.role)
        if timing:
            entry["snap_at_s"] = timing["snap_at_s"]
            entry["snap_confidence"] = timing["snap_confidence"]
            entry["rep_duration_s"] = timing["rep_duration_s"]
            entry["clip_duration_s"] = timing["clip_duration_s"]
        view_summaries.append(entry)

    fused["multiview"] = {
        "enabled": True,
        "primary_role": views[primary_idx].role,
        "coverage": coverage,
        "alignment": alignment,
        "module_sources": module_sources,
        "views": view_summaries,
    }
    return fused


def analyze_multiview(
    views: list[ViewInput],
    base_config: AnalysisConfig | None = None,
    output_json: str | None = None,
    artifact_dir: str | None = None,
    artifact_prefix: str = "clip",
    progress_cb: ProgressCB | None = None,
    primary_role: str = "sideline",
) -> tuple[dict[str, Any], list[ViewArtifacts]]:
    """Run the single-view pipeline per camera, then fuse.

    Returns the fused result dict and the per-view artifacts (each carrying its
    own overlay/JSON paths and coverage). Single view in → single view out
    (no fusion), so callers can pass one or two views uniformly.
    """
    if not views:
        raise ValueError("analyze_multiview requires at least one view")

    base_config = base_config or AnalysisConfig()
    out_root = Path(artifact_dir) if artifact_dir else Path(views[0].video_path).parent

    n = len(views)
    span = 1.0 / n

    artifacts: list[ViewArtifacts] = []
    for i, view in enumerate(views):
        lo = i * span * 92.0
        hi = (i + 1) * span * 92.0

        def _sub(pct: float, msg: str, stage: str, _lo=lo, _hi=hi, _role=view.role):
            if progress_cb is None:
                return
            scaled = _lo + (_hi - _lo) * (float(pct) / 100.0)
            label = f"[{_role}] {msg}" if n > 1 else msg
            try:
                progress_cb(scaled, label, stage)
            except Exception:
                pass

        cfg = copy.deepcopy(base_config)
        cfg.athlete_pick_xy = view.pick_xy
        cfg.target_jersey = view.jersey
        if view.snap_frame is not None:
            cfg.snap_frame_override = view.snap_frame

        view_json = str(out_root / f"{artifact_prefix}__{view.role}_analysis.json")
        view_overlay = str(out_root / f"{artifact_prefix}__{view.role}_overlay.mp4")

        result = analyze_video(
            view.video_path,
            config=cfg,
            output_json=view_json,
            overlay_path=view_overlay,
            progress_cb=_sub,
        )
        art = ViewArtifacts(
            role=view.role,
            result=result,
            analysis_json=view_json,
            overlay_path=result.get("overlay_video", view_overlay),
        )
        artifacts.append(art)

    if progress_cb is not None:
        try:
            progress_cb(93.0, "Fusing views…", "metrics")
        except Exception:
            pass

    fused = fuse_view_results(artifacts, primary_role=primary_role)

    if output_json is not None:
        Path(output_json).write_text(json.dumps(fused, indent=2), encoding="utf-8")
        fused["output_json"] = output_json

    return fused, artifacts
