"""Associate third-party person detections with OUR locked offensive lineman.

WHAM's preprocessing detects every person in frame and runs its own tracker to
decide who to reconstruct. That decision must never be trusted here: the whole
point of the BoT-SORT identity stack is that we already know which body is the
target. This module is the gate between the two.

Per frame we take the TrackManifest target bbox as ground truth and score every
candidate detection against it. A candidate must clear all of:

    IoU                 overlap with the target box
    center distance     as a fraction of the target box diagonal
    area ratio          detection area vs target area

A frame with no passing candidate is marked INVALID and dropped. It is never
back-filled with the best-of-a-bad-bunch, because a defender in contact is
exactly the detection that would win such a fallback. Preferring a hole over a
wrong body is the same rule the 2D tracker already follows.

Pure numpy so the identical scoring runs on the Windows side (for the debug
video) and inside the WSL conda env (for the real reconstruction).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


BRIDGED = "bridged"


class RejectReason:
    LOW_IOU = "low_iou"
    FAR_CENTER = "far_center"
    AREA_MISMATCH = "area_mismatch"
    LOW_SCORE = "low_score"
    NOT_BEST = "not_best"
    AMBIGUOUS = "ambiguous"


class InvalidReason:
    NO_DETECTIONS = "no_detections"
    ALL_REJECTED = "all_rejected"
    NO_TARGET_BBOX = "no_target_bbox"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class AssociationThresholds:
    """Configurable gates. Defaults are tuned for sideline OL film.

    Contact plays put a defender's box heavily over the lineman's, so the gates
    lean strict; loosening min_iou is the fastest way to start reconstructing the
    wrong player.
    """

    min_iou: float = 0.35
    max_center_dist_frac: float = 0.45  # of the target bbox diagonal
    min_area_ratio: float = 0.45
    max_area_ratio: float = 2.20
    min_score: float = 0.45
    # Two candidates this close in score means we cannot tell OL from defender.
    ambiguous_margin: float = 0.08
    reject_ambiguous: bool = False
    # Bridging: a detector that blinks for a frame or two should not split a run,
    # but anything longer is a real loss of evidence and stays a hole.
    max_bridge_gap: int = 3
    # Per-frame budget for how far the body may travel across a bridged gap, as a
    # fraction of its box diagonal. An OL at full speed covers well under 0.1.
    max_bridge_step_frac: float = 0.25
    # Bridged frames are interpolation, not observation, so their confidence is
    # scaled down to keep them visually distinguishable downstream.
    bridge_confidence_penalty: float = 0.5
    w_iou: float = 0.60
    w_center: float = 0.30
    w_area: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_iou": self.min_iou,
            "max_center_dist_frac": self.max_center_dist_frac,
            "min_area_ratio": self.min_area_ratio,
            "max_area_ratio": self.max_area_ratio,
            "min_score": self.min_score,
            "ambiguous_margin": self.ambiguous_margin,
            "reject_ambiguous": self.reject_ambiguous,
            "max_bridge_gap": self.max_bridge_gap,
            "max_bridge_step_frac": self.max_bridge_step_frac,
            "bridge_confidence_penalty": self.bridge_confidence_penalty,
            "weights": {"iou": self.w_iou, "center": self.w_center, "area": self.w_area},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> AssociationThresholds:
        w = d.get("weights") or {}
        return cls(
            min_iou=float(d.get("min_iou", 0.35)),
            max_center_dist_frac=float(d.get("max_center_dist_frac", 0.45)),
            min_area_ratio=float(d.get("min_area_ratio", 0.45)),
            max_area_ratio=float(d.get("max_area_ratio", 2.20)),
            min_score=float(d.get("min_score", 0.45)),
            ambiguous_margin=float(d.get("ambiguous_margin", 0.08)),
            reject_ambiguous=bool(d.get("reject_ambiguous", False)),
            max_bridge_gap=int(d.get("max_bridge_gap", 3)),
            max_bridge_step_frac=float(d.get("max_bridge_step_frac", 0.25)),
            bridge_confidence_penalty=float(d.get("bridge_confidence_penalty", 0.5)),
            w_iou=float(w.get("iou", 0.60)),
            w_center=float(w.get("center", 0.30)),
            w_area=float(w.get("area", 0.10)),
        )


@dataclass
class Candidate:
    """One detection scored against the target box."""

    index: int
    bbox: list[float]
    detection_confidence: float
    iou: float = 0.0
    center_dist_frac: float = 0.0
    area_ratio: float = 0.0
    score: float = 0.0
    accepted: bool = False
    selected: bool = False
    reject_reason: str | None = None
    # True when this box was interpolated across a bridged gap rather than detected.
    interpolated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "bbox": [round(float(v), 2) for v in self.bbox],
            "detection_confidence": round(float(self.detection_confidence), 4),
            "iou": round(float(self.iou), 4),
            "center_dist_frac": round(float(self.center_dist_frac), 4),
            "area_ratio": round(float(self.area_ratio), 4),
            "score": round(float(self.score), 4),
            "accepted": bool(self.accepted),
            "selected": bool(self.selected),
            "reject_reason": self.reject_reason,
            "interpolated": bool(self.interpolated),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Candidate:
        return cls(
            index=int(d["index"]),
            bbox=[float(v) for v in d["bbox"]],
            detection_confidence=float(d.get("detection_confidence", 0.0)),
            iou=float(d.get("iou", 0.0)),
            center_dist_frac=float(d.get("center_dist_frac", 0.0)),
            area_ratio=float(d.get("area_ratio", 0.0)),
            score=float(d.get("score", 0.0)),
            accepted=bool(d.get("accepted", False)),
            selected=bool(d.get("selected", False)),
            reject_reason=d.get("reject_reason"),
            interpolated=bool(d.get("interpolated", False)),
        )


@dataclass
class FrameAssociation:
    frame_index: int
    target_bbox: list[float] | None
    valid: bool
    candidates: list[Candidate] = field(default_factory=list)
    invalid_reason: str | None = None
    ambiguous: bool = False
    # BoT-SORT track id for this frame. Bridging requires it to be unchanged
    # across the gap, so an identity switch can never be papered over.
    track_id: int | None = None
    bridged: bool = False
    # Confidence override for bridged frames, which are interpolated not observed.
    bridged_confidence: float | None = None

    @property
    def selected(self) -> Candidate | None:
        for c in self.candidates:
            if c.selected:
                return c
        return None

    @property
    def confidence(self) -> float:
        if self.bridged and self.bridged_confidence is not None:
            return float(self.bridged_confidence)
        sel = self.selected
        return float(sel.score) if sel else 0.0

    @property
    def interpolated(self) -> bool:
        sel = self.selected
        return bool(sel.interpolated) if sel else False

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "track_id": self.track_id,
            "target_bbox": (
                None if self.target_bbox is None else [round(float(v), 2) for v in self.target_bbox]
            ),
            "valid": bool(self.valid),
            "ambiguous": bool(self.ambiguous),
            "bridged": bool(self.bridged),
            "invalid_reason": self.invalid_reason,
            "confidence": round(self.confidence, 4),
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FrameAssociation:
        bridged = bool(d.get("bridged", False))
        return cls(
            frame_index=int(d["frame_index"]),
            track_id=(None if d.get("track_id") is None else int(d["track_id"])),
            target_bbox=(
                None if d.get("target_bbox") is None else [float(v) for v in d["target_bbox"]]
            ),
            valid=bool(d.get("valid", False)),
            candidates=[Candidate.from_dict(c) for c in (d.get("candidates") or [])],
            invalid_reason=d.get("invalid_reason"),
            ambiguous=bool(d.get("ambiguous", False)),
            bridged=bridged,
            bridged_confidence=(float(d["confidence"]) if bridged and "confidence" in d else None),
        )


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = (float(v) for v in a)
    bx0, by0, bx1, by1 = (float(v) for v in b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-9 else 0.0


def _center(b: Sequence[float]) -> np.ndarray:
    return np.array([(float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0])


def _diag(b: Sequence[float]) -> float:
    return float(np.hypot(float(b[2]) - float(b[0]), float(b[3]) - float(b[1])))


def _area(b: Sequence[float]) -> float:
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def associate_frame(
    frame_index: int,
    target_bbox: Sequence[float] | None,
    detections: Sequence[Sequence[float]],
    confidences: Sequence[float] | None = None,
    thresholds: AssociationThresholds | None = None,
) -> FrameAssociation:
    """Pick the detection that is our lineman, or mark the frame invalid.

    ``detections`` are xyxy person boxes in full-frame pixels, in the same space
    as ``target_bbox``.
    """
    th = thresholds or AssociationThresholds()

    if target_bbox is None:
        return FrameAssociation(
            frame_index=frame_index,
            target_bbox=None,
            valid=False,
            invalid_reason=InvalidReason.NO_TARGET_BBOX,
        )

    target = [float(v) for v in target_bbox]
    if not detections:
        return FrameAssociation(
            frame_index=frame_index,
            target_bbox=target,
            valid=False,
            invalid_reason=InvalidReason.NO_DETECTIONS,
        )

    t_center = _center(target)
    t_diag = max(_diag(target), 1e-6)
    t_area = max(_area(target), 1e-6)
    confs = list(confidences or [0.0] * len(detections))

    candidates: list[Candidate] = []
    for i, det in enumerate(detections):
        box = [float(v) for v in det]
        iou = iou_xyxy(target, box)
        center_frac = float(np.linalg.norm(_center(box) - t_center) / t_diag)
        area_ratio = _area(box) / t_area

        cand = Candidate(
            index=i,
            bbox=box,
            detection_confidence=float(confs[i] if i < len(confs) else 0.0),
            iou=iou,
            center_dist_frac=center_frac,
            area_ratio=area_ratio,
        )

        # Gates, checked in order of how decisive they are for identity.
        if iou < th.min_iou:
            cand.reject_reason = RejectReason.LOW_IOU
        elif center_frac > th.max_center_dist_frac:
            cand.reject_reason = RejectReason.FAR_CENTER
        elif not (th.min_area_ratio <= area_ratio <= th.max_area_ratio):
            cand.reject_reason = RejectReason.AREA_MISMATCH

        area_agreement = min(area_ratio, 1.0 / area_ratio) if area_ratio > 1e-9 else 0.0
        center_agreement = max(0.0, 1.0 - center_frac / max(th.max_center_dist_frac, 1e-6))
        cand.score = float(
            th.w_iou * iou + th.w_center * center_agreement + th.w_area * area_agreement
        )

        if cand.reject_reason is None and cand.score < th.min_score:
            cand.reject_reason = RejectReason.LOW_SCORE
        cand.accepted = cand.reject_reason is None
        candidates.append(cand)

    accepted = sorted(
        (c for c in candidates if c.accepted), key=lambda c: c.score, reverse=True
    )
    if not accepted:
        return FrameAssociation(
            frame_index=frame_index,
            target_bbox=target,
            valid=False,
            candidates=candidates,
            invalid_reason=InvalidReason.ALL_REJECTED,
        )

    best = accepted[0]
    ambiguous = len(accepted) > 1 and (best.score - accepted[1].score) < th.ambiguous_margin

    if ambiguous and th.reject_ambiguous:
        for c in accepted:
            c.accepted = False
            c.reject_reason = RejectReason.AMBIGUOUS
        return FrameAssociation(
            frame_index=frame_index,
            target_bbox=target,
            valid=False,
            candidates=candidates,
            invalid_reason=InvalidReason.AMBIGUOUS,
            ambiguous=True,
        )

    best.selected = True
    for c in accepted[1:]:
        c.reject_reason = RejectReason.NOT_BEST
    return FrameAssociation(
        frame_index=frame_index,
        target_bbox=target,
        valid=True,
        candidates=candidates,
        ambiguous=ambiguous,
    )


def associate_sequence(
    target_frames: Iterable[Any],
    detections_by_frame: dict[int, tuple[Sequence[Sequence[float]], Sequence[float]]],
    thresholds: AssociationThresholds | None = None,
) -> list[FrameAssociation]:
    """Associate a whole segment. ``target_frames`` are TrackFrame records."""
    out: list[FrameAssociation] = []
    for fr in target_frames:
        dets, confs = detections_by_frame.get(fr.frame_index, ([], []))
        assoc = associate_frame(fr.frame_index, fr.bbox, dets, confs, thresholds=thresholds)
        assoc.track_id = getattr(fr, "track_id", None)
        out.append(assoc)
    return out


def bridge_gaps(
    associations: list[FrameAssociation],
    thresholds: AssociationThresholds | None = None,
) -> dict[str, Any]:
    """Fill short detector dropouts by interpolation, in place.

    A gap is only bridged when every one of these holds:

      * it is at most ``max_bridge_gap`` frames long
      * it is bounded by an accepted detection on BOTH sides
      * the BoT-SORT track id is unchanged across the gap, and no frame inside the
        gap carries a different id
      * the body did not move further than the per-frame step budget allows

    Anything else stays a hole. A longer gap means we genuinely stopped seeing the
    player, and an id change means the thing on the far side may be someone else —
    interpolating across either would invent motion and risk swapping bodies,
    which is the exact failure this whole layer exists to prevent.

    Bridged frames get an interpolated box and a reduced confidence so they stay
    distinguishable from observations everywhere downstream.
    """
    th = thresholds or AssociationThresholds()
    stats = {"bridged_frames": [], "skipped_gaps": []}

    if th.max_bridge_gap <= 0:
        return stats

    n = len(associations)
    i = 0
    while i < n:
        if associations[i].valid:
            i += 1
            continue

        start = i
        while i < n and not associations[i].valid:
            i += 1
        end = i - 1  # inclusive
        gap = associations[start : end + 1]
        gap_len = len(gap)

        prev_a = associations[start - 1] if start > 0 else None
        next_a = associations[end + 1] if end + 1 < n else None

        def skip(reason: str) -> None:
            stats["skipped_gaps"].append(
                {
                    "frames": [gap[0].frame_index, gap[-1].frame_index],
                    "length": gap_len,
                    "reason": reason,
                }
            )

        if prev_a is None or next_a is None:
            skip("unbounded")
            continue
        if gap_len > th.max_bridge_gap:
            skip("too_long")
            continue

        prev_sel, next_sel = prev_a.selected, next_a.selected
        if prev_sel is None or next_sel is None:
            skip("unbounded")
            continue

        # Identity must be the same track on both sides, and nothing inside the
        # gap may claim a different one.
        ids = {a.track_id for a in gap if a.track_id is not None}
        if prev_a.track_id != next_a.track_id:
            skip("identity_switch")
            continue
        if ids and ids != {prev_a.track_id}:
            skip("identity_switch")
            continue

        # Plausible travel: the body cannot teleport across a 1-3 frame gap.
        span = gap_len + 1
        diag = max(_diag(prev_sel.bbox), 1e-6)
        step = float(np.linalg.norm(_center(next_sel.bbox) - _center(prev_sel.bbox)) / diag)
        if step > th.max_bridge_step_frac * span:
            skip("moved_too_far")
            continue

        base_conf = min(prev_a.confidence, next_a.confidence) * th.bridge_confidence_penalty
        p0 = np.asarray(prev_sel.bbox, dtype=float)
        p1 = np.asarray(next_sel.bbox, dtype=float)
        for k, a in enumerate(gap, start=1):
            t = k / span
            box = (1.0 - t) * p0 + t * p1
            cand = Candidate(
                index=-1,
                bbox=[float(v) for v in box],
                detection_confidence=0.0,
                iou=float(iou_xyxy(a.target_bbox, box)) if a.target_bbox else 0.0,
                score=float(base_conf),
                accepted=True,
                selected=True,
                interpolated=True,
            )
            # Keep the rejected detections for the debug view, drop stale selections.
            for c in a.candidates:
                c.selected = False
                if c.reject_reason is None:
                    c.reject_reason = RejectReason.NOT_BEST
            a.candidates.append(cand)
            a.valid = True
            a.bridged = True
            a.invalid_reason = None
            a.bridged_confidence = float(base_conf)
            a.track_id = prev_a.track_id
            stats["bridged_frames"].append(a.frame_index)

    return stats


def summarize(associations: Sequence[FrameAssociation]) -> dict[str, Any]:
    """Counts a human needs to judge whether the association held."""
    total = len(associations)
    valid = [a for a in associations if a.valid]
    invalid = [a for a in associations if not a.valid]
    reasons: dict[str, int] = {}
    for a in invalid:
        key = a.invalid_reason or "unknown"
        reasons[key] = reasons.get(key, 0) + 1
    observed = [a for a in valid if not a.bridged]
    bridged = [a for a in valid if a.bridged]
    scores = [a.confidence for a in observed]
    ious = [a.selected.iou for a in observed if a.selected]
    return {
        "frames": total,
        "valid": len(valid),
        "observed": len(observed),
        "bridged": len(bridged),
        "bridged_frames": [a.frame_index for a in bridged],
        "invalid": len(invalid),
        "ambiguous": sum(1 for a in associations if a.ambiguous),
        "invalid_reasons": reasons,
        "unmatched_frames": [a.frame_index for a in invalid],
        "mean_confidence": round(float(np.mean(scores)), 4) if scores else 0.0,
        "min_confidence": round(float(np.min(scores)), 4) if scores else 0.0,
        "mean_iou": round(float(np.mean(ious)), 4) if ious else 0.0,
        "min_iou": round(float(np.min(ious)), 4) if ious else 0.0,
        "mean_candidates": (
            round(sum(len(a.candidates) for a in associations) / total, 2) if total else 0.0
        ),
    }


def longest_valid_run(associations: Sequence[FrameAssociation]) -> tuple[int, int] | None:
    """Longest contiguous run of valid frames, as (start_frame, end_frame).

    WHAM's temporal model needs a contiguous sequence, so when association drops
    frames mid-segment we reconstruct the longest clean run rather than stitching
    across a hole.
    """
    best: tuple[int, int] | None = None
    best_len = 0
    start: int | None = None
    prev: int | None = None

    for a in associations:
        if not a.valid:
            start = prev = None
            continue
        if start is None or (prev is not None and a.frame_index != prev + 1):
            start = a.frame_index
        prev = a.frame_index
        length = prev - start + 1
        if length > best_len:
            best_len, best = length, (start, prev)
    return best


def save_associations(
    path: str | Path,
    associations: Sequence[FrameAssociation],
    thresholds: AssociationThresholds,
    extra: dict[str, Any] | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "thresholds": thresholds.to_dict(),
        "summary": summarize(associations),
        "frames": [a.to_dict() for a in associations],
    }
    if extra:
        payload.update(extra)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


def load_associations(
    path: str | Path,
) -> tuple[list[FrameAssociation], AssociationThresholds, dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = [FrameAssociation.from_dict(f) for f in (data.get("frames") or [])]
    th = AssociationThresholds.from_dict(data.get("thresholds") or {})
    return frames, th, data.get("summary") or {}
