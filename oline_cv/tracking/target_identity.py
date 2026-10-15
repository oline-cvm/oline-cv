"""Locked-target identity: keep one tapped player across a clip.

OLINE-specific; not part of Deep HM-SORT. It sits on top of the generic
tracker's per-frame output and decides, conservatively, which detection (if
any) is the player the user tapped. Wrong-person tracking is worse than
missing tracking, so under doubt the target has no box rather than someone
else's.

States
------
VISIBLE_CONFIDENT        bound tracklet seen, consistent with the target's
                         clean appearance gallery, no close competitor
VISIBLE_AMBIGUOUS        bound tracklet seen but a competitor is close in
                         cost, or its appearance stopped matching; no box
OCCLUDED                 not seen recently (< target_occluded_max_s)
LOST                     not seen for longer
REACQUISITION_CANDIDATE  a candidate passes every gate; waiting for
                         confirmation over consecutive frames
REACQUIRED               confirmed reacquisition (box emitted), then back to
                         VISIBLE_CONFIDENT

Reacquisition gates (all must pass)
-----------------------------------
* spatial: the candidate overlaps the last confident box expanded by a scale
  that grows with time unseen (capped at ``target_max_expansion``), with
  Expansion-IoU distance <= ``target_reacquire_max_iou_distance`` on its own
* not a tracklet seen alongside the target (within ``target_exclusion_s``,
  default the whole clip): two simultaneously visible boxes are two people
* gallery appearance distance <= ``target_reid_max_distance``, tightened to
  a low quantile of the distances measured to those co-visible people when
  enough are known (generic embeddings put teammates close together)
* Deep HM cost <= ``target_reacquire_max_cost``
* margin to the runner-up >= ``target_reacquire_min_margin``
* the same candidate wins ``target_reacquire_confirm_frames`` frames in a row
  (immediate if it is the target's own tracklet re-found by the tracker)

The gallery holds the lock embedding plus recent *clean* samples only: high
score, not overlapping another detection, not cut by the frame edge, not
tiny, consistent with the gallery, and spaced in time.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from oline_cv.tracking.camera_motion import invert, warp_boxes
from oline_cv.tracking.costs import cosine_distance_matrix, harmonic_mean_distance
from oline_cv.tracking.deep_hm_sort import FrameResult
from oline_cv.tracking.expansion_iou import expansion_iou, iou_matrix
from oline_cv.tracking.occlusion_risk import evaluate_risk
from oline_cv.tracking.target_mask import MaskEvidence, mask_box_iou, mask_coverage


class TargetState(str, Enum):
    VISIBLE_CONFIDENT = "VISIBLE_CONFIDENT"
    VISIBLE_AMBIGUOUS = "VISIBLE_AMBIGUOUS"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"
    REACQUISITION_CANDIDATE = "REACQUISITION_CANDIDATE"
    REACQUIRED = "REACQUIRED"


BOXED_STATES = (TargetState.VISIBLE_CONFIDENT, TargetState.REACQUIRED)


def _box_center(box) -> np.ndarray:
    b = np.asarray(box, dtype=float).reshape(4)
    return np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=float)


@dataclass
class TargetFrame:
    frame_idx: int
    state: TargetState
    det_index: int | None = None
    bbox: np.ndarray | None = None
    track_id: int | None = None
    score: float | None = None
    cost: float | None = None
    d_iou: float | None = None
    d_app: float | None = None
    margin: float | None = None
    frames_since_seen: int = 0
    expansion: float | None = None
    appearance_updated: bool = False
    update_skip: str | None = None
    reason: str = ""
    candidates: list[dict] = field(default_factory=list)
    risk: str | None = None
    sam_active: bool = False
    sam_mask_available: bool = False
    sam_mask_confidence: float | None = None
    sam_mask_area: int | None = None
    mask_coverage: float | None = None
    recovery: bool = False

    @property
    def has_box(self) -> bool:
        return self.state in BOXED_STATES and self.bbox is not None

    def to_dict(self) -> dict:
        def f(x):
            return None if x is None else round(float(x), 4)

        return {
            "frame": self.frame_idx,
            "state": self.state.value,
            "det_index": self.det_index,
            "bbox": None if self.bbox is None else [round(float(v), 1) for v in self.bbox],
            "track_id": self.track_id,
            "score": f(self.score),
            "cost": f(self.cost),
            "d_iou": f(self.d_iou),
            "d_app": f(self.d_app),
            "margin": f(self.margin),
            "frames_since_seen": self.frames_since_seen,
            "expansion": f(self.expansion),
            "appearance_updated": self.appearance_updated,
            "update_skip": self.update_skip,
            "reason": self.reason,
            "candidates": self.candidates,
            "risk": self.risk,
            "sam_active": self.sam_active,
            "sam_mask_available": self.sam_mask_available,
            "sam_mask_confidence": f(self.sam_mask_confidence),
            "sam_mask_area": self.sam_mask_area,
            "mask_coverage": f(self.mask_coverage),
            "recovery": self.recovery,
        }


class LockedTargetIdentity:
    def __init__(self, cfg, fps: float, image_wh: tuple[int, int] | None, logical_id: int = 1):
        self.cfg = cfg
        self.fps = max(float(fps), 1.0)
        self.image_wh = image_wh
        self.logical_id = logical_id
        self.reid_max = (
            cfg.target_reid_max_distance
            if cfg.target_reid_max_distance is not None
            else cfg.appearance_cost_threshold
        )
        self._occl_frames = max(1, int(round(cfg.target_occluded_max_s * self.fps)))
        self._expand_every = max(1, int(round(cfg.target_expand_every_s * self.fps)))
        self._excl_frames = (
            None if cfg.target_exclusion_s is None
            else max(1, int(round(cfg.target_exclusion_s * self.fps)))
        )
        self.impostors: deque = deque(maxlen=400)
        self._gallery_gap = max(1, int(round(cfg.target_gallery_min_interval_s * self.fps)))
        self.anchor: np.ndarray | None = None
        self.gallery: deque = deque(maxlen=max(1, cfg.locked_target_gallery_size - 1))
        self.last_box: np.ndarray | None = None
        self.last_seen: int | None = None
        self.last_gallery_frame: int | None = None
        self.bound_tid: int | None = None
        self.state = TargetState.LOST
        self.fail_count = 0
        self.pending: dict | None = None
        self.excluded: dict[int, int] = {}
        self.reacquisitions = 0
        self.rejected_candidates = 0
        self.gallery_skips: dict[str, int] = {}
        # Constant-velocity prior (SAMURAI-style). Image pixels / frame.
        self._vel = np.zeros(2, dtype=float)
        self._prev_c: np.ndarray | None = None

    # -- gallery ---------------------------------------------------------------
    def gallery_matrix(self) -> np.ndarray | None:
        rows = ([self.anchor] if self.anchor is not None else []) + list(self.gallery)
        return np.stack(rows) if rows else None

    def _gallery_distance(self, emb: np.ndarray) -> np.ndarray:
        g = self.gallery_matrix()
        if g is None or len(emb) == 0:
            return np.ones(len(emb))
        return cosine_distance_matrix(g, emb).min(axis=0)

    def _clean_reason(self, fr: FrameResult, i: int, d_app: float, margin: float) -> str | None:
        c = self.cfg
        b = fr.boxes[i]
        if fr.scores[i] < c.target_update_min_score:
            return "low_score"
        if not np.all(np.isfinite(fr.embeddings[i])):
            return "no_embedding"
        if b[3] - b[1] < c.target_min_crop_height_px:
            return "small_crop"
        if self.image_wh is not None:
            w, h = self.image_wh
            m = c.target_edge_margin_px
            if b[0] <= m or b[1] <= m or b[2] >= w - m or b[3] >= h - m:
                return "frame_edge"
        if len(fr) > 1:
            ov = iou_matrix(b[None], np.delete(fr.boxes, i, axis=0))[0]
            if ov.max() >= c.appearance_update_overlap_threshold:
                return "overlap"
        if margin < c.target_reacquire_min_margin:
            return "ambiguous"
        if d_app > self.reid_max:
            return "appearance_jump"
        if self.last_gallery_frame is not None and abs(fr.frame_idx - self.last_gallery_frame) < self._gallery_gap:
            return "interval"
        return None

    def _maybe_update_gallery(self, fr: FrameResult, i: int, d_app: float, margin: float):
        why = self._clean_reason(fr, i, d_app, margin)
        if why is None:
            e = fr.embeddings[i].astype(np.float64)
            self.gallery.append(e / max(np.linalg.norm(e), 1e-12))
            self.last_gallery_frame = fr.frame_idx
            return True, None
        self.gallery_skips[why] = self.gallery_skips.get(why, 0) + 1
        return False, why

    # -- core ------------------------------------------------------------------
    def initialize(self, fr: FrameResult, det_index: int) -> TargetFrame:
        b = fr.boxes[det_index].copy()
        e = fr.embeddings[det_index]
        if np.all(np.isfinite(e)):
            self.anchor = e.astype(np.float64) / max(np.linalg.norm(e), 1e-12)
        self.last_box = b
        self.last_seen = fr.frame_idx
        self._vel = np.zeros(2, dtype=float)
        self._prev_c = _box_center(b)
        tid = int(fr.track_ids[det_index])
        self.bound_tid = tid if tid >= 0 else None
        self.state = TargetState.VISIBLE_CONFIDENT
        self._mark_covisible(fr, det_index)
        self._record_impostors(fr, det_index, self._gallery_distance(fr.embeddings))
        return TargetFrame(
            frame_idx=fr.frame_idx, state=self.state, det_index=det_index, bbox=b,
            track_id=self.bound_tid, score=float(fr.scores[det_index]), cost=0.0,
            d_iou=0.0, d_app=0.0, reason="lock",
        )

    def _mark_covisible(self, fr: FrameResult, det_index: int) -> None:
        for j, tid in enumerate(fr.track_ids):
            if j != det_index and tid >= 0 and tid != self.bound_tid:
                self.excluded[int(tid)] = fr.frame_idx

    def _is_excluded(self, tid: int, frame_idx: int) -> bool:
        if tid < 0 or tid == self.bound_tid or tid not in self.excluded:
            return False
        return self._excl_frames is None or abs(frame_idx - self.excluded[tid]) <= self._excl_frames

    def _record_impostors(self, fr: FrameResult, i: int, d_app: np.ndarray) -> None:
        """Appearance distances of clearly separate people seen with the target."""
        if len(fr) < 2:
            return
        ov = iou_matrix(fr.boxes[i][None], fr.boxes)[0]
        for j in range(len(fr)):
            if j != i and ov[j] < self.cfg.appearance_update_overlap_threshold \
                    and np.all(np.isfinite(fr.embeddings[j])):
                self.impostors.append(float(d_app[j]))

    def reacquire_appearance_threshold(self) -> float:
        c = self.cfg
        if not c.target_adaptive_appearance or len(self.impostors) < c.target_impostor_min_samples:
            return self.reid_max
        return float(min(self.reid_max, np.quantile(np.asarray(self.impostors), c.target_impostor_quantile)))

    def expansion_for(self, k: int, mask_visible: bool = False) -> float:
        c = self.cfg
        steps = max(0, (k - 1) // self._expand_every)
        cap = c.target_max_expansion_lost if (k > self._occl_frames or mask_visible) else c.target_max_expansion
        return float(min(cap, c.expansion_initial + c.expansion_step * steps))

    def _unseen_state(self, k: int) -> TargetState:
        return TargetState.OCCLUDED if k <= self._occl_frames else TargetState.LOST

    def _update_motion(self, box, frame_idx: int) -> None:
        c = _box_center(box)
        if self._prev_c is not None and self.last_seen is not None:
            dt = max(1, abs(int(frame_idx) - int(self.last_seen)))
            inst = (c - self._prev_c) / dt
            self._vel = 0.6 * self._vel + 0.4 * inst
        self._prev_c = c

    def _motion_ok(self, box, k: int) -> bool:
        """SAMURAI-style: stay near the predicted location, or overlap last box."""
        if self.last_box is None or box is None:
            return False
        h = max(1.0, float(self.last_box[3] - self.last_box[1]))
        pred = _box_center(self.last_box) + self._vel * max(1, k)
        err = float(np.hypot(*(_box_center(box) - pred))) / h
        iou = float(iou_matrix(
            np.asarray(box, float).reshape(1, 4), self.last_box.reshape(1, 4),
        )[0, 0])
        grow = float(getattr(self.cfg, "mask_motion_err_box_h", 0.6)) * max(1.0, float(k) ** 0.5)
        lim = min(1.15, grow)
        return iou >= 0.12 or err < lim

    def _commit_mask_box(self, f: int, ev: MaskEvidence, k: int, E: float, reason: str) -> TargetFrame:
        box = np.asarray(ev.bbox, dtype=float).reshape(4).copy()
        self._update_motion(box, f)
        self.last_box = box
        self.last_seen = f
        self.fail_count = 0
        self.pending = None
        self.state = TargetState.VISIBLE_CONFIDENT
        return self._decorate(
            TargetFrame(
                f, self.state, bbox=box, track_id=self.bound_tid,
                frames_since_seen=0, expansion=E, reason=reason,
            ),
            ev, risk="confident",
        )

    def _costs(self, fr: FrameResult, expansion: float):
        d_iou = 1.0 - expansion_iou(self.last_box[None], fr.boxes, expansion, self.image_wh)[0]
        d_app = self._gallery_distance(fr.embeddings)
        app = np.where(d_app > self.reid_max, 1.0, d_app)
        cost = harmonic_mean_distance(d_iou, app)
        # Hard spatial gate: never consider a box outside the expanded region.
        cost = np.where(d_iou >= 1.0 - 1e-9, 1.0, cost)
        excluded = np.array([self._is_excluded(int(t), fr.frame_idx) for t in fr.track_ids], dtype=bool)
        cost = np.where(excluded, 1.0, cost)
        return cost, d_iou, d_app, excluded

    def _decorate(self, out: TargetFrame, ev: MaskEvidence | None, coverage=None, risk=None) -> TargetFrame:
        if ev is not None:
            out.sam_active = ev.backend not in ("none",)
            out.sam_mask_available = ev.mask is not None
            out.sam_mask_confidence = ev.score
            out.sam_mask_area = ev.area
        if coverage is not None:
            out.mask_coverage = float(coverage)
        if risk is not None:
            out.risk = risk if isinstance(risk, str) else risk.value
        return out

    def step(self, fr: FrameResult, motion: np.ndarray | None = None, mask: MaskEvidence | None = None) -> TargetFrame:
        """Decide the target in ``fr``. ``motion`` (2x3) maps the previously
        stepped frame's coordinates into this frame's (camera compensation)."""
        c = self.cfg
        f = fr.frame_idx
        ev = mask
        mask_vis = bool(ev is not None and ev.visible and ev.mask is not None)
        if motion is not None and self.last_box is not None:
            self.last_box = warp_boxes(self.last_box[None], motion)[0]
            if self._prev_c is not None:
                stub = np.array([self._prev_c[0], self._prev_c[1], self._prev_c[0] + 1, self._prev_c[1] + 1], float)
                w = warp_boxes(stub[None], motion)[0]
                self._prev_c = np.array([w[0], w[1]], dtype=float)
            if self.pending is not None:
                self.pending["box"] = warp_boxes(self.pending["box"][None], motion)[0]
        k = abs(f - self.last_seen) if self.last_seen is not None else f
        E = self.expansion_for(k, mask_vis)
        if len(fr) == 0:
            self.pending = None
            if (mask_vis and ev is not None and ev.bbox is not None
                    and getattr(c, "mask_emit_without_detection", True)
                    and self._motion_ok(ev.bbox, max(1, k))):
                return self._commit_mask_box(f, ev, k, E, "mask_holds_through_dropout")
            self.state = self._unseen_state(k)
            reason = "sam_propagating" if mask_vis else "target_detector_dropout"
            return self._decorate(
                TargetFrame(f, self.state, frames_since_seen=k, expansion=E, reason=reason),
                ev, risk="full_occlusion" if k <= self._occl_frames else "lost",
            )

        cost, d_iou, d_app, excluded = self._costs(fr, E)
        cov = np.zeros(len(fr))
        miou = np.zeros(len(fr))
        if mask_vis:
            cov = np.array([mask_coverage(ev.mask, b) for b in fr.boxes])
            miou = np.array([mask_box_iou(ev.mask, b) for b in fr.boxes])
            # Mask identity can reopen the spatial gate (SAM-Deep-EIoU).
            reopen = (cov >= c.mask_candidate_min_coverage) | (miou >= c.mask_candidate_min_iou)
            cost = np.where(reopen, harmonic_mean_distance(d_iou, np.clip(d_app, 0, 1)), cost)
        order = np.argsort(cost, kind="stable")
        cands = [
            {"det": int(i), "track_id": int(fr.track_ids[i]), "cost": round(float(cost[i]), 4),
             "d_iou": round(float(d_iou[i]), 4), "d_app": round(float(d_app[i]), 4),
             "excluded": bool(excluded[i]), "mask_coverage": round(float(cov[i]), 4),
             "mask_iou": round(float(miou[i]), 4), "recovery": bool(fr.is_recovery(i)),
             "score": round(float(fr.scores[i]), 3)}
            for i in order[:5]
        ]

        def margin_for(i: int) -> float:
            others = np.delete(cost, i)
            return float((others.min() if len(others) else 1.0) - cost[i])

        def cover(i):
            return float(cov[i]) if i is not None else None

        def frame(state, i=None, reason="", **kw):
            rec = bool(i is not None and fr.is_recovery(i))
            return TargetFrame(
                frame_idx=f, state=state, det_index=None if i is None else int(i),
                bbox=None if i is None or state not in BOXED_STATES else fr.boxes[i].copy(),
                track_id=None if i is None else int(fr.track_ids[i]),
                score=None if i is None else float(fr.scores[i]),
                cost=None if i is None else float(cost[i]),
                d_iou=None if i is None else float(d_iou[i]),
                d_app=None if i is None else float(d_app[i]),
                frames_since_seen=k, expansion=E, reason=reason, candidates=cands,
                recovery=rec, mask_coverage=cover(i), **kw,
            )

        def finish(out, i=None, risk=None):
            rsk = risk
            if rsk is None and i is not None:
                rsk, _ = evaluate_risk(
                    fr, det_index=i, bound_present=bound is not None, margin=out.margin,
                    d_app=out.d_app, reid_max=self.reid_max, last_box=self.last_box, cfg=c, unseen=k,
                )
            return self._decorate(out, ev, cover(i), rsk)

        def accept(i, state, reason):
            m = margin_for(i)
            tid = int(fr.track_ids[i])
            self.bound_tid = tid if tid >= 0 else None
            self._update_motion(fr.boxes[i], f)
            self.last_box = fr.boxes[i].copy()
            self.last_seen = f
            self.fail_count = 0
            self.pending = None
            self.state = state
            self._mark_covisible(fr, i)
            if state == TargetState.VISIBLE_CONFIDENT:
                self._record_impostors(fr, i, d_app)
            upd, skip = (False, "reacquired") if state == TargetState.REACQUIRED else \
                self._maybe_update_gallery(fr, i, float(d_app[i]), m)
            out = frame(state, i, reason, appearance_updated=upd, update_skip=skip)
            out.margin = m
            out.frames_since_seen = 0
            return finish(out, i, "confident")

        bound = None
        if self.bound_tid is not None:
            hit = np.flatnonzero((fr.track_ids == self.bound_tid) & fr.confirmed)
            bound = int(hit[0]) if len(hit) else None

        visible_chain = self.state in (
            TargetState.VISIBLE_CONFIDENT, TargetState.REACQUIRED, TargetState.VISIBLE_AMBIGUOUS
        )

        def mask_supports(i: int) -> bool:
            return mask_vis and (cov[i] >= c.mask_candidate_min_coverage or miou[i] >= c.mask_candidate_min_iou)

        def mask_contradicts(i: int) -> bool:
            if not mask_vis:
                return False
            other = [j for j in range(len(fr)) if j != i and mask_supports(j)]
            return cov[i] <= c.mask_veto_max_coverage and len(other) > 0

        if bound is not None and visible_chain:
            m = margin_for(bound)
            if mask_contradicts(bound):
                self.state = TargetState.VISIBLE_AMBIGUOUS
                out = frame(self.state, bound, "deep_hm_assignment_rejected_by_mask")
                out.margin = m
                return finish(out, bound, "ambiguous")
            app_ok = not np.all(np.isfinite(fr.embeddings[bound])) or d_app[bound] <= self.reid_max
            ok = d_iou[bound] < 1.0 - 1e-9 and app_ok and cost[bound] <= c.assignment_cost_threshold
            from oline_cv.tracking.occlusion_risk import overlap_with_others
            ov = overlap_with_others(fr.boxes, bound)
            if ok and ov >= c.target_occlusion_overlap_threshold:
                holds = mask_supports(bound) or (
                    (not mask_vis) and self._motion_ok(fr.boxes[bound], 1)
                )
                self.gallery_skips["overlap"] = self.gallery_skips.get("overlap", 0) + 1
                if holds:
                    return accept(bound, TargetState.VISIBLE_CONFIDENT, "overlap_identity_holds")
                self.state = TargetState.VISIBLE_AMBIGUOUS
                out = frame(self.state, bound, "target_overlap_detected")
                out.margin = m
                out.update_skip = "overlap"
                return finish(out, bound, "partial_occlusion")
            if ok and m >= c.target_visible_min_margin:
                return accept(bound, TargetState.VISIBLE_CONFIDENT, "deep_hm_confident")
            if ok:
                self.state = TargetState.VISIBLE_AMBIGUOUS
                out = frame(self.state, bound, "close_competitor")
                out.margin = m
                return finish(out, bound, "ambiguous")
            self.fail_count += 1
            if self.fail_count < c.target_identity_fail_frames:
                self.state = TargetState.VISIBLE_AMBIGUOUS
                out = frame(self.state, bound, "identity_check_failed")
                out.margin = m
                return finish(out, bound, "ambiguous")
            self.excluded[self.bound_tid] = f
            self.bound_tid = None
            self.fail_count = 0
            self.state = self._unseen_state(k)
            return finish(frame(self.state, None, "unbound_identity_failed"), None, "lost")

        best = int(order[0])
        if visible_chain and k == 1 and bound is None:
            mb = margin_for(best)
            if (d_iou[best] <= c.low_assignment_cost_threshold and not excluded[best]
                    and d_app[best] <= self.reid_max and mb >= c.target_reacquire_min_margin
                    and not fr.is_recovery(best)):
                return accept(best, TargetState.VISIBLE_CONFIDENT, "continuity")

        app_thr = self.reacquire_appearance_threshold()
        for tf_c in cands:
            tf_c["app_thr"] = round(app_thr, 4)

        def passes(i: int, mi: float, thr: float) -> tuple[bool, str]:
            weak = fr.is_recovery(i)
            if excluded[i]:
                return False, "excluded_covisible"
            if weak and not mask_supports(i):
                return False, "low_confidence_without_identity"
            if mask_vis and not mask_supports(i) and d_app[i] > thr:
                return False, "candidate_failed_mask_gate"
            if (not mask_supports(i)) and d_iou[i] > c.target_reacquire_max_iou_distance:
                return False, "spatial_gate"
            if d_app[i] > thr and not mask_supports(i):
                return False, "candidate_failed_reid"
            if mask_supports(i) and d_app[i] > self.reid_max and d_app[i] > 0.55:
                return False, "candidate_failed_reid"  # ReID strongly contradicts mask
            if cost[i] > c.target_reacquire_max_cost and not mask_supports(i):
                return False, "no_valid_reacquisition"
            if mi < c.target_reacquire_min_margin:
                return False, "candidate_margin_too_small"
            return True, ("reacquired_via_mask_and_reid" if mask_supports(i) else "reacquired_via_reid")

        m = margin_for(best)
        own_thr = self.reid_max if k <= self._occl_frames else app_thr
        if bound is not None:
            ok_b, why_b = passes(bound, margin_for(bound), own_thr)
            if ok_b:
                self.reacquisitions += 1
                return accept(bound, TargetState.REACQUIRED, why_b if mask_supports(bound) else "own_tracklet_refound")

        # Prefer the detection the mask sits on when it qualifies.
        pick = best
        if mask_vis:
            ranked = [int(i) for i in np.argsort(-cov, kind="stable")]
            for i in ranked:
                ok_i, _ = passes(i, margin_for(i), app_thr)
                if ok_i and mask_supports(i):
                    pick = i
                    m = margin_for(i)
                    break
        qualifies, why = passes(pick, m, app_thr)
        if not qualifies:
            if (mask_vis and ev is not None and ev.bbox is not None
                    and getattr(c, "mask_emit_without_detection", True)
                    and self._motion_ok(ev.bbox, max(1, k))):
                snap = None
                for i in (int(j) for j in np.argsort(-cov, kind="stable")):
                    if mask_supports(i) and self._motion_ok(fr.boxes[i], max(1, k)):
                        snap = i
                        break
                if snap is not None:
                    self.reacquisitions += 1
                    return accept(snap, TargetState.REACQUIRED, "reacquired_via_mask")
                return self._commit_mask_box(f, ev, k, E, "mask_identity")
            if cost[pick] < 1.0:
                self.rejected_candidates += 1
            self.pending = None
            self.state = self._unseen_state(k)
            out = frame(self.state, None, why if why else "no_valid_reacquisition")
            out.margin = m
            return finish(out, None, "full_occlusion" if k <= self._occl_frames else "lost")

        tid = int(fr.track_ids[pick])
        p = self.pending
        same = (
            p is not None
            and abs(f - p["frame"]) == 1
            and ((tid >= 0 and p["track_id"] == tid)
                 or (tid < 0 and p["track_id"] < 0
                     and iou_matrix(p["box"][None], fr.boxes[pick][None])[0, 0] > 0.3)
                 or (mask_supports(pick) and p.get("det") == pick))
        )
        count = p["count"] + 1 if same else 1
        self.pending = {"track_id": tid, "box": fr.boxes[pick].copy(), "frame": f, "count": count, "det": pick}
        if count >= c.target_reacquire_confirm_frames:
            self.reacquisitions += 1
            return accept(pick, TargetState.REACQUIRED, why)
        self.state = TargetState.REACQUISITION_CANDIDATE
        out = frame(self.state, pick, f"candidate_{count}")
        out.margin = m
        return finish(out, pick, "ambiguous")

    def summary(self) -> dict:
        return {
            "logical_target_id": self.logical_id,
            "gallery_size": len(self.gallery) + (1 if self.anchor is not None else 0),
            "reacquisitions": self.reacquisitions,
            "rejected_candidates": self.rejected_candidates,
            "gallery_skips": dict(self.gallery_skips),
            "reid_max_distance": self.reid_max,
            "impostor_samples": len(self.impostors),
            "reacquire_appearance_threshold": round(self.reacquire_appearance_threshold(), 4),
        }


def _needs_mask(tf: TargetFrame | None) -> bool:
    if tf is None:
        return True
    return tf.state not in (TargetState.VISIBLE_CONFIDENT,)


def run_target_identity(
    frames: list[FrameResult],
    lock_pos: int,
    lock_det: int,
    cfg,
    fps: float,
    image_wh: tuple[int, int] | None,
    bidirectional: bool = True,
    mask_tracker=None,
) -> tuple[list[TargetFrame | None], dict]:
    """Run the target layer forward (and backward) from the lock frame.

    ``frames`` is the tracker output in frame order; ``lock_pos`` indexes it.
    Frames before the lock are only resolved when ``bidirectional``.
    When ``mask_tracker`` is provided it is seeded at the lock. By default the
    mask is propagated through the whole clip (SAM 2 / SAMURAI), then used as
    identity when Deep HM-SORT association is unsafe.
    """
    out: list[TargetFrame | None] = [None] * len(frames)
    window = max(1, int(round(getattr(cfg, "max_occlusion_window_s", 4.0) * max(fps, 1.0))))
    fwd = LockedTargetIdentity(cfg, fps, image_wh)
    lock_ev = None
    if mask_tracker is not None:
        if getattr(cfg, "sam2_propagate_full_video", True) and hasattr(mask_tracker, "propagate_all"):
            mask_tracker.propagate_all(lock_pos, len(frames))
        lock_ev = mask_tracker.get(lock_pos)
    out[lock_pos] = fwd.initialize(frames[lock_pos], lock_det)
    if lock_ev is not None:
        fwd._decorate(out[lock_pos], lock_ev, risk="confident")
        out[lock_pos].reason = "lock"
    sam_used = 0
    prefilled = bool(
        mask_tracker is not None and getattr(cfg, "sam2_propagate_full_video", True)
    )

    def mask_for(t: int, prev: TargetFrame | None, reverse: bool):
        nonlocal sam_used
        if mask_tracker is None:
            return None
        if prefilled:
            return mask_tracker.get(t)
        if prev is not None and not _needs_mask(prev) and t != lock_pos:
            return mask_tracker.get(t) if mask_tracker.get(t).mask is not None else None
        ev = mask_tracker.get(t)
        if ev.mask is None:
            if reverse:
                mask_tracker.propagate_backward(t, min(window, abs(t - (mask_tracker._last_propagated or t))))
            else:
                start = (mask_tracker._last_propagated or t) + 1
                mask_tracker.propagate_forward(start, min(window, max(1, t - start + 1)))
            ev = mask_tracker.ensure(t)
            sam_used += 1
        return ev

    for t in range(lock_pos + 1, len(frames)):
        ev = mask_for(t, out[t - 1], False)
        out[t] = fwd.step(frames[t], frames[t].camera, ev)
        if out[t].state == TargetState.VISIBLE_CONFIDENT and mask_tracker is not None and out[t].has_box:
            # Keep SAM memory aligned with the last clean box without forcing a box.
            if getattr(cfg, "enable_target_segmentation", False) and out[t].update_skip is None:
                pass
    meta = {"forward": fwd.summary(), "sam_propagate_calls": sam_used}
    if bidirectional and lock_pos > 0:
        bwd = LockedTargetIdentity(cfg, fps, image_wh)
        bwd.initialize(frames[lock_pos], lock_det)
        for t in range(lock_pos - 1, -1, -1):
            cam = frames[t + 1].camera
            ev = mask_for(t, out[t + 1], True)
            out[t] = bwd.step(frames[t], None if cam is None else invert(cam), ev)
        meta["backward"] = bwd.summary()
    counts: dict[str, int] = {}
    for tf in out:
        key = tf.state.value if tf is not None else "UNRESOLVED"
        counts[key] = counts.get(key, 0) + 1
    meta["state_counts"] = counts
    meta["reacquisitions"] = fwd.reacquisitions + (meta.get("backward") or {}).get("reacquisitions", 0)
    return out, meta
