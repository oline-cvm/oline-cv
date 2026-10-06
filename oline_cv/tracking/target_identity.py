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


class TargetState(str, Enum):
    VISIBLE_CONFIDENT = "VISIBLE_CONFIDENT"
    VISIBLE_AMBIGUOUS = "VISIBLE_AMBIGUOUS"
    OCCLUDED = "OCCLUDED"
    LOST = "LOST"
    REACQUISITION_CANDIDATE = "REACQUISITION_CANDIDATE"
    REACQUIRED = "REACQUIRED"


BOXED_STATES = (TargetState.VISIBLE_CONFIDENT, TargetState.REACQUIRED)


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

    def expansion_for(self, k: int) -> float:
        c = self.cfg
        steps = max(0, (k - 1) // self._expand_every)
        return float(min(c.target_max_expansion, c.expansion_initial + c.expansion_step * steps))

    def _unseen_state(self, k: int) -> TargetState:
        return TargetState.OCCLUDED if k <= self._occl_frames else TargetState.LOST

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

    def step(self, fr: FrameResult, motion: np.ndarray | None = None) -> TargetFrame:
        """Decide the target in ``fr``. ``motion`` (2x3) maps the previously
        stepped frame's coordinates into this frame's (camera compensation)."""
        c = self.cfg
        f = fr.frame_idx
        if motion is not None:
            self.last_box = warp_boxes(self.last_box[None], motion)[0]
            if self.pending is not None:
                self.pending["box"] = warp_boxes(self.pending["box"][None], motion)[0]
        k = abs(f - self.last_seen)
        E = self.expansion_for(k)
        if len(fr) == 0:
            self.pending = None
            self.state = self._unseen_state(k)
            return TargetFrame(f, self.state, frames_since_seen=k, expansion=E, reason="no_detections")

        cost, d_iou, d_app, excluded = self._costs(fr, E)
        order = np.argsort(cost, kind="stable")
        cands = [
            {"det": int(i), "track_id": int(fr.track_ids[i]), "cost": round(float(cost[i]), 4),
             "d_iou": round(float(d_iou[i]), 4), "d_app": round(float(d_app[i]), 4),
             "excluded": bool(excluded[i])}
            for i in order[:3]
        ]

        def margin_for(i: int) -> float:
            others = np.delete(cost, i)
            return float((others.min() if len(others) else 1.0) - cost[i])

        def frame(state, i=None, reason="", **kw):
            return TargetFrame(
                frame_idx=f, state=state, det_index=None if i is None else int(i),
                bbox=None if i is None or state not in BOXED_STATES else fr.boxes[i].copy(),
                track_id=None if i is None else int(fr.track_ids[i]),
                score=None if i is None else float(fr.scores[i]),
                cost=None if i is None else float(cost[i]),
                d_iou=None if i is None else float(d_iou[i]),
                d_app=None if i is None else float(d_app[i]),
                frames_since_seen=k, expansion=E, reason=reason, candidates=cands, **kw,
            )

        def accept(i, state, reason):
            m = margin_for(i)
            tid = int(fr.track_ids[i])
            self.bound_tid = tid if tid >= 0 else None
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
            return out

        bound = None
        if self.bound_tid is not None:
            hit = np.flatnonzero((fr.track_ids == self.bound_tid) & fr.confirmed)
            bound = int(hit[0]) if len(hit) else None

        visible_chain = self.state in (
            TargetState.VISIBLE_CONFIDENT, TargetState.REACQUIRED, TargetState.VISIBLE_AMBIGUOUS
        )
        if bound is not None and visible_chain:
            # Visible-target protection: follow the bound tracklet; a nearby
            # box can only make the frame ambiguous, never take the target.
            m = margin_for(bound)
            app_ok = not np.all(np.isfinite(fr.embeddings[bound])) or d_app[bound] <= self.reid_max
            ok = d_iou[bound] < 1.0 - 1e-9 and app_ok and cost[bound] <= c.assignment_cost_threshold
            if ok and m >= c.target_visible_min_margin:
                return accept(bound, TargetState.VISIBLE_CONFIDENT, "bound_track")
            if ok:
                self.state = TargetState.VISIBLE_AMBIGUOUS
                out = frame(self.state, bound, "close_competitor")
                out.margin = m
                return out
            self.fail_count += 1
            if self.fail_count < c.target_identity_fail_frames:
                self.state = TargetState.VISIBLE_AMBIGUOUS
                out = frame(self.state, bound, "identity_check_failed")
                out.margin = m
                return out
            # Bound tracklet no longer looks like the target: drop it.
            self.excluded[self.bound_tid] = f
            self.bound_tid = None
            self.fail_count = 0
            self.state = self._unseen_state(k)
            return frame(self.state, None, "unbound_identity_failed")

        # Visible last frame but no bound tracklet in this one (the detection
        # has no tracker id yet): continue only on tight frame-to-frame
        # overlap with a clear margin, as the tracker's IoU stage would.
        best = int(order[0])
        if visible_chain and k == 1 and bound is None:
            mb = margin_for(best)
            if (d_iou[best] <= c.low_assignment_cost_threshold and not excluded[best]
                    and d_app[best] <= self.reid_max and mb >= c.target_reacquire_min_margin):
                return accept(best, TargetState.VISIBLE_CONFIDENT, "continuity")

        # Not visible: conservative reacquisition.
        app_thr = self.reacquire_appearance_threshold()
        for tf_c in cands:
            tf_c["app_thr"] = round(app_thr, 4)

        def passes(i: int, mi: float, thr: float) -> bool:
            return bool(
                cost[i] < 1.0
                and cost[i] <= c.target_reacquire_max_cost
                and d_iou[i] <= c.target_reacquire_max_iou_distance
                and d_app[i] <= thr
                and mi >= c.target_reacquire_min_margin
            )

        m = margin_for(best)
        # The target's own tracklet re-found after a short occlusion carries
        # the tracker's association as independent evidence: static gate.
        own_thr = self.reid_max if k <= self._occl_frames else app_thr
        if bound is not None and passes(bound, margin_for(bound), own_thr):
            self.reacquisitions += 1
            return accept(bound, TargetState.REACQUIRED, "own_tracklet_refound")
        qualifies = passes(best, m, app_thr)
        if not qualifies:
            if cost[best] < 1.0:
                self.rejected_candidates += 1
            self.pending = None
            self.state = self._unseen_state(k)
            out = frame(self.state, None, "no_qualified_candidate")
            out.margin = m
            return out

        tid = int(fr.track_ids[best])
        p = self.pending
        same = (
            p is not None
            and abs(f - p["frame"]) == 1
            and ((tid >= 0 and p["track_id"] == tid)
                 or (tid < 0 and p["track_id"] < 0
                     and iou_matrix(p["box"][None], fr.boxes[best][None])[0, 0] > 0.3))
        )
        count = p["count"] + 1 if same else 1
        self.pending = {"track_id": tid, "box": fr.boxes[best].copy(), "frame": f, "count": count}
        if count >= c.target_reacquire_confirm_frames:
            self.reacquisitions += 1
            return accept(best, TargetState.REACQUIRED, f"confirmed_{count}_frames")
        self.state = TargetState.REACQUISITION_CANDIDATE
        out = frame(self.state, best, f"candidate_{count}")
        out.margin = m
        return out

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


def run_target_identity(
    frames: list[FrameResult],
    lock_pos: int,
    lock_det: int,
    cfg,
    fps: float,
    image_wh: tuple[int, int] | None,
    bidirectional: bool = True,
) -> tuple[list[TargetFrame | None], dict]:
    """Run the target layer forward (and backward) from the lock frame.

    ``frames`` is the tracker output in frame order; ``lock_pos`` indexes it.
    Frames before the lock are only resolved when ``bidirectional``.
    """
    out: list[TargetFrame | None] = [None] * len(frames)
    fwd = LockedTargetIdentity(cfg, fps, image_wh)
    out[lock_pos] = fwd.initialize(frames[lock_pos], lock_det)
    for t in range(lock_pos + 1, len(frames)):
        out[t] = fwd.step(frames[t], frames[t].camera)
    meta = {"forward": fwd.summary()}
    if bidirectional and lock_pos > 0:
        bwd = LockedTargetIdentity(cfg, fps, image_wh)
        bwd.initialize(frames[lock_pos], lock_det)
        for t in range(lock_pos - 1, -1, -1):
            cam = frames[t + 1].camera
            out[t] = bwd.step(frames[t], None if cam is None else invert(cam))
        meta["backward"] = bwd.summary()
    counts: dict[str, int] = {}
    for tf in out:
        key = tf.state.value if tf is not None else "UNRESOLVED"
        counts[key] = counts.get(key, 0) + 1
    meta["state_counts"] = counts
    meta["reacquisitions"] = fwd.reacquisitions + (meta.get("backward") or {}).get("reacquisitions", 0)
    return out, meta
