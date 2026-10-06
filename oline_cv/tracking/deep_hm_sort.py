"""Deep HM-SORT: a general multi-object tracker for sports film.

Independent implementation after:

* Deep HM-SORT — Gran-Henriksen et al., "Deep HM-SORT: Enhancing
  Multi-Object Tracking in Sports with Deep Features, Harmonic Mean, and
  Expansion IOU", arXiv:2406.12081.
* Deep-EIoU — Huang et al., "Iterative Scale-Up ExpansionIoU and Deep
  Features Association for Multi-Object Tracking in Sports", WACV Workshops
  2024.

Per frame:

1. High-confidence detections are matched to active and lost tracklets with
   cost ``H(d_iou, d_app)`` (harmonic mean of the Expansion-IoU distance and
   the appearance distance), over iteratively larger expansion scales.
   Lost tracklets only take part in the first pass.
2. Low-confidence detections are matched to the still-unmatched active
   tracklets on Expansion IoU alone.
3. Tentative tracklets are confirmed by a second match or dropped.
4. Remaining high-confidence detections start new tentative tracklets.

Confirmed tracklets are never deleted: once unmatched they go LOST, then
HISTORICAL, and stay eligible for re-identification for the rest of the
video. Track IDs only ever increase.

There is no Kalman prediction in the cost (as in Deep-EIoU, matching uses the
last observed box). OLINE additions: optional camera-motion compensation
moves last boxes with the camera (``update(..., camera=M)``); a tracklet's
appearance is not refreshed from a box that overlaps another detection; and
the active/lost duplicate pruning of ByteTrack-style trackers is not applied
(it would delete persistent tracklets).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from oline_cv.tracking.costs import (
    cosine_distance_matrix,
    gated_hm_cost,
    linear_assignment,
    normalize_rows,
)
from oline_cv.tracking.camera_motion import warp_boxes
from oline_cv.tracking.expansion_iou import expansion_iou, expansion_schedule, iou_matrix


class TrackletState(str, Enum):
    TENTATIVE = "TENTATIVE"
    ACTIVE = "ACTIVE"
    REACTIVATED = "REACTIVATED"  # active again this frame after being lost
    LOST = "LOST"
    HISTORICAL = "HISTORICAL"
    REMOVED = "REMOVED"  # only tentative tracklets, or keep_all_tracklets=False


_VISIBLE = (TrackletState.ACTIVE, TrackletState.REACTIVATED)
_DORMANT = (TrackletState.LOST, TrackletState.HISTORICAL)


@dataclass
class Tracklet:
    track_id: int
    box: np.ndarray  # last observed xyxy
    score: float
    start_frame: int
    last_frame: int
    state: TrackletState = TrackletState.TENTATIVE
    confirmed: bool = False
    feature: np.ndarray | None = None  # EMA appearance, L2-normalised
    hits: int = 1
    reactivations: int = 0
    feature_updates: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=64))

    def summary(self) -> dict:
        return {
            "track_id": self.track_id,
            "state": self.state.value,
            "start_frame": self.start_frame,
            "last_frame": self.last_frame,
            "hits": self.hits,
            "reactivations": self.reactivations,
        }


@dataclass
class TrackObservation:
    """One detection assigned to a tracklet in one frame."""

    track_id: int
    det_index: int  # index into FrameResult.boxes
    state: TrackletState
    confirmed: bool
    reactivated: bool
    stage: str
    cost: float | None = None
    d_iou: float | None = None
    d_app: float | None = None
    appearance_updated: bool = False


@dataclass
class FrameResult:
    """Tracker output for one frame (only detections above detector_threshold)."""

    frame_idx: int
    boxes: np.ndarray  # (N, 4)
    scores: np.ndarray  # (N,)
    embeddings: np.ndarray  # (N, D), NaN rows = no appearance
    track_ids: np.ndarray  # (N,) int, -1 = unassigned
    confirmed: np.ndarray  # (N,) bool — track_ids[i] belongs to a confirmed tracklet
    source_index: np.ndarray  # (N,) index into the detector's raw output
    observations: list[TrackObservation] = field(default_factory=list)
    camera: np.ndarray | None = None  # 2x3, previous frame -> this frame

    def __len__(self) -> int:
        return int(len(self.scores))

    def visible_track_ids(self) -> list[int]:
        return [int(t) for t, c in zip(self.track_ids, self.confirmed) if t >= 0 and c]


def _feature_matrix(tracks: list[Tracklet], dim: int) -> np.ndarray:
    m = np.full((len(tracks), dim), np.nan)
    for i, t in enumerate(tracks):
        if t.feature is not None and t.feature.shape[0] == dim:
            m[i] = t.feature
    return m


class DeepHMSort:
    """Online tracker; call ``update`` once per frame in order."""

    def __init__(self, cfg, fps: float = 30.0, image_wh: tuple[int, int] | None = None) -> None:
        self.cfg = cfg
        self.fps = max(float(fps), 1.0)
        self.image_wh = image_wh
        self._next_id = 1
        self.tracks: dict[int, Tracklet] = {}  # every confirmed or tentative tracklet
        self.removed_tentative = 0
        self._first_frame: int | None = None
        self._schedule = expansion_schedule(
            cfg.expansion_initial, cfg.expansion_step, cfg.expansion_iterations
        )
        self._hist_frames = max(1, int(round(cfg.historical_after_s * self.fps)))
        self._max_lost = (
            None if cfg.max_lost_s is None else max(1, int(round(cfg.max_lost_s * self.fps)))
        )

    # -- helpers -------------------------------------------------------------
    def _cost(self, tracks, dets, boxes, emb, expansion, use_appearance=True):
        tb = np.array([t.box for t in tracks], dtype=np.float64)
        db = boxes[dets]
        d_iou = 1.0 - expansion_iou(tb, db, expansion, self.image_wh)
        if use_appearance:
            d_app = cosine_distance_matrix(_feature_matrix(tracks, emb.shape[1]), emb[dets])
            cost = gated_hm_cost(
                d_iou,
                d_app,
                appearance_threshold=self.cfg.appearance_cost_threshold,
                proximity_threshold=self.cfg.proximity_threshold,
            )
        else:
            d_app = np.ones_like(d_iou)
            cost = d_iou
        return cost, d_iou, d_app

    def _update_feature(self, t: Tracklet, f: np.ndarray) -> bool:
        if f is None or not np.all(np.isfinite(f)):
            return False
        if t.feature is None or t.feature.shape != f.shape:
            t.feature = f.copy()
        else:
            a = self.cfg.appearance_ema_alpha
            v = a * t.feature + (1.0 - a) * f
            n = np.linalg.norm(v)
            if n <= 1e-12:
                return False
            t.feature = v / n
        t.feature_updates += 1
        return True

    def _apply(self, t, d, frame_idx, boxes, scores, emb, clear, stage, c=None, di=None, da=None,
               update_feature=True):
        was_dormant = t.state in _DORMANT
        t.box = boxes[d].copy()
        t.score = float(scores[d])
        t.last_frame = frame_idx
        t.hits += 1
        t.history.append((frame_idx, t.box.copy()))
        upd = bool(update_feature and clear[d] and self._update_feature(t, emb[d]))
        if was_dormant:
            t.state = TrackletState.REACTIVATED
            t.reactivations += 1
        else:
            t.state = TrackletState.ACTIVE
        return TrackObservation(
            track_id=t.track_id, det_index=int(d), state=t.state, confirmed=t.confirmed,
            reactivated=was_dormant, stage=stage,
            cost=None if c is None else float(c), d_iou=None if di is None else float(di),
            d_app=None if da is None else float(da), appearance_updated=upd,
        )

    # -- main ----------------------------------------------------------------
    def update(self, frame_idx: int, boxes, scores, embeddings=None, camera=None) -> FrameResult:
        """Associate one frame. ``camera`` (2x3, previous -> this frame) moves
        every tracklet's last box into this frame's coordinates first."""
        cfg = self.cfg
        if camera is not None:
            for t in self.tracks.values():
                t.box = warp_boxes(t.box[None], camera)[0]
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        n_raw = len(scores)
        if embeddings is None or np.asarray(embeddings).size == 0:
            emb_raw = np.full((n_raw, 1), np.nan)
        else:
            emb_raw = np.asarray(embeddings, dtype=np.float64).reshape(n_raw, -1)
        valid = (
            np.isfinite(scores)
            & np.all(np.isfinite(boxes), axis=1)
            & (boxes[:, 2] > boxes[:, 0])
            & (boxes[:, 3] > boxes[:, 1])
            & (scores >= cfg.detector_threshold)
        )
        src = np.flatnonzero(valid)
        boxes, scores = boxes[src], scores[src]
        emb = normalize_rows(emb_raw[src]) if len(src) else emb_raw[src]
        n = len(scores)
        if self._first_frame is None:
            self._first_frame = frame_idx

        for t in self.tracks.values():
            if t.state == TrackletState.REACTIVATED:
                t.state = TrackletState.ACTIVE

        # Appearance refresh only from boxes that do not overlap a neighbour.
        if n > 1:
            ov = iou_matrix(boxes, boxes)
            np.fill_diagonal(ov, 0.0)
            clear = ov.max(axis=1) < cfg.appearance_update_overlap_threshold
        else:
            clear = np.ones(n, dtype=bool)

        track_ids = np.full(n, -1, dtype=int)
        observations: list[TrackObservation] = []
        high = [i for i in range(n) if scores[i] >= cfg.high_confidence_threshold]
        low = [i for i in range(n) if scores[i] < cfg.high_confidence_threshold]

        live = [t for t in self.tracks.values() if t.state != TrackletState.REMOVED]
        confirmed_active = [t for t in live if t.confirmed and t.state in _VISIBLE]
        dormant = [t for t in live if t.confirmed and t.state in _DORMANT]
        tentative = [t for t in live if not t.confirmed]

        def record(obs: TrackObservation) -> None:
            track_ids[obs.det_index] = obs.track_id
            observations.append(obs)

        # Stage 1: iterative Expansion-IoU + appearance, harmonic mean cost.
        pool = confirmed_active + dormant
        rem = list(high)
        for e in self._schedule:
            if not pool or not rem:
                break
            cost, d_iou, d_app = self._cost(pool, rem, boxes, emb, e)
            matches, ut, ud = linear_assignment(cost, cfg.assignment_cost_threshold)
            for r, c in matches:
                record(self._apply(pool[r], rem[c], frame_idx, boxes, scores, emb, clear,
                                   f"high@{e:g}", cost[r, c], d_iou[r, c], d_app[r, c]))
            pool = [pool[r] for r in ut if pool[r].state in _VISIBLE]
            rem = [rem[c] for c in ud]

        # Stage 2: low-confidence detections, Expansion IoU only.
        pool = [t for t in pool if t.state in _VISIBLE]
        if pool and low:
            cost, d_iou, _ = self._cost(pool, low, boxes, emb, cfg.low_stage_expansion,
                                        use_appearance=False)
            matches, ut, _ = linear_assignment(cost, cfg.low_assignment_cost_threshold)
            for r, c in matches:
                record(self._apply(pool[r], low[c], frame_idx, boxes, scores, emb, clear,
                                   "low", cost[r, c], d_iou[r, c], None, update_feature=False))
            pool = [pool[r] for r in ut]
        for t in pool:
            t.state = TrackletState.LOST

        # Stage 3: tentative tracklets need a second match to be confirmed.
        if tentative and rem:
            cost, d_iou, d_app = self._cost(tentative, rem, boxes, emb, cfg.low_stage_expansion)
            matches, ut, ud = linear_assignment(cost, cfg.unconfirmed_assignment_cost_threshold)
            for r, c in matches:
                tentative[r].confirmed = True
                record(self._apply(tentative[r], rem[c], frame_idx, boxes, scores, emb, clear,
                                   "confirm", cost[r, c], d_iou[r, c], d_app[r, c]))
            leftover = [tentative[r] for r in ut]
            rem = [rem[c] for c in ud]
        else:
            leftover = tentative
        for t in leftover:
            t.state = TrackletState.REMOVED
            del self.tracks[t.track_id]
            self.removed_tentative += 1

        # Stage 4: new tracklets.
        first = frame_idx == self._first_frame
        for d in rem:
            if scores[d] < cfg.new_track_threshold:
                continue
            t = Tracklet(
                track_id=self._next_id, box=boxes[d].copy(), score=float(scores[d]),
                start_frame=frame_idx, last_frame=frame_idx,
                state=TrackletState.ACTIVE if first else TrackletState.TENTATIVE,
                confirmed=first,
            )
            self._next_id += 1
            t.history.append((frame_idx, t.box.copy()))
            upd = bool(clear[d] and self._update_feature(t, emb[d]))
            self.tracks[t.track_id] = t
            record(TrackObservation(
                track_id=t.track_id, det_index=int(d), state=t.state, confirmed=t.confirmed,
                reactivated=False, stage="new", appearance_updated=upd,
            ))

        # Dormant bookkeeping: LOST -> HISTORICAL; optional expiry.
        for t in list(self.tracks.values()):
            if t.state not in _DORMANT:
                continue
            gap = frame_idx - t.last_frame
            if self._max_lost is not None and not cfg.keep_all_tracklets and gap > self._max_lost:
                t.state = TrackletState.REMOVED
                del self.tracks[t.track_id]
            elif gap > self._hist_frames:
                t.state = TrackletState.HISTORICAL

        confirmed = np.array(
            [tid >= 0 and self.tracks.get(int(tid)) is not None and self.tracks[int(tid)].confirmed
             for tid in track_ids], dtype=bool,
        )
        return FrameResult(
            frame_idx=frame_idx, boxes=boxes, scores=scores,
            embeddings=emb if emb.ndim == 2 else emb.reshape(n, -1),
            track_ids=track_ids, confirmed=confirmed, source_index=src,
            observations=observations, camera=None if camera is None else np.asarray(camera, float),
        )

    def visible(self) -> list[Tracklet]:
        """Tracklets matched in the latest frame (never stale LOST tracks)."""
        return [t for t in self.tracks.values() if t.confirmed and t.state in _VISIBLE]

    def summary(self) -> dict:
        states: dict[str, int] = {}
        for t in self.tracks.values():
            states[t.state.value] = states.get(t.state.value, 0) + 1
        return {
            "tracklets": len(self.tracks),
            "next_id": self._next_id,
            "removed_tentative": self.removed_tentative,
            "states": states,
            "reactivations": int(sum(t.reactivations for t in self.tracks.values())),
        }
