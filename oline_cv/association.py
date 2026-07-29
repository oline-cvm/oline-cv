"""Identity association — separate from detection.

Scores every candidate against a permanent target. Prefer returning no target
over transferring the target ID to another player.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from oline_cv.appearance_ref import FrozenAppearanceRef, crop_torso


class TrackState(str, Enum):
    TRACKED = "TRACKED"
    UNCERTAIN = "UNCERTAIN"
    LOST = "LOST"
    REIDENTIFIED = "REIDENTIFIED"


@dataclass
class CandidateScore:
    frame_idx: int
    track_id: int | None
    det_conf: float
    iou: float
    appearance: float
    motion_dist: float
    motion_score: float
    jersey_sim: float
    size_score: float
    formation_score: float
    weighted: float
    accepted: bool
    reason: str


@dataclass
class AssociationDecision:
    state: TrackState
    confidence: float
    best: CandidateScore | None
    candidates: list[CandidateScore] = field(default_factory=list)
    target_id: int = 1
    botsort_id: int | None = None


@dataclass
class AssociationWeights:
    appearance: float = 0.35
    motion: float = 0.20
    iou: float = 0.20
    jersey: float = 0.12
    size: float = 0.08
    formation: float = 0.05


@dataclass
class AssociationThresholds:
    """None = calibration mode (log only; do not hard-reject on that signal)."""

    min_appearance: float | None = None
    min_jersey: float | None = None
    min_weighted: float | None = None
    min_iou_tracked: float | None = None
    uncertain_weighted: float | None = None


class AssociationLogger:
    def __init__(self, out_dir: str | Path | None):
        self.rows: list[dict] = []
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "crops").mkdir(exist_ok=True)

    def log_candidates(self, scores: list[CandidateScore]) -> None:
        for s in scores:
            self.rows.append(asdict(s))

    def save_debug_image(
        self,
        frame: np.ndarray,
        tag: str,
        frame_idx: int,
        box: np.ndarray | None = None,
        crop: np.ndarray | None = None,
        note: str = "",
    ) -> None:
        if self.out_dir is None:
            return
        img = frame.copy()
        if box is not None:
            b = box.astype(int)
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 80, 255), 2)
        cv2.putText(
            img,
            f"{tag} f{frame_idx} {note}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        path = self.out_dir / f"{frame_idx:05d}_{tag}.jpg"
        cv2.imwrite(str(path), img)
        if crop is not None:
            cv2.imwrite(str(self.out_dir / "crops" / f"{frame_idx:05d}_{tag}.jpg"), crop)

    def flush(self) -> Path | None:
        if self.out_dir is None:
            return None
        csv_path = self.out_dir / "candidate_matches.csv"
        if self.rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                w.writeheader()
                w.writerows(self.rows)
        json_path = self.out_dir / "candidate_matches.json"
        json_path.write_text(json.dumps(self.rows, indent=2), encoding="utf-8")
        return csv_path


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = aa + bb - inter
    return float(inter / union) if union > 1e-6 else 0.0


def _bbox_center(b: np.ndarray) -> np.ndarray:
    return (b[:2] + b[2:]) / 2.0


def _bbox_area(b: np.ndarray) -> float:
    return max(1.0, float(b[2] - b[0]) * float(b[3] - b[1]))


class IdentityAssociator:
    """Match detections to a permanent target without transferring the ID."""

    def __init__(
        self,
        ref: FrozenAppearanceRef,
        *,
        weights: AssociationWeights | None = None,
        thresholds: AssociationThresholds | None = None,
        lost_buffer: int = 45,
        logger: AssociationLogger | None = None,
        reject_wrong_team: bool = True,
        team_jersey_floor: float | None = None,
    ):
        self.ref = ref
        self.weights = weights or AssociationWeights()
        self.thresholds = thresholds or AssociationThresholds()
        self.lost_buffer = int(lost_buffer)
        self.logger = logger or AssociationLogger(None)
        self.reject_wrong_team = reject_wrong_team
        # None = calib: don't reject on team color yet
        self.team_jersey_floor = team_jersey_floor

        self.state = TrackState.TRACKED
        self.confidence = 1.0
        self.lost_frames = 0
        self.botsort_id: int | None = None  # last matched MOT id (may change on re-id only)
        self.locked_botsort_id: int | None = None  # preferred MOT id from initial lock
        self.prev_bbox: np.ndarray | None = None
        self.prev_center: np.ndarray | None = (
            None if ref.formation_xy is None else ref.formation_xy.copy()
        )
        self.velocity = np.zeros(2, dtype=float)
        self.last_decision: AssociationDecision | None = None

    def associate(
        self,
        frame_idx: int,
        frame: np.ndarray,
        boxes: np.ndarray,
        confs: np.ndarray,
        track_ids: np.ndarray | None = None,
    ) -> AssociationDecision:
        pred = (
            self.prev_center + self.velocity
            if self.prev_center is not None
            else self.ref.formation_xy
        )
        prior_box = self.prev_bbox
        prior_area = _bbox_area(prior_box) if prior_box is not None else None
        diag = max(self.ref.lock_diag, 1.0)

        scores: list[CandidateScore] = []
        n = len(confs)
        for i in range(n):
            box = boxes[i].astype(float)
            tid = int(track_ids[i]) if track_ids is not None else None
            crop = crop_torso(frame, box)
            if crop is None:
                scores.append(
                    CandidateScore(
                        frame_idx=frame_idx,
                        track_id=tid,
                        det_conf=float(confs[i]),
                        iou=0.0,
                        appearance=0.0,
                        motion_dist=1e9,
                        motion_score=0.0,
                        jersey_sim=0.0,
                        size_score=0.0,
                        formation_score=0.0,
                        weighted=0.0,
                        accepted=False,
                        reason="no_torso_crop",
                    )
                )
                continue

            app = self.ref.appearance_sim(crop)
            # Blend frozen + recent (recent never replaces frozen)
            app = 0.75 * app + 0.25 * self.ref.recent_sim(crop)
            jer = self.ref.jersey_sim(crop)
            iou = _bbox_iou(prior_box, box) if prior_box is not None else 0.0
            c = _bbox_center(box)
            if pred is None:
                motion_dist = 0.0
                motion_score = 1.0
            else:
                motion_dist = float(np.linalg.norm(c - pred))
                motion_score = max(0.0, 1.0 - motion_dist / (diag * 1.2))

            if prior_area is None:
                size_score = 1.0
            else:
                ar = _bbox_area(box) / prior_area
                size_score = 1.0 - min(1.0, abs(np.log(max(ar, 1e-3))) / 1.2)

            if self.ref.formation_xy is None:
                formation_score = 1.0
            else:
                form_dist = float(np.linalg.norm(c - self.ref.formation_xy))
                formation_score = max(0.0, 1.0 - form_dist / (diag * 2.5))

            w = self.weights
            weighted = (
                w.appearance * app
                + w.motion * motion_score
                + w.iou * iou
                + w.jersey * jer
                + w.size * size_score
                + w.formation * formation_score
            )

            reason = "candidate"
            rejected = False
            # Hard rejects that do not require tuned thresholds
            if self.reject_wrong_team and self.team_jersey_floor is not None:
                if jer < self.team_jersey_floor:
                    rejected = True
                    reason = "wrong_team_jersey"

            thr = self.thresholds
            if not rejected and thr.min_appearance is not None and app < thr.min_appearance:
                rejected = True
                reason = f"appearance_below_{thr.min_appearance:.3f}"
            if not rejected and thr.min_jersey is not None and jer < thr.min_jersey:
                rejected = True
                reason = f"jersey_below_{thr.min_jersey:.3f}"
            if not rejected and thr.min_weighted is not None and weighted < thr.min_weighted:
                rejected = True
                reason = f"weighted_below_{thr.min_weighted:.3f}"
            if (
                not rejected
                and thr.min_iou_tracked is not None
                and self.state in (TrackState.TRACKED, TrackState.UNCERTAIN, TrackState.REIDENTIFIED)
                and prior_box is not None
                and iou < thr.min_iou_tracked
                and app < (thr.min_appearance or 0.99)
            ):
                # Only apply IoU floor while actively tracked and appearance not rock-solid
                rejected = True
                reason = f"iou_below_{thr.min_iou_tracked:.3f}"

            scores.append(
                CandidateScore(
                    frame_idx=frame_idx,
                    track_id=tid,
                    det_conf=float(confs[i]),
                    iou=float(iou),
                    appearance=float(app),
                    motion_dist=float(motion_dist),
                    motion_score=float(motion_score),
                    jersey_sim=float(jer),
                    size_score=float(size_score),
                    formation_score=float(formation_score),
                    weighted=float(weighted),
                    accepted=False,  # set after selection
                    reason=reason if rejected else "scored",
                )
            )
            if rejected:
                scores[-1].accepted = False

        # Eligible = not hard-rejected
        eligible = [s for s in scores if s.reason in ("scored", "candidate")]
        # Prefer the LOCKED BoT-SORT id — never silently transfer to a higher-scoring neighbor
        preferred_id = self.locked_botsort_id if self.locked_botsort_id is not None else self.botsort_id
        chosen: CandidateScore | None = None
        if preferred_id is not None:
            same = [s for s in eligible if s.track_id == preferred_id]
            if same:
                cand = max(same, key=lambda s: s.weighted)
                # Keep preferred id unless appearance collapses (occlusion / wrong body under id)
                thr_app = self.thresholds.min_appearance
                if thr_app is None or cand.appearance >= thr_app:
                    chosen = cand
                else:
                    cand.reason = "preferred_id_appearance_fail"
                    cand.accepted = False

        if chosen is None and eligible:
            # Re-id path: highest weighted, not closest; require clear margin
            ordered = sorted(eligible, key=lambda s: s.weighted, reverse=True)
            top = ordered[0]
            if len(ordered) >= 2 and (ordered[0].weighted - ordered[1].weighted) < 0.06:
                for s in ordered[:2]:
                    if s.reason == "scored":
                        s.reason = "ambiguous_top2"
                        s.accepted = False
                chosen = None
            else:
                # While actively tracked, do not adopt a different MOT id without LOST first
                if (
                    preferred_id is not None
                    and self.state in (TrackState.TRACKED, TrackState.UNCERTAIN, TrackState.REIDENTIFIED)
                    and top.track_id != preferred_id
                    and self.lost_frames < max(5, self.lost_buffer // 6)
                ):
                    top.reason = "blocked_id_transfer"
                    top.accepted = False
                    chosen = None
                else:
                    # Strict re-id: need strong appearance vs frozen template
                    app_floor = self.thresholds.min_appearance
                    if app_floor is None:
                        app_floor = max(0.88, float(self.ref.self_similarity) - 0.12)
                    if top.appearance < app_floor:
                        top.reason = "reid_appearance_fail"
                        top.accepted = False
                        chosen = None
                    else:
                        chosen = top

        if chosen is not None:
            for i, s in enumerate(scores):
                if s is chosen:
                    s.accepted = True
                    s.reason = "accepted"
                elif s.reason == "scored":
                    s.reason = "rejected_not_best"
                    if s.weighted >= chosen.weighted - 0.08:
                        self.logger.save_debug_image(
                            frame,
                            "reject",
                            frame_idx,
                            box=boxes[i],
                            note=f"w={s.weighted:.2f}",
                        )

        # Log after accept/reject flags are finalized
        self.logger.log_candidates(scores)

        prev_state = self.state
        if chosen is None:
            self.lost_frames += 1
            self.velocity *= 0.85
            if self.lost_frames == 1 or prev_state != TrackState.LOST:
                self.logger.save_debug_image(frame, "LOST", frame_idx, box=prior_box)
            if self.lost_frames > self.lost_buffer:
                # stay LOST; keep target_id; do not adopt anyone
                pass
            self.state = TrackState.LOST
            self.confidence = 0.0
            decision = AssociationDecision(
                state=self.state,
                confidence=self.confidence,
                best=None,
                candidates=scores,
                target_id=self.ref.target_id,
                botsort_id=self.botsort_id,
            )
            self.last_decision = decision
            return decision

        # Accept chosen — reconnect spatial state; never change target_id
        idx = None
        for i, s in enumerate(scores):
            if s is chosen:
                idx = i
                break
        box = boxes[idx].astype(float)
        crop = crop_torso(frame, box)
        c = _bbox_center(box)

        if prev_state == TrackState.LOST:
            self.state = TrackState.REIDENTIFIED
            self.logger.save_debug_image(
                frame, "REIDENTIFIED", frame_idx, box=box, crop=crop, note=f"w={chosen.weighted:.2f}"
            )
        else:
            unc = self.thresholds.uncertain_weighted
            if unc is not None and chosen.weighted < unc:
                self.state = TrackState.UNCERTAIN
            else:
                self.state = TrackState.TRACKED

        if self.botsort_id is not None and chosen.track_id is not None and chosen.track_id != self.botsort_id:
            self.logger.save_debug_image(
                frame,
                "id_change",
                frame_idx,
                box=box,
                crop=crop,
                note=f"{self.botsort_id}->{chosen.track_id}",
            )
        self.botsort_id = chosen.track_id
        # Bind locked MOT id once. Rebind only after sustained LOST + strong appearance.
        if self.locked_botsort_id is None:
            self.locked_botsort_id = chosen.track_id
        elif (
            prev_state == TrackState.LOST
            and chosen.track_id is not None
            and chosen.track_id != self.locked_botsort_id
            and self.lost_frames >= max(8, self.lost_buffer // 4)
            and chosen.appearance >= max(0.90, float(self.ref.self_similarity) - 0.10)
        ):
            self.locked_botsort_id = chosen.track_id
            self.logger.save_debug_image(
                frame,
                "lock_id_rebind",
                frame_idx,
                box=box,
                crop=crop,
                note=f"locked->{chosen.track_id}",
            )

        if self.prev_center is not None:
            measured_v = c - self.prev_center
            speed = float(np.linalg.norm(measured_v))
            vmax = diag * 0.25
            if speed > vmax:
                measured_v = measured_v * (vmax / speed)
            self.velocity = 0.7 * self.velocity + 0.3 * measured_v
        self.prev_center = c
        self.prev_bbox = box
        self.lost_frames = 0
        self.confidence = float(chosen.weighted)

        if crop is not None:
            occluded = chosen.iou < 0.15 and chosen.appearance < 0.85
            self.ref.maybe_update_recent(
                crop,
                det_conf=chosen.det_conf,
                occluded=occluded,
                min_conf=0.55,
                min_frozen_sim=max(0.75, self.ref.self_similarity * 0.95),
            )

        decision = AssociationDecision(
            state=self.state,
            confidence=self.confidence,
            best=chosen,
            candidates=scores,
            target_id=self.ref.target_id,
            botsort_id=self.botsort_id,
        )
        self.last_decision = decision
        return decision

    def recommend_thresholds(self) -> dict[str, float]:
        """Suggest thresholds from logged scores + frozen self-similarity (no guessing)."""
        rows = self.logger.rows
        base = {
            "self_similarity": round(float(self.ref.self_similarity), 3),
            "n_logged": len(rows),
        }
        if not rows:
            return {**base, "note": "no_rows"}

        accepted = [r for r in rows if r.get("accepted")]
        # Proxy true positives: early frames with high IoU to prior (pre-contact)
        early = [
            r
            for r in rows
            if int(r.get("frame_idx", 9999)) <= 45 and float(r.get("iou", 0)) >= 0.5
        ]
        if accepted:
            pos = accepted
        elif early:
            pos = early
        else:
            pos = []

        others = [
            r
            for r in rows
            if (not r.get("accepted"))
            and r.get("reason")
            not in ("no_torso_crop", "accepted")
            and float(r.get("iou", 0)) < 0.25
        ]

        def pct(xs: list[float], p: float) -> float | None:
            if not xs:
                return None
            return float(np.percentile(xs, p))

        pos_app = [float(r["appearance"]) for r in pos]
        pos_w = [float(r["weighted"]) for r in pos]
        pos_j = [float(r["jersey_sim"]) for r in pos]
        neg_app = [float(r["appearance"]) for r in others]

        # Appearance floor: below typical positives, above strong negatives when separable
        min_app = pct(pos_app, 10)
        if min_app is None:
            min_app = float(self.ref.self_similarity) * 0.9
        neg90 = pct(neg_app, 90)
        if neg90 is not None and neg90 < min_app:
            min_app = (min_app + neg90) / 2.0
        # Never use near-perfect pre-snap self-sim as the live floor
        min_app = min(min_app, float(self.ref.self_similarity) - 0.02)
        min_app = max(0.50, min_app)

        out = {
            **base,
            "min_appearance": round(float(min_app), 3),
            "min_jersey": round(float(pct(pos_j, 10) or 0.5), 3),
            "min_weighted": round(float(pct(pos_w, 10) or 0.4), 3),
            "uncertain_weighted": round(float(pct(pos_w, 25) or 0.5), 3),
            "n_accepted": len(accepted),
            "n_pos_proxy": len(pos),
            "n_neg_proxy": len(others),
            "pos_appearance_p10": None if not pos_app else round(float(pct(pos_app, 10)), 3),
            "neg_appearance_p90": None if neg90 is None else round(float(neg90), 3),
        }
        return out
