"""YOLO-pose detection + separate identity association.

Detection: Ultralytics YOLO-pose with BoT-SORT (GMC + optional ReID features).
Identity: permanent target_id + frozen multi-frame appearance (classical, no NN train).
Prefer LOST (no target) over transferring identity to another player.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from oline_cv.appearance_ref import build_frozen_appearance, crop_torso
from oline_cv.association import (
    AssociationLogger,
    AssociationThresholds,
    AssociationWeights,
    IdentityAssociator,
    TrackState,
)
from oline_cv.config import AnalysisConfig, KEYPOINT_NAMES
from oline_cv.geometry import hip_mid
from oline_cv.ol_select import lock_ol_from_frames


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


_load_dotenv()

BOTSORT_YAML = str(Path(__file__).resolve().parent / "trackers" / "botsort_oline.yaml")
PERMANENT_TARGET_ID = 1


@dataclass
class FramePose:
    frame_idx: int
    timestamp_ms: float
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    bbox_xyxy: np.ndarray | None
    person_confidence: float
    low_confidence: bool
    usable: bool
    track_state: str = TrackState.LOST.value
    track_confidence: float = 0.0
    track_id: int | None = None  # BoT-SORT id (may change); logical target is always permanent
    target_id: int = PERMANENT_TARGET_ID


def _bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(float(a[0]), float(b[0]))
    y0 = max(float(a[1]), float(b[1]))
    x1 = min(float(a[2]), float(b[2]))
    y1 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-6 else 0.0


def _bbox_center(b: np.ndarray) -> np.ndarray:
    return (b[:2] + b[2:]) / 2.0


def _bbox_diag(b: np.ndarray) -> float:
    return float(np.linalg.norm(b[2:] - b[:2]))


class PoseTracker:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._apply_env_overrides()

        model_name = os.environ.get("OLINE_POSE_MODEL") or config.pose_model or "yolov8m-pose.pt"
        if model_name in ("mediapipe-pose",):
            model_name = "yolov8m-pose.pt"
        self.model = YOLO(model_name)

        self._anchor_center: np.ndarray | None = None
        self._anchor_bbox: np.ndarray | None = None
        self._lock_origin: np.ndarray | None = None
        self._last_hip: np.ndarray | None = None
        self._dl_center: np.ndarray | None = None
        self._pick_xy: tuple[float, float] | None = config.athlete_pick_xy
        self._associator: IdentityAssociator | None = None
        self._assoc_logger: AssociationLogger | None = None
        self.lock_meta: dict = {}
        self.track_states: list[str] = []
        self.recommended_thresholds: dict = {}

    def _apply_env_overrides(self) -> None:
        freeze_env = os.environ.get("OLINE_FREEZE_IDENTITY", "").strip()
        if freeze_env in ("0", "false", "False"):
            self.config.track_freeze_identity = False
        elif freeze_env in ("1", "true", "True"):
            self.config.track_freeze_identity = True
        dbg = os.environ.get("OLINE_TRACK_DEBUG_DIR", "").strip()
        if dbg:
            self.config.track_debug_dir = dbg
        calib = os.environ.get("OLINE_TRACK_CALIB", "").strip()
        if calib in ("1", "true", "True"):
            self.config.track_calib_mode = True
        elif calib in ("0", "false", "False"):
            self.config.track_calib_mode = False

    def extract_all(
        self, video_path: str
    ) -> tuple[float, int, int, int, list[FramePose], list[FramePose | None], list[np.ndarray]]:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        # --- Step A: initial lock (selection only; not ongoing identity) ---
        if self._pick_xy is not None:
            self._anchor_center = np.array(
                [self._pick_xy[0] * width, self._pick_xy[1] * height], dtype=float
            )
            cx, cy = self._anchor_center
            self._anchor_bbox = np.array(
                [cx - 60, cy - 100, cx + 60, cy + 100], dtype=float
            )
            self._lock_origin = self._anchor_center.copy()
            self.lock_meta = {"method": "manual_pick_xy", "pick_xy": list(self._pick_xy)}
            print(f"  OL lock: manual pick {self._pick_xy}", flush=True)
            self._snap_lock_to_detection(frames[0] if frames else None)
        else:
            center, bbox, meta = lock_ol_from_frames(self.model, frames, self.config)
            self._anchor_center = center
            self._anchor_bbox = bbox
            self._lock_origin = center.copy()
            self.lock_meta = meta
            method = meta.get("method")
            if method == "jersey_ocr":
                print(
                    f"  OL lock: jersey_ocr #{meta.get('jersey')} "
                    f"conf={meta.get('ocr_confidence', 0):.2f} "
                    f"@ ({center[0]:.0f},{center[1]:.0f})",
                    flush=True,
                )
            else:
                score = meta.get("score")
                score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
                print(
                    f"  OL lock: {method} score={score_s} "
                    f"@ ({center[0]:.0f},{center[1]:.0f})",
                    flush=True,
                )

        # --- Frozen multi-frame appearance (no NN training) ---
        self._build_associator(frames, fps)

        # Reset BoT-SORT state for a clean run
        if hasattr(self.model, "predictor") and self.model.predictor is not None:
            self.model.predictor = None

        ol_poses: list[FramePose] = []
        dl_poses: list[FramePose | None] = []
        self.track_states = []
        for idx, frame in enumerate(frames):
            ol, dl = self._infer_frame(frame, idx, fps)
            ol_poses.append(ol)
            dl_poses.append(dl)
            self.track_states.append(ol.track_state)
            if (idx + 1) % 30 == 0:
                print(f"  pose {idx + 1} frames...", flush=True)

        if self._associator is not None:
            self.recommended_thresholds = self._associator.recommend_thresholds()
            csv_path = self._associator.logger.flush()
            self.lock_meta["recommended_thresholds"] = self.recommended_thresholds
            self.lock_meta["assoc_log"] = str(csv_path) if csv_path else None
            print(f"  track thresholds (from data): {self.recommended_thresholds}", flush=True)
            if csv_path:
                print(f"  track assoc log: {csv_path}", flush=True)

        # Save frozen reference crops for debugging
        self._save_reference_crops()

        lost_n = sum(1 for s in self.track_states if s == TrackState.LOST.value)
        self.lock_meta["target_id"] = PERMANENT_TARGET_ID
        self.lock_meta["frames_lost"] = lost_n
        self.lock_meta["identity_freeze"] = True
        self.lock_meta["tracker"] = "botsort+appearance_assoc"
        return fps, len(frames), width, height, ol_poses, dl_poses, frames

    def _snap_lock_to_detection(self, frame: np.ndarray | None) -> None:
        if frame is None or self._anchor_center is None:
            return
        results = self.model.predict(
            frame,
            verbose=False,
            conf=self.config.min_person_confidence,
            imgsz=self.config.pose_imgsz,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return
        boxes = results[0].boxes.xyxy.cpu().numpy()
        centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        i = int(np.argmin(np.linalg.norm(centers - self._anchor_center[None, :], axis=1)))
        self._anchor_bbox = boxes[i].astype(float)
        self._anchor_center = _bbox_center(self._anchor_bbox)
        self._lock_origin = self._anchor_center.copy()

    def _pre_snap_sample_indices(self, n_frames: int, fps: float) -> list[int]:
        # Several clean frames before contact (~4s issue) — sample first ~1.5s
        horizon = min(n_frames - 1, max(8, int(fps * 1.5)))
        raw = [0, horizon // 4, horizon // 2, (3 * horizon) // 4, horizon]
        return sorted({max(0, min(n_frames - 1, i)) for i in raw})

    def _build_associator(self, frames: list[np.ndarray], fps: float) -> None:
        if not frames or self._anchor_bbox is None or self._anchor_center is None:
            return

        idxs = self._pre_snap_sample_indices(len(frames), fps)
        ref_frames: list[np.ndarray] = []
        ref_boxes: list[np.ndarray] = []
        for i in idxs:
            frame = frames[i]
            results = self.model.predict(
                frame,
                verbose=False,
                conf=self.config.min_person_confidence,
                imgsz=self.config.pose_imgsz,
            )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            boxes = results[0].boxes.xyxy.cpu().numpy()
            centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
            j = int(np.argmin(np.linalg.norm(centers - self._anchor_center[None, :], axis=1)))
            # Only keep if still near lock origin (don't grab neighbor on later sample)
            if float(np.linalg.norm(centers[j] - self._lock_origin)) > _bbox_diag(self._anchor_bbox) * 0.8:
                continue
            ref_frames.append(frame)
            ref_boxes.append(boxes[j].astype(float))

        if not ref_boxes:
            ref_frames = [frames[0]]
            ref_boxes = [self._anchor_bbox.astype(float)]

        # Snap primary lock to first clean box
        self._anchor_bbox = ref_boxes[0].copy()
        self._anchor_center = _bbox_center(self._anchor_bbox)
        self._lock_origin = self._anchor_center.copy()

        ref = build_frozen_appearance(
            ref_frames,
            ref_boxes,
            target_id=PERMANENT_TARGET_ID,
            formation_xy=self._lock_origin,
            lock_diag=_bbox_diag(self._anchor_bbox),
        )
        if ref is None:
            print("  appearance: failed to build frozen reference", flush=True)
            return

        debug_dir = self.config.track_debug_dir
        if not debug_dir:
            debug_dir = str(Path("outputs") / "track_debug")
        self._assoc_logger = AssociationLogger(debug_dir)

        # Save original lock crop + clean refs immediately
        for k, crop in enumerate(ref.reference_crops):
            import cv2

            cv2.imwrite(str(Path(debug_dir) / "crops" / f"ref_{k:02d}.jpg"), crop)

        # Calib: do not hard-gate on appearance yet — log scores for threshold recommendation.
        # Production (calib=False): use configured floors, falling back to a soft self-sim discount
        # only when explicit values are provided.
        if self.config.track_calib_mode:
            thr = AssociationThresholds(
                min_appearance=None,
                min_jersey=None,
                min_weighted=None,
                min_iou_tracked=None,
                uncertain_weighted=None,
            )
        else:
            thr = AssociationThresholds(
                min_appearance=self.config.track_min_appearance,
                min_jersey=self.config.track_min_jersey,
                min_weighted=self.config.track_min_weighted,
                min_iou_tracked=self.config.track_min_iou_assoc,
                uncertain_weighted=self.config.track_uncertain_weighted,
            )

        self._associator = IdentityAssociator(
            ref,
            weights=AssociationWeights(
                appearance=self.config.track_w_appearance,
                motion=self.config.track_w_motion,
                iou=self.config.track_w_iou,
                jersey=self.config.track_w_jersey,
                size=self.config.track_w_size,
                formation=self.config.track_w_formation,
            ),
            thresholds=thr,
            lost_buffer=self.config.track_lost_buffer,
            logger=self._assoc_logger,
            reject_wrong_team=self.config.track_reject_wrong_team,
            team_jersey_floor=None if self.config.track_calib_mode else self.config.track_min_jersey,
        )
        self._associator.prev_bbox = self._anchor_bbox.copy()
        self._associator.prev_center = self._anchor_center.copy()

        frozen = ref.frozen.copy()
        print(
            f"  appearance: frozen from {len(ref.reference_crops)} pre-snap crops "
            f"(self_sim={ref.self_similarity:.3f}, calib={self.config.track_calib_mode})",
            flush=True,
        )
        # Prove freeze: store hash of frozen vector
        self.lock_meta["frozen_appearance_norm"] = float(np.linalg.norm(frozen))
        self.lock_meta["frozen_self_similarity"] = float(ref.self_similarity)
        self._frozen_vector_snapshot = frozen

    def _save_reference_crops(self) -> None:
        if self._associator is None or self._assoc_logger is None:
            return
        ref = self._associator.ref
        # Verify frozen was not overwritten
        if hasattr(self, "_frozen_vector_snapshot"):
            same = bool(np.allclose(self._frozen_vector_snapshot, ref.frozen))
            self.lock_meta["frozen_unchanged"] = same
            print(f"  appearance: frozen unchanged={same}", flush=True)

    def _empty(self, idx: int, ts: float, state: str = TrackState.LOST.value, conf: float = 0.0) -> FramePose:
        return FramePose(
            frame_idx=idx,
            timestamp_ms=ts,
            keypoints_xy=np.full((17, 2), np.nan),
            keypoints_conf=np.zeros(17),
            bbox_xyxy=None,
            person_confidence=0.0,
            low_confidence=True,
            usable=False,
            track_state=state,
            track_confidence=conf,
            track_id=None,
            target_id=PERMANENT_TARGET_ID,
        )

    def _pack(
        self,
        idx: int,
        ts: float,
        kxy: np.ndarray,
        kcf: np.ndarray,
        bbox: np.ndarray,
        conf: float,
        *,
        state: str,
        track_conf: float,
        track_id: int | None,
    ) -> FramePose:
        missing = (kxy[:, 0] <= 1.0) & (kxy[:, 1] <= 1.0)
        kcf = kcf.copy()
        kxy = kxy.copy()
        kcf[missing] = 0.0
        kxy[missing] = np.nan
        confident = int((kcf >= self.config.min_keypoint_confidence).sum())
        ratio = confident / 17.0
        low = ratio < self.config.min_frame_keypoint_ratio
        usable = (not low) and conf >= self.config.min_person_confidence
        # LOST/UNCERTAIN with weak conf → not usable for metrics
        if state == TrackState.LOST.value:
            usable = False
        return FramePose(
            frame_idx=idx,
            timestamp_ms=ts,
            keypoints_xy=kxy,
            keypoints_conf=kcf,
            bbox_xyxy=bbox.astype(float),
            person_confidence=float(conf),
            low_confidence=low,
            usable=usable,
            track_state=state,
            track_confidence=float(track_conf),
            track_id=track_id,
            target_id=PERMANENT_TARGET_ID,
        )

    def _infer_frame(
        self, frame: np.ndarray, idx: int, fps: float
    ) -> tuple[FramePose, FramePose | None]:
        ts = (idx / fps) * 1000.0 if fps > 0 else 0.0

        # --- Detection step (BoT-SORT + GMC); full frame so occlusion recovery works ---
        results = self.model.track(
            frame,
            persist=True,
            tracker=BOTSORT_YAML,
            verbose=False,
            conf=self.config.min_person_confidence,
            imgsz=self.config.pose_imgsz,
        )
        if not results:
            return self._empty(idx, ts), None
        r0 = results[0]
        if r0.keypoints is None or r0.boxes is None or len(r0.boxes) == 0:
            if self._associator is not None:
                decision = self._associator.associate(idx, frame, np.zeros((0, 4)), np.zeros(0))
                return self._empty(idx, ts, decision.state.value, decision.confidence), None
            return self._empty(idx, ts), None

        xy = r0.keypoints.xy.cpu().numpy().copy()
        kconf = (
            r0.keypoints.conf.cpu().numpy()
            if r0.keypoints.conf is not None
            else np.ones(xy.shape[:2])
        )
        box_xyxy = r0.boxes.xyxy.cpu().numpy().copy()
        box_conf = r0.boxes.conf.cpu().numpy()
        track_ids = None
        if r0.boxes.id is not None:
            track_ids = r0.boxes.id.cpu().numpy().astype(int)

        # --- Association step (permanent target; may return no target) ---
        if self._associator is None:
            # Fallback: nearest to lock (should be rare)
            centers = (box_xyxy[:, :2] + box_xyxy[:, 2:]) / 2.0
            i = int(np.argmin(np.linalg.norm(centers - self._anchor_center[None, :], axis=1)))
            ol = self._pack(
                idx, ts, xy[i], kconf[i], box_xyxy[i], float(box_conf[i]),
                state=TrackState.TRACKED.value, track_conf=1.0,
                track_id=int(track_ids[i]) if track_ids is not None else None,
            )
            return ol, None

        # Seed locked BoT-SORT id on first frame near lock
        if self._associator.locked_botsort_id is None and track_ids is not None and self._anchor_center is not None:
            centers = (box_xyxy[:, :2] + box_xyxy[:, 2:]) / 2.0
            i0 = int(np.argmin(np.linalg.norm(centers - self._anchor_center[None, :], axis=1)))
            self._associator.locked_botsort_id = int(track_ids[i0])
            self._associator.botsort_id = int(track_ids[i0])
            print(f"  track: locked BoT-SORT id={track_ids[i0]}", flush=True)

        decision = self._associator.associate(idx, frame, box_xyxy, box_conf, track_ids)

        if decision.best is None:
            return self._empty(idx, ts, decision.state.value, decision.confidence), None

        # Map chosen score back to detection index
        ol_i = None
        for i in range(len(box_conf)):
            tid = int(track_ids[i]) if track_ids is not None else None
            if decision.best.track_id is not None and tid == decision.best.track_id:
                ol_i = i
                break
            if decision.best.track_id is None:
                # match by IoU to associator prev box (just set)
                if self._associator.prev_bbox is not None:
                    if _bbox_iou(self._associator.prev_bbox, box_xyxy[i]) > 0.9:
                        ol_i = i
                        break
        if ol_i is None:
            # last resort: highest IoU to prev
            if self._associator.prev_bbox is not None:
                ious = [_bbox_iou(self._associator.prev_bbox, box_xyxy[i]) for i in range(len(box_conf))]
                ol_i = int(np.argmax(ious))
            else:
                return self._empty(idx, ts, TrackState.LOST.value, 0.0), None

        ol = self._pack(
            idx,
            ts,
            xy[ol_i],
            kconf[ol_i],
            box_xyxy[ol_i],
            float(box_conf[ol_i]),
            state=decision.state.value,
            track_conf=decision.confidence,
            track_id=decision.botsort_id,
        )
        if ol.usable:
            self._anchor_bbox = box_xyxy[ol_i].astype(float)
            self._anchor_center = _bbox_center(self._anchor_bbox)
            try:
                hip = hip_mid(xy[ol_i])
                if hip is not None and not np.any(np.isnan(hip)):
                    self._last_hip = hip.astype(float)
            except Exception:
                pass

        dl = None
        if self.config.track_defender and len(box_conf) > 1:
            dl_i = self._select_dl(box_xyxy, box_conf, ol_i)
            if dl_i is not None:
                dl = self._pack(
                    idx,
                    ts,
                    xy[dl_i],
                    kconf[dl_i],
                    box_xyxy[dl_i],
                    float(box_conf[dl_i]),
                    state=TrackState.TRACKED.value,
                    track_conf=float(box_conf[dl_i]),
                    track_id=int(track_ids[dl_i]) if track_ids is not None else None,
                )
                dc = _bbox_center(box_xyxy[dl_i])
                self._dl_center = dc if self._dl_center is None else 0.7 * self._dl_center + 0.3 * dc

        return ol, dl

    def _select_dl(self, box_xyxy, box_conf, ol_i: int) -> int | None:
        centers = (box_xyxy[:, :2] + box_xyxy[:, 2:]) / 2.0
        ol_c = centers[ol_i]
        idxs = [i for i in range(len(box_conf)) if i != ol_i]
        if not idxs:
            return None
        if self._dl_center is not None:
            dist = np.linalg.norm(centers[idxs] - self._dl_center[None, :], axis=1)
            near_ol = np.linalg.norm(centers[idxs] - ol_c[None, :], axis=1)
            score = -dist - 0.35 * near_ol
            return int(idxs[int(np.argmax(score))])
        dist = np.linalg.norm(centers[idxs] - ol_c[None, :], axis=1)
        return int(idxs[int(np.argmin(dist))])

    # --- Kept for unit tests / spatial fallback ---
    def _select_ol(self, box_xyxy, box_conf, width, height):
        """Spatial-only scorer (tests + fallback). Production uses IdentityAssociator."""
        n = len(box_conf)
        if n == 0:
            return None, 0.0
        centers = (box_xyxy[:, :2] + box_xyxy[:, 2:]) / 2.0
        if self._anchor_center is None:
            return 0, 1.0
        prior = self._anchor_bbox
        if prior is None:
            i = int(np.argmin(np.linalg.norm(centers - self._anchor_center[None, :], axis=1)))
            return i, 1.0
        prior_diag = max(_bbox_diag(prior), 1.0)
        pred = self._anchor_center
        scores = np.full(n, -1e9, dtype=float)
        ious = np.zeros(n, dtype=float)
        for i in range(n):
            dist = float(np.linalg.norm(centers[i] - pred))
            if dist > prior_diag * self.config.track_max_jump_mult * 1.5:
                continue
            if dist > prior_diag * self.config.track_max_center_frac * 2.5:
                continue
            iou = _bbox_iou(prior, box_xyxy[i])
            ious[i] = iou
            dist_score = max(0.0, 1.0 - dist / max(prior_diag * self.config.track_max_center_frac, 1.0))
            scores[i] = 0.55 * iou + 0.45 * dist_score
        best = int(np.argmax(scores))
        if scores[best] < -1e8:
            return None, 0.0
        return best, float(scores[best])


def keypoints_as_dict(pose: FramePose) -> dict:
    out = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        xy = pose.keypoints_xy[i]
        out[name] = {
            "x": None if np.isnan(xy[0]) else float(xy[0]),
            "y": None if np.isnan(xy[1]) else float(xy[1]),
            "confidence": float(pose.keypoints_conf[i]),
        }
    return out
