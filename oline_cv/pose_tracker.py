"""YOLO pose tracking locked to one offensive lineman.

Identity is sticky: IoU + predicted center + size gates stop mid-play
switches to a nearby teammate/defender. The posture NN never chooses the athlete.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from oline_cv.config import AnalysisConfig, KEYPOINT_NAMES
from oline_cv.geometry import hip_mid
from oline_cv.ol_select import lock_ol_from_frames


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


def _bbox_area(b: np.ndarray) -> float:
    return max(1.0, float(b[2] - b[0]) * float(b[3] - b[1]))


def _bbox_diag(b: np.ndarray) -> float:
    return float(np.linalg.norm(b[2:] - b[:2]))


class PoseTracker:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        model_name = config.pose_model or "yolov8m-pose.pt"
        if model_name in ("mediapipe-pose",):
            model_name = "yolov8m-pose.pt"
        self.model = YOLO(model_name)
        self._anchor_center: np.ndarray | None = None
        self._anchor_bbox: np.ndarray | None = None
        self._anchor_vel: np.ndarray = np.zeros(2, dtype=float)
        self._lock_origin: np.ndarray | None = None
        self._last_hip: np.ndarray | None = None
        self._dl_center: np.ndarray | None = None
        self._pick_xy: tuple[float, float] | None = config.athlete_pick_xy
        self._lost_frames: int = 0
        self._switch_rejects: int = 0
        self.lock_meta: dict = {}

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
                    f"hits={meta.get('agreement')}/{meta.get('votes')} "
                    f"@ ({center[0]:.0f},{center[1]:.0f})",
                    flush=True,
                )
            else:
                score = meta.get("score")
                score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
                print(
                    f"  OL lock: {method} score={score_s} "
                    f"knee={meta.get('knee_flex')} @ ({center[0]:.0f},{center[1]:.0f})",
                    flush=True,
                )

        ol_poses: list[FramePose] = []
        dl_poses: list[FramePose | None] = []
        for idx, frame in enumerate(frames):
            ol, dl = self._infer_frame(frame, idx, fps)
            ol_poses.append(ol)
            dl_poses.append(dl)
            if (idx + 1) % 30 == 0:
                print(f"  pose {idx + 1} frames...", flush=True)

        if self._switch_rejects:
            print(f"  track: rejected {self._switch_rejects} identity-switch candidates", flush=True)
        self.lock_meta["track_switch_rejects"] = self._switch_rejects
        return fps, len(frames), width, height, ol_poses, dl_poses, frames

    def _crop_pad_frac(self) -> float:
        """Widen crop briefly only when the lock is lost — never while sticky."""
        base = self.config.track_crop_pad
        if self._lost_frames >= self.config.track_lost_expand_frames:
            return min(1.1, base * 1.35)
        return base

    def _crop_around_anchor(self, frame: np.ndarray):
        if self._anchor_bbox is None:
            return None
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self._anchor_bbox
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
        pad = self._crop_pad_frac()
        pad_x = bw * pad
        pad_y = bh * pad
        # Prefer predicted center if we have velocity (follow the athlete, not old box)
        if self._anchor_center is not None:
            pred = self._anchor_center + self._anchor_vel
            cx0 = int(max(0, pred[0] - 0.5 * bw - pad_x))
            cy0 = int(max(0, pred[1] - 0.5 * bh - pad_y))
            cx1 = int(min(w, pred[0] + 0.5 * bw + pad_x))
            cy1 = int(min(h, pred[1] + 0.5 * bh + pad_y))
        else:
            cx0 = int(max(0, x0 - pad_x))
            cy0 = int(max(0, y0 - pad_y))
            cx1 = int(min(w, x1 + pad_x))
            cy1 = int(min(h, y1 + pad_y))
        if cx1 - cx0 < 32 or cy1 - cy0 < 32:
            return None
        return frame[cy0:cy1, cx0:cx1].copy(), cx0, cy0

    def _empty(self, idx: int, ts: float) -> FramePose:
        return FramePose(
            frame_idx=idx,
            timestamp_ms=ts,
            keypoints_xy=np.full((17, 2), np.nan),
            keypoints_conf=np.zeros(17),
            bbox_xyxy=None,
            person_confidence=0.0,
            low_confidence=True,
            usable=False,
        )

    def _pack(
        self, idx: int, ts: float, kxy: np.ndarray, kcf: np.ndarray, bbox: np.ndarray, conf: float
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
        return FramePose(
            frame_idx=idx,
            timestamp_ms=ts,
            keypoints_xy=kxy,
            keypoints_conf=kcf,
            bbox_xyxy=bbox.astype(float),
            person_confidence=float(conf),
            low_confidence=low,
            usable=usable,
        )

    def _infer_frame(
        self, frame: np.ndarray, idx: int, fps: float
    ) -> tuple[FramePose, FramePose | None]:
        ts = (idx / fps) * 1000.0 if fps > 0 else 0.0
        empty = self._empty(idx, ts)
        h, w = frame.shape[:2]
        offset = (0, 0)
        infer_img = frame
        crop_info = self._crop_around_anchor(frame)
        if crop_info is not None:
            infer_img, ox, oy = crop_info
            offset = (ox, oy)

        results = self.model.predict(
            infer_img,
            verbose=False,
            conf=self.config.min_person_confidence,
            imgsz=self.config.pose_imgsz if crop_info is None else min(960, self.config.pose_imgsz),
        )
        if not results:
            self._on_miss()
            return empty, None
        r0 = results[0]
        if r0.keypoints is None or r0.boxes is None or len(r0.boxes) == 0:
            self._on_miss()
            return empty, None

        xy = r0.keypoints.xy.cpu().numpy().copy()
        conf = (
            r0.keypoints.conf.cpu().numpy()
            if r0.keypoints.conf is not None
            else np.ones(xy.shape[:2])
        )
        box_xyxy = r0.boxes.xyxy.cpu().numpy().copy()
        box_conf = r0.boxes.conf.cpu().numpy()
        ox, oy = offset
        xy[:, :, 0] += ox
        xy[:, :, 1] += oy
        box_xyxy[:, [0, 2]] += ox
        box_xyxy[:, [1, 3]] += oy

        ol_i, match_score = self._select_ol(box_xyxy, box_conf, w, h)
        if ol_i is None:
            self._on_miss()
            return empty, None

        # Refuse to emit a different body even for one frame — holds identity sticky
        if not self._accept_ol(box_xyxy[ol_i], match_score, xy[ol_i], conf[ol_i]):
            self._on_miss()
            return empty, None

        ol = self._pack(idx, ts, xy[ol_i], conf[ol_i], box_xyxy[ol_i], float(box_conf[ol_i]))
        if ol.usable:
            self._commit_ol(box_xyxy[ol_i], match_score, xy[ol_i], conf[ol_i])
            self._lost_frames = 0
        else:
            self._on_miss()

        dl = None
        if self.config.track_defender and len(box_conf) > 1:
            dl_i = self._select_dl(box_xyxy, box_conf, ol_i)
            if dl_i is not None:
                dl = self._pack(idx, ts, xy[dl_i], conf[dl_i], box_xyxy[dl_i], float(box_conf[dl_i]))
                dc = _bbox_center(box_xyxy[dl_i])
                self._dl_center = dc if self._dl_center is None else 0.7 * self._dl_center + 0.3 * dc

        return ol, dl

    def _on_miss(self) -> None:
        self._lost_frames += 1
        # Decay velocity so we don't coast into another body
        self._anchor_vel *= 0.85

    def _origin_leash(self) -> float:
        """Dynamic travel budget from lock — scales with athlete size + play type."""
        diag = (
            _bbox_diag(self._anchor_bbox)
            if self._anchor_bbox is not None
            else 200.0
        )
        mult = self.config.track_max_origin_diag_mult
        if self.config.play_type == "run":
            mult = self.config.track_max_origin_diag_mult_run
        return float(diag * mult)

    def _accept_ol(
        self,
        box: np.ndarray,
        match_score: float,
        kxy: np.ndarray | None = None,
        kcf: np.ndarray | None = None,
    ) -> bool:
        """True only if this detection is the same locked athlete (clip-agnostic)."""
        if self._anchor_center is None or self._anchor_bbox is None:
            return True
        c = _bbox_center(box)
        jump = float(np.linalg.norm(c - self._anchor_center))
        diag = max(_bbox_diag(self._anchor_bbox), 1.0)
        iou = _bbox_iou(self._anchor_bbox, box)
        if jump > diag * self.config.track_teleport_frac:
            self._switch_rejects += 1
            return False
        if iou < 0.28:
            self._switch_rejects += 1
            return False
        if match_score < 0.45:
            self._switch_rejects += 1
            return False
        if self._lock_origin is not None:
            if float(np.linalg.norm(c - self._lock_origin)) > self._origin_leash():
                self._switch_rejects += 1
                return False
        # Hip continuity — catches neighbor steals when boxes overlap
        if kxy is not None and self._last_hip is not None:
            try:
                hip = hip_mid(kxy)
            except Exception:
                hip = None
            if hip is not None and not np.any(np.isnan(hip)):
                hip_jump = float(np.linalg.norm(hip - self._last_hip))
                if hip_jump > diag * self.config.track_hip_jump_frac:
                    self._switch_rejects += 1
                    return False
                if self._lock_origin is not None:
                    if abs(float(hip[1] - self._lock_origin[1])) > diag * self.config.track_hip_vert_frac:
                        self._switch_rejects += 1
                        return False
                    if float(np.linalg.norm(hip - self._lock_origin)) > diag * self.config.track_hip_origin_frac:
                        self._switch_rejects += 1
                        return False
        return True

    def _commit_ol(
        self,
        box: np.ndarray,
        match_score: float,
        kxy: np.ndarray | None = None,
        kcf: np.ndarray | None = None,
    ) -> None:
        """Update lock — caller already gated with _accept_ol."""
        c = _bbox_center(box)
        if self._anchor_center is None or self._anchor_bbox is None:
            self._anchor_center = c.astype(float)
            self._anchor_bbox = box.astype(float)
            self._anchor_vel = np.zeros(2, dtype=float)
            if kxy is not None:
                try:
                    hip = hip_mid(kxy)
                    if hip is not None and not np.any(np.isnan(hip)):
                        self._last_hip = hip.astype(float)
                except Exception:
                    pass
            return

        iou = _bbox_iou(self._anchor_bbox, box)
        ema = float(self.config.track_ema)
        if match_score >= 0.70 and iou >= 0.45:
            ema = min(0.94, ema + 0.04)

        diag = max(_bbox_diag(self._anchor_bbox), 1.0)
        new_c = ema * self._anchor_center + (1.0 - ema) * c
        measured_v = c - self._anchor_center
        vmax = diag * 0.22
        speed = float(np.linalg.norm(measured_v))
        if speed > vmax:
            measured_v = measured_v * (vmax / speed)
        self._anchor_vel = 0.70 * self._anchor_vel + 0.30 * measured_v
        self._anchor_center = new_c
        self._anchor_bbox = ema * self._anchor_bbox + (1.0 - ema) * box.astype(float)
        if kxy is not None:
            try:
                hip = hip_mid(kxy)
                if hip is not None and not np.any(np.isnan(hip)):
                    # Keep raw last hip for continuity gate (no EMA drift)
                    self._last_hip = hip.astype(float)
            except Exception:
                pass

    def _select_ol(self, box_xyxy, box_conf, width, height) -> tuple[int | None, float]:
        """Multi-filter identity match. Returns (index, score) or (None, 0)."""
        n = len(box_conf)
        if n == 0:
            return None, 0.0

        centers = (box_xyxy[:, :2] + box_xyxy[:, 2:]) / 2.0
        areas = (box_xyxy[:, 2] - box_xyxy[:, 0]) * (box_xyxy[:, 3] - box_xyxy[:, 1])
        origin_leash = self._origin_leash() if self._anchor_bbox is not None else float(width) * 0.25

        # Cold start without anchor
        if self._anchor_center is None:
            if self._pick_xy is not None:
                px, py = self._pick_xy[0] * width, self._pick_xy[1] * height
                i = int(np.argmin(np.linalg.norm(centers - np.array([px, py]), axis=1)))
                return i, 1.0
            x0, y0, x1, y1 = self.config.athlete_roi
            in_roi = (
                (centers[:, 0] >= x0 * width)
                & (centers[:, 0] <= x1 * width)
                & (centers[:, 1] >= y0 * height)
                & (centers[:, 1] <= y1 * height)
            )
            cand = np.where(in_roi)[0]
            if len(cand) == 0:
                cand = np.arange(n)
            i = int(cand[int(np.argmax(areas[cand] * box_conf[cand]))])
            return i, 1.0

        pred = self._anchor_center + self._anchor_vel
        prior = self._anchor_bbox if self._anchor_bbox is not None else None
        prior_area = _bbox_area(prior) if prior is not None else float(np.median(areas))
        prior_diag = _bbox_diag(prior) if prior is not None else float(np.median(
            np.linalg.norm(box_xyxy[:, 2:] - box_xyxy[:, :2], axis=1)
        ))
        max_center = prior_diag * self.config.track_max_center_frac
        if self._lost_frames >= self.config.track_lost_expand_frames:
            max_center *= 1.12

        scores = np.full(n, -1e9, dtype=float)
        ious = np.zeros(n, dtype=float)
        for i in range(n):
            dist = float(np.linalg.norm(centers[i] - pred))
            dist_lock = float(np.linalg.norm(centers[i] - self._anchor_center))
            if dist > max_center * self.config.track_max_jump_mult * 1.5:
                continue
            if dist_lock > prior_diag * 0.55:
                continue
            # Stay near the original lock — pass-set shouldn't wander across the screen
            if self._lock_origin is not None:
                if float(np.linalg.norm(centers[i] - self._lock_origin)) > origin_leash:
                    continue

            iou = _bbox_iou(prior, box_xyxy[i]) if prior is not None else 0.0
            ious[i] = iou

            if (
                prior is not None
                and dist > prior_diag * self.config.track_teleport_frac
                and iou < max(self.config.track_min_iou, 0.40)
            ):
                continue

            ar = float(areas[i] / prior_area)
            if ar < self.config.track_area_ratio_min or ar > self.config.track_area_ratio_max:
                if iou < self.config.track_min_iou:
                    continue

            dist_score = max(0.0, 1.0 - dist / max(max_center, 1.0))
            size_score = 1.0 - min(1.0, abs(np.log(max(ar, 1e-3))) / 1.2)
            conf_score = float(np.clip(box_conf[i], 0.0, 1.0))
            scores[i] = 0.55 * iou + 0.25 * dist_score + 0.12 * size_score + 0.08 * conf_score

        best = int(np.argmax(scores))
        best_score = float(scores[best])
        if best_score < -1e8:
            return None, 0.0

        order = np.argsort(-scores)
        if len(order) >= 2 and scores[order[1]] > -1e8:
            second = int(order[1])
            if (
                ious[best] < self.config.track_min_iou
                and (scores[best] - scores[second]) < self.config.track_switch_iou_margin
            ):
                self._switch_rejects += 1
                return None, 0.0

        if (
            prior is not None
            and self._lost_frames < self.config.track_lost_expand_frames
            and ious[best] < self.config.track_min_iou * 0.55
            and best_score < 0.45
        ):
            self._switch_rejects += 1
            return None, 0.0

        return best, best_score

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
