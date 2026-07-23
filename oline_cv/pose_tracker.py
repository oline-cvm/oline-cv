"""YOLO pose tracking locked to a single athlete (e.g. #76).

After the first lock, inference runs on a padded crop around that player only —
other people on the field are never posed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from oline_cv.config import AnalysisConfig, KEYPOINT_NAMES


@dataclass
class FramePose:
    frame_idx: int
    timestamp_ms: float
    keypoints_xy: np.ndarray  # (17, 2) full-frame coords
    keypoints_conf: np.ndarray  # (17,)
    bbox_xyxy: np.ndarray | None
    person_confidence: float
    low_confidence: bool
    usable: bool


class PoseTracker:
    """Track exactly one offensive lineman; ignore everyone else."""

    def __init__(self, config: AnalysisConfig):
        self.config = config
        model_name = config.pose_model
        if model_name in ("mediapipe-pose", "", None):
            model_name = "yolov8n-pose.pt"
        self.model = YOLO(model_name)
        self._anchor_center: np.ndarray | None = None
        self._anchor_bbox: np.ndarray | None = None
        self._pick_xy: tuple[float, float] | None = config.athlete_pick_xy
        # Default pick for jersey #76 (LT) on elevated All-22 / sideline film
        if self._pick_xy is None and config.target_jersey == 76:
            self._pick_xy = (0.272, 0.53)

    def extract_all(
        self, video_path: str
    ) -> tuple[float, int, int, int, list[FramePose], list[np.ndarray]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        poses: list[FramePose] = []
        frames: list[np.ndarray] = []
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
                poses.append(self._infer_frame(frame, idx, fps))
                idx += 1
                if idx % 30 == 0:
                    print(f"  pose {idx} frames...", flush=True)
        finally:
            cap.release()

        return fps, len(frames), width, height, poses, frames

    def _crop_around_anchor(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, int, int] | None:
        """Return (crop, offset_x, offset_y) tightly around the locked athlete."""
        if self._anchor_bbox is None:
            return None
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self._anchor_bbox
        bw, bh = x1 - x0, y1 - y0
        pad_x = bw * self.config.track_crop_pad
        pad_y = bh * self.config.track_crop_pad
        cx0 = int(max(0, x0 - pad_x))
        cy0 = int(max(0, y0 - pad_y))
        cx1 = int(min(w, x1 + pad_x))
        cy1 = int(min(h, y1 + pad_y))
        if cx1 - cx0 < 32 or cy1 - cy0 < 32:
            return None
        return frame[cy0:cy1, cx0:cx1].copy(), cx0, cy0

    def _infer_frame(self, frame: np.ndarray, idx: int, fps: float) -> FramePose:
        ts = (idx / fps) * 1000.0 if fps > 0 else 0.0
        empty = FramePose(
            frame_idx=idx,
            timestamp_ms=ts,
            keypoints_xy=np.full((17, 2), np.nan),
            keypoints_conf=np.zeros(17),
            bbox_xyxy=None,
            person_confidence=0.0,
            low_confidence=True,
            usable=False,
        )

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
            imgsz=self.config.pose_imgsz if crop_info is None else min(640, self.config.pose_imgsz),
        )
        if not results:
            return empty

        r0 = results[0]
        if r0.keypoints is None or r0.boxes is None or len(r0.boxes) == 0:
            return empty

        xy = r0.keypoints.xy.cpu().numpy().copy()
        conf = (
            r0.keypoints.conf.cpu().numpy()
            if r0.keypoints.conf is not None
            else np.ones(xy.shape[:2])
        )
        box_xyxy = r0.boxes.xyxy.cpu().numpy().copy()
        box_conf = r0.boxes.conf.cpu().numpy()

        # Map crop coords → full frame
        ox, oy = offset
        xy[:, :, 0] += ox
        xy[:, :, 1] += oy
        box_xyxy[:, [0, 2]] += ox
        box_xyxy[:, [1, 3]] += oy

        pick = self._select_person(xy, box_xyxy, box_conf, w, h)
        if pick is None:
            return empty

        kxy = xy[pick].astype(float).copy()
        kcf = conf[pick].astype(float).copy()
        missing = (kxy[:, 0] <= 1.0) & (kxy[:, 1] <= 1.0)
        kcf[missing] = 0.0
        kxy[missing] = np.nan

        confident = int((kcf >= self.config.min_keypoint_confidence).sum())
        ratio = confident / 17.0
        low = ratio < self.config.min_frame_keypoint_ratio
        usable = (not low) and float(box_conf[pick]) >= self.config.min_person_confidence

        bbox = box_xyxy[pick].astype(float)
        center = (bbox[:2] + bbox[2:]) / 2.0
        if usable:
            if self._anchor_center is None:
                self._anchor_center = center
                self._anchor_bbox = bbox
            else:
                self._anchor_center = 0.75 * self._anchor_center + 0.25 * center
                self._anchor_bbox = 0.75 * self._anchor_bbox + 0.25 * bbox

        return FramePose(
            frame_idx=idx,
            timestamp_ms=ts,
            keypoints_xy=kxy,
            keypoints_conf=kcf,
            bbox_xyxy=bbox,
            person_confidence=float(box_conf[pick]),
            low_confidence=low,
            usable=usable,
        )

    def _select_person(
        self,
        xy: np.ndarray,
        box_xyxy: np.ndarray,
        box_conf: np.ndarray,
        width: int,
        height: int,
    ) -> int | None:
        n = len(box_conf)
        if n == 0:
            return None

        centers = (box_xyxy[:, :2] + box_xyxy[:, 2:]) / 2.0
        areas = (box_xyxy[:, 2] - box_xyxy[:, 0]) * (box_xyxy[:, 3] - box_xyxy[:, 1])

        # Once locked, ONLY accept the nearest detection within jump radius.
        if self._anchor_center is not None:
            dist = np.linalg.norm(centers - self._anchor_center[None, :], axis=1)
            diags = np.linalg.norm(box_xyxy[:, 2:] - box_xyxy[:, :2], axis=1)
            max_jump = float(np.median(diags) * self.config.track_max_jump_mult)
            if self._anchor_bbox is not None:
                ab = self._anchor_bbox
                max_jump = max(max_jump, float(np.linalg.norm(ab[2:] - ab[:2]) * 1.2))
            valid = np.where(dist <= max_jump)[0]
            if len(valid) == 0:
                return None
            # Closest to anchor wins — never switch to another player nearby
            return int(valid[int(np.argmin(dist[valid]))])

        # First lock: click / jersey default pick, else largest in ROI
        x0, y0, x1, y1 = self.config.athlete_roi
        rx0, ry0, rx1, ry1 = x0 * width, y0 * height, x1 * width, y1 * height
        in_roi = (
            (centers[:, 0] >= rx0)
            & (centers[:, 0] <= rx1)
            & (centers[:, 1] >= ry0)
            & (centers[:, 1] <= ry1)
        )

        if self._pick_xy is not None:
            px = self._pick_xy[0] * width
            py = self._pick_xy[1] * height
            dist = np.linalg.norm(centers - np.array([px, py]), axis=1)
            return int(np.argmin(dist))

        candidates = np.where(in_roi)[0]
        if len(candidates) == 0:
            candidates = np.arange(n)
        score = areas[candidates] * box_conf[candidates]
        return int(candidates[int(np.argmax(score))])


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
