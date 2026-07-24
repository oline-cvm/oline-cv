"""YOLO pose tracking locked to one auto-detected offensive lineman.

After lock, inference uses a padded crop around that OL. The nearest other
person in-crop is treated as the matchup defender (mirror / hands / anchor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from oline_cv.config import AnalysisConfig, KEYPOINT_NAMES
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


class PoseTracker:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        model_name = config.pose_model or "yolov8m-pose.pt"
        if model_name in ("mediapipe-pose",):
            model_name = "yolov8m-pose.pt"
        self.model = YOLO(model_name)
        self._anchor_center: np.ndarray | None = None
        self._anchor_bbox: np.ndarray | None = None
        self._dl_center: np.ndarray | None = None
        self._pick_xy: tuple[float, float] | None = config.athlete_pick_xy
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

        # Auto-lock OL unless user provided an explicit click
        if self._pick_xy is not None:
            self._anchor_center = np.array(
                [self._pick_xy[0] * width, self._pick_xy[1] * height], dtype=float
            )
            self.lock_meta = {"method": "manual_pick_xy", "pick_xy": list(self._pick_xy)}
            print(f"  OL lock: manual pick {self._pick_xy}", flush=True)
        else:
            center, bbox, meta = lock_ol_from_frames(self.model, frames, self.config)
            self._anchor_center = center
            self._anchor_bbox = bbox
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

        return fps, len(frames), width, height, ol_poses, dl_poses, frames

    def _crop_around_anchor(self, frame: np.ndarray):
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
            return empty, None
        r0 = results[0]
        if r0.keypoints is None or r0.boxes is None or len(r0.boxes) == 0:
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

        ol_i = self._select_ol(box_xyxy, box_conf, w, h)
        if ol_i is None:
            return empty, None

        ol = self._pack(idx, ts, xy[ol_i], conf[ol_i], box_xyxy[ol_i], float(box_conf[ol_i]))
        if ol.usable:
            c = (box_xyxy[ol_i][:2] + box_xyxy[ol_i][2:]) / 2.0
            if self._anchor_center is None:
                self._anchor_center = c
                self._anchor_bbox = box_xyxy[ol_i].astype(float)
            else:
                self._anchor_center = 0.75 * self._anchor_center + 0.25 * c
                self._anchor_bbox = 0.75 * self._anchor_bbox + 0.25 * box_xyxy[ol_i]

        dl = None
        if self.config.track_defender and len(box_conf) > 1:
            dl_i = self._select_dl(box_xyxy, box_conf, ol_i)
            if dl_i is not None:
                dl = self._pack(idx, ts, xy[dl_i], conf[dl_i], box_xyxy[dl_i], float(box_conf[dl_i]))
                dc = (box_xyxy[dl_i][:2] + box_xyxy[dl_i][2:]) / 2.0
                self._dl_center = dc if self._dl_center is None else 0.7 * self._dl_center + 0.3 * dc

        return ol, dl

    def _select_ol(self, box_xyxy, box_conf, width, height) -> int | None:
        n = len(box_conf)
        if n == 0:
            return None
        centers = (box_xyxy[:, :2] + box_xyxy[:, 2:]) / 2.0
        areas = (box_xyxy[:, 2] - box_xyxy[:, 0]) * (box_xyxy[:, 3] - box_xyxy[:, 1])

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
            return int(valid[int(np.argmin(dist[valid]))])

        if self._pick_xy is not None:
            px, py = self._pick_xy[0] * width, self._pick_xy[1] * height
            return int(np.argmin(np.linalg.norm(centers - np.array([px, py]), axis=1)))

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
        return int(cand[int(np.argmax(areas[cand] * box_conf[cand]))])

    def _select_dl(self, box_xyxy, box_conf, ol_i: int) -> int | None:
        centers = (box_xyxy[:, :2] + box_xyxy[:, 2:]) / 2.0
        ol_c = centers[ol_i]
        idxs = [i for i in range(len(box_conf)) if i != ol_i]
        if not idxs:
            return None
        if self._dl_center is not None:
            dist = np.linalg.norm(centers[idxs] - self._dl_center[None, :], axis=1)
            # Prefer continuity, but must stay near OL
            near_ol = np.linalg.norm(centers[idxs] - ol_c[None, :], axis=1)
            score = -dist - 0.35 * near_ol
            return int(idxs[int(np.argmax(score))])
        # First DL lock: nearest other person to OL
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
