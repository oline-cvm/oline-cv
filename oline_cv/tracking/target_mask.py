"""Target-only video object segmentation for the locked tackle.

Deep HM-SORT still tracks every player. This module tracks **one** visual
object — the user-selected OL — through occlusions, after SAM-Deep-EIoU
(selective mask propagation when association is ambiguous) and SAM 2.1
video memory.

Backends
--------
``FakeTargetMaskTracker``
    Tests; no SAM, no weights.
``SAM2TargetMaskTracker``
    Official ``sam2`` package, lazy-loaded. Requires a local checkpoint
    (``sam2_checkpoint`` or ``OLINE_SAM2_CHECKPOINT``). Never downloads.
``NoneBackend``
    Segmentation disabled; every frame reports no mask.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class TargetMaskUnavailableError(RuntimeError):
    """SAM2 was requested but cannot run (missing package or checkpoint)."""


@dataclass
class MaskEvidence:
    frame_idx: int
    mask: np.ndarray | None  # bool (H, W) at video resolution, or None
    score: float = 0.0
    area: int = 0
    bbox: np.ndarray | None = None  # xyxy of the mask, or None
    visible: bool = False
    backend: str = "none"

    def to_dict(self) -> dict:
        return {
            "available": self.mask is not None,
            "visible": self.visible,
            "score": round(float(self.score), 4),
            "area": int(self.area),
            "bbox": None if self.bbox is None else [round(float(v), 1) for v in self.bbox],
            "backend": self.backend,
        }


def mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float64)


def mask_coverage(mask: np.ndarray, box) -> float:
    """Fraction of mask pixels that fall inside ``box`` (xyxy)."""
    area = int(mask.sum())
    if area <= 0:
        return 0.0
    x1, y1, x2, y2 = (int(np.floor(v)) for v in np.asarray(box, float)[:4])
    h, w = mask.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float(mask[y1:y2, x1:x2].sum()) / area


def mask_box_iou(mask: np.ndarray, box) -> float:
    from oline_cv.tracking.expansion_iou import iou_matrix

    mb = mask_bbox(mask)
    if mb is None:
        return 0.0
    return float(iou_matrix(mb[None], np.asarray(box, float).reshape(1, 4))[0, 0])


class TargetMaskTracker:
    """Interface: prompt one box, propagate through time, query per frame."""

    name = "base"

    def initialize(self, frames, frame_idx: int, bbox, image_wh=None) -> MaskEvidence:
        raise NotImplementedError

    def reseed(self, frame_idx: int, bbox) -> MaskEvidence:
        raise NotImplementedError

    def propagate_forward(self, start: int, n: int | None = None) -> None:
        raise NotImplementedError

    def propagate_backward(self, start: int, n: int | None = None) -> None:
        raise NotImplementedError

    def propagate_all(self, lock_frame: int, n_frames: int, progress=None) -> None:
        """Run the whole clip once from the lock (SAMURAI / SAM 2 VOT usage)."""
        if lock_frame + 1 < n_frames:
            if progress:
                progress(f"SAM forward {lock_frame + 1}→{n_frames - 1}")
            self.propagate_forward(lock_frame + 1, n_frames - lock_frame - 1)
        if lock_frame > 0:
            if progress:
                progress(f"SAM backward {lock_frame - 1}→0")
            self.propagate_backward(lock_frame - 1, lock_frame)

    def get(self, frame_idx: int) -> MaskEvidence:
        raise NotImplementedError

    def ensure(self, frame_idx: int) -> MaskEvidence:
        ev = self.get(frame_idx)
        if ev.mask is None and frame_idx not in getattr(self, "_empty", set()):
            last = getattr(self, "_last_propagated", None)
            if last is None:
                return ev
            if frame_idx > last:
                self.propagate_forward(last + 1, frame_idx - last)
            elif frame_idx < last:
                self.propagate_backward(last - 1, last - frame_idx)
            ev = self.get(frame_idx)
        return ev

    def reset(self) -> None:
        pass

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name}


class NoneTargetMaskTracker(TargetMaskTracker):
    name = "none"

    def __init__(self) -> None:
        self._n = 0

    def initialize(self, frames, frame_idx: int, bbox, image_wh=None) -> MaskEvidence:
        self._n = len(frames)
        return MaskEvidence(frame_idx, None, backend=self.name)

    def reseed(self, frame_idx: int, bbox) -> MaskEvidence:
        return MaskEvidence(frame_idx, None, backend=self.name)

    def propagate_forward(self, start: int, n: int | None = None) -> None:
        return None

    def propagate_backward(self, start: int, n: int | None = None) -> None:
        return None

    def get(self, frame_idx: int) -> MaskEvidence:
        return MaskEvidence(frame_idx, None, backend=self.name)


class FakeTargetMaskTracker(TargetMaskTracker):
    """Deterministic masks for tests: ``masks[frame_idx] = bool array``."""

    name = "fake"

    def __init__(self, masks: dict[int, np.ndarray] | None = None, scores: dict[int, float] | None = None):
        self.masks = dict(masks or {})
        self.scores = dict(scores or {})
        self.anchor_frame: int | None = None
        self.anchor_bbox = None
        self._last_propagated = None

    def initialize(self, frames, frame_idx: int, bbox, image_wh=None) -> MaskEvidence:
        self.anchor_frame = int(frame_idx)
        self.anchor_bbox = np.asarray(bbox, float).copy()
        self._last_propagated = int(frame_idx)
        return self.get(frame_idx)

    def reseed(self, frame_idx: int, bbox) -> MaskEvidence:
        self.anchor_frame = int(frame_idx)
        self.anchor_bbox = np.asarray(bbox, float).copy()
        return self.get(frame_idx)

    def propagate_forward(self, start: int, n: int | None = None) -> None:
        self._last_propagated = start if n is None else start + n

    def propagate_backward(self, start: int, n: int | None = None) -> None:
        self._last_propagated = start if n is None else start - n

    def get(self, frame_idx: int) -> MaskEvidence:
        m = self.masks.get(int(frame_idx))
        if m is None:
            return MaskEvidence(frame_idx, None, backend=self.name)
        m = np.asarray(m, dtype=bool)
        area = int(m.sum())
        score = float(self.scores.get(int(frame_idx), 1.0 if area else 0.0))
        return MaskEvidence(
            frame_idx=frame_idx, mask=m, score=score, area=area,
            bbox=mask_bbox(m), visible=area > 0 and score > 0,
            backend=self.name,
        )

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "n_masks": len(self.masks)}


def _resolve_checkpoint(cfg) -> Path | None:
    raw = cfg.sam2_checkpoint or os.environ.get("OLINE_SAM2_CHECKPOINT") or ""
    if not raw:
        root = Path(__file__).resolve().parent.parent.parent
        cand = root / "models" / "sam2" / "sam2.1_hiera_tiny.pt"
        return cand if cand.is_file() else None
    p = Path(raw).expanduser()
    return p if p.is_file() else None


def sam2_setup_error(cfg) -> str:
    cfg_name = cfg.sam2_model_cfg
    return (
        "Target video segmentation (SAM 2.1) is not available. Place the official "
        f"checkpoint locally and set DeepHMSortConfig.sam2_checkpoint or "
        f"OLINE_SAM2_CHECKPOINT to that file. Expected config: {cfg_name} "
        "(from the sam2 package). Weights are never downloaded. See docs/tracking.md."
    )


class SAM2TargetMaskTracker(TargetMaskTracker):
    """Official SAM 2.1 video predictor, one object, lazy-loaded."""

    name = "sam2"

    def __init__(self, cfg, jpeg_quality: int = 90) -> None:
        self.cfg = cfg
        ckpt = _resolve_checkpoint(cfg)
        if ckpt is None:
            raise TargetMaskUnavailableError(sam2_setup_error(cfg))
        self.checkpoint = ckpt
        self.model_cfg = os.environ.get("OLINE_SAM2_MODEL_CONFIG") or cfg.sam2_model_cfg
        self._device_req = cfg.sam2_device
        self.jpeg_quality = jpeg_quality
        self._predictor = None
        self._state = None
        self._tmpdir = None
        self._cache: dict[int, MaskEvidence] = {}
        self._n = 0
        self._hw: tuple[int, int] = (0, 0)
        self._last_propagated: int | None = None
        self._empty: set[int] = set()
        self.min_area_frac = cfg.mask_visible_min_area_frac
        self.min_score = cfg.mask_visible_min_score

    def _device(self) -> str:
        if self._device_req:
            return self._device_req
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self) -> None:
        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as e:
            raise TargetMaskUnavailableError(
                "target_segmentation_backend='sam2' needs the sam2 package "
                f"(pip install sam2). Import failed: {e}"
            ) from e
        device = self._device()
        self._predictor = build_sam2_video_predictor(
            self.model_cfg, str(self.checkpoint), device=device
        )
        log.info("SAM 2.1 loaded from %s (%s) on %s", self.checkpoint, self.model_cfg, device)

    def _write_jpegs(self, frames: list[np.ndarray]) -> str:
        import cv2

        td = tempfile.TemporaryDirectory(prefix="oline_sam2_")
        self._tmpdir = td
        root = Path(td.name)
        for i, f in enumerate(frames):
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f.ndim == 3 else f
            cv2.imwrite(
                str(root / f"{i:05d}.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
        return str(root)

    def _store_logits(self, frame_idx: int, logits) -> MaskEvidence:
        if logits is None:
            ev = MaskEvidence(frame_idx, None, backend=self.name)
            self._cache[frame_idx] = ev
            self._empty.add(frame_idx)
            return ev
        t = logits
        if hasattr(t, "detach"):
            arr = t.detach().float().cpu().numpy()
        else:
            arr = np.asarray(t, dtype=np.float32)
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])
        mask = arr > 0
        fg = arr[mask]
        score = float(fg.mean()) if fg.size else 0.0
        if mask.shape != self._hw:
            import cv2
            mask = cv2.resize(mask.astype(np.uint8), (self._hw[1], self._hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        area = int(mask.sum())
        frac = area / max(1, self._hw[0] * self._hw[1])
        visible = bool(area > 0 and score >= self.min_score and frac >= self.min_area_frac)
        ev = MaskEvidence(
            frame_idx=frame_idx, mask=mask if visible or area > 0 else None,
            score=score, area=area, bbox=mask_bbox(mask) if area else None,
            visible=visible, backend=self.name,
        )
        if not visible:
            ev.mask = mask if area else None
        self._cache[frame_idx] = ev
        return ev

    def initialize(self, frames, frame_idx: int, bbox, image_wh=None) -> MaskEvidence:
        if self._predictor is None:
            self._load()
        self._n = len(frames)
        h, w = frames[0].shape[:2]
        self._hw = (h, w) if image_wh is None else (int(image_wh[1]), int(image_wh[0]))
        video_dir = self._write_jpegs(frames)
        self._state = self._predictor.init_state(
            video_path=video_dir, offload_video_to_cpu=True, offload_state_to_cpu=True
        )
        box = np.asarray(bbox, dtype=np.float32).reshape(4)
        _, _, logits = self._predictor.add_new_points_or_box(
            self._state, frame_idx=int(frame_idx), obj_id=1, box=box,
        )
        self._last_propagated = int(frame_idx)
        return self._store_logits(int(frame_idx), logits)

    def reseed(self, frame_idx: int, bbox) -> MaskEvidence:
        if self._predictor is None or self._state is None:
            raise TargetMaskUnavailableError("SAM2 tracker is not initialized")
        box = np.asarray(bbox, dtype=np.float32).reshape(4)
        _, _, logits = self._predictor.add_new_points_or_box(
            self._state, frame_idx=int(frame_idx), obj_id=1, box=box,
        )
        self._last_propagated = int(frame_idx)
        return self._store_logits(int(frame_idx), logits)

    def _propagate(self, start: int, n: int | None, reverse: bool) -> None:
        if self._predictor is None or self._state is None:
            return
        max_n = n
        n_done = 0
        for fidx, _ids, logits in self._predictor.propagate_in_video(
            self._state, start_frame_idx=int(start), max_frame_num_to_track=max_n, reverse=reverse,
        ):
            self._store_logits(int(fidx), logits)
            self._last_propagated = int(fidx)
            n_done += 1
            if n_done % 15 == 0:
                print(f"  SAM 2.1 {n_done} frames ({'rev' if reverse else 'fwd'})...", flush=True)

    def propagate_forward(self, start: int, n: int | None = None) -> None:
        self._propagate(start, n, False)

    def propagate_backward(self, start: int, n: int | None = None) -> None:
        self._propagate(start, n, True)

    def get(self, frame_idx: int) -> MaskEvidence:
        return self._cache.get(int(frame_idx), MaskEvidence(frame_idx, None, backend=self.name))

    def reset(self) -> None:
        self._cache.clear()
        self._empty.clear()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
        self._state = None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "checkpoint": str(self.checkpoint),
            "model_cfg": self.model_cfg,
            "device": self._device(),
            "frames_cached": len(self._cache),
        }


def build_target_mask_tracker(cfg, frames) -> tuple[TargetMaskTracker, dict[str, Any]]:
    """Construct the configured backend. Never downloads weights.

    Returns ``(tracker, status)``. ``status['available']`` is False when SAM
    was requested but cannot run; the caller should record degraded mode.
    """
    backend = cfg.target_segmentation_backend
    if not cfg.enable_target_segmentation or backend in ("none", "", None):
        return NoneTargetMaskTracker(), {
            "available": False, "backend": "none", "tracking_mode": "deep_hm_sort",
            "reason": "target_segmentation_disabled",
        }
    if backend == "sam2":
        try:
            tr = SAM2TargetMaskTracker(cfg)
            return tr, {
                "available": True, "backend": "sam2", "tracking_mode": "robust_target",
                **tr.describe(),
            }
        except TargetMaskUnavailableError as e:
            log.warning("%s", e)
            return NoneTargetMaskTracker(), {
                "available": False, "backend": "sam2", "tracking_mode": "degraded",
                "reason": str(e),
            }
    raise ValueError(f"unknown target_segmentation_backend {backend!r}")
