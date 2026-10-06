"""Appearance (ReID) embeddings for tracking.

Backends
--------
``osnet``
    torchreid OSNet loaded from an explicit checkpoint (``reid_weights`` or
    env ``OLINE_REID_WEIGHTS``), e.g. a sports-trained OSNet such as the one
    released with Deep-EIoU. Nothing is ever downloaded: a missing path or a
    checkpoint that does not fit the architecture is a hard error, never
    random weights.
``yolo_embed``
    Pooled features from the person detector OLINE already uses. Generic,
    not trained for re-identification, and clearly lower quality; reported
    as ``quality="generic"`` in the analysis metadata.
``auto``
    ``osnet`` when weights are configured, otherwise ``yolo_embed``.

Models load on the first ``extract`` call, so constructing an extractor (and
importing this module) does not import torch.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ReIDUnavailableError(RuntimeError):
    """A ReID backend was requested but cannot be loaded."""


def crop_boxes(frame: np.ndarray, boxes, min_px: int = 12) -> list[np.ndarray | None]:
    """Clip ``xyxy`` boxes to the frame and cut crops; tiny boxes give None."""
    h, w = frame.shape[:2]
    out: list[np.ndarray | None] = []
    for b in np.asarray(boxes, dtype=np.float64).reshape(-1, 4):
        if not np.all(np.isfinite(b)):
            out.append(None)
            continue
        x1, y1 = int(max(0, np.floor(b[0]))), int(max(0, np.floor(b[1])))
        x2, y2 = int(min(w, np.ceil(b[2]))), int(min(h, np.ceil(b[3])))
        if x2 - x1 < min_px or y2 - y1 < min_px:
            out.append(None)
            continue
        out.append(frame[y1:y2, x1:x2])
    return out


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    out = x / np.maximum(n, 1e-12)
    out[n[:, 0] <= 1e-12] = np.nan
    return out.astype(np.float32)


class ReIDExtractor:
    """Base class: ``extract(frame, boxes) -> (N, D)`` L2-normalised rows.

    Rows for crops that could not be embedded are NaN, which the cost code
    treats as "no appearance evidence" (distance 1).
    """

    name = "base"
    quality = "unknown"

    def __init__(self, *, batch_size: int = 32, min_crop_px: int = 12) -> None:
        self.batch_size = max(1, int(batch_size))
        self.min_crop_px = int(min_crop_px)
        self.dim: int | None = None

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def extract(self, frame: np.ndarray, boxes) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        crops = crop_boxes(frame, boxes, self.min_crop_px)
        valid = [i for i, c in enumerate(crops) if c is not None]
        if not valid:
            return np.full((len(boxes), self.dim or 1), np.nan, dtype=np.float32)
        feats = []
        for s in range(0, len(valid), self.batch_size):
            feats.append(self._embed([crops[i] for i in valid[s : s + self.batch_size]]))
        f = _l2(np.concatenate(feats, axis=0).astype(np.float32))
        self.dim = f.shape[1]
        out = np.full((len(boxes), self.dim), np.nan, dtype=np.float32)
        out[valid] = f
        return out

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "quality": self.quality, "dim": self.dim}


def _resolve_device(device: str | None) -> str:
    if device:
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class OSNetExtractor(ReIDExtractor):
    """torchreid OSNet from an explicit, validated checkpoint."""

    name = "osnet"
    quality = "sports_reid"

    def __init__(
        self,
        weights: str | os.PathLike,
        *,
        model_name: str = "osnet_x1_0",
        device: str | None = None,
        input_hw: tuple[int, int] = (256, 128),
        batch_size: int = 32,
        min_crop_px: int = 12,
    ) -> None:
        super().__init__(batch_size=batch_size, min_crop_px=min_crop_px)
        self.weights = Path(weights).expanduser()
        if not self.weights.is_file():
            raise ReIDUnavailableError(
                f"OSNet ReID weights not found at '{self.weights}'. Set "
                "DeepHMSortConfig.reid_weights or OLINE_REID_WEIGHTS to a local "
                "checkpoint (see docs/tracking.md); weights are never downloaded."
            )
        self.model_name = model_name
        self._device_req = device
        self.input_hw = input_hw
        self._model = None
        self._device = None

    def _load(self) -> None:
        try:
            import torch
            import torchreid
        except ImportError as e:
            raise ReIDUnavailableError(
                "reid_backend='osnet' needs torchreid (pip install torchreid). "
                f"Import failed: {e}"
            ) from e
        self._device = _resolve_device(self._device_req)
        model = torchreid.models.build_model(
            name=self.model_name, num_classes=1, pretrained=False
        )
        ckpt = torch.load(str(self.weights), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        if not isinstance(state, dict):
            raise ReIDUnavailableError(f"Unrecognised checkpoint format in {self.weights}")
        state = {k.removeprefix("module."): v for k, v in state.items()}
        own = model.state_dict()
        fit = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        backbone = [k for k in own if not k.startswith("classifier")]
        loaded = sum(1 for k in backbone if k in fit)
        if loaded < 0.9 * len(backbone):
            raise ReIDUnavailableError(
                f"{self.weights} does not match {self.model_name}: only "
                f"{loaded}/{len(backbone)} backbone tensors loaded. Refusing to "
                "run ReID with mostly random weights."
            )
        model.load_state_dict(fit, strict=False)
        model.eval().to(self._device)
        self._model = model
        log.info("OSNet ReID loaded from %s on %s (%d/%d tensors)",
                 self.weights, self._device, loaded, len(backbone))

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
        import cv2
        import torch

        if self._model is None:
            self._load()
        h, w = self.input_hw
        batch = np.stack([
            (cv2.cvtColor(cv2.resize(c, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
             - _IMAGENET_MEAN) / _IMAGENET_STD
            for c in crops
        ]).transpose(0, 3, 1, 2)
        with torch.no_grad():
            f = self._model(torch.from_numpy(batch).to(self._device))
        return f.detach().cpu().numpy()

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(weights=str(self.weights), model=self.model_name, device=self._device)
        return d


class YoloEmbedExtractor(ReIDExtractor):
    """Generic fallback: pooled features from the YOLO person detector.

    Not trained for re-identification — expect weaker separation between
    players in similar uniforms than a sports ReID model.
    """

    name = "yolo_embed"
    quality = "generic"

    def __init__(
        self,
        model_path: str,
        *,
        imgsz: int = 160,
        device: str | None = None,
        batch_size: int = 32,
        min_crop_px: int = 12,
    ) -> None:
        super().__init__(batch_size=batch_size, min_crop_px=min_crop_px)
        self.model_path = model_path
        self.imgsz = imgsz
        self._device_req = device
        self._model = None

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if self._model is None:
            from ultralytics import YOLO

            # Own instance: embed() changes predictor args, which must not leak
            # into the detector's predict calls.
            self._model = YOLO(self.model_path)
        kw = {"imgsz": self.imgsz, "verbose": False}
        if self._device_req:
            kw["device"] = self._device_req
        feats = self._model.embed(crops, **kw)
        return np.stack([np.asarray(f.detach().cpu().numpy(), dtype=np.float32).ravel() for f in feats])

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(model=self.model_path,
                 warning="generic detector features, not a sports ReID model")
        return d


def build_reid_extractor(cfg, detect_model: str) -> ReIDExtractor:
    """Build the extractor named by ``cfg.reid_backend`` (a DeepHMSortConfig)."""
    weights = cfg.reid_weights or os.environ.get("OLINE_REID_WEIGHTS") or None
    backend = cfg.reid_backend
    common = {"batch_size": cfg.reid_batch_size, "min_crop_px": cfg.reid_min_crop_px}
    if backend == "osnet" or (backend == "auto" and weights):
        if not weights:
            raise ReIDUnavailableError(
                "reid_backend='osnet' needs reid_weights or OLINE_REID_WEIGHTS."
            )
        return OSNetExtractor(weights, model_name=cfg.reid_model_name,
                              device=cfg.reid_device, **common)
    if backend in ("auto", "yolo_embed"):
        if backend == "auto":
            log.warning("No ReID weights configured; using generic detector "
                        "embeddings (lower quality). See docs/tracking.md.")
        return YoloEmbedExtractor(detect_model, device=cfg.reid_device, **common)
    raise ValueError(f"unknown reid_backend {backend!r}")
