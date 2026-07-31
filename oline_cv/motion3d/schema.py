"""Serialization contract between BoT-SORT tracking and the 3D motion stage.

This is a stable, versioned artifact. The HMR stage (WHAM/GVHMR) runs in a
separate environment and consumes only this file plus the exported images, so
any change here is a cross-process breaking change — bump SCHEMA_VERSION.

Coordinate conventions used throughout this module (nothing is implicit):

    image pixels    x right, y DOWN, origin top-left, units = pixels
    normalized      image pixels divided by (width, height), range [0, 1]
    crop pixels     x right, y DOWN, origin at crop box top-left, units = pixels

Conversions:
    crop_px  = (image_px - crop.box[:2]) * crop.scale
    image_px = crop_px / crop.scale + crop.box[:2]

No 3D or field coordinates appear in this file. Phase 1 is strictly 2D.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0.0"


class BboxSource(str, Enum):
    """Where a frame's bbox came from, so downstream stages can weight it."""

    DETECTED = "detected"          # BoT-SORT produced a box this frame
    INTERPOLATED = "interpolated"  # linearly filled across a short gap
    CARRIED = "carried"            # copied from the nearest valid neighbour
    MISSING = "missing"            # no box available; frame is not reconstructable


@dataclass
class CropRef:
    """A player crop on disk plus the exact transform used to make it.

    ``box`` is the source rectangle in full-frame pixels [x0, y0, x1, y1].
    ``scale`` maps source pixels to output pixels. Crops are square and
    aspect-preserving, so a single scalar scale is sufficient.
    """

    path: str
    box: list[float]
    size: int
    scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "box": [round(float(v), 3) for v in self.box],
            "size": int(self.size),
            "scale": round(float(self.scale), 6),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CropRef:
        return cls(
            path=str(d["path"]),
            box=[float(v) for v in d["box"]],
            size=int(d["size"]),
            scale=float(d["scale"]),
        )

    def image_to_crop(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.box[0]) * self.scale, (y - self.box[1]) * self.scale

    def crop_to_image(self, x: float, y: float) -> tuple[float, float]:
        return x / self.scale + self.box[0], y / self.scale + self.box[1]


@dataclass
class TrackFrame:
    """One frame of the locked OL's track.

    ``keypoints_2d`` is COCO-17 in full-frame pixels, ordered by
    ``config.KEYPOINT_NAMES``. Each entry is [x, y, confidence]; a missing joint
    is [None, None, 0.0] rather than being dropped, so the array length is
    always 17 and index == joint id.
    """

    frame_index: int
    timestamp: float
    track_id: int | None
    target_id: int
    bbox: list[float] | None
    bbox_source: str
    detection_confidence: float
    track_state: str
    track_confidence: float
    keypoints_2d: list[list[float | None]]
    low_confidence: bool
    usable: bool
    frame_path: str | None = None
    crop: CropRef | None = None

    @property
    def reconstructable(self) -> bool:
        """Whether the HMR stage can use this frame at all."""
        return self.bbox is not None and self.bbox_source != BboxSource.MISSING.value

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "frame_index": int(self.frame_index),
            "timestamp": round(float(self.timestamp), 6),
            "track_id": None if self.track_id is None else int(self.track_id),
            "target_id": int(self.target_id),
            "bbox": None if self.bbox is None else [round(float(v), 2) for v in self.bbox],
            "bbox_source": str(self.bbox_source),
            "detection_confidence": round(float(self.detection_confidence), 4),
            "track_state": str(self.track_state),
            "track_confidence": round(float(self.track_confidence), 4),
            "keypoints_2d": [
                [
                    None if kp[0] is None else round(float(kp[0]), 2),
                    None if kp[1] is None else round(float(kp[1]), 2),
                    round(float(kp[2] or 0.0), 4),
                ]
                for kp in self.keypoints_2d
            ],
            "low_confidence": bool(self.low_confidence),
            "usable": bool(self.usable),
        }
        if self.frame_path:
            d["frame_path"] = self.frame_path
        if self.crop:
            d["crop"] = self.crop.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrackFrame:
        return cls(
            frame_index=int(d["frame_index"]),
            timestamp=float(d["timestamp"]),
            track_id=None if d.get("track_id") is None else int(d["track_id"]),
            target_id=int(d.get("target_id", 1)),
            bbox=None if d.get("bbox") is None else [float(v) for v in d["bbox"]],
            bbox_source=str(d.get("bbox_source", BboxSource.MISSING.value)),
            detection_confidence=float(d.get("detection_confidence", 0.0)),
            track_state=str(d.get("track_state", "LOST")),
            track_confidence=float(d.get("track_confidence", 0.0)),
            keypoints_2d=[
                [
                    None if kp[0] is None else float(kp[0]),
                    None if kp[1] is None else float(kp[1]),
                    float(kp[2] or 0.0),
                ]
                for kp in d.get("keypoints_2d", [])
            ],
            low_confidence=bool(d.get("low_confidence", True)),
            usable=bool(d.get("usable", False)),
            frame_path=d.get("frame_path"),
            crop=CropRef.from_dict(d["crop"]) if d.get("crop") else None,
        )


@dataclass
class TrackManifest:
    """Everything the HMR stage needs to run without re-reading our Python code."""

    video: dict[str, Any]
    target: dict[str, Any]
    export: dict[str, Any]
    frames: list[TrackFrame] = field(default_factory=list)
    segments: list[list[int]] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    @property
    def fps(self) -> float:
        return float(self.video.get("fps") or 30.0)

    def frame_by_index(self, frame_index: int) -> TrackFrame | None:
        for fr in self.frames:
            if fr.frame_index == frame_index:
                return fr
        return None

    def stats(self) -> dict[str, Any]:
        total = len(self.frames)
        by_source: dict[str, int] = {}
        for fr in self.frames:
            by_source[fr.bbox_source] = by_source.get(fr.bbox_source, 0) + 1
        return {
            "frames": total,
            "reconstructable": sum(1 for f in self.frames if f.reconstructable),
            "usable": sum(1 for f in self.frames if f.usable),
            "bbox_source": by_source,
            "segments": len(self.segments),
            "longest_segment": max((b - a + 1 for a, b in self.segments), default=0),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video": self.video,
            "target": self.target,
            "export": self.export,
            "segments": [[int(a), int(b)] for a, b in self.segments],
            "stats": self.stats(),
            "frames": [f.to_dict() for f in self.frames],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TrackManifest:
        return cls(
            schema_version=str(d.get("schema_version", "")),
            video=dict(d.get("video") or {}),
            target=dict(d.get("target") or {}),
            export=dict(d.get("export") or {}),
            segments=[[int(a), int(b)] for a, b in (d.get("segments") or [])],
            frames=[TrackFrame.from_dict(f) for f in (d.get("frames") or [])],
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


def load_manifest(path: str | Path) -> TrackManifest:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = str(data.get("schema_version", ""))
    if version.split(".")[0] != SCHEMA_VERSION.split(".")[0]:
        raise ValueError(
            f"tracks manifest schema {version!r} is incompatible with {SCHEMA_VERSION!r}"
        )
    return TrackManifest.from_dict(data)


def asdict_shallow(obj: Any) -> dict[str, Any]:
    return asdict(obj)
