"""Contract for the RAW WHAM reconstruction (motion_raw.npz + motion_metadata.json).

Raw means exactly what WHAM produced, with only frame-index bookkeeping added.
No smoothing, no foot locking, no field calibration — those belong to later
phases and must be written to a separate cleaned artifact so the two can always
be compared.

Coordinate conventions, stated explicitly because silent axis swaps are the
classic failure mode here:

    WHAM camera space   x right, y DOWN, z forward (OpenCV-style)
    WHAM world space    y-UP, gravity-aligned (network called with return_y_up=True)
    Three.js world      x right, y UP, z toward viewer

    So world -> Three.js is a handedness change on z, NOT a y/z swap. The
    conversion is deliberately NOT applied here; it happens once, in the
    viewer-facing stage, and is documented there.

SMPL pose layout: axis-angle, 24 joints x 3 = 72 values per frame, joint 0 is
the root. ``pose_cam`` has the root in camera space, ``pose_world`` has the same
body pose with the root in gravity-aligned world space. Body joints 1..23 are
identical between the two.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MOTION_SCHEMA_VERSION = "1.0.0"

# Arrays the runner always writes. (name, expected trailing shape, dtype)
REQUIRED_ARRAYS: tuple[tuple[str, tuple[int, ...] | None], ...] = (
    ("frame_indices", ()),
    ("timestamps", ()),
    ("pose_cam", (72,)),
    ("pose_world", (72,)),
    ("betas", (10,)),
    ("trans_cam", (3,)),
    ("trans_world", (3,)),
    ("contact", (4,)),
)

# Arrays written when available / requested.
OPTIONAL_ARRAYS: tuple[str, ...] = (
    "joints_cam",
    "verts_cam",
    "keypoints_2d",
    "bbox_cxcys",
    # Per-frame provenance: association confidence, and whether the frame's box
    # was interpolated across a bridged detector dropout rather than observed.
    "frame_confidence",
    "interpolated",
)


class ReconstructionStatus(str):
    OK = "ok"
    OK_LOCAL_ONLY = "ok_local_only"  # ran, but no SLAM: world trajectory is not gravity-grounded
    FAILED = "failed"


@dataclass
class MotionMetadata:
    """Sidecar describing a reconstruction run. Written even on failure."""

    status: str
    video: str
    fps: float
    segment: dict[str, Any]
    frame_range: list[int]
    frame_count: int
    runtime_seconds: float
    outputs: dict[str, str] = field(default_factory=dict)
    wham: dict[str, Any] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)
    # Which detections were accepted as the tracked lineman, and which frames had
    # none. A reconstruction is only trustworthy if this holds up.
    association: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    schema_version: str = MOTION_SCHEMA_VERSION

    @property
    def world_grounded(self) -> bool:
        return bool(self.wham.get("world_grounded"))

    @property
    def ok(self) -> bool:
        return self.status in (ReconstructionStatus.OK, ReconstructionStatus.OK_LOCAL_ONLY)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "video": self.video,
            "fps": self.fps,
            "segment": self.segment,
            "frame_range": self.frame_range,
            "frame_count": self.frame_count,
            "runtime_seconds": round(float(self.runtime_seconds), 3),
            "outputs": self.outputs,
            "wham": self.wham,
            "device": self.device,
            "association": self.association,
            "warnings": self.warnings,
            "errors": self.errors,
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MotionMetadata:
        return cls(
            schema_version=str(d.get("schema_version", "")),
            status=str(d.get("status", ReconstructionStatus.FAILED)),
            video=str(d.get("video", "")),
            fps=float(d.get("fps") or 0.0),
            segment=dict(d.get("segment") or {}),
            frame_range=[int(v) for v in (d.get("frame_range") or [0, 0])],
            frame_count=int(d.get("frame_count") or 0),
            runtime_seconds=float(d.get("runtime_seconds") or 0.0),
            outputs=dict(d.get("outputs") or {}),
            wham=dict(d.get("wham") or {}),
            device=dict(d.get("device") or {}),
            association=dict(d.get("association") or {}),
            warnings=list(d.get("warnings") or []),
            errors=list(d.get("errors") or []),
            stats=dict(d.get("stats") or {}),
        )

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return p


def load_metadata(path: str | Path) -> MotionMetadata:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    version = str(data.get("schema_version", ""))
    if version and version.split(".")[0] != MOTION_SCHEMA_VERSION.split(".")[0]:
        raise ValueError(
            f"motion metadata schema {version!r} incompatible with {MOTION_SCHEMA_VERSION!r}"
        )
    return MotionMetadata.from_dict(data)


def validate_motion_npz(path: str | Path, expected_frames: int | None = None) -> dict[str, Any]:
    """Structural check of motion_raw.npz. Raises ValueError on contract breach.

    Returns a summary dict. This runs on the Windows side after the WSL job so a
    malformed reconstruction fails loudly here rather than in the viewer.
    """
    import numpy as np

    p = Path(path)
    if not p.exists():
        raise ValueError(f"motion npz not found: {p}")

    with np.load(p, allow_pickle=False) as data:
        keys = set(data.files)
        missing = [name for name, _ in REQUIRED_ARRAYS if name not in keys]
        if missing:
            raise ValueError(f"motion npz missing required arrays: {missing}")

        n = int(data["frame_indices"].shape[0])
        if expected_frames is not None and n != expected_frames:
            raise ValueError(
                f"motion npz has {n} frames, expected {expected_frames}"
            )

        for name, tail in REQUIRED_ARRAYS:
            arr = data[name]
            if arr.shape[0] != n:
                raise ValueError(
                    f"array {name!r} has {arr.shape[0]} rows, expected {n}"
                )
            if tail and tuple(arr.shape[1:]) != tail:
                raise ValueError(
                    f"array {name!r} has trailing shape {tuple(arr.shape[1:])}, expected {tail}"
                )

        idx = data["frame_indices"]
        if n > 1 and not bool(np.all(np.diff(idx) > 0)):
            raise ValueError("frame_indices must be strictly increasing")

        for name in ("pose_world", "trans_world"):
            if not bool(np.all(np.isfinite(data[name]))):
                raise ValueError(f"array {name!r} contains non-finite values")

        return {
            "frames": n,
            "frame_range": [int(idx[0]), int(idx[-1])],
            "arrays": sorted(keys),
            "optional_present": sorted(k for k in OPTIONAL_ARRAYS if k in keys),
        }
