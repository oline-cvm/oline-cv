"""Phase 1 — export the locked OL's BoT-SORT track, full frames, and crops.

This module never re-runs or second-guesses tracking. It serializes the
``FramePose`` records ``PoseTracker`` already produced, so identity is exactly
what the existing association stack decided.

Why full frames are exported alongside crops: world-grounded HMR (WHAM/GVHMR)
needs scene context to separate camera motion from player motion. Running the
model on isolated crops throws that away and is the main reason the previous
3D replay could not tell a camera pan from a kick slide.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from oline_cv.config import KEYPOINT_NAMES
from oline_cv.motion3d.schema import (
    SCHEMA_VERSION,
    BboxSource,
    CropRef,
    TrackFrame,
    TrackManifest,
)

DEFAULT_CROP_SIZE = 256
DEFAULT_CROP_PAD = 0.25
DEFAULT_MAX_INTERP_GAP = 8


def square_crop_box(
    bbox: Sequence[float],
    pad: float = DEFAULT_CROP_PAD,
    min_side: float = 32.0,
) -> list[float]:
    """Square, aspect-preserving crop box around a bbox, in full-frame pixels.

    The box is intentionally NOT clamped to the frame. Clamping would break the
    square aspect and silently change the pixel-to-crop scale, which corrupts
    keypoint round-trips. Out-of-frame regions are letterboxed with black at
    render time instead, keeping ``CropRef.scale`` a single exact scalar.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(float(x1 - x0), float(y1 - y0)) * (1.0 + 2.0 * float(pad))
    side = max(side, float(min_side))
    half = side / 2.0
    return [cx - half, cy - half, cx + half, cy + half]


def render_crop(
    frame_bgr: np.ndarray,
    box: Sequence[float],
    size: int = DEFAULT_CROP_SIZE,
) -> tuple[np.ndarray, float]:
    """Cut ``box`` out of ``frame_bgr``, letterboxing out-of-frame area.

    Returns the (size, size, 3) crop and the source->output scale factor.
    """
    import cv2

    fh, fw = frame_bgr.shape[:2]
    x0, y0, x1, y1 = (int(round(float(v))) for v in box)
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    canvas = np.zeros((bh, bw, 3), dtype=frame_bgr.dtype)

    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(fw, x1), min(fh, y1)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = frame_bgr[sy0:sy1, sx0:sx1]

    interp = cv2.INTER_AREA if bw > size else cv2.INTER_LINEAR
    out = cv2.resize(canvas, (size, size), interpolation=interp)
    return out, float(size) / float(bw)


def fill_bbox_gaps(
    bboxes: Sequence[Sequence[float] | None],
    max_interp_gap: int = DEFAULT_MAX_INTERP_GAP,
) -> tuple[list[list[float] | None], list[str]]:
    """Fill short gaps in a bbox sequence; leave long gaps as MISSING.

    Interpolation is linear between the bracketing detections. Gaps at the head
    or tail of the sequence, and gaps longer than ``max_interp_gap``, are not
    invented — the spec is explicit that we must not fabricate long stretches of
    motion. Short one-sided gaps carry the nearest neighbour instead.
    """
    n = len(bboxes)
    out: list[list[float] | None] = [None if b is None else [float(v) for v in b] for b in bboxes]
    sources: list[str] = [
        BboxSource.DETECTED.value if b is not None else BboxSource.MISSING.value for b in bboxes
    ]

    detected = [i for i, b in enumerate(out) if b is not None]
    if not detected:
        return out, sources

    for a, b in zip(detected, detected[1:]):
        gap = b - a - 1
        if gap <= 0 or gap > max_interp_gap:
            continue
        left, right = out[a], out[b]
        assert left is not None and right is not None
        for k in range(1, gap + 1):
            t = k / (gap + 1)
            out[a + k] = [left[j] + t * (right[j] - left[j]) for j in range(4)]
            sources[a + k] = BboxSource.INTERPOLATED.value

    first, last = detected[0], detected[-1]
    for i in range(max(0, first - max_interp_gap), first):
        out[i] = list(out[first])  # type: ignore[arg-type]
        sources[i] = BboxSource.CARRIED.value
    for i in range(last + 1, min(n, last + 1 + max_interp_gap)):
        out[i] = list(out[last])  # type: ignore[arg-type]
        sources[i] = BboxSource.CARRIED.value

    return out, sources


def valid_segments(sources: Sequence[str], min_length: int = 2) -> list[list[int]]:
    """Contiguous [start, end] index ranges that the HMR stage can reconstruct."""
    segments: list[list[int]] = []
    start: int | None = None
    for i, s in enumerate(sources):
        ok = s != BboxSource.MISSING.value
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            if i - start >= min_length:
                segments.append([start, i - 1])
            start = None
    if start is not None and len(sources) - start >= min_length:
        segments.append([start, len(sources) - 1])
    return segments


def _keypoints_to_list(xy: np.ndarray, conf: np.ndarray) -> list[list[float | None]]:
    out: list[list[float | None]] = []
    for i in range(len(KEYPOINT_NAMES)):
        if i >= len(xy):
            out.append([None, None, 0.0])
            continue
        x, y = float(xy[i][0]), float(xy[i][1])
        c = float(conf[i]) if i < len(conf) else 0.0
        if not np.isfinite(x) or not np.isfinite(y):
            out.append([None, None, c])
        else:
            out.append([x, y, c])
    return out


def export_tracks(
    video_path: str,
    ol_poses: Sequence[Any],
    frames: Sequence[np.ndarray],
    fps: float,
    width: int,
    height: int,
    out_dir: str | Path,
    *,
    target_jersey: int | None = None,
    ol_lock: dict[str, Any] | None = None,
    snap_frame: int | None = None,
    set_end: int | None = None,
    crop_size: int = DEFAULT_CROP_SIZE,
    crop_pad: float = DEFAULT_CROP_PAD,
    max_interp_gap: int = DEFAULT_MAX_INTERP_GAP,
    save_full_frames: bool = True,
    jpeg_quality: int = 92,
    progress_cb=None,
) -> TrackManifest:
    """Write ``tracks.json`` plus ``frames/`` and ``crops/`` for the locked OL.

    ``ol_poses`` and ``frames`` come straight from ``PoseTracker.extract_all``.
    The full clip is exported, not just the snap..set_end window, because HMR
    needs temporal lead-in and lead-out to stabilize.
    """
    import cv2

    out = Path(out_dir)
    crops_dir = out / "crops"
    frames_dir = out / "frames"
    crops_dir.mkdir(parents=True, exist_ok=True)
    if save_full_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)

    def _prog(pct: float, msg: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(float(pct), str(msg), "export")
        except Exception:
            pass

    raw_boxes: list[list[float] | None] = []
    for p in ol_poses:
        bb = getattr(p, "bbox_xyxy", None)
        raw_boxes.append(None if bb is None else [float(v) for v in bb])
    boxes, sources = fill_bbox_gaps(raw_boxes, max_interp_gap=max_interp_gap)

    encode = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    track_frames: list[TrackFrame] = []
    n = len(ol_poses)

    for i, pose in enumerate(ol_poses):
        frame_bgr = frames[i] if i < len(frames) else None
        box = boxes[i]
        source = sources[i]

        frame_path: str | None = None
        if save_full_frames and frame_bgr is not None:
            fp = frames_dir / f"{i:06d}.jpg"
            cv2.imwrite(str(fp), frame_bgr, encode)
            frame_path = f"frames/{i:06d}.jpg"

        crop_ref: CropRef | None = None
        if box is not None and frame_bgr is not None:
            cbox = square_crop_box(box, pad=crop_pad)
            crop_img, scale = render_crop(frame_bgr, cbox, size=crop_size)
            cp = crops_dir / f"{i:06d}.jpg"
            cv2.imwrite(str(cp), crop_img, encode)
            crop_ref = CropRef(
                path=f"crops/{i:06d}.jpg", box=cbox, size=crop_size, scale=scale
            )

        track_frames.append(
            TrackFrame(
                frame_index=int(getattr(pose, "frame_idx", i)),
                timestamp=float(getattr(pose, "timestamp_ms", (i / fps) * 1000.0)) / 1000.0,
                track_id=getattr(pose, "track_id", None),
                target_id=int(getattr(pose, "target_id", 1)),
                bbox=box,
                bbox_source=source,
                detection_confidence=float(getattr(pose, "person_confidence", 0.0) or 0.0),
                track_state=str(getattr(pose, "track_state", "LOST")),
                track_confidence=float(getattr(pose, "track_confidence", 0.0) or 0.0),
                keypoints_2d=_keypoints_to_list(
                    getattr(pose, "keypoints_xy", np.full((17, 2), np.nan)),
                    getattr(pose, "keypoints_conf", np.zeros(17)),
                ),
                low_confidence=bool(getattr(pose, "low_confidence", True)),
                usable=bool(getattr(pose, "usable", False)),
                frame_path=frame_path,
                crop=crop_ref,
            )
        )

        if n and i % 25 == 0:
            _prog(100.0 * i / n, f"Exporting track frames ({i}/{n})…")

    manifest = TrackManifest(
        schema_version=SCHEMA_VERSION,
        video={
            "path": str(video_path),
            "fps": float(fps),
            "frame_count": int(n),
            "width": int(width),
            "height": int(height),
        },
        target={
            "target_id": int(getattr(ol_poses[0], "target_id", 1)) if ol_poses else 1,
            "jersey": target_jersey,
            "snap_frame": snap_frame,
            "set_end": set_end,
            "lock": ol_lock or {},
        },
        export={
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "crop_size": int(crop_size),
            "crop_pad": float(crop_pad),
            "max_interp_gap": int(max_interp_gap),
            "full_frames": bool(save_full_frames),
            "keypoint_format": "coco17",
            "keypoint_names": list(KEYPOINT_NAMES),
            "coordinate_space": "image_pixels_x_right_y_down_origin_topleft",
        },
        frames=track_frames,
        # segments are reported in frame_index space, not list-position space
        segments=[
            [track_frames[a].frame_index, track_frames[b].frame_index]
            for a, b in valid_segments(sources)
        ],
    )
    manifest.save(out / "tracks.json")
    _prog(100.0, "Track export complete")
    return manifest


def iter_segment_frames(
    manifest: TrackManifest, segment: Sequence[int]
) -> Iterable[TrackFrame]:
    a, b = int(segment[0]), int(segment[1])
    for fr in manifest.frames:
        if a <= fr.frame_index <= b:
            yield fr
