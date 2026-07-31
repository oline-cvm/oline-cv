"""Render the association debug video.

One frame of output answers the only question that matters before trusting a
reconstruction: is the box WHAM is about to reconstruct the same body the 2D
tracker locked onto?

    cyan solid      TrackManifest target bbox (ground truth identity)
    green solid     selected detection -> this is what WHAM reconstructs
    red dashed      rejected candidates, labelled with why
    banner          frame index, confidence, IoU, validity, reject counts

Runs on the exported frames/ directory, so it works on the Windows side with no
WHAM present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from oline_cv.motion3d.target_association import FrameAssociation

COLOR_TARGET = (232, 196, 88)      # BGR - cyan/gold
COLOR_SELECTED = (110, 220, 130)   # green
COLOR_REJECTED = (70, 70, 220)     # red
COLOR_AMBIGUOUS = (60, 190, 250)   # amber
COLOR_BRIDGED = (240, 170, 60)     # orange - interpolated, not observed
COLOR_PANEL = (18, 20, 18)
COLOR_TEXT = (235, 238, 232)

FONT_SCALE = 0.5
LINE = 1


def _dashed_rect(img, p0, p1, color, thickness=2, dash=12) -> None:
    import cv2

    x0, y0 = p0
    x1, y1 = p1
    for x in range(x0, x1, dash * 2):
        cv2.line(img, (x, y0), (min(x + dash, x1), y0), color, thickness)
        cv2.line(img, (x, y1), (min(x + dash, x1), y1), color, thickness)
    for y in range(y0, y1, dash * 2):
        cv2.line(img, (x0, y), (x0, min(y + dash, y1)), color, thickness)
        cv2.line(img, (x1, y), (x1, min(y + dash, y1)), color, thickness)


def _label(img, text, org, color, scale=FONT_SCALE) -> None:
    import cv2

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, LINE)
    x, y = org
    y = max(y, th + 4)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 6, y + 3), COLOR_PANEL, -1)
    cv2.putText(
        img, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, LINE, cv2.LINE_AA
    )


def draw_association(img, assoc: FrameAssociation, scale: float = 1.0):
    """Draw one frame's association state onto ``img`` (modified in place)."""
    import cv2

    def pt(x, y):
        return int(round(float(x) * scale)), int(round(float(y) * scale))

    for cand in assoc.candidates:
        if cand.selected:
            continue
        color = COLOR_AMBIGUOUS if cand.reject_reason == "not_best" else COLOR_REJECTED
        p0, p1 = pt(cand.bbox[0], cand.bbox[1]), pt(cand.bbox[2], cand.bbox[3])
        _dashed_rect(img, p0, p1, color, thickness=2)
        # Only the candidates that actually competed for the identity get labels;
        # labelling all ~16 detections buries the frame in text.
        if cand.iou > 0.02 or cand.center_dist_frac < 1.0:
            _label(
                img,
                f"{cand.reject_reason or 'rejected'} iou{cand.iou:.2f} s{cand.score:.2f}",
                (p0[0], p0[1] - 4),
                color,
                scale=0.42,
            )

    if assoc.target_bbox is not None:
        p0, p1 = pt(assoc.target_bbox[0], assoc.target_bbox[1]), pt(
            assoc.target_bbox[2], assoc.target_bbox[3]
        )
        cv2.rectangle(img, p0, p1, COLOR_TARGET, 2)
        _label(img, "TARGET (BoT-SORT)", (p0[0], p1[1] + 18), COLOR_TARGET, scale=0.45)

    sel = assoc.selected
    if sel is not None:
        p0, p1 = pt(sel.bbox[0], sel.bbox[1]), pt(sel.bbox[2], sel.bbox[3])
        if sel.interpolated:
            # Interpolated, not observed: drawn dashed so it can never be mistaken
            # for a detection in a debug review.
            _dashed_rect(img, p0, p1, COLOR_BRIDGED, thickness=3, dash=16)
            _label(
                img,
                f"BRIDGED (interpolated)  conf {assoc.confidence:.3f}",
                (p0[0], p0[1] - 22),
                COLOR_BRIDGED,
                scale=0.5,
            )
        else:
            cv2.rectangle(img, p0, p1, COLOR_SELECTED, 3)
            _label(
                img,
                f"WHAM SELECTED  conf {sel.score:.3f}  iou {sel.iou:.3f}",
                (p0[0], p0[1] - 22),
                COLOR_SELECTED,
                scale=0.5,
            )

    h, w = img.shape[:2]
    banner_h = 30
    cv2.rectangle(img, (0, 0), (w, banner_h), COLOR_PANEL, -1)
    n_rej = sum(1 for c in assoc.candidates if not c.selected)
    if assoc.bridged:
        status = "BRIDGED"
        status_color = COLOR_BRIDGED
    elif assoc.valid:
        status = "AMBIGUOUS" if assoc.ambiguous else "MATCHED"
        status_color = COLOR_AMBIGUOUS if assoc.ambiguous else COLOR_SELECTED
    else:
        status = f"INVALID: {assoc.invalid_reason}"
        status_color = COLOR_REJECTED
    text = (
        f"frame {assoc.frame_index:>5}   {status}   "
        f"conf {assoc.confidence:.3f}   candidates {len(assoc.candidates)}   rejected {n_rej}"
    )
    cv2.putText(
        img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, LINE, cv2.LINE_AA
    )
    return img


def render_association_video(
    associations: Sequence[FrameAssociation],
    frames_dir: str | Path,
    out_path: str | Path,
    fps: float = 30.0,
    max_width: int = 1280,
    frame_pattern: str = "{:06d}.jpg",
) -> dict[str, Any]:
    """Write the debug video from exported full frames + association records."""
    import cv2

    frames_dir = Path(frames_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    scale = 1.0
    written = 0
    missing: list[int] = []

    for assoc in associations:
        fp = frames_dir / frame_pattern.format(assoc.frame_index)
        img = cv2.imread(str(fp))
        if img is None:
            missing.append(assoc.frame_index)
            continue
        if writer is None:
            h, w = img.shape[:2]
            scale = min(1.0, max_width / float(w))
            size = (int(round(w * scale)), int(round(h * scale)))
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size
            )
            if not writer.isOpened():
                raise RuntimeError(f"could not open video writer for {out_path}")
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        draw_association(img, assoc, scale=scale)
        writer.write(img)
        written += 1

    if writer is not None:
        writer.release()

    return {
        "path": str(out_path),
        "frames_written": written,
        "frames_missing": missing,
        "scale": round(scale, 4),
    }


def render_selected_crops_sheet(
    associations: Sequence[FrameAssociation],
    frames_dir: str | Path,
    out_path: str | Path,
    count: int = 10,
    cols: int = 5,
    tile_height: int = 300,
    pad: float = 0.12,
    frame_pattern: str = "{:06d}.jpg",
) -> dict[str, Any]:
    """Zoom on the selected body only, big enough to read a jersey number.

    The whole-frame view proves the selected box sits on the target box; this
    proves the target box is the right human.
    """
    import cv2
    import numpy as np

    frames_dir = Path(frames_dir)
    valid = [a for a in associations if a.valid and a.selected is not None]
    if not valid:
        return {"path": str(out_path), "tiles": 0}

    if len(valid) <= count:
        picked = valid
    else:
        idxs = sorted({int(round(i)) for i in np.linspace(0, len(valid) - 1, count)})
        picked = [valid[i] for i in idxs]

    tiles = []
    for assoc in picked:
        img = cv2.imread(str(frames_dir / frame_pattern.format(assoc.frame_index)))
        if img is None:
            continue
        h, w = img.shape[:2]
        x0, y0, x1, y1 = assoc.selected.bbox
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
        x1, y1 = int(min(w, x1 + px)), int(min(h, y1 + py))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        scale = tile_height / float(crop.shape[0])
        crop = cv2.resize(
            crop, (max(1, int(round(crop.shape[1] * scale))), tile_height),
            interpolation=cv2.INTER_CUBIC,
        )
        _label(crop, f"f{assoc.frame_index}  {assoc.confidence:.2f}", (4, 16), COLOR_SELECTED, 0.45)
        tiles.append(crop)

    if not tiles:
        return {"path": str(out_path), "tiles": 0}

    tw = max(t.shape[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * tile_height, cols * tw, 3), dtype=tiles[0].dtype)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * tile_height : (r + 1) * tile_height, c * tw : c * tw + t.shape[1]] = t

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return {"path": str(out_path), "tiles": len(tiles)}


def render_association_sheet(
    associations: Sequence[FrameAssociation],
    frames_dir: str | Path,
    out_path: str | Path,
    count: int = 12,
    cols: int = 4,
    tile_width: int = 480,
    frame_pattern: str = "{:06d}.jpg",
    prefer: str = "spread",
) -> dict[str, Any]:
    """Contact sheet of association frames, for a quick visual check in chat.

    ``prefer='invalid'`` samples the problem frames instead of spreading evenly.
    """
    import cv2
    import numpy as np

    frames_dir = Path(frames_dir)
    pool = list(associations)
    if prefer == "invalid":
        # Anything a reviewer should look at twice: no match, uncertain identity,
        # or a box that was interpolated rather than observed.
        bad = [a for a in pool if not a.valid or a.ambiguous or a.bridged]
        pool = bad or pool
    if not pool:
        return {"path": str(out_path), "tiles": 0}

    if len(pool) <= count:
        picked = pool
    else:
        idxs = np.linspace(0, len(pool) - 1, count)
        picked = [pool[int(round(i))] for i in sorted({int(round(i)) for i in idxs})]

    tiles = []
    for assoc in picked:
        img = cv2.imread(str(frames_dir / frame_pattern.format(assoc.frame_index)))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = tile_width / float(w)
        img = cv2.resize(img, (tile_width, int(round(h * scale))), interpolation=cv2.INTER_AREA)
        draw_association(img, assoc, scale=scale)
        tiles.append(img)

    if not tiles:
        return {"path": str(out_path), "tiles": 0}

    th, tw = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), dtype=tiles[0].dtype)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * th : r * th + t.shape[0], c * tw : c * tw + t.shape[1]] = t

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return {"path": str(out_path), "tiles": len(tiles)}
