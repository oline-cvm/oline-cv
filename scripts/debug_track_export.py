"""Phase 1 debug — contact sheet of exported crops with 2D keypoints overlaid.

Keypoints are drawn using CropRef.image_to_crop, so if the crop transform is
wrong the skeleton will visibly detach from the body. This is the cheapest
guard against a silent coordinate bug reaching the HMR stage.

    python scripts/debug_track_export.py outputs/motion3d/footage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.config import KEYPOINT_NAMES  # noqa: E402
from oline_cv.motion3d import BboxSource, load_manifest  # noqa: E402

SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

SOURCE_COLOR = {
    BboxSource.DETECTED.value: (110, 220, 130),
    BboxSource.INTERPOLATED.value: (80, 190, 240),
    BboxSource.CARRIED.value: (60, 150, 240),
    BboxSource.MISSING.value: (70, 70, 220),
}


def draw_frame(manifest_dir: Path, fr, min_conf: float) -> np.ndarray | None:
    if fr.crop is None:
        return None
    img = cv2.imread(str(manifest_dir / fr.crop.path))
    if img is None:
        return None

    pts: dict[int, tuple[int, int]] = {}
    for i, kp in enumerate(fr.keypoints_2d):
        if kp[0] is None or kp[2] < min_conf:
            continue
        cx, cy = fr.crop.image_to_crop(float(kp[0]), float(kp[1]))
        pts[i] = (int(round(cx)), int(round(cy)))

    for a, b in SKELETON:
        if a in pts and b in pts:
            cv2.line(img, pts[a], pts[b], (230, 230, 235), 1, cv2.LINE_AA)
    for i, p in pts.items():
        color = (90, 200, 255) if i >= 11 else (120, 240, 160)
        cv2.circle(img, p, 2, color, -1, cv2.LINE_AA)

    color = SOURCE_COLOR.get(fr.bbox_source, (200, 200, 200))
    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, 2)
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (18, 20, 18), -1)
    label = f"{fr.frame_index}  {fr.track_state[:4]}  {fr.bbox_source[:3]}  k{len(pts)}"
    cv2.putText(img, label, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description="Contact sheet for an exported track")
    ap.add_argument("export_dir", help="dir containing tracks.json")
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--count", type=int, default=32, help="frames to sample across the clip")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = Path(args.export_dir)
    manifest = load_manifest(d / "tracks.json")
    frames = [f for f in manifest.frames if f.crop is not None]
    if not frames:
        print("no crops in export", file=sys.stderr)
        return 1

    idxs = np.linspace(0, len(frames) - 1, min(args.count, len(frames)))
    picked = [frames[int(round(i))] for i in sorted({int(round(i)) for i in idxs})]

    tiles = [t for t in (draw_frame(d, f, args.min_conf) for f in picked) if t is not None]
    if not tiles:
        print("could not read any crops", file=sys.stderr)
        return 1

    h, w = tiles[0].shape[:2]
    cols = max(1, args.cols)
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = t

    out = Path(args.out) if args.out else d / "debug_contact_sheet.jpg"
    cv2.imwrite(str(out), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    stats = manifest.stats()
    print(f"wrote {out}  ({len(tiles)} tiles)")
    print(f"  frames {stats['frames']}  reconstructable {stats['reconstructable']}  usable {stats['usable']}")
    print(f"  bbox sources {stats['bbox_source']}")
    print(f"  segments {manifest.segments}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
