#!/usr/bin/env python3
"""Run tracking on one clip with a tap and write a contact sheet of the result.

  python scripts/check_lock.py VIDEO X Y [--t SECONDS]

The sheet (outputs/track_eval/<stem>_lock.jpg) shows the tracked box and
skeleton on 12 frames; each tile is cropped around the player so you can see
who it is.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.config import AnalysisConfig
from oline_cv.pose_tracker import PoseTracker

OUT = Path(__file__).resolve().parent.parent / "outputs" / "track_eval"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("x", type=float)
    p.add_argument("y", type=float)
    p.add_argument("--t", type=float, default=0.2)
    a = p.parse_args()

    cfg = AnalysisConfig(athlete_pick_xy=(a.x, a.y), athlete_pick_time_s=a.t)
    tr = PoseTracker(cfg)
    fps, n, w, h, ol, _dl, frames = tr.extract_all(a.video)
    print({k: tr.lock_meta.get(k) for k in ("tracker", "anchor_bbox", "global_link", "frames_lost")})

    tiles = []
    for i in np.linspace(0, n - 1, 12).astype(int):
        img = frames[i].copy()
        pose = ol[i]
        if pose.bbox_xyxy is not None:
            b = pose.bbox_xyxy.astype(int)
            col = (0, 165, 255) if pose.interpolated else (0, 255, 0)
            cv2.rectangle(img, tuple(b[:2]), tuple(b[2:]), col, 2)
            for (x, y), c in zip(pose.keypoints_xy, pose.keypoints_conf):
                if not np.isnan(x) and c >= cfg.min_keypoint_confidence:
                    cv2.circle(img, (int(x), int(y)), 3, (255, 0, 255), -1)
            cx, cy = (b[:2] + b[2:]) // 2
        else:
            cx, cy = w // 2, h // 2
            cv2.putText(img, "NO TARGET", (cx - 150, cy), 0, 1.5, (0, 0, 255), 3)
        half = int(max(160, 2.5 * (0 if pose.bbox_xyxy is None else b[3] - b[1])))
        x0, y0 = max(0, cx - half), max(0, cy - half)
        crop = img[y0 : y0 + 2 * half, x0 : x0 + 2 * half]
        crop = cv2.resize(crop, (360, 360))
        cv2.putText(crop, f"f{i} {pose.track_state}", (8, 24), 0, 0.7, (255, 255, 255), 2)
        tiles.append(crop)
    sheet = np.vstack([np.hstack(tiles[r : r + 4]) for r in range(0, 12, 4)])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{Path(a.video).stem}_lock.jpg"
    cv2.imwrite(str(path), sheet)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
