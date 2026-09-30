#!/usr/bin/env python3
"""Compare legacy frame-by-frame identity vs whole-clip linking on real clips.

YOLO + BoT-SORT runs once per clip; both identity methods are replayed on the
same detections, so the only difference is how the target is picked.

  python scripts/eval_tracking.py
  python scripts/eval_tracking.py --videos data/reference/clips/casey_01.mov

Per clip it reports post-snap coverage (share of frames with the target),
jumps (target centre moving > 0.5 box diagonals between consecutive frames —
a switch onto another player), and BoT-SORT id changes followed. Contact sheets
land in outputs/track_eval/ so you can check the box stays on the tackle.
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.config import AnalysisConfig
from oline_cv.global_link import camera_motion
from oline_cv.pose_tracker import PoseTracker, _bbox_iou
from oline_cv.snap_detection import detect_snap

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "track_eval"


def default_videos() -> list[str]:
    vids = sorted(glob.glob(str(ROOT / "data" / "reference" / "clips" / "*.mov")))
    vids += sorted(glob.glob(str(ROOT / "uploads" / "*__sideline.*")))
    return vids


def pick_for(video: Path) -> tuple[float, float] | None:
    """Reuse the tap from an earlier dashboard run of the same upload."""
    job = video.stem.split("__")[0]
    analysis = ROOT / "outputs" / f"{job}_analysis.json"
    if not analysis.exists():
        return None
    lock = json.loads(analysis.read_text(encoding="utf-8")).get("ol_lock") or {}
    pick = lock.get("pick_xy")
    return (float(pick[0]), float(pick[1])) if pick else None


def read_frames(path: str) -> tuple[list[np.ndarray], float, int, int]:
    cap = cv2.VideoCapture(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps, w, h


def legacy_replay(tr: PoseTracker, frames, raw) -> list[tuple[np.ndarray | None, int | None]]:
    """Re-run the old greedy IdentityAssociator over stored detections."""
    assoc = tr._associator
    out: list[tuple[np.ndarray | None, int | None]] = []
    for idx, frame in enumerate(frames):
        c = raw[idx]["cands"]
        if len(c) == 0:
            assoc.associate(idx, frame, np.zeros((0, 4)), np.zeros(0))
            out.append((None, None))
            continue
        ids = c.track_ids if (c.track_ids >= 0).any() else None
        if assoc.locked_botsort_id is None and ids is not None:
            centers = (c.boxes[:, :2] + c.boxes[:, 2:]) / 2.0
            i0 = int(np.argmin(np.linalg.norm(centers - tr._anchor_center[None, :], axis=1)))
            assoc.locked_botsort_id = assoc.botsort_id = int(ids[i0])
        d = assoc.associate(idx, frame, c.boxes, c.confs, ids)
        if d.best is None:
            out.append((None, None))
            continue
        pick = None
        if d.best.track_id is not None and ids is not None:
            hits = np.nonzero(ids == d.best.track_id)[0]
            pick = int(hits[0]) if len(hits) else None
        if pick is None and assoc.prev_bbox is not None:
            pick = int(np.argmax([_bbox_iou(assoc.prev_bbox, b) for b in c.boxes]))
        out.append((c.boxes[pick], d.best.track_id))
    return out


def max_drift(track, motion: np.ndarray, snap: int) -> float:
    """Farthest the box strays from its snap position, in player heights, with
    camera pan/zoom removed. A pass-pro tackle stays within a few heights;
    large values mean the box walked onto someone else."""
    origin, h0, worst = None, None, 0.0
    for t in range(snap, len(track)):
        b = track[t][0]
        if b is None:
            continue
        a, bx, by = motion[t]
        c = ((b[:2] + b[2:]) / 2.0 - (bx, by)) / a
        h = (b[3] - b[1]) / a
        if origin is None:
            origin, h0 = c, max(h, 1.0)
            continue
        worst = max(worst, float(np.linalg.norm(c - origin)) / h0)
    return round(worst, 2)


def metrics(track: list[tuple[np.ndarray | None, int | None]], snap: int, motion: np.ndarray) -> dict:
    post = track[snap:]
    covered = sum(1 for b, _ in post if b is not None)
    jumps = 0
    id_changes = 0
    prev_b, prev_id = None, None
    for b, tid in post:
        if b is not None and prev_b is not None:
            diag = float(np.linalg.norm(prev_b[2:] - prev_b[:2]))
            move = float(np.linalg.norm((b[:2] + b[2:]) / 2 - (prev_b[:2] + prev_b[2:]) / 2))
            if diag > 0 and move / diag > 0.5:
                jumps += 1
        if tid is not None and prev_id is not None and tid != prev_id:
            id_changes += 1
        prev_b = b
        if tid is not None:
            prev_id = tid
    return {
        "post_snap_frames": len(post),
        "coverage": round(covered / max(1, len(post)), 3),
        "jumps": jumps,
        "id_changes_followed": id_changes,
        "max_drift_h": max_drift(track, motion, snap),
    }


def contact_sheet(frames, track, path: Path, color, n: int = 12, cols: int = 4) -> None:
    idxs = np.linspace(0, len(frames) - 1, n).astype(int)
    tiles = []
    for i in idxs:
        img = frames[i].copy()
        b, _ = track[i]
        if b is not None:
            b = b.astype(int)
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, max(3, img.shape[1] // 400))
        else:
            cv2.putText(img, "NO TARGET", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 6)
        cv2.putText(img, f"f{i}", (40, img.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        scale = 480 / img.shape[1]
        tiles.append(cv2.resize(img, (480, int(img.shape[0] * scale))))
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[r : r + cols]) for r in range(0, len(tiles), cols)]
    cv2.imwrite(str(path), np.vstack(rows))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--videos", nargs="*", default=None)
    p.add_argument(
        "--tapped",
        action="append",
        default=[],
        metavar="VIDEO=X,Y",
        help="Extra clip locked by a tap at normalized X,Y (repeatable)",
    )
    p.add_argument("--model", default="yolov8m-pose.pt")
    p.add_argument("--redetect", action="store_true", help="Ignore cached detections")
    args = p.parse_args(argv)
    jobs: list[tuple[str, tuple[float, float] | None]] = [
        (v, pick_for(Path(v)))
        for v in (default_videos() if args.videos is None else args.videos)
    ]
    for spec in args.tapped:
        path, _, xy = spec.rpartition("=")
        x, y = (float(s) for s in xy.split(","))
        jobs.append((path, (x, y)))
    OUT.mkdir(parents=True, exist_ok=True)

    summary = []
    for v, pick in jobs:
        video = Path(v)
        print(f"\n== {video.name} (lock: {'tap ' + str(pick) if pick else 'auto'})", flush=True)
        cfg = AnalysisConfig(
            pose_model=args.model,
            athlete_pick_xy=pick,
            track_debug_dir=str(OUT / "_debug"),
        )
        frames, fps, w, h = read_frames(str(video))
        tag = f"{video.stem}_{'tap' if pick else 'auto'}"
        cache = OUT / "_cache" / f"{tag}_{cfg.imgsz_for(frames[0])}.pkl"
        tr = PoseTracker(cfg)
        if cache.exists() and not args.redetect:
            with cache.open("rb") as fh:
                c = pickle.load(fh)
            raw, snap, old, tr._anchor_center = c["raw"], c["snap"], c["legacy"], c["anchor"]
        else:
            tr.lock_initial(frames, w, h)
            tr._build_associator(frames, fps)
            tr.model.predictor = None
            raw = tr._detect_all(frames, lambda *a: None)
            snap = detect_snap(frames, cfg).snap_frame
            tr._build_associator(frames, fps)  # fresh associator state for the replay
            old = legacy_replay(tr, frames, raw)
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as fh:
                pickle.dump(
                    {"raw": raw, "snap": snap, "legacy": old, "anchor": tr._anchor_center}, fh
                )

        ol_poses, _ = tr._extract_global(frames, fps, lambda *a: None, raw=raw)
        new = [
            (None, None) if (pp.bbox_xyxy is None or pp.interpolated) else (pp.bbox_xyxy, pp.track_id)
            for pp in ol_poses
        ]
        motion = camera_motion([r["cands"] for r in raw])

        row = {
            "clip": video.name,
            "lock": "tap" if pick else "auto",
            "snap_frame": snap,
            "legacy": metrics(old, snap, motion),
            "global_link": metrics(new, snap, motion),
        }
        summary.append(row)
        contact_sheet(frames, old, OUT / f"{tag}_legacy.jpg", (0, 0, 255))
        contact_sheet(frames, new, OUT / f"{tag}_global.jpg", (0, 220, 0))
        print(f"   legacy      {row['legacy']}", flush=True)
        print(f"   global_link {row['global_link']}", flush=True)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        "\nclip                         cover old -> new   jumps old -> new"
        "   id-follow old -> new   drift(h) old -> new"
    )
    for r in summary:
        o, n = r["legacy"], r["global_link"]
        print(
            f"{r['clip'][:28]:<28} {o['coverage']:>6.1%} -> {n['coverage']:<6.1%}"
            f"   {o['jumps']:>3} -> {n['jumps']:<3}"
            f"          {o['id_changes_followed']:>3} -> {n['id_changes_followed']:<3}"
            f"          {o['max_drift_h']:>5} -> {n['max_drift_h']:<5}"
        )
    print(f"\nContact sheets + summary.json in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
