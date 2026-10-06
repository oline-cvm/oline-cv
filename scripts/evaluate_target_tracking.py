#!/usr/bin/env python3
"""Evaluate locked-target tracking on one clip.

  python scripts/evaluate_target_tracking.py VIDEO --pick-xy X,Y [--pick-t S]
      [--gt ground_truth.json] [--tracker deep_hm_sort|legacy_botsort|both]
      [--debug-tracking] [--out outputs/track_eval]

With ``--gt`` (format in ``oline_cv/tracking/evaluation.py`` and
docs/tracking.md) it reports target recall / precision, wrong-person frames,
identity switches and per-occlusion reacquisition latency. Without it, a
ground-truth-free health summary (coverage, states, tracker-id changes,
suspicious per-frame jumps).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.config import AnalysisConfig  # noqa: E402
from oline_cv.pose_tracker import PoseTracker  # noqa: E402
from oline_cv.tracking.evaluation import (  # noqa: E402
    load_ground_truth,
    target_metrics,
    tracking_health,
)

ROOT = Path(__file__).resolve().parent.parent


def run_one(video: str, pick_xy, pick_t, backend: str, debug: bool, reid_weights: str | None):
    cfg = AnalysisConfig(
        athlete_pick_xy=pick_xy,
        athlete_pick_time_s=pick_t,
        tracker_backend=backend,
        debug_tracking=debug,
    )
    if reid_weights:
        cfg.deep_hm.reid_weights = reid_weights
    tr = PoseTracker(cfg)
    fps, _n, _w, _h, ol, _dl, _frames = tr.extract_all(video)
    return fps, ol, tr.lock_meta


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("--pick-xy", type=str, default=None, help="normalised tap x,y")
    p.add_argument("--pick-t", type=float, default=None, help="tap time in seconds")
    p.add_argument("--gt", type=str, default=None, help="ground-truth JSON")
    p.add_argument("--tracker", choices=["deep_hm_sort", "legacy_botsort", "both"], default="deep_hm_sort")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--reid-weights", type=str, default=None)
    p.add_argument("--debug-tracking", action="store_true")
    p.add_argument("--out", type=str, default=str(ROOT / "outputs" / "track_eval"))
    a = p.parse_args()

    gt = load_ground_truth(a.gt) if a.gt else None
    pick = a.pick_xy or (",".join(str(v) for v in gt["pick_xy"]) if gt and gt.get("pick_xy") else None)
    if pick is None:
        p.error("--pick-xy is required (or pick_xy in the ground-truth file)")
    pick_xy = tuple(float(v) for v in pick.split(","))
    pick_t = a.pick_t if a.pick_t is not None else (gt.get("pick_t") if gt else None)

    backends = ["deep_hm_sort", "legacy_botsort"] if a.tracker == "both" else [a.tracker]
    report = {"video": a.video, "pick_xy": list(pick_xy), "pick_t": pick_t, "results": {}}
    for backend in backends:
        print(f"== {backend}", flush=True)
        fps, ol, meta = run_one(a.video, pick_xy, pick_t, backend, a.debug_tracking, a.reid_weights)
        pred = [None if (p_.bbox_xyxy is None or p_.interpolated) else p_.bbox_xyxy for p_ in ol]
        entry = {"health": tracking_health(ol, fps), "lock": {
            k: meta.get(k) for k in ("tracker", "reid", "target", "deep_hm_sort", "global_link", "debug")
            if k in meta
        }}
        if gt is not None:
            entry["metrics"] = target_metrics(pred, gt["boxes"], a.iou)
        report["results"][backend] = entry
        print(json.dumps({k: v for k, v in entry.items() if k != "lock"}, indent=2, default=str))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{Path(a.video).stem}_target_eval.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
