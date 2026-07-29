"""Debug tracking association on a clip — logs scores around contact (~4s).

Usage:
  .\\.venv\\Scripts\\python.exe scripts/debug_track_assoc.py footage.mp4
  .\\.venv\\Scripts\\python.exe scripts/debug_track_assoc.py footage.mp4 --pick-xy 0.48,0.55
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oline_cv.config import AnalysisConfig
from oline_cv.pose_tracker import PoseTracker
from oline_cv.visualize import write_overlay_video
from oline_cv.snap_detection import detect_snap
from oline_cv.initial_quicks import analyze_initial_quicks
from oline_cv.body_position import compute_frame_body_metrics, smooth_posture_sequence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=str)
    ap.add_argument("--pick-xy", type=str, default=None, help="normalized x,y")
    ap.add_argument("--jersey", type=int, default=None)
    ap.add_argument("--out", type=str, default="outputs/track_debug")
    ap.add_argument("--contact-sec", type=float, default=4.0)
    ap.add_argument("--window", type=float, default=1.5, help="seconds around contact to summarize")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pick = None
    if args.pick_xy:
        a, b = args.pick_xy.split(",")
        pick = (float(a), float(b))

    cfg = AnalysisConfig(
        target_jersey=args.jersey,
        athlete_pick_xy=pick,
        track_calib_mode=True,
        track_debug_dir=str(out),
        overlay_zoom_on_athlete=False,
    )
    tracker = PoseTracker(cfg)
    fps, n, w, h, ol, dl, frames = tracker.extract_all(args.video)

    # Overlay with track state HUD
    snap = detect_snap(frames, cfg)
    quicks = analyze_initial_quicks(ol, snap.snap_frame, fps, cfg)
    body = [
        compute_frame_body_metrics(p, quicks.standing_height_px, cfg) for p in ol
    ]
    smooth_posture_sequence(body)
    overlay = str(out / "overlay_debug.mp4")
    write_overlay_video(frames, ol, body, snap, quicks, fps, overlay, cfg)

    contact_f = int(args.contact_sec * fps)
    lo = max(0, contact_f - int(args.window * fps))
    hi = min(n - 1, contact_f + int(args.window * fps))

    states = [p.track_state for p in ol]
    window_states = states[lo : hi + 1]
    summary = {
        "fps": fps,
        "n_frames": n,
        "contact_frame": contact_f,
        "window": [lo, hi],
        "lock_meta": tracker.lock_meta,
        "recommended_thresholds": tracker.recommended_thresholds,
        "state_counts_full": {s: states.count(s) for s in sorted(set(states))},
        "state_counts_window": {s: window_states.count(s) for s in sorted(set(window_states))},
        "usable_rate_window": float(np.mean([p.usable for p in ol[lo : hi + 1]])),
        "mean_conf_window": float(
            np.mean([p.track_confidence for p in ol[lo : hi + 1]])
        ),
        "overlay": overlay,
        "assoc_csv": tracker.lock_meta.get("assoc_log"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Print candidate score ranges near contact from CSV rows
    rows = tracker._associator.logger.rows if tracker._associator else []
    near = [r for r in rows if lo <= int(r["frame_idx"]) <= hi]
    accepted = [r for r in near if r.get("accepted")]
    rejected = [r for r in near if not r.get("accepted")]

    def stats(xs):
        if not xs:
            return None
        a = np.array(xs, dtype=float)
        return {
            "n": int(len(a)),
            "min": float(a.min()),
            "p10": float(np.percentile(a, 10)),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "max": float(a.max()),
        }

    score_report = {
        "accepted_appearance": stats([r["appearance"] for r in accepted]),
        "accepted_weighted": stats([r["weighted"] for r in accepted]),
        "accepted_jersey": stats([r["jersey_sim"] for r in accepted]),
        "rejected_appearance": stats([r["appearance"] for r in rejected]),
        "rejected_weighted": stats([r["weighted"] for r in rejected]),
        "rejected_jersey": stats([r["jersey_sim"] for r in rejected]),
    }
    (out / "contact_score_report.json").write_text(
        json.dumps(score_report, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("--- contact window score report ---")
    print(json.dumps(score_report, indent=2))
    print(f"Wrote {overlay}")
    print(f"Refs/crops under {out / 'crops'}")


if __name__ == "__main__":
    main()
