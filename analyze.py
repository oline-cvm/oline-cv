#!/usr/bin/env python3
"""CLI for offensive lineman analysis.

Example:
  python analyze.py footage.mp4
  python analyze.py footage.mp4 --jersey 76
  python analyze.py footage.mp4 --pick-xy 0.28,0.55
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oline_cv.config import AnalysisConfig
from oline_cv.pipeline import analyze_video, result_brief


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OL pass-set / run CV analysis")
    p.add_argument("video", type=str, help="Path to sideline/endzone video")
    p.add_argument("-o", "--output", type=str, default=None, help="Output JSON path")
    p.add_argument("--overlay", type=str, default=None, help="Overlay video path")
    p.add_argument("--no-overlay", action="store_true", help="Skip overlay video")
    p.add_argument(
        "--jersey",
        type=int,
        default=None,
        help="Optional jersey label for UI (does not drive tracking)",
    )
    p.add_argument(
        "--play-type",
        choices=["pass", "run"],
        default="pass",
        help="Pass protection or run blocking analysis",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8m-pose.pt",
        help="Ultralytics YOLO-pose weights",
    )
    p.add_argument("--snap-frame", type=int, default=None, help="Manual snap frame override")
    p.add_argument("--set-end-frame", type=int, default=None, help="Manual set end frame")
    p.add_argument(
        "--pick-xy",
        type=str,
        default=None,
        help="Optional normalized click x,y to force athlete lock",
    )
    p.add_argument("--movement-threshold", type=float, default=None)
    p.add_argument("--min-keypoint-confidence", type=float, default=None)
    p.add_argument("--brief", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video = Path(args.video)
    if not video.exists():
        print(f"Video not found: {video}", file=sys.stderr)
        return 1

    cfg = AnalysisConfig(
        pose_model=args.model,
        target_jersey=args.jersey,
        play_type=args.play_type,
    )
    if args.no_overlay:
        cfg.write_overlay_video = False
    if args.snap_frame is not None:
        cfg.snap_frame_override = args.snap_frame
    if args.set_end_frame is not None:
        cfg.set_end_frame_override = args.set_end_frame
    if args.movement_threshold is not None:
        cfg.movement_threshold_frac = args.movement_threshold
    if args.min_keypoint_confidence is not None:
        cfg.min_keypoint_confidence = args.min_keypoint_confidence
    if args.pick_xy:
        parts = [float(x.strip()) for x in args.pick_xy.split(",")]
        if len(parts) != 2:
            print("--pick-xy must be x,y", file=sys.stderr)
            return 1
        cfg.athlete_pick_xy = (parts[0], parts[1])

    print(f"Analyzing {video} ...")
    result = analyze_video(
        str(video),
        config=cfg,
        output_json=args.output,
        overlay_path=args.overlay,
    )
    if args.brief:
        print(result_brief(result))
    else:
        print(result_brief(result))
        print(f"JSON: {result['output_json']}")
        if "overlay_video" in result:
            print(f"Overlay: {result['overlay_video']}")
        print(json.dumps(result["rep_summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
