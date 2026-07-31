"""Phase 1 CLI — export the locked OL's track, full frames, and crops.

The output directory is the hand-off point to the world-grounded HMR stage,
which runs in a separate CUDA environment (WSL2) and reads only tracks.json.

    python scripts/export_tracks.py footage.mp4 --jersey 76 --out outputs/motion3d/footage

Existing tracking behaviour is untouched; this only serializes its output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.config import AnalysisConfig  # noqa: E402
from oline_cv.motion3d.track_export import export_tracks  # noqa: E402
from oline_cv.pose_tracker import PoseTracker  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Export BoT-SORT track + crops for 3D HMR")
    ap.add_argument("video", help="path to the clip")
    ap.add_argument("--out", default=None, help="output dir (default outputs/motion3d/<stem>)")
    ap.add_argument("--jersey", type=int, default=None, help="target jersey number")
    ap.add_argument("--play-type", default="pass", choices=["pass", "run", "auto"])
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--crop-pad", type=float, default=0.25)
    ap.add_argument("--max-interp-gap", type=int, default=8)
    ap.add_argument(
        "--no-full-frames",
        action="store_true",
        help="skip full-frame export (saves disk, but HMR loses scene context)",
    )
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        print(f"error: {video} not found", file=sys.stderr)
        return 2

    out_dir = Path(args.out) if args.out else Path("outputs") / "motion3d" / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = AnalysisConfig(
        target_jersey=args.jersey,
        play_type=args.play_type,
        write_overlay_video=False,
    )

    def progress(pct, msg, stage="track"):
        print(f"  [{stage}] {float(pct):5.1f}%  {msg}", flush=True)

    print(f"Tracking {video} …", flush=True)
    tracker = PoseTracker(cfg)
    fps, n_frames, width, height, ol_poses, _dl, frames = tracker.extract_all(
        str(video), progress_cb=progress
    )

    print(f"Exporting {len(ol_poses)} frames → {out_dir}", flush=True)
    manifest = export_tracks(
        str(video),
        ol_poses,
        frames,
        fps,
        width,
        height,
        out_dir,
        target_jersey=args.jersey,
        ol_lock=getattr(tracker, "lock_meta", {}) or {},
        crop_size=args.crop_size,
        crop_pad=args.crop_pad,
        max_interp_gap=args.max_interp_gap,
        save_full_frames=not args.no_full_frames,
        progress_cb=progress,
    )

    stats = manifest.stats()
    print("\nExport complete")
    print(f"  tracks.json       {out_dir / 'tracks.json'}")
    print(f"  frames            {stats['frames']}")
    print(f"  reconstructable   {stats['reconstructable']}")
    print(f"  usable            {stats['usable']}")
    print(f"  bbox sources      {stats['bbox_source']}")
    print(f"  segments          {stats['segments']} (longest {stats['longest_segment']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
