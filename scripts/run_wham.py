"""Windows CLI for the WHAM reconstruction stage (Phase 2).

Never imports WHAM — it drives the WSL job through the bridge.

    # check the WSL side is ready
    python scripts/run_wham.py --doctor

    # reconstruct the good segment of a clip
    python scripts/run_wham.py --tracks outputs/motion3d/footage/tracks.json ^
        --video footage.mp4 --segment 0:164
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.motion3d.association_debug import (  # noqa: E402
    render_association_video,
    render_selected_crops_sheet,
)
from oline_cv.motion3d.segments import describe_segments  # noqa: E402
from oline_cv.motion3d.schema import load_manifest  # noqa: E402
from oline_cv.motion3d.target_association import load_associations  # noqa: E402
from oline_cv.motion3d.wham_bridge import (  # noqa: E402
    WhamBridgeError,
    WhamConfig,
    doctor,
    run_wham_job,
)


def _print_doctor(report: dict) -> int:
    print("WHAM environment report")
    print(f"  distro python   {report.get('python')}")
    print(f"  conda env       {report.get('conda_env')}")
    print(f"  wham root       {report.get('wham_root')}")
    gpu = report.get("gpu") or {}
    print(f"  cuda            {gpu.get('cuda_available')}  {gpu.get('name') or ''}")
    print(f"  ViTPose ready   {report.get('vitpose_available')}")
    print(f"  DPVO/SLAM ready {report.get('slam_available')}")

    mods = report.get("modules") or {}
    bad = [name for name, info in mods.items() if not info.get("ok")]
    if bad:
        print(f"  missing modules {', '.join(sorted(bad))}")
        for name in sorted(bad):
            err = (mods[name] or {}).get("error")
            if err:
                print(f"    - {name}: {err}")

    files = report.get("files") or {}
    absent = [info["path"] for info in files.values() if not info.get("exists")]
    if absent:
        print("  missing files:")
        for p in sorted(absent):
            print(f"    - {p}")

    if report.get("ready"):
        print("\nREADY — reconstruction can run.")
        return 0
    print(f"\nNOT READY — {len(report.get('missing') or [])} blocking item(s):")
    for m in report.get("missing") or []:
        print(f"  - {m}")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the WHAM stage via WSL")
    ap.add_argument("--tracks", help="path to tracks.json from Phase 1")
    ap.add_argument("--video", help="original full-resolution video")
    ap.add_argument("--out", default=None, help="output dir (default: alongside tracks.json)")
    ap.add_argument("--segment", default=None, help="frame range START:END (default: auto)")
    ap.add_argument("--list-segments", action="store_true", help="show segments and exit")
    ap.add_argument("--doctor", action="store_true", help="check the WSL environment and exit")
    ap.add_argument("--distro", default=None)
    ap.add_argument("--conda-env", default=None)
    ap.add_argument("--wham-root", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keypoints", default="vitpose", choices=["vitpose", "manifest"])
    ap.add_argument("--no-flip-eval", action="store_true")
    ap.add_argument("--no-slam", dest="run_slam", action="store_false", default=True)
    ap.add_argument("--save-verts", action="store_true")
    ap.add_argument(
        "--no-associate",
        dest="associate",
        action="store_false",
        default=True,
        help="trust the BoT-SORT box without checking it against WHAM's detections",
    )
    ap.add_argument("--assoc-min-iou", type=float, default=None)
    ap.add_argument("--assoc-max-center", type=float, default=None)
    ap.add_argument("--assoc-min-score", type=float, default=None)
    ap.add_argument("--assoc-reject-ambiguous", action="store_true")
    ap.add_argument(
        "--max-bridge-gap",
        type=int,
        default=None,
        help="interpolate detector dropouts up to N frames (0 disables)",
    )
    ap.add_argument("--min-frames", type=int, default=None)
    ap.add_argument(
        "--no-debug-video",
        dest="debug_video",
        action="store_false",
        default=True,
        help="skip rendering association_debug.mp4 after the run",
    )
    ap.add_argument("--verbose", action="store_true", help="stream raw WSL output")
    args = ap.parse_args()

    cfg = WhamConfig(device=args.device, keypoints_source=args.keypoints)
    if args.distro:
        cfg.distro = args.distro
    if args.conda_env:
        cfg.conda_env = args.conda_env
    if args.wham_root:
        cfg.wham_root = args.wham_root
    cfg.flip_eval = not args.no_flip_eval
    cfg.run_slam = args.run_slam
    cfg.save_verts = args.save_verts
    cfg.associate = args.associate
    cfg.assoc_min_iou = args.assoc_min_iou
    cfg.assoc_max_center = args.assoc_max_center
    cfg.assoc_min_score = args.assoc_min_score
    cfg.assoc_reject_ambiguous = args.assoc_reject_ambiguous
    cfg.max_bridge_gap = args.max_bridge_gap
    cfg.min_frames = args.min_frames

    if args.doctor:
        return _print_doctor(doctor(cfg))

    if not args.tracks:
        ap.error("--tracks is required (or use --doctor)")

    tracks = Path(args.tracks)
    manifest = load_manifest(tracks)

    if args.list_segments:
        print(f"{tracks}  ({manifest.video.get('frame_count')} frames @ {manifest.fps} fps)")
        for seg in describe_segments(manifest):
            print(
                f"  {seg.start:>4}-{seg.end:<4}  {seg.frames:>4} frames  "
                f"detected {seg.detected_ratio:.0%}  usable {seg.usable_ratio:.0%}"
            )
        return 0

    if not args.video:
        ap.error("--video is required")

    out_dir = Path(args.out) if args.out else tracks.parent
    segment = None
    if args.segment:
        a, b = args.segment.split(":")
        segment = (int(a), int(b))

    def progress(pct, msg, stage="wham"):
        print(f"  [{stage}] {float(pct):5.1f}%  {msg}", flush=True)

    try:
        result = run_wham_job(
            tracks,
            args.video,
            out_dir,
            segment=segment,
            config=cfg,
            progress_cb=progress,
            log_cb=(lambda line: print(f"    | {line}", flush=True)) if args.verbose else None,
        )
    except WhamBridgeError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1

    meta = result.metadata
    assert meta is not None
    print("\nReconstruction complete")
    print(f"  status          {meta.status}")
    print(f"  frames          {meta.frame_count}  (range {meta.frame_range})")
    print(f"  world grounded  {meta.world_grounded}")
    print(f"  runtime         {meta.runtime_seconds:.1f}s")
    print(f"  npz             {result.npz_path}")
    print(f"  arrays          {', '.join(result.npz_summary.get('arrays', []))}")

    assoc = meta.association or {}
    if assoc:
        print("\nIdentity association (target vs WHAM detections)")
        print(f"  matched         {assoc.get('valid')} / {assoc.get('frames')}"
              f"  (observed {assoc.get('observed')}, bridged {assoc.get('bridged')})")
        print(f"  reconstructed   {assoc.get('reconstructed_frames')} "
              f"frames {assoc.get('reconstructed_range')}")
        if assoc.get("bridged_frames"):
            print(f"  bridged frames  {assoc['bridged_frames']}")
        for skipped in assoc.get("bridge_skipped_gaps") or []:
            print(f"  gap not bridged {skipped['frames']} ({skipped['length']}f): "
                  f"{skipped['reason']}")
        print(f"  mean/min conf   {assoc.get('mean_confidence')} / {assoc.get('min_confidence')}")
        print(f"  mean/min iou    {assoc.get('mean_iou')} / {assoc.get('min_iou')}")
        print(f"  ambiguous       {assoc.get('ambiguous')}")
        unmatched = assoc.get("unmatched_frames") or []
        if unmatched:
            print(f"  unmatched       {unmatched[:40]}{' …' if len(unmatched) > 40 else ''}")

    assoc_json = out_dir / "association.json"
    if args.debug_video and assoc_json.exists():
        frames_dir = tracks.parent / "frames"
        if frames_dir.is_dir():
            print("\nRendering association debug video…")
            records, _, _ = load_associations(assoc_json)
            info = render_association_video(
                records, frames_dir, out_dir / "association_debug.mp4", fps=manifest.fps
            )
            zoom = render_selected_crops_sheet(
                records, frames_dir, out_dir / "association_selected.jpg"
            )
            print(f"  debug video     {info['path']} ({info['frames_written']} frames)")
            print(f"  selected zoom   {zoom['path']}")
        else:
            print(f"\n  (no frames/ dir at {frames_dir}; skipped debug video)")

    if meta.warnings:
        print("  warnings:")
        for w in meta.warnings:
            print(f"    - {w}")
    if meta.stats:
        print(f"  stats           {json.dumps(meta.stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
