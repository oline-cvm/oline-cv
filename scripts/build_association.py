"""Verify target association independently of WHAM.

WHAM's preprocessing detects every person in frame; the identity of the body it
reconstructs is decided by association against our BoT-SORT target box. That
association logic is the risky part, and it does not need a GPU or WHAM to
validate — only a person detector and the exported frames.

This script runs a local YOLO person detector over the segment, associates every
detection against the TrackManifest target, and writes:

    association.json          per-frame candidates, scores, accept/reject reasons
    association_debug.mp4     target vs selected vs rejected, with confidence
    association_sheet.jpg     contact sheet for a quick visual check

The exact same `target_association` module runs inside the WSL reconstruction, so
what you verify here is what WHAM will use.

    python scripts/build_association.py outputs/motion3d/footage --segment 0:164
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.motion3d.association_debug import (  # noqa: E402
    render_association_sheet,
    render_association_video,
    render_selected_crops_sheet,
)
from oline_cv.motion3d.schema import load_manifest  # noqa: E402
from oline_cv.motion3d.segments import frames_in_segment, select_segment  # noqa: E402
from oline_cv.motion3d.target_association import (  # noqa: E402
    AssociationThresholds,
    associate_sequence,
    bridge_gaps,
    load_associations,
    longest_valid_run,
    save_associations,
    summarize,
)


def detect_people(video: str, frame_indices: set[int], model_name: str, conf: float, imgsz: int):
    """Run a person detector on the requested frames of the original video.

    Stands in for WHAM's YOLOv8x preprocessing detector: same job (every person
    in frame), same output space (full-frame xyxy).
    """
    import cv2
    from ultralytics import YOLO

    model = YOLO(model_name)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")

    out: dict[int, tuple[list[list[float]], list[float]]] = {}
    last = max(frame_indices)
    idx = 0
    done = 0
    while idx <= last:
        ok, img = cap.read()
        if not ok:
            break
        if idx in frame_indices:
            res = model.predict(
                img, classes=0, conf=conf, imgsz=imgsz, verbose=False, save=False
            )[0]
            boxes = res.boxes
            xyxy = boxes.xyxy.cpu().numpy().tolist() if boxes is not None else []
            confs = boxes.conf.cpu().numpy().tolist() if boxes is not None else []
            out[idx] = ([[float(v) for v in b] for b in xyxy], [float(c) for c in confs])
            done += 1
            if done % 20 == 0:
                print(f"  detect {done}/{len(frame_indices)}…", flush=True)
        idx += 1
    cap.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build + verify target association")
    ap.add_argument("export_dir", help="Phase 1 export dir containing tracks.json")
    ap.add_argument("--video", default=None, help="source video (default: manifest path)")
    ap.add_argument("--segment", default=None, help="frame range START:END (default: auto)")
    ap.add_argument("--model", default="yolov8m-pose.pt", help="person detector")
    ap.add_argument("--det-conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-iou", type=float, default=None)
    ap.add_argument("--max-center", type=float, default=None)
    ap.add_argument("--min-score", type=float, default=None)
    ap.add_argument("--reject-ambiguous", action="store_true")
    ap.add_argument("--no-video", action="store_true", help="skip mp4, write sheet only")
    ap.add_argument(
        "--reuse-detections",
        action="store_true",
        help="re-score the detections already in association.json (skips the detector)",
    )
    args = ap.parse_args()

    export_dir = Path(args.export_dir)
    manifest = load_manifest(export_dir / "tracks.json")
    video = args.video or manifest.video.get("path")
    if not video or not Path(video).exists():
        print(f"error: source video not found ({video})", file=sys.stderr)
        return 2

    explicit = None
    if args.segment:
        a, b = args.segment.split(":")
        explicit = (int(a), int(b))
    segment = select_segment(manifest, explicit=explicit)
    frames = frames_in_segment(manifest, segment)
    print(f"segment {segment.start}-{segment.end}  {len(frames)} frames  video {video}")

    kwargs = {}
    if args.min_iou is not None:
        kwargs["min_iou"] = args.min_iou
    if args.max_center is not None:
        kwargs["max_center_dist_frac"] = args.max_center
    if args.min_score is not None:
        kwargs["min_score"] = args.min_score
    if args.reject_ambiguous:
        kwargs["reject_ambiguous"] = True
    thresholds = AssociationThresholds(**kwargs)

    if args.reuse_detections:
        prior, _, _ = load_associations(export_dir / "association.json")
        detections = {
            a.frame_index: (
                [c.bbox for c in a.candidates],
                [c.detection_confidence for c in a.candidates],
            )
            for a in prior
        }
        print(f"reusing detections for {len(detections)} frames")
    else:
        print(f"detecting people with {args.model}…")
        detections = detect_people(
            str(video), {f.frame_index for f in frames}, args.model, args.det_conf, args.imgsz
        )

    associations = associate_sequence(frames, detections, thresholds=thresholds)
    bridge_stats = bridge_gaps(associations, thresholds=thresholds)
    stats = summarize(associations)
    run = longest_valid_run(associations)

    save_associations(
        export_dir / "association.json",
        associations,
        thresholds,
        extra={
            "segment": segment.to_dict(),
            "detector": {"model": args.model, "conf": args.det_conf, "imgsz": args.imgsz},
            "longest_valid_run": list(run) if run else None,
            "bridge_skipped_gaps": bridge_stats["skipped_gaps"],
        },
    )

    sheet = render_association_sheet(
        associations, export_dir / "frames", export_dir / "association_sheet.jpg", count=12, cols=4
    )
    zoom = render_selected_crops_sheet(
        associations, export_dir / "frames", export_dir / "association_selected.jpg"
    )
    video_info = {"frames_written": 0}
    if not args.no_video:
        print("rendering debug video…")
        video_info = render_association_video(
            associations,
            export_dir / "frames",
            export_dir / "association_debug.mp4",
            fps=manifest.fps,
        )

    print("\nAssociation summary")
    print(f"  frames            {stats['frames']}")
    print(f"  matched           {stats['valid']} "
          f"(observed {stats['observed']}, bridged {stats['bridged']})")
    if stats["bridged_frames"]:
        print(f"  bridged frames    {stats['bridged_frames']}")
    for skipped in bridge_stats["skipped_gaps"]:
        print(f"  gap not bridged   {skipped['frames']} ({skipped['length']}f): "
              f"{skipped['reason']}")
    print(f"  invalid           {stats['invalid']}  {stats['invalid_reasons']}")
    print(f"  ambiguous         {stats['ambiguous']}")
    print(f"  mean/min conf     {stats['mean_confidence']} / {stats['min_confidence']}")
    print(f"  mean/min iou      {stats['mean_iou']} / {stats['min_iou']}")
    print(f"  mean candidates   {stats['mean_candidates']}")
    print(f"  longest valid run {run}")
    if stats["unmatched_frames"]:
        shown = stats["unmatched_frames"][:40]
        print(f"  unmatched frames  {shown}{' …' if len(stats['unmatched_frames']) > 40 else ''}")
    print(f"\n  association.json  {export_dir / 'association.json'}")
    print(f"  sheet             {sheet['path']} ({sheet['tiles']} tiles)")
    print(f"  selected zoom     {zoom['path']} ({zoom['tiles']} tiles)")
    if not args.no_video:
        print(f"  debug video       {video_info['path']} ({video_info['frames_written']} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
