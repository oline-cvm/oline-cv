#!/usr/bin/env python3
"""Run the existing analyzer over every annotated clip in the reference dataset.

This is an orchestrator, not an analyzer: it calls
``oline_cv.pipeline.analyze_video`` unchanged, once per clip, and files the
output as ``analyses/<rep_id>_analysis.json``.

Each finished analysis is stamped with the play context from that rep's
annotation (rep_id, position, technique, side) because the benchmark's
``extract_features`` reads those fields and the pipeline does not produce
them. The expert's *quality verdict and notes deliberately stay out* of the
analysis file — the measurement artifact must never carry the label.

  python scripts/analyze_reference_dataset.py --dataset data/reference
  python scripts/analyze_reference_dataset.py --dataset data/reference --force

A clip that fails is reported and skipped; the run continues.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.benchmark.storage import load_json, save_json
from oline_cv.reference.dataset import ReferenceDataset
from oline_cv.reference.schema import annotation_context, validate_annotation

DEFAULT_DATASET = "data/reference"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch-run analyze_video over annotated reference clips"
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Reference dataset root")
    p.add_argument(
        "--play-type",
        choices=["pass", "run"],
        default=None,
        help="Fallback when an annotation leaves play_type unknown (default pass)",
    )
    p.add_argument("--model", default="yolov8m-pose.pt", help="YOLO-pose weights")
    p.add_argument("--rep-id", action="append", default=None, help="Only these reps")
    p.add_argument("--limit", type=int, default=None, help="Stop after N clips")
    p.add_argument("--force", action="store_true", help="Re-analyze reps that already have output")
    p.add_argument("--overlay", action="store_true", help="Also write overlay videos")
    p.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip reps whose annotation fails validation instead of stopping",
    )
    return p


def _stamp_context(
    analysis_path: Path, rep_id: str, context: dict[str, str]
) -> None:
    """Add join keys + play context to a finished analysis JSON.

    Keeps the measured data exactly as the pipeline wrote it and only adds
    identity/context fields the benchmark feature extractor looks for.
    """
    analysis = load_json(analysis_path)
    analysis["rep_id"] = rep_id
    for field in ("position", "technique", "side"):
        if context.get(field) and context[field] != "unknown":
            analysis[field] = context[field]
    save_json(analysis_path, analysis)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = ReferenceDataset.at(args.dataset)

    if not dataset.root.is_dir():
        print(f"Dataset root not found: {dataset.root}", file=sys.stderr)
        return 1

    # Imported here so --help works on a machine without torch/ultralytics.
    from oline_cv.config import AnalysisConfig
    from oline_cv.pipeline import analyze_video

    clips = dataset.discover_clips()
    annotations = dataset.discover_annotations()
    wanted = set(args.rep_id) if args.rep_id else None

    dataset.analyses_dir.mkdir(parents=True, exist_ok=True)

    done: list[str] = []
    skipped: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for rep_id in dataset.rep_ids():
        if wanted is not None and rep_id not in wanted:
            continue
        if args.limit is not None and len(done) >= args.limit:
            break

        ann_path = annotations.get(rep_id)
        if ann_path is None:
            skipped.append((rep_id, "no annotation"))
            continue

        clip_list = clips.get(rep_id) or []
        if not clip_list:
            skipped.append((rep_id, "no clip in clips/"))
            continue

        try:
            annotation = load_json(ann_path)
        except (OSError, ValueError) as exc:
            failed.append((rep_id, f"unreadable annotation: {exc}"))
            continue

        problems = validate_annotation(annotation, rep_id=rep_id)
        if problems:
            message = f"invalid annotation: {problems[0]}"
            if args.skip_invalid:
                skipped.append((rep_id, message))
                continue
            failed.append((rep_id, message))
            continue

        out_json = dataset.analysis_path(rep_id)
        if out_json.is_file() and not args.force:
            skipped.append((rep_id, "analysis exists (use --force)"))
            continue

        context = annotation_context(annotation)
        play_type = context["play_type"]
        if play_type == "unknown":
            play_type = args.play_type or "pass"

        video = clip_list[0]
        print(f"[{len(done) + 1}] analyzing {rep_id} ({video.name}) play_type={play_type}")

        cfg = AnalysisConfig(
            pose_model=args.model,
            play_type=play_type,
            overlay_zoom_on_athlete=False,
        )
        cfg.write_overlay_video = bool(args.overlay)

        overlay_path = (
            str(dataset.analyses_dir / f"{rep_id}_overlay.mp4") if args.overlay else None
        )

        try:
            analyze_video(
                str(video),
                config=cfg,
                output_json=str(out_json),
                overlay_path=overlay_path,
            )
            _stamp_context(out_json, rep_id, context)
        except Exception as exc:  # keep going through the rest of the dataset
            failed.append((rep_id, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc(limit=3, file=sys.stderr)
            continue

        done.append(rep_id)

    print("\nBatch analysis")
    print("--------------")
    print(f"Analyzed: {len(done)}")
    print(f"Skipped:  {len(skipped)}")
    print(f"Failed:   {len(failed)}")
    for rep_id, reason in skipped:
        print(f"  skip {rep_id}: {reason}")
    for rep_id, reason in failed:
        print(f"  FAIL {rep_id}: {reason}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
