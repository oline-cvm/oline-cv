#!/usr/bin/env python3
"""Create, edit, or validate an expert annotation for a reference clip.

The annotation is the human football label. This script only writes down what
the annotator says — it never inspects the video or infers technique.

Create:
  python scripts/create_annotation.py \\
      --video rep_001.mp4 --rep-id rep_001 \\
      --position LT --play-type pass --technique vertical_set --side left \\
      --quality good --note "good initial set" --note "stays square"

Add notes to an existing annotation:
  python scripts/create_annotation.py --rep-id rep_001 --edit \\
      --note "hands a touch wide"

Validate what is already on disk:
  python scripts/create_annotation.py --rep-id rep_001 --validate-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.benchmark.schema import PLAY_TYPES, POSITIONS, SIDES
from oline_cv.benchmark.storage import load_json, save_json
from oline_cv.reference.dataset import ReferenceDataset
from oline_cv.reference.schema import (
    COMMON_LEVELS,
    QUALITY_VALUES,
    make_annotation,
    validate_annotation,
)

DEFAULT_DATASET = "data/reference"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create or edit an expert annotation for a reference clip",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Reference dataset root")
    p.add_argument("--rep-id", help="Rep id (defaults to the video filename stem)")
    p.add_argument("--video", help="Video filename inside clips/, e.g. rep_001.mp4")

    p.add_argument("--position", choices=[*POSITIONS], help="OL position")
    p.add_argument("--level", default=None, help=f"e.g. {', '.join(COMMON_LEVELS)}")
    p.add_argument("--play-type", choices=[*PLAY_TYPES], help="pass or run")
    p.add_argument("--technique", help="Free-form, e.g. vertical_set, inside_zone")
    p.add_argument("--side", choices=[*SIDES], help="left / right / interior")

    p.add_argument("--quality", choices=[*QUALITY_VALUES], help="Expert verdict")
    p.add_argument(
        "--note",
        action="append",
        default=[],
        dest="notes",
        help="Free-form expert note; repeat for multiple notes",
    )

    p.add_argument("--force", action="store_true", help="Overwrite an existing annotation")
    p.add_argument("--edit", action="store_true", help="Update an existing annotation in place")
    p.add_argument("--validate-only", action="store_true", help="Validate without writing")
    p.add_argument(
        "--require-video",
        action="store_true",
        help="Fail if the referenced clip is not present in clips/",
    )
    return p


def _resolve_ids(args: argparse.Namespace) -> tuple[str, str] | None:
    """Work out (rep_id, video_file) from whichever the user supplied."""
    rep_id = args.rep_id
    video = args.video
    if not rep_id and not video:
        return None
    if not rep_id:
        rep_id = Path(str(video)).stem
    if not video:
        video = f"{rep_id}.mp4"
    return str(rep_id), Path(str(video)).name


def _report(errors: list[str], target: Path) -> int:
    if errors:
        print(f"Invalid annotation ({target}):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"Annotation valid: {target}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = ReferenceDataset.at(args.dataset)

    ids = _resolve_ids(args)
    if ids is None:
        print("Provide --rep-id or --video.", file=sys.stderr)
        return 1
    rep_id, video_file = ids

    target = dataset.annotations_dir / f"{rep_id}.json"

    # --- validate an existing file -------------------------------------
    if args.validate_only:
        if not target.is_file():
            print(f"No annotation at {target}", file=sys.stderr)
            return 1
        try:
            existing = load_json(target)
        except (OSError, ValueError) as exc:
            print(f"Unreadable annotation {target}: {exc}", file=sys.stderr)
            return 1
        return _report(validate_annotation(existing, rep_id=rep_id), target)

    # --- edit in place --------------------------------------------------
    if args.edit:
        if not target.is_file():
            print(f"No annotation to edit at {target}", file=sys.stderr)
            return 1
        annotation = load_json(target)
        player = annotation.setdefault("player", {})
        play = annotation.setdefault("play", {})
        expert = annotation.setdefault("expert", {})

        if args.position:
            player["position"] = args.position
        if args.level:
            player["level"] = args.level
        if args.play_type:
            play["play_type"] = args.play_type
        if args.technique:
            play["technique"] = args.technique
        if args.side:
            play["side"] = args.side
        if args.quality:
            expert["quality"] = args.quality
        if args.notes:
            # Append, never replace: earlier expert observations are kept.
            expert["notes"] = list(expert.get("notes") or []) + list(args.notes)

        errors = validate_annotation(annotation, rep_id=rep_id)
        if errors:
            return _report(errors, target)
        save_json(target, annotation)
        print(f"Updated {target}")
        return 0

    # --- create ---------------------------------------------------------
    if not args.quality:
        print(
            f"--quality is required when creating "
            f"(one of {', '.join(QUALITY_VALUES)}).",
            file=sys.stderr,
        )
        return 1

    if target.is_file() and not args.force:
        print(
            f"Annotation already exists: {target}\n"
            "Pass --force to overwrite, or --edit to update it in place.",
            file=sys.stderr,
        )
        return 1

    if args.require_video:
        clips = dataset.discover_clips().get(rep_id) or []
        if not clips:
            print(
                f"No clip for {rep_id!r} in {dataset.clips_dir}", file=sys.stderr
            )
            return 1

    annotation = make_annotation(
        rep_id=rep_id,
        video_file=video_file,
        position=args.position,
        level=args.level,
        play_type=args.play_type,
        technique=args.technique,
        side=args.side,
        quality=args.quality,
        notes=args.notes,
    )

    errors = validate_annotation(annotation, rep_id=rep_id)
    if errors:
        return _report(errors, target)

    dataset.annotations_dir.mkdir(parents=True, exist_ok=True)
    save_json(target, annotation)
    print(f"Wrote {target}")

    unknown = [
        field
        for field, value in (
            ("position", annotation["player"]["position"]),
            ("play_type", annotation["play"]["play_type"]),
            ("technique", annotation["play"]["technique"]),
            ("side", annotation["play"]["side"]),
        )
        if value == "unknown"
    ]
    if unknown:
        print(f"  note: {', '.join(unknown)} left unknown; add later with --edit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
