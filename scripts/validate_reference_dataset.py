#!/usr/bin/env python3
"""Validate the reference dataset at a given pipeline stage.

Stages are cumulative, so the dataset can be checked as it is built up:

  --stage annotations      annotations valid, clips present, ids consistent
  --stage analyzed         plus a readable analysis JSON for every rep
  --stage benchmark-ready  plus a usable trust score on every "good" rep

A missing analysis is not an error at the ``annotations`` stage, so this is
safe to run the moment the first clips are annotated.

  python scripts/validate_reference_dataset.py --dataset data/reference \\
      --stage annotations
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.benchmark.schema import DEFAULT_MIN_TRUST_SCORE
from oline_cv.reference.dataset import (
    STAGES,
    ReferenceDataset,
    format_validation_report,
    validate_dataset,
    write_manifest,
)

DEFAULT_DATASET = "data/reference"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate the OL reference dataset")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Reference dataset root")
    p.add_argument("--stage", default="annotations", choices=[*STAGES])
    p.add_argument(
        "--min-trust",
        type=float,
        default=DEFAULT_MIN_TRUST_SCORE,
        help=f"Trust floor used by the benchmark-ready stage (default {DEFAULT_MIN_TRUST_SCORE})",
    )
    p.add_argument(
        "--write-manifest",
        action="store_true",
        help="Also refresh manifests/reference_manifest.json",
    )
    p.add_argument("--json", action="store_true", help="Emit the report as JSON")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = ReferenceDataset.at(args.dataset)

    if not dataset.root.is_dir():
        print(f"Dataset root not found: {dataset.root}", file=sys.stderr)
        return 1

    report = validate_dataset(dataset, stage=args.stage, min_trust=args.min_trust)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_validation_report(report))

    if args.write_manifest:
        path = write_manifest(dataset)
        if not args.json:
            print(f"\nManifest: {path}")

    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
