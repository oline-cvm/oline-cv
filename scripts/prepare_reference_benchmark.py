#!/usr/bin/env python3
"""Select benchmark-eligible reps and write a small selection manifest.

Filters run in a fixed order — annotation validity, expert quality, play
context, then measured trust — so every excluded rep carries exactly one
reason and the football reasons are reported before the CV ones. The output
manifest lists only rep ids and analysis paths; it never copies analysis data.

  python scripts/prepare_reference_benchmark.py \\
      --dataset data/reference \\
      --position LT --play-type pass --technique vertical_set --quality good \\
      --output data/reference/manifests/lt_pass_vertical_set_v1.json

Feed the result straight to the existing builder:

  python scripts/build_benchmark.py \\
      --manifest data/reference/manifests/lt_pass_vertical_set_v1.json \\
      --name lt_pass_vertical_set_v1 \\
      --output benchmarks/lt_pass_vertical_set_v1.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.benchmark.schema import (
    DEFAULT_MIN_TRUST_SCORE,
    PLAY_TYPES,
    POSITIONS,
    SIDES,
)
from oline_cv.benchmark.storage import save_json
from oline_cv.reference.dataset import (
    DEFAULT_DATASET_NAME,
    ReferenceDataset,
    select_benchmark_reps,
)
from oline_cv.reference.schema import QUALITY_VALUES

DEFAULT_DATASET = "data/reference"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Select benchmark-eligible reference reps")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="Reference dataset root")
    p.add_argument("--output", required=True, help="Output selection manifest path")
    p.add_argument("--name", default=DEFAULT_DATASET_NAME, help="Dataset name to record")

    p.add_argument("--position", choices=[*POSITIONS], default=None)
    p.add_argument("--play-type", choices=[*PLAY_TYPES], default=None)
    p.add_argument("--technique", default=None, help="e.g. vertical_set")
    p.add_argument("--side", choices=[*SIDES], default=None)

    p.add_argument(
        "--quality",
        action="append",
        choices=[*QUALITY_VALUES],
        default=None,
        help="Expert verdicts to accept (repeatable; default: good)",
    )
    p.add_argument(
        "--min-trust",
        type=float,
        default=DEFAULT_MIN_TRUST_SCORE,
        help=f"Trust floor, matching the benchmark default ({DEFAULT_MIN_TRUST_SCORE})",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = ReferenceDataset.at(args.dataset)

    if not dataset.root.is_dir():
        print(f"Dataset root not found: {dataset.root}", file=sys.stderr)
        return 1

    manifest = select_benchmark_reps(
        dataset,
        position=args.position,
        play_type=args.play_type,
        technique=args.technique,
        side=args.side,
        quality=args.quality or ("good",),
        min_trust=args.min_trust,
        dataset_name=args.name,
    )

    out = save_json(args.output, manifest)

    print(f"Wrote {out}")
    print(f"eligible={manifest['n_reps']} excluded={manifest['n_excluded']}")
    if manifest["reps"]:
        print("\nIncluded:")
        for rep_id in manifest["reps"]:
            print(f"  + {rep_id}")
    if manifest["excluded"]:
        print("\nExcluded:")
        for row in manifest["excluded"]:
            detail = f" ({row['detail']})" if row.get("detail") else ""
            print(f"  - {row['rep_id']}: {row['reason']}{detail}")

    if not manifest["reps"]:
        print("\nNo eligible reps; nothing to build a benchmark from.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
