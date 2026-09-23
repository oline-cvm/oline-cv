#!/usr/bin/env python3
"""Build an OL reference profile from approved analyze_video() JSON files.

Example:
  python scripts/build_benchmark.py \\
      --inputs data/good_vertical_sets/*.json \\
      --name elite_lt_vertical_set_v1 \\
      --position LT \\
      --play-type pass \\
      --technique vertical_set \\
      --output benchmarks/elite_lt_vertical_set_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.benchmark.builder import build_benchmark
from oline_cv.benchmark.schema import DEFAULT_MIN_TRUST_SCORE, PLAY_TYPES, POSITIONS, SIDES
from oline_cv.benchmark.storage import expand_input_paths, load_analysis_json, save_benchmark


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build a statistical OL benchmark from good-rep JSON")
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Analysis JSON paths, globs, or directories",
    )
    p.add_argument("--name", required=True, help="Benchmark name")
    p.add_argument("--output", required=True, help="Output benchmark JSON path")
    p.add_argument("--position", default="unknown", choices=POSITIONS)
    p.add_argument("--play-type", default="pass", choices=PLAY_TYPES)
    p.add_argument("--technique", default="unknown", help="e.g. vertical_set, inside_zone")
    p.add_argument("--side", default="unknown", choices=SIDES)
    p.add_argument(
        "--min-trust",
        type=float,
        default=DEFAULT_MIN_TRUST_SCORE,
        help=f"Skip reps with trust.overall.score below this (default {DEFAULT_MIN_TRUST_SCORE})",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = expand_input_paths(args.inputs)
    if not paths:
        print("No input JSON files matched.", file=sys.stderr)
        return 1

    analyses: list[tuple[dict, str]] = []
    for path in paths:
        try:
            analyses.append((load_analysis_json(path), str(path)))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"skip unreadable {path}: {exc}", file=sys.stderr)

    if not analyses:
        print("No readable analysis JSON files.", file=sys.stderr)
        return 1

    benchmark = build_benchmark(
        analyses,
        name=args.name,
        position=args.position,
        play_type=args.play_type,
        technique=args.technique,
        side=args.side,
        min_trust_score=args.min_trust,
    )
    out = save_benchmark(args.output, benchmark)
    print(f"Wrote {out}")
    print(f"accepted={benchmark['n_reps']} skipped={benchmark['n_skipped']}")
    covered = sum(1 for n in benchmark["feature_coverage"].values() if n > 0)
    print(f"features_with_data={covered}")
    for row in benchmark["skipped"]:
        print(f"  skipped {row['rep_id']}: {row['reason']} trust={row['trust_overall']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
