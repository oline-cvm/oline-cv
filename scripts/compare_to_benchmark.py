#!/usr/bin/env python3
"""Compare one analyze_video() JSON file to a saved benchmark profile.

Example:
  python scripts/compare_to_benchmark.py \\
      --analysis outputs/new_rep_analysis.json \\
      --benchmark benchmarks/elite_lt_vertical_set_v1.json \\
      --position LT --technique vertical_set
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.benchmark.compare import compare_rep
from oline_cv.benchmark.features import extract_features
from oline_cv.benchmark.storage import load_analysis_json, load_benchmark, save_json


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare one OL analysis JSON to a benchmark")
    p.add_argument("--analysis", required=True, help="Path to *_analysis.json")
    p.add_argument("--benchmark", required=True, help="Path to benchmark JSON")
    p.add_argument("--output", default=None, help="Optional comparison JSON path")
    p.add_argument("--position", default=None)
    p.add_argument("--technique", default=None)
    p.add_argument("--side", default=None)
    p.add_argument("--play-type", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    analysis = load_analysis_json(args.analysis)
    benchmark = load_benchmark(args.benchmark)
    record = extract_features(
        analysis,
        source_analysis_path=args.analysis,
        position=args.position,
        technique=args.technique,
        side=args.side,
        play_type=args.play_type,
    )
    comparison = compare_rep(record, benchmark)
    comparison["player_record"] = record

    print(f"benchmark={comparison['benchmark']}")
    print(f"valid_comparisons={comparison['valid_comparisons']}")
    if comparison["context_warnings"]:
        print("context_warnings:")
        for w in comparison["context_warnings"]:
            print(f"  {w}")
    print("largest_deviations:")
    for row in comparison["largest_deviations"]:
        print(
            f"  {row['feature']}: player={row['player_value']} "
            f"ref_mean={row['benchmark_mean']} z={row['z_score']} {row['status']}"
        )
    print("closest_matches:")
    for row in comparison["closest_matches"]:
        print(
            f"  {row['feature']}: player={row['player_value']} "
            f"ref_mean={row['benchmark_mean']} z={row['z_score']} {row['status']}"
        )

    if args.output:
        path = save_json(args.output, comparison)
        print(f"Wrote {path}")
    else:
        print(json.dumps({k: comparison[k] for k in ("valid_comparisons", "largest_deviations", "closest_matches")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
