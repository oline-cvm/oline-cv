#!/usr/bin/env python3
"""Fit five-attribute good/bad rules from reference analyses + expert labels.

  python scripts/fit_attribute_model.py \\
      --dataset data/reference \\
      --output benchmarks/attribute_model_v1.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.benchmark.attributes import fit_attribute_model
from oline_cv.benchmark.storage import load_json, save_json
from oline_cv.reference.dataset import ReferenceDataset
from oline_cv.reference.schema import ATTRIBUTE_KEYS, annotation_attributes


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit attribute good/bad rules from labeled analyses")
    p.add_argument("--dataset", default="data/reference")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    dataset = ReferenceDataset.at(args.dataset)
    labeled: list[tuple[dict, dict]] = []
    missing: list[str] = []
    for rep_id in dataset.rep_ids():
        if rep_id not in dataset.discover_annotations():
            continue
        if not dataset.has_analysis(rep_id):
            missing.append(rep_id)
            continue
        analysis = load_json(dataset.analysis_path(rep_id))
        annotation = dataset.load_annotation(rep_id)
        labeled.append((analysis, annotation))

    if not labeled:
        print("No labeled analyses found. Run analyze_reference_dataset.py first.", file=sys.stderr)
        if missing:
            print(f"Missing analysis for: {', '.join(missing)}", file=sys.stderr)
        return 1

    model = fit_attribute_model(labeled)
    save_json(args.output, model)
    print(f"Wrote {args.output} from {len(labeled)} reps")
    if missing:
        print(f"Skipped (no analysis): {', '.join(missing)}")
    for attr in ATTRIBUTE_KEYS:
        n = model["attributes"][attr]["n_rules"]
        print(f"  {attr}: {n} separating metric(s)")
        labels = [annotation_attributes(a)[attr] for _, a in labeled]
        print(f"    expert labels: good={labels.count('good')} bad={labels.count('bad')} n/a={labels.count('n/a')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
