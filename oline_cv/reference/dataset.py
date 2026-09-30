"""Reference-dataset layout, manifests, validation stages, and rep selection.

Layout (all paths relative to a dataset root, by default ``data/reference``)::

    clips/       rep_001.mp4          raw film, never committed
    annotations/ rep_001.json         expert football label
    analyses/    rep_001_analysis.json    analyze_video() output
    manifests/   reference_manifest.json  what belongs to what

``rep_id`` is the join key across all four directories. The
``<rep_id>_analysis.json`` naming is deliberate: the benchmark's existing
``infer_rep_id()`` strips the ``_analysis`` suffix, so reps keep their identity
through to the built benchmark with no change to that code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from oline_cv.benchmark.features import trust_overall_score
from oline_cv.benchmark.schema import (
    DEFAULT_MIN_TRUST_SCORE,
    normalize_play_type,
    normalize_position,
    normalize_side,
    normalize_technique,
)
from oline_cv.benchmark.storage import load_json, save_json
from oline_cv.reference.schema import (
    annotation_context,
    annotation_quality,
    validate_annotation,
)

MANIFEST_SCHEMA_VERSION = "1.0"
DEFAULT_DATASET_NAME = "reference_ol_v1"
ANALYSIS_SUFFIX = "_analysis.json"

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")

STAGES: tuple[str, ...] = ("annotations", "analyzed", "benchmark-ready")


@dataclass(frozen=True)
class ReferenceDataset:
    """Filesystem view of a reference dataset root."""

    root: Path

    @classmethod
    def at(cls, root: str | Path) -> "ReferenceDataset":
        return cls(Path(root))

    @property
    def clips_dir(self) -> Path:
        return self.root / "clips"

    @property
    def annotations_dir(self) -> Path:
        return self.root / "annotations"

    @property
    def analyses_dir(self) -> Path:
        return self.root / "analyses"

    @property
    def manifests_dir(self) -> Path:
        return self.root / "manifests"

    def ensure_dirs(self) -> None:
        for d in (
            self.clips_dir,
            self.annotations_dir,
            self.analyses_dir,
            self.manifests_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    # --- discovery ----------------------------------------------------

    def discover_clips(self) -> dict[str, list[Path]]:
        """Map rep_id -> clip files. A list so duplicates stay visible."""
        found: dict[str, list[Path]] = {}
        if not self.clips_dir.is_dir():
            return found
        for path in sorted(self.clips_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            found.setdefault(path.stem, []).append(path)
        return found

    def discover_annotations(self) -> dict[str, Path]:
        """Map rep_id -> annotation file."""
        found: dict[str, Path] = {}
        if not self.annotations_dir.is_dir():
            return found
        for path in sorted(self.annotations_dir.glob("*.json")):
            found[path.stem] = path
        return found

    def analysis_path(self, rep_id: str) -> Path:
        return self.analyses_dir / f"{rep_id}{ANALYSIS_SUFFIX}"

    def has_analysis(self, rep_id: str) -> bool:
        return self.analysis_path(rep_id).is_file()

    def rep_ids(self) -> list[str]:
        """Every rep_id implied by a clip or an annotation."""
        return sorted(set(self.discover_clips()) | set(self.discover_annotations()))

    def load_annotation(self, rep_id: str) -> dict[str, Any]:
        return load_json(self.discover_annotations()[rep_id])

    def relpath(self, path: Path) -> str:
        """Path relative to the dataset root, POSIX-style, for manifests."""
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()


# --- manifest -----------------------------------------------------------


def build_manifest(
    dataset: ReferenceDataset,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> dict[str, Any]:
    """Connect each rep's video, annotation, and analysis by rep_id.

    Stores relative paths only — the manifest points at data, it never copies
    it. ``analysis`` is null until the batch analyzer has run.
    """
    clips = dataset.discover_clips()
    annotations = dataset.discover_annotations()

    reps: list[dict[str, Any]] = []
    for rep_id in dataset.rep_ids():
        clip_list = clips.get(rep_id) or []
        analysis = dataset.analysis_path(rep_id)
        reps.append(
            {
                "rep_id": rep_id,
                "video": dataset.relpath(clip_list[0]) if clip_list else None,
                "annotation": (
                    dataset.relpath(annotations[rep_id])
                    if rep_id in annotations
                    else None
                ),
                "analysis": dataset.relpath(analysis) if analysis.is_file() else None,
            }
        )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_reps": len(reps),
        "reps": reps,
    }


def write_manifest(
    dataset: ReferenceDataset,
    *,
    dataset_name: str = DEFAULT_DATASET_NAME,
    output: str | Path | None = None,
) -> Path:
    manifest = build_manifest(dataset, dataset_name=dataset_name)
    target = Path(output) if output else dataset.manifests_dir / "reference_manifest.json"
    return save_json(target, manifest)


# --- validation ---------------------------------------------------------


def validate_dataset(
    dataset: ReferenceDataset,
    *,
    stage: str = "annotations",
    min_trust: float = DEFAULT_MIN_TRUST_SCORE,
) -> dict[str, Any]:
    """Check dataset integrity for a given pipeline stage.

    Stages are cumulative:

    * ``annotations``    — annotation valid, video present, ids consistent.
    * ``analyzed``       — plus a readable analysis JSON for every rep.
    * ``benchmark-ready``— plus a usable trust score on every "good" rep.

    A missing analysis is only an error from the ``analyzed`` stage onward, so
    running this right after annotating does not fail the dataset.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r} (expected one of {list(STAGES)})")

    clips = dataset.discover_clips()
    annotations = dataset.discover_annotations()
    rep_ids = dataset.rep_ids()

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    quality_counts: dict[str, int] = {"good": 0, "not_good": 0, "uncertain": 0}
    annotated = 0
    missing_analysis = 0
    below_trust = 0

    def add_error(rep_id: str, message: str) -> None:
        errors.append({"rep_id": rep_id, "error": message})

    def add_warning(rep_id: str, message: str) -> None:
        warnings.append({"rep_id": rep_id, "warning": message})

    # Duplicate rep ids: same stem appearing as more than one video file.
    for rep_id, paths in sorted(clips.items()):
        if len(paths) > 1:
            names = ", ".join(p.name for p in paths)
            add_error(rep_id, f"duplicate rep id across clips: {names}")

    for rep_id in rep_ids:
        clip_list = clips.get(rep_id) or []
        if not clip_list:
            add_error(rep_id, "annotation has no matching video in clips/")

        ann_path = annotations.get(rep_id)
        if ann_path is None:
            add_error(rep_id, "video has no annotation in annotations/")
            continue

        try:
            annotation = load_json(ann_path)
        except (OSError, ValueError) as exc:
            add_error(rep_id, f"unreadable annotation: {exc}")
            continue

        problems = validate_annotation(annotation, rep_id=rep_id)
        if problems:
            for problem in problems:
                add_error(rep_id, problem)
            continue

        annotated += 1
        quality = annotation_quality(annotation)
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

        # The referenced video_file must be the file actually on disk.
        video_file = str(annotation.get("video_file") or "")
        if clip_list and video_file and clip_list[0].name != video_file:
            add_error(
                rep_id,
                f"video_file {video_file!r} but clips/ holds {clip_list[0].name!r}",
            )

        context = annotation_context(annotation)
        for field in ("position", "play_type", "technique", "side"):
            if context[field] == "unknown":
                add_warning(rep_id, f"{field} is unknown; rep cannot match a {field} filter")

        analysis_file = dataset.analysis_path(rep_id)
        if not analysis_file.is_file():
            missing_analysis += 1
            if stage in ("analyzed", "benchmark-ready"):
                add_error(rep_id, "missing analysis; run analyze_reference_dataset.py")
            continue

        if stage in ("analyzed", "benchmark-ready"):
            try:
                analysis = load_json(analysis_file)
            except (OSError, ValueError) as exc:
                add_error(rep_id, f"unreadable analysis: {exc}")
                continue

            if stage == "benchmark-ready" and quality == "good":
                trust = trust_overall_score(analysis)
                if trust is None:
                    add_error(rep_id, "analysis has no trust.overall.score")
                elif trust < min_trust:
                    below_trust += 1
                    add_warning(
                        rep_id,
                        f"trust {trust:.3f} below {min_trust:.2f}; will be excluded",
                    )

    return {
        "dataset_root": str(dataset.root),
        "stage": stage,
        "min_trust": float(min_trust),
        "total_clips": len(clips),
        "total_reps": len(rep_ids),
        "annotated": annotated,
        "quality_counts": quality_counts,
        "missing_analysis": missing_analysis,
        "below_trust": below_trust,
        "errors": errors,
        "warnings": warnings,
        "ok": not errors,
    }


def format_validation_report(report: dict[str, Any]) -> str:
    """Human-readable summary for the CLI."""
    counts = report["quality_counts"]
    lines = [
        "Reference dataset",
        "-----------------",
        f"Root:            {report['dataset_root']}",
        f"Stage:           {report['stage']}",
        f"Total clips:     {report['total_clips']}",
        f"Annotated:       {report['annotated']}",
        f"Good:            {counts.get('good', 0)}",
        f"Not good:        {counts.get('not_good', 0)}",
        f"Uncertain:       {counts.get('uncertain', 0)}",
        f"Missing analysis:{report['missing_analysis']:>3}",
    ]
    if report["warnings"]:
        lines.append("")
        lines.append(f"Warnings ({len(report['warnings'])}):")
        for row in report["warnings"]:
            lines.append(f"  - {row['rep_id']}: {row['warning']}")
    if report["errors"]:
        lines.append("")
        lines.append(f"Errors ({len(report['errors'])}):")
        for row in report["errors"]:
            lines.append(f"  - {row['rep_id']}: {row['error']}")
    else:
        lines.append("")
        lines.append("No errors.")
    return "\n".join(lines)


# --- benchmark-eligible selection ---------------------------------------


def select_benchmark_reps(
    dataset: ReferenceDataset,
    *,
    position: str | None = None,
    play_type: str | None = None,
    technique: str | None = None,
    side: str | None = None,
    quality: Iterable[str] = ("good",),
    min_trust: float = DEFAULT_MIN_TRUST_SCORE,
    dataset_name: str = DEFAULT_DATASET_NAME,
) -> dict[str, Any]:
    """Pick reps eligible for a benchmark and record why each one was dropped.

    Filters are applied in a fixed order — annotation validity, expert
    quality, football context, then measured trust — so every excluded rep has
    exactly one reason, and the football reasons are reported before the CV
    ones. A filter left as ``None`` is simply not applied.
    """
    wanted_quality = {str(q).strip().lower() for q in quality}
    filters = {
        "position": normalize_position(position) if position else None,
        "play_type": normalize_play_type(play_type) if play_type else None,
        "technique": normalize_technique(technique) if technique else None,
        "side": normalize_side(side) if side else None,
    }

    annotations = dataset.discover_annotations()
    selected: list[str] = []
    analyses: list[str] = []
    excluded: list[dict[str, Any]] = []

    def drop(rep_id: str, reason: str, detail: str | None = None) -> None:
        row: dict[str, Any] = {"rep_id": rep_id, "reason": reason}
        if detail:
            row["detail"] = detail
        excluded.append(row)

    for rep_id in dataset.rep_ids():
        ann_path = annotations.get(rep_id)
        if ann_path is None:
            drop(rep_id, "missing_annotation")
            continue

        try:
            annotation = load_json(ann_path)
        except (OSError, ValueError) as exc:
            drop(rep_id, "annotation_unreadable", str(exc))
            continue

        problems = validate_annotation(annotation, rep_id=rep_id)
        if problems:
            drop(rep_id, "annotation_invalid", problems[0])
            continue

        # 1. Expert football label decides eligibility before any CV number.
        rep_quality = annotation_quality(annotation)
        if rep_quality not in wanted_quality:
            drop(rep_id, f"expert_quality_{rep_quality}")
            continue

        # 2. Play context.
        context = annotation_context(annotation)
        mismatch = next(
            (
                field
                for field, want in filters.items()
                if want is not None and context[field] != want
            ),
            None,
        )
        if mismatch is not None:
            drop(
                rep_id,
                f"{mismatch}_mismatch",
                f"annotation says {context[mismatch]!r}, filter wants {filters[mismatch]!r}",
            )
            continue

        # 3. Measured data must exist and be trustworthy enough.
        analysis_file = dataset.analysis_path(rep_id)
        if not analysis_file.is_file():
            drop(rep_id, "missing_analysis")
            continue

        try:
            analysis = load_json(analysis_file)
        except (OSError, ValueError) as exc:
            drop(rep_id, "analysis_unreadable", str(exc))
            continue

        trust = trust_overall_score(analysis)
        if trust is None:
            drop(rep_id, "missing_trust")
            continue
        if trust < min_trust:
            drop(rep_id, "trust_below_threshold", f"trust {trust:.3f} < {min_trust:.2f}")
            continue

        selected.append(rep_id)
        analyses.append(dataset.relpath(analysis_file))

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(dataset.root),
        "context": {
            "position": filters["position"] or "unknown",
            "play_type": filters["play_type"] or "pass",
            "technique": filters["technique"] or "unknown",
            "side": filters["side"] or "unknown",
        },
        "selection": {
            "quality": sorted(wanted_quality),
            "min_trust_score": float(min_trust),
            "filters": {k: v for k, v in filters.items() if v is not None},
        },
        "n_reps": len(selected),
        "n_excluded": len(excluded),
        "reps": selected,
        "analyses": analyses,
        "excluded": excluded,
    }


def resolve_manifest_analyses(manifest_path: str | Path) -> tuple[list[Path], dict[str, Any]]:
    """Resolve a prepared manifest into analysis paths + benchmark context.

    Used by ``scripts/build_benchmark.py --manifest``. Relative paths resolve
    against ``dataset_root`` when it still exists, otherwise against the
    manifest's own directory, so a dataset stays usable after being moved.
    """
    path = Path(manifest_path)
    manifest = load_json(path)

    if "reps" not in manifest:
        raise ValueError(f"{path} is not a reference manifest (missing 'reps')")

    root = Path(manifest.get("dataset_root") or "")
    if not root.is_absolute():
        root = (path.parent / root).resolve() if str(root) else path.parent
    if not root.is_dir():
        root = path.parent.parent if path.parent.name == "manifests" else path.parent

    listed = manifest.get("analyses") or []
    if listed:
        candidates = [Path(p) for p in listed]
    else:
        candidates = [Path("analyses") / f"{rep}{ANALYSIS_SUFFIX}" for rep in manifest["reps"]]

    resolved: list[Path] = []
    for candidate in candidates:
        full = candidate if candidate.is_absolute() else root / candidate
        if full.is_file():
            resolved.append(full)

    context = dict(manifest.get("context") or {})
    selection = manifest.get("selection") or {}
    if "min_trust_score" in selection:
        context["min_trust_score"] = selection["min_trust_score"]
    context["dataset_name"] = manifest.get("dataset_name")
    return resolved, context
