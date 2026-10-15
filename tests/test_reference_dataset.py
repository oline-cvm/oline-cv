"""Reference-dataset discovery, manifests, staged validation, and selection.

Everything here runs on synthetic JSON and empty placeholder files in
``tmp_path``. No video is decoded and no model is loaded, so the suite stays
runnable on a machine without YOLO, CUDA, WHAM, or SMPL.
"""

import json

import pytest

from oline_cv.benchmark.schema import DEFAULT_MIN_TRUST_SCORE
from oline_cv.reference.dataset import (
    ReferenceDataset,
    build_manifest,
    resolve_manifest_analyses,
    select_benchmark_reps,
    validate_dataset,
    write_manifest,
)
from oline_cv.reference.schema import make_annotation


def _analysis(trust: float | None = 0.8) -> dict:
    """Minimal stand-in for analyze_video() output."""
    result = {
        "video": {"path": "clips/rep.mp4", "fps": 30.0, "frame_count": 120},
        "play_type": "pass",
        "rep_summary": {
            "reaction_time_ms": 200.0,
            "mean_knee_flexion_deg": 128.0,
            "set_depth": 0.42,
        },
        "modules": {},
    }
    if trust is not None:
        result["trust"] = {"overall": {"score": trust, "level": "high"}}
    return result


def _dataset(tmp_path):
    ds = ReferenceDataset.at(tmp_path / "reference")
    ds.ensure_dirs()
    return ds


def _add_rep(
    ds,
    rep_id,
    *,
    quality="good",
    position="LT",
    play_type="pass",
    technique="vertical_set",
    side="left",
    trust=0.8,
    with_clip=True,
    with_analysis=True,
    notes=None,
):
    if with_clip:
        (ds.clips_dir / f"{rep_id}.mp4").write_bytes(b"")
    annotation = make_annotation(
        rep_id=rep_id,
        video_file=f"{rep_id}.mp4",
        position=position,
        play_type=play_type,
        technique=technique,
        side=side,
        quality=quality,
        notes=notes or [],
    )
    (ds.annotations_dir / f"{rep_id}.json").write_text(json.dumps(annotation))
    if with_analysis:
        ds.analysis_path(rep_id).write_text(json.dumps(_analysis(trust)))
    return annotation


# --- discovery + manifest ----------------------------------------------


def test_manifest_links_video_annotation_and_analysis(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001")
    _add_rep(ds, "rep_002", with_analysis=False)

    manifest = build_manifest(ds, dataset_name="reference_ol_v1")

    assert manifest["dataset_name"] == "reference_ol_v1"
    assert manifest["n_reps"] == 2
    by_id = {r["rep_id"]: r for r in manifest["reps"]}
    assert by_id["rep_001"]["video"] == "clips/rep_001.mp4"
    assert by_id["rep_001"]["annotation"] == "annotations/rep_001.json"
    assert by_id["rep_001"]["analysis"] == "analyses/rep_001_analysis.json"
    # Not analyzed yet, so the link is explicitly absent rather than guessed.
    assert by_id["rep_002"]["analysis"] is None


def test_manifest_stores_paths_not_copies_of_the_data(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", notes=["good initial set"])
    manifest = build_manifest(ds)
    blob = json.dumps(manifest)
    assert "good initial set" not in blob
    assert "reaction_time_ms" not in blob


def test_write_manifest_defaults_into_manifests_dir(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001")
    path = write_manifest(ds)
    assert path == ds.manifests_dir / "reference_manifest.json"
    assert json.loads(path.read_text())["n_reps"] == 1


def test_analysis_naming_keeps_rep_id_recoverable(tmp_path):
    """<rep_id>_analysis.json is what the benchmark's infer_rep_id() expects."""
    from oline_cv.benchmark.features import infer_rep_id

    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_007")
    path = ds.analysis_path("rep_007")
    assert path.name == "rep_007_analysis.json"
    assert infer_rep_id(json.loads(path.read_text()), str(path)) == "rep_007"


# --- validation ---------------------------------------------------------


def test_duplicate_rep_id_across_clips_is_detected(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001")
    (ds.clips_dir / "rep_001.mov").write_bytes(b"")

    report = validate_dataset(ds, stage="annotations")

    assert not report["ok"]
    assert any("duplicate rep id" in e["error"] for e in report["errors"])


def test_annotation_without_a_clip_is_an_error(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", with_clip=False, with_analysis=False)

    report = validate_dataset(ds, stage="annotations")

    assert not report["ok"]
    assert any("no matching video" in e["error"] for e in report["errors"])


def test_clip_without_an_annotation_is_an_error(tmp_path):
    ds = _dataset(tmp_path)
    (ds.clips_dir / "rep_099.mp4").write_bytes(b"")

    report = validate_dataset(ds, stage="annotations")

    assert not report["ok"]
    assert any("no annotation" in e["error"] for e in report["errors"])


def test_invalid_annotation_is_reported(tmp_path):
    ds = _dataset(tmp_path)
    (ds.clips_dir / "rep_001.mp4").write_bytes(b"")
    (ds.annotations_dir / "rep_001.json").write_text(
        json.dumps({"schema_version": "1.0", "rep_id": "rep_001"})
    )

    report = validate_dataset(ds, stage="annotations")

    assert not report["ok"]
    assert any("video_file" in e["error"] for e in report["errors"])


def test_unreadable_annotation_is_reported(tmp_path):
    ds = _dataset(tmp_path)
    (ds.clips_dir / "rep_001.mp4").write_bytes(b"")
    (ds.annotations_dir / "rep_001.json").write_text("{not json")

    report = validate_dataset(ds, stage="annotations")

    assert not report["ok"]
    assert any("unreadable annotation" in e["error"] for e in report["errors"])


def test_missing_analysis_does_not_fail_the_annotations_stage(tmp_path):
    """Validating right after annotating must not fail for un-run analyses."""
    ds = _dataset(tmp_path)
    for rep in ("rep_001", "rep_002"):
        _add_rep(ds, rep, with_analysis=False)

    report = validate_dataset(ds, stage="annotations")

    assert report["ok"]
    assert report["missing_analysis"] == 2


def test_missing_analysis_fails_the_analyzed_stage(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", with_analysis=False)

    report = validate_dataset(ds, stage="analyzed")

    assert not report["ok"]
    assert any("missing analysis" in e["error"] for e in report["errors"])


def test_benchmark_ready_stage_requires_trust_on_good_reps(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", quality="good", trust=None)

    report = validate_dataset(ds, stage="benchmark-ready")

    assert not report["ok"]
    assert any("trust" in e["error"] for e in report["errors"])


def test_low_trust_is_a_warning_not_an_error(tmp_path):
    """Below-threshold reps get excluded at selection, not failed here."""
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", quality="good", trust=0.20)

    report = validate_dataset(ds, stage="benchmark-ready")

    assert report["ok"]
    assert report["below_trust"] == 1
    assert any("below" in w["warning"] for w in report["warnings"])


def test_validation_summary_counts_each_quality(tmp_path):
    ds = _dataset(tmp_path)
    for i in range(4):
        _add_rep(ds, f"rep_{i:03d}", quality="good")
    _add_rep(ds, "rep_100", quality="not_good")
    _add_rep(ds, "rep_101", quality="uncertain")

    report = validate_dataset(ds, stage="annotations")

    assert report["ok"]
    assert report["total_clips"] == 6
    assert report["annotated"] == 6
    assert report["quality_counts"] == {"good": 4, "not_good": 1, "uncertain": 1}


def test_unknown_stage_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        validate_dataset(_dataset(tmp_path), stage="whenever")


# --- selection ----------------------------------------------------------


def test_quality_filter_excludes_non_good_reps_with_reasons(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", quality="good")
    _add_rep(ds, "rep_002", quality="not_good")
    _add_rep(ds, "rep_003", quality="uncertain")

    manifest = select_benchmark_reps(ds, quality=("good",))

    assert manifest["reps"] == ["rep_001"]
    reasons = {r["rep_id"]: r["reason"] for r in manifest["excluded"]}
    assert reasons["rep_002"] == "expert_quality_not_good"
    assert reasons["rep_003"] == "expert_quality_uncertain"


def test_quality_filter_can_accept_several_verdicts(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", quality="good")
    _add_rep(ds, "rep_002", quality="uncertain")

    manifest = select_benchmark_reps(ds, quality=("good", "uncertain"))

    assert manifest["reps"] == ["rep_001", "rep_002"]


def test_context_filters_exclude_mismatches(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", position="LT", technique="vertical_set")
    _add_rep(ds, "rep_002", position="RT", technique="vertical_set")
    _add_rep(ds, "rep_003", position="LT", technique="jump_set")
    _add_rep(ds, "rep_004", position="LT", technique="vertical_set", play_type="run")

    manifest = select_benchmark_reps(
        ds, position="LT", play_type="pass", technique="vertical_set"
    )

    assert manifest["reps"] == ["rep_001"]
    reasons = {r["rep_id"]: r["reason"] for r in manifest["excluded"]}
    assert reasons["rep_002"] == "position_mismatch"
    assert reasons["rep_003"] == "technique_mismatch"
    assert reasons["rep_004"] == "play_type_mismatch"


def test_unapplied_filters_do_not_exclude_anything(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", position="LT")
    _add_rep(ds, "rep_002", position="RG")

    manifest = select_benchmark_reps(ds)

    assert manifest["reps"] == ["rep_001", "rep_002"]


def test_rep_with_unknown_context_never_matches_a_filter(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", technique=None)

    manifest = select_benchmark_reps(ds, technique="vertical_set")

    assert manifest["reps"] == []
    assert manifest["excluded"][0]["reason"] == "technique_mismatch"


def test_trust_filter_uses_the_benchmark_default_threshold(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", trust=0.80)
    _add_rep(ds, "rep_002", trust=0.20)

    manifest = select_benchmark_reps(ds)

    assert manifest["selection"]["min_trust_score"] == DEFAULT_MIN_TRUST_SCORE
    assert manifest["reps"] == ["rep_001"]
    assert manifest["excluded"][0]["reason"] == "trust_below_threshold"


def test_trust_threshold_is_configurable(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", trust=0.30)

    assert select_benchmark_reps(ds, min_trust=0.25)["reps"] == ["rep_001"]
    assert select_benchmark_reps(ds, min_trust=0.55)["reps"] == []


def test_missing_analysis_and_trust_are_distinct_reasons(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", with_analysis=False)
    _add_rep(ds, "rep_002", trust=None)

    manifest = select_benchmark_reps(ds)

    reasons = {r["rep_id"]: r["reason"] for r in manifest["excluded"]}
    assert reasons["rep_001"] == "missing_analysis"
    assert reasons["rep_002"] == "missing_trust"


def test_football_quality_is_judged_before_cv_trust(tmp_path):
    """A not_good rep is excluded for football reasons even if trust is high."""
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", quality="not_good", trust=0.99)

    manifest = select_benchmark_reps(ds)

    assert manifest["excluded"][0]["reason"] == "expert_quality_not_good"


def test_invalid_annotation_is_excluded_with_a_reason(tmp_path):
    ds = _dataset(tmp_path)
    (ds.clips_dir / "rep_001.mp4").write_bytes(b"")
    (ds.annotations_dir / "rep_001.json").write_text(json.dumps({"rep_id": "rep_001"}))

    manifest = select_benchmark_reps(ds)

    assert manifest["reps"] == []
    assert manifest["excluded"][0]["reason"] == "annotation_invalid"


def test_every_rep_is_either_included_or_explained(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", quality="good")
    _add_rep(ds, "rep_002", quality="not_good")
    _add_rep(ds, "rep_003", trust=0.1)
    _add_rep(ds, "rep_004", position="RT")
    _add_rep(ds, "rep_005", with_analysis=False)

    manifest = select_benchmark_reps(ds, position="LT")

    accounted = set(manifest["reps"]) | {r["rep_id"] for r in manifest["excluded"]}
    assert accounted == set(ds.rep_ids())
    assert manifest["n_reps"] + manifest["n_excluded"] == len(ds.rep_ids())


def test_selection_manifest_records_its_own_criteria(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001")

    manifest = select_benchmark_reps(
        ds, position="LT", play_type="pass", technique="vertical_set", side="left"
    )

    assert manifest["context"] == {
        "position": "LT",
        "play_type": "pass",
        "technique": "vertical_set",
        "side": "left",
    }
    assert manifest["selection"]["quality"] == ["good"]
    assert manifest["analyses"] == ["analyses/rep_001_analysis.json"]


# --- handoff to the existing benchmark builder --------------------------


def test_prepared_manifest_resolves_to_analysis_paths(tmp_path):
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001")
    _add_rep(ds, "rep_002")
    _add_rep(ds, "rep_003", quality="not_good")

    manifest = select_benchmark_reps(ds, position="LT", technique="vertical_set")
    manifest_path = ds.manifests_dir / "lt_pass_vertical_set_v1.json"
    manifest_path.write_text(json.dumps(manifest))

    paths, context = resolve_manifest_analyses(manifest_path)

    assert [p.name for p in paths] == [
        "rep_001_analysis.json",
        "rep_002_analysis.json",
    ]
    assert context["position"] == "LT"
    assert context["technique"] == "vertical_set"
    assert context["min_trust_score"] == DEFAULT_MIN_TRUST_SCORE


def test_manifest_resolution_survives_a_moved_dataset(tmp_path):
    """dataset_root is a hint; the manifest's own location is the fallback."""
    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001")
    manifest = select_benchmark_reps(ds)
    manifest["dataset_root"] = "/nowhere/that/exists"
    manifest_path = ds.manifests_dir / "sel.json"
    manifest_path.write_text(json.dumps(manifest))

    paths, _ = resolve_manifest_analyses(manifest_path)

    assert [p.name for p in paths] == ["rep_001_analysis.json"]


def test_resolve_rejects_a_non_manifest(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"hello": "world"}))
    with pytest.raises(ValueError):
        resolve_manifest_analyses(bad)


def test_selected_reps_build_a_benchmark_end_to_end(tmp_path):
    """The prepared manifest feeds the existing builder unchanged."""
    from oline_cv.benchmark.builder import build_benchmark
    from oline_cv.benchmark.storage import load_analysis_json

    ds = _dataset(tmp_path)
    _add_rep(ds, "rep_001", trust=0.8)
    _add_rep(ds, "rep_002", trust=0.7)
    _add_rep(ds, "rep_003", quality="not_good", trust=0.9)

    selection = select_benchmark_reps(ds, position="LT", technique="vertical_set")
    manifest_path = ds.manifests_dir / "sel.json"
    manifest_path.write_text(json.dumps(selection))
    paths, context = resolve_manifest_analyses(manifest_path)

    benchmark = build_benchmark(
        [(load_analysis_json(p), str(p)) for p in paths],
        name="lt_pass_vertical_set_v1",
        position=context["position"],
        play_type=context["play_type"],
        technique=context["technique"],
        min_trust_score=context["min_trust_score"],
    )

    assert benchmark["n_reps"] == 2
    assert sorted(benchmark["input_rep_ids"]) == ["rep_001", "rep_002"]
    # The expert-rejected rep never reaches the builder.
    assert "rep_003" not in benchmark["input_rep_ids"]
    assert benchmark["context"]["position"] == "LT"
    assert benchmark["features"]["reaction_time_ms"]["count"] == 2
