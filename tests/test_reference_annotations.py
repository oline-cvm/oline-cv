"""Expert-annotation schema: what is required, what may stay unknown.

The annotation is the football label, so validation has to be strict about
identity (rep_id / video_file) and the quality vocabulary, but permissive
about context the annotator genuinely cannot read off the film.
"""

import pytest

from oline_cv.reference.schema import (
    ANNOTATION_SCHEMA_VERSION,
    QUALITY_VALUES,
    annotation_context,
    annotation_quality,
    make_annotation,
    validate_annotation,
)


def _valid() -> dict:
    return make_annotation(
        rep_id="rep_001",
        video_file="rep_001.mp4",
        position="LT",
        level="high_school",
        play_type="pass",
        technique="vertical_set",
        side="left",
        quality="good",
        notes=["good initial set", "stays square"],
    )


def test_make_annotation_round_trips_clean():
    ann = _valid()
    assert validate_annotation(ann, rep_id="rep_001") == []
    assert ann["schema_version"] == ANNOTATION_SCHEMA_VERSION
    assert ann["player"]["position"] == "LT"
    assert ann["play"]["technique"] == "vertical_set"


def test_expert_notes_are_preserved_verbatim():
    notes = ["heavy hands", "  late off the ball  ", "SETS TOO FLAT"]
    ann = make_annotation(
        rep_id="rep_002", video_file="rep_002.mp4", quality="not_good", notes=notes
    )
    # No trimming, casing, reordering, or rewriting of what the expert wrote.
    assert ann["expert"]["notes"] == notes


def test_quality_vocabulary_is_neutral_and_not_numeric():
    assert set(QUALITY_VALUES) == {"good", "not_good", "uncertain"}
    ann = _valid()
    assert ann["expert"]["quality"] == "good"
    assert not isinstance(ann["expert"]["quality"], (int, float))


@pytest.mark.parametrize("quality", ["good", "not_good", "uncertain"])
def test_all_quality_values_validate(quality):
    ann = make_annotation(rep_id="r", video_file="r.mp4", quality=quality)
    assert validate_annotation(ann, rep_id="r") == []


def test_unknown_context_is_allowed():
    """An annotator who cannot tell the technique must not be forced to guess."""
    ann = make_annotation(rep_id="rep_003", video_file="rep_003.mp4", quality="good")
    assert validate_annotation(ann, rep_id="rep_003") == []
    ctx = annotation_context(ann)
    assert ctx["position"] == "unknown"
    assert ctx["technique"] == "unknown"
    assert ctx["side"] == "unknown"


def test_unannotated_play_type_does_not_become_pass():
    """normalize_play_type() defaults to "pass"; an unknown rep must not."""
    ann = make_annotation(rep_id="rep_004", video_file="rep_004.mp4", quality="good")
    assert ann["play"]["play_type"] == "unknown"
    assert annotation_context(ann)["play_type"] == "unknown"


def test_missing_quality_is_an_error():
    ann = _valid()
    del ann["expert"]["quality"]
    errors = validate_annotation(ann, rep_id="rep_001")
    assert any("expert.quality" in e for e in errors)


def test_invalid_quality_is_rejected():
    ann = _valid()
    ann["expert"]["quality"] = "elite"
    errors = validate_annotation(ann, rep_id="rep_001")
    assert any("invalid expert.quality" in e for e in errors)


def test_make_annotation_rejects_invalid_quality():
    with pytest.raises(ValueError):
        make_annotation(rep_id="r", video_file="r.mp4", quality="amazing")


def test_typoed_position_is_an_error_not_silently_unknown():
    ann = _valid()
    ann["player"]["position"] = "LTT"
    errors = validate_annotation(ann, rep_id="rep_001")
    assert any("invalid position" in e for e in errors)


def test_invalid_play_type_is_rejected():
    ann = _valid()
    ann["play"]["play_type"] = "screen"
    errors = validate_annotation(ann, rep_id="rep_001")
    assert any("invalid play_type" in e for e in errors)


def test_rep_id_must_match_filename():
    ann = _valid()
    errors = validate_annotation(ann, rep_id="rep_999")
    assert any("does not match its filename" in e for e in errors)


def test_video_file_must_match_rep_id():
    ann = _valid()
    ann["video_file"] = "some_other_clip.mp4"
    errors = validate_annotation(ann, rep_id="rep_001")
    assert any("does not match rep_id" in e for e in errors)


def test_missing_required_fields_are_reported():
    errors = validate_annotation({"schema_version": "1.0"})
    joined = " ".join(errors)
    assert "rep_id" in joined and "video_file" in joined and "expert.quality" in joined


def test_unsupported_schema_version_is_reported():
    ann = _valid()
    ann["schema_version"] = "99.0"
    errors = validate_annotation(ann, rep_id="rep_001")
    assert any("schema_version" in e for e in errors)


def test_notes_must_be_a_list_of_strings():
    ann = _valid()
    ann["expert"]["notes"] = "good set"
    errors = validate_annotation(ann, rep_id="rep_001")
    assert any("expert.notes" in e for e in errors)


def test_non_object_annotation_is_rejected():
    assert validate_annotation(["not", "an", "object"]) == [
        "annotation is not a JSON object"
    ]


def test_annotation_quality_defaults_to_uncertain():
    assert annotation_quality({}) == "uncertain"
    assert annotation_quality({"expert": {"quality": "GOOD"}}) == "good"
