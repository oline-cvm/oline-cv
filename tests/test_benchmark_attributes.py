"""Five-attribute good/bad calibration — synthetic analyses only."""

from oline_cv.benchmark.attributes import fit_attribute_model, score_attributes
from oline_cv.reference.schema import annotation_attributes, make_annotation, validate_annotation


def _analysis(rep_id: str, **metrics):
    s = {"play_type": "pass", **metrics}
    initiated = s.pop("initiated_by", None)
    if initiated:
        s["initiated_by"] = initiated
    return {
        "rep_id": rep_id,
        "play_type": "pass",
        "rep_summary": s,
        "modules": {
            "initial_quicks": {"initiated_by": initiated, "first_step_acceleration": s.get("first_step_acceleration")},
            "hands": {"time_to_first_contact_ms": s.get("punch_ms")},
        },
    }


def test_structured_attributes_round_trip():
    ann = make_annotation(
        rep_id="r",
        video_file="r.mp4",
        quality="good",
        attributes={"stance": "good", "first_step": "n/a", "feet": "bad"},
    )
    assert validate_annotation(ann, rep_id="r") == []
    attrs = annotation_attributes(ann)
    assert attrs["stance"] == "good"
    assert attrs["first_step"] == "n/a"
    assert attrs["feet"] == "bad"
    assert attrs["hands"] == "n/a"


def test_attributes_parse_from_notes_if_missing_object():
    ann = make_annotation(
        rep_id="r",
        video_file="r.mp4",
        quality="not_good",
        notes=["stance: good", "first_step: bad", "player: Casey"],
    )
    del ann["expert"]["attributes"]
    attrs = annotation_attributes(ann)
    assert attrs["stance"] == "good"
    assert attrs["first_step"] == "bad"


def test_fit_and_score_first_step_from_reaction_time():
    labeled = []
    for i, (ms, label) in enumerate([(140, "good"), (150, "good"), (280, "bad"), (300, "bad")]):
        ann = make_annotation(
            rep_id=f"r{i}",
            video_file=f"r{i}.mp4",
            quality="uncertain",
            attributes={"first_step": label, "stance": "good"},
        )
        labeled.append((_analysis(f"r{i}", reaction_time_ms=ms, initiated_by="foot"), ann))
    model = fit_attribute_model(labeled)
    assert model["attributes"]["first_step"]["n_rules"] >= 1
    fast = score_attributes(_analysis("new", reaction_time_ms=145, initiated_by="foot"), model)
    slow = score_attributes(_analysis("new", reaction_time_ms=290, initiated_by="foot"), model)
    assert fast["attributes"]["first_step"]["verdict"] == "good"
    assert slow["attributes"]["first_step"]["verdict"] == "bad"


def test_n_a_labels_are_not_used_for_fit():
    labeled = [
        (
            _analysis("a", reaction_time_ms=140),
            make_annotation(rep_id="a", video_file="a.mp4", quality="good", attributes={"first_step": "n/a"}),
        ),
        (
            _analysis("b", reaction_time_ms=300),
            make_annotation(rep_id="b", video_file="b.mp4", quality="not_good", attributes={"first_step": "bad"}),
        ),
    ]
    model = fit_attribute_model(labeled)
    # Only one class for first_step → no midpoint rule.
    assert model["attributes"]["first_step"]["n_rules"] == 0
    scored = score_attributes(_analysis("c", reaction_time_ms=200), model)
    assert scored["attributes"]["first_step"]["verdict"] == "uncertain"
