"""Tests for target association.

The behaviour worth defending: a nearby defender must never inherit the tracked
lineman's identity, and a frame with no good match must stay invalid rather than
fall back to second best.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from oline_cv.motion3d.target_association import (
    AssociationThresholds,
    InvalidReason,
    RejectReason,
    associate_frame,
    associate_sequence,
    bridge_gaps,
    iou_xyxy,
    load_associations,
    longest_valid_run,
    save_associations,
    summarize,
)

TARGET = [100.0, 100.0, 200.0, 300.0]  # 100x200 box


def test_iou_basics():
    assert iou_xyxy(TARGET, TARGET) == pytest.approx(1.0)
    assert iou_xyxy(TARGET, [400, 400, 500, 600]) == 0.0
    # Half-overlap in x only.
    assert iou_xyxy([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_exact_box_is_selected():
    a = associate_frame(7, TARGET, [TARGET], [0.9])
    assert a.valid
    sel = a.selected
    assert sel is not None and sel.index == 0
    assert sel.iou == pytest.approx(1.0)
    assert a.confidence > 0.95
    assert a.frame_index == 7


def test_picks_best_of_several_and_labels_the_rest():
    dets = [
        [400.0, 100.0, 500.0, 300.0],  # far away
        [105.0, 104.0, 203.0, 302.0],  # our guy
        [150.0, 100.0, 250.0, 300.0],  # overlapping neighbour
    ]
    a = associate_frame(0, TARGET, dets, [0.8, 0.9, 0.85])
    assert a.valid
    assert a.selected.index == 1
    assert a.candidates[0].reject_reason == RejectReason.LOW_IOU
    # The neighbour is either gated out or simply not best, but never selected.
    assert a.candidates[2].selected is False
    assert a.candidates[2].reject_reason is not None


def test_defender_in_contact_does_not_steal_identity():
    """A defender overlapping the lineman must lose on center + area agreement."""
    defender = [160.0, 90.0, 275.0, 320.0]
    lineman = [98.0, 102.0, 199.0, 298.0]
    a = associate_frame(0, TARGET, [defender, lineman], [0.95, 0.60])
    assert a.selected.index == 1, "higher detector confidence must not win"


def test_no_detections_is_invalid():
    a = associate_frame(3, TARGET, [], [])
    assert not a.valid
    assert a.invalid_reason == InvalidReason.NO_DETECTIONS
    assert a.confidence == 0.0


def test_all_rejected_never_falls_back_to_second_best():
    """The failure mode we are defending against: reconstructing the wrong body."""
    a = associate_frame(4, TARGET, [[210.0, 100.0, 310.0, 300.0]], [0.99])
    assert not a.valid
    assert a.invalid_reason == InvalidReason.ALL_REJECTED
    assert a.selected is None


def test_missing_target_bbox_is_invalid():
    a = associate_frame(5, None, [TARGET], [0.9])
    assert not a.valid
    assert a.invalid_reason == InvalidReason.NO_TARGET_BBOX


def test_area_mismatch_rejects_same_center_wrong_size():
    tiny = [145.0, 195.0, 155.0, 205.0]  # concentric but far too small
    a = associate_frame(0, TARGET, [tiny], [0.9])
    assert not a.valid
    assert a.candidates[0].reject_reason in {
        RejectReason.LOW_IOU,
        RejectReason.AREA_MISMATCH,
    }


def test_thresholds_are_configurable():
    det = [150.0, 100.0, 250.0, 300.0]  # IoU = 1/3
    strict = associate_frame(0, TARGET, [det], [0.9], AssociationThresholds(min_iou=0.5))
    loose = associate_frame(
        0, TARGET, [det], [0.9], AssociationThresholds(min_iou=0.2, min_score=0.2)
    )
    assert not strict.valid
    assert loose.valid


def test_ambiguous_flagged_and_optionally_rejected():
    """Two near-identical boxes mean we cannot tell which body is which."""
    a = [100.0, 100.0, 200.0, 300.0]
    b = [102.0, 101.0, 202.0, 301.0]
    flagged = associate_frame(0, TARGET, [a, b], [0.9, 0.9])
    assert flagged.valid and flagged.ambiguous

    rejected = associate_frame(
        0, TARGET, [a, b], [0.9, 0.9], AssociationThresholds(reject_ambiguous=True)
    )
    assert not rejected.valid
    assert rejected.invalid_reason == InvalidReason.AMBIGUOUS


def test_associate_sequence_uses_frame_indices():
    frames = [SimpleNamespace(frame_index=i, bbox=TARGET) for i in (10, 11, 12)]
    dets = {10: ([TARGET], [0.9]), 12: ([TARGET], [0.9])}  # 11 has no detections
    out = associate_sequence(frames, dets)
    assert [a.frame_index for a in out] == [10, 11, 12]
    assert [a.valid for a in out] == [True, False, True]
    assert out[1].invalid_reason == InvalidReason.NO_DETECTIONS


def test_longest_valid_run_stops_at_holes():
    def mk(idx, valid):
        return SimpleNamespace(frame_index=idx, valid=valid, ambiguous=False)

    assocs = [mk(i, True) for i in range(5)]
    assocs += [mk(5, False)]
    assocs += [mk(i, True) for i in range(6, 20)]
    assert longest_valid_run(assocs) == (6, 19)
    assert longest_valid_run([mk(0, False)]) is None
    assert longest_valid_run([]) is None


def test_summary_reports_unmatched_frames():
    frames = [SimpleNamespace(frame_index=i, bbox=TARGET) for i in range(4)]
    dets = {0: ([TARGET], [0.9]), 2: ([TARGET], [0.9]), 3: ([TARGET], [0.9])}
    stats = summarize(associate_sequence(frames, dets))
    assert stats["frames"] == 4
    assert stats["valid"] == 3
    assert stats["invalid"] == 1
    assert stats["unmatched_frames"] == [1]
    assert stats["invalid_reasons"] == {InvalidReason.NO_DETECTIONS: 1}
    assert stats["min_iou"] == pytest.approx(1.0)


# --- gap bridging ----------------------------------------------------------


def _seq(pattern, track_id=1, bbox=TARGET, drift=0.0):
    """Build a sequence from a valid/invalid pattern like 'VV..V'."""
    frames = []
    for i, ch in enumerate(pattern):
        box = [bbox[0] + drift * i, bbox[1], bbox[2] + drift * i, bbox[3]]
        tid = track_id[i] if isinstance(track_id, (list, tuple)) else track_id
        frames.append(SimpleNamespace(frame_index=i, bbox=box, track_id=tid))
    dets = {
        f.frame_index: ([f.bbox], [0.9]) for f, ch in zip(frames, pattern) if ch == "V"
    }
    return associate_sequence(frames, dets)


def test_short_gap_is_bridged_and_marked_interpolated():
    assocs = _seq("VVV..VVV")
    stats = bridge_gaps(assocs)

    assert stats["bridged_frames"] == [3, 4]
    for i in (3, 4):
        assert assocs[i].valid
        assert assocs[i].bridged
        assert assocs[i].interpolated
        assert assocs[i].invalid_reason is None
    # Bridged frames must be lower confidence than the observations around them.
    assert assocs[3].confidence < assocs[2].confidence
    assert longest_valid_run(assocs) == (0, 7)


def test_bridged_box_is_interpolated_between_neighbours():
    assocs = _seq("VV..VV", drift=10.0)
    bridge_gaps(assocs)
    x0 = [a.selected.bbox[0] for a in assocs]
    # Monotonic and strictly between the observed endpoints.
    assert x0[1] < x0[2] < x0[3] < x0[4]


def test_gap_longer_than_max_is_not_bridged():
    assocs = _seq("VV....VV")  # 4-frame gap
    stats = bridge_gaps(assocs)
    assert stats["bridged_frames"] == []
    assert stats["skipped_gaps"][0]["reason"] == "too_long"
    assert not any(a.bridged for a in assocs)


def test_max_bridge_gap_is_configurable_and_zero_disables():
    assert bridge_gaps(_seq("VV....VV"), AssociationThresholds(max_bridge_gap=4))[
        "bridged_frames"
    ] == [2, 3, 4, 5]
    assert bridge_gaps(_seq("VV.VV"), AssociationThresholds(max_bridge_gap=0))[
        "bridged_frames"
    ] == []


def test_identity_switch_is_never_bridged():
    """The gap is short, but the body on the far side is a different track."""
    assocs = _seq("VV..VV", track_id=[1, 1, 1, 1, 7, 7])
    stats = bridge_gaps(assocs)
    assert stats["bridged_frames"] == []
    assert stats["skipped_gaps"][0]["reason"] == "identity_switch"


def test_identity_switch_inside_the_gap_blocks_bridging():
    assocs = _seq("VV..VV", track_id=[1, 1, 9, 1, 1, 1])
    stats = bridge_gaps(assocs)
    assert stats["bridged_frames"] == []
    assert stats["skipped_gaps"][0]["reason"] == "identity_switch"


def test_teleport_across_gap_is_not_bridged():
    """Same track id, short gap, but the box jumps implausibly far."""
    frames = [SimpleNamespace(frame_index=i, bbox=TARGET, track_id=1) for i in range(4)]
    far = [900.0, 100.0, 1000.0, 300.0]
    dets = {0: ([TARGET], [0.9]), 3: ([far], [0.9])}
    assocs = associate_sequence(frames, dets)
    # Frame 3 associates against its own target box, so force the far selection.
    assocs[3] = associate_frame(3, far, [far], [0.9])
    assocs[3].track_id = 1
    stats = bridge_gaps(assocs)
    assert stats["bridged_frames"] == []
    assert stats["skipped_gaps"][0]["reason"] == "moved_too_far"


def test_unbounded_gaps_at_the_edges_are_not_bridged():
    leading = bridge_gaps(_seq("..VVV"))
    trailing = bridge_gaps(_seq("VVV.."))
    assert leading["bridged_frames"] == []
    assert trailing["bridged_frames"] == []
    assert leading["skipped_gaps"][0]["reason"] == "unbounded"
    assert trailing["skipped_gaps"][0]["reason"] == "unbounded"


def test_bridging_is_idempotent():
    assocs = _seq("VV.VV")
    first = bridge_gaps(assocs)["bridged_frames"]
    second = bridge_gaps(assocs)["bridged_frames"]
    assert first == [2]
    assert second == []


def test_summary_separates_observed_from_bridged():
    assocs = _seq("VVV..VVV")
    bridge_gaps(assocs)
    stats = summarize(assocs)
    assert stats["valid"] == 8
    assert stats["observed"] == 6
    assert stats["bridged"] == 2
    assert stats["bridged_frames"] == [3, 4]
    assert stats["invalid"] == 0
    # Confidence stats describe observations only, not interpolation.
    assert stats["mean_iou"] == pytest.approx(1.0)


def test_bridged_frames_survive_json_round_trip(tmp_path):
    assocs = _seq("VV.VV")
    bridge_gaps(assocs)
    path = save_associations(tmp_path / "a.json", assocs, AssociationThresholds())
    loaded, th, _ = load_associations(path)
    assert th.max_bridge_gap == 3
    assert loaded[2].bridged
    assert loaded[2].interpolated
    assert loaded[2].track_id == 1
    assert loaded[2].confidence == pytest.approx(assocs[2].confidence, abs=1e-3)


def test_association_json_round_trip(tmp_path):
    frames = [SimpleNamespace(frame_index=i, bbox=TARGET) for i in range(3)]
    dets = {0: ([TARGET, [400, 400, 500, 600]], [0.9, 0.7]), 1: ([TARGET], [0.9])}
    original = associate_sequence(frames, dets)
    th = AssociationThresholds(min_iou=0.4)
    path = save_associations(tmp_path / "association.json", original, th)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["valid"] == 2
    assert payload["thresholds"]["min_iou"] == 0.4

    loaded, loaded_th, summary = load_associations(path)
    assert loaded_th.min_iou == 0.4
    assert [a.frame_index for a in loaded] == [0, 1, 2]
    assert [a.valid for a in loaded] == [True, True, False]
    assert loaded[0].selected is not None
    assert len(loaded[0].candidates) == 2
    assert loaded[0].confidence == pytest.approx(original[0].confidence, abs=1e-3)
    assert summary["unmatched_frames"] == [2]
