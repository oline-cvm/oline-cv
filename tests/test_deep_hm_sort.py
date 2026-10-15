"""Deep HM-SORT primitives and generic tracker (synthetic data, no models)."""

from __future__ import annotations

import numpy as np
import pytest

from oline_cv.config import DeepHMSortConfig
from oline_cv.tracking.costs import (
    cosine_distance_matrix,
    gated_hm_cost,
    harmonic_mean_distance,
    linear_assignment,
)
from oline_cv.tracking.deep_hm_sort import DeepHMSort, TrackletState
from oline_cv.tracking.expansion_iou import (
    expand_boxes,
    expansion_iou,
    expansion_schedule,
    iou_matrix,
)


def unit(*v):
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)


def box(cx, cy, w=40, h=100):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# --- 1. harmonic mean ------------------------------------------------------
def test_harmonic_mean_basic_values():
    assert harmonic_mean_distance(0.5, 0.5) == pytest.approx(0.5)
    assert harmonic_mean_distance(0.2, 0.8) == pytest.approx(2 * 0.2 * 0.8 / 1.0)
    assert harmonic_mean_distance(1.0, 1.0) == pytest.approx(1.0)


def test_harmonic_mean_zero_and_bad_inputs_are_finite():
    out = harmonic_mean_distance([0.0, 0.0, 0.3, np.nan, np.inf], [0.0, 0.7, 0.0, 0.2, 0.2])
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[1] == 0.0 and out[2] == 0.0
    # NaN means no evidence: treated as distance 1.
    assert out[3] == pytest.approx(2 * 1.0 * 0.2 / 1.2)
    assert out[4] == pytest.approx(2 * 1.0 * 0.2 / 1.2)


def test_harmonic_mean_empty():
    assert harmonic_mean_distance(np.zeros((0, 3)), np.zeros((0, 3))).shape == (0, 3)


def test_harmonic_mean_leans_toward_smaller_term_but_stays_above_it():
    h = harmonic_mean_distance(0.1, 0.9)
    assert 0.1 < h < 0.5


# --- 2. cosine distance ----------------------------------------------------
def test_cosine_distance_identical_orthogonal_opposite():
    a = np.array([unit(1, 0, 0), unit(0, 1, 0), unit(-1, 0, 0)])
    d = cosine_distance_matrix(unit(1, 0, 0), a)[0]
    assert d[0] == pytest.approx(0.0, abs=1e-9)
    assert d[1] == pytest.approx(1.0)
    assert d[2] == pytest.approx(1.0)  # clipped, not 2


def test_cosine_distance_missing_embeddings():
    b = np.array([[np.nan, np.nan], [0.0, 0.0], [1.0, 0.0]])
    d = cosine_distance_matrix(np.array([[1.0, 0.0]]), b)[0]
    assert list(d) == [1.0, 1.0, 0.0]
    assert cosine_distance_matrix(np.zeros((0, 2)), b).shape == (0, 3)


def test_cosine_distance_dimension_mismatch_raises():
    with pytest.raises(ValueError):
        cosine_distance_matrix(np.ones((1, 3)), np.ones((1, 4)))


# --- 3. Expansion IoU ------------------------------------------------------
def test_expand_boxes_preserves_centre_and_scales():
    b = np.array([[10.0, 20.0, 30.0, 60.0]])
    e = expand_boxes(b, 0.5)[0]
    assert (e[0] + e[2]) / 2 == pytest.approx(20.0)
    assert (e[1] + e[3]) / 2 == pytest.approx(40.0)
    assert e[2] - e[0] == pytest.approx(20.0 * 2.0)
    assert e[3] - e[1] == pytest.approx(40.0 * 2.0)


def test_expand_boxes_zero_is_identity_and_negative_rejected():
    b = np.array([[1.0, 2.0, 3.0, 4.0]])
    assert np.allclose(expand_boxes(b, 0.0), b)
    with pytest.raises(ValueError):
        expand_boxes(b, -0.1)


def test_expand_boxes_clips_at_image_edge():
    e = expand_boxes([[0.0, 0.0, 20.0, 20.0]], 1.0, image_wh=(100, 50))[0]
    assert e.min() >= 0 and e[2] <= 100 and e[3] <= 50


def test_expansion_iou_recovers_non_overlapping_fast_motion():
    a, b = [box(100, 100)], [box(150, 100)]
    assert iou_matrix(a, b)[0, 0] == 0.0
    assert expansion_iou(a, b, 0.3)[0, 0] > 0.0
    assert expansion_iou(a, b, 0.6)[0, 0] > expansion_iou(a, b, 0.3)[0, 0]


def test_expansion_iou_deterministic_and_empty():
    a = np.random.default_rng(0).uniform(0, 100, (5, 2))
    boxes = np.hstack([a, a + 20])
    assert np.array_equal(expansion_iou(boxes, boxes, 0.3), expansion_iou(boxes, boxes, 0.3))
    assert expansion_iou(np.zeros((0, 4)), boxes, 0.3).shape == (0, 5)


def test_expansion_schedule():
    assert expansion_schedule(0.3, 0.3, 2) == pytest.approx([0.3, 0.6])
    with pytest.raises(ValueError):
        expansion_schedule(0.3, 0.3, 0)


# --- 4. assignment ---------------------------------------------------------
def test_hungarian_one_to_one():
    cost = np.array([[0.1, 0.2], [0.15, 0.9]])
    m, ur, uc = linear_assignment(cost, 0.8)
    assert sorted(m) == [(0, 1), (1, 0)]  # global optimum, not greedy
    assert ur == [] and uc == []


def test_assignment_threshold_and_empty():
    m, ur, uc = linear_assignment(np.array([[0.95]]), 0.8)
    assert m == [] and ur == [0] and uc == [0]
    m, ur, uc = linear_assignment(np.zeros((0, 3)), 0.8)
    assert m == [] and ur == [] and uc == [0, 1, 2]


def test_gated_cost_appearance_ignored_far_away():
    # Same look but far apart: proximity gate keeps appearance from matching.
    c = gated_hm_cost(np.array([0.95]), np.array([0.0]), appearance_threshold=0.3,
                      proximity_threshold=0.5)
    assert c[0] == pytest.approx(harmonic_mean_distance(0.95, 1.0))


# --- 5. generic tracker ----------------------------------------------------
def _cfg(**kw):
    return DeepHMSortConfig(**kw)


def test_tracker_ids_stable_and_monotonic():
    tr = DeepHMSort(_cfg(), fps=30)
    e = np.array([unit(1, 0, 0), unit(0, 1, 0)])
    ids = []
    for f in range(5):
        r = tr.update(f, [box(100 + 2 * f, 100), box(300 - 2 * f, 100)], [0.9, 0.9], e)
        ids.append(tuple(r.track_ids))
    assert len(set(ids)) == 1 and ids[0] == (1, 2)


def test_tracker_crossing_players_keep_ids_by_appearance():
    tr = DeepHMSort(_cfg(), fps=30)
    ea, eb = unit(1, 0, 0), unit(0, 1, 0)
    xs_a = np.linspace(100, 200, 11)
    xs_b = np.linspace(200, 100, 11)
    first = None
    for f, (xa, xb) in enumerate(zip(xs_a, xs_b)):
        r = tr.update(f, [box(xa, 100), box(xb, 100)], [0.9, 0.9], np.array([ea, eb]))
        if first is None:
            first = tuple(r.track_ids)
        if abs(xa - xb) > 1:
            assert tuple(r.track_ids) == first, f"swap at frame {f}"


def test_tracker_keeps_lost_tracklets_and_reactivates():
    cfg = _cfg(historical_after_s=0.1)
    tr = DeepHMSort(cfg, fps=30)
    ea = unit(1, 0, 0)
    r = tr.update(0, [box(100, 100)], [0.9], ea[None])
    tid = int(r.track_ids[0])
    for f in range(1, 120):  # 4 s gone
        tr.update(f, [], [], None)
    assert tr.tracks[tid].state == TrackletState.HISTORICAL
    assert tr.visible() == []
    r = tr.update(120, [box(110, 100)], [0.9], ea[None])
    assert int(r.track_ids[0]) == tid
    assert r.observations[0].reactivated
    assert tr.tracks[tid].state == TrackletState.REACTIVATED


def test_tracker_never_reuses_ids():
    tr = DeepHMSort(_cfg(), fps=30)
    seen = set()
    rng = np.random.default_rng(1)
    for f in range(30):
        n = rng.integers(0, 4)
        xs = rng.uniform(0, 2000, n)
        emb = rng.normal(size=(n, 8))
        r = tr.update(f, [box(x, 200) for x in xs], [0.9] * n, emb)
        for o in r.observations:
            if o.stage == "new":
                assert o.track_id not in seen
                seen.add(o.track_id)


def test_tracker_low_confidence_second_stage_and_threshold():
    tr = DeepHMSort(_cfg(), fps=30)
    e = unit(1, 0)[None]
    r = tr.update(0, [box(100, 100)], [0.9], e)
    tid = int(r.track_ids[0])
    r = tr.update(1, [box(101, 100)], [0.5], e)  # low band: IoU-only stage
    assert int(r.track_ids[0]) == tid and r.observations[0].stage == "low"
    r = tr.update(2, [box(101, 100)], [0.3], e)  # below detector threshold
    assert len(r) == 0 and tr.tracks[tid].state == TrackletState.LOST


def test_tracker_empty_frames_and_missing_embeddings():
    tr = DeepHMSort(_cfg(), fps=30)
    r = tr.update(0, np.zeros((0, 4)), np.zeros(0), None)
    assert len(r) == 0
    r = tr.update(1, [box(100, 100)], [0.9], None)
    assert len(r) == 1
    r = tr.update(2, [box(102, 100)], [0.9], np.full((1, 4), np.nan))
    assert int(r.track_ids[0]) == int(tr.visible()[0].track_id)


def test_tracker_does_not_refresh_appearance_from_overlapping_box():
    tr = DeepHMSort(_cfg(), fps=30)
    ea, eb = unit(1, 0, 0), unit(0, 1, 0)
    tr.update(0, [box(100, 100), box(400, 100)], [0.9, 0.9], np.array([ea, eb]))
    tid = 1
    before = tr.tracks[tid].feature.copy()
    # Player B steps into A's box; A's crop is now half B.
    r = tr.update(1, [box(100, 100), box(110, 100)], [0.9, 0.9],
                  np.array([unit(1, 1, 0), eb]))
    obs = next(o for o in r.observations if o.track_id == tid)
    assert not obs.appearance_updated
    assert np.allclose(tr.tracks[tid].feature, before)


def test_tracker_feature_ema_is_normalised():
    tr = DeepHMSort(_cfg(appearance_ema_alpha=0.9), fps=30)
    tr.update(0, [box(100, 100)], [0.9], unit(1, 0)[None])
    tr.update(1, [box(101, 100)], [0.9], unit(0.9, 0.3)[None])
    f = tr.tracks[1].feature
    assert np.linalg.norm(f) == pytest.approx(1.0)
    expect = 0.9 * unit(1, 0) + 0.1 * unit(0.9, 0.3)
    assert np.allclose(f, expect / np.linalg.norm(expect))


def test_warp_boxes_translation_zoom_and_inverse():
    from oline_cv.tracking.camera_motion import invert, warp_boxes

    b = np.array([[100.0, 100.0, 140.0, 200.0]])
    shift = np.array([[1.0, 0.0, 50.0], [0.0, 1.0, -10.0]])
    assert np.allclose(warp_boxes(b, shift), [[150, 90, 190, 190]])
    zoom = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    z = warp_boxes(b, zoom)[0]
    assert z[2] - z[0] == pytest.approx(80) and (z[0] + z[2]) / 2 == pytest.approx(240)
    assert np.allclose(warp_boxes(warp_boxes(b, zoom), invert(zoom)), b)
    assert np.allclose(warp_boxes(b, None), b)


def test_camera_motion_estimator_recovers_pan():
    from oline_cv.tracking.camera_motion import CameraMotionEstimator

    rng = np.random.default_rng(3)
    base = (rng.uniform(0, 255, (400, 1200)) > 200).astype(np.uint8) * 255
    import cv2

    base = cv2.GaussianBlur(base, (5, 5), 0)
    img0 = np.dstack([base] * 3)
    img1 = np.roll(img0, 24, axis=1)  # camera pans: scene moves +24 px
    est = CameraMotionEstimator(work_width=1200)
    est.estimate(img0, np.zeros((0, 4)))
    m = est.estimate(img1, np.zeros((0, 4)))
    assert m[0, 2] == pytest.approx(24, abs=1.5)
    assert m[1, 2] == pytest.approx(0, abs=1.5)


def test_tracker_keeps_id_through_camera_pan():
    pan = np.array([[1.0, 0.0, 80.0], [0.0, 1.0, 0.0]])  # 2 box widths per frame
    ea = unit(1, 0)
    without = DeepHMSort(_cfg(), fps=30)
    with_cmc = DeepHMSort(_cfg(), fps=30)
    for tr, cam in ((without, None), (with_cmc, pan)):
        tr.update(0, [box(100, 100)], [0.9], ea[None])
        for f in range(1, 6):
            tr.update(f, [box(100 + 80 * f, 100)], [0.9], ea[None], camera=None if f == 0 else cam)
    assert {t.track_id for t in with_cmc.visible()} == {1}
    assert 1 not in {t.track_id for t in without.visible()}


def test_tracker_tentative_needs_confirmation():
    tr = DeepHMSort(_cfg(), fps=30)
    tr.update(0, [], [], None)
    r = tr.update(1, [box(100, 100)], [0.9], unit(1, 0)[None])
    assert not r.confirmed[0] and tr.visible() == []
    r = tr.update(2, [box(101, 100)], [0.9], unit(1, 0)[None])
    assert r.confirmed[0] and len(tr.visible()) == 1
