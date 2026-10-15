"""Locked-target identity layer (OLINE) on synthetic tracker output."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from oline_cv.config import AnalysisConfig, DeepHMSortConfig
from oline_cv.tracking.deep_hm_sort import DeepHMSort, FrameResult
from oline_cv.tracking.target_identity import (
    LockedTargetIdentity,
    TargetState,
    run_target_identity,
)

W, H = 1920, 1080


def unit(*v):
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)


EA = unit(1, 0, 0, 0)  # target
EB = unit(0, 1, 0, 0)  # different player
EA_LOOKALIKE = unit(1, 0.25, 0, 0)  # same uniform, close to EA (d ~ 0.03)


def box(cx, cy=500, w=60, h=150):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def fr(f, dets, conf_all=True):
    """FrameResult from [(box, emb, track_id, score), ...]."""
    n = len(dets)
    return FrameResult(
        frame_idx=f,
        boxes=np.array([d[0] for d in dets], dtype=float).reshape(n, 4),
        scores=np.array([d[3] if len(d) > 3 else 0.9 for d in dets], dtype=float),
        embeddings=np.array([d[1] for d in dets], dtype=float).reshape(n, -1) if n else np.zeros((0, 4)),
        track_ids=np.array([d[2] for d in dets], dtype=int),
        confirmed=np.array([conf_all and d[2] >= 0 for d in dets], dtype=bool),
        source_index=np.arange(n),
    )


def run(frames, lock_pos=0, lock_det=0, **cfg):
    out, meta = run_target_identity(frames, lock_pos, lock_det, DeepHMSortConfig(**cfg), 30.0, (W, H))
    return out, meta


def boxed_track_ids(out):
    return [t.track_id for t in out if t is not None and t.has_box]


# --- occlusion by a crossing player ----------------------------------------
def test_occluder_never_inherits_target():
    frames = []
    for f in range(10):  # A at 500, B at 800 approaching
        frames.append(fr(f, [(box(500), EA, 1), (box(800 - 20 * f), EB, 2)]))
    for f in range(10, 30):  # B in front of A; A not detected
        frames.append(fr(f, [(box(505), EB, 2)]))
    for f in range(30, 40):  # B moves on, A visible again
        frames.append(fr(f, [(box(505), EA, 1), (box(505 - 15 * (f - 29)), EB, 2)]))
    out, _ = run(frames)
    assert all(t.track_id != 2 for t in out if t.has_box)
    assert all(not out[f].has_box for f in range(10, 30))
    assert out[39].has_box and out[39].track_id == 1


def test_reacquire_with_new_tracker_id_keeps_logical_identity():
    frames = [fr(f, [(box(500), EA, 1), (box(900), EB, 2)]) for f in range(10)]
    frames += [fr(f, [(box(900), EB, 2)]) for f in range(10, 40)]
    # Tracker failed to re-link: A comes back as tracklet 7.
    frames += [fr(f, [(box(520), EA, 7), (box(900), EB, 2)]) for f in range(40, 50)]
    out, meta = run(frames)
    assert out[40].state == TargetState.REACQUISITION_CANDIDATE and not out[40].has_box
    assert out[41].state == TargetState.REACQUIRED and out[41].track_id == 7
    assert out[45].state == TargetState.VISIBLE_CONFIDENT
    assert meta["reacquisitions"] == 1
    assert set(boxed_track_ids(out)) == {1, 7}


# --- nearby player cannot steal a visible target ----------------------------
def test_visible_target_protection_against_adjacent_lookalike():
    frames = [fr(f, [(box(500), EA, 1)]) for f in range(5)]
    # A look-alike steps right next to A (overlapping) with nearly identical
    # appearance; the tracker keeps both ids.
    frames += [fr(f, [(box(500), EA, 1), (box(520), EA_LOOKALIKE, 3)]) for f in range(5, 25)]
    out, _ = run(frames)
    assert 3 not in boxed_track_ids(out)
    for t in out[5:]:
        assert t.state in (TargetState.VISIBLE_CONFIDENT, TargetState.VISIBLE_AMBIGUOUS)


def test_bound_tracklet_that_stops_looking_like_target_is_dropped():
    frames = [fr(f, [(box(500), EA, 1)]) for f in range(5)]
    # Tracker swap: tracklet 1 now sits on a different-looking player.
    frames += [fr(f, [(box(505), EB, 1)]) for f in range(5, 15)]
    out, _ = run(frames, target_identity_fail_frames=3)
    assert all(not t.has_box for t in out[5:])
    assert out[5].state == TargetState.VISIBLE_AMBIGUOUS
    assert out[10].state in (TargetState.OCCLUDED, TargetState.LOST)


# --- similar uniforms / ambiguity ------------------------------------------
def test_ambiguous_reacquisition_stays_lost():
    frames = [fr(f, [(box(500), EA, 1)]) for f in range(5)]
    frames += [fr(f, []) for f in range(5, 20)]
    # Two equally plausible candidates either side of the last box.
    frames += [fr(f, [(box(470), EA, 5), (box(530), EA_LOOKALIKE, 6)]) for f in range(20, 35)]
    out, _ = run(frames)
    assert not any(t.has_box for t in out[5:])
    assert out[25].state in (TargetState.OCCLUDED, TargetState.LOST)


def test_covisible_player_excluded_from_reacquisition():
    # B (same uniform) is seen alongside A, so B is known to be someone else.
    frames = [fr(f, [(box(500), EA, 1), (box(600), EA_LOOKALIKE, 2)]) for f in range(10)]
    frames += [fr(f, [(box(600 - 8 * (f - 9)), EA_LOOKALIKE, 2)]) for f in range(10, 25)]
    out, _ = run(frames)
    assert 2 not in boxed_track_ids(out)


def test_reacquisition_outside_spatial_region_rejected():
    frames = [fr(f, [(box(300), EA, 1)]) for f in range(5)]
    frames += [fr(f, []) for f in range(5, 10)]
    # Identical appearance but far across the field after 5 frames.
    frames += [fr(f, [(box(1500), EA, 9)]) for f in range(10, 20)]
    out, _ = run(frames)
    assert not any(t.has_box for t in out[5:])


def test_reacquisition_after_long_absence():
    frames = [fr(f, [(box(500), EA, 1), (box(1200), EB, 2)]) for f in range(10)]
    frames += [fr(f, [(box(1200), EB, 2)]) for f in range(10, 100)]  # 3 s gone
    frames += [fr(f, [(box(560), EA, 1), (box(1200), EB, 2)]) for f in range(100, 110)]
    out, _ = run(frames)
    assert out[50].state == TargetState.LOST
    # Tracker re-found its own tracklet -> immediate reacquisition.
    assert out[100].state == TargetState.REACQUIRED and out[100].track_id == 1
    assert out[105].state == TargetState.VISIBLE_CONFIDENT


def test_harmonic_mean_cannot_carry_barely_overlapping_box():
    # Edge of the expanded region (d_iou ~ 0.97) with perfect appearance: the
    # HM cost alone would be low; the separate spatial gate rejects it.
    frames = [fr(f, [(box(500), EA, 1)]) for f in range(3)]
    frames += [fr(f, []) for f in range(3, 6)]
    frames += [fr(f, [(box(578), EA, 4)]) for f in range(6, 12)]
    out, _ = run(frames, target_adaptive_appearance=False)
    assert not any(t.has_box for t in out[3:])
    assert out[6].d_iou is None or out[6].candidates[0]["d_iou"] > 0.7


def test_adaptive_appearance_gate_from_covisible_players():
    # Teammates seen alongside the target sit at d ~ 0.03-0.05 from him; a
    # stranger at d ~ 0.06 would pass the static 0.3 gate but not the
    # adaptive one.
    mates = [unit(1, 0.25 + 0.01 * k, 0, 0) for k in range(4)]
    stranger = unit(1, 0.36, 0, 0)
    frames = [
        fr(f, [(box(500), EA, 1)] + [(box(800 + 150 * k), m, 2 + k) for k, m in enumerate(mates)])
        for f in range(10)
    ]
    frames += [fr(f, []) for f in range(10, 14)]
    frames += [fr(f, [(box(505), stranger, 20)]) for f in range(14, 22)]
    adaptive, meta = run(frames)
    static, _ = run(frames, target_adaptive_appearance=False)
    assert meta["forward"]["reacquire_appearance_threshold"] < 0.06
    assert not any(t.has_box for t in adaptive[10:])
    assert any(t.has_box and t.track_id == 20 for t in static[14:])


def test_own_tracklet_after_short_occlusion_uses_static_gate_strangers_do_not():
    mates = [unit(1, 0.25 + 0.01 * k, 0, 0) for k in range(4)]
    drifted = unit(1, 0.5, 0, 0)  # d ~ 0.11 from EA: contact pose / blur
    base = [
        fr(f, [(box(500), EA, 1)] + [(box(800 + 150 * k), m, 2 + k) for k, m in enumerate(mates)])
        for f in range(10)
    ]
    gap = [fr(f, []) for f in range(10, 14)]
    own = base + gap + [fr(f, [(box(510), drifted, 1)]) for f in range(14, 20)]
    stranger = base + gap + [fr(f, [(box(510), drifted, 30)]) for f in range(14, 20)]
    out_own, _ = run(own)
    out_str, _ = run(stranger)
    assert out_own[14].state == TargetState.REACQUIRED and out_own[14].reason == "own_tracklet_refound"
    assert not any(t.has_box for t in out_str[10:])


def test_target_follows_camera_pan_in_both_directions():
    pan = np.array([[1.0, 0.0, 70.0], [0.0, 1.0, 0.0]])
    frames = []
    for f in range(12):
        r = fr(f, [(box(300 + 70 * f), EA, 1 if f < 6 else -1)])
        r.camera = pan if f > 0 else None
        frames.append(r)
    # Detector misses frame 6; target returns without a tracker id.
    frames[6] = fr(6, [])
    frames[6].camera = pan
    out, _ = run(frames, lock_pos=3)
    assert all(out[f].has_box for f in range(0, 6))
    assert out[6].state == TargetState.OCCLUDED
    assert out[8].has_box and out[11].has_box


def test_detector_dropout_single_frame():
    frames = [fr(f, [(box(500 + f), EA, 1)]) for f in range(5)]
    frames.append(fr(5, []))
    frames += [fr(f, [(box(500 + f), EA, 1)]) for f in range(6, 10)]
    out, _ = run(frames)
    assert out[5].state == TargetState.OCCLUDED and not out[5].has_box
    assert out[6].state == TargetState.REACQUIRED
    assert all(out[f].state == TargetState.VISIBLE_CONFIDENT for f in range(7, 10))


def test_all_players_disappear():
    frames = [fr(0, [(box(500), EA, 1)])] + [fr(f, []) for f in range(1, 60)]
    out, _ = run(frames)
    assert out[10].state == TargetState.OCCLUDED
    assert out[59].state == TargetState.LOST
    assert not any(t.has_box for t in out[1:])


def test_missing_embedding_never_reacquires():
    frames = [fr(f, [(box(500), EA, 1)]) for f in range(3)]
    frames += [fr(f, []) for f in range(3, 8)]
    frames += [fr(f, [(box(500), np.full(4, np.nan), 4)]) for f in range(8, 15)]
    out, _ = run(frames)
    assert not any(t.has_box for t in out[3:])


# --- clean gallery ---------------------------------------------------------
def test_gallery_skips_overlap_low_score_edge_and_small():
    cfg = DeepHMSortConfig(target_gallery_min_interval_s=0.0)
    lt = LockedTargetIdentity(cfg, 30.0, (W, H))
    lt.initialize(fr(0, [(box(500), EA, 1)]), 0)
    lt.step(fr(1, [(box(500), EA, 1), (box(515), EB, 2)]))  # overlap
    lt.step(fr(2, [(box(500), EA, 1, 0.45)]))  # low score
    assert len(lt.gallery) == 0
    assert {"overlap", "low_score"} <= set(lt.gallery_skips)
    lt_edge = LockedTargetIdentity(cfg, 30.0, (W, H))
    lt_edge.initialize(fr(0, [(box(40), EA, 1)]), 0)
    lt_edge.step(fr(1, [([1, 425, 61, 575], EA, 1)]))  # touches frame edge
    assert len(lt_edge.gallery) == 0 and lt_edge.gallery_skips.get("frame_edge") == 1
    lt2 = LockedTargetIdentity(cfg, 30.0, (W, H))
    lt2.initialize(fr(0, [(box(500, h=20, w=10), EA, 1)]), 0)
    lt2.step(fr(1, [(box(500, h=20, w=10), EA, 1)]))
    assert lt2.gallery_skips.get("small_crop") == 1


def test_gallery_rejects_contaminated_embedding_and_is_bounded():
    cfg = DeepHMSortConfig(target_gallery_min_interval_s=0.0, locked_target_gallery_size=5)
    lt = LockedTargetIdentity(cfg, 30.0, (W, H))
    lt.initialize(fr(0, [(box(500), EA, 1)]), 0)
    for f in range(1, 40):
        lt.step(fr(f, [(box(500), unit(1, 0.05 * (f % 3), 0, 0), 1)]))
    assert len(lt.gallery_matrix()) == 5
    np.testing.assert_allclose(lt.gallery_matrix()[0], EA)  # anchor kept
    # A crop that is half someone else: appearance jump, not added.
    n_before = len(lt.gallery)
    snapshot = lt.gallery_matrix().copy()
    lt.step(fr(40, [(box(500), unit(1, 1.3, 0, 0), 1)]))
    assert lt.gallery_skips.get("appearance_jump", 0) >= 1 or lt.state != TargetState.VISIBLE_CONFIDENT
    assert len(lt.gallery) == n_before
    np.testing.assert_allclose(lt.gallery_matrix(), snapshot)


def test_gallery_respects_min_interval():
    cfg = DeepHMSortConfig(target_gallery_min_interval_s=0.2)  # 6 frames @30
    lt = LockedTargetIdentity(cfg, 30.0, (W, H))
    lt.initialize(fr(0, [(box(500), EA, 1)]), 0)
    for f in range(1, 13):
        lt.step(fr(f, [(box(500), EA, 1)]))
    assert len(lt.gallery) == 2  # frames 1 and 7


# --- bidirectional / through the real tracker -------------------------------
def test_bidirectional_resolves_frames_before_lock():
    frames = [fr(f, [(box(500), EA, 1), (box(900), EB, 2)]) for f in range(20)]
    out, meta = run(frames, lock_pos=10)
    assert all(t.has_box and t.track_id == 1 for t in out)
    assert "backward" in meta


def test_end_to_end_with_tracker_crossing_occlusion():
    cfg = DeepHMSortConfig()
    tr = DeepHMSort(cfg, 30.0, (W, H))
    results = []
    for f in range(60):
        dets, embs = [], []
        bx = 900 - 15 * f  # B walks left across A at x=500
        a_hidden = abs(bx - 500) < 40
        if not a_hidden:
            dets.append(box(500))
            embs.append(EA)
        dets.append(box(bx))
        embs.append(EB)
        results.append(tr.update(f, dets, [0.9] * len(dets), np.array(embs)))
    out, _ = run_target_identity(results, 0, 0, cfg, 30.0, (W, H))
    lock_tid = int(results[0].track_ids[0])
    b_tids = {int(r.track_ids[i]) for r in results for i in range(len(r))
              if abs(r.boxes[i][0] + 30 - (900 - 15 * r.frame_idx)) < 1}
    assert lock_tid not in b_tids
    assert not (set(boxed_track_ids(out)) & b_tids)
    assert out[59].has_box


# --- evaluation metrics ---------------------------------------------------
def test_target_metrics_counts_wrong_person_and_reacquisition():
    from oline_cv.tracking.evaluation import target_metrics

    g = np.array(box(500))
    other = np.array(box(800))
    gt = {0: g, 1: g, 2: None, 3: None, 4: g, 5: g, 6: g}
    pred = [g, g, None, other, None, g, other]
    m = target_metrics(pred, gt)
    assert m["correct_frames"] == 3
    assert m["wrong_person_frames"] == 2  # box while hidden, and box on someone else
    assert m["missed_frames"] == 1
    assert m["identity_switches"] == 1
    assert m["occlusions"][0]["wrong_during"] == 1
    assert m["occlusions"][0]["reacquire_latency_frames"] == 1


# --- downstream contract ----------------------------------------------------
def _fake_tracker(cfg: AnalysisConfig):
    from oline_cv.pose_tracker import PoseTracker

    tr = PoseTracker.__new__(PoseTracker)
    tr.config = cfg
    tr._dl_center = None

    def pose_in_box(frame, b):
        x0, y0, x1, y1 = b
        xs = np.linspace(x0 + 5, x1 - 5, 17)
        ys = np.linspace(y0 + 5, y1 - 5, 17)
        return np.stack([xs, ys], axis=1), np.full(17, 0.9)

    tr._pose_in_box = pose_in_box
    return tr


def test_target_frames_to_poses_and_track_export(tmp_path):
    from oline_cv.motion3d.track_export import export_tracks
    from oline_cv.series import build_series

    cfg = AnalysisConfig()
    frames_r = [fr(f, [(box(500 + f), EA, 1), (box(900), EB, 2)]) for f in range(12)]
    frames_r[5] = fr(5, [(box(900), EB, 2)])  # dropout
    out, _ = run_target_identity(frames_r, 0, 0, cfg.deep_hm, 30.0, (W, H))
    tr = _fake_tracker(cfg)
    imgs = [np.zeros((H, W, 3), dtype=np.uint8) for _ in frames_r]
    ol, dl = tr._poses_from_targets(imgs, frames_r, out, 30.0)
    assert len(ol) == len(dl) == 12
    assert ol[0].track_state == "TRACKED" and ol[0].target_state == "VISIBLE_CONFIDENT"
    assert ol[5].track_state == "LOST" and ol[5].interpolated and not ol[5].usable
    assert ol[6].track_state == "REIDENTIFIED"
    assert all(p.target_id == 1 for p in ol)
    assert ol[0].association is not None and "cost" in ol[0].association
    assert dl[0] is not None and dl[0].track_id == 2

    manifest = export_tracks("clip.mp4", ol, imgs, 30.0, W, H, tmp_path,
                             save_full_frames=False, ol_lock={"tracker": "deep_hm_sort"})
    assert (tmp_path / "tracks.json").exists()
    assert len(manifest.frames) == 12
    series = build_series(ol, 0, 11, 150.0, 30.0, cfg)
    assert series is not None


def test_backend_config_defaults_and_legacy_switch():
    cfg = AnalysisConfig()
    assert cfg.tracker_backend == "deep_hm_sort"
    d = cfg.deep_hm
    assert (d.detector_threshold, d.high_confidence_threshold,
            d.assignment_cost_threshold, d.appearance_cost_threshold) == (0.4, 0.6, 0.8, 0.3)
    assert (d.expansion_initial, d.expansion_step) == (0.3, 0.3)
    assert "deep_hm" in cfg.to_dict()
    assert AnalysisConfig(tracker_backend="legacy_botsort").tracker_backend == "legacy_botsort"


def test_import_has_no_heavy_side_effects_and_reid_fails_clearly(tmp_path):
    import importlib

    import oline_cv.tracking as t

    importlib.reload(t)
    assert "torchreid" not in sys.modules
    from oline_cv.tracking.reid import ReIDUnavailableError, build_reid_extractor

    cfg = DeepHMSortConfig(reid_backend="osnet", reid_weights=str(tmp_path / "missing.pth"))
    with pytest.raises(ReIDUnavailableError, match="never downloaded"):
        build_reid_extractor(cfg, "yolov8m.pt")
    with pytest.raises(ReIDUnavailableError):
        build_reid_extractor(DeepHMSortConfig(reid_backend="osnet"), "yolov8m.pt")
    fb = build_reid_extractor(DeepHMSortConfig(reid_backend="yolo_embed"), "yolov8m.pt")
    assert fb.describe()["quality"] == "generic" and fb._model is None  # lazy
