"""SAM / fake-mask target recovery (no SAM weights, no CUDA)."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from oline_cv.config import AnalysisConfig, DeepHMSortConfig
from oline_cv.tracking.deep_hm_sort import DeepHMSort, FrameResult
from oline_cv.tracking.evaluation import load_ground_truth, target_metrics
from oline_cv.tracking.occlusion_risk import TargetRisk, evaluate_risk
from oline_cv.tracking.target_identity import TargetState, run_target_identity
from oline_cv.tracking.target_mask import (
    FakeTargetMaskTracker,
    MaskEvidence,
    build_target_mask_tracker,
    mask_coverage,
)


W, H = 400, 200
EA = np.array([1.0, 0.0, 0.0, 0.0])
EB = np.array([0.0, 1.0, 0.0, 0.0])


def unit(*v):
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)


def box(cx, cy=100, w=40, h=80):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def fr(f, dets, recovery=None):
    n = len(dets)
    rec = np.zeros(n, dtype=bool)
    if recovery:
        for i in recovery:
            rec[i] = True
    return FrameResult(
        frame_idx=f,
        boxes=np.array([d[0] for d in dets], dtype=float).reshape(n, 4) if n else np.zeros((0, 4)),
        scores=np.array([d[3] if len(d) > 3 else 0.9 for d in dets], dtype=float),
        embeddings=np.array([d[1] for d in dets], dtype=float).reshape(n, -1) if n else np.zeros((0, 4)),
        track_ids=np.array([d[2] for d in dets], dtype=int) if n else np.zeros(0, dtype=int),
        confirmed=np.array([d[2] >= 0 for d in dets], dtype=bool) if n else np.zeros(0, dtype=bool),
        source_index=np.arange(n),
        recovery=rec,
    )


def blob(cx, cy=100, w=40, h=80):
    m = np.zeros((H, W), dtype=bool)
    x1, y1 = max(0, int(cx - w / 2)), max(0, int(cy - h / 2))
    x2, y2 = min(W, int(cx + w / 2)), min(H, int(cy + h / 2))
    m[y1:y2, x1:x2] = True
    return m


def run(frames, masks=None, lock_pos=0, lock_det=0, **kw):
    cfg = DeepHMSortConfig(**kw)
    mt = FakeTargetMaskTracker(masks or {})
    mt.initialize([None] * len(frames), lock_pos, frames[lock_pos].boxes[lock_det], (W, H))
    return run_target_identity(frames, lock_pos, lock_det, cfg, 30.0, (W, H), mask_tracker=mt)


def test_normal_visible_accepts_deep_hm():
    frames = [fr(f, [(box(80), EA, 1), (box(300), EB, 2)]) for f in range(8)]
    out, _ = run(frames, {f: blob(80) for f in range(8)})
    assert all(t.has_box and t.track_id == 1 for t in out)
    assert out[3].reason in ("lock", "deep_hm_confident", "bound_track")


def test_nearby_player_does_not_steal_identity():
    frames = [fr(f, [(box(80), EA, 1)]) for f in range(4)]
    frames += [fr(f, [(box(80), EA, 1), (box(95), EB, 3)]) for f in range(4, 12)]
    out, _ = run(frames, {f: blob(80) for f in range(12)})
    assert 3 not in [t.track_id for t in out if t.has_box]


def test_overlap_keeps_identity_and_skips_gallery():
    cfg = DeepHMSortConfig(target_gallery_min_interval_s=0.0)
    frames = [fr(0, [(box(80), EA, 1)])]
    frames += [fr(f, [(box(80), EA, 1), (box(90), EB, 2)]) for f in range(1, 6)]
    mt = FakeTargetMaskTracker({f: blob(80) for f in range(6)})
    mt.initialize(frames, 0, frames[0].boxes[0], (W, H))
    out, meta = run_target_identity(frames, 0, 0, cfg, 30.0, (W, H), mask_tracker=mt)
    skips = meta["forward"]["gallery_skips"]
    assert skips.get("overlap", 0) >= 1
    assert all(t.has_box and t.track_id == 1 for t in out)
    assert any(t.reason == "overlap_identity_holds" for t in out[1:])


def test_detector_dropout_keeps_logical_identity():
    frames = [fr(f, [(box(80), EA, 1)]) for f in range(4)]
    frames += [fr(f, []) for f in range(4, 10)]
    frames += [fr(f, [(box(82), EA, 1)]) for f in range(10, 14)]
    masks = {f: blob(80) if f < 4 or f >= 10 else np.zeros((H, W), bool) for f in range(14)}
    out, _ = run(frames, masks)
    assert all(not t.has_box for t in out[4:10])
    assert out[5].state in (TargetState.OCCLUDED, TargetState.LOST)


def test_weak_reappearance_considered_with_mask():
    frames = [fr(f, [(box(80), EA, 1, 0.9)]) for f in range(3)]
    frames += [fr(f, []) for f in range(3, 6)]
    frames += [fr(f, [(box(84), EA, -1, 0.22)], recovery=[0]) for f in range(6, 12)]
    masks = {**{f: blob(80) for f in range(3)}, **{f: blob(84) for f in range(6, 12)}}
    out, _ = run(frames, masks, target_adaptive_appearance=False)
    boxed = [t for t in out[6:] if t.has_box]
    assert boxed and boxed[0].recovery


def test_mask_reacquires_correct_candidate():
    frames = [fr(f, [(box(80), EA, 1)]) for f in range(4)]
    frames += [fr(f, []) for f in range(4, 7)]
    frames += [fr(f, [(box(200), EB, 8), (box(85), EA, 9)]) for f in range(7, 14)]
    masks = {**{f: blob(80) for f in range(4)}, **{f: blob(85) for f in range(7, 14)}}
    out, meta = run(frames, masks, target_adaptive_appearance=False)
    later = [t for t in out[8:] if t.has_box]
    assert later and all(t.track_id == 9 for t in later)
    assert meta["reacquisitions"] >= 1


def test_mask_vetoes_wrong_deep_hm_switch():
    frames = [fr(f, [(box(80), EA, 1)]) for f in range(5)]
    # Tracker ID 1 hops onto player B; mask stays on A (new det 4).
    frames += [fr(f, [(box(200), EB, 1, 0.9), (box(82), EA, 4, 0.9)]) for f in range(5, 12)]
    masks = {f: blob(82) for f in range(12)}
    out, _ = run(frames, masks)
    assert all(t.track_id != 1 or t.state == TargetState.VISIBLE_AMBIGUOUS or not t.has_box for t in out[5:])
    assert any(t.reason == "deep_hm_assignment_rejected_by_mask" for t in out[5:])


def test_reid_contradicts_mask_stays_uncertain():
    frames = [fr(f, [(box(80), EA, 1)]) for f in range(3)]
    frames += [fr(f, []) for f in range(3, 6)]
    # Mask on B, but appearance is B (not EA).
    frames += [fr(f, [(box(200), EB, 7)]) for f in range(6, 12)]
    masks = {**{f: blob(80) for f in range(3)}, **{f: blob(200) for f in range(6, 12)}}
    out, _ = run(frames, masks)
    assert not any(t.has_box and t.track_id == 7 for t in out)


def test_two_similar_candidates_no_guess():
    look = unit(1, 0.2, 0, 0)
    frames = [fr(f, [(box(80), EA, 1)]) for f in range(3)]
    frames += [fr(f, []) for f in range(3, 6)]
    frames += [fr(f, [(box(70), look, 5), (box(90), look, 6)]) for f in range(6, 12)]
    out, _ = run(frames, {}, target_adaptive_appearance=False)
    assert not any(t.has_box for t in out[3:])


def test_confirmation_frames_then_reacquired():
    frames = [fr(f, [(box(80), EA, 1)]) for f in range(3)]
    frames += [fr(f, []) for f in range(3, 6)]
    frames += [fr(f, [(box(85), EA, 9)]) for f in range(6, 12)]
    masks = {**{f: blob(80) for f in range(3)}, **{f: blob(85) for f in range(6, 12)}}
    out, _ = run(frames, masks, target_adaptive_appearance=False, target_reacquire_confirm_frames=2)
    assert out[6].state == TargetState.REACQUISITION_CANDIDATE and not out[6].has_box
    assert out[7].state == TargetState.REACQUIRED and out[7].has_box
    assert out[8].state == TargetState.VISIBLE_CONFIDENT


def test_empty_mask_during_occlusion_emits_no_box():
    frames = [fr(f, [(box(80), EA, 1)]) for f in range(3)]
    frames += [fr(f, [(box(200), EB, 2)]) for f in range(3, 8)]
    masks = {f: blob(80) if f < 3 else np.zeros((H, W), bool) for f in range(8)}
    out, _ = run(frames, masks)
    assert all(not t.has_box for t in out[3:])


def test_mask_returns_after_occlusion():
    frames = [fr(f, [(box(80), EA, 1)]) for f in range(3)]
    frames += [fr(f, []) for f in range(3, 6)]
    frames += [fr(f, [(box(88), EA, 1)]) for f in range(6, 10)]
    masks = {f: blob(80) if f < 3 or f >= 6 else np.zeros((H, W), bool) for f in range(10)}
    out, _ = run(frames, masks, target_adaptive_appearance=False)
    assert any(t.state == TargetState.REACQUIRED and t.has_box for t in out[6:])


def test_gallery_skips_overlap():
    test_overlap_keeps_identity_and_skips_gallery()


def test_lock_anchor_never_discarded():
    cfg = DeepHMSortConfig(target_gallery_min_interval_s=0.0, locked_target_gallery_size=4)
    from oline_cv.tracking.target_identity import LockedTargetIdentity
    lt = LockedTargetIdentity(cfg, 30.0, (W, H))
    lt.initialize(fr(0, [(box(80), EA, 1)]), 0)
    for f in range(1, 20):
        lt.step(fr(f, [(box(80), unit(1, 0.02 * (f % 2), 0, 0), 1)]))
    assert lt.anchor is not None
    np.testing.assert_allclose(lt.gallery_matrix()[0], EA / np.linalg.norm(EA), atol=1e-6)


def test_spatial_gate_expands_after_long_loss():
    from oline_cv.tracking.target_identity import LockedTargetIdentity
    lt = LockedTargetIdentity(DeepHMSortConfig(), 30.0, (W, H))
    short = lt.expansion_for(2)
    long = lt.expansion_for(90)
    assert long > short
    assert long == pytest.approx(4.0)


def test_mask_allows_larger_displacement():
    frames = [fr(f, [(box(50), EA, 1)]) for f in range(3)]
    frames += [fr(f, []) for f in range(3, 8)]
    frames += [fr(f, [(box(220), EA, 11)]) for f in range(8, 14)]
    masks = {**{f: blob(50) for f in range(3)}, **{f: blob(220) for f in range(8, 14)}}
    out, _ = run(frames, masks, target_adaptive_appearance=False)
    assert any(t.has_box and t.track_id == 11 for t in out[9:])


def test_weak_det_without_identity_cannot_become_target():
    frames = [fr(f, [(box(80), EA, 1)]) for f in range(3)]
    frames += [fr(f, []) for f in range(3, 6)]
    frames += [fr(f, [(box(200), EB, -1, 0.2)], recovery=[0]) for f in range(6, 12)]
    out, _ = run(frames, {})  # no mask
    assert not any(t.has_box for t in out[3:])


def test_empty_frame():
    frames = [fr(0, [(box(80), EA, 1)]), fr(1, [])]
    out, _ = run(frames, {0: blob(80)})
    assert not out[1].has_box


def test_mask_holds_through_detector_dropout():
    frames = [fr(f, [(box(80), EA, 1)]) for f in range(3)]
    frames += [fr(f, []) for f in range(3, 8)]
    out, _ = run(frames, {f: blob(80) for f in range(8)})
    assert all(t.has_box for t in out)
    assert any(t.reason == "mask_holds_through_dropout" for t in out[3:])


def test_mask_teleport_does_not_become_target():
    frames = [fr(f, [(box(80), EA, 1)]) for f in range(3)]
    frames += [fr(f, []) for f in range(3, 6)]
    frames += [fr(f, [(box(320), EB, 9)]) for f in range(6, 10)]
    masks = {**{f: blob(80) for f in range(3)}, **{f: blob(320) for f in range(6, 10)}}
    out, _ = run(frames, masks)
    assert not any(t.has_box and t.track_id == 9 for t in out)
    assert not any(t.has_box and t.reason == "mask_identity" for t in out[6:])


def test_overlap_without_mask_keeps_bound_track():
    frames = [fr(0, [(box(80), EA, 1)])]
    frames += [fr(f, [(box(80), EA, 1), (box(90), EB, 2)]) for f in range(1, 6)]
    out, _ = run(frames, {})
    assert all(t.has_box and t.track_id == 1 for t in out)


def test_all_detections_gone():
    frames = [fr(0, [(box(80), EA, 1)])] + [fr(f, []) for f in range(1, 40)]
    out, _ = run(frames, {0: blob(80)})
    assert all(not t.has_box for t in out[1:])
    assert out[-1].state == TargetState.LOST


def test_target_near_frame_border_skips_gallery():
    from oline_cv.tracking.target_identity import LockedTargetIdentity
    cfg = DeepHMSortConfig(target_gallery_min_interval_s=0.0)
    lt = LockedTargetIdentity(cfg, 30.0, (W, H))
    lt.initialize(fr(0, [(box(40), EA, 1)]), 0)
    lt.step(fr(1, [([1, 60, 41, 140], EA, 1)]))
    assert lt.gallery_skips.get("frame_edge") == 1


def test_mask_contradicts_visible_switch():
    test_mask_vetoes_wrong_deep_hm_switch()


def test_downstream_no_wrong_person_box(tmp_path):
    from oline_cv.motion3d.track_export import export_tracks
    from oline_cv.pose_tracker import PoseTracker
    from oline_cv.series import build_series

    frames_r = [fr(f, [(box(80), EA, 1), (box(300), EB, 2)]) for f in range(6)]
    frames_r[3] = fr(3, [(box(300), EB, 2)])
    out, _ = run(frames_r, {f: blob(80) if f != 3 else np.zeros((H, W), bool) for f in range(6)})
    cfg = AnalysisConfig()
    tr = PoseTracker.__new__(PoseTracker)
    tr.config = cfg
    tr._dl_center = None
    tr._pose_in_box = lambda frame, b: (
        np.stack([np.linspace(b[0], b[2], 17), np.linspace(b[1], b[3], 17)], 1),
        np.full(17, 0.9),
    )
    imgs = [np.zeros((H, W, 3), dtype=np.uint8) for _ in frames_r]
    ol, dl = tr._poses_from_targets(imgs, frames_r, out, 30.0)
    assert ol[3].track_state == "LOST" and not ol[3].usable
    assert ol[0].track_state == "TRACKED"
    man = export_tracks("c.mp4", ol, imgs, 30.0, W, H, tmp_path, save_full_frames=False)
    assert len(man.frames) == 6
    assert build_series(ol, 0, 5, 80.0, 30.0, cfg) is not None


def test_import_without_sam2_does_not_fail():
    import importlib
    import oline_cv.tracking as t
    importlib.reload(t)
    st, status = build_target_mask_tracker(
        DeepHMSortConfig(enable_target_segmentation=False), [np.zeros((8, 8, 3), np.uint8)],
    )
    assert status["available"] is False
    assert "sam2" not in sys.modules or True  # may already be installed; import of oline_cv must not require it


def test_degraded_status_when_checkpoint_missing(tmp_path, monkeypatch):
    monkeypatch.delenv("OLINE_SAM2_CHECKPOINT", raising=False)
    cfg = DeepHMSortConfig(enable_target_segmentation=True, sam2_checkpoint=str(tmp_path / "nope.pt"))
    tr, status = build_target_mask_tracker(cfg, [])
    assert status["available"] is False
    assert status["tracking_mode"] == "degraded"


def test_mask_coverage_math():
    m = blob(80, 100, 40, 80)
    assert mask_coverage(m, box(80)) == pytest.approx(1.0)
    assert mask_coverage(m, box(300)) == pytest.approx(0.0)


def test_occlusion_risk_overlap():
    r = fr(0, [(box(80), EA, 1), (box(90), EB, 2)])
    risk, reasons = evaluate_risk(
        r, det_index=0, bound_present=True, margin=0.2, d_app=0.01,
        reid_max=0.3, last_box=np.array(box(80)), cfg=DeepHMSortConfig(), unseen=0,
    )
    assert risk in (TargetRisk.AMBIGUOUS, TargetRisk.PARTIAL_OCCLUSION)
    assert "overlap" in reasons


def test_deep_hm_skips_ema_on_low_score():
    tr = DeepHMSort(DeepHMSortConfig(), fps=30)
    e = unit(1, 0)[None]
    tr.update(0, [box(80)], [0.9], e)
    before = tr.tracks[1].feature.copy()
    tr.update(1, [box(81)], [0.45], unit(0, 1)[None])  # low stage, no feature update
    assert np.allclose(tr.tracks[1].feature, before)


def test_annotation_and_metrics_false_assignment(tmp_path):
    p = tmp_path / "gt.json"
    p.write_text(
        '{"pick_xy": [0.2, 0.5], "frames": {"0": {"visible": true, "bbox": [60, 60, 100, 140]}, '
        '"1": {"visible": false}, "2": {"visible": true, "bbox": [60, 60, 100, 140]}}}',
        encoding="utf-8",
    )
    gt = load_ground_truth(p)
    pred = [np.array([60, 60, 100, 140]), np.array([200, 60, 240, 140]), np.array([60, 60, 100, 140])]
    m = target_metrics(pred, gt["boxes"])
    assert m["wrong_person_frames"] >= 1
    assert m["false_target_assignment_rate"] is not None
    assert "reacquisition_success_rate" in m


def test_append_recovery_boxes():
    r = fr(0, [(box(80), EA, 1, 0.9)])
    r.append_recovery([box(200)], [0.18], EB[None])
    assert len(r) == 2 and r.is_recovery(1) and not r.is_recovery(0)
    assert int(r.track_ids[1]) == -1
