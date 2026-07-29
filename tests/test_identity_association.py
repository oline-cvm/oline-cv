"""Association / frozen-appearance unit tests (no NN training)."""

from __future__ import annotations

import numpy as np

from oline_cv.appearance_ref import (
    FrozenAppearanceRef,
    appearance_vector,
    build_frozen_appearance,
    jersey_color_vector,
)
from oline_cv.association import (
    AssociationThresholds,
    IdentityAssociator,
    TrackState,
)


def _solid_frame(color_bgr, box, size=(400, 600)) -> tuple[np.ndarray, np.ndarray]:
    h, w = size
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:] = (30, 30, 30)
    x0, y0, x1, y1 = [int(v) for v in box]
    frame[y0:y1, x0:x1] = color_bgr
    return frame, np.array(box, dtype=float)


def test_frozen_reference_not_overwritten():
    box = [200, 100, 320, 300]
    frames = []
    boxes = []
    for _ in range(4):
        f, b = _solid_frame((40, 160, 40), box)
        frames.append(f)
        boxes.append(b)
    ref = build_frozen_appearance(frames, boxes, formation_xy=np.array([260.0, 200.0]))
    assert ref is not None
    frozen_copy = ref.frozen.copy()
    # Recent update must not touch frozen
    crop = frames[0][100:300, 200:320]
    crop = np.ascontiguousarray(crop)
    # Force a valid torso-sized crop via rebuild path
    from oline_cv.appearance_ref import crop_torso

    c = crop_torso(frames[0], boxes[0])
    assert c is not None
    ok = ref.maybe_update_recent(c, det_conf=0.9, occluded=False, min_conf=0.5, min_frozen_sim=0.1)
    assert ok
    assert np.allclose(ref.frozen, frozen_copy)


def test_reject_low_confidence_and_wrong_jersey():
    lock_box = [200, 100, 320, 300]
    other_box = [400, 100, 520, 300]
    f_lock, b_lock = _solid_frame((40, 180, 40), lock_box)  # green
    f_mix = f_lock.copy()
    x0, y0, x1, y1 = other_box
    f_mix[y0:y1, x0:x1] = (40, 40, 200)  # red neighbor

    ref = build_frozen_appearance([f_lock] * 3, [b_lock] * 3, formation_xy=np.array([260.0, 200.0]))
    assert ref is not None
    assoc = IdentityAssociator(
        ref,
        thresholds=AssociationThresholds(min_appearance=0.85, min_jersey=0.55, min_weighted=0.4),
        team_jersey_floor=0.55,
        lost_buffer=10,
    )
    assoc.prev_bbox = b_lock.copy()
    assoc.prev_center = np.array([260.0, 200.0])

    boxes = np.stack([b_lock, np.array(other_box, dtype=float)])
    confs = np.array([0.9, 0.95])
    decision = assoc.associate(5, f_mix, boxes, confs, track_ids=np.array([1, 2]))
    # Green lock should win; red rejected on jersey/appearance
    assert decision.best is not None
    assert decision.best.track_id == 1
    red = [c for c in decision.candidates if c.track_id == 2][0]
    assert red.accepted is False


def test_occlusion_does_not_transfer_target_id():
    lock_box = [200, 100, 320, 300]
    neighbor = [250, 110, 370, 310]  # heavy overlap
    f_lock, b_lock = _solid_frame((40, 180, 40), lock_box)
    f_occ = f_lock.copy()
    x0, y0, x1, y1 = neighbor
    f_occ[y0:y1, x0:x1] = (20, 20, 220)  # different color overlapping

    ref = build_frozen_appearance([f_lock] * 3, [b_lock] * 3, target_id=1, formation_xy=np.array([260.0, 200.0]))
    assoc = IdentityAssociator(
        ref,
        thresholds=AssociationThresholds(min_appearance=0.80, min_weighted=0.35),
        team_jersey_floor=0.5,
        lost_buffer=20,
    )
    assoc.prev_bbox = b_lock.copy()
    assoc.prev_center = np.array([260.0, 200.0])
    assoc.botsort_id = 7
    assoc.locked_botsort_id = 7

    boxes = np.stack([np.array(neighbor, dtype=float)])  # only neighbor visible
    confs = np.array([0.99])
    decision = assoc.associate(120, f_occ, boxes, confs, track_ids=np.array([99]))
    # Wrong appearance / different id while not LOST long enough → no transfer
    assert decision.target_id == 1
    assert decision.best is None or decision.best.track_id != 99 or decision.state == TrackState.LOST
    assert decision.target_id == 1


def test_blocked_transfer_to_nearby_player_while_tracked():
    lock_box = [200, 100, 320, 300]
    neighbor = [330, 100, 450, 300]
    f_lock, b_lock = _solid_frame((40, 180, 40), lock_box)
    f_both = f_lock.copy()
    x0, y0, x1, y1 = neighbor
    f_both[y0:y1, x0:x1] = (40, 180, 40)  # same color — spatial steal risk

    ref = build_frozen_appearance([f_lock] * 3, [b_lock] * 3, target_id=1, formation_xy=np.array([260.0, 200.0]))
    assoc = IdentityAssociator(
        ref,
        thresholds=AssociationThresholds(),  # calib-like
        lost_buffer=20,
    )
    assoc.prev_bbox = b_lock.copy()
    assoc.prev_center = np.array([260.0, 200.0])
    assoc.locked_botsort_id = 6
    assoc.botsort_id = 6
    assoc.state = TrackState.TRACKED

    # Only neighbor present (id 9) — must NOT adopt while not lost
    boxes = np.stack([np.array(neighbor, dtype=float)])
    confs = np.array([0.99])
    decision = assoc.associate(120, f_both, boxes, confs, track_ids=np.array([9]))
    assert decision.target_id == 1
    assert decision.state == TrackState.LOST
    assert decision.best is None
    assert assoc.locked_botsort_id == 6


def test_reidentify_after_lost():
    lock_box = [200, 100, 320, 300]
    f_lock, b_lock = _solid_frame((40, 180, 40), lock_box)
    ref = build_frozen_appearance([f_lock] * 3, [b_lock] * 3, formation_xy=np.array([260.0, 200.0]))
    assoc = IdentityAssociator(
        ref,
        thresholds=AssociationThresholds(min_appearance=0.7, min_weighted=0.3),
        lost_buffer=30,
    )
    assoc.prev_bbox = b_lock.copy()
    assoc.prev_center = np.array([260.0, 200.0])
    assoc.state = TrackState.LOST
    assoc.lost_frames = 5

    decision = assoc.associate(
        130,
        f_lock,
        np.stack([b_lock]),
        np.array([0.9]),
        track_ids=np.array([7]),
    )
    assert decision.state == TrackState.REIDENTIFIED
    assert decision.best is not None
    assert decision.target_id == 1
