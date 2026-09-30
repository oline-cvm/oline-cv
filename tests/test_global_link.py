import numpy as np

from oline_cv.global_link import (
    FrameCandidates,
    LinkParams,
    gap_segments,
    interpolate_boxes,
    link_target,
)

W, H = 60.0, 140.0  # player box size; diagonal ~152 px


def _box(cx: float, cy: float) -> list[float]:
    return [cx - W / 2, cy - H / 2, cx + W / 2, cy + H / 2]


def _frame(players: list[tuple[float, float, int]], app: float = 0.9) -> FrameCandidates:
    """players: (cx, cy, botsort_id)."""
    if not players:
        return FrameCandidates.empty()
    return FrameCandidates(
        boxes=np.array([_box(x, y) for x, y, _ in players], dtype=float),
        confs=np.full(len(players), 0.9),
        track_ids=np.array([tid for _, _, tid in players], dtype=int),
        appearance=np.full(len(players), app),
    )


def _chosen_x(cands, choice):
    return [
        None if j is None else float((cands[t].boxes[j, 0] + cands[t].boxes[j, 2]) / 2)
        for t, j in enumerate(choice)
    ]


def test_ignores_botsort_id_swap_between_identical_teammates():
    # Target drifts right slowly; a teammate sits 0.6 diagonals away. At frame 20
    # BoT-SORT swaps their ids. The path must stay on the target trajectory.
    cands = []
    for t in range(40):
        tx = 300 + 2.0 * t
        nx = 390 + 1.0 * t
        ids = (1, 2) if t < 20 else (2, 1)
        cands.append(_frame([(nx, 400, ids[1]), (tx, 400, ids[0])]))
    lock_idx = 1
    res = link_target(cands, 0, lock_idx)
    xs = _chosen_x(cands, res.choice)
    for t, x in enumerate(xs):
        assert x is not None
        assert abs(x - (300 + 2.0 * t)) < 1e-6


def test_bridges_occlusion_without_jumping_to_neighbour():
    cands = []
    for t in range(40):
        tx = 300 + 2.0 * t
        players = [(tx + 95, 400, 2)]  # neighbour always visible
        if not (15 <= t < 21):  # target hidden for 6 frames
            players.append((tx, 400, 1))
        cands.append(_frame(players))
    res = link_target(cands, 0, 1)
    xs = _chosen_x(cands, res.choice)
    for t, x in enumerate(xs):
        if 15 <= t < 21:
            assert x is None
        else:
            assert x is not None and abs(x - (300 + 2.0 * t)) < 1e-6
    assert gap_segments(res.choice) == [(15, 20)]
    filled = interpolate_boxes(cands, res.choice, max_gap=12)
    assert sorted(filled) == list(range(15, 21))
    cx = (filled[17][0] + filled[17][2]) / 2
    assert abs(cx - (300 + 2.0 * 17)) < 1e-6


def test_never_teleports_past_speed_gate():
    # Target vanishes at frame 10; a far player (3 diagonals away) is the only detection.
    cands = []
    for t in range(30):
        players = [(1000, 400, 7)]
        if t < 10:
            players.append((300, 400, 1))
        cands.append(_frame(players))
    res = link_target(cands, 0, 1, LinkParams(max_gap_frames=15))
    diag = np.hypot(W, H)
    prev = None
    for t, j in enumerate(res.choice):
        if j is None:
            continue
        c = cands[t].boxes[j]
        cx = (c[0] + c[2]) / 2
        if prev is not None:
            pt, px = prev
            assert abs(cx - px) / diag <= 0.12 * (t - pt) + 0.10 + 1e-9
        prev = (t, cx)
    assert all(j is None for j in res.choice[10:])


def test_camera_pan_compensated_so_target_is_not_lost():
    # Broadcast pan: every player slides 40 px/frame (> speed gate) while the
    # target barely moves on the field. A teammate one step away stays close.
    cands = []
    field = [(100.0, 700.0), (250.0, 900.0), (400.0, 650.0), (600.0, 800.0), (800.0, 700.0)]
    for t in range(40):
        pan = -40.0 * t
        players = [(fx + pan, fy, 10 + i) for i, (fx, fy) in enumerate(field)]
        players.append((320 + 1.0 * t + pan, 400, 1))  # target
        players.append((420 + 1.0 * t + pan, 400, 2))  # neighbour
        cands.append(_frame(players))
    tgt = 5
    res = link_target(cands, 0, tgt)
    assert all(j == tgt for j in res.choice)
    assert res.diagnostics["camera_pan_px"] > 1000

    res_raw = link_target(cands, 0, tgt, compensate_camera=False)
    assert any(j != tgt for j in res_raw.choice)


def test_leash_stops_path_riding_a_player_who_walks_away():
    # Target is hidden from frame 10. A teammate starting beside him jogs off
    # 8 px/frame (well inside the speed gate) and ends ~5 heights away.
    cands = []
    for t in range(60):
        players = [(380 + 8.0 * max(0, t - 5), 400, 2)]
        if t < 10:
            players.append((300, 400, 1))
        cands.append(_frame(players))
    free = link_target(cands, 0, 1, LinkParams(max_gap_frames=15))
    assert free.choice[-1] is not None  # without a leash it follows him off

    res = link_target(cands, 0, 1, LinkParams(max_gap_frames=15, leash_h=2.0, w_leash=4.0))
    for t, j in enumerate(res.choice):
        if j is None:
            continue
        cx = (cands[t].boxes[j, 0] + cands[t].boxes[j, 2]) / 2
        assert abs(cx - 300) / H < 2.0 + 0.8  # net-positive cost past ~2.5 heights


def test_lock_frame_is_honoured_and_search_runs_backward():
    # Lock in the middle on the lower-confidence, lower-appearance player.
    cands = []
    for t in range(21):
        f = _frame([(300 + t, 400, 1), (500 + t, 400, 2)])
        f.appearance = np.array([0.95, 0.60])
        cands.append(f)
    res = link_target(cands, 10, 1)
    assert res.choice[10] == 1
    assert all(j == 1 for j in res.choice)
