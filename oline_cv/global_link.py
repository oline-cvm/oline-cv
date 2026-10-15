"""Whole-clip target path search (offline identity linking).

Frame-by-frame association fails on sideline OL film: teammates in identical
jerseys overlap, BoT-SORT swaps ids between neighbours, and every close call
becomes LOST. Because the whole clip is in memory, we instead pick the single
most consistent path for the locked player over all frames at once.

Model: each frame's state is one of its detections or a gap. A path pays

* a per-detection reward (so it prefers covering frames over skipping),
* a unary cost for weak appearance / detection confidence,
* a transition cost from the previous chosen box: squared centre jump
  (scaled by elapsed frames, hard-gated by a max speed), size change, and a
  small BoT-SORT id-change penalty.

Appearance is deliberately weak — teammates look the same — so geometry
decides and colour only breaks near-ties. The lock frame is a hard constraint;
the search runs forward and backward from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameCandidates:
    """All detections in one frame. ``track_ids`` uses -1 for "no id"."""

    boxes: np.ndarray  # (N, 4) xyxy
    confs: np.ndarray  # (N,)
    track_ids: np.ndarray  # (N,) int
    appearance: np.ndarray  # (N,) similarity to the frozen target, 0..1

    @classmethod
    def empty(cls) -> "FrameCandidates":
        return cls(
            boxes=np.zeros((0, 4)),
            confs=np.zeros(0),
            track_ids=np.zeros(0, dtype=int),
            appearance=np.zeros(0),
        )

    def __len__(self) -> int:
        return int(len(self.confs))


@dataclass
class LinkParams:
    max_gap_frames: int = 90  # longest occlusion a path may bridge
    # A tackle's kick-step moves ~0.03 box diagonals per frame at 60 fps; 0.12
    # leaves headroom for camera pans while blocking hops to a neighbour.
    max_speed_diag: float = 0.12  # max centre motion per frame, x box diagonal
    gate_slack_diag: float = 0.10
    max_gate_diag: float = 2.0
    w_motion: float = 8.0
    # Resuming after a gap pays for distance without the 1/k discount. Without
    # this, jumping onto a neighbouring teammate is cheaper than waiting out an
    # occlusion (the neighbour keeps earning the per-frame detection reward).
    w_gap_distance: float = 20.0
    w_size: float = 1.0
    w_id_change: float = 0.3
    w_appearance: float = 0.5
    w_conf: float = 0.3
    detection_reward: float = 1.0
    reacquire_penalty: float = 0.4  # one-off cost for resuming after a gap
    # Pass-pro tackles stay within a few body lengths of their stance spot.
    # Past ``leash_h`` lock-box heights (camera-stabilized) each frame pays
    # ``w_leash`` per squared extra height, so a far-off referee or a back
    # running by is worth less than a gap. None disables the leash.
    leash_h: float | None = None
    w_leash: float = 0.5


@dataclass
class LinkResult:
    choice: list[int | None]  # detection index per frame, None = gap
    cost: float
    diagnostics: dict = field(default_factory=dict)


def _centers(b: np.ndarray) -> np.ndarray:
    return (b[:, :2] + b[:, 2:]) / 2.0


def _diags(b: np.ndarray) -> np.ndarray:
    return np.maximum(np.linalg.norm(b[:, 2:] - b[:, :2], axis=1), 1.0)


def _areas(b: np.ndarray) -> np.ndarray:
    return np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1.0)


def _unary(
    fc: FrameCandidates, p: LinkParams, anchor: tuple[np.ndarray, float] | None = None
) -> np.ndarray:
    app = np.clip(fc.appearance, 0.0, 1.0)
    conf = np.clip(fc.confs, 0.0, 1.0)
    cost = -p.detection_reward + p.w_appearance * (1.0 - app) + p.w_conf * (1.0 - conf)
    if anchor is not None and p.leash_h is not None and len(fc):
        origin, h0 = anchor
        d = np.linalg.norm(_centers(fc.boxes) - origin[None, :], axis=1) / h0
        cost = cost + p.w_leash * np.maximum(0.0, d - p.leash_h) ** 2
    return cost


def _transition(
    prev: FrameCandidates, cur: FrameCandidates, k: int, p: LinkParams
) -> np.ndarray:
    """(N_prev, N_cur) cost of moving from prev detections to cur after k frames."""
    cp, cc = _centers(prev.boxes), _centers(cur.boxes)
    diag = (_diags(prev.boxes)[:, None] + _diags(cur.boxes)[None, :]) / 2.0
    dist = np.linalg.norm(cp[:, None, :] - cc[None, :, :], axis=2) / diag
    gate = min(p.max_gate_diag, p.max_speed_diag * k + p.gate_slack_diag)

    cost = p.w_motion * dist**2 / k
    cost += p.w_size * np.abs(
        np.log(_areas(cur.boxes)[None, :] / _areas(prev.boxes)[:, None])
    )
    ids_p, ids_c = prev.track_ids[:, None], cur.track_ids[None, :]
    changed = (ids_p >= 0) & (ids_c >= 0) & (ids_p != ids_c)
    cost += p.w_id_change * changed
    if k > 1:
        cost += p.reacquire_penalty + p.w_gap_distance * dist**2
    cost[dist > gate] = np.inf
    return cost


def _forward(
    cands: list[FrameCandidates], start_idx: int, p: LinkParams
) -> tuple[list[int | None], float]:
    """Best path over ``cands`` whose frame 0 is fixed to detection ``start_idx``."""
    T = len(cands)
    cost: list[np.ndarray] = [np.full(len(c), np.inf) for c in cands]
    back: list[list[tuple[int, int] | None]] = [[None] * len(c) for c in cands]
    cost[0][start_idx] = 0.0
    b0 = cands[0].boxes[start_idx]
    anchor = (_centers(b0[None, :])[0], max(float(b0[3] - b0[1]), 1.0))
    unary = [_unary(c, p, anchor) for c in cands]

    for t in range(1, T):
        n = len(cands[t])
        if n == 0:
            continue
        best = np.full(n, np.inf)
        best_bp: list[tuple[int, int] | None] = [None] * n
        for k in range(1, min(p.max_gap_frames + 1, t) + 1):
            s = t - k
            prev_cost = cost[s]
            if len(prev_cost) == 0 or not np.isfinite(prev_cost).any():
                continue
            total = prev_cost[:, None] + _transition(cands[s], cands[t], k, p)
            arg = np.argmin(total, axis=0)
            vals = total[arg, np.arange(n)]
            better = vals < best
            for j in np.nonzero(better)[0]:
                best[j] = vals[j]
                best_bp[j] = (s, int(arg[j]))
        cost[t] = best + unary[t]
        back[t] = best_bp

    # End on the reachable state with the lowest cost (trailing frames are gaps).
    end_t, end_j, end_cost = 0, start_idx, 0.0
    for t in range(T):
        if len(cost[t]) and np.isfinite(cost[t]).any():
            j = int(np.argmin(cost[t]))
            if cost[t][j] < end_cost:
                end_t, end_j, end_cost = t, j, float(cost[t][j])

    choice: list[int | None] = [None] * T
    node: tuple[int, int] | None = (end_t, end_j)
    while node is not None:
        t, j = node
        choice[t] = j
        node = back[t][j]
    choice[0] = start_idx
    return choice, end_cost


def camera_motion(cands: list[FrameCandidates], min_matches: int = 4) -> np.ndarray:
    """Per-frame zoom/pan ``(a, bx, by)`` so that ``x_t = a * x_0 + b``.

    Broadcast film pans and zooms; in image space the tackle can move faster
    than any sane speed gate while barely moving on the field. Consecutive
    frames are related through BoT-SORT ids seen in both: zoom is the median
    height ratio, pan the median residual shift. Medians shrug off the odd id
    swap and players running in one direction. Frames without enough matches
    inherit the previous transform.
    """
    T = len(cands)
    out = np.zeros((T, 3))
    out[:, 0] = 1.0
    for t in range(1, T):
        a_prev, b_prev = out[t - 1, 0], out[t - 1, 1:]
        prev, cur = cands[t - 1], cands[t]
        step_a, step_b = 1.0, np.zeros(2)
        if len(prev) and len(cur):
            ids_p = {int(i): k for k, i in enumerate(prev.track_ids) if i >= 0}
            pairs = [(ids_p[int(i)], k) for k, i in enumerate(cur.track_ids) if int(i) in ids_p]
            if len(pairs) >= min_matches:
                ip = np.array([p for p, _ in pairs])
                ic = np.array([c for _, c in pairs])
                hp = np.maximum(prev.boxes[ip, 3] - prev.boxes[ip, 1], 1.0)
                hc = np.maximum(cur.boxes[ic, 3] - cur.boxes[ic, 1], 1.0)
                step_a = float(np.clip(np.median(hc / hp), 0.8, 1.25))
                step_b = np.median(
                    _centers(cur.boxes[ic]) - step_a * _centers(prev.boxes[ip]), axis=0
                )
        out[t, 0] = step_a * a_prev
        out[t, 1:] = step_a * b_prev + step_b
    return out


def stabilize(cands: list[FrameCandidates], motion: np.ndarray) -> list[FrameCandidates]:
    """Map every frame's boxes into frame-0 coordinates (undo pan/zoom)."""
    out = []
    for fc, (a, bx, by) in zip(cands, motion):
        if len(fc) == 0:
            out.append(fc)
            continue
        b = fc.boxes.astype(float).copy()
        b[:, [0, 2]] = (b[:, [0, 2]] - bx) / a
        b[:, [1, 3]] = (b[:, [1, 3]] - by) / a
        out.append(
            FrameCandidates(boxes=b, confs=fc.confs, track_ids=fc.track_ids, appearance=fc.appearance)
        )
    return out


def link_target(
    cands: list[FrameCandidates],
    lock_frame: int,
    lock_idx: int,
    params: LinkParams | None = None,
    compensate_camera: bool = True,
) -> LinkResult:
    """Pick the target's detection in every frame (or a gap).

    ``lock_frame`` / ``lock_idx`` pin the path to the locked detection. With
    ``compensate_camera`` the search runs on pan/zoom-stabilized boxes; the
    returned indices still refer to the original detections.
    """
    p = params or LinkParams()
    T = len(cands)
    if T == 0:
        return LinkResult(choice=[], cost=0.0)
    if not (0 <= lock_frame < T) or not (0 <= lock_idx < len(cands[lock_frame])):
        raise ValueError("lock frame/index out of range")

    motion = camera_motion(cands) if compensate_camera else None
    work = stabilize(cands, motion) if motion is not None else cands

    fwd, c_f = _forward(work[lock_frame:], lock_idx, p)
    bwd, c_b = _forward(work[: lock_frame + 1][::-1], lock_idx, p)
    choice = list(reversed(bwd))[:-1] + fwd

    covered = sum(1 for c in choice if c is not None)
    diag = {"frames": T, "covered": covered, "coverage": covered / T}
    if motion is not None:
        diag["camera_zoom_range"] = [float(motion[:, 0].min()), float(motion[:, 0].max())]
        diag["camera_pan_px"] = float(np.abs(motion[:, 1:]).max())
    return LinkResult(choice=choice, cost=c_f + c_b, diagnostics=diag)


def gap_segments(choice: list[int | None]) -> list[tuple[int, int]]:
    """Inclusive (start, end) runs of None that sit between two chosen frames."""
    out: list[tuple[int, int]] = []
    last = None
    for t, c in enumerate(choice):
        if c is None:
            continue
        if last is not None and t - last > 1:
            out.append((last + 1, t - 1))
        last = t
    return out


def interpolate_boxes(
    cands: list[FrameCandidates], choice: list[int | None], max_gap: int
) -> dict[int, np.ndarray]:
    """Linear box fill for bridged gaps no longer than ``max_gap`` frames."""
    filled: dict[int, np.ndarray] = {}
    for a, b in gap_segments(choice):
        if b - a + 1 > max_gap:
            continue
        box0 = cands[a - 1].boxes[choice[a - 1]]
        box1 = cands[b + 1].boxes[choice[b + 1]]
        span = (b + 1) - (a - 1)
        for t in range(a, b + 1):
            w = (t - (a - 1)) / span
            filled[t] = (1.0 - w) * box0 + w * box1
    return filled
