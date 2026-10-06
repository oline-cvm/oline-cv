"""Target-tracking metrics for one locked player.

Ground-truth JSON (frames not listed are unannotated and ignored)::

    {
      "video": "uploads/clip.mp4",          # informational
      "pick_xy": [0.43, 0.72],              # optional default tap
      "pick_t": 0.2,                        # optional
      "target": {
        "boxes": {
          "0":  [x1, y1, x2, y2],           # target visible here
          "57": null,                       # target hidden / off screen
          ...
        }
      }
    }

A predicted box is correct when IoU with the ground-truth box is at least
``iou_threshold``. Any predicted box on a frame where the target is hidden,
or with IoU below the threshold, is a wrong-person frame.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oline_cv.tracking.expansion_iou import iou_matrix


def load_ground_truth(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = (d.get("target") or {}).get("boxes") or {}
    boxes = {int(k): (None if v is None else np.asarray(v, dtype=float)) for k, v in raw.items()}
    return {**d, "boxes": boxes}


def _runs(flags: list[bool]) -> list[tuple[int, int]]:
    out, start = [], None
    for i, f in enumerate(flags + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    return out


def target_metrics(
    pred: list[np.ndarray | None],
    gt: dict[int, np.ndarray | None],
    iou_threshold: float = 0.5,
) -> dict:
    frames = sorted(f for f in gt if 0 <= f < len(pred))
    outcome: dict[int, str] = {}
    for f in frames:
        g, p = gt[f], pred[f]
        if g is None:
            outcome[f] = "wrong" if p is not None else "correct_absent"
        elif p is None:
            outcome[f] = "missed"
        else:
            outcome[f] = "correct" if iou_matrix(p[None], g[None])[0, 0] >= iou_threshold else "wrong"
    n = len(frames)
    visible = [f for f in frames if gt[f] is not None]
    boxed = [f for f in frames if pred[f] is not None]
    correct = sum(1 for f in frames if outcome[f] == "correct")
    wrong = sum(1 for f in frames if outcome[f] == "wrong")
    missed = sum(1 for f in frames if outcome[f] == "missed")

    seq = [outcome[f] for f in frames]
    switches = sum(1 for a, b in zip(seq, seq[1:]) if a == "correct" and b == "wrong")
    wrong_runs = _runs([s == "wrong" for s in seq])

    # Occlusions: annotated hidden runs followed by a visible frame.
    hidden = [gt[f] is None for f in frames]
    occlusions = []
    for a, b in _runs(hidden):
        after = [frames[i] for i in range(b + 1, n) if gt[frames[i]] is not None]
        latency = None
        for f in after:
            if outcome[f] == "correct":
                latency = f - frames[b + 1] if b + 1 < n else None
                break
            if outcome[f] == "wrong":
                break
        occlusions.append({
            "start": frames[a], "end": frames[b],
            "wrong_during": sum(1 for i in range(a, b + 1) if seq[i] == "wrong"),
            "reacquired": latency is not None,
            "reacquire_latency_frames": latency,
        })

    def rate(x, d):
        return None if d == 0 else round(x / d, 4)

    return {
        "annotated_frames": n,
        "gt_visible_frames": len(visible),
        "predicted_box_frames": len(boxed),
        "correct_frames": correct,
        "wrong_person_frames": wrong,
        "missed_frames": missed,
        "target_recall": rate(correct, len(visible)),
        "target_precision": rate(correct, len(boxed)),
        "wrong_person_rate": rate(wrong, len(boxed)),
        "identity_switches": switches,
        "longest_wrong_run": max((b - a + 1 for a, b in wrong_runs), default=0),
        "occlusions": occlusions,
    }


def tracking_health(poses, fps: float) -> dict:
    """Ground-truth-free summary of a FramePose sequence."""
    states: dict[str, int] = {}
    tstates: dict[str, int] = {}
    for p in poses:
        states[p.track_state] = states.get(p.track_state, 0) + 1
        if getattr(p, "target_state", None):
            tstates[p.target_state] = tstates.get(p.target_state, 0) + 1
    boxed = [p for p in poses if p.bbox_xyxy is not None and not p.interpolated]
    jumps = []
    for a, b in zip(boxed, boxed[1:]):
        if b.frame_idx - a.frame_idx != 1:
            continue
        ca = (a.bbox_xyxy[:2] + a.bbox_xyxy[2:]) / 2
        cb = (b.bbox_xyxy[:2] + b.bbox_xyxy[2:]) / 2
        h = max(1.0, float(a.bbox_xyxy[3] - a.bbox_xyxy[1]))
        jumps.append(float(np.linalg.norm(cb - ca) / h))
    tid_changes = sum(
        1 for a, b in zip(boxed, boxed[1:])
        if a.track_id is not None and b.track_id is not None and a.track_id != b.track_id
    )
    return {
        "frames": len(poses),
        "boxed_frames": len(boxed),
        "coverage": round(len(boxed) / max(1, len(poses)), 4),
        "track_states": states,
        "target_states": tstates,
        "track_id_changes": tid_changes,
        "max_frame_jump_box_h": round(max(jumps), 3) if jumps else 0.0,
        "jumps_over_0.5h": sum(1 for j in jumps if j > 0.5),
        "fps": fps,
    }
