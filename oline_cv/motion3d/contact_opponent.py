"""Find and track the person the OL goes into contact with.

Uses the per-frame WHAM/YOLO detections already scored in association.json.
We never steal the selected lineman's box — the opponent is always chosen from
the rejected (or secondary) candidates, then linked across frames by IoU so the
same rusher stays identity-stable through the engagement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from oline_cv.motion3d.target_association import iou_xyxy


@dataclass
class OpponentFrame:
    frame_index: int
    bbox: list[float]
    score: float  # proximity score vs target (higher = closer engagement)
    linked_iou: float  # IoU with previous opponent box (1.0 on first frame)


@dataclass
class OpponentTrack:
    frames: list[OpponentFrame]
    start: int
    end: int
    method: str = "nearest_overlap_link"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "start": self.start,
            "end": self.end,
            "frames": len(self.frames),
            "frame_indices": [f.frame_index for f in self.frames],
            "bboxes": [f.bbox for f in self.frames],
            "scores": [round(f.score, 4) for f in self.frames],
        }


def _center(b: Sequence[float]) -> np.ndarray:
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])


def _diag(b: Sequence[float]) -> float:
    return float(np.hypot(b[2] - b[0], b[3] - b[1]))


def proximity_score(target: Sequence[float], cand: Sequence[float]) -> float:
    """Higher when the candidate is near / overlapping the OL (contact)."""
    iou = iou_xyxy(target, cand)
    cd = float(np.linalg.norm(_center(cand) - _center(target)) / max(_diag(target), 1e-6))
    # Prefer overlap, fall back to nearness. Far background players score ~0.
    return float(0.7 * iou + 0.3 * max(0.0, 1.0 - cd / 1.2))


def extract_opponent_track(
    association_path: str | Path,
    min_score: float = 0.18,
    min_link_iou: float = 0.20,
    min_run: int = 20,
    seed_score: float = 0.25,
    max_pre_contact: int = 35,
) -> OpponentTrack | None:
    """Build a contiguous opponent track through the contact window.

    Strategy:
      1. Score every non-selected detection by proximity to the OL.
      2. Seed at the peak-contact frame (strongest proximity).
      3. Expand forward while the same body stays linkable.
      4. Expand backward only a short approach window (``max_pre_contact``),
         so a pre-snap neighbor never becomes a 165-frame false track.
    """
    data = json.loads(Path(association_path).read_text(encoding="utf-8"))
    frames = data.get("frames") or []
    if not frames:
        return None

    per: dict[int, list[dict[str, Any]]] = {}
    targets: dict[int, list[float]] = {}
    for fr in frames:
        idx = int(fr["frame_index"])
        if fr.get("target_bbox") is None:
            continue
        targets[idx] = [float(v) for v in fr["target_bbox"]]
        cands = []
        for c in fr.get("candidates") or []:
            if c.get("selected") or c.get("interpolated"):
                continue
            bbox = [float(v) for v in c["bbox"]]
            cands.append({"bbox": bbox, "score": proximity_score(targets[idx], bbox)})
        cands.sort(key=lambda x: -x["score"])
        per[idx] = cands

    order = sorted(targets)
    ranked = []
    for idx in order:
        if per.get(idx):
            ranked.append((per[idx][0]["score"], idx, per[idx][0]["bbox"]))
    if not ranked:
        return None
    ranked.sort(reverse=True)
    peak_score, seed_i, seed_box = ranked[0]
    if peak_score < min_score:
        return None

    # Prefer the earliest frame that clears the seed bar near the peak, so we
    # start at the beginning of engagement rather than mid-collision.
    for score, idx, box in sorted(ranked, key=lambda r: r[1]):
        if idx >= seed_i - 15 and score >= seed_score:
            seed_i, seed_box = idx, box
            break

    def pick_next(idx: int, prev_box: list[float]) -> tuple[dict[str, Any], float] | None:
        cands = per.get(idx) or []
        if not cands:
            return None
        linked = max(cands, key=lambda c: iou_xyxy(prev_box, c["bbox"]))
        link_iou = iou_xyxy(prev_box, linked["bbox"])
        near = cands[0]
        pick = linked if link_iou >= min_link_iou else near
        pick_iou = iou_xyxy(prev_box, pick["bbox"])
        if pick["score"] < min_score and pick_iou < min_link_iou:
            return None
        # Reject a jump to a far body even if IoU to prev somehow survives.
        if pick["score"] < min_score * 0.5:
            return None
        return pick, float(pick_iou)

    track: list[OpponentFrame] = [
        OpponentFrame(seed_i, seed_box, proximity_score(targets[seed_i], seed_box), 1.0)
    ]
    prev = seed_box
    for idx in order:
        if idx <= seed_i:
            continue
        got = pick_next(idx, prev)
        if got is None:
            break
        pick, pick_iou = got
        track.append(OpponentFrame(idx, pick["bbox"], pick["score"], pick_iou))
        prev = pick["bbox"]

    prev = seed_box
    backward: list[OpponentFrame] = []
    for idx in reversed(order):
        if idx >= seed_i:
            continue
        if seed_i - idx > max_pre_contact:
            break
        got = pick_next(idx, prev)
        if got is None:
            break
        pick, pick_iou = got
        # Approach frames may be weaker; still require some nearness.
        if pick["score"] < min_score * 0.75 and pick_iou < min_link_iou:
            break
        backward.append(OpponentFrame(idx, pick["bbox"], pick["score"], pick_iou))
        prev = pick["bbox"]
    backward.reverse()
    track = backward + track

    if len(track) < min_run:
        return None
    return OpponentTrack(
        frames=track,
        start=track[0].frame_index,
        end=track[-1].frame_index,
    )


def write_opponent_manifest(
    track: OpponentTrack,
    source_manifest_path: str | Path,
    out_path: str | Path,
    video_path: str | None = None,
) -> Path:
    """Write a minimal TrackManifest the WHAM runner can consume for the opponent."""
    from oline_cv.motion3d.schema import (
        BboxSource,
        TrackFrame,
        TrackManifest,
        load_manifest,
    )

    src = load_manifest(source_manifest_path)
    by_idx = {f.frame_index: f for f in src.frames}
    empty_kp = [[None, None, 0.0] for _ in range(17)]
    frames: list[TrackFrame] = []
    for of in track.frames:
        base = by_idx.get(of.frame_index)
        frames.append(
            TrackFrame(
                frame_index=of.frame_index,
                timestamp=float(of.frame_index) / float(src.fps or 30.0),
                track_id=9001,  # synthetic opponent id — never collide with OL
                target_id=9001,
                bbox=list(of.bbox),
                bbox_source=BboxSource.DETECTED.value,
                detection_confidence=float(of.score),
                track_state="OK",
                track_confidence=float(of.score),
                keypoints_2d=list(base.keypoints_2d) if base and base.keypoints_2d else empty_kp,
                low_confidence=False,
                usable=True,
                frame_path=base.frame_path if base else None,
                crop=None,
            )
        )

    video = dict(src.video)
    if video_path:
        video["path"] = video_path
    manifest = TrackManifest(
        video=video,
        target={
            "target_id": 9001,
            "jersey": None,
            "role": "contact_opponent",
            "source_track": src.target,
        },
        export={
            "source": "contact_opponent",
            "parent_manifest": str(source_manifest_path),
        },
        frames=frames,
        segments=[[track.start, track.end]],
        schema_version=src.schema_version,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.save(out)
    (out.parent / "contact_opponent.json").write_text(
        json.dumps(track.to_dict(), indent=2), encoding="utf-8"
    )
    return out
