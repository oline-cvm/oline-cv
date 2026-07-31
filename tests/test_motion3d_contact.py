"""Contact opponent track extraction."""

from __future__ import annotations

import json
from pathlib import Path

from oline_cv.motion3d.contact_opponent import extract_opponent_track, proximity_score


def _assoc(tmp_path: Path, frames: list[dict]) -> Path:
    p = tmp_path / "association.json"
    p.write_text(json.dumps({"frames": frames, "thresholds": {}}), encoding="utf-8")
    return p


def test_proximity_prefers_overlap():
    target = [100.0, 100.0, 200.0, 300.0]
    near = [150.0, 120.0, 250.0, 320.0]
    far = [800.0, 100.0, 900.0, 300.0]
    assert proximity_score(target, near) > proximity_score(target, far)


def test_extract_track_from_engagement(tmp_path):
    frames = []
    # Pre-contact: neighbor far away.
    for i in range(0, 10):
        frames.append(
            {
                "frame_index": i,
                "target_bbox": [100, 100, 200, 300],
                "valid": True,
                "candidates": [
                    {"bbox": [100, 100, 200, 300], "selected": True, "interpolated": False},
                    {"bbox": [600, 100, 700, 300], "selected": False, "interpolated": False},
                ],
            }
        )
    # Contact window: overlapping defender.
    for i in range(10, 40):
        frames.append(
            {
                "frame_index": i,
                "target_bbox": [100, 100, 200, 300],
                "valid": True,
                "candidates": [
                    {"bbox": [100, 100, 200, 300], "selected": True, "interpolated": False},
                    {
                        "bbox": [140 + (i - 10), 110, 240 + (i - 10), 310],
                        "selected": False,
                        "interpolated": False,
                    },
                ],
            }
        )
    track = extract_opponent_track(
        _assoc(tmp_path, frames), min_run=10, seed_score=0.2, min_score=0.15, max_pre_contact=5
    )
    assert track is not None
    assert track.start >= 5  # not the whole pre-snap far neighbor
    assert track.end == 39
    assert len(track.frames) >= 10


def test_no_contact_returns_none(tmp_path):
    frames = [
        {
            "frame_index": i,
            "target_bbox": [100, 100, 200, 300],
            "valid": True,
            "candidates": [
                {"bbox": [100, 100, 200, 300], "selected": True, "interpolated": False},
                {"bbox": [900, 100, 1000, 300], "selected": False, "interpolated": False},
            ],
        }
        for i in range(20)
    ]
    assert extract_opponent_track(_assoc(tmp_path, frames), min_run=5, min_score=0.3) is None
