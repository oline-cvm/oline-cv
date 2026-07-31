"""Phase 1 tests — track/crop export contract.

These cover the pure geometry and gap-handling logic plus a full round-trip on
a synthetic clip, so a schema or coordinate regression fails here rather than
silently corrupting the 3D reconstruction downstream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from oline_cv.motion3d.schema import (
    SCHEMA_VERSION,
    BboxSource,
    CropRef,
    TrackManifest,
    load_manifest,
)
from oline_cv.motion3d.track_export import (
    export_tracks,
    fill_bbox_gaps,
    render_crop,
    square_crop_box,
    valid_segments,
)


@dataclass
class FakePose:
    frame_idx: int
    timestamp_ms: float
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    bbox_xyxy: np.ndarray | None
    person_confidence: float = 0.9
    low_confidence: bool = False
    usable: bool = True
    track_state: str = "TRACKED"
    track_confidence: float = 0.8
    track_id: int | None = 6
    target_id: int = 1


def make_poses(n=12, missing=(), fps=30.0, w=640, h=480):
    poses = []
    for i in range(n):
        cx = 100.0 + 5.0 * i
        cy = 240.0
        kp = np.full((17, 2), np.nan)
        conf = np.zeros(17)
        for j in (5, 6, 11, 12, 15, 16):
            kp[j] = [cx + j, cy + j]
            conf[j] = 0.8
        bbox = None if i in missing else np.array([cx - 40, cy - 90, cx + 40, cy + 90])
        poses.append(
            FakePose(
                frame_idx=i,
                timestamp_ms=(i / fps) * 1000.0,
                keypoints_xy=kp,
                keypoints_conf=conf,
                bbox_xyxy=bbox,
                usable=i not in missing,
                track_state="TRACKED" if i not in missing else "LOST",
            )
        )
    return poses


def make_frames(n=12, w=640, h=480):
    return [np.full((h, w, 3), 60, dtype=np.uint8) for _ in range(n)]


# --- crop geometry ---------------------------------------------------------


def test_square_crop_box_is_square_and_centered():
    box = square_crop_box([100, 200, 140, 380], pad=0.0)
    width = box[2] - box[0]
    height = box[3] - box[1]
    assert width == pytest.approx(height)
    assert width == pytest.approx(180.0)  # long side of the bbox
    assert (box[0] + box[2]) / 2 == pytest.approx(120.0)
    assert (box[1] + box[3]) / 2 == pytest.approx(290.0)


def test_square_crop_box_pad_expands_both_sides():
    box = square_crop_box([0, 0, 100, 100], pad=0.25)
    assert box[2] - box[0] == pytest.approx(150.0)
    assert box[0] == pytest.approx(-25.0)  # deliberately not clamped to frame


def test_crop_box_stays_square_off_frame_edge():
    """Clamping to the frame would break the single-scalar scale contract."""
    box = square_crop_box([-30, -50, 10, 30], pad=0.1)
    assert (box[2] - box[0]) == pytest.approx(box[3] - box[1])


def test_render_crop_letterboxes_and_reports_exact_scale():
    frame = np.full((480, 640, 3), 200, dtype=np.uint8)
    box = [-50.0, -50.0, 50.0, 50.0]
    crop, scale = render_crop(frame, box, size=100)
    assert crop.shape == (100, 100, 3)
    assert scale == pytest.approx(1.0)
    assert crop[10, 10].max() == 0  # out-of-frame region is black
    assert crop[75, 75].max() > 0  # in-frame region carries pixels


def test_keypoint_round_trip_through_crop_transform():
    box = square_crop_box([100, 200, 180, 400], pad=0.2)
    _, scale = render_crop(np.zeros((480, 640, 3), np.uint8), box, size=256)
    ref = CropRef(path="crops/000000.jpg", box=box, size=256, scale=scale)
    for x, y in [(120.0, 260.0), (175.0, 395.0), (100.0, 200.0)]:
        cx, cy = ref.image_to_crop(x, y)
        bx, by = ref.crop_to_image(cx, cy)
        assert bx == pytest.approx(x, abs=1e-6)
        assert by == pytest.approx(y, abs=1e-6)


# --- gap handling ----------------------------------------------------------


def test_short_gap_is_linearly_interpolated():
    boxes = [[0, 0, 10, 10], None, None, [30, 30, 40, 40]]
    filled, sources = fill_bbox_gaps(boxes, max_interp_gap=8)
    assert sources == [
        BboxSource.DETECTED.value,
        BboxSource.INTERPOLATED.value,
        BboxSource.INTERPOLATED.value,
        BboxSource.DETECTED.value,
    ]
    assert filled[1][0] == pytest.approx(10.0)
    assert filled[2][0] == pytest.approx(20.0)


def test_long_gap_is_left_missing_not_invented():
    boxes = [[0, 0, 10, 10]] + [None] * 20 + [[50, 50, 60, 60]]
    filled, sources = fill_bbox_gaps(boxes, max_interp_gap=5)
    assert sources[10] == BboxSource.MISSING.value
    assert filled[10] is None


def test_head_and_tail_gaps_carry_nearest_but_are_bounded():
    boxes = [None, None, [0, 0, 10, 10], None, None]
    filled, sources = fill_bbox_gaps(boxes, max_interp_gap=1)
    assert sources[1] == BboxSource.CARRIED.value
    assert sources[0] == BboxSource.MISSING.value  # beyond the carry budget
    assert sources[3] == BboxSource.CARRIED.value
    assert filled[1] == [0, 0, 10, 10]


def test_all_missing_yields_no_segments():
    filled, sources = fill_bbox_gaps([None, None, None])
    assert all(f is None for f in filled)
    assert valid_segments(sources) == []


def test_valid_segments_splits_on_missing():
    sources = ["detected"] * 3 + ["missing"] * 2 + ["detected"] * 4
    assert valid_segments(sources) == [[0, 2], [5, 8]]


# --- full export round trip ------------------------------------------------


def test_export_round_trip_and_schema(tmp_path):
    poses = make_poses(n=12, missing=(5,))
    frames = make_frames(n=12)
    manifest = export_tracks(
        "fake.mp4", poses, frames, 30.0, 640, 480, tmp_path,
        target_jersey=76, crop_size=128, save_full_frames=True,
    )

    reloaded = load_manifest(tmp_path / "tracks.json")
    assert reloaded.schema_version == SCHEMA_VERSION
    assert len(reloaded.frames) == 12
    assert reloaded.video["fps"] == 30.0
    assert reloaded.target["jersey"] == 76
    assert reloaded.export["keypoint_format"] == "coco17"

    # single-frame dropout is interpolated, so every frame stays reconstructable
    assert reloaded.stats()["reconstructable"] == 12
    assert reloaded.frame_by_index(5).bbox_source == BboxSource.INTERPOLATED.value
    assert reloaded.frame_by_index(5).usable is False

    for fr in reloaded.frames:
        assert len(fr.keypoints_2d) == 17
        assert (tmp_path / fr.crop.path).exists()
        assert (tmp_path / fr.frame_path).exists()


def test_export_preserves_botsort_identity_fields(tmp_path):
    poses = make_poses(n=4)
    poses[2].track_id = 9  # BoT-SORT re-id; logical target must not change
    manifest = export_tracks("fake.mp4", poses, make_frames(4), 30.0, 640, 480, tmp_path)
    ids = [f.track_id for f in manifest.frames]
    assert ids == [6, 6, 9, 6]
    assert {f.target_id for f in manifest.frames} == {1}


def test_export_without_full_frames_still_writes_crops(tmp_path):
    manifest = export_tracks(
        "fake.mp4", make_poses(4), make_frames(4), 30.0, 640, 480, tmp_path,
        save_full_frames=False,
    )
    assert not (tmp_path / "frames").exists()
    for fr in manifest.frames:
        assert fr.frame_path is None
        assert (tmp_path / fr.crop.path).exists()


def test_timestamps_are_seconds_and_monotonic(tmp_path):
    manifest = export_tracks("fake.mp4", make_poses(6), make_frames(6), 30.0, 640, 480, tmp_path)
    ts = [f.timestamp for f in manifest.frames]
    assert ts[0] == pytest.approx(0.0)
    assert ts[1] == pytest.approx(1 / 30.0, abs=1e-6)
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_incompatible_schema_version_is_rejected(tmp_path):
    p = tmp_path / "tracks.json"
    p.write_text(json.dumps({"schema_version": "99.0.0", "frames": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        load_manifest(p)


def test_missing_keypoints_serialize_as_null_not_nan(tmp_path):
    poses = make_poses(2)
    manifest = export_tracks("fake.mp4", poses, make_frames(2), 30.0, 640, 480, tmp_path)
    raw = (tmp_path / "tracks.json").read_text(encoding="utf-8")
    assert "NaN" not in raw  # NaN is invalid JSON and breaks the WSL-side reader
    nose = manifest.frames[0].keypoints_2d[0]
    assert nose[0] is None and nose[2] == 0.0
