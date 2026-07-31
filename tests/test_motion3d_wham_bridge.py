"""Phase 2 tests — WSL bridge, path translation, segment selection, motion contract.

These run entirely on Windows with no WHAM present. Anything requiring the GPU
or the WHAM env belongs in the doctor preflight, not the test suite.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from oline_cv.motion3d.motion_schema import (
    MOTION_SCHEMA_VERSION,
    MotionMetadata,
    ReconstructionStatus,
    load_metadata,
    validate_motion_npz,
)
from oline_cv.motion3d.schema import TrackFrame, TrackManifest
from oline_cv.motion3d.segments import (
    Segment,
    describe_segments,
    contiguity_gaps,
    frames_in_segment,
    select_segment,
)
from oline_cv.motion3d.wham_bridge import (
    LOG_PREFIX,
    WhamConfig,
    build_job_args,
    build_remote_script,
    parse_event,
)
from oline_cv.motion3d.wsl_paths import (
    is_windows_path,
    quote_posix,
    windows_to_wsl,
    wsl_to_windows,
)


# --- job arguments ---------------------------------------------------------


def _args(**kw):
    cfg = WhamConfig(**kw)
    return build_job_args("/t/tracks.json", "/t/v.mp4", "/t/out", "/repo", (0, 164), cfg)


def test_association_is_on_by_default():
    args = _args()
    assert "--no-associate" not in args
    assert "--segment" in args and "0:164" in args


def test_association_can_be_disabled():
    assert "--no-associate" in _args(associate=False)


def test_association_thresholds_are_passed_through():
    args = _args(
        assoc_min_iou=0.5,
        assoc_max_center=0.3,
        assoc_min_score=0.6,
        min_frames=45,
        assoc_reject_ambiguous=True,
    )
    for flag, value in (
        ("--assoc-min-iou", "0.5"),
        ("--assoc-max-center", "0.3"),
        ("--assoc-min-score", "0.6"),
        ("--min-frames", "45"),
    ):
        assert args[args.index(flag) + 1] == value
    assert "--assoc-reject-ambiguous" in args


def test_unset_thresholds_are_omitted_so_runner_defaults_win():
    args = _args()
    for flag in ("--assoc-min-iou", "--assoc-max-center", "--assoc-min-score", "--min-frames"):
        assert flag not in args


def test_no_segment_requests_auto_selection():
    cfg = WhamConfig()
    args = build_job_args("/t/t.json", "/t/v.mp4", "/t/o", "/repo", None, cfg)
    assert "--auto-segment" in args
    assert "--segment" not in args


# --- path translation ------------------------------------------------------


@pytest.mark.parametrize(
    "win,posix",
    [
        (r"C:\Users\rishb\proj\a.mp4", "/mnt/c/Users/rishb/proj/a.mp4"),
        (r"D:\data\clip.mp4", "/mnt/d/data/clip.mp4"),
        (r"c:\lower\drive.txt", "/mnt/c/lower/drive.txt"),
        (r"C:\with space\a b.json", "/mnt/c/with space/a b.json"),
    ],
)
def test_windows_to_wsl(win, posix):
    assert windows_to_wsl(win) == posix


def test_posix_input_passes_through():
    assert windows_to_wsl("/home/rishul/WHAM") == "/home/rishul/WHAM"


def test_drive_root():
    assert windows_to_wsl("C:\\") == "/mnt/c"


def test_unc_paths_rejected():
    with pytest.raises(ValueError, match="UNC"):
        windows_to_wsl(r"\\server\share\file.mp4")


def test_round_trip_windows_wsl():
    win = r"C:\Users\rishb\OneDrive\Desktop\oline-cv\outputs\tracks.json"
    assert wsl_to_windows(windows_to_wsl(win)) == win


def test_is_windows_path():
    assert is_windows_path(r"C:\x")
    assert not is_windows_path("/mnt/c/x")


def test_quote_posix_escapes_single_quotes():
    assert quote_posix("/a/it's") == "'/a/it'\\''s'"


# --- remote command construction -------------------------------------------


def test_remote_script_activates_env_and_cds_to_wham():
    cfg = WhamConfig(conda_env="wham", wham_root="/home/rishul/WHAM")
    script = build_remote_script(cfg, "/mnt/c/repo/scripts/run_wham_manifest.py", ["--doctor"])
    assert "conda activate 'wham'" in script
    assert "cd '/home/rishul/WHAM'" in script
    assert "run_wham_manifest.py" in script
    assert "python -u" in script  # unbuffered, so progress streams live


def test_remote_script_quotes_path_arguments_with_spaces():
    cfg = WhamConfig()
    script = build_remote_script(
        cfg, "/mnt/c/r/run.py", ["--video", "/mnt/c/my films/a.mp4"]
    )
    assert "'/mnt/c/my films/a.mp4'" in script
    assert "--video" in script


def test_remote_script_has_distinct_exit_codes_for_env_failures():
    script = build_remote_script(WhamConfig(), "/mnt/c/r/run.py", [])
    assert "exit 78" in script  # conda activate failed
    assert "exit 79" in script  # wham root missing


# --- structured log protocol -----------------------------------------------


def test_parse_event_extracts_json_after_prefix():
    line = f'2026-01-01 INFO noise {LOG_PREFIX} {{"event": "progress", "percent": 42}}'
    ev = parse_event(line)
    assert ev == {"event": "progress", "percent": 42}


def test_parse_event_ignores_unrelated_and_malformed_lines():
    assert parse_event("Preprocess: ####  50%") is None
    assert parse_event(f"{LOG_PREFIX} not-json") is None


# --- segment selection -----------------------------------------------------


def _manifest(segments, n=200, missing_range=None):
    frames = []
    for i in range(n):
        inside = any(a <= i <= b for a, b in segments)
        missing = missing_range and missing_range[0] <= i <= missing_range[1]
        frames.append(
            TrackFrame(
                frame_index=i,
                timestamp=i / 30.0,
                track_id=6,
                target_id=1,
                bbox=None if (missing or not inside) else [0, 0, 10, 10],
                bbox_source="detected" if inside and not missing else "missing",
                detection_confidence=0.9,
                track_state="TRACKED" if inside else "LOST",
                track_confidence=0.8,
                keypoints_2d=[[1.0, 2.0, 0.9]] * 17,
                low_confidence=False,
                usable=inside and not missing,
            )
        )
    return TrackManifest(
        video={"fps": 30.0, "frame_count": n, "width": 1920, "height": 1080},
        target={"target_id": 1},
        export={},
        frames=frames,
        segments=[list(s) for s in segments],
    )


def test_auto_select_prefers_segment_with_most_detections():
    m = _manifest([(0, 164), (192, 199)])
    seg = select_segment(m)
    assert (seg.start, seg.end) == (0, 164)
    assert seg.frames == 165


def test_explicit_segment_inside_a_valid_range_is_accepted():
    m = _manifest([(0, 164)])
    seg = select_segment(m, explicit=(0, 164))
    assert seg.frames == 165
    assert seg.detected == 165


def test_explicit_subrange_is_accepted():
    m = _manifest([(0, 164)])
    seg = select_segment(m, explicit=(20, 120))
    assert (seg.start, seg.end, seg.frames) == (20, 120, 101)


def test_explicit_segment_outside_valid_range_is_refused():
    m = _manifest([(0, 164)])
    with pytest.raises(ValueError, match="not inside a reconstructable"):
        select_segment(m, explicit=(150, 250))


def test_short_segments_are_ignored():
    m = _manifest([(0, 10)])
    with pytest.raises(ValueError, match="no reconstructable segment"):
        select_segment(m)


def test_too_short_explicit_range_is_refused():
    m = _manifest([(0, 164)])
    with pytest.raises(ValueError, match="below the"):
        select_segment(m, explicit=(0, 5))


def test_describe_segments_scores_detection_ratio():
    m = _manifest([(0, 99)], n=100)
    segs = describe_segments(m)
    assert len(segs) == 1
    assert segs[0].detected_ratio == pytest.approx(1.0)
    assert segs[0].usable_ratio == pytest.approx(1.0)


def test_frames_in_segment_excludes_holes_and_sorts():
    m = _manifest([(0, 164)], missing_range=(50, 52))
    seg = Segment(start=0, end=164, frames=165, detected=162, interpolated=0, usable=162)
    frames = frames_in_segment(m, seg)
    assert len(frames) == 162
    assert [f.frame_index for f in frames] == sorted(f.frame_index for f in frames)
    assert contiguity_gaps(frames) == [(49, 53)]


def test_segment_contains_operator():
    seg = Segment(start=0, end=164, frames=165, detected=165, interpolated=0, usable=165)
    assert 0 in seg and 164 in seg and 165 not in seg


# --- motion metadata / npz contract ----------------------------------------


def _meta(**kw):
    base = dict(
        status=ReconstructionStatus.OK,
        video="footage.mp4",
        fps=30.0,
        segment={"start": 0, "end": 164},
        frame_range=[0, 164],
        frame_count=165,
        runtime_seconds=42.5,
        wham={"world_grounded": True},
    )
    base.update(kw)
    return MotionMetadata(**base)


def test_metadata_round_trip(tmp_path):
    p = _meta(warnings=["w1"]).save(tmp_path / "motion_metadata.json")
    back = load_metadata(p)
    assert back.status == ReconstructionStatus.OK
    assert back.frame_count == 165
    assert back.world_grounded is True
    assert back.warnings == ["w1"]
    assert back.ok


def test_local_only_status_is_ok_but_not_world_grounded():
    m = _meta(status=ReconstructionStatus.OK_LOCAL_ONLY, wham={"world_grounded": False})
    assert m.ok
    assert not m.world_grounded


def test_failed_status_is_not_ok():
    assert not _meta(status=ReconstructionStatus.FAILED).ok


def test_metadata_schema_mismatch_rejected(tmp_path):
    p = tmp_path / "motion_metadata.json"
    p.write_text(json.dumps({"schema_version": "9.0.0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        load_metadata(p)


def _write_npz(path, n=165, **overrides):
    arrays = {
        "frame_indices": np.arange(n, dtype=np.int32),
        "timestamps": (np.arange(n) / 30.0).astype(np.float32),
        "pose_cam": np.zeros((n, 72), np.float32),
        "pose_world": np.zeros((n, 72), np.float32),
        "betas": np.zeros((n, 10), np.float32),
        "trans_cam": np.zeros((n, 3), np.float32),
        "trans_world": np.zeros((n, 3), np.float32),
        "contact": np.zeros((n, 4), np.float32),
    }
    arrays.update(overrides)
    np.savez_compressed(path, **arrays)
    return path


def test_validate_npz_accepts_well_formed(tmp_path):
    p = _write_npz(tmp_path / "motion_raw.npz")
    summary = validate_motion_npz(p, expected_frames=165)
    assert summary["frames"] == 165
    assert summary["frame_range"] == [0, 164]


def test_validate_npz_rejects_missing_array(tmp_path):
    p = tmp_path / "m.npz"
    np.savez_compressed(p, frame_indices=np.arange(5, dtype=np.int32))
    with pytest.raises(ValueError, match="missing required arrays"):
        validate_motion_npz(p)


def test_validate_npz_rejects_wrong_pose_width(tmp_path):
    p = _write_npz(tmp_path / "m.npz", pose_world=np.zeros((165, 69), np.float32))
    with pytest.raises(ValueError, match="trailing shape"):
        validate_motion_npz(p)


def test_validate_npz_rejects_frame_count_mismatch(tmp_path):
    p = _write_npz(tmp_path / "m.npz", n=100)
    with pytest.raises(ValueError, match="expected 165"):
        validate_motion_npz(p, expected_frames=165)


def test_validate_npz_rejects_non_monotonic_frame_indices(tmp_path):
    idx = np.arange(165, dtype=np.int32)
    idx[10] = 200
    p = _write_npz(tmp_path / "m.npz", frame_indices=idx)
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_motion_npz(p)


def test_validate_npz_rejects_nan_world_trajectory(tmp_path):
    bad = np.zeros((165, 3), np.float32)
    bad[5, 1] = np.nan
    p = _write_npz(tmp_path / "m.npz", trans_world=bad)
    with pytest.raises(ValueError, match="non-finite"):
        validate_motion_npz(p)


def test_validate_npz_reports_optional_arrays(tmp_path):
    p = _write_npz(tmp_path / "m.npz", joints_cam=np.zeros((165, 17, 3), np.float32))
    summary = validate_motion_npz(p)
    assert "joints_cam" in summary["optional_present"]


def test_schema_version_is_pinned():
    assert MOTION_SCHEMA_VERSION == "1.0.0"
