"""Two-view fusion core: coverage math + per-metric view selection.

Builds two synthetic single-view results where each camera sees a
complementary half of the rep (the classic occlusion case: the lineman
leaves the sideline frame at contact, but the endzone still has him). The
fusion should recover the union coverage and pull each metric from the view
that actually saw it.
"""

from oline_cv.multiview import ViewArtifacts, fuse_view_results
from oline_cv.trust import compute_trust

FPS = 30.0
STEP = 1000.0 / FPS
N = 20


def _module(available=True, **kw):
    return {"available": available, "notes": [], "coach_flags": [], **kw}


def _frames(covered: set[int]):
    return [
        {
            "frame_idx": i,
            "timestamp_ms": i * STEP,
            "low_confidence": i not in covered,
            "keypoints": {},
        }
        for i in range(N)
    ]


def _sideline():
    result = {
        "video": {"path": "sideline.mp4", "fps": FPS, "frame_count": N, "width": 1920, "height": 1080},
        "play_type": "pass",
        "target_jersey": 76,
        "ol_lock": {"frames_lost": 10, "method": "manual_pick_xy"},
        "snap": {"snap_frame": 0},
        "set": {"start_frame": 0, "end_frame": N - 1},
        "frames": _frames(set(range(0, 10))),  # sideline covers the first half
        "modules": {
            "initial_quicks": _module(
                reaction_time_ms=200, reaction_time_frames=6,
                initiated_by="foot", late_off_the_ball=False,
                coach_flags=["foot_quickness"],
            ),
            "footwork": _module(step_cadence_hz=3.2, set_depth=1.0, set_width=0.3, mean_base_width=0.3, overset=False),
            "mirror_redirect": _module(available=False),
            "anchor": _module(available=False, notes=["lost_contact"]),
            "body_position": _module(
                posture_confidence=0.85, posture_classification="balanced", posture_mixed=False,
                mean_knee_flexion_deg=150, min_knee_flexion_deg=140, mean_torso_angle_deg=25,
                hip_height_at_lowest=0.6, mean_hip_height=0.65,
            ),
            "balance": _module(com_jitter=0.02),
            "hands": _module(available=False),
            "sustain": _module(available=False),
            "point_of_attack": _module(available=False, notes=["skipped_pass_play"]),
            "movement_in_space": _module(available=False, notes=["skipped_pass_play"]),
        },
        "rep_summary": {"target_jersey": 76, "ol_lock": {"method": "manual_pick_xy"}, "play_type": "pass", "defender_tracked": False},
    }
    result["trust"] = compute_trust(result)
    return result


def _endzone():
    result = {
        "video": {"path": "endzone.mp4", "fps": FPS, "frame_count": N, "width": 1920, "height": 1080},
        "play_type": "pass",
        "target_jersey": 76,
        "ol_lock": {"frames_lost": 10, "method": "manual_pick_xy"},
        "snap": {"snap_frame": 0},
        "set": {"start_frame": 0, "end_frame": N - 1},
        "frames": _frames(set(range(10, 20))),  # endzone covers the second half
        "modules": {
            "initial_quicks": _module(available=False),
            "footwork": _module(step_cadence_hz=3.0, set_depth=1.2, set_width=0.35, mean_base_width=0.35, overset=True, coach_flags=["overset"]),
            "mirror_redirect": _module(lateral_match_correlation=0.8),
            "anchor": _module(max_hip_displacement_after_contact=0.1, contact_detected=True),
            "body_position": _module(
                posture_confidence=0.6, posture_classification="balanced", posture_mixed=False,
                mean_knee_flexion_deg=148, min_knee_flexion_deg=138, mean_torso_angle_deg=24,
                hip_height_at_lowest=0.58, mean_hip_height=0.63,
            ),
            "balance": _module(com_jitter=0.03),
            "hands": _module(hand_visibility_confidence=0.8, time_to_first_contact_ms=250),
            "sustain": _module(contacted=True, engagement_ms=300),
            "point_of_attack": _module(available=False, notes=["skipped_pass_play"]),
            "movement_in_space": _module(available=False, notes=["skipped_pass_play"]),
        },
        "rep_summary": {"target_jersey": 76, "ol_lock": {"method": "manual_pick_xy"}, "play_type": "pass", "defender_tracked": True},
    }
    result["trust"] = compute_trust(result)
    return result


def _fuse():
    views = [
        ViewArtifacts(role="sideline", result=_sideline()),
        ViewArtifacts(role="endzone", result=_endzone()),
    ]
    return fuse_view_results(views, primary_role="sideline"), views


def test_combined_coverage_recovers_occlusion():
    fused, _ = _fuse()
    cov = fused["multiview"]["coverage"]
    # Each camera alone saw half the rep; together they cover all of it.
    assert cov["best_single_coverage"] == 0.5
    assert cov["combined_coverage"] == 1.0
    assert cov["occlusion_reduction"] == 0.5


def test_modules_taken_from_the_view_that_saw_them():
    fused, _ = _fuse()
    src = fused["multiview"]["module_sources"]
    # Contact-phase reads come from the endzone (sideline lost the player there).
    assert src["anchor"] == "endzone"
    assert src["hands"] == "endzone"
    assert src["sustain"] == "endzone"
    # Get-off and pad level come from the sideline (endzone missed the first half).
    assert src["initial_quicks"] == "sideline"
    assert src["body_position"] == "sideline"


def test_rep_summary_reflects_selected_views():
    fused, _ = _fuse()
    s = fused["rep_summary"]
    assert s["reaction_time_ms"] == 200          # sideline get-off
    assert s["anchor_give"] == 0.1               # endzone anchor
    assert s["punch_ms"] == 250                  # endzone hands
    assert s["lateral_match"] == 0.8             # endzone mirror


def test_fused_trust_beats_single_view_because_coverage_is_higher():
    fused, views = _fuse()
    endzone_anchor = views[1].result["trust"]["modules"]["anchor"]["score"]
    fused_anchor = fused["trust"]["modules"]["anchor"]["score"]
    # Same underlying anchor read, but fused coverage removes the sparse-pose
    # penalty the endzone carried on its own.
    assert fused_anchor > endzone_anchor


def test_single_view_passthrough():
    only = ViewArtifacts(role="sideline", result=_sideline())
    fused = fuse_view_results([only], primary_role="sideline")
    assert "multiview" not in fused  # nothing to fuse


# ── alignment validation ──────────────────────────────────────────────

def test_alignment_same_play_different_start_times():
    """Two clips of the same play that started recording at different times.

    The sideline camera began filming earlier (snap at frame 90 = 3.0s into
    the clip) while the endzone started later (snap at frame 15 = 0.5s).
    Both reps last ~20 frames after snap, so durations match and alignment
    should report high confidence.
    """
    side = _sideline()
    side["snap"]["snap_frame"] = 90
    side["set"]["end_frame"] = 110
    side["video"]["frame_count"] = 120
    side["frames"] = [
        {"frame_idx": i, "timestamp_ms": i * STEP, "low_confidence": False, "keypoints": {}}
        for i in range(90, 110)
    ]
    side["trust"] = compute_trust(side)

    end = _endzone()
    end["snap"]["snap_frame"] = 15
    end["set"]["end_frame"] = 35
    end["video"]["frame_count"] = 40
    end["frames"] = [
        {"frame_idx": i, "timestamp_ms": i * STEP, "low_confidence": False, "keypoints": {}}
        for i in range(15, 35)
    ]
    end["trust"] = compute_trust(end)

    views = [
        ViewArtifacts(role="sideline", result=side),
        ViewArtifacts(role="endzone", result=end),
    ]
    fused = fuse_view_results(views, primary_role="sideline")
    align = fused["multiview"]["alignment"]
    assert align["same_play_confidence"] == "high"
    assert len(align["warnings"]) == 0
    assert align["timeline_overlap_frac"] >= 0.95
    # Per-view snap timing surfaced for the user.
    side_view = next(v for v in fused["multiview"]["views"] if v["role"] == "sideline")
    end_view = next(v for v in fused["multiview"]["views"] if v["role"] == "endzone")
    assert side_view["snap_at_s"] == 3.0
    assert end_view["snap_at_s"] == 0.5


def test_alignment_different_plays_detected():
    """Clips from two different plays: the rep durations are very different."""
    side = _sideline()
    side["set"]["end_frame"] = 100
    side["snap"]["snap_frame"] = 0
    side["video"]["frame_count"] = 110
    side["frames"] = [
        {"frame_idx": i, "timestamp_ms": i * STEP, "low_confidence": False, "keypoints": {}}
        for i in range(0, 100)
    ]
    side["trust"] = compute_trust(side)

    end = _endzone()
    end["set"]["end_frame"] = 10
    end["snap"]["snap_frame"] = 0
    end["video"]["frame_count"] = 15
    end["frames"] = [
        {"frame_idx": i, "timestamp_ms": i * STEP, "low_confidence": False, "keypoints": {}}
        for i in range(0, 10)
    ]
    end["trust"] = compute_trust(end)

    views = [
        ViewArtifacts(role="sideline", result=side),
        ViewArtifacts(role="endzone", result=end),
    ]
    fused = fuse_view_results(views, primary_role="sideline")
    align = fused["multiview"]["alignment"]
    # 100 frames vs 10 frames at 30fps → 3.33s vs 0.33s → delta > 2s
    assert align["same_play_confidence"] in ("low", "medium")
    assert any("differ" in w or "overlap" in w for w in align["warnings"])


def test_alignment_low_snap_confidence_warns():
    """One view with low snap confidence → medium confidence + warning."""
    side = _sideline()
    side["snap"]["confidence"] = "low"
    side["trust"] = compute_trust(side)

    end = _endzone()
    end["trust"] = compute_trust(end)

    views = [
        ViewArtifacts(role="sideline", result=side),
        ViewArtifacts(role="endzone", result=end),
    ]
    fused = fuse_view_results(views, primary_role="sideline")
    align = fused["multiview"]["alignment"]
    assert align["same_play_confidence"] in ("medium", "low")
    assert any("uncertain" in w.lower() or "snap" in w.lower() for w in align["warnings"])


def test_alignment_different_fps_warns():
    """Views at different frame rates → informational warning."""
    side = _sideline()
    side["video"]["fps"] = 30.0
    side["trust"] = compute_trust(side)

    end = _endzone()
    end["video"]["fps"] = 60.0
    end["trust"] = compute_trust(end)

    views = [
        ViewArtifacts(role="sideline", result=side),
        ViewArtifacts(role="endzone", result=end),
    ]
    fused = fuse_view_results(views, primary_role="sideline")
    align = fused["multiview"]["alignment"]
    assert any("frame rate" in w.lower() for w in align["warnings"])
