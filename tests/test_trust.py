"""Tests for trust scoring."""

from oline_cv.trust import compute_trust


def test_trust_overall_present():
    result = {
        "video": {"fps": 30},
        "play_type": "pass",
        "ol_lock": {"track_switch_rejects": 2},
        "rep_summary": {"defender_tracked": True},
        "modules": {
            "initial_quicks": {
                "available": True,
                "reaction_time_frames": 6,
                "notes": [],
                "coach_flags": ["initial_quicks"],
            },
            "body_position": {
                "available": True,
                "posture_confidence": 0.8,
                "posture_mixed": False,
                "posture_classification": "balanced",
                "valid_frame_count": 50,
                "flagged_frame_count": 0,
            },
            "hands": {
                "available": True,
                "hand_visibility_confidence": 0.2,
                "notes": ["hand_placement_approximate_v1"],
                "coach_flags": ["hands_occluded"],
            },
            "footwork": {"available": True, "step_cadence_hz": 3.2, "notes": []},
            "mirror_redirect": {"available": True, "notes": []},
            "anchor": {"available": True, "notes": []},
            "sustain": {"available": True, "contacted": True, "notes": []},
            "balance": {"available": True, "com_jitter": 0.02, "notes": []},
            "point_of_attack": {"available": False, "notes": ["skipped_pass_play"]},
            "movement_in_space": {"available": False, "notes": ["skipped_pass_play"]},
        },
        "frames": [{"low_confidence": False}] * 10,
    }
    trust = compute_trust(result)
    assert "overall" in trust and "modules" in trust
    assert 0 <= trust["overall"]["score"] <= 1
    assert trust["modules"]["hands"]["level"] == "low"
    assert trust["modules"]["body_position"]["level"] in ("high", "medium")
    assert trust["modules"]["point_of_attack"]["score"] == 0.0
