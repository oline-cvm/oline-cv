"""Benchmark schema, context enums, and feature-definition registry.

Units match what analyze_video() already writes — nothing is rescaled here.
Spatial values in this repo are fractions of standing height unless noted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SCHEMA_VERSION = "1.0"

# Conservative default: skip trust.overall.level == "low" (score < 0.45).
DEFAULT_MIN_TRUST_SCORE = 0.45

# Need this many valid observations before range / z-score statuses are assigned.
MIN_COMPARE_N = 3
STD_EPS = 1e-9

Position = Literal["LT", "LG", "C", "RG", "RT", "unknown"]
PlayType = Literal["pass", "run"]
Side = Literal["left", "right", "interior", "unknown"]

POSITIONS: tuple[str, ...] = ("LT", "LG", "C", "RG", "RT", "unknown")
PLAY_TYPES: tuple[str, ...] = ("pass", "run")
SIDES: tuple[str, ...] = ("left", "right", "interior", "unknown")

STATUSES: tuple[str, ...] = (
    "within_reference_range",
    "above_reference_range",
    "below_reference_range",
    "insufficient_reference_data",
    "missing_player_data",
)

STAT_KEYS: tuple[str, ...] = (
    "count",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "p10",
    "p25",
    "p75",
    "p90",
)


@dataclass(frozen=True)
class FeatureDef:
    key: str
    display_name: str
    unit: str
    description: str
    category: str
    # Lookup paths tried in order; first present numeric value wins.
    sources: tuple[tuple[str, ...], ...]


# Canonical numeric features already produced by the pipeline.
FEATURE_DEFS: tuple[FeatureDef, ...] = (
    FeatureDef(
        "reaction_time_ms",
        "Reaction Time",
        "ms",
        "Time from snap to first monotonic foot or hip movement.",
        "initial_quicks",
        (("rep_summary", "reaction_time_ms"), ("modules", "initial_quicks", "reaction_time_ms")),
    ),
    FeatureDef(
        "reaction_time_frames",
        "Reaction Time (frames)",
        "frames",
        "Reaction time in source-video frames.",
        "initial_quicks",
        (("rep_summary", "reaction_time_frames"), ("modules", "initial_quicks", "reaction_time_frames")),
    ),
    FeatureDef(
        "first_step_acceleration",
        "First-Step Acceleration",
        "standing_heights_per_s2",
        "Ankle-path acceleration along the first step, divided by standing height.",
        "initial_quicks",
        (("modules", "initial_quicks", "first_step_acceleration"),),
    ),
    FeatureDef(
        "first_step_direction_deg",
        "First-Step Direction",
        "degrees",
        "First-step heading; 0 is image upfield (negative y).",
        "initial_quicks",
        (("modules", "initial_quicks", "first_step_direction_deg"),),
    ),
    FeatureDef(
        "mean_knee_flexion_deg",
        "Knee Flexion",
        "degrees",
        "Mean interior hip–knee–ankle angle (180° is a straight leg).",
        "body_position",
        (("rep_summary", "mean_knee_flexion_deg"), ("modules", "body_position", "mean_knee_flexion_deg")),
    ),
    FeatureDef(
        "min_knee_flexion_deg",
        "Knee Flexion (deepest)",
        "degrees",
        "Minimum interior knee angle in the set window.",
        "body_position",
        (("rep_summary", "min_knee_flexion_deg"), ("modules", "body_position", "min_knee_flexion_deg")),
    ),
    FeatureDef(
        "mean_torso_angle_deg",
        "Torso Lean",
        "degrees",
        "Mean angle of the hip-to-shoulder vector from vertical.",
        "body_position",
        (("rep_summary", "mean_torso_angle_deg"), ("modules", "body_position", "mean_torso_angle_deg")),
    ),
    FeatureDef(
        "max_torso_angle_deg",
        "Torso Lean (max)",
        "degrees",
        "Maximum torso lean from vertical in the set window.",
        "body_position",
        (("modules", "body_position", "max_torso_angle_deg"),),
    ),
    FeatureDef(
        "mean_hip_height",
        "Hip Height (mean)",
        "standing_height_frac",
        "Mean hip height above the feet, divided by standing height.",
        "body_position",
        (("rep_summary", "mean_hip_height"), ("modules", "body_position", "mean_hip_height")),
    ),
    FeatureDef(
        "hip_height_at_lowest",
        "Pad Level (lowest hips)",
        "standing_height_frac",
        "Lowest hip height in the set, divided by standing height.",
        "body_position",
        (("rep_summary", "hip_height_at_lowest"), ("modules", "body_position", "hip_height_at_lowest")),
    ),
    FeatureDef(
        "posture_confidence",
        "Posture Confidence",
        "fraction",
        "Weighted plurality fraction of the posture label (0–1).",
        "body_position",
        (("rep_summary", "posture_confidence"), ("modules", "body_position", "posture_confidence")),
    ),
    FeatureDef(
        "step_cadence_hz",
        "Step Cadence",
        "Hz",
        "Detected ankle-speed peaks per second over the set.",
        "footwork",
        (("rep_summary", "step_cadence_hz"), ("modules", "footwork", "step_cadence_hz")),
    ),
    FeatureDef(
        "mean_step_length",
        "Step Length",
        "standing_height_frac",
        "Mean ankle-mid travel between consecutive steps, divided by standing height.",
        "footwork",
        (("modules", "footwork", "mean_step_length"),),
    ),
    FeatureDef(
        "second_step_gain",
        "Second-Step Gain",
        "standing_height_frac",
        "Second-step displacement along the set's principal travel axis / standing height.",
        "footwork",
        (("modules", "footwork", "second_step_gain"),),
    ),
    FeatureDef(
        "set_depth",
        "Set Depth",
        "standing_height_frac",
        "Max early-set hip travel along the principal axis, divided by standing height.",
        "footwork",
        (("rep_summary", "set_depth"), ("modules", "footwork", "set_depth")),
    ),
    FeatureDef(
        "set_width",
        "Set Width",
        "standing_height_frac",
        "Max early-set |lateral| hip travel, divided by standing height.",
        "footwork",
        (("rep_summary", "set_width"), ("modules", "footwork", "set_width")),
    ),
    FeatureDef(
        "mean_base_width",
        "Base Width",
        "standing_height_frac",
        "Mean ankle-to-ankle distance divided by standing height.",
        "footwork",
        (("rep_summary", "mean_base_width"), ("modules", "footwork", "mean_base_width")),
    ),
    FeatureDef(
        "min_base_width",
        "Base Width (narrowest)",
        "standing_height_frac",
        "Minimum ankle-to-ankle distance divided by standing height.",
        "footwork",
        (("modules", "footwork", "min_base_width"),),
    ),
    FeatureDef(
        "mean_lateral_velocity",
        "Lateral Speed (mean)",
        "standing_heights_per_s",
        "Mean |lateral hip velocity| divided by standing height.",
        "footwork",
        (("modules", "footwork", "mean_lateral_velocity"),),
    ),
    FeatureDef(
        "peak_lateral_velocity",
        "Lateral Speed (peak)",
        "standing_heights_per_s",
        "Peak |lateral hip velocity| divided by standing height.",
        "footwork",
        (("modules", "footwork", "peak_lateral_velocity"),),
    ),
    FeatureDef(
        "feet_close_to_ground_score",
        "Feet-to-Ground Score",
        "fraction",
        "1 minus normalized ankle-y variance (higher = less bounce).",
        "footwork",
        (("modules", "footwork", "feet_close_to_ground_score"),),
    ),
    FeatureDef(
        "lateral_match",
        "Lateral Match",
        "correlation",
        "Pearson correlation of OL vs defender lateral hip velocities.",
        "engagement",
        (
            ("rep_summary", "lateral_match"),
            ("modules", "mirror_redirect", "lateral_match_correlation"),
        ),
    ),
    FeatureDef(
        "mean_separation",
        "Separation (mean)",
        "standing_height_frac",
        "Mean hip-to-hip distance to the defender, divided by standing height.",
        "engagement",
        (("modules", "mirror_redirect", "mean_separation"),),
    ),
    FeatureDef(
        "min_separation",
        "Separation (min)",
        "standing_height_frac",
        "Closest hip-to-hip distance to the defender, divided by standing height.",
        "engagement",
        (("modules", "mirror_redirect", "min_separation"),),
    ),
    FeatureDef(
        "recovery_ms_after_redirect",
        "Redirect Recovery",
        "ms",
        "Time for the OL to match the defender's new lateral direction after a cut.",
        "engagement",
        (("modules", "mirror_redirect", "recovery_ms_after_redirect"),),
    ),
    FeatureDef(
        "anchor_give",
        "Anchor Give",
        "standing_height_frac",
        "Max hip displacement in the first ~0.8s after contact, divided by standing height.",
        "engagement",
        (
            ("rep_summary", "anchor_give"),
            ("modules", "anchor", "max_hip_displacement_after_contact"),
        ),
    ),
    FeatureDef(
        "max_torso_displacement_after_contact",
        "Torso Give",
        "standing_height_frac",
        "Max shoulder displacement after contact, divided by standing height.",
        "engagement",
        (("modules", "anchor", "max_torso_displacement_after_contact"),),
    ),
    FeatureDef(
        "punch_ms",
        "Punch Timing",
        "ms",
        "Time from snap to first wrist-near-defender contact (approximate; wrists often occluded).",
        "engagement",
        (("rep_summary", "punch_ms"), ("modules", "hands", "time_to_first_contact_ms")),
    ),
    FeatureDef(
        "approximate_hand_placement_score",
        "Hand Placement Score",
        "fraction",
        "Fraction of visible punch frames with the wrist inside the defender torso (V1 approximate).",
        "engagement",
        (("modules", "hands", "approximate_hand_placement_score"),),
    ),
    FeatureDef(
        "engagement_ms",
        "Engagement Duration",
        "ms",
        "Longest run of close hip-to-hip separation, in milliseconds.",
        "engagement",
        (("rep_summary", "engagement_ms"), ("modules", "sustain", "engagement_ms")),
    ),
    FeatureDef(
        "com_jitter",
        "CoM Jitter",
        "standing_height_frac",
        "Detrended mean frame-to-frame center-of-mass travel, divided by standing height.",
        "balance",
        (("modules", "balance", "com_jitter"), ("modules", "balance", "jitter")),
    ),
    FeatureDef(
        "max_defender_displacement",
        "Defender Displacement (max)",
        "standing_height_frac",
        "Max defender hip travel from first valid point (run plays).",
        "run_game",
        (("modules", "point_of_attack", "max_defender_displacement"),),
    ),
    FeatureDef(
        "mean_defender_displacement",
        "Defender Displacement (mean)",
        "standing_height_frac",
        "Mean defender hip travel from first valid point (run plays).",
        "run_game",
        (("modules", "point_of_attack", "mean_defender_displacement"),),
    ),
    FeatureDef(
        "mean_leverage_angle_deg",
        "Leverage Angle",
        "degrees",
        "Mean OL→DL hip vector angle vs image vertical (run plays).",
        "run_game",
        (("modules", "point_of_attack", "mean_leverage_angle_deg"),),
    ),
    FeatureDef(
        "path_efficiency",
        "Path Efficiency",
        "fraction",
        "Straight-line hip travel divided by path length (run / pull / climb).",
        "run_game",
        (("modules", "movement_in_space", "path_efficiency"),),
    ),
    FeatureDef(
        "travel_distance",
        "Travel Distance",
        "standing_height_frac",
        "Hip path length divided by standing height (run plays).",
        "run_game",
        (("modules", "movement_in_space", "travel_distance"),),
    ),
)

FEATURE_REGISTRY: dict[str, FeatureDef] = {d.key: d for d in FEATURE_DEFS}
FEATURE_KEYS: tuple[str, ...] = tuple(d.key for d in FEATURE_DEFS)


def normalize_position(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return "unknown"
    v = str(value).strip().upper()
    aliases = {"TACKLE": "unknown", "GUARD": "unknown", "LEFT_TACKLE": "LT", "RIGHT_TACKLE": "RT"}
    v = aliases.get(v, v)
    return v if v in POSITIONS else "unknown"


def normalize_play_type(value: str | None) -> str:
    if value is None:
        return "pass"
    v = str(value).strip().lower()
    return v if v in PLAY_TYPES else "pass"


def normalize_side(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return "unknown"
    v = str(value).strip().lower()
    return v if v in SIDES else "unknown"


def normalize_technique(value: str | None) -> str:
    if value is None or str(value).strip() == "":
        return "unknown"
    return str(value).strip().lower().replace(" ", "_")
