"""Configurable thresholds for OL analysis (Yeager pass-pro / run framework).

Spatial thresholds are fractions of standing height unless noted.
Tune against coach-labeled reps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


PlayType = Literal["pass", "run", "auto"]

NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


@dataclass
class AnalysisConfig:
    # --- Pose ---
    # Medium pose model: better ankles/wrists than nano for stadium film.
    pose_model: str = "yolov8m-pose.pt"
    pose_imgsz: int = 1280
    min_keypoint_confidence: float = 0.35
    min_frame_keypoint_ratio: float = 0.40
    min_person_confidence: float = 0.25
    athlete_roi: tuple[float, float, float, float] = (0.10, 0.20, 0.90, 0.90)
    athlete_pick_xy: tuple[float, float] | None = None
    # Optional jersey number — when set, lock prefers OCR match (e.g. #76).
    target_jersey: int | None = None
    # Crop pad around locked OL. Lower = fewer distractors in-frame.
    track_crop_pad: float = 0.55
    track_max_jump_mult: float = 0.85
    # Identity stickiness — reject switches unless clearly the same body.
    track_min_iou: float = 0.32
    track_switch_iou_margin: float = 0.15
    track_max_center_frac: float = 0.42  # vs prior bbox diagonal
    track_area_ratio_min: float = 0.50
    track_area_ratio_max: float = 2.0
    track_ema: float = 0.88  # higher = stickier anchor
    track_lost_expand_frames: int = 10
    track_teleport_frac: float = 0.38  # hard reject 1-frame jumps beyond this × athlete diag
    # Max travel from lock, as multiples of the locked athlete's bbox diagonal.
    # Pass sets stay compact; run/pull needs more room — scaled by play_type at runtime.
    track_max_origin_diag_mult: float = 1.8
    track_max_origin_diag_mult_run: float = 3.2
    track_hip_jump_frac: float = 0.30  # vs athlete diag
    track_hip_origin_frac: float = 1.4  # hip vs lock origin, × diag
    track_hip_vert_frac: float = 0.60  # |Δy| hip vs lock origin, × diag
    # Track nearest defender inside the OL crop for mirror / anchor / hands.
    track_defender: bool = True
    overlay_zoom_on_athlete: bool = False
    overlay_zoom_size: int = 720
    play_type: PlayType = "pass"

    # --- Height ---
    standing_height_pre_snap_frames: int = 30
    nose_to_hip_height_factor: float = 1.65

    # --- Snap ---
    snap_motion_zscore: float = 3.5
    snap_baseline_frames: int = 20
    snap_sustained_frames: int = 3
    snap_max_full_to_roi_ratio: float = 8.0
    snap_max_full_frame_energy: float = 25.0
    snap_search_margin_frac: float = 0.05
    snap_roi: tuple[float, float, float, float] = (0.35, 0.45, 0.55, 0.72)
    snap_frame_override: int | None = None

    # --- Initial quicks / get-off ---
    movement_threshold_frac: float = 0.025
    movement_min_px: float = 10.0
    movement_sustain_frames: int = 2
    movement_monotonic_window: int = 3
    reaction_search_max_frames: int = 45
    baseline_lookback_frames: int = 15
    baseline_gap_frames: int = 3
    # Coach flag: "late off the ball" if reaction exceeds this (ms). @30fps ~150ms≈4.5f
    late_off_ball_ms: float = 200.0

    # --- Footwork ---
    step_peak_prominence_frac: float = 0.022
    step_min_separation_frames: int = 3
    base_width_narrow_frac: float = 0.22
    base_width_wide_frac: float = 0.38
    # Sideline perspective inflates lateral travel; require clear overset.
    overset_width_frac: float = 0.85
    crossover_min_frames: int = 2
    # Plausible OL shuffle cadence band (Hz). Outside → measurement noise.
    foot_quickness_cadence_min_hz: float = 2.8
    foot_quickness_cadence_max_hz: float = 4.8

    # --- Body position ---
    # Tuned to sideline YOLO hip/torso distributions (hips often 0.55–0.80 H).
    knee_bender_flexion_deg: float = 145.0
    waist_bender_torso_deg: float = 22.0
    low_hip_height_frac: float = 0.62
    posture_majority_frac: float = 0.50
    com_stability_jitter_frac: float = 0.075  # pass-set CoM wander is expected
    use_nn_posture: bool = True
    nn_posture_min_confidence: float = 0.75
    nn_only_on_borderline: bool = False  # NN fills unknown only; never overrides geometry

    # --- Contact / hands / sustain / mirror ---
    contact_distance_frac: float = 0.48  # hip-mid to hip-mid
    hand_reach_frac: float = 0.40
    hand_inside_shoulder_tol_frac: float = 0.15
    separation_close_frac: float = 0.42
    redirect_vel_flip_frac: float = 0.012  # defender lateral vel sign flip
    anchor_give_frac: float = 0.12  # hip retreat after contact vs height
    sustain_min_frames: int = 5
    # Require real contact before judging sustain / early_disengage.
    sustain_require_contact: bool = True

    # --- Set end ---
    set_max_frames: int = 180
    set_end_frame_override: int | None = None
    pose_lost_end_frames: int = 10

    # --- Output ---
    write_overlay_video: bool = True
    overlay_suffix: str = "_overlay.mp4"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["snap_roi"] = list(self.snap_roi)
        d["athlete_roi"] = list(self.athlete_roi)
        d["athlete_pick_xy"] = list(self.athlete_pick_xy) if self.athlete_pick_xy else None
        return d
