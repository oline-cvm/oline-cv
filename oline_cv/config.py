"""Configurable thresholds for OL pass-set analysis.

All spatial thresholds are fractions of the player's standing height so they
remain invariant to camera distance. Tune against coach-labeled reps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# COCO-17 keypoint indices (Ultralytics / YOLO pose)
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
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass
class AnalysisConfig:
    """All tunable constants for Modules 1 and 2.

    Documented defaults are starting points — calibrate on coach-labeled film.
    """

    # --- Pose model ---
    # YOLOv8-pose weights (COCO-17). Auto-downloads on first run.
    # yolov8n-pose.pt is fast; use yolov8m-pose.pt for higher keypoint quality.
    pose_model: str = "yolov8n-pose.pt"
    pose_imgsz: int = 960
    # Minimum keypoint confidence to trust a joint (else frame is flagged).
    min_keypoint_confidence: float = 0.40
    # Fraction of tracked keypoints that must be confident for a usable frame.
    min_frame_keypoint_ratio: float = 0.45
    # Person detection confidence for selecting the OL.
    min_person_confidence: float = 0.25
    # ROI (x0,y0,x1,y1) as frame fractions — where the target OL is expected.
    # Tuned for elevated sideline/endzone film of the OL.
    athlete_roi: tuple[float, float, float, float] = (0.15, 0.25, 0.85, 0.85)
    # Optional normalized click (x,y) to pick a specific player on first lock.
    athlete_pick_xy: tuple[float, float] | None = None
    # Target jersey — used for labeling + default pick for #76.
    target_jersey: int = 76
    # After lock, pose only inside a padded crop around this athlete.
    track_crop_pad: float = 0.55
    # Max center jump (× bbox diagonal) before rejecting a detection as "other player".
    track_max_jump_mult: float = 1.35
    # Overlay zooms in on the tracked athlete instead of full field.
    overlay_zoom_on_athlete: bool = True
    overlay_zoom_size: int = 720

    # --- Standing height estimation ---
    # Use median of (ankle_mid → shoulder_mid) distance over the first N
    # pre-snap frames as standing height in pixels.
    standing_height_pre_snap_frames: int = 30
    # Fallback if ankles are occluded: nose-to-hip * this factor ≈ full height.
    nose_to_hip_height_factor: float = 1.65

    # --- Module 1: snap detection ---
    # Optical-flow / motion energy spike relative to rolling baseline.
    snap_motion_zscore: float = 3.5
    # Frames of quiet baseline immediately before candidate snap.
    snap_baseline_frames: int = 20
    # Require elevated motion for this many consecutive frames.
    snap_sustained_frames: int = 3
    # Reject candidates where full-frame motion dwarfs ROI motion (broadcast
    # graphics / camera cuts). Ratio = full_energy / roi_energy.
    snap_max_full_to_roi_ratio: float = 8.0
    # Absolute cap on full-frame mean absdiff — graphics flashes exceed this.
    snap_max_full_frame_energy: float = 25.0
    # Search window: ignore the first/last N% of the clip when auto-finding snap.
    snap_search_margin_frac: float = 0.05
    # ROI for snap: ball / center hands region (frame fractions x0,y0,x1,y1).
    snap_roi: tuple[float, float, float, float] = (0.35, 0.45, 0.55, 0.72)
    # Optional manual override (None = auto-detect).
    snap_frame_override: int | None = None

    # --- Module 1: initial quicks ---
    # Displacement threshold as fraction of standing height.
    # Spec starting point was 1.5%; distant stadium film needs a higher floor
    # because keypoint jitter is often 5–10px.
    movement_threshold_frac: float = 0.025
    # Absolute pixel floor (applied after height-normalized threshold).
    movement_min_px: float = 10.0
    # Require displacement above threshold for this many consecutive frames.
    movement_sustain_frames: int = 2
    # Require monotonic increase in cumulative displacement over this window
    # to reject single-frame pre-snap stance wiggles.
    movement_monotonic_window: int = 3
    # Max frames after snap to search for first movement (else null).
    reaction_search_max_frames: int = 45
    # Pre-snap baseline window: [snap - baseline_lookback, snap - baseline_gap).
    baseline_lookback_frames: int = 15
    baseline_gap_frames: int = 3

    # --- Module 2: body position / posture ---
    # Knee flexion is the interior angle at the knee (hip–knee–ankle).
    # 180° ≈ fully extended; smaller = more flexed.
    # "High knee flexion" for knee-bender classification.
    knee_bender_flexion_deg: float = 140.0  # angle below this = flexed
    # Torso angle from vertical (degrees). Near 0 = upright.
    waist_bender_torso_deg: float = 25.0  # lean above this = waist bend
    # Normalized hip height (hip_y / standing_height, origin at feet ≈ 0–1).
    # Lower values = hips closer to ground. Used as "low pad" cue.
    low_hip_height_frac: float = 0.55
    # Fraction of valid post-snap frames needed to assign a rep summary label.
    posture_majority_frac: float = 0.55

    # --- Set end detection ---
    # End of set: either explicit frame override, or N frames after snap,
    # or when pose is lost for this many consecutive frames.
    set_max_frames: int = 180  # 3.0s @ 60fps
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
