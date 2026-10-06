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


TrackerBackend = Literal["deep_hm_sort", "legacy_botsort"]


@dataclass
class DeepHMSortConfig:
    """Deep HM-SORT tracker plus the OLINE locked-target layer.

    Tracker defaults follow Deep HM-SORT (Gran-Henriksen et al., 2024) /
    Deep-EIoU (Huang et al., 2024). ``target_*`` fields belong to the OLINE
    locked-target layer, which is not part of either paper.
    """

    # Detection confidence bands.
    detector_threshold: float = 0.4  # detections below this are discarded
    high_confidence_threshold: float = 0.6  # first-stage / new-track band
    new_track_threshold: float = 0.6
    # Association.
    assignment_cost_threshold: float = 0.8  # Hungarian cutoff, first stage
    appearance_cost_threshold: float = 0.3  # d_app above this counts as 1
    proximity_threshold: float = 0.5  # d_iou above this disables appearance
    low_assignment_cost_threshold: float = 0.5  # second (low-score) stage
    unconfirmed_assignment_cost_threshold: float = 0.7
    expansion_initial: float = 0.3
    expansion_step: float = 0.3
    expansion_iterations: int = 2
    low_stage_expansion: float = 0.5
    appearance_ema_alpha: float = 0.9
    # Tracklets are kept for the whole video and never re-numbered.
    keep_all_tracklets: bool = True
    historical_after_s: float = 1.0  # LOST -> HISTORICAL label after this
    max_lost_s: float | None = None  # only used when keep_all_tracklets=False
    # OLINE addition to the generic tracker: do not refresh a tracklet's
    # appearance from a box that overlaps another detection this much.
    appearance_update_overlap_threshold: float = 0.2
    # OLINE addition (after BoT-SORT GMC): move last boxes with camera pan/zoom.
    camera_motion_compensation: bool = True

    # ReID: "auto" uses OSNet when weights are configured, otherwise the
    # generic detector-feature fallback (lower quality, flagged in lock meta).
    reid_backend: Literal["auto", "osnet", "yolo_embed"] = "auto"
    reid_weights: str | None = None  # or env OLINE_REID_WEIGHTS
    reid_model_name: str = "osnet_x1_0"
    reid_device: str | None = None  # None = cuda when available
    reid_batch_size: int = 32
    reid_min_crop_px: int = 12

    # --- OLINE locked-target layer ---
    target_reacquire_max_cost: float = 0.5
    target_reacquire_min_margin: float = 0.1
    target_reid_max_distance: float | None = None  # None = appearance_cost_threshold
    # The harmonic mean lets one strong cue carry a weak one; reacquisition
    # also needs real overlap with the (expanded) last box on its own.
    target_reacquire_max_iou_distance: float = 0.7
    # Tighten the reacquisition appearance gate to the q-quantile of distances
    # to players seen alongside the target (known other people). Generic
    # embeddings put teammates close; this keeps them out.
    target_adaptive_appearance: bool = True
    target_impostor_quantile: float = 0.05
    target_impostor_min_samples: int = 20
    target_reacquire_confirm_frames: int = 2
    target_visible_min_margin: float = 0.03
    target_identity_fail_frames: int = 3
    target_occluded_max_s: float = 1.0  # OCCLUDED label, then LOST
    target_expand_every_s: float = 0.2
    target_max_expansion: float = 1.5
    # A tracklet seen alongside the target is someone else; None = for the
    # whole clip.
    target_exclusion_s: float | None = None
    locked_target_gallery_size: int = 16
    target_gallery_min_interval_s: float = 0.2
    target_update_min_score: float = 0.6
    target_min_crop_height_px: float = 24.0
    target_edge_margin_px: float = 2.0
    target_bidirectional: bool = True


@dataclass
class AnalysisConfig:
    # --- Tracking backend ---
    # "deep_hm_sort" (default) or "legacy_botsort" (deprecated detector +
    # BoT-SORT + whole-clip link path, kept for A/B comparison only).
    tracker_backend: TrackerBackend = "deep_hm_sort"
    deep_hm: DeepHMSortConfig = field(default_factory=DeepHMSortConfig)
    # Writes <stem>_tracking.jsonl and <stem>_tracking_debug.mp4 to
    # track_debug_dir (or the output dir). Also enabled by OLINE_TRACK_DEBUG=1.
    debug_tracking: bool = False

    # --- Pose ---
    # Medium pose model: better ankles/wrists than nano for stadium film.
    pose_model: str = "yolov8m-pose.pt"
    # Upper bound; capped at the video's own long side (see ``imgsz_for``).
    # Sideline film stacks linemen in depth: at 1280 a 2.8k frame merges the
    # interior line into one box, at 1920 each lineman gets his own.
    pose_imgsz: int = 1920
    # Person detector for tracking. The pose model misses small players on
    # wide shots (it may see only the referees); a plain detector finds
    # everyone, and pose then runs on an upscaled crop of the chosen player.
    detect_model: str = "yolov8m.pt"
    pose_crop_px: int = 640  # crop around a tracked player is resized to this
    min_keypoint_confidence: float = 0.35
    min_frame_keypoint_ratio: float = 0.40
    min_person_confidence: float = 0.25
    athlete_roi: tuple[float, float, float, float] = (0.10, 0.20, 0.90, 0.90)
    athlete_pick_xy: tuple[float, float] | None = None
    athlete_pick_time_s: float | None = None  # video time the tap was made on
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
    # Once locked, never adopt another player — frozen appearance + association.
    track_freeze_identity: bool = True
    track_id_embed_min: float = 0.58  # legacy; prefer track_min_appearance
    track_id_train_steps: int = 80  # unused (no lock-time NN train)
    # Association (None = leave unset; calib mode derives appearance floor from self-sim)
    track_calib_mode: bool = True
    track_debug_dir: str | None = None
    track_lost_buffer: int = 45
    # Pick the locked player's path over the whole clip (offline) instead of
    # deciding identity greedily frame by frame. False = legacy associator.
    track_global_link: bool = True
    track_interp_gap_s: float = 0.2
    # Whole-clip link gates, in the locked player's box units (per second so
    # 30 and 60 fps film behave alike). A pass-setting tackle stays within ~2
    # stance-box heights of his spot; the leash keeps the path from walking
    # onto a referee or a back running past. None disables the leash.
    track_max_speed_diag_s: float = 3.0
    track_leash_h: float | None = 2.0  # fill the box across occlusions up to this long
    track_reject_wrong_team: bool = True
    track_min_appearance: float | None = None
    track_min_jersey: float | None = None
    track_min_weighted: float | None = None
    track_min_iou_assoc: float | None = None
    track_uncertain_weighted: float | None = None
    track_w_appearance: float = 0.35
    track_w_motion: float = 0.20
    track_w_iou: float = 0.20
    track_w_jersey: float = 0.12
    track_w_size: float = 0.08
    track_w_formation: float = 0.05
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

    # --- 3D motion reconstruction (Phase 1 export) ---
    # When set, analyze_video writes tracks.json + frames/ + crops/ here for the
    # world-grounded HMR stage. Off by default: the export is disk-heavy.
    motion3d_export_dir: str | None = None
    motion3d_crop_size: int = 256
    motion3d_crop_pad: float = 0.25
    motion3d_max_interp_gap: int = 8
    motion3d_save_full_frames: bool = True

    # --- Optional statistical benchmark (off by default) ---
    # When set, analyze_video() extracts numeric features from the finished
    # result and appends result["benchmark_comparison"]. Never required to import.
    benchmark_path: str | None = None
    benchmark_position: str | None = None
    benchmark_technique: str | None = None
    benchmark_side: str | None = None
    # Optional five-attribute good/bad model (stance/first_step/feet/hands/hips).
    attribute_model_path: str | None = None

    # --- Output ---
    write_overlay_video: bool = True
    overlay_suffix: str = "_overlay.mp4"
    extra: dict[str, Any] = field(default_factory=dict)

    def imgsz_for(self, frame: Any) -> int:
        """Pose input size for ``frame``: ``pose_imgsz`` but never upscaled."""
        long_side = max(frame.shape[:2])
        return max(32, min(self.pose_imgsz, -(-long_side // 32) * 32))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["snap_roi"] = list(self.snap_roi)
        d["athlete_roi"] = list(self.athlete_roi)
        d["athlete_pick_xy"] = list(self.athlete_pick_xy) if self.athlete_pick_xy else None
        return d
