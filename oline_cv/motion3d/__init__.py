"""Monocular 3D human-motion reconstruction for the locked OL.

Pipeline stages (see docs in each module):

    BoT-SORT target track  -> track_export   (Phase 1)
    world-grounded HMR     -> hmr_runner     (Phase 2)
    field calibration      -> calibration    (Phase 4)
    cleanup / foot lock    -> smoothing      (Phase 5)

Phase 1 is the only stage that touches the existing tracking pipeline, and it is
read-only with respect to it: it serializes what ``PoseTracker`` already produces.
"""

from oline_cv.motion3d.motion_schema import (
    MOTION_SCHEMA_VERSION,
    MotionMetadata,
    ReconstructionStatus,
    load_metadata,
    validate_motion_npz,
)
from oline_cv.motion3d.schema import (
    SCHEMA_VERSION,
    BboxSource,
    CropRef,
    TrackFrame,
    TrackManifest,
    load_manifest,
)
from oline_cv.motion3d.segments import Segment, describe_segments, select_segment
from oline_cv.motion3d.target_association import (
    AssociationThresholds,
    Candidate,
    FrameAssociation,
    associate_frame,
    associate_sequence,
    load_associations,
    save_associations,
    summarize,
)

__all__ = [
    "SCHEMA_VERSION",
    "BboxSource",
    "CropRef",
    "TrackFrame",
    "TrackManifest",
    "load_manifest",
    "MOTION_SCHEMA_VERSION",
    "MotionMetadata",
    "ReconstructionStatus",
    "load_metadata",
    "validate_motion_npz",
    "Segment",
    "describe_segments",
    "select_segment",
    "AssociationThresholds",
    "Candidate",
    "FrameAssociation",
    "associate_frame",
    "associate_sequence",
    "load_associations",
    "save_associations",
    "summarize",
]

# oline_cv.motion3d.wham_bridge is intentionally NOT imported here: it shells out
# to WSL and should only be pulled in by callers that actually run the stage.
