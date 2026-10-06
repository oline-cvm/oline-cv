"""Multi-object tracking for OLINE.

``deep_hm_sort`` is a general sports tracker after Deep HM-SORT
(Gran-Henriksen et al., arXiv:2406.12081), which builds on Deep-EIoU
(Huang et al., WACV Workshops 2024). ``target_identity`` is an OLINE-specific
layer that keeps one tapped player's identity on top of the tracker; it is not
part of either paper.

Importing this package does not import torch, torchreid or ultralytics; ReID
backends load lazily on first use.
"""

from oline_cv.tracking.costs import (
    cosine_distance_matrix,
    harmonic_mean_distance,
    linear_assignment,
)
from oline_cv.tracking.deep_hm_sort import (
    DeepHMSort,
    TrackObservation,
    Tracklet,
    TrackletState,
)
from oline_cv.tracking.expansion_iou import expand_boxes, expansion_iou
from oline_cv.tracking.target_identity import (
    LockedTargetIdentity,
    TargetFrame,
    TargetState,
    run_target_identity,
)

__all__ = [
    "DeepHMSort",
    "LockedTargetIdentity",
    "TargetFrame",
    "TargetState",
    "TrackObservation",
    "Tracklet",
    "TrackletState",
    "cosine_distance_matrix",
    "expand_boxes",
    "expansion_iou",
    "harmonic_mean_distance",
    "linear_assignment",
    "run_target_identity",
]
