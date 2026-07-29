"""Identity stickiness tests for pose tracking."""

from __future__ import annotations

import numpy as np

from oline_cv.config import AnalysisConfig
from oline_cv.pose_tracker import PoseTracker, _bbox_iou


def test_bbox_iou_identical():
    a = np.array([10.0, 10.0, 50.0, 80.0])
    assert _bbox_iou(a, a) > 0.99


def test_bbox_iou_disjoint():
    a = np.array([0.0, 0.0, 10.0, 10.0])
    b = np.array([20.0, 20.0, 30.0, 30.0])
    assert _bbox_iou(a, b) == 0.0


def test_select_ol_prefers_iou_over_nearby_distractor():
    cfg = AnalysisConfig(track_min_iou=0.25, track_max_center_frac=0.55)
    tr = PoseTracker(cfg)
    tr._anchor_bbox = np.array([100.0, 100.0, 180.0, 260.0])
    tr._anchor_center = np.array([140.0, 180.0])
    tr._anchor_vel = np.zeros(2)

    # Same athlete (high IoU) vs nearby bigger distractor
    boxes = np.array(
        [
            [105.0, 105.0, 175.0, 255.0],  # true OL
            [160.0, 120.0, 260.0, 300.0],  # distractor overlapping edge
        ]
    )
    conf = np.array([0.7, 0.95])
    idx, score = tr._select_ol(boxes, conf, 800, 600)
    assert idx == 0
    assert score > 0.4


def test_select_ol_rejects_far_jump():
    cfg = AnalysisConfig(track_min_iou=0.25, track_max_center_frac=0.45, track_max_jump_mult=0.85)
    tr = PoseTracker(cfg)
    tr._anchor_bbox = np.array([100.0, 100.0, 180.0, 260.0])
    tr._anchor_center = np.array([140.0, 180.0])
    tr._anchor_vel = np.zeros(2)

    boxes = np.array(
        [
            [500.0, 400.0, 600.0, 560.0],  # far away — different player
        ]
    )
    conf = np.array([0.9])
    idx, score = tr._select_ol(boxes, conf, 800, 600)
    assert idx is None
