"""When ordinary Deep HM association is unsafe for the locked target.

OLINE-only. Signals come from overlap, detector score, association margin,
appearance jump, unmatched target, and sudden box jumps — any one can raise
the risk; they are not all required.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from oline_cv.tracking.deep_hm_sort import FrameResult
from oline_cv.tracking.expansion_iou import iou_matrix


class TargetRisk(str, Enum):
    CONFIDENT = "confident"
    AMBIGUOUS = "ambiguous"
    PARTIAL_OCCLUSION = "partial_occlusion"
    FULL_OCCLUSION = "full_occlusion"
    LOST = "lost"


def overlap_with_others(boxes: np.ndarray, i: int) -> float:
    if len(boxes) < 2:
        return 0.0
    ov = iou_matrix(boxes[i][None], np.delete(boxes, i, axis=0))[0]
    return float(ov.max()) if len(ov) else 0.0


def evaluate_risk(
    fr: FrameResult,
    *,
    det_index: int | None,
    bound_present: bool,
    margin: float | None,
    d_app: float | None,
    reid_max: float,
    last_box: np.ndarray | None,
    cfg,
    unseen: int,
) -> tuple[TargetRisk, list[str]]:
    """Return (risk, reasons). ``det_index`` is the currently considered box."""
    reasons: list[str] = []
    if len(fr) == 0 or det_index is None:
        if unseen <= 0:
            return TargetRisk.FULL_OCCLUSION, ["unmatched"]
        return (TargetRisk.FULL_OCCLUSION if unseen < 8 else TargetRisk.LOST), ["unmatched"]

    i = int(det_index)
    ov = overlap_with_others(fr.boxes, i)
    if ov >= cfg.target_occlusion_overlap_threshold:
        reasons.append("overlap")
    if fr.scores[i] < cfg.high_confidence_threshold:
        reasons.append("low_score")
    if margin is not None and margin < cfg.target_ambiguity_margin:
        reasons.append("close_competitor")
    if d_app is not None and np.isfinite(d_app) and d_app > reid_max:
        reasons.append("appearance_jump")
    if last_box is not None:
        h = max(1.0, float(last_box[3] - last_box[1]))
        ca = (last_box[:2] + last_box[2:]) / 2
        cb = (fr.boxes[i, :2] + fr.boxes[i, 2:]) / 2
        if float(np.linalg.norm(cb - ca) / h) > cfg.target_jump_box_h:
            reasons.append("bbox_jump")
    if not bound_present:
        reasons.append("unmatched")

    if "overlap" in reasons and ("low_score" in reasons or "unmatched" in reasons):
        return TargetRisk.PARTIAL_OCCLUSION, reasons
    if reasons and any(r in reasons for r in ("overlap", "close_competitor", "appearance_jump", "bbox_jump")):
        return TargetRisk.AMBIGUOUS, reasons
    if "unmatched" in reasons or "low_score" in reasons:
        return TargetRisk.AMBIGUOUS, reasons
    return TargetRisk.CONFIDENT, reasons
