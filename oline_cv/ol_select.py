"""Auto-detect an offensive lineman from pre-snap stance / LOS geometry.

No jersey hardcoding. Optional click override still wins when provided.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oline_cv.config import (
    AnalysisConfig,
    L_ANKLE,
    L_HIP,
    L_KNEE,
    L_WRIST,
    R_ANKLE,
    R_HIP,
    R_KNEE,
    R_WRIST,
)
from oline_cv.geometry import hip_mid, joint_angle_deg


@dataclass
class OLCandidate:
    index: int
    center: np.ndarray
    bbox: np.ndarray
    score: float
    knee_flex: float | None
    hip_sink: float | None
    area: float


def _knee_mean(xy: np.ndarray, conf: np.ndarray, min_conf: float) -> float | None:
    angles = []
    for hip, knee, ankle in ((L_HIP, L_KNEE, L_ANKLE), (R_HIP, R_KNEE, R_ANKLE)):
        if min(float(conf[hip]), float(conf[knee]), float(conf[ankle])) < min_conf:
            continue
        if np.any(np.isnan(xy[[hip, knee, ankle]])):
            continue
        a = joint_angle_deg(xy[hip], xy[knee], xy[ankle])
        if a is not None:
            angles.append(a)
    return float(np.mean(angles)) if angles else None


def _hip_sink(xy: np.ndarray, conf: np.ndarray, bbox: np.ndarray, min_conf: float) -> float | None:
    if float(conf[L_HIP]) < min_conf or float(conf[R_HIP]) < min_conf:
        return None
    hips = hip_mid(xy)
    if np.any(np.isnan(hips)):
        return None
    y0, y1 = float(bbox[1]), float(bbox[3])
    return float((hips[1] - y0) / max(y1 - y0, 1.0))


def _hand_near_ground(xy: np.ndarray, conf: np.ndarray, bbox: np.ndarray, min_conf: float) -> bool:
    y1 = float(bbox[3])
    h = max(float(bbox[3] - bbox[1]), 1.0)
    for w in (L_WRIST, R_WRIST):
        if float(conf[w]) < min_conf or np.any(np.isnan(xy[w])):
            continue
        if (y1 - float(xy[w][1])) / h < 0.22:
            return True
    return False


def score_ol_candidate(
    xy, conf, bbox, box_conf, frame_area, config: AnalysisConfig
) -> OLCandidate | None:
    area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    if area < frame_area * 0.004:
        return None
    knee = _knee_mean(xy, conf, config.min_keypoint_confidence)
    sink = _hip_sink(xy, conf, bbox, config.min_keypoint_confidence)
    center = (bbox[:2] + bbox[2:]) / 2.0

    crouch = 0.0
    # Hip sink is the reliable OL stance cue on sideline film. Extreme knee
    # angles are often keypoint noise and were crowning skill players upfield.
    if sink is not None:
        crouch += max(0.0, (sink - 0.42) * 3.0)
    if knee is not None:
        if knee <= 145:
            crouch += min(0.40, (145.0 - knee) / 70.0)
        elif knee <= 160:
            crouch += 0.15
    if sink is not None and sink < 0.40:
        crouch *= 0.55  # upright-in-bbox → unlikely set OL

    size = min(1.5, area / (frame_area * 0.02))
    three_pt = 0.35 if _hand_near_ground(xy, conf, bbox, config.min_keypoint_confidence) else 0.0
    upright_penalty = 0.8 if (knee is not None and knee > 165 and (sink is None or sink < 0.42)) else 0.0
    score = 1.4 * crouch + 0.9 * size + three_pt + 0.25 * float(box_conf) - upright_penalty
    return OLCandidate(-1, center.astype(float), bbox.astype(float), float(score), knee, sink, area)


def _longest_ol_line(cands: list[OLCandidate], width: int, height: int) -> list[OLCandidate]:
    if len(cands) < 3:
        return cands
    ordered = sorted(cands, key=lambda c: c.center[0])
    max_dx, max_dy = 0.14 * width, 0.08 * height
    chains: list[list[OLCandidate]] = []
    for i in range(len(ordered)):
        chain = [ordered[i]]
        for j in range(i + 1, len(ordered)):
            prev, cur = chain[-1], ordered[j]
            dx = cur.center[0] - prev.center[0]
            if dx > max_dx:
                break
            if abs(cur.center[1] - prev.center[1]) <= max_dy:
                chain.append(cur)
        chains.append(chain)
    chains.sort(key=lambda ch: (len(ch), sum(c.area for c in ch)), reverse=True)
    return chains[0]


def select_ol_from_detections(
    xy, conf, box_xyxy, box_conf, width: int, height: int, config: AnalysisConfig
) -> OLCandidate | None:
    n = len(box_conf)
    if n == 0:
        return None
    frame_area = float(width * height)
    x0, y0, x1, y1 = config.athlete_roi
    cands: list[OLCandidate] = []
    for i in range(n):
        cx = 0.5 * (box_xyxy[i, 0] + box_xyxy[i, 2])
        cy = 0.5 * (box_xyxy[i, 1] + box_xyxy[i, 3])
        if not (x0 * width <= cx <= x1 * width and y0 * height <= cy <= y1 * height):
            continue
        c = score_ol_candidate(xy[i], conf[i], box_xyxy[i], float(box_conf[i]), frame_area, config)
        if c is None or c.score < 0.35:
            continue
        c.index = i
        cands.append(c)

    if not cands:
        centers = (box_xyxy[:, :2] + box_xyxy[:, 2:]) / 2.0
        areas = (box_xyxy[:, 2] - box_xyxy[:, 0]) * (box_xyxy[:, 3] - box_xyxy[:, 1])
        return OLCandidate(
            int(np.argmax(areas)),
            centers[int(np.argmax(areas))],
            box_xyxy[int(np.argmax(areas))].astype(float),
            0.0,
            None,
            None,
            float(np.max(areas)),
        )

    # Offense band ≈ closer to camera (larger y) on elevated All-22 / sideline.
    ys = np.array([c.center[1] for c in cands], dtype=float)
    if len(cands) >= 4:
        m1, m2 = float(ys.min()), float(ys.max())
        for _ in range(8):
            d1, d2 = np.abs(ys - m1), np.abs(ys - m2)
            g1, g2 = ys[d1 <= d2], ys[d1 > d2]
            if len(g1) and len(g2):
                m1, m2 = float(g1.mean()), float(g2.mean())
        offense_y = max(m1, m2)
        mid_y = 0.5 * (m1 + m2)
        # Must be on the camera-side of the mid-gap, not merely "near" offense_y.
        ol_band = [c for c in cands if c.center[1] >= mid_y - 0.02 * height]
        if len(ol_band) >= 2:
            cands = ol_band
            # Prefer the densest horizontal LOS inside that band
            line = _longest_ol_line(cands, width, height)
            if len(line) >= 3:
                cands = line

    crouched = [
        c
        for c in cands
        if (c.hip_sink is not None and c.hip_sink >= 0.46)
        or (c.knee_flex is not None and 125.0 <= c.knee_flex <= 155.0)
    ]
    pool = crouched if len(crouched) >= 2 else cands

    if len(pool) >= 3 and len(cands) < 3:
        line = _longest_ol_line(pool, width, height)
        if len(line) >= 3:
            pool = line

    # Best crouched body on the LOS cluster. Do not force LT/RT wing —
    # that flip-flopped lock between sideline samples and poisoned every metric.
    return max(pool, key=lambda c: c.score + 1e-6 * c.area)


def lock_ol_from_frames(
    model,
    frames: list,
    config: AnalysisConfig,
    sample_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not frames:
        raise RuntimeError("No frames for OL lock")
    h, w = frames[0].shape[:2]
    if sample_indices is None:
        n = len(frames)
        sample_indices = sorted(
            {0, max(0, min(n - 1, 15)), max(0, min(n - 1, 30)), max(0, min(n - 1, 45)), max(0, min(n - 1, 60))}
        )

    # Prefer explicit jersey when requested (e.g. #76).
    if config.target_jersey is not None:
        jersey_hit = _lock_by_jersey(model, frames, sample_indices, config, w, h)
        if jersey_hit is not None:
            return jersey_hit

    votes: list[OLCandidate] = []
    for idx in sample_indices:
        results = model.predict(
            frames[idx],
            verbose=False,
            conf=config.min_person_confidence,
            imgsz=config.pose_imgsz,
        )
        if not results or results[0].keypoints is None or results[0].boxes is None:
            continue
        r0 = results[0]
        if len(r0.boxes) == 0:
            continue
        kxy = r0.keypoints.xy.cpu().numpy()
        kcf = r0.keypoints.conf.cpu().numpy() if r0.keypoints.conf is not None else np.ones(kxy.shape[:2])
        pick = select_ol_from_detections(
            kxy, kcf, r0.boxes.xyxy.cpu().numpy(), r0.boxes.conf.cpu().numpy(), w, h, config
        )
        if pick is not None:
            votes.append(pick)

    if not votes:
        cx = 0.5 * (config.athlete_roi[0] + config.athlete_roi[2]) * w
        cy = 0.5 * (config.athlete_roi[1] + config.athlete_roi[3]) * h
        return (
            np.array([cx, cy]),
            np.array([cx - 40, cy - 80, cx + 40, cy + 80]),
            {"method": "fallback_roi_center", "score": 0.0},
        )

    # Modal spatial bin — prefer the OL the detector agrees on, not a one-off DE.
    # Tighter bins (2% width) + require spatial median of the winning cluster.
    bins: dict[int, list[OLCandidate]] = {}
    for v in votes:
        key = int(v.center[0] / max(0.02 * w, 1.0))
        bins.setdefault(key, []).append(v)
    modal = max(bins.values(), key=lambda vs: (len(vs), sum(v.score for v in vs)))
    # Robust center: median of agreeing votes (not single highest-score sample)
    xs = np.array([v.center[0] for v in modal])
    ys = np.array([v.center[1] for v in modal])
    center = np.array([float(np.median(xs)), float(np.median(ys))])
    # Bbox from vote nearest to median center
    pick = min(modal, key=lambda v: float(np.linalg.norm(v.center - center)))
    meta = {
        "method": "stance_los_cluster",
        "score": float(pick.score),
        "knee_flex": pick.knee_flex,
        "hip_sink": pick.hip_sink,
        "votes": len(votes),
        "agreement": len(modal),
        "sample_frames": sample_indices,
        "lock_xy": [float(center[0]), float(center[1])],
        "target_jersey": config.target_jersey,
    }
    return center, np.asarray(pick.bbox, dtype=float), meta


def _lock_by_jersey(
    model,
    frames: list,
    sample_indices: list[int],
    config: AnalysisConfig,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Vote across sample frames for the box whose jersey OCR matches target."""
    from oline_cv.jersey_ocr import find_jersey_match

    target = int(config.target_jersey)  # type: ignore[arg-type]
    hits: list[tuple[np.ndarray, np.ndarray, float, int]] = []  # center, bbox, conf, frame
    for idx in sample_indices:
        results = model.predict(
            frames[idx],
            verbose=False,
            conf=config.min_person_confidence,
            imgsz=config.pose_imgsz,
        )
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue
        boxes = results[0].boxes.xyxy.cpu().numpy()
        match = find_jersey_match(frames[idx], boxes, target, min_conf=0.35)
        if match is None:
            continue
        i, conf = match
        box = boxes[i].astype(float)
        center = (box[:2] + box[2:]) / 2.0
        hits.append((center, box, conf, idx))

    if not hits:
        return None

    # Cluster by x-bin; prefer the jersey cluster that appears most often.
    bins: dict[int, list[tuple[np.ndarray, np.ndarray, float, int]]] = {}
    for center, box, conf, idx in hits:
        key = int(center[0] / max(0.025 * width, 1.0))
        bins.setdefault(key, []).append((center, box, conf, idx))
    modal = max(bins.values(), key=lambda vs: (len(vs), sum(v[2] for v in vs)))
    # Need ≥2 agreeing frames, or one strong OCR read
    if len(modal) < 2 and not (len(modal) == 1 and modal[0][2] >= 0.55):
        return None

    xs = np.array([v[0][0] for v in modal])
    ys = np.array([v[0][1] for v in modal])
    center = np.array([float(np.median(xs)), float(np.median(ys))])
    pick = min(modal, key=lambda v: float(np.linalg.norm(v[0] - center)))
    meta = {
        "method": "jersey_ocr",
        "jersey": target,
        "ocr_confidence": float(np.mean([v[2] for v in modal])),
        "votes": len(hits),
        "agreement": len(modal),
        "sample_frames": sample_indices,
        "lock_xy": [float(center[0]), float(center[1])],
        "hit_frames": [int(v[3]) for v in modal],
    }
    return center, np.asarray(pick[1], dtype=float), meta
