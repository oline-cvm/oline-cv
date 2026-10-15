"""Tracking diagnostics: per-frame JSONL trace and an annotated debug video."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oline_cv.tracking.deep_hm_sort import FrameResult
from oline_cv.tracking.target_identity import TargetFrame, TargetState

_STATE_BGR = {
    TargetState.VISIBLE_CONFIDENT: (60, 220, 60),
    TargetState.REACQUIRED: (255, 200, 0),
    TargetState.VISIBLE_AMBIGUOUS: (0, 200, 255),
    TargetState.REACQUISITION_CANDIDATE: (255, 0, 255),
    TargetState.OCCLUDED: (0, 140, 255),
    TargetState.LOST: (0, 0, 255),
}


def frame_record(fr: FrameResult, tf: TargetFrame | None) -> dict:
    tracks = []
    for o in fr.observations:
        b = fr.boxes[o.det_index]
        tracks.append({
            "track_id": o.track_id,
            "state": o.state.value,
            "confirmed": o.confirmed,
            "reactivated": o.reactivated,
            "stage": o.stage,
            "bbox": [round(float(v), 1) for v in b],
            "score": round(float(fr.scores[o.det_index]), 3),
            "cost": None if o.cost is None else round(o.cost, 4),
            "d_iou": None if o.d_iou is None else round(o.d_iou, 4),
            "d_app": None if o.d_app is None else round(o.d_app, 4),
            "appearance_updated": o.appearance_updated,
        })
    return {
        "frame": fr.frame_idx,
        "n_detections": len(fr),
        "tracks": tracks,
        "target": None if tf is None else tf.to_dict(),
    }


def write_trace(path: str | Path, results: list[FrameResult], targets: list[TargetFrame | None],
                header: dict | None = None) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        if header is not None:
            fh.write(json.dumps({"header": header}, default=str) + "\n")
        for fr, tf in zip(results, targets):
            fh.write(json.dumps(frame_record(fr, tf)) + "\n")
    return p


def _short_state(state: TargetState) -> str:
    return {
        TargetState.VISIBLE_CONFIDENT: "VISIBLE",
        TargetState.VISIBLE_AMBIGUOUS: "AMBIGUOUS",
        TargetState.OCCLUDED: "OCCLUDED",
        TargetState.LOST: "LOST",
        TargetState.REACQUISITION_CANDIDATE: "REACQ_CAND",
        TargetState.REACQUIRED: "REACQUIRED",
    }.get(state, state.value)


def draw_debug_frame(
    img: np.ndarray,
    fr: FrameResult,
    tf: TargetFrame | None,
    mask: np.ndarray | None = None,
    overlay_info: dict | None = None,
) -> np.ndarray:
    import cv2

    out = img.copy()
    scale = max(1.0, out.shape[0] / 720.0)
    th = max(1, int(round(scale)))
    if mask is not None and mask.shape[:2] == out.shape[:2] and mask.any():
        tint = out.copy()
        tint[mask.astype(bool)] = (40, 40, 220)
        out = cv2.addWeighted(out, 0.72, tint, 0.28, 0)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (40, 40, 255), th)
    for i in range(len(fr)):
        x1, y1, x2, y2 = (int(v) for v in fr.boxes[i])
        tid = int(fr.track_ids[i])
        rec = fr.is_recovery(i)
        col = (80, 80, 200) if rec else ((170, 170, 170) if fr.confirmed[i] else (90, 90, 90))
        cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
        label = ("r" if rec else "") + (f"{tid}" if tid >= 0 else "-")
        cv2.putText(out, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * scale, col, th, cv2.LINE_AA)
    if tf is not None:
        col = _STATE_BGR.get(tf.state, (255, 255, 255))
        if tf.has_box and tf.bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in tf.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2 * th + 2)
        elif tf.det_index is not None and tf.det_index < len(fr):
            x1, y1, x2, y2 = (int(v) for v in fr.boxes[tf.det_index])
            cv2.rectangle(out, (x1, y1), (x2, y2), col, th)
        info = overlay_info or {}
        text = (
            f"TARGET {_short_state(tf.state)} id=1 trk={tf.track_id} "
            f"unseen={tf.frames_since_seen}"
        )
        if tf.cost is not None:
            text += f" cost={tf.cost:.2f} app={tf.d_app:.2f} iou={tf.d_iou:.2f}"
        if tf.mask_coverage is not None:
            text += f" mask={tf.mask_coverage:.2f}"
        extra = f" reid={info.get('reid', '?')} seg={info.get('seg', '?')}"
        cv2.rectangle(out, (0, 0), (out.shape[1], int(44 * scale)), (0, 0, 0), -1)
        cv2.putText(out, f"f{fr.frame_idx} {text}  [{tf.reason}]", (6, int(18 * scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, col, th, cv2.LINE_AA)
        cv2.putText(out, extra, (6, int(36 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45 * scale, (200, 200, 200), th, cv2.LINE_AA)
    return out


def write_debug_video(path: str | Path, frames: list[np.ndarray], results: list[FrameResult],
                      targets: list[TargetFrame | None], fps: float,
                      masks: list | None = None, overlay_info: dict | None = None) -> Path | None:
    import cv2

    if not frames:
        return None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    try:
        for i, (img, fr, tf) in enumerate(zip(frames, results, targets)):
            m = None
            if masks is not None and i < len(masks) and masks[i] is not None:
                ev = masks[i]
                m = ev.mask if hasattr(ev, "mask") else ev
            vw.write(draw_debug_frame(img, fr, tf, m, overlay_info))
    finally:
        vw.release()
    return p
