"""Coach PDF report with keyframe stills from the analyzed clip."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from oline_cv.coach_brief import build_coach_brief


def select_keyframe_indices(result: dict[str, Any], n_video_frames: int) -> list[tuple[int, str]]:
    """Pick labeled frames for the report (snap, get-off, contact, finish)."""
    snap = int((result.get("snap") or {}).get("snap_frame") or 0)
    q = result.get("initial_quicks") or result.get("modules", {}).get("initial_quicks") or {}
    set_info = result.get("set") or {}
    end = int(set_info.get("end_frame") or min(n_video_frames - 1, snap + 60))
    foot = q.get("first_foot_movement_frame")
    hip = q.get("first_hip_movement_frame")
    react = foot if foot is not None else hip

    picks: list[tuple[int, str]] = [(snap, "Snap")]
    if react is not None and int(react) != snap:
        picks.append((int(react), "First move / get-off"))
    mid = int(round((snap + end) / 2))
    if mid not in {p[0] for p in picks}:
        picks.append((mid, "Mid-rep"))
    # Late contact / finish
    late = min(n_video_frames - 1, max(end, snap + 1))
    if late not in {p[0] for p in picks}:
        picks.append((late, "Finish / contact"))

    # Dedupe + clamp
    out: list[tuple[int, str]] = []
    seen = set()
    for idx, label in picks:
        idx = int(np.clip(idx, 0, max(0, n_video_frames - 1)))
        if idx in seen:
            continue
        seen.add(idx)
        out.append((idx, label))
    return out[:5]


def extract_keyframes(
    video_path: str,
    result: dict[str, Any],
    out_dir: str | Path,
    ol_poses: list | None = None,
) -> list[dict[str, Any]]:
    """Write JPEG stills + optional skeleton for the report."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    picks = select_keyframe_indices(result, n if n > 0 else 1)
    saved: list[dict[str, Any]] = []

    for frame_idx, label in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        draw = frame.copy()
        if ol_poses is not None and 0 <= frame_idx < len(ol_poses):
            pose = ol_poses[frame_idx]
            _draw_simple_skeleton(draw, pose)
        path = out_dir / f"kf_{frame_idx:05d}.jpg"
        cv2.imwrite(str(path), draw, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        saved.append(
            {
                "frame_idx": frame_idx,
                "label": label,
                "path": str(path),
                "url": None,  # filled by caller
            }
        )
    cap.release()
    return saved


def _draw_simple_skeleton(img: np.ndarray, pose) -> None:
    try:
        xy = pose.keypoints_xy
        conf = pose.keypoints_conf
        bbox = pose.bbox_xyxy
    except Exception:
        return
    bones = [(5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (5, 7), (7, 9), (6, 8), (8, 10)]
    for a, b in bones:
        if conf[a] < 0.3 or conf[b] < 0.3:
            continue
        pa, pb = xy[a], xy[b]
        if np.any(np.isnan(pa)) or np.any(np.isnan(pb)):
            continue
        cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (60, 220, 140), 2, cv2.LINE_AA)
    if bbox is not None:
        b = bbox.astype(int)
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (40, 200, 120), 2)


def _pdf_text(s: str) -> str:
    """Core Helvetica is Latin-1 — strip fancy punctuation."""
    return (
        str(s)
        .replace("\u2014", "-")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2026", "...")
        .encode("latin-1", "replace")
        .decode("latin-1")
    )


def write_coach_pdf(
    packed: dict[str, Any],
    full_result: dict[str, Any],
    keyframes: list[dict[str, Any]],
    out_path: str | Path,
) -> Path:
    """Generate a simple one/two-page coach PDF."""
    try:
        from fpdf import FPDF
    except ImportError as exc:
        raise RuntimeError("fpdf2 is required for PDF export — pip install fpdf2") from exc

    brief = build_coach_brief(packed)
    out_path = Path(out_path)

    pdf = FPDF(unit="mm", format="Letter")
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()
    pdf.set_margins(16, 14, 16)

    # Header
    pdf.set_font("Helvetica", "B", 18)
    jersey = packed.get("jersey")
    title = f"OLINE Coach Report - #{jersey}" if jersey is not None else "OLINE Coach Report"
    pdf.cell(0, 10, _pdf_text(title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    play = "Pass protection" if brief["play"] == "pass" else "Run blocking"
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 6, _pdf_text(f"{play}  |  {brief['verdict']}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 5, _pdf_text(brief["summary"]))
    pdf.ln(4)

    def _section(title: str, items: list[dict[str, str]], color: tuple[int, int, int]) -> None:
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(*color)
        pdf.cell(0, 8, _pdf_text(title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        if not items:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 6, "None flagged for this rep.", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(2)
            return
        for it in items:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 6, _pdf_text(f"- {it['title']}"), new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            pdf.multi_cell(0, 5, _pdf_text(it["detail"]))
            pdf.set_text_color(0, 0, 0)
            pdf.ln(1)
        pdf.ln(2)

    _section("Fix next", brief["fix"], (160, 70, 40))
    _section("Keep doing", brief["keep"], (70, 120, 60))

    # Keyframes page
    if keyframes:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, "Keyframe reference stills", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(80, 80, 80)
        pdf.multi_cell(
            0,
            5,
            "Stills from the analyzed film at coaching moments (snap, get-off, mid-rep, finish).",
        )
        pdf.set_text_color(0, 0, 0)
        pdf.ln(3)

        usable = [k for k in keyframes if k.get("path") and Path(k["path"]).exists()]
        x0 = pdf.l_margin
        y = pdf.get_y()
        col_w = 90
        img_h = 52
        for i, kf in enumerate(usable[:4]):
            col = i % 2
            row = i // 2
            x = x0 + col * (col_w + 6)
            yy = y + row * (img_h + 16)
            pdf.set_xy(x, yy)
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(
                col_w,
                5,
                _pdf_text(f"{kf['label']}  (frame {kf['frame_idx']})"),
                new_x="LMARGIN",
                new_y="NEXT",
            )
            try:
                pdf.image(kf["path"], x=x, y=yy + 6, w=col_w, h=img_h)
            except Exception:
                pdf.set_xy(x, yy + 6)
                pdf.set_font("Helvetica", "I", 9)
                pdf.cell(col_w, 6, "(image unavailable)")

    pdf.set_y(-18)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(
        0,
        5,
        "Generated by OLINE - for coaching use. Prefer film + cues over raw numbers.",
        align="C",
    )

    pdf.output(str(out_path))
    return out_path


def build_field_pose_payload(
    full_result: dict[str, Any],
    *,
    video_url: str | None = None,
) -> dict[str, Any]:
    """Pose stream + film URL for the immersive on-field viewer (any clip)."""
    video = full_result.get("video") or {}
    w = float(video.get("width") or 1920)
    h = float(video.get("height") or 1080)
    fps = float(video.get("fps") or 30)
    frames_out = []
    for fr in full_result.get("frames") or []:
        kps = fr.get("keypoints") or {}
        joints = {}
        for name, pt in kps.items():
            if not isinstance(pt, dict):
                continue
            x, y = pt.get("x"), pt.get("y")
            c = float(pt.get("confidence") or 0)
            if x is None or y is None or c < 0.25:
                continue
            joints[name] = {"x": float(x) / w, "y": float(y) / h, "c": c}
        if len(joints) < 4:
            continue
        xs = [j["x"] for j in joints.values()]
        ys = [j["y"] for j in joints.values()]
        pad_x, pad_y = 0.04, 0.06
        bbox = [
            max(0.0, min(xs) - pad_x),
            max(0.0, min(ys) - pad_y),
            min(1.0, max(xs) + pad_x),
            min(1.0, max(ys) + pad_y),
        ]
        frames_out.append(
            {
                "frame_idx": fr.get("frame_idx"),
                "t": float(fr.get("timestamp_ms") or 0) / 1000.0,
                "posture": fr.get("posture"),
                "joints": joints,
                "bbox": bbox,
            }
        )
    return {
        "fps": fps,
        "width": w,
        "height": h,
        "snap_frame": (full_result.get("snap") or {}).get("snap_frame"),
        "play_type": full_result.get("play_type"),
        "video_url": video_url,
        "frames": frames_out,
    }
