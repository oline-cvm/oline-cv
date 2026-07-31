"""Monocular 3D pose lift for the locked OL (MediaPipe Pose Landmarker).

Uses Google MediaPipe world landmarks (meters, hip-relative) on crops guided by
our 2D tracker — works for any uploaded clip, nothing clip-specific.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

# MediaPipe BlazePose 33 → names we care about for the field viewer
MP_INDEX = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

MODEL_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task",
    Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task",
]

# In-memory build status for progress UI (job_id → dict)
POSE3D_STATUS: dict[str, dict[str, Any]] = {}
_POSE3D_LOCK = threading.Lock()


def _model_path() -> Path:
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    dest = MODEL_CANDIDATES[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
    )
    print(f"  downloading MediaPipe pose model → {dest}", flush=True)
    import urllib.request

    urllib.request.urlretrieve(url, dest)
    if not dest.exists() or dest.stat().st_size < 1_000_000:
        raise FileNotFoundError("Failed to download pose_landmarker_full.task")
    return dest


def _bbox_from_joints(joints: dict[str, dict], pad: float = 0.08) -> list[float] | None:
    if len(joints) < 4:
        return None
    xs = [j["x"] for j in joints.values()]
    ys = [j["y"] for j in joints.values()]
    return [
        max(0.0, min(xs) - pad),
        max(0.0, min(ys) - pad),
        min(1.0, max(xs) + pad),
        min(1.0, max(ys) + pad),
    ]


def _norm_joints_from_analysis_frame(fr: dict[str, Any], w: float, h: float) -> dict[str, dict]:
    joints = {}
    for name, pt in (fr.get("keypoints") or {}).items():
        if not isinstance(pt, dict):
            continue
        x, y = pt.get("x"), pt.get("y")
        c = float(pt.get("confidence") or 0)
        if x is None or y is None or c < 0.2:
            continue
        joints[name] = {"x": float(x) / w, "y": float(y) / h, "c": c}
    return joints


def _crop_rgb(frame_bgr: np.ndarray, bbox_norm: list[float]) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x0 = int(bbox_norm[0] * w)
    y0 = int(bbox_norm[1] * h)
    x1 = int(bbox_norm[2] * w)
    y1 = int(bbox_norm[3] * h)
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    x0 = max(0, x0 - int(0.15 * bw))
    y0 = max(0, y0 - int(0.10 * bh))
    x1 = min(w, x1 + int(0.15 * bw))
    y1 = min(h, y1 + int(0.15 * bh))
    if x1 - x0 < 32 or y1 - y0 < 32:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
    # Cap crop size for speed
    ch, cw = rgb.shape[:2]
    max_side = 480
    if max(ch, cw) > max_side:
        scale = max_side / max(ch, cw)
        rgb = cv2.resize(rgb, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


def _world_to_dict(world_landmarks) -> dict[str, dict[str, float]]:
    out = {}
    for name, idx in MP_INDEX.items():
        if idx >= len(world_landmarks):
            continue
        lm = world_landmarks[idx]
        vis = float(getattr(lm, "visibility", 1.0) or 1.0)
        if vis < 0.2:
            continue
        out[name] = {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "v": vis,
        }
    return out


def _plant_feet(joints3d: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    hip_ys = [joints3d[n]["y"] for n in ("left_hip", "right_hip") if n in joints3d]
    ankle_ys = [joints3d[n]["y"] for n in ("left_ankle", "right_ankle") if n in joints3d]
    if hip_ys and ankle_ys and (sum(ankle_ys) / len(ankle_ys)) > (sum(hip_ys) / len(hip_ys)):
        joints3d = {
            k: {"x": j["x"], "y": -j["y"], "z": j["z"], "v": j.get("v", 1.0)}
            for k, j in joints3d.items()
        }
        ankle_ys = [-a for a in ankle_ys]
    if not ankle_ys:
        ankle_ys = [j["y"] for j in joints3d.values()]
    ground = min(ankle_ys) if ankle_ys else 0.0
    return {
        k: {"x": j["x"], "y": j["y"] - ground, "z": j["z"], "v": j.get("v", 1.0)}
        for k, j in joints3d.items()
    }


def _root_on_field(nx: float, ny: float) -> dict[str, float]:
    return {
        "x": (nx - 0.5) * 18.0,
        "y": 0.0,
        "z": (0.78 - ny) * 14.0,
    }


def _sample_frames(frames_meta: list[dict[str, Any]], max_frames: int = 40) -> list[dict[str, Any]]:
    """Evenly sample analysis frames so lift finishes in a reasonable time."""
    if len(frames_meta) <= max_frames:
        return list(frames_meta)
    idxs = np.linspace(0, len(frames_meta) - 1, max_frames)
    picked = sorted({int(round(i)) for i in idxs})
    return [frames_meta[i] for i in picked]


def _set_status(job_id: str, **kwargs: Any) -> None:
    with _POSE3D_LOCK:
        cur = POSE3D_STATUS.get(job_id) or {}
        cur.update(kwargs)
        POSE3D_STATUS[job_id] = cur


def get_pose3d_status(job_id: str) -> dict[str, Any]:
    with _POSE3D_LOCK:
        return dict(POSE3D_STATUS.get(job_id) or {"status": "idle", "percent": 0})


def lift_video_to_3d(
    video_path: str,
    analysis: dict[str, Any],
    *,
    progress_cb: Callable[[int, int, str], None] | None = None,
    max_frames: int = 40,
) -> dict[str, Any]:
    """Run MediaPipe 3D lift. Sequential decode + frame subsample for speed."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    BaseOptions = mp_python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    video = analysis.get("video") or {}
    w = float(video.get("width") or 1920)
    h = float(video.get("height") or 1080)
    fps = float(video.get("fps") or 30.0)
    frames_meta = _sample_frames(analysis.get("frames") or [], max_frames=max_frames)
    if not frames_meta:
        return {"fps": fps, "width": w, "height": h, "frames": [], "engine": "mediapipe"}

    by_idx = {int(fr.get("frame_idx") or 0): fr for fr in frames_meta}
    wanted = set(by_idx.keys())
    max_wanted = max(wanted)

    # IMAGE mode: sparse subsampled frames (VIDEO mode needs dense monotonic timestamps)
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(_model_path())),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video for 3D lift: {video_path}")

    out_frames: list[dict[str, Any]] = []
    n_target = len(wanted)
    done = 0
    cur = 0

    if progress_cb:
        progress_cb(0, n_target, "Loading MediaPipe…")

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cur <= max_wanted:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if cur in wanted:
                fr = by_idx[cur]
                t_ms = float(fr.get("timestamp_ms") or (cur / fps) * 1000.0)
                joints2d = _norm_joints_from_analysis_frame(fr, w, h)
                bbox = _bbox_from_joints(joints2d) or [0.2, 0.2, 0.8, 0.9]
                try:
                    rgb = _crop_rgb(frame, bbox)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect(mp_image)
                    if result.pose_world_landmarks:
                        world = result.pose_world_landmarks[0]
                        joints3d = _plant_feet(_world_to_dict(world))
                        if len(joints3d) >= 6:
                            hip = joints2d.get("left_hip") or joints2d.get("right_hip")
                            if hip is None:
                                hip = {"x": (bbox[0] + bbox[2]) / 2, "y": (bbox[1] + bbox[3]) / 2}
                            out_frames.append(
                                {
                                    "frame_idx": cur,
                                    "t": t_ms / 1000.0,
                                    "posture": fr.get("posture"),
                                    "bbox": bbox,
                                    "joints2d": joints2d,
                                    "joints3d": joints3d,
                                    "root": _root_on_field(hip["x"], hip["y"]),
                                }
                            )
                except Exception as exc:
                    print(f"  pose3d skip frame {cur}: {exc}", flush=True)
                done += 1
                if progress_cb:
                    progress_cb(done, n_target, f"Lifting 3D pose ({done}/{n_target})…")
            cur += 1

    cap.release()
    out_frames.sort(key=lambda f: f["frame_idx"])
    if progress_cb:
        progress_cb(n_target, n_target, "3D pose ready")
    return {
        "fps": fps,
        "width": w,
        "height": h,
        "engine": "mediapipe_pose_landmarker",
        "unit": "meters",
        "sampled": n_target,
        "frames": out_frames,
    }


def ensure_pose3d_cache(
    job_id: str,
    analysis_path: Path,
    video_path: Path,
    cache_path: Path,
    *,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("frames"):
                return data
        except Exception:
            pass
    analysis = json.loads(Path(analysis_path).read_text(encoding="utf-8"))
    payload = lift_video_to_3d(str(video_path), analysis, progress_cb=progress_cb)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def start_pose3d_job(
    job_id: str,
    analysis_path: Path,
    video_path: Path,
    cache_path: Path,
) -> dict[str, Any]:
    """Kick off background 3D lift; returns current status immediately."""
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if data.get("frames"):
                _set_status(
                    job_id,
                    status="done",
                    percent=100,
                    message="3D pose ready",
                    frames=len(data["frames"]),
                )
                return get_pose3d_status(job_id)
        except Exception:
            pass

    with _POSE3D_LOCK:
        cur = POSE3D_STATUS.get(job_id) or {}
        if cur.get("status") == "running":
            return dict(cur)

    _set_status(job_id, status="running", percent=1, message="Starting 3D lift…", frames=0)

    def _run() -> None:
        try:

            def on_prog(done: int, total: int, msg: str) -> None:
                pct = 5 + int(90 * (done / max(1, total)))
                _set_status(job_id, status="running", percent=pct, message=msg, done=done, total=total)

            payload = ensure_pose3d_cache(
                job_id, analysis_path, video_path, cache_path, progress_cb=on_prog
            )
            n = len(payload.get("frames") or [])
            if n == 0:
                _set_status(
                    job_id,
                    status="error",
                    percent=100,
                    message="No 3D frames produced — try re-running Analyze",
                    frames=0,
                )
            else:
                _set_status(
                    job_id,
                    status="done",
                    percent=100,
                    message="3D pose ready",
                    frames=n,
                )
        except Exception as exc:
            _set_status(job_id, status="error", percent=100, message=str(exc), frames=0)

    threading.Thread(target=_run, daemon=True).start()
    return get_pose3d_status(job_id)
