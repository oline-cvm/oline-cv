"""OLINE dashboard — FastAPI app for pass-set analysis."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from oline_cv.config import AnalysisConfig
from oline_cv.pipeline import analyze_video
from oline_cv.web_video import ensure_web_mp4

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
OUTPUT_DIR = ROOT / "outputs"
STATIC_DIR = ROOT / "dashboard"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="OLINE", description="Offensive lineman pass-set analysis")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# In-memory job store
JOBS: dict[str, dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    jersey: int = Form(76),
    snap_frame: int | None = Form(None),
) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"
    with video_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "jersey": jersey,
        "progress": "Queued",
        "result": None,
        "error": None,
    }

    background_tasks.add_task(_run_job, job_id, str(video_path), jersey, snap_frame)
    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return JSONResponse(job)


def _run_job(job_id: str, video_path: str, jersey: int, snap_frame: int | None) -> None:
    job = JOBS[job_id]
    try:
        job["status"] = "running"
        job["progress"] = f"Tracking #{jersey}…"

        out_json = str(OUTPUT_DIR / f"{job_id}_analysis.json")
        out_overlay = str(OUTPUT_DIR / f"{job_id}_overlay.mp4")

        cfg = AnalysisConfig(
            target_jersey=jersey,
            write_overlay_video=True,
            overlay_zoom_on_athlete=True,
            pose_model="yolov8n-pose.pt",
        )
        if jersey == 76 and cfg.athlete_pick_xy is None:
            cfg.athlete_pick_xy = (0.272, 0.53)
        if snap_frame is not None:
            cfg.snap_frame_override = snap_frame

        result = analyze_video(
            video_path,
            config=cfg,
            output_json=out_json,
            overlay_path=out_overlay,
        )

        job["progress"] = "Encoding preview…"
        web_path = ensure_web_mp4(out_overlay, str(OUTPUT_DIR / f"{job_id}_web.mp4"))

        summary = result.get("rep_summary", {})
        quicks = result.get("initial_quicks", {})
        snap = result.get("snap", {})

        job["result"] = {
            "jersey": jersey,
            "snap_frame": snap.get("snap_frame"),
            "reaction_time_ms": summary.get("reaction_time_ms"),
            "reaction_time_frames": summary.get("reaction_time_frames"),
            "initiated_by": summary.get("initiated_by"),
            "posture_classification": summary.get("posture_classification"),
            "mean_knee_flexion_deg": summary.get("mean_knee_flexion_deg"),
            "min_knee_flexion_deg": summary.get("min_knee_flexion_deg"),
            "mean_torso_angle_deg": summary.get("mean_torso_angle_deg"),
            "max_torso_angle_deg": summary.get("max_torso_angle_deg"),
            "hip_height_at_lowest": summary.get("hip_height_at_lowest"),
            "mean_hip_height": summary.get("mean_hip_height"),
            "first_foot_movement_frame": quicks.get("first_foot_movement_frame"),
            "first_hip_movement_frame": quicks.get("first_hip_movement_frame"),
            "posture_frame_counts": summary.get("posture_frame_counts"),
            "video_fps": result.get("video", {}).get("fps"),
            "overlay_url": f"/outputs/{Path(web_path).name}",
            "json_url": f"/outputs/{Path(out_json).name}",
            "notes": quicks.get("notes", []),
        }
        job["status"] = "done"
        job["progress"] = "Done"
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        job["progress"] = "Failed"


@app.get("/api/demo")
def demo_existing() -> JSONResponse:
    """Load last local footage analysis if present."""
    local = ROOT / "footage_analysis.json"
    overlay = ROOT / "footage_overlay.mp4"
    if not local.exists():
        return JSONResponse({"error": "no_demo"}, status_code=404)
    data = json.loads(local.read_text(encoding="utf-8"))
    web = ensure_web_mp4(str(overlay), str(OUTPUT_DIR / "footage_web.mp4")) if overlay.exists() else None
    s = data.get("rep_summary", {})
    q = data.get("initial_quicks", {})
    return JSONResponse(
        {
            "jersey": 76,
            "snap_frame": data.get("snap", {}).get("snap_frame"),
            "reaction_time_ms": s.get("reaction_time_ms"),
            "reaction_time_frames": s.get("reaction_time_frames"),
            "initiated_by": s.get("initiated_by"),
            "posture_classification": s.get("posture_classification"),
            "mean_knee_flexion_deg": s.get("mean_knee_flexion_deg"),
            "min_knee_flexion_deg": s.get("min_knee_flexion_deg"),
            "mean_torso_angle_deg": s.get("mean_torso_angle_deg"),
            "max_torso_angle_deg": s.get("max_torso_angle_deg"),
            "hip_height_at_lowest": s.get("hip_height_at_lowest"),
            "mean_hip_height": s.get("mean_hip_height"),
            "first_foot_movement_frame": q.get("first_foot_movement_frame"),
            "first_hip_movement_frame": q.get("first_hip_movement_frame"),
            "posture_frame_counts": s.get("posture_frame_counts"),
            "video_fps": data.get("video", {}).get("fps"),
            "overlay_url": f"/outputs/{Path(web).name}" if web else None,
            "json_url": None,
            "notes": q.get("notes", []),
        }
    )
