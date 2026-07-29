"""OLINE dashboard — FastAPI."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
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

app = FastAPI(title="OLINE")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

JOBS: dict[str, dict[str, Any]] = {}
# In-memory last two packed results for server-side compare helpers
RECENT: list[dict[str, Any]] = []


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _pack_result(
    jersey: int | None, result: dict[str, Any], web_path: str | None, json_name: str | None
) -> dict:
    s = result.get("rep_summary", {})
    q = result.get("initial_quicks", {})
    snap = result.get("snap", {})
    packed = {
        "id": json_name or uuid.uuid4().hex[:10],
        "jersey": jersey if jersey is not None else s.get("target_jersey"),
        "ol_lock": result.get("ol_lock") or s.get("ol_lock"),
        "play_type": result.get("play_type", s.get("play_type")),
        "snap_frame": snap.get("snap_frame"),
        "reaction_time_ms": s.get("reaction_time_ms"),
        "reaction_time_frames": s.get("reaction_time_frames"),
        "initiated_by": s.get("initiated_by"),
        "late_off_the_ball": s.get("late_off_the_ball"),
        "posture_classification": s.get("posture_classification"),
        "posture_confidence": s.get("posture_confidence"),
        "mean_knee_flexion_deg": s.get("mean_knee_flexion_deg"),
        "mean_torso_angle_deg": s.get("mean_torso_angle_deg"),
        "hip_height_at_lowest": s.get("hip_height_at_lowest"),
        "step_cadence_hz": s.get("step_cadence_hz"),
        "set_depth": s.get("set_depth"),
        "set_width": s.get("set_width"),
        "mean_base_width": s.get("mean_base_width"),
        "lateral_match": s.get("lateral_match"),
        "anchor_give": s.get("anchor_give"),
        "punch_ms": s.get("punch_ms"),
        "engagement_ms": s.get("engagement_ms"),
        "coach_language": s.get("coach_language", []),
        "modules": result.get("modules", {}),
        "trust": result.get("trust") or {"overall": s.get("trust_overall")},
        "video_fps": result.get("video", {}).get("fps"),
        "overlay_url": f"/outputs/{Path(web_path).name}" if web_path else None,
        "json_url": f"/outputs/{json_name}" if json_name else None,
        "notes": q.get("notes", []),
    }
    return packed


def _remember(packed: dict[str, Any]) -> None:
    RECENT.append(packed)
    if len(RECENT) > 8:
        del RECENT[0 : len(RECENT) - 8]


@app.post("/api/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    jersey: str = Form(""),
    play_type: str = Form("pass"),
    snap_frame: int | None = Form(None),
    pick_x: str = Form(""),
    pick_y: str = Form(""),
) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"
    with video_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    jersey_i: int | None = None
    if str(jersey).strip().isdigit():
        jersey_i = int(str(jersey).strip())

    pick_xy: tuple[float, float] | None = None
    try:
        if str(pick_x).strip() != "" and str(pick_y).strip() != "":
            px, py = float(pick_x), float(pick_y)
            if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0:
                pick_xy = (px, py)
    except ValueError:
        pick_xy = None

    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "jersey": jersey_i,
        "pick_xy": pick_xy,
        "progress": "Queued",
        "result": None,
        "error": None,
    }
    background_tasks.add_task(
        _run_job, job_id, str(video_path), jersey_i, play_type, snap_frame, pick_xy
    )
    return JSONResponse({"job_id": job_id})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if not job:
        return JSONResponse({"error": "not_found"}, status_code=404)
    return JSONResponse(job)


@app.get("/api/recent")
def recent_results() -> JSONResponse:
    return JSONResponse({"results": RECENT[-8:]})


def _run_job(
    job_id: str,
    video_path: str,
    jersey: int | None,
    play_type: str,
    snap_frame: int | None,
    pick_xy: tuple[float, float] | None,
) -> None:
    job = JOBS[job_id]
    try:
        job["status"] = "running"
        if pick_xy is not None:
            job["progress"] = f"Locking click ({pick_xy[0]:.2f},{pick_xy[1]:.2f})…"
        elif jersey is not None:
            job["progress"] = f"Detecting #{jersey}…"
        else:
            job["progress"] = "Detecting offensive lineman…"
        out_json = str(OUTPUT_DIR / f"{job_id}_analysis.json")
        out_overlay = str(OUTPUT_DIR / f"{job_id}_overlay.mp4")
        cfg = AnalysisConfig(
            target_jersey=jersey,
            athlete_pick_xy=pick_xy,
            write_overlay_video=True,
            overlay_zoom_on_athlete=False,
            play_type="run" if play_type == "run" else "pass",
            pose_model="yolov8m-pose.pt",
        )
        if snap_frame is not None:
            cfg.snap_frame_override = snap_frame

        result = analyze_video(video_path, config=cfg, output_json=out_json, overlay_path=out_overlay)
        job["progress"] = "Encoding preview…"
        web_path = ensure_web_mp4(out_overlay, str(OUTPUT_DIR / f"{job_id}_web.mp4"))
        packed = _pack_result(jersey, result, web_path, Path(out_json).name)
        packed["id"] = job_id
        job["result"] = packed
        _remember(packed)
        job["status"] = "done"
        job["progress"] = "Done"
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        job["progress"] = "Failed"


@app.get("/api/demo")
def demo_existing() -> JSONResponse:
    local = ROOT / "footage_analysis.json"
    overlay = ROOT / "footage_overlay.mp4"
    if not local.exists():
        return JSONResponse({"error": "no_demo"}, status_code=404)
    data = json.loads(local.read_text(encoding="utf-8"))
    if "trust" not in data:
        from oline_cv.trust import compute_trust

        data["trust"] = compute_trust(data)
    web = ensure_web_mp4(str(overlay), str(OUTPUT_DIR / "footage_web.mp4")) if overlay.exists() else None
    packed = _pack_result(data.get("target_jersey"), data, web, None)
    packed["id"] = "demo"
    _remember(packed)
    return JSONResponse(packed)
