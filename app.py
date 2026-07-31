"""OLINE dashboard — FastAPI."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from oline_cv.coach_brief import build_coach_brief
from oline_cv.config import AnalysisConfig
from oline_cv.pipeline import analyze_video
from oline_cv.report import build_field_pose_payload, extract_keyframes, write_coach_pdf
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


MOTION3D_DIR = OUTPUT_DIR / "motion3d"


@app.get("/motion3d", response_class=HTMLResponse)
def motion3d_page() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "motion3d.html").read_text(encoding="utf-8"))


@app.get("/viewer3d", response_class=HTMLResponse)
def viewer3d_page() -> HTMLResponse:
    """Phase 3: interactive SMPL body in Three.js (blank scene)."""
    return HTMLResponse((STATIC_DIR / "viewer3d.html").read_text(encoding="utf-8"))


@app.get("/api/motion3d/{clip}")
def motion3d_data(clip: str) -> JSONResponse:
    """Review payload for one reconstructed clip: metadata + world-space checks.

    Read-only summary of Phase 2 artifacts; the heavy arrays stay in the npz.
    """
    import numpy as np

    from oline_cv.motion3d.motion_schema import load_metadata
    from oline_cv.motion3d.schema import load_manifest
    from oline_cv.motion3d.world_checks import world_report

    base = (MOTION3D_DIR / clip).resolve()
    if not str(base).startswith(str(MOTION3D_DIR.resolve())):
        return JSONResponse({"error": "invalid clip"}, status_code=400)
    npz_path = base / "motion_raw.npz"
    meta_path = base / "motion_metadata.json"
    if not npz_path.exists() or not meta_path.exists():
        return JSONResponse({"error": f"no reconstruction for {clip}"}, status_code=404)

    meta = load_metadata(meta_path)
    payload: dict[str, Any] = {"clip": clip, "metadata": meta.to_dict()}

    tracks = base / "tracks.json"
    if tracks.exists():
        try:
            payload["target"] = load_manifest(tracks).target
        except Exception:
            payload["target"] = None

    with np.load(npz_path, allow_pickle=False) as d:
        payload["arrays"] = [
            {"name": k, "shape": list(d[k].shape)} for k in sorted(d.files)
        ]
        payload["confidence"] = (
            [round(float(v), 4) for v in d["frame_confidence"]]
            if "frame_confidence" in d.files else []
        )
        payload["interpolated"] = (
            [int(v) for v in d["interpolated"]] if "interpolated" in d.files else []
        )

        payload["world"] = world_report(
            d["pose_world"], d["pose_cam"], d["trans_world"], fps=meta.fps
        )

    return JSONResponse(payload)


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
    brief = build_coach_brief(packed)
    packed["brief"] = brief
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
        "percent": 0,
        "stage": "queued",
        "stages_done": [],
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


STAGE_ORDER = ("ingest", "lock", "track", "metrics", "overlay", "encode", "done")


def _set_progress(job: dict[str, Any], percent: float, message: str, stage: str) -> None:
    job["percent"] = int(max(0, min(100, round(percent))))
    job["progress"] = message
    job["stage"] = stage
    done = list(job.get("stages_done") or [])
    # Mark prior stages complete when we advance
    if stage in STAGE_ORDER:
        idx = STAGE_ORDER.index(stage)
        for s in STAGE_ORDER[:idx]:
            if s not in done:
                done.append(s)
    job["stages_done"] = done


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
            _set_progress(
                job, 3, f"Locking click ({pick_xy[0]:.2f},{pick_xy[1]:.2f})…", "lock"
            )
        elif jersey is not None:
            _set_progress(job, 3, f"Detecting #{jersey}…", "lock")
        else:
            _set_progress(job, 3, "Detecting offensive lineman…", "lock")

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

        def on_progress(pct: float, msg: str, stage: str = "analyze") -> None:
            _set_progress(job, pct, msg, stage)

        result = analyze_video(
            video_path,
            config=cfg,
            output_json=out_json,
            overlay_path=out_overlay,
            progress_cb=on_progress,
        )
        _set_progress(job, 95, "Encoding web preview…", "encode")
        web_path = ensure_web_mp4(out_overlay, str(OUTPUT_DIR / f"{job_id}_web.mp4"))

        # Keyframe stills + PDF coach report
        _set_progress(job, 97, "Building coach PDF report…", "encode")
        kf_dir = OUTPUT_DIR / f"{job_id}_keyframes"
        # Prefer overlay film so stills include skeleton
        still_src = out_overlay if Path(out_overlay).exists() else video_path
        keyframes = extract_keyframes(still_src, result, kf_dir)
        for kf in keyframes:
            kf["url"] = f"/outputs/{job_id}_keyframes/{Path(kf['path']).name}"

        packed = _pack_result(jersey, result, web_path, Path(out_json).name)
        packed["id"] = job_id
        packed["keyframes"] = [{"label": k["label"], "frame_idx": k["frame_idx"], "url": k["url"]} for k in keyframes]

        pdf_path = OUTPUT_DIR / f"{job_id}_report.pdf"
        try:
            write_coach_pdf(packed, result, keyframes, pdf_path)
            packed["report_url"] = f"/outputs/{pdf_path.name}"
        except Exception as pdf_exc:
            packed["report_url"] = None
            packed["report_error"] = str(pdf_exc)

        job["video_path"] = video_path
        job["analysis_json"] = out_json
        job["result"] = packed
        _remember(packed)
        job["status"] = "done"
        _set_progress(job, 100, "Done", "done")
        job["stages_done"] = list(STAGE_ORDER)
    except Exception as exc:
        job["status"] = "error"
        job["error"] = str(exc)
        job["progress"] = "Failed"
        job["stage"] = "error"


@app.get("/api/jobs/{job_id}/report.pdf")
def job_report_pdf(job_id: str):
    """Download coach PDF (regenerate if missing)."""
    job = JOBS.get(job_id)
    pdf_path = OUTPUT_DIR / f"{job_id}_report.pdf"
    if pdf_path.exists():
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"oline_report_{job_id}.pdf",
        )
    if not job or not job.get("result"):
        # try analysis json on disk
        analysis = OUTPUT_DIR / f"{job_id}_analysis.json"
        if not analysis.exists():
            return JSONResponse({"error": "not_found"}, status_code=404)
        full = json.loads(analysis.read_text(encoding="utf-8"))
        packed = _pack_result(full.get("target_jersey"), full, None, analysis.name)
        packed["id"] = job_id
        kf_dir = OUTPUT_DIR / f"{job_id}_keyframes"
        video_guess = next(UPLOAD_DIR.glob(f"{job_id}.*"), None)
        keyframes = []
        if video_guess:
            keyframes = extract_keyframes(str(video_guess), full, kf_dir)
        write_coach_pdf(packed, full, keyframes, pdf_path)
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            filename=f"oline_report_{job_id}.pdf",
        )
    # regenerate from in-memory job
    analysis = Path(job.get("analysis_json") or OUTPUT_DIR / f"{job_id}_analysis.json")
    full = json.loads(analysis.read_text(encoding="utf-8")) if analysis.exists() else {}
    packed = job["result"]
    kf_dir = OUTPUT_DIR / f"{job_id}_keyframes"
    keyframes = []
    if kf_dir.exists():
        for p in sorted(kf_dir.glob("kf_*.jpg")):
            keyframes.append({"path": str(p), "label": p.stem, "frame_idx": 0})
    write_coach_pdf(packed, full or packed, keyframes, pdf_path)
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"oline_report_{job_id}.pdf",
    )


def _job_analysis_path(job_id: str) -> Path | None:
    analysis = OUTPUT_DIR / f"{job_id}_analysis.json"
    if analysis.exists():
        return analysis
    job = JOBS.get(job_id)
    if job and job.get("analysis_json") and Path(job["analysis_json"]).exists():
        return Path(job["analysis_json"])
    return None


def _job_video_url(job_id: str, packed: dict[str, Any] | None = None) -> str | None:
    packed = packed or {}
    video_url = packed.get("overlay_url")
    if video_url:
        return video_url
    web = OUTPUT_DIR / f"{job_id}_web.mp4"
    if web.exists():
        return f"/outputs/{web.name}"
    if (OUTPUT_DIR / f"{job_id}_overlay.mp4").exists():
        return f"/outputs/{job_id}_overlay.mp4"
    return None


def _job_source_video(job_id: str, video_url: str | None = None) -> Path | None:
    """Prefer original upload for MediaPipe (cleaner than burned overlay)."""
    job = JOBS.get(job_id)
    if job and job.get("video_path") and Path(job["video_path"]).exists():
        return Path(job["video_path"])
    hits = list(UPLOAD_DIR.glob(f"{job_id}.*"))
    if hits:
        return hits[0]
    if video_url:
        cand = OUTPUT_DIR / Path(video_url).name
        if cand.exists():
            return cand
    return None


def _read_pose3d_cache(cache: Path) -> dict[str, Any] | None:
    if not cache.exists():
        return None
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if data.get("frames"):
            return data
    except Exception:
        pass
    return None


@app.get("/api/jobs/{job_id}/pose3d")
def job_pose3d_status(job_id: str) -> JSONResponse:
    """Start / poll MediaPipe 3D lift (async — never blocks the HTTP worker)."""
    from oline_cv.pose3d import get_pose3d_status, start_pose3d_job

    analysis = _job_analysis_path(job_id)
    if analysis is None:
        return JSONResponse({"error": "not_found", "status": "error"}, status_code=404)

    job = JOBS.get(job_id)
    packed = (job or {}).get("result") or {}
    video_url = _job_video_url(job_id, packed)
    video_path = _job_source_video(job_id, video_url)
    cache = OUTPUT_DIR / f"{job_id}_pose3d.json"

    cached = _read_pose3d_cache(cache)
    if cached is not None:
        return JSONResponse(
            {
                "status": "done",
                "percent": 100,
                "message": "3D pose ready",
                "frames": len(cached["frames"]),
            }
        )

    if video_path is None:
        return JSONResponse(
            {
                "status": "error",
                "percent": 100,
                "message": "No source video for 3D lift",
                "frames": 0,
            },
            status_code=400,
        )

    status = start_pose3d_job(job_id, analysis, video_path, cache)
    return JSONResponse(status)


@app.get("/api/demo/pose3d")
def demo_pose3d_status() -> JSONResponse:
    from oline_cv.pose3d import start_pose3d_job

    local = ROOT / "footage_analysis.json"
    if not local.exists():
        return JSONResponse({"error": "no_demo", "status": "error"}, status_code=404)
    video_path = ROOT / "footage.mp4"
    if not video_path.exists():
        overlay = ROOT / "footage_overlay.mp4"
        video_path = overlay if overlay.exists() else None
    if video_path is None:
        return JSONResponse(
            {"status": "error", "percent": 100, "message": "No demo video", "frames": 0},
            status_code=400,
        )
    cache = OUTPUT_DIR / "demo_pose3d.json"
    status = start_pose3d_job("demo", local, video_path, cache)
    return JSONResponse(status)


@app.get("/api/jobs/{job_id}/field-data")
def job_field_data(job_id: str) -> JSONResponse:
    """Field payload. Reads pose3d cache only — never runs MediaPipe inline."""
    from oline_cv.report import build_field_pose_payload

    analysis = _job_analysis_path(job_id)
    if analysis is None:
        return JSONResponse({"error": "not_found"}, status_code=404)

    full = json.loads(analysis.read_text(encoding="utf-8"))
    job = JOBS.get(job_id)
    packed = (job or {}).get("result") or {}
    video_url = _job_video_url(job_id, packed)

    pose3d = _read_pose3d_cache(OUTPUT_DIR / f"{job_id}_pose3d.json")
    payload = build_field_pose_payload(full, video_url=video_url)
    payload["jersey"] = packed.get("jersey") or full.get("target_jersey")
    payload["brief"] = packed.get("brief") or build_coach_brief(
        _pack_result(full.get("target_jersey"), full, None, None)
    )
    payload["pose3d"] = pose3d
    payload["mode"] = "3d" if pose3d and pose3d.get("frames") else "building"
    return JSONResponse(payload)


@app.get("/api/demo/field-data")
def demo_field_data() -> JSONResponse:
    from oline_cv.report import build_field_pose_payload

    local = ROOT / "footage_analysis.json"
    if not local.exists():
        return JSONResponse({"error": "no_demo"}, status_code=404)
    full = json.loads(local.read_text(encoding="utf-8"))
    web = OUTPUT_DIR / "footage_web.mp4"
    overlay = ROOT / "footage_overlay.mp4"
    video_url = None
    if web.exists():
        video_url = f"/outputs/{web.name}"
    elif overlay.exists():
        try:
            ensure_web_mp4(str(overlay), str(web))
            video_url = f"/outputs/{web.name}"
        except Exception:
            video_url = None

    pose3d = _read_pose3d_cache(OUTPUT_DIR / "demo_pose3d.json")
    payload = build_field_pose_payload(full, video_url=video_url)
    payload["jersey"] = full.get("target_jersey")
    payload["brief"] = build_coach_brief(_pack_result(full.get("target_jersey"), full, None, None))
    payload["pose3d"] = pose3d
    payload["mode"] = "3d" if pose3d and pose3d.get("frames") else "building"
    return JSONResponse(payload)


@app.get("/field")
def field_page():
    # The MediaPipe capsule avatar is the old path. When a WHAM SMPL mesh pack
    # exists, send people to the real reconstruction viewer instead.
    mesh = MOTION3D_DIR / "footage" / "mesh_threejs.bin"
    if mesh.exists():
        return RedirectResponse(url="/viewer3d?clip=footage", status_code=302)
    return HTMLResponse((STATIC_DIR / "field.html").read_text(encoding="utf-8"))


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
    packed["field_job_id"] = "demo"
    _remember(packed)
    return JSONResponse(packed)
