"""Windows -> WSL bridge for the WHAM reconstruction stage.

This is the ONLY place the Windows application talks to WHAM, and it does so
strictly through a subprocess:

    Windows Python
      -> subprocess.run(wsl.exe ...)
        -> bash -lc "conda activate <env> && cd <wham_root> && python run_wham_manifest.py"
          -> motion_raw.npz + motion_metadata.json

Nothing in this module imports torch, mmpose, or WHAM. It builds an argv list
(never a shell string on the Windows side, so PowerShell cannot mangle it),
streams the child's structured log lines, and validates the artifacts that come
back.

The runner emits machine-readable progress as lines of the form

    @@OLINE@@ {"event": "progress", "percent": 42.0, "message": "..."}

interleaved with WHAM's own noisy stdout. We parse only our own prefix, so
upstream logging changes cannot break progress reporting.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from oline_cv.motion3d.motion_schema import (
    MotionMetadata,
    load_metadata,
    validate_motion_npz,
)
from oline_cv.motion3d.wsl_paths import quote_posix, windows_to_wsl

LOG_PREFIX = "@@OLINE@@"

# This box has two distros; the WHAM install (checkpoints, body models, wham39)
# lives in Ubuntu-22.04, while "Ubuntu" is the default and has a bare clone.
# Probing the default silently reports a broken environment, so pin it here.
DEFAULT_DISTRO = "Ubuntu-22.04"
DEFAULT_CONDA_ENV = "wham39"
DEFAULT_WHAM_ROOT = "/home/rishul/WHAM"
DEFAULT_CONDA_SH = "~/miniconda3/etc/profile.d/conda.sh"
RUNNER_RELPATH = "scripts/run_wham_manifest.py"


class WhamBridgeError(RuntimeError):
    """Base class for reconstruction bridge failures."""


class WhamEnvironmentError(WhamBridgeError):
    """WSL, conda, WHAM, or a required weight file is unavailable."""


class WhamRuntimeError(WhamBridgeError):
    """The WSL job started but exited non-zero."""


class WhamOutputError(WhamBridgeError):
    """The job reported success but the artifacts are missing or malformed."""


@dataclass
class WhamConfig:
    """Where WHAM lives inside WSL and how to activate it."""

    distro: str = DEFAULT_DISTRO
    conda_env: str = DEFAULT_CONDA_ENV
    wham_root: str = DEFAULT_WHAM_ROOT
    conda_sh: str = DEFAULT_CONDA_SH
    device: str = "cuda"
    flip_eval: bool = True
    keypoints_source: str = "vitpose"  # vitpose | manifest
    run_slam: bool = True
    save_verts: bool = False
    timeout_seconds: int = 3600
    # Identity association against the TrackManifest target. Disabling this makes
    # the run trust the BoT-SORT box blindly, so it stays opt-out and recorded.
    associate: bool = True
    assoc_min_iou: float | None = None
    assoc_max_center: float | None = None
    assoc_min_score: float | None = None
    assoc_reject_ambiguous: bool = False
    max_bridge_gap: int | None = None
    min_frames: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "distro": self.distro,
            "conda_env": self.conda_env,
            "wham_root": self.wham_root,
            "device": self.device,
            "flip_eval": self.flip_eval,
            "keypoints_source": self.keypoints_source,
            "run_slam": self.run_slam,
            "associate": self.associate,
        }


@dataclass
class WhamJobResult:
    returncode: int
    metadata: MotionMetadata | None
    npz_path: Path | None
    metadata_path: Path | None
    npz_summary: dict[str, Any] = field(default_factory=dict)
    log_lines: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.metadata is not None and self.metadata.ok


def _wsl_exe() -> str:
    exe = shutil.which("wsl.exe") or shutil.which("wsl")
    if not exe:
        raise WhamEnvironmentError(
            "wsl.exe not found on PATH — the WHAM stage requires WSL2"
        )
    return exe


def build_remote_script(
    config: WhamConfig,
    runner_posix: str,
    args: list[str],
) -> str:
    """The bash -lc payload. Kept as one auditable string for logging."""
    arg_str = " ".join(quote_posix(a) if _needs_quote(a) else a for a in args)
    return (
        f"set -o pipefail; "
        f"source {config.conda_sh} >/dev/null 2>&1 || true; "
        f"conda activate {quote_posix(config.conda_env)} || "
        f"{{ echo '{LOG_PREFIX} {{\"event\":\"error\",\"message\":\"conda activate failed\"}}'; exit 78; }}; "
        f"cd {quote_posix(config.wham_root)} || "
        f"{{ echo '{LOG_PREFIX} {{\"event\":\"error\",\"message\":\"wham_root missing\"}}'; exit 79; }}; "
        f"exec python -u {quote_posix(runner_posix)} {arg_str}"
    )


def _needs_quote(arg: str) -> bool:
    return not arg.startswith("--") or "=" in arg or " " in arg


def build_command(
    config: WhamConfig,
    runner_posix: str,
    args: list[str],
) -> list[str]:
    """argv for subprocess. A list (not a string) so no shell re-tokenizes it."""
    return [
        _wsl_exe(),
        "-d",
        config.distro,
        "--",
        "bash",
        "-lc",
        build_remote_script(config, runner_posix, args),
    ]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _runner_posix(relpath: str = RUNNER_RELPATH) -> str:
    runner = _repo_root() / relpath
    if not runner.exists():
        raise WhamEnvironmentError(f"runner script not found: {runner}")
    return windows_to_wsl(runner)


def run_wsl_script(
    script_relpath: str,
    args: list[str],
    config: WhamConfig | None = None,
    progress_cb: Callable[[float, str, str], None] | None = None,
    log_cb: Callable[[str], None] | None = None,
    stage: str = "wham",
) -> tuple[int, list[dict[str, Any]], list[str]]:
    """Run any repo script inside the WHAM env, streaming our structured events.

    Shares one activation path with the reconstruction so a working run and a
    working render can never disagree about which env or distro they used.
    """
    config = config or WhamConfig()
    cmd = build_command(config, _runner_posix(script_relpath), args)

    log_lines: list[str] = []
    events: list[dict[str, Any]] = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise WhamEnvironmentError(f"could not launch WSL: {exc}") from exc

    try:
        for line in _stream(proc):
            log_lines.append(line)
            if log_cb is not None:
                try:
                    log_cb(line)
                except Exception:
                    pass
            ev = parse_event(line)
            if ev is None:
                continue
            events.append(ev)
            if ev.get("event") == "progress" and progress_cb is not None:
                try:
                    progress_cb(
                        float(ev.get("percent") or 0.0), str(ev.get("message") or ""), stage
                    )
                except Exception:
                    pass
        returncode = proc.wait(timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise WhamRuntimeError(f"{script_relpath} exceeded {config.timeout_seconds}s") from exc

    return returncode, events, log_lines


def _stream(proc: subprocess.Popen) -> Iterator[str]:
    assert proc.stdout is not None
    for raw in proc.stdout:
        yield raw.rstrip("\r\n")


def parse_event(line: str) -> dict[str, Any] | None:
    """Extract one of our structured log events, if this line is one."""
    idx = line.find(LOG_PREFIX)
    if idx < 0:
        return None
    payload = line[idx + len(LOG_PREFIX) :].strip()
    try:
        obj = json.loads(payload)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def doctor(config: WhamConfig | None = None) -> dict[str, Any]:
    """Preflight the WSL side. Never raises for a missing dependency.

    Returns the runner's environment report so the caller can tell the user
    exactly which piece is absent instead of failing mid-reconstruction.
    """
    config = config or WhamConfig()
    cmd = build_command(config, _runner_posix(), ["--doctor"])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
    except FileNotFoundError as exc:
        return {"ok": False, "error": f"wsl not launchable: {exc}"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "doctor timed out after 180s"}

    report: dict[str, Any] = {}
    for line in (proc.stdout or "").splitlines():
        ev = parse_event(line)
        if ev and ev.get("event") == "doctor":
            report = ev.get("report") or {}
    if not report:
        return {
            "ok": False,
            "error": "no doctor report returned",
            "returncode": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
    report["ok"] = bool(report.get("ready"))
    return report


def build_job_args(
    tracks_posix: str,
    video_posix: str,
    out_posix: str,
    repo_posix: str,
    segment: tuple[int, int] | None,
    config: WhamConfig,
) -> list[str]:
    """argv for run_wham_manifest.py. Split out so it can be asserted on."""
    args = [
        "--tracks",
        tracks_posix,
        "--video",
        video_posix,
        "--out",
        out_posix,
        "--repo",
        repo_posix,
        "--device",
        config.device,
        "--keypoints",
        config.keypoints_source,
    ]
    if segment is not None:
        args += ["--segment", f"{int(segment[0])}:{int(segment[1])}"]
    else:
        args += ["--auto-segment"]
    if not config.flip_eval:
        args.append("--no-flip-eval")
    if not config.run_slam:
        args.append("--no-slam")
    if config.save_verts:
        args.append("--save-verts")
    if not config.associate:
        args.append("--no-associate")
    if config.assoc_reject_ambiguous:
        args.append("--assoc-reject-ambiguous")
    for flag, value in (
        ("--assoc-min-iou", config.assoc_min_iou),
        ("--assoc-max-center", config.assoc_max_center),
        ("--assoc-min-score", config.assoc_min_score),
        ("--max-bridge-gap", config.max_bridge_gap),
        ("--min-frames", config.min_frames),
    ):
        if value is not None:
            args += [flag, str(value)]
    return args


def run_wham_job(
    tracks_json: str | Path,
    video: str | Path,
    out_dir: str | Path,
    segment: tuple[int, int] | None = None,
    config: WhamConfig | None = None,
    progress_cb: Callable[[float, str, str], None] | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> WhamJobResult:
    """Run the reconstruction for one clip segment and validate the artifacts.

    ``segment`` pins the frame range (e.g. ``(0, 164)``); ``None`` lets the
    runner pick the best reconstructable segment from the manifest.
    """
    config = config or WhamConfig()
    # Resolve before translating: a relative path has no drive letter to map, so
    # it would cross into WSL unchanged and resolve against the WHAM root.
    tracks_json = Path(tracks_json).resolve()
    out_dir = Path(out_dir).resolve()
    if not tracks_json.exists():
        raise WhamEnvironmentError(f"tracks.json not found: {tracks_json}")
    video_path = Path(video).resolve()
    if not video_path.exists():
        raise WhamEnvironmentError(f"source video not found: {video_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    args = build_job_args(
        windows_to_wsl(tracks_json),
        windows_to_wsl(video_path),
        windows_to_wsl(out_dir),
        windows_to_wsl(_repo_root()),
        segment,
        config,
    )

    cmd = build_command(config, _runner_posix(), args)

    def _emit(pct: float, msg: str, stage: str = "wham") -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(float(pct), str(msg), str(stage))
        except Exception:
            pass

    _emit(1.0, "Launching WHAM in WSL…")
    started = time.time()
    log_lines: list[str] = []
    events: list[dict[str, Any]] = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except FileNotFoundError as exc:
        raise WhamEnvironmentError(f"could not launch WSL: {exc}") from exc

    try:
        for line in _stream(proc):
            log_lines.append(line)
            if log_cb is not None:
                try:
                    log_cb(line)
                except Exception:
                    pass
            ev = parse_event(line)
            if ev is None:
                continue
            events.append(ev)
            kind = ev.get("event")
            if kind == "progress":
                _emit(float(ev.get("percent") or 0.0), str(ev.get("message") or "Working…"))
            elif kind == "error":
                _emit(100.0, f"WHAM error: {ev.get('message')}")
        returncode = proc.wait(timeout=config.timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise WhamRuntimeError(
            f"WHAM job exceeded {config.timeout_seconds}s and was killed"
        ) from exc

    elapsed = time.time() - started
    npz_path = out_dir / "motion_raw.npz"
    meta_path = out_dir / "motion_metadata.json"

    metadata: MotionMetadata | None = None
    if meta_path.exists():
        try:
            metadata = load_metadata(meta_path)
        except Exception as exc:
            raise WhamOutputError(f"motion_metadata.json is unreadable: {exc}") from exc

    if returncode == 78:
        raise WhamEnvironmentError(
            f"conda env {config.conda_env!r} could not be activated in {config.distro}"
        )
    if returncode == 79:
        raise WhamEnvironmentError(f"WHAM root {config.wham_root!r} not found in {config.distro}")
    if returncode != 0:
        detail = "; ".join(str(e.get("message")) for e in events if e.get("event") == "error")
        tail = "\n".join(log_lines[-25:])
        raise WhamRuntimeError(
            f"WHAM job failed (exit {returncode})"
            + (f": {detail}" if detail else "")
            + f"\n--- last output ---\n{tail}"
        )

    if metadata is None:
        raise WhamOutputError(f"job exited 0 but {meta_path.name} was not written")
    if not metadata.ok:
        raise WhamOutputError(
            f"reconstruction status is {metadata.status!r}: {'; '.join(metadata.errors) or 'no detail'}"
        )

    summary = validate_motion_npz(npz_path, expected_frames=metadata.frame_count)
    _emit(100.0, f"Reconstruction complete ({summary['frames']} frames in {elapsed:.0f}s)")

    return WhamJobResult(
        returncode=returncode,
        metadata=metadata,
        npz_path=npz_path,
        metadata_path=meta_path,
        npz_summary=summary,
        log_lines=log_lines,
        events=events,
    )
