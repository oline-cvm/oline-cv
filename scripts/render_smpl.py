"""Windows CLI: render the reconstructed SMPL body over the film via WSL.

    python scripts/render_smpl.py --motion outputs/motion3d/footage --video footage.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.motion3d.wham_bridge import (  # noqa: E402
    WhamBridgeError,
    WhamConfig,
    run_wsl_script,
)
from oline_cv.motion3d.wsl_paths import windows_to_wsl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Render SMPL motion over the source video")
    ap.add_argument("--motion", required=True, help="dir containing motion_raw.npz")
    ap.add_argument("--video", required=True, help="source video")
    ap.add_argument("--out", default=None, help="output mp4")
    ap.add_argument("--sheet", default=None, help="contact sheet jpg")
    ap.add_argument("--sheet-count", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--distro", default=None)
    ap.add_argument("--conda-env", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    motion = Path(args.motion).resolve()
    video = Path(args.video).resolve()
    if not (motion / "motion_raw.npz").exists():
        print(f"error: no motion_raw.npz in {motion}", file=sys.stderr)
        return 2

    cfg = WhamConfig()
    if args.distro:
        cfg.distro = args.distro
    if args.conda_env:
        cfg.conda_env = args.conda_env

    out = Path(args.out).resolve() if args.out else motion / "smpl_overlay.mp4"
    sheet = Path(args.sheet).resolve() if args.sheet else motion / "smpl_sheet.jpg"

    script_args = [
        "--motion", windows_to_wsl(motion),
        "--video", windows_to_wsl(video),
        "--out", windows_to_wsl(out),
        "--sheet", windows_to_wsl(sheet),
        "--sheet-count", str(args.sheet_count),
        "--wham-root", cfg.wham_root,
    ]
    if args.max_frames:
        script_args += ["--max-frames", str(args.max_frames)]

    def progress(pct, msg, stage="render"):
        print(f"  [{stage}] {float(pct):5.1f}%  {msg}", flush=True)

    try:
        code, events, logs = run_wsl_script(
            "scripts/render_motion.py",
            script_args,
            config=cfg,
            progress_cb=progress,
            log_cb=(lambda line: print(f"    | {line}", flush=True)) if args.verbose else None,
            stage="render",
        )
    except WhamBridgeError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    if code != 0:
        detail = "; ".join(str(e.get("message")) for e in events if e.get("event") == "error")
        print(f"FAILED (exit {code}): {detail or 'see output'}", file=sys.stderr)
        print("\n".join(logs[-25:]), file=sys.stderr)
        return 1

    done = next((e for e in events if e.get("event") == "done"), {})
    print("\nRender complete")
    print(f"  frames  {done.get('frames')}")
    print(f"  video   {out}")
    if sheet.exists():
        print(f"  sheet   {sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
