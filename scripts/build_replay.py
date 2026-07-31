"""Windows CLI: bake motion_raw.npz into browser assets for the 3D replay.

    python scripts/build_replay.py --motion outputs/motion3d/footage
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
    ap = argparse.ArgumentParser(description="Bake SMPL motion for the 3D replay viewer")
    ap.add_argument("--motion", required=True, help="dir containing motion_raw.npz")
    ap.add_argument("--out", default=None, help="output dir (default: --motion)")
    ap.add_argument("--per-frame-betas", action="store_true")
    ap.add_argument("--distro", default=None)
    ap.add_argument("--conda-env", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    motion = Path(args.motion).resolve()
    if not (motion / "motion_raw.npz").exists():
        print(f"error: no motion_raw.npz in {motion}", file=sys.stderr)
        return 2
    out = Path(args.out).resolve() if args.out else motion

    cfg = WhamConfig()
    if args.distro:
        cfg.distro = args.distro
    if args.conda_env:
        cfg.conda_env = args.conda_env

    script_args = [
        "--motion", windows_to_wsl(motion),
        "--out", windows_to_wsl(out),
        "--wham-root", cfg.wham_root,
    ]
    if args.per_frame_betas:
        script_args.append("--per-frame-betas")

    def progress(pct, msg, stage="bake"):
        print(f"  [{stage}] {float(pct):5.1f}%  {msg}", flush=True)

    try:
        code, events, logs = run_wsl_script(
            "scripts/export_web_motion.py",
            script_args,
            config=cfg,
            progress_cb=progress,
            log_cb=(lambda line: print(f"    | {line}", flush=True)) if args.verbose else None,
            stage="bake",
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
    print("\nReplay assets built")
    print(f"  frames            {done.get('frames')}")
    print(f"  binary            {done.get('bin')}  ({done.get('bin_mb')} MB)")
    print(f"  metadata          {done.get('json')}")
    print(f"  quantise error    {done.get('quant_error_mm')} mm")
    print(f"  ground offset     {done.get('ground_y_removed')} m removed")
    print("\n  open http://127.0.0.1:8000/replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
