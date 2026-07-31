"""Windows CLI: bake SMPL mesh for the Three.js viewer via WSL.

    python scripts/bake_mesh.py --motion outputs/motion3d/footage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.motion3d.wham_bridge import WhamBridgeError, WhamConfig, run_wsl_script  # noqa: E402
from oline_cv.motion3d.wsl_paths import windows_to_wsl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Bake SMPL mesh for Three.js")
    ap.add_argument("--motion", required=True, help="dir containing motion_raw.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    motion = Path(args.motion).resolve()
    if not (motion / "motion_raw.npz").exists():
        print(f"error: no motion_raw.npz in {motion}", file=sys.stderr)
        return 2

    out = Path(args.out).resolve() if args.out else motion / "mesh_threejs.bin"
    cfg = WhamConfig()
    script_args = [
        "--motion", windows_to_wsl(motion),
        "--out", windows_to_wsl(out),
        "--wham-root", cfg.wham_root,
    ]

    def progress(pct, msg, stage="bake"):
        print(f"  [{stage}] {float(pct):5.1f}%  {msg}", flush=True)

    try:
        code, events, logs = run_wsl_script(
            "scripts/bake_smpl_mesh.py",
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
        print("\n".join(logs[-20:]), file=sys.stderr)
        return 1

    done = next((e for e in events if e.get("event") == "done"), {})
    print("\nMesh bake complete")
    print(f"  frames  {done.get('frames')}")
    print(f"  verts   {done.get('n_verts')} × {done.get('frames')}")
    print(f"  faces   {done.get('n_faces')}")
    print(f"  size    {done.get('bytes', 0) / 1e6:.1f} MB")
    print(f"  out     {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
