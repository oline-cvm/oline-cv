"""Check that motion_raw.npz world output is actually gravity-aligned.

`world_grounded: true` in the metadata only records that DPVO ran. This measures
the property that matters: in a gravity-aligned frame the body's up axis stays
near +Y regardless of where the camera points, so its angle to +Y should be both
small and stable. In camera space the same axis tumbles as the camera pans, which
gives us a built-in control to compare against.

    python scripts/verify_world.py outputs/motion3d/footage
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.motion3d.world_checks import (  # noqa: E402
    MAX_WORLD_UP_STD_DEG,
    world_report,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify world-space output")
    ap.add_argument("motion", help="dir containing motion_raw.npz")
    ap.add_argument("--max-up-std", type=float, default=MAX_WORLD_UP_STD_DEG,
                    help="fail if world up-axis angle std exceeds this (deg)")
    args = ap.parse_args()

    npz = Path(args.motion) / "motion_raw.npz"
    if not npz.exists():
        print(f"error: {npz} not found", file=sys.stderr)
        return 2
    d = np.load(npz)

    required = ("pose_world", "trans_world", "pose_cam", "trans_cam")
    missing = [k for k in required if k not in d.files]
    if missing:
        print(f"error: missing arrays {missing}", file=sys.stderr)
        return 2

    fps = 30.0
    if "timestamps" in d.files and len(d["timestamps"]) > 1:
        dt = float(np.median(np.diff(d["timestamps"])))
        if dt > 1e-6:
            fps = 1.0 / dt

    r = world_report(d["pose_world"], d["pose_cam"], d["trans_world"], fps=fps)
    span = r["span_m"]

    print(f"frames            {r['frames']}  @ {r['fps']:.2f} fps")
    print(f"trans_world span  x {span[0]:.3f}  y {span[1]:.3f}  z {span[2]:.3f}  (m)")
    print(f"horizontal path   {r['path_length_m']:.2f} m   "
          f"net displacement {r['net_displacement_m']:.2f} m")
    print(f"peak speed        {r['peak_speed_ms']:.2f} m/s   "
          f"mean {r['mean_speed_ms']:.2f} m/s")
    print()
    print("Body up-axis angle to +Y (gravity alignment)")
    print(f"  world  mean {r['world_up_mean']:6.2f}deg  std {r['world_up_std']:5.2f}  "
          f"range {r['world_up_range'][0]:.1f}-{r['world_up_range'][1]:.1f}")
    print(f"  camera mean {r['cam_up_mean']:6.2f}deg  std {r['cam_up_std']:5.2f}  "
          f"range {r['cam_up_range'][0]:.1f}-{r['cam_up_range'][1]:.1f}")

    if "contact" in d.files:
        print(f"\nmean foot contact {d['contact'].mean(axis=0).round(3).tolist()}")
    if "interpolated" in d.files:
        interp = np.flatnonzero(d["interpolated"] > 0)
        idx = d["frame_indices"]
        print(f"interpolated      {len(interp)} frames "
              f"{idx[interp].tolist() if len(interp) else ''}")
    if "frame_confidence" in d.files:
        fc = d["frame_confidence"]
        print(f"confidence        mean {fc.mean():.3f}  min {fc.min():.3f}")

    ok = True
    if not r["finite"]:
        print("\nFAIL: trans_world contains non-finite values")
        ok = False
    if r["world_up_std"] > args.max_up_std:
        print(f"\nFAIL: world up-axis std {r['world_up_std']:.2f}deg exceeds "
              f"{args.max_up_std}deg — trajectory does not look gravity-aligned")
        ok = False
    if not r["steadier_than_camera"]:
        print("\nWARN: world up axis is no more stable than camera space; "
              "check that SLAM actually contributed")

    print("\nPASS: world output is present and gravity-aligned" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
