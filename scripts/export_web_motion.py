"""Bake the reconstructed SMPL body into a browser-loadable mesh animation.

Runs inside WSL, where the SMPL model lives. The browser has no SMPL, so rather
than shipping the model plus skinning weights and re-deriving the body in JS
(which risks the viewer disagreeing with the reconstruction), we evaluate SMPL
here and ship the resulting vertices. What the viewer draws is then exactly what
WHAM produced.

Cost: 6890 verts x 3 x int16 per frame, about 41 KB/frame, so ~7 MB for a 165
frame rep. Fine for a local review tool; a longer clip should stream or move to
skinned playback.

Two conversions happen here, once, and nowhere else:

  1. Handedness. WHAM world space and Three.js are both Y-up, so the tempting
     fixes (leave it alone, or swap Y and Z) are both wrong. The difference is a
     flip on Z, and getting it wrong silently mirrors the player.
  2. Ground. A single constant Y offset puts the lowest vertex of the whole clip
     at Y=0, so the body stands on the floor. It is deliberately NOT per-frame:
     that would be foot locking, which belongs to Phase 5 and would hide sliding
     we still need to see.

Outputs <out>/web_motion.json and <out>/web_motion.bin.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

LOG_PREFIX = "@@OLINE@@"


def emit(event: str, **payload) -> None:
    payload["event"] = event
    print(f"{LOG_PREFIX} {json.dumps(payload)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bake SMPL world motion for the browser")
    ap.add_argument("--motion", required=True, help="dir containing motion_raw.npz")
    ap.add_argument("--out", default=None, help="output dir (default: --motion)")
    ap.add_argument("--wham-root", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--per-frame-betas",
        action="store_true",
        help="use raw per-frame shape instead of the clip mean (adds shape jitter)",
    )
    args = ap.parse_args()

    wham_root = args.wham_root or os.getcwd()
    if wham_root not in sys.path:
        sys.path.insert(0, wham_root)

    import numpy as np
    import torch

    from lib.models.smpl import SMPL

    npz_path = osp.join(args.motion, "motion_raw.npz")
    if not osp.exists(npz_path):
        emit("error", message=f"motion_raw.npz not found in {args.motion}")
        return 2
    out_dir = args.out or args.motion
    os.makedirs(out_dir, exist_ok=True)

    d = np.load(npz_path)
    pose_world = d["pose_world"]
    trans_world = d["trans_world"]
    betas = d["betas"]
    frame_indices = d["frame_indices"].astype(int)
    n = len(frame_indices)
    emit("loaded", frames=n)

    device = args.device if torch.cuda.is_available() else "cpu"
    smpl = SMPL(model_path=osp.join(wham_root, "dataset/body_models/smpl")).to(device)

    # One body per clip: it is the same human throughout, so the clip-mean shape
    # removes per-frame shape wobble without touching the pose.
    shape = betas if args.per_frame_betas else np.repeat(betas.mean(0, keepdims=True), n, axis=0)

    with torch.no_grad():
        out = smpl.get_output(
            body_pose=torch.from_numpy(pose_world[:, 3:]).float().to(device),
            global_orient=torch.from_numpy(pose_world[:, :3]).float().to(device),
            betas=torch.from_numpy(shape).float().to(device),
            pose2rot=True,
        )
        # get_output returns pelvis-centred vertices and trans_world is the pelvis
        # position in world space, so adding them places the body correctly.
        t = torch.from_numpy(trans_world).float().to(device)
        verts = (out.vertices + t.unsqueeze(1)).cpu().numpy()
        joints = (out.joints + t.unsqueeze(1)).cpu().numpy()

    faces = np.asarray(smpl.faces, dtype=np.uint32)
    emit("posed", verts=list(verts.shape), faces=int(len(faces)))

    # --- conversion 1: WHAM world (Y-up, RH) -> Three.js (Y-up, LH) ---
    verts[..., 2] *= -1.0
    joints[..., 2] *= -1.0
    # Winding order must flip with handedness or every triangle faces inward.
    faces = faces[:, ::-1].copy()

    # --- conversion 2: stand the body on Y=0 with one constant offset ---
    ground_y = float(verts[..., 1].min())
    verts[..., 1] -= ground_y
    joints[..., 1] -= ground_y

    # Quantise to int16 against one global box so decode is a single scale+bias.
    lo = verts.reshape(-1, 3).min(0)
    hi = verts.reshape(-1, 3).max(0)
    extent = np.maximum(hi - lo, 1e-6)
    scale = extent / 65534.0
    quant = np.rint((verts - lo) / scale).astype(np.int32) - 32767
    quant = np.clip(quant, -32767, 32767).astype(np.int16)

    # Round-trip check: quantisation must not be visible at body scale.
    decoded = (quant.astype(np.float32) + 32767.0) * scale + lo
    err_mm = float(np.abs(decoded - verts).max() * 1000.0)

    index_dtype = np.uint16 if verts.shape[1] < 65536 else np.uint32
    bin_path = osp.join(out_dir, "web_motion.bin")
    with open(bin_path, "wb") as f:
        f.write(faces.astype(index_dtype).tobytes())
        f.write(quant.tobytes())

    meta = {
        "schema": "oline.web_motion/1",
        "frames": int(n),
        "n_verts": int(verts.shape[1]),
        "n_faces": int(len(faces)),
        "index_type": "uint16" if index_dtype is np.uint16 else "uint32",
        "fps": float(1.0 / np.median(np.diff(d["timestamps"]))) if n > 1 else 30.0,
        "frame_indices": frame_indices.tolist(),
        "timestamps": [round(float(v), 4) for v in d["timestamps"]],
        "quant": {"lo": lo.tolist(), "scale": scale.tolist(), "bias": 32767},
        "ground_y_removed": ground_y,
        "max_quant_error_mm": round(err_mm, 4),
        "axes": "three.js: X right, Y up, Z toward viewer (WHAM world Z negated)",
        "shape_source": "per_frame" if args.per_frame_betas else "clip_mean",
        "joints": [[round(float(v), 4) for v in j] for j in joints.reshape(n, -1)],
        "n_joints": int(joints.shape[1]),
        "trans_world": [[round(float(v), 4) for v in t] for t in trans_world],
        "bounds": {"lo": lo.tolist(), "hi": (hi - np.array([0, ground_y, 0])).tolist()},
    }
    for key in ("contact", "frame_confidence", "interpolated"):
        if key in d.files:
            arr = d[key]
            meta[key] = (
                [[round(float(v), 4) for v in row] for row in arr]
                if arr.ndim > 1 else [round(float(v), 4) for v in arr]
            )

    json_path = osp.join(out_dir, "web_motion.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    emit(
        "done",
        json=osp.abspath(json_path),
        bin=osp.abspath(bin_path),
        bin_mb=round(osp.getsize(bin_path) / 1e6, 2),
        frames=int(n),
        quant_error_mm=round(err_mm, 4),
        ground_y_removed=round(ground_y, 4),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
