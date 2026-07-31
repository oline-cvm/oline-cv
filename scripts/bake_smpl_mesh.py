"""Bake SMPL vertices into a browser-ready binary mesh pack. Runs inside WSL.

WHAM world space is y-up. Three.js is also y-up but opposite-handed, so we
negate Z once here — the viewer must never apply a second conversion.

Uses the same get_output(transl=trans_world) path as WHAM's own global vis, then
plants the sequence so the lowest vertex sits on y=0.

Binary layout (little-endian), magic OLMESH01:

    u32 n_frames, n_verts, n_faces
    f32 fps
    i32 frame_indices[n_frames]
    u8  interpolated[n_frames]
    f32 confidence[n_frames]
    u32 faces[n_faces * 3]
    f32 verts[n_frames * n_verts * 3]   # Three.js coords
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import struct
import sys

LOG_PREFIX = "@@OLINE@@"
MAGIC = b"OLMESH01"


def emit(event: str, **payload) -> None:
    payload["event"] = event
    print(f"{LOG_PREFIX} {json.dumps(payload)}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bake SMPL mesh for Three.js")
    ap.add_argument("--motion", required=True, help="dir with motion_raw.npz")
    ap.add_argument("--out", default=None, help="output .bin path")
    ap.add_argument("--wham-root", default=None)
    ap.add_argument("--device", default="cuda")
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

    data = np.load(npz_path)
    pose = data["pose_world"].astype(np.float32)
    betas = data["betas"].astype(np.float32)
    trans = data["trans_world"].astype(np.float32)
    frame_indices = data["frame_indices"].astype(np.int32)
    n = len(frame_indices)
    fps = 30.0
    if "timestamps" in data.files and len(data["timestamps"]) > 1:
        dt = float(np.median(np.diff(data["timestamps"])))
        if dt > 1e-6:
            fps = 1.0 / dt

    interpolated = (
        data["interpolated"].astype(np.uint8)
        if "interpolated" in data.files
        else np.zeros(n, dtype=np.uint8)
    )
    confidence = (
        data["frame_confidence"].astype(np.float32)
        if "frame_confidence" in data.files
        else np.ones(n, dtype=np.float32)
    )

    emit("loaded", frames=n, frame_range=[int(frame_indices[0]), int(frame_indices[-1])])

    device = args.device if torch.cuda.is_available() else "cpu"
    smpl = SMPL(model_path=osp.join(wham_root, "dataset/body_models/smpl")).to(device)
    faces = np.asarray(smpl.faces, dtype=np.uint32).reshape(-1, 3)

    with torch.no_grad():
        # Same call WHAM's run_vis uses for the global figure.
        out = smpl.get_output(
            body_pose=torch.from_numpy(pose[:, 3:]).to(device),
            global_orient=torch.from_numpy(pose[:, :3]).to(device),
            betas=torch.from_numpy(betas).to(device),
            transl=torch.from_numpy(trans).to(device),
            pose2rot=True,
        )
        verts = out.vertices.detach().cpu().numpy().astype(np.float32)

    # Plant the whole sequence on the ground plane (WHAM global vis convention).
    verts[..., 1] -= float(verts[..., 1].min())

    # WHAM world (y-up, right-handed OpenGL-ish after return_y_up) -> Three.js
    # (y-up, right-handed with z toward viewer): flip Z once.
    verts[..., 2] *= -1.0

    n_verts = int(verts.shape[1])
    n_faces = int(faces.shape[0])
    emit("progress", percent=80.0, message=f"packing {n}×{n_verts} verts…")

    out_path = args.out or osp.join(args.motion, "mesh_threejs.bin")
    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", n, n_verts, n_faces))
        f.write(struct.pack("<f", float(fps)))
        f.write(frame_indices.tobytes())
        f.write(interpolated.tobytes())
        # Pad so Float32Array views in the browser stay 4-byte aligned.
        pad = (4 - (f.tell() % 4)) % 4
        if pad:
            f.write(b"\x00" * pad)
        f.write(confidence.tobytes())
        f.write(faces.astype(np.uint32).ravel().tobytes())
        f.write(np.ascontiguousarray(verts).tobytes())

    meta = {
        "path": osp.abspath(out_path),
        "frames": n,
        "n_verts": n_verts,
        "n_faces": n_faces,
        "fps": fps,
        "bytes": osp.getsize(out_path),
        "coords": "threejs_y_up_z_negated",
        "planted": True,
        "source": "pose_world + trans_world via WHAM SMPL.get_output(transl=...)",
    }
    meta_path = osp.splitext(out_path)[0] + ".json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    emit("done", **meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
