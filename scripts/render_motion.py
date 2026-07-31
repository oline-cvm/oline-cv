"""Render the reconstructed SMPL body back onto the film. Runs inside WSL.

This is the verification gate for Phase 2: a clean exit code proves nothing, but
a mesh that tracks the selected lineman's limbs frame by frame does. The mesh is
posed from motion_raw.npz in CAMERA space and projected with WHAM's own renderer,
so any mismatch between mesh and film is a real reconstruction error rather than
a viewer bug.

Only the reconstructed subject is drawn — there is exactly one, by construction.

    python scripts/render_motion.py --motion <dir> --video <mp4> --out overlay.mp4
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
    ap = argparse.ArgumentParser(description="Render SMPL motion over the source video")
    ap.add_argument("--motion", required=True, help="dir containing motion_raw.npz")
    ap.add_argument("--video", required=True, help="source video")
    ap.add_argument("--out", default=None, help="output mp4 (default: <motion>/smpl_overlay.mp4")
    ap.add_argument("--wham-root", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--sheet", default=None, help="also write a contact sheet here")
    ap.add_argument("--sheet-count", type=int, default=8)
    args = ap.parse_args()

    wham_root = args.wham_root or os.getcwd()
    if wham_root not in sys.path:
        sys.path.insert(0, wham_root)

    import cv2
    import numpy as np
    import torch

    from lib.models.smpl import SMPL
    from lib.vis.renderer import Renderer

    npz_path = osp.join(args.motion, "motion_raw.npz")
    if not osp.exists(npz_path):
        emit("error", message=f"motion_raw.npz not found in {args.motion}")
        return 2
    data = np.load(npz_path)
    frame_indices = data["frame_indices"].astype(int)
    pose_cam = data["pose_cam"]
    betas = data["betas"]
    trans_cam = data["trans_cam"]
    n = len(frame_indices)
    emit("loaded", frames=n, frame_range=[int(frame_indices[0]), int(frame_indices[-1])])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        emit("error", message=f"could not open video: {args.video}")
        return 2
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    device = args.device if torch.cuda.is_available() else "cpu"
    smpl = SMPL(model_path=osp.join(wham_root, "dataset/body_models/smpl")).to(device)

    # WHAM's demo focal length convention: the crop-independent default used when
    # no intrinsics are known.
    focal = (width ** 2 + height ** 2) ** 0.5
    renderer = Renderer(width, height, focal, device, smpl.faces)

    with torch.no_grad():
        out = smpl.get_output(
            body_pose=torch.from_numpy(pose_cam[:, 3:]).float().to(device),
            global_orient=torch.from_numpy(pose_cam[:, :3]).float().to(device),
            betas=torch.from_numpy(betas).float().to(device),
            pose2rot=True,
        )
        # get_output returns pelvis-centred vertices (WHAM's verts_cam), while the
        # npz stores trans_cam MINUS that same pelvis offset, matching WHAM's own
        # pkl convention. Rendering needs verts_cam + raw trans_cam, so the offset
        # has to be added back or the mesh lands a pelvis-height away from the body.
        trans = torch.from_numpy(trans_cam).float().to(device) + out.offset
        verts = (out.vertices + trans.unsqueeze(1)).cpu().numpy()

    out_path = args.out or osp.join(args.motion, "smpl_overlay.mp4")
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        emit("error", message=f"could not open writer: {out_path}")
        return 2

    wanted = {int(f): i for i, f in enumerate(frame_indices)}
    limit = args.max_frames or n
    sheet_at = set()
    if args.sheet:
        picks = np.linspace(0, min(limit, n) - 1, args.sheet_count)
        sheet_at = {int(frame_indices[int(round(p))]) for p in picks}
    tiles = []

    idx = 0
    written = 0
    last = int(frame_indices[min(limit, n) - 1])
    while idx <= last:
        ok, img = cap.read()
        if not ok:
            break
        if idx in wanted:
            i = wanted[idx]
            frame = renderer.render_mesh(torch.from_numpy(verts[i]).to(device), img.copy())
            cv2.putText(
                frame, f"frame {idx}  SMPL (camera space)", (12, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 240, 140), 2, cv2.LINE_AA,
            )
            if idx in sheet_at:
                tiles.append((idx, frame.copy()))
            writer.write(frame)
            written += 1
            if written % 25 == 0:
                emit("progress", percent=round(100.0 * written / min(limit, n), 1),
                     message=f"rendered {written}/{min(limit, n)}")
        idx += 1

    writer.release()
    cap.release()

    sheet_path = None
    if tiles and args.sheet:
        cols = 4
        tile_w = 420
        # Crop to the player before downscaling: a full 2040px frame shrunk to a
        # tile makes the mesh a few pixels tall, which is useless for judging fit.
        bboxes = data["bbox_cxcys"] if "bbox_cxcys" in data.files else None
        resized = []
        for fidx, frame in tiles:
            if bboxes is not None:
                cx, cy, scale = bboxes[wanted[fidx]]
                half = max(scale * 200.0, 120.0) * 1.1
                x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
                x1, y1 = int(min(width, cx + half)), int(min(height, cy + half))
                view = frame[y0:y1, x0:x1]
                if view.size == 0:
                    view = frame
            else:
                view = frame
            s = tile_w / float(view.shape[1])
            resized.append(cv2.resize(view, (tile_w, int(round(view.shape[0] * s)))))
        th = max(t.shape[0] for t in resized)
        resized = [
            cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if t.shape[0] < th else t
            for t in resized
        ]
        rows = (len(resized) + cols - 1) // cols
        sheet = np.zeros((rows * th, cols * tile_w, 3), dtype=resized[0].dtype)
        for i, t in enumerate(resized):
            r, c = divmod(i, cols)
            sheet[r * th : r * th + t.shape[0], c * tile_w : c * tile_w + t.shape[1]] = t
        sheet_path = args.sheet
        cv2.imwrite(sheet_path, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    emit("done", out=osp.abspath(out_path), frames=written,
         sheet=osp.abspath(sheet_path) if sheet_path else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
