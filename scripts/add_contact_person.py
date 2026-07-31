"""Add the contact opponent to the 3D replay.

  1. Link the nearest non-selected detection across the engagement.
  2. Reconstruct that person with WHAM (same WSL env as the OL).
  3. Bake a mesh and rigidly align it into the OL's Three.js scene so both
     share one ground plane and meet at contact.

    python scripts/add_contact_person.py --motion outputs/motion3d/footage --video footage.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oline_cv.motion3d.contact_opponent import (  # noqa: E402
    extract_opponent_track,
    write_opponent_manifest,
)
from oline_cv.motion3d.wham_bridge import (  # noqa: E402
    WhamBridgeError,
    WhamConfig,
    run_wham_job,
    run_wsl_script,
)
from oline_cv.motion3d.wsl_paths import windows_to_wsl  # noqa: E402


def _pelvis_xz(verts: np.ndarray) -> np.ndarray:
    """Rough pelvis proxy: mean of mid-torso vertices (stable enough to align)."""
    return verts.mean(axis=0)[[0, 2]]


def align_contact_mesh(
    ol_bin: Path,
    contact_bin: Path,
    association_json: Path,
    contact_meta: Path,
    out_bin: Path,
) -> dict:
    """Translate the contact mesh into the OL scene using the first shared frame.

    WHAM's world origin is per-reconstruction, so a second person lands in a
    different place. We keep the contact body's pose, plant them on the same
    ground (y already planted at bake), and slide XZ so their pelvis sits next
    to the OL at the first overlapping frame — offset by the image-space
    separation scaled to metres via the OL bbox height (~1.85 m).
    """
    from oline_cv.motion3d.target_association import load_associations

    def read_pack(path: Path):
        raw = path.read_bytes()
        import struct

        assert raw[:8] == b"OLMESH01"
        n, nv, nf = struct.unpack_from("<III", raw, 8)
        fps = struct.unpack_from("<f", raw, 20)[0]
        o = 24
        fi = np.frombuffer(raw, dtype=np.int32, count=n, offset=o); o += n * 4
        interp = np.frombuffer(raw, dtype=np.uint8, count=n, offset=o); o += n
        o += (4 - (o % 4)) % 4
        conf = np.frombuffer(raw, dtype=np.float32, count=n, offset=o); o += n * 4
        faces = np.frombuffer(raw, dtype=np.uint32, count=nf * 3, offset=o); o += nf * 3 * 4
        verts = np.frombuffer(raw, dtype=np.float32, count=n * nv * 3, offset=o).reshape(n, nv, 3).copy()
        return {
            "raw_header": raw[:24],
            "n": n, "nv": nv, "nf": nf, "fps": fps,
            "fi": fi.copy(), "interp": interp.copy(), "conf": conf.copy(),
            "faces": faces.copy(), "verts": verts,
            "tail_start": 24,  # rebuilt fully below
        }

    ol = read_pack(ol_bin)
    ct = read_pack(contact_bin)
    assocs, _, _ = load_associations(association_json)
    by_assoc = {a.frame_index: a for a in assocs}
    contact_info = json.loads(contact_meta.read_text(encoding="utf-8")) if contact_meta.exists() else {}
    contact_idxs = set(contact_info.get("frame_indices") or ct["fi"].tolist())

    # First shared frame that has a target bbox.
    shared = [int(f) for f in ct["fi"] if int(f) in {int(x) for x in ol["fi"]} and int(f) in by_assoc]
    if not shared:
        raise RuntimeError("no shared frames between OL and contact meshes")
    f0 = shared[0]
    ol_i = int(np.where(ol["fi"] == f0)[0][0])
    ct_i = int(np.where(ct["fi"] == f0)[0][0])

    assoc = by_assoc[f0]
    ol_box = assoc.target_bbox
    # Best non-selected candidate ≈ opponent box at this frame.
    opp = None
    for c in assoc.candidates:
        if c.selected or c.interpolated:
            continue
        if opp is None or c.score > opp.score:
            # prefer high proximity; Candidate.score is association score not proximity
            opp = c
    # Prefer the stored contact bbox if we have it.
    opp_box = None
    if contact_info.get("bboxes") and contact_info.get("frame_indices"):
        try:
            j = contact_info["frame_indices"].index(f0)
            opp_box = contact_info["bboxes"][j]
        except ValueError:
            opp_box = None
    if opp_box is None and opp is not None:
        opp_box = opp.bbox
    if ol_box is None or opp_box is None:
        raise RuntimeError("missing boxes for alignment frame")

    ol_h = max(ol_box[3] - ol_box[1], 1.0)
    mpp = 1.85 / ol_h  # metres per pixel from standing height
    ol_cx, ol_cy = (ol_box[0] + ol_box[2]) / 2.0, (ol_box[1] + ol_box[3]) / 2.0
    op_cx, op_cy = (opp_box[0] + opp_box[2]) / 2.0, (opp_box[1] + opp_box[3]) / 2.0
    # Image x → Three.js +X; image y down → Three.js −Z (depth into field).
    desired = np.array([(op_cx - ol_cx) * mpp, 0.0, -((op_cy - ol_cy) * mpp * 0.55)], dtype=np.float32)

    ol_pelvis = ol["verts"][ol_i].mean(axis=0)
    ct_pelvis = ct["verts"][ct_i].mean(axis=0)
    target_pelvis = ol_pelvis + desired
    # Keep contact on the ground (y planted); only slide on XZ.
    delta = target_pelvis - ct_pelvis
    delta[1] = 0.0
    ct["verts"] += delta

    # Replant if any vertex went below ground after numerical noise.
    ymin = float(ct["verts"][..., 1].min())
    if ymin < 0:
        ct["verts"][..., 1] -= ymin

    import struct

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(b"OLMESH01")
        f.write(struct.pack("<III", ct["n"], ct["nv"], ct["nf"]))
        f.write(struct.pack("<f", float(ct["fps"])))
        f.write(ct["fi"].astype(np.int32).tobytes())
        f.write(ct["interp"].astype(np.uint8).tobytes())
        pad = (4 - (f.tell() % 4)) % 4
        if pad:
            f.write(b"\x00" * pad)
        f.write(ct["conf"].astype(np.float32).tobytes())
        f.write(ct["faces"].astype(np.uint32).tobytes())
        f.write(np.ascontiguousarray(ct["verts"], dtype=np.float32).tobytes())

    return {
        "align_frame": f0,
        "delta_m": [round(float(v), 4) for v in delta],
        "desired_offset_m": [round(float(v), 4) for v in desired],
        "shared_frames": len(shared),
        "contact_frames": sorted(contact_idxs),
        "out": str(out_bin),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Add contact opponent to 3D replay")
    ap.add_argument("--motion", required=True, help="dir with tracks.json + motion_raw.npz")
    ap.add_argument("--video", default=None)
    ap.add_argument("--min-run", type=int, default=20)
    ap.add_argument("--skip-wham", action="store_true", help="reuse existing motion_contact.npz")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    motion = Path(args.motion).resolve()
    tracks = motion / "tracks.json"
    assoc = motion / "association.json"
    if not tracks.exists() or not assoc.exists():
        print("error: need tracks.json and association.json", file=sys.stderr)
        return 2

    print("Extracting contact opponent track…")
    track = extract_opponent_track(assoc, min_run=args.min_run)
    if track is None:
        print("error: could not find a stable contact opponent track", file=sys.stderr)
        return 1
    print(f"  opponent frames {track.start}-{track.end} ({len(track.frames)} frames)")

    contact_dir = motion / "contact"
    contact_dir.mkdir(parents=True, exist_ok=True)
    video = args.video
    if not video:
        from oline_cv.motion3d.schema import load_manifest
        video = load_manifest(tracks).video.get("path")
    manifest_path = write_opponent_manifest(
        track, tracks, contact_dir / "tracks.json", video_path=video
    )
    print(f"  wrote {manifest_path}")

    cfg = WhamConfig()
    # Boxes were already chosen from detections; re-associating can steal a
    # nearby teammate. Trust the linked opponent boxes and run ViTPose on them.
    cfg.associate = False
    npz = contact_dir / "motion_raw.npz"
    if not args.skip_wham or not npz.exists():
        print("Running WHAM on contact opponent…")

        def progress(pct, msg, stage="wham"):
            print(f"  [{stage}] {float(pct):5.1f}%  {msg}", flush=True)

        try:
            result = run_wham_job(
                manifest_path,
                video,
                contact_dir,
                segment=(track.start, track.end),
                config=cfg,
                progress_cb=progress,
                log_cb=(lambda line: print(f"    | {line}", flush=True)) if args.verbose else None,
            )
        except WhamBridgeError as exc:
            print(f"FAILED: {exc}", file=sys.stderr)
            return 1
        print(f"  contact status {result.metadata.status}  frames {result.metadata.frame_count}")
    else:
        print(f"  reusing {npz}")

    print("Baking contact SMPL mesh…")

    def progress(pct, msg, stage="bake"):
        print(f"  [{stage}] {float(pct):5.1f}%  {msg}", flush=True)

    raw_bin = contact_dir / "mesh_threejs_raw.bin"
    code, events, logs = run_wsl_script(
        "scripts/bake_smpl_mesh.py",
        [
            "--motion", windows_to_wsl(contact_dir),
            "--out", windows_to_wsl(raw_bin),
            "--wham-root", cfg.wham_root,
        ],
        config=cfg,
        progress_cb=progress,
        stage="bake",
    )
    if code != 0:
        print(f"bake failed: {events}", file=sys.stderr)
        print("\n".join(logs[-20:]), file=sys.stderr)
        return 1

    ol_bin = motion / "mesh_threejs.bin"
    if not ol_bin.exists():
        print("error: OL mesh_threejs.bin missing — bake the primary mesh first", file=sys.stderr)
        return 2

    print("Aligning contact mesh into OL scene…")
    info = align_contact_mesh(
        ol_bin,
        raw_bin,
        assoc,
        contact_dir / "contact_opponent.json",
        motion / "mesh_contact.bin",
    )
    (motion / "mesh_contact.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"  align frame {info['align_frame']}  delta {info['delta_m']}")
    print(f"  wrote {info['out']}")
    print("\nDone. Open /viewer3d?clip=footage — both players should appear.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
