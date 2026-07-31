"""WSL-side WHAM runner. Executed inside the `wham` conda env, cwd = WHAM root.

This is the production interface to WHAM. It does NOT call demo.py; it imports
WHAM's own components and drives them with OUR tracking identity:

    - WHAM's own tracker is deliberately bypassed. Its get_track_id pass would
      pick whichever person it liked, discarding the BoT-SORT identity lock that
      the rest of this project exists to maintain. We still run WHAM's YOLOv8x
      detector to enumerate every person in frame, but which of those detections
      is "the player" is decided by association against the TrackManifest target
      box (see oline_cv.motion3d.target_association). A frame with no accepted
      candidate is dropped, never reassigned to a nearby defender.
    - WHAM's ViTPose is still used for 2D keypoints, run on the ASSOCIATED
      detection box. WHAM was trained on ViTPose keypoints, so feeding YOLO
      keypoints would be a domain shift; ViTPose-on-the-associated-box gives us
      WHAM's expected input distribution while identity stays ours.
    - Everything downstream (FeatureExtractor, SLAMModel, CustomDataset,
      Network) is WHAM's, unmodified.

Never imported by the Windows application — it is launched via wsl.exe.

Structured events go to stdout as `@@OLINE@@ {json}` lines so the Windows bridge
can track progress without parsing WHAM's own logging.
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
import traceback
from collections import defaultdict
from typing import Any

LOG_PREFIX = "@@OLINE@@"

# WHAM's own thresholds, mirrored so our injected data matches its expectations.
VIS_THRESH = 0.3
BBOX_S_FACTOR = 1.2
MINIMUM_JOINTS = 6

REQUIRED_CHECKPOINTS = {
    "wham": "checkpoints/wham_vit_bedlam_w_3dpw.pth.tar",
    "hmr2a": "checkpoints/hmr2a.ckpt",
}
OPTIONAL_CHECKPOINTS = {
    "vitpose": "checkpoints/vitpose-h-multi-coco.pth",
    "dpvo": "checkpoints/dpvo.pth",
    "yolo": "checkpoints/yolov8x.pt",
}
# Only the assets WHAM's inference path actually loads. constants.py also declares
# FACES (smpl_faces.npy) and JOINTS_REGRESSOR_EXTRA (J_regressor_extra.npy), but
# neither is referenced anywhere in lib/ or configs/ — treating them as required
# reports a working install as broken.
REQUIRED_BODY_MODELS = [
    "dataset/body_models/smpl/SMPL_NEUTRAL.pkl",
    "dataset/body_models/smplx2smpl.pkl",
    "dataset/body_models/smpl_mean_params.npz",
    "dataset/body_models/J_regressor_wham.npy",
    "dataset/body_models/J_regressor_feet.npy",
    "dataset/body_models/J_regressor_h36m.npy",
    "dataset/body_models/coco_aug_dict.pth",
]
UNUSED_BODY_MODELS = [
    "dataset/body_models/smpl_faces.npy",
    "dataset/body_models/J_regressor_extra.npy",
]


def emit(event: str, **payload: Any) -> None:
    payload["event"] = event
    print(f"{LOG_PREFIX} {json.dumps(payload)}", flush=True)


def progress(percent: float, message: str) -> None:
    emit("progress", percent=round(float(percent), 2), message=message)


# --------------------------------------------------------------------------
# environment preflight
# --------------------------------------------------------------------------


def _probe_module(name: str) -> dict[str, Any]:
    try:
        import importlib

        mod = importlib.import_module(name)
        return {"ok": True, "version": getattr(mod, "__version__", None)}
    except Exception as exc:  # noqa: BLE001 - report, never raise
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def run_doctor(wham_root: str) -> dict[str, Any]:
    """Report exactly which pieces of the WHAM stack are present."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
        "wham_root": wham_root,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "modules": {},
        "files": {},
        "missing": [],
        "gpu": None,
    }

    for name in (
        "numpy", "torch", "cv2", "joblib", "loguru", "yacs",
        "progress", "smplx", "mmpose", "mmcv", "dpvo",
    ):
        report["modules"][name] = _probe_module(name)

    torch_info = report["modules"].get("torch", {})
    if torch_info.get("ok"):
        try:
            import torch

            report["gpu"] = {
                "cuda_available": bool(torch.cuda.is_available()),
                "device_count": int(torch.cuda.device_count()),
                "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            }
        except Exception as exc:  # noqa: BLE001
            report["gpu"] = {"error": str(exc)}

    for label, rel in {**REQUIRED_CHECKPOINTS, **OPTIONAL_CHECKPOINTS}.items():
        p = osp.join(wham_root, rel)
        report["files"][label] = {"path": rel, "exists": osp.exists(p)}
    for rel in REQUIRED_BODY_MODELS + UNUSED_BODY_MODELS:
        p = osp.join(wham_root, rel)
        report["files"][osp.basename(rel)] = {"path": rel, "exists": osp.exists(p)}
    cfg_rel = "configs/yamls/demo.yaml"
    report["files"]["demo_yaml"] = {
        "path": cfg_rel,
        "exists": osp.exists(osp.join(wham_root, cfg_rel)),
    }

    for name in ("numpy", "torch", "cv2", "joblib", "loguru", "yacs", "progress", "smplx"):
        if not report["modules"][name]["ok"]:
            report["missing"].append(f"module:{name}")
    for label in REQUIRED_CHECKPOINTS:
        if not report["files"][label]["exists"]:
            report["missing"].append(f"checkpoint:{REQUIRED_CHECKPOINTS[label]}")
    for rel in REQUIRED_BODY_MODELS:
        if not report["files"][osp.basename(rel)]["exists"]:
            report["missing"].append(f"body_model:{rel}")
    if not report["files"]["demo_yaml"]["exists"]:
        report["missing"].append("config:configs/yamls/demo.yaml")
    if not (report.get("gpu") or {}).get("cuda_available"):
        report["missing"].append("cuda:no_gpu_visible")

    report["vitpose_available"] = (
        report["modules"]["mmpose"]["ok"] and report["files"]["vitpose"]["exists"]
    )
    # Probe the import WHAM actually performs, not just `import dpvo`: the SLAM
    # wrapper pulls in dpvo's compiled extensions and config, either of which can
    # be missing while the bare package imports fine.
    if wham_root not in sys.path:
        sys.path.insert(0, wham_root)
    report["modules"]["lib.models.preproc.slam"] = _probe_module("lib.models.preproc.slam")
    report["slam_available"] = (
        report["modules"]["lib.models.preproc.slam"]["ok"] and report["files"]["dpvo"]["exists"]
    )
    report["ready"] = not report["missing"]
    return report


# --------------------------------------------------------------------------
# manifest -> WHAM tracking_results
# --------------------------------------------------------------------------


def load_our_manifest(repo: str, tracks: str):
    """Import the oline-cv schema from the Windows repo (stdlib-only module)."""
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from oline_cv.motion3d.schema import load_manifest

    return load_manifest(tracks)


def select_our_segment(repo: str, manifest, explicit: tuple[int, int] | None):
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from oline_cv.motion3d.segments import contiguity_gaps, frames_in_segment, select_segment

    segment = select_segment(manifest, explicit=explicit)
    frames = frames_in_segment(manifest, segment)
    return segment, frames, contiguity_gaps(frames)


def xyxy_to_cxcys(bbox, s_factor: float = 1.05):
    """WHAM's bbox convention: [cx, cy, scale] with scale = max(w,h)/200*s."""
    import numpy as np

    b = np.asarray(bbox, dtype=float)
    cx, cy = b[[0, 2]].mean(), b[[1, 3]].mean()
    scale = max(b[2] - b[0], b[3] - b[1]) / 200.0 * s_factor
    return np.array([cx, cy, scale], dtype=float)


def bbox_from_keypoints(kp, fallback_xyxy, s_factor: float = BBOX_S_FACTOR):
    """Replicates WHAM's compute_bboxes_from_keypoints for a single frame.

    WHAM derives its crop box from keypoints, not the detector box, so we do the
    same to stay inside its trained input distribution. When too few keypoints
    are visible we fall back to the BoT-SORT box instead of emitting garbage.
    """
    import numpy as np

    mask = kp[:, -1] > VIS_THRESH
    if mask.sum() < MINIMUM_JOINTS:
        return xyxy_to_cxcys(fallback_xyxy, s_factor=1.05), True
    xs, ys = kp[mask, 0], kp[mask, 1]
    cx, cy = (xs.max() + xs.min()) / 2, (ys.max() + ys.min()) / 2
    s = max(xs.max() - xs.min(), ys.max() - ys.min())
    return np.array([cx, cy, s * s_factor / 200.0], dtype=float), False


def keypoints_from_manifest(frame):
    import numpy as np

    kp = np.zeros((17, 3), dtype=float)
    for i, entry in enumerate(frame.keypoints_2d[:17]):
        if entry[0] is None or entry[1] is None:
            continue
        kp[i] = [float(entry[0]), float(entry[1]), float(entry[2] or 0.0)]
    return kp


class PersonDetector:
    """WHAM's preprocessing person detector, without WHAM's identity decisions.

    Same YOLOv8x weights WHAM's DetectionModel loads, same class filter. We only
    skip its get_track_id step, because that is precisely the decision we are
    taking over.
    """

    def __init__(self, wham_root: str, device: str, conf: float, imgsz: int):
        from ultralytics import YOLO

        ckpt = osp.join(wham_root, OPTIONAL_CHECKPOINTS["yolo"])
        self.model = YOLO(ckpt if osp.exists(ckpt) else "yolov8x.pt")
        self.device = device
        self.conf = conf
        self.imgsz = imgsz

    def __call__(self, img) -> tuple[list[list[float]], list[float]]:
        res = self.model.predict(
            img,
            device=self.device,
            classes=0,
            conf=self.conf,
            imgsz=self.imgsz,
            save=False,
            verbose=False,
        )[0]
        boxes = res.boxes
        if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
            return [], []
        xyxy = boxes.xyxy.detach().cpu().numpy().tolist()
        confs = boxes.conf.detach().cpu().numpy().tolist()
        return [[float(v) for v in b] for b in xyxy], [float(c) for c in confs]


def _read_frames(video: str, wanted: list[int]):
    """Yield ``(image, frame_index)`` for the requested indices, in order.

    Sequential decode rather than seeking: frame-accurate seeks on H.264 are
    unreliable, and getting the wrong frame here silently corrupts the whole
    reconstruction.
    """
    import cv2

    if not wanted:
        return
    targets = set(wanted)
    last = max(wanted)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video}")
    try:
        idx = 0
        while idx <= last:
            ok, img = cap.read()
            if not ok:
                break
            if idx in targets:
                yield img, idx
            idx += 1
    finally:
        cap.release()


def build_tracking_results(
    video: str,
    frames: list[Any],
    fps: float,
    keypoints_source: str,
    wham_root: str,
    device: str,
    warnings: list[str],
    thresholds: Any,
    args: Any,
) -> tuple[dict[int, dict[str, Any]], list[Any], dict[str, Any]]:
    """Assemble WHAM's tracking_results for our single locked OL.

    Returns ``(tracking_results, associations, association_summary)``. Only frames
    whose detection was successfully associated with the TrackManifest target are
    included, and the sequence is trimmed to the longest contiguous accepted run
    so WHAM's temporal model never sees a stitch across a hole.
    """
    import cv2
    import numpy as np
    import scipy.signal as signal

    from oline_cv.motion3d.target_association import (
        associate_frame,
        bridge_gaps,
        longest_valid_run,
        summarize,
    )

    wanted = {f.frame_index: f for f in frames}
    order = sorted(wanted)
    total = len(order)

    detector = None
    if args.associate:
        try:
            detector = PersonDetector(wham_root, device, args.det_conf, args.det_imgsz)
        except Exception as exc:  # noqa: BLE001
            warnings.append(
                f"person detector unavailable ({type(exc).__name__}: {exc}); "
                "falling back to the BoT-SORT box with no association check"
            )

    # Pass 1: enumerate detections and decide identity for every frame. Bridging
    # needs the whole sequence before keypoints are extracted, because a bridged
    # frame's box only exists once its neighbours are known.
    associations: list[Any] = []
    for cursor, (img, fidx) in enumerate(_read_frames(video, order), start=1):
        fr = wanted[fidx]
        if detector is not None:
            dets, confs = detector(img)
            assoc = associate_frame(fidx, fr.bbox, dets, confs, thresholds=thresholds)
        else:
            # No detector: our own box is the only candidate, so it is trivially
            # the target. Recorded explicitly so metadata never overstates this.
            assoc = associate_frame(fidx, fr.bbox, [fr.bbox] if fr.bbox else [], [1.0])
        assoc.track_id = getattr(fr, "track_id", None)
        associations.append(assoc)
        if cursor % 20 == 0:
            progress(10.0 + 15.0 * cursor / max(1, total), f"Associating ({cursor}/{total})…")

    if len(associations) != total:
        warnings.append(
            f"video yielded {len(associations)} of {total} requested frames "
            "(clip shorter than the manifest?)"
        )

    bridge_stats = bridge_gaps(associations, thresholds=thresholds)
    if bridge_stats["bridged_frames"]:
        warnings.append(
            f"bridged {len(bridge_stats['bridged_frames'])} short-dropout frames by "
            f"interpolation: {bridge_stats['bridged_frames']}; these are marked "
            "interpolated and carry reduced confidence"
        )
    for skipped in bridge_stats["skipped_gaps"]:
        warnings.append(
            f"gap {skipped['frames']} ({skipped['length']} frames) not bridged: "
            f"{skipped['reason']}"
        )

    # Pass 2: 2D keypoints on whichever box each valid frame ended up with.
    pose_model = None
    if keypoints_source == "vitpose":
        from mmpose.apis import init_pose_model

        cfg_path = osp.join(
            wham_root,
            "third-party/ViTPose/configs/body/2d_kpt_sview_rgb_img/"
            "topdown_heatmap/coco/ViTPose_huge_coco_256x192.py",
        )
        ckpt = osp.join(wham_root, OPTIONAL_CHECKPOINTS["vitpose"])
        pose_model = init_pose_model(cfg_path, ckpt, device=device.lower())

    valid_order = [a.frame_index for a in associations if a.valid]
    by_index = {a.frame_index: a for a in associations}
    per_frame: dict[int, dict[str, Any]] = {}

    for cursor, (img, fidx) in enumerate(_read_frames(video, valid_order), start=1):
        assoc = by_index[fidx]
        sel_bbox = assoc.selected.bbox
        if pose_model is not None:
            from mmpose.apis import inference_top_down_pose_model

            results, _ = inference_top_down_pose_model(
                pose_model,
                img,
                person_results=[{"bbox": np.asarray(sel_bbox, dtype=float)}],
                format="xyxy",
                return_heatmap=False,
                outputs=None,
            )
            kp = (
                np.asarray(results[0]["keypoints"], dtype=float)
                if results
                else np.zeros((17, 3))
            )
        else:
            kp = keypoints_from_manifest(wanted[fidx])
        bbox, used_fallback = bbox_from_keypoints(kp, sel_bbox)
        per_frame[fidx] = {
            "kp": kp,
            "bbox": bbox,
            "fallback": used_fallback,
            "interpolated": bool(assoc.interpolated),
            "confidence": float(assoc.confidence),
        }
        if cursor % 20 == 0:
            progress(
                25.0 + 20.0 * cursor / max(1, len(valid_order)),
                f"2D keypoints ({cursor}/{len(valid_order)})…",
            )

    summary = summarize(associations)
    summary["bridge_skipped_gaps"] = bridge_stats["skipped_gaps"]
    emit("association", **{k: summary[k] for k in
         ("frames", "valid", "observed", "bridged", "invalid", "ambiguous",
          "mean_confidence", "min_iou")})
    if summary["invalid"]:
        warnings.append(
            f"{summary['invalid']} of {summary['frames']} frames had no detection matching "
            f"the tracked lineman ({summary['invalid_reasons']}); those frames are excluded"
        )
    if summary["ambiguous"]:
        warnings.append(
            f"{summary['ambiguous']} frames had two candidates within the ambiguity margin; "
            "identity may be uncertain there"
        )

    run = longest_valid_run(associations)
    if run is None:
        raise RuntimeError(
            "no frame could be associated with the tracked lineman; "
            "refusing to reconstruct an unidentified person"
        )
    start, end = run
    kept = [i for i in sorted(per_frame) if start <= i <= end]
    if len(kept) < summary["valid"]:
        warnings.append(
            f"association was discontinuous; reconstructing the longest clean run "
            f"{start}-{end} ({len(kept)} of {summary['valid']} matched frames)"
        )
    if len(kept) < args.min_frames:
        raise RuntimeError(
            f"only {len(kept)} contiguous associated frames (need >= {args.min_frames}); "
            "WHAM's temporal model needs a longer clean run"
        )

    fallback_count = sum(1 for i in kept if per_frame[i]["fallback"])
    if fallback_count:
        warnings.append(
            f"{fallback_count} frames had <{MINIMUM_JOINTS} visible keypoints; "
            "used the associated detection box for those crops"
        )

    keypoints = np.stack([per_frame[i]["kp"] for i in kept])
    bbox_arr = np.stack([per_frame[i]["bbox"] for i in kept])
    summary["interpolated_kept"] = [i for i in kept if per_frame[i]["interpolated"]]
    summary["frame_confidence"] = {int(i): round(per_frame[i]["confidence"], 4) for i in kept}
    # WHAM median-filters bbox params over ~fps/2 to kill detector jitter.
    kernel = int(int(fps / 2) / 2) * 2 + 1
    if kernel > 1 and len(bbox_arr) > kernel:
        bbox_arr = np.array([signal.medfilt(p, kernel) for p in bbox_arr.T]).T

    results: dict[int, dict[str, Any]] = defaultdict(lambda: defaultdict(list))
    subject = 1  # our permanent logical target id
    results[subject]["frame_id"] = np.asarray(kept, dtype=np.int64)
    results[subject]["keypoints"] = keypoints
    results[subject]["bbox"] = bbox_arr

    summary["reconstructed_range"] = [start, end]
    summary["reconstructed_frames"] = len(kept)
    summary["associated"] = detector is not None
    return results, associations, summary


def get_slam_results(
    video: str,
    output_pth: str,
    width: int,
    height: int,
    n_frames_total: int,
    start: int,
    count: int,
    run_slam: bool,
    warnings: list[str],
):
    """DPVO camera trajectory sliced to our segment, or an identity fallback.

    Camera angular velocity is what makes WHAM's trajectory world-grounded, so a
    missing DPVO is a correctness downgrade, not a cosmetic one. We record that
    honestly rather than pretending the output is gravity-aligned.
    """
    import numpy as np

    def identity(n: int):
        arr = np.zeros((n, 7), dtype=float)
        arr[:, 3] = 1.0  # unit quaternion
        return arr

    if not run_slam:
        warnings.append("SLAM disabled by request: camera motion is not compensated")
        return identity(count), False

    try:
        from lib.models.preproc.slam import SLAMModel
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"DPVO unavailable ({type(exc).__name__}): local-only reconstruction")
        return identity(count), False

    try:
        import cv2

        progress(38.0, "Running DPVO visual odometry over the full clip…")
        slam = SLAMModel(video, output_pth, width, height, None)
        cap = cv2.VideoCapture(video)
        seen = 0
        while cap.isOpened():
            ok, _img = cap.read()
            if not ok:
                break
            slam.track()
            seen += 1
            if seen % 50 == 0:
                progress(38.0 + 12.0 * seen / max(1, n_frames_total), f"SLAM ({seen} frames)…")
        cap.release()
        traj = np.asarray(slam.process())
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"SLAM failed ({type(exc).__name__}: {exc}); local-only reconstruction")
        return identity(count), False

    # DPVO runs over the whole clip for continuity; slice to our segment so the
    # per-frame camera angular velocity aligns with the keypoint sequence.
    if traj.ndim != 2 or traj.shape[1] != 7:
        warnings.append(f"unexpected SLAM output shape {traj.shape}; using identity camera")
        return identity(count), False
    if traj.shape[0] < start + count:
        warnings.append(
            f"SLAM returned {traj.shape[0]} poses, need {start + count}; padding with last pose"
        )
        pad = np.repeat(traj[-1:], start + count - traj.shape[0], axis=0)
        traj = np.concatenate([traj, pad], axis=0)
    return traj[start : start + count], True


# --------------------------------------------------------------------------
# reconstruction
# --------------------------------------------------------------------------


def reconstruct(args) -> int:
    import numpy as np
    import torch

    started = time.time()
    warnings: list[str] = []
    wham_root = args.wham_root or os.getcwd()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    npz_path = osp.join(out_dir, "motion_raw.npz")
    meta_path = osp.join(out_dir, "motion_metadata.json")

    if repo := args.repo:
        if repo not in sys.path:
            sys.path.insert(0, repo)
    from oline_cv.motion3d.motion_schema import MotionMetadata, ReconstructionStatus
    from oline_cv.motion3d.target_association import AssociationThresholds, save_associations

    def write_meta(status: str, **extra: Any) -> None:
        meta = MotionMetadata(
            status=status,
            video=args.video,
            fps=float(extra.pop("fps", 0.0) or 0.0),
            segment=extra.pop("segment", {}),
            frame_range=extra.pop("frame_range", [0, 0]),
            frame_count=int(extra.pop("frame_count", 0) or 0),
            runtime_seconds=time.time() - started,
            outputs=extra.pop("outputs", {}),
            wham=extra.pop("wham", {}),
            device=extra.pop("device", {}),
            association=extra.pop("association", {}),
            warnings=warnings,
            errors=extra.pop("errors", []),
            stats=extra.pop("stats", {}),
        )
        meta.save(meta_path)

    # ---- deterministic-as-possible setup ----
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    explicit = None
    if args.segment:
        a, b = args.segment.split(":")
        explicit = (int(a), int(b))

    progress(3.0, "Loading track manifest…")
    manifest = load_our_manifest(args.repo, args.tracks)
    segment, frames, gaps = select_our_segment(args.repo, manifest, explicit)
    if gaps:
        warnings.append(f"segment has {len(gaps)} index discontinuities: {gaps[:5]}")

    fps = manifest.fps
    width = int(manifest.video.get("width") or 0)
    height = int(manifest.video.get("height") or 0)
    n_total = int(manifest.video.get("frame_count") or 0)
    emit(
        "segment",
        start=segment.start,
        end=segment.end,
        frames=segment.frames,
        detected_ratio=round(segment.detected_ratio, 4),
    )

    # This script lives on the Windows mount, so Python puts /mnt/c/... on
    # sys.path rather than the WHAM root we cd'd into. WHAM's packages (configs,
    # lib) are imported by bare name, so the root has to be added explicitly.
    if wham_root not in sys.path:
        sys.path.insert(0, wham_root)

    try:
        from configs.config import get_cfg_defaults
        from lib.data.datasets import CustomDataset
        from lib.models import build_body_model, build_network
        from lib.models.preproc.extractor import FeatureExtractor
        from lib.utils.imutils import avg_preds
        from lib.utils.transforms import matrix_to_axis_angle
    except Exception as exc:  # noqa: BLE001
        emit("error", message=f"WHAM import failed: {type(exc).__name__}: {exc}")
        write_meta(
            ReconstructionStatus.FAILED,
            fps=fps,
            segment=segment.to_dict(),
            frame_range=[segment.start, segment.end],
            errors=[f"WHAM import failed: {exc}"],
        )
        return 3

    cfg = get_cfg_defaults()
    cfg.merge_from_file(osp.join(wham_root, "configs/yamls/demo.yaml"))
    cfg.DEVICE = args.device
    if args.no_flip_eval:
        cfg.FLIP_EVAL = False

    device_info = {
        "requested": args.device,
        "cuda_available": bool(torch.cuda.is_available()),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }

    progress(8.0, "Building WHAM network…")
    smpl_batch_size = cfg.TRAIN.BATCH_SIZE * cfg.DATASET.SEQLEN
    smpl = build_body_model(cfg.DEVICE, smpl_batch_size)
    network = build_network(cfg, smpl)
    network.eval()

    keypoints_source = args.keypoints
    if keypoints_source == "vitpose":
        vit_ckpt = osp.join(wham_root, OPTIONAL_CHECKPOINTS["vitpose"])
        if not osp.exists(vit_ckpt):
            warnings.append("ViTPose checkpoint missing; using tracker (YOLO) keypoints instead")
            keypoints_source = "manifest"

    thresholds = AssociationThresholds(
        min_iou=args.assoc_min_iou,
        max_center_dist_frac=args.assoc_max_center,
        min_score=args.assoc_min_score,
        reject_ambiguous=args.assoc_reject_ambiguous,
        max_bridge_gap=args.max_bridge_gap,
    )

    progress(10.0, "Associating detections with the tracked lineman…")
    with torch.no_grad():
        tracking_results, associations, assoc_summary = build_tracking_results(
            args.video, frames, fps, keypoints_source, wham_root, cfg.DEVICE.lower(),
            warnings, thresholds, args,
        )

        assoc_path = osp.join(out_dir, "association.json")
        save_associations(
            assoc_path,
            associations,
            thresholds,
            extra={"segment": segment.to_dict(), "video": args.video},
        )

        first_frame = int(tracking_results[1]["frame_id"][0])
        n = int(tracking_results[1]["frame_id"].shape[0])
        slam_results, world_grounded = get_slam_results(
            args.video, out_dir, width, height, n_total, first_frame, n, args.run_slam, warnings
        )

        progress(52.0, "Extracting HMR2 image features…")
        extractor = FeatureExtractor(cfg.DEVICE.lower(), cfg.FLIP_EVAL)
        tracking_results = extractor.run(args.video, tracking_results)

        progress(72.0, "Running WHAM inference…")
        dataset = CustomDataset(cfg, tracking_results, slam_results, width, height, fps)
        if len(dataset) != 1:
            warnings.append(f"expected exactly 1 subject, dataset has {len(dataset)}")

        if cfg.FLIP_EVAL:
            flipped_batch = dataset.load_data(0, True)
            _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = flipped_batch
            flipped_pred = network(
                x, inits, features, mask=mask, init_root=init_root,
                cam_angvel=cam_angvel, return_y_up=True, **kwargs
            )
            batch = dataset.load_data(0)
            _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
            pred = network(
                x, inits, features, mask=mask, init_root=init_root,
                cam_angvel=cam_angvel, return_y_up=True, **kwargs
            )

            flipped_pose = flipped_pred["pose"].squeeze(0).reshape(-1, 24, 6)
            flipped_shape = flipped_pred["betas"].squeeze(0)
            pose = pred["pose"].squeeze(0).reshape(-1, 24, 6)
            shape = pred["betas"].squeeze(0)
            avg_pose, avg_shape = avg_preds(pose, shape, flipped_pose, flipped_shape)
            avg_pose = avg_pose.reshape(-1, 144)
            avg_contact = (flipped_pred["contact"][..., [2, 3, 0, 1]] + pred["contact"]) / 2

            network.pred_pose = avg_pose.view_as(network.pred_pose)
            network.pred_shape = avg_shape.view_as(network.pred_shape)
            network.pred_contact = avg_contact.view_as(network.pred_contact)
            output = network.forward_smpl(**kwargs)
            pred = network.refine_trajectory(output, cam_angvel, return_y_up=True)
        else:
            batch = dataset.load_data(0)
            _id, x, inits, features, mask, init_root, cam_angvel, frame_id, kwargs = batch
            pred = network(
                x, inits, features, mask=mask, init_root=init_root,
                cam_angvel=cam_angvel, return_y_up=True, **kwargs
            )

        progress(90.0, "Packing raw motion…")
        body_pose = matrix_to_axis_angle(pred["poses_body"]).cpu().numpy().reshape(-1, 69)
        root_cam = matrix_to_axis_angle(pred["poses_root_cam"]).cpu().numpy().reshape(-1, 3)
        root_world = matrix_to_axis_angle(pred["poses_root_world"]).cpu().numpy().reshape(-1, 3)
        pose_cam = np.concatenate((root_cam, body_pose), axis=-1)
        pose_world = np.concatenate((root_world, body_pose), axis=-1)
        trans_cam = (pred["trans_cam"] - network.output.offset).cpu().numpy().reshape(-1, 3)
        trans_world = pred["trans_world"].cpu().squeeze(0).numpy().reshape(-1, 3)
        betas = pred["betas"].cpu().squeeze(0).numpy().reshape(-1, 10)
        contact = pred["contact"].cpu().squeeze(0).numpy().reshape(-1, 4)

        frame_indices = np.asarray(frame_id, dtype=np.int64).reshape(-1)
        timestamps = frame_indices.astype(np.float64) / float(fps or 30.0)

        arrays: dict[str, Any] = {
            "frame_indices": frame_indices.astype(np.int32),
            "timestamps": timestamps.astype(np.float32),
            "pose_cam": pose_cam.astype(np.float32),
            "pose_world": pose_world.astype(np.float32),
            "betas": betas.astype(np.float32),
            "trans_cam": trans_cam.astype(np.float32),
            "trans_world": trans_world.astype(np.float32),
            "contact": contact.astype(np.float32),
            "keypoints_2d": tracking_results[1]["keypoints"].astype(np.float32),
            "bbox_cxcys": np.asarray(tracking_results[1]["bbox"], dtype=np.float32),
        }

        # Per-frame provenance so the viewer can flag interpolated motion instead
        # of presenting it as observed.
        conf_map = assoc_summary.get("frame_confidence") or {}
        interp = set(assoc_summary.get("interpolated_kept") or [])
        arrays["frame_confidence"] = np.asarray(
            [float(conf_map.get(int(f), conf_map.get(str(int(f)), 0.0))) for f in frame_indices],
            dtype=np.float32,
        )
        arrays["interpolated"] = np.asarray(
            [int(int(f) in interp) for f in frame_indices], dtype=np.int8
        )

        joints = getattr(network.output, "joints", None)
        if joints is not None:
            try:
                arrays["joints_cam"] = (
                    joints.detach().cpu().numpy().reshape(len(frame_indices), -1, 3).astype(np.float32)
                )
            except Exception:  # noqa: BLE001 - joints are derivable from SMPL params
                warnings.append("could not pack joints_cam; derive joints from SMPL params")
        if args.save_verts and "verts_cam" in pred:
            arrays["verts_cam"] = pred["verts_cam"].cpu().numpy().astype(np.float32)

    lengths = {k: int(v.shape[0]) for k, v in arrays.items()}
    if len(set(lengths.values())) != 1:
        warnings.append(f"array length mismatch: {lengths}")

    np.savez_compressed(npz_path, **arrays)

    status = ReconstructionStatus.OK if world_grounded else ReconstructionStatus.OK_LOCAL_ONLY
    write_meta(
        status,
        fps=fps,
        segment=segment.to_dict(),
        frame_range=[int(frame_indices[0]), int(frame_indices[-1])],
        frame_count=int(len(frame_indices)),
        outputs={
            "motion_raw": osp.abspath(npz_path),
            "motion_metadata": osp.abspath(meta_path),
            "association": osp.abspath(assoc_path),
        },
        wham={
            "root": wham_root,
            "checkpoint": cfg.TRAIN.CHECKPOINT,
            "flip_eval": bool(cfg.FLIP_EVAL),
            "keypoints_source": keypoints_source,
            "world_grounded": bool(world_grounded),
            "backbone": cfg.MODEL.BACKBONE,
            "note": "raw output; no smoothing, foot locking, or field calibration applied",
        },
        device=device_info,
        association=assoc_summary,
        stats={
            "arrays": sorted(arrays),
            "mean_contact": [round(float(v), 4) for v in contact.mean(axis=0)],
            "trans_world_span_m": [
                round(float(trans_world[:, i].max() - trans_world[:, i].min()), 4)
                for i in range(3)
            ],
        },
    )
    progress(100.0, f"Wrote motion_raw.npz ({len(frame_indices)} frames)")
    emit(
        "done",
        status=status,
        frames=int(len(frame_indices)),
        world_grounded=bool(world_grounded),
        runtime_seconds=round(time.time() - started, 2),
        warnings=warnings,
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run WHAM on an oline-cv TrackManifest")
    ap.add_argument("--tracks", help="path to tracks.json (POSIX path)")
    ap.add_argument("--video", help="original full-resolution video (POSIX path)")
    ap.add_argument("--out", help="output dir for motion_raw.npz")
    ap.add_argument("--repo", default=None, help="oline-cv repo root, for schema import")
    ap.add_argument("--wham-root", default=None, help="WHAM root (default: cwd)")
    ap.add_argument("--segment", default=None, help="explicit frame range START:END")
    ap.add_argument("--auto-segment", action="store_true", help="pick the best segment")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--keypoints", default="vitpose", choices=["vitpose", "manifest"])
    ap.add_argument(
        "--no-associate",
        dest="associate",
        action="store_false",
        default=True,
        help="skip detector association and trust the BoT-SORT box directly",
    )
    ap.add_argument("--det-conf", type=float, default=0.30, help="person detector confidence")
    ap.add_argument("--det-imgsz", type=int, default=1280)
    ap.add_argument("--assoc-min-iou", type=float, default=0.35)
    ap.add_argument("--assoc-max-center", type=float, default=0.45)
    ap.add_argument("--assoc-min-score", type=float, default=0.45)
    ap.add_argument("--assoc-reject-ambiguous", action="store_true")
    ap.add_argument(
        "--max-bridge-gap",
        type=int,
        default=3,
        help="interpolate dropouts up to this many frames (0 disables bridging)",
    )
    ap.add_argument(
        "--min-frames",
        type=int,
        default=30,
        help="minimum contiguous associated frames required to reconstruct",
    )
    ap.add_argument("--no-flip-eval", action="store_true")
    ap.add_argument("--no-slam", dest="run_slam", action="store_false", default=True)
    ap.add_argument("--save-verts", action="store_true")
    ap.add_argument("--doctor", action="store_true", help="report environment readiness and exit")
    args = ap.parse_args()

    wham_root = args.wham_root or os.getcwd()

    if args.doctor:
        emit("doctor", report=run_doctor(wham_root))
        return 0

    missing = [n for n in ("tracks", "video", "out") if not getattr(args, n)]
    if missing:
        emit("error", message=f"missing required arguments: {missing}")
        return 2

    args.wham_root = wham_root
    try:
        return reconstruct(args)
    except Exception as exc:  # noqa: BLE001 - must always report structurally
        emit("error", message=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        try:
            if args.repo and args.repo not in sys.path:
                sys.path.insert(0, args.repo)
            from oline_cv.motion3d.motion_schema import MotionMetadata, ReconstructionStatus

            MotionMetadata(
                status=ReconstructionStatus.FAILED,
                video=args.video or "",
                fps=0.0,
                segment={},
                frame_range=[0, 0],
                frame_count=0,
                runtime_seconds=0.0,
                errors=[f"{type(exc).__name__}: {exc}"],
            ).save(osp.join(args.out, "motion_metadata.json"))
        except Exception:  # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
