# Player tracking

OLINE follows one tapped offensive lineman through a clip. Tracking has two
layers:

1. **Deep HM-SORT** (`oline_cv/tracking/deep_hm_sort.py`) — a general
   multi-object tracker for every person the detector finds.
2. **Locked-target identity** (`oline_cv/tracking/target_identity.py`) — an
   OLINE-specific layer that decides which tracklet, if any, is the tapped
   player in each frame.

Policy: **wrong-person tracking is worse than missing tracking.** When the
target layer is unsure it outputs no box (`track_state = "LOST"`) instead of
moving to another player.

## Attribution

The generic tracker is an independent implementation after:

- Deep HM-SORT — M. Gran-Henriksen, H. A. Lindgaard, G. Kiss, F. Lindseth,
  "Deep HM-SORT: Enhancing Multi-Object Tracking in Sports with Deep
  Features, Harmonic Mean, and Expansion IOU", arXiv:2406.12081, 2024.
- Deep-EIoU — H.-W. Huang, C.-Y. Yang, W. Ramkumar, C.-I. Huang, J.-N. Hwang,
  P.-K. Kim, K. Lee, K. Kim, "Iterative Scale-Up ExpansionIoU and Deep
  Features Association for Multi-Object Tracking in Sports", WACV Workshops
  2024.

No third-party tracker source is copied; the official Deep-EIoU repository
has no license file. The locked-target layer, its gates and its gallery are
OLINE additions and are not part of either paper.

## Pipeline

```
video frames
  -> YOLO person detector (config.detect_model, conf >= detector_threshold)
  -> ReID embeddings for every box (batched)
  -> camera motion (previous -> current frame)
  -> DeepHMSort.update per frame          -> tracklets, per-frame FrameResult
  -> tap -> detection on the tapped frame -> LockedTargetIdentity
  -> forward from the tap, backward to frame 0
  -> top-down pose on the target box (unchanged)
  -> FramePose list (unchanged contract) -> metrics / overlay / TrackManifest
```

### Deep HM-SORT

- **Expansion IoU**: both the tracklet's last observed box and the detection
  are grown about their centres to `w(2E+1) x h(2E+1)` before IoU, so fast
  movers still overlap. `d_iou = 1 - EIoU`.
- **Appearance**: `d_app = 1 - cos(track feature, detection embedding)`,
  clipped to `[0, 1]`; missing embeddings give 1. Track features are an
  EMA `v = a v + (1-a) f` (a = 0.9), re-normalised.
- **Cost**: harmonic mean `H = 2ab/(a+b)` of `d_iou` and `d_app` (the
  zero-safe form of `2/(1/a + 1/b)`). As in Deep-EIoU, `d_app` is set to 1
  above `appearance_cost_threshold` and when `d_iou > proximity_threshold`.
- **Association** (Hungarian, scipy):
  1. detections with score >= 0.6 vs active and lost tracklets, iterative
     expansion `E = 0.3, 0.6`; lost tracklets only in the first pass;
     cutoff 0.8
  2. detections 0.4-0.6 vs remaining active tracklets, EIoU only
  3. tentative tracklets vs remaining high detections (confirmation)
  4. new tentative tracklets from what is left
- **Persistent tracklets**: confirmed tracklets are never deleted. They go
  `ACTIVE -> LOST -> HISTORICAL` and can be `REACTIVATED` at any later
  frame. IDs only increase. LOST/HISTORICAL tracklets are never returned as
  visible.

OLINE additions and deviations (documented in code):

- **Camera-motion compensation** (`camera_motion_compensation`, on by
  default; after BoT-SORT's global motion compensation, not part of Deep
  HM-SORT): a similarity transform between consecutive frames is fitted to
  background optical flow with player boxes masked out, and every
  tracklet's last box — and the target layer's last box, inverted for the
  backward pass — is moved with the camera. Sideline film pans and zooms
  far faster than players move.
- A tracklet's appearance is not refreshed from a box overlapping another
  detection (`appearance_update_overlap_threshold`).
- ByteTrack-style active/lost duplicate pruning is not applied because it
  would delete persistent tracklets.
- No Kalman prediction is used in the cost (Deep-EIoU matches on the last
  observed box too).

### Locked-target identity

Separate from tracker IDs: the logical target is always `target_id = 1`,
while `track_id` reports the underlying tracklet (it may change after a
reacquisition).

| State | Box emitted | `track_state` |
|---|---|---|
| `VISIBLE_CONFIDENT` | yes | `TRACKED` |
| `REACQUIRED` | yes | `REIDENTIFIED` |
| `VISIBLE_AMBIGUOUS` | no | `LOST` |
| `REACQUISITION_CANDIDATE` | no | `LOST` |
| `OCCLUDED` / `LOST` | no | `LOST` |

Short gaps between two boxed frames still get an interpolated box
(`interpolated=True`, NaN keypoints, not usable), as before.

- **Visible-target protection**: while visible the layer follows its bound
  tracklet. A nearby box can make the frame ambiguous (no box) but can never
  take the target. If the bound tracklet stops matching the target's
  gallery for `target_identity_fail_frames` frames it is dropped. If the
  target's detection has no tracker id yet, it continues only on tight
  frame-to-frame overlap (Expansion-IoU distance <=
  `low_assignment_cost_threshold`) with a clear margin.
- **Reacquisition** needs every gate: inside the expanded region around the
  last confident box (expansion grows with time unseen), Expansion-IoU
  distance <= `target_reacquire_max_iou_distance`, not a tracklet that was
  ever seen alongside the target, gallery distance <= the appearance gate,
  HM cost <= `target_reacquire_max_cost`, margin to the runner-up >=
  `target_reacquire_min_margin`, and the same winner for
  `target_reacquire_confirm_frames` frames (immediate when it is the
  target's own tracklet re-found by the tracker).
- **Adaptive appearance gate**: every confident frame measures the gallery
  distance to clearly separate players in view — people known not to be the
  target. Once enough are known, reacquisition requires a distance below
  their `target_impostor_quantile` (and below `target_reid_max_distance`).
  With generic embeddings teammates in the same uniform score close to the
  target; this keeps them out at the cost of fewer reacquisitions. The one
  exception: the target's own tracklet re-found by the tracker within
  `target_occluded_max_s` uses the static `target_reid_max_distance` (the
  tracker's association is independent evidence; every other gate still
  applies).
- **Clean gallery**: the lock embedding plus up to
  `locked_target_gallery_size - 1` recent samples, added only from
  confident frames with score >= `target_update_min_score`, no overlap with
  another detection, not at the frame edge, not tiny, unambiguous,
  consistent with the gallery, and at least `target_gallery_min_interval_s`
  apart. Skips are counted by reason in `ol_lock.target.*.gallery_skips`.

## ReID weights

`DeepHMSortConfig.reid_backend`:

- `auto` (default): OSNet when weights are configured, otherwise the generic
  fallback.
- `osnet`: torchreid OSNet from a local checkpoint. Requires
  `pip install torchreid` and either `reid_weights` or the environment
  variable `OLINE_REID_WEIGHTS` pointing at the file, e.g. the sports-trained
  OSNet released with Deep-EIoU (download it yourself from the project's
  README and place it anywhere, e.g. `models/reid/sports_osnet.pth`). The
  checkpoint is validated against the architecture; a missing file or a
  mismatched checkpoint is an error. **Weights are never downloaded and
  random weights are never used.**
- `yolo_embed`: pooled features from the person detector already used for
  detection. Generic, not trained for re-identification, and clearly weaker
  — teammates in the same uniform are hard to tell apart. Reported as
  `ol_lock.reid.quality = "generic"`.

```
set OLINE_REID_WEIGHTS=C:\path\to\sports_osnet.pth      (Windows)
export OLINE_REID_WEIGHTS=/path/to/sports_osnet.pth     (macOS / Linux)
```

## Configuration

`AnalysisConfig.tracker_backend`: `"deep_hm_sort"` (default) or
`"legacy_botsort"` (deprecated detector + BoT-SORT + whole-clip link, kept
only for A/B comparison). Environment override: `OLINE_TRACKER_BACKEND`.

`AnalysisConfig.deep_hm` (`DeepHMSortConfig`) holds every threshold. Paper
defaults: detector 0.4, high confidence 0.6, assignment cutoff 0.8,
appearance cutoff 0.3, expansion 0.3 + 0.3 per iteration.

## Debugging

`--debug-tracking` (CLI), `debug_tracking=True`, or `OLINE_TRACK_DEBUG=1`
writes to `track_debug_dir` (default `outputs/track_debug/`):

- `<clip>_tracking.jsonl` — header with config, then one line per frame:
  every tracklet (id, state, stage, cost terms, appearance update) and the
  target decision (state, cost, `d_iou`, `d_app`, margin, frames unseen,
  expansion, gallery update / skip reason, top candidates).
- `<clip>_tracking_debug.mp4` — all tracklets in grey with IDs, the target
  coloured by state, and a status line.

## Evaluation

```
python scripts/evaluate_target_tracking.py uploads/clip.mp4 --pick-xy 0.43,0.72 --pick-t 0.2
python scripts/evaluate_target_tracking.py uploads/clip.mp4 --gt gt.json --tracker both
```

Ground truth (frames not listed are ignored; `null` = target hidden):

```json
{
  "pick_xy": [0.43, 0.72],
  "pick_t": 0.2,
  "target": {"boxes": {"0": [804, 735, 837, 834], "57": null}}
}
```

Reported: target recall / precision, wrong-person frames and rate, identity
switches, longest wrong run, and for each annotated occlusion the
wrong-person frames during it and reacquisition latency. Without ground
truth: coverage, state counts, tracker-id changes and per-frame jumps.

## Limitations

- With the generic `yolo_embed` backend, appearance barely separates
  teammates, so the target layer reacquires rarely and drops more frames
  during line play. A sports ReID checkpoint is the main lever for coverage.
- While the target is visible the layer trusts the tracker's continuity; a
  tracker identity swap between two look-alike players that stay in contact
  is only caught if appearance diverges.
- Camera compensation follows the background; it does not predict player
  motion, so long occlusions rely on the growing expansion region.
- CPU runtime is dominated by detection at `pose_imgsz` on every frame.
