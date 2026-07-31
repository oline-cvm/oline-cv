"""Segment selection for 3D reconstruction.

A clip's track is rarely reconstructable end to end: the OL leaves frame, gets
occluded, or the association layer correctly prefers LOST over a wrong player.
Feeding those stretches to WHAM injects noise into the global trajectory, so we
reconstruct contiguous high-quality segments only.

Stdlib only, so the same selection logic runs on the Windows side and inside the
WSL conda env without pulling numpy into the import path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

# WHAM's own temporal models need a reasonable window to stabilize; below this
# the trajectory decoder output is not trustworthy.
MIN_SEGMENT_FRAMES = 30


@dataclass(frozen=True)
class Segment:
    """A contiguous, reconstructable frame range in original frame-index space."""

    start: int
    end: int
    frames: int
    detected: int
    interpolated: int
    usable: int

    @property
    def detected_ratio(self) -> float:
        return self.detected / self.frames if self.frames else 0.0

    @property
    def usable_ratio(self) -> float:
        return self.usable / self.frames if self.frames else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "frames": self.frames,
            "detected": self.detected,
            "interpolated": self.interpolated,
            "usable": self.usable,
            "detected_ratio": round(self.detected_ratio, 4),
            "usable_ratio": round(self.usable_ratio, 4),
        }

    def __contains__(self, frame_index: int) -> bool:
        return self.start <= int(frame_index) <= self.end


def describe_segments(manifest: Any, min_frames: int = MIN_SEGMENT_FRAMES) -> list[Segment]:
    """Turn a TrackManifest's raw segments into scored Segment records."""
    out: list[Segment] = []
    for pair in manifest.segments or []:
        start, end = int(pair[0]), int(pair[1])
        frames = [f for f in manifest.frames if start <= f.frame_index <= end]
        if len(frames) < min_frames:
            continue
        out.append(
            Segment(
                start=start,
                end=end,
                frames=len(frames),
                detected=sum(1 for f in frames if f.bbox_source == "detected"),
                interpolated=sum(1 for f in frames if f.bbox_source == "interpolated"),
                usable=sum(1 for f in frames if f.usable),
            )
        )
    return out


def select_segment(
    manifest: Any,
    explicit: tuple[int, int] | None = None,
    min_frames: int = MIN_SEGMENT_FRAMES,
) -> Segment:
    """Pick the segment to reconstruct.

    With ``explicit`` the caller pins a range (used in production, where clip
    review already determined the good window). The range must be covered by a
    reconstructable segment — we refuse to silently reconstruct frames the
    tracker never resolved.

    Without ``explicit``, the best segment wins: most detected frames, ties
    broken by length then by earliest start (the rep usually precedes the
    post-play milling around).
    """
    segments = describe_segments(manifest, min_frames=min_frames)
    if not segments:
        raise ValueError(
            f"no reconstructable segment of at least {min_frames} frames in this track"
        )

    if explicit is not None:
        start, end = int(explicit[0]), int(explicit[1])
        if end < start:
            raise ValueError(f"invalid segment range {start}-{end}")
        for seg in segments:
            if seg.start <= start and end <= seg.end:
                frames = [f for f in manifest.frames if start <= f.frame_index <= end]
                if len(frames) < min_frames:
                    raise ValueError(
                        f"segment {start}-{end} has {len(frames)} frames, "
                        f"below the {min_frames}-frame minimum"
                    )
                return Segment(
                    start=start,
                    end=end,
                    frames=len(frames),
                    detected=sum(1 for f in frames if f.bbox_source == "detected"),
                    interpolated=sum(1 for f in frames if f.bbox_source == "interpolated"),
                    usable=sum(1 for f in frames if f.usable),
                )
        available = ", ".join(f"{s.start}-{s.end}" for s in segments)
        raise ValueError(
            f"requested segment {start}-{end} is not inside a reconstructable "
            f"segment (available: {available})"
        )

    return sorted(segments, key=lambda s: (-s.detected, -s.frames, s.start))[0]


def frames_in_segment(manifest: Any, segment: Segment) -> list[Any]:
    """Manifest frames inside a segment, ordered by original frame index.

    Frames without a bbox are excluded: WHAM cannot be given a hole. Because a
    Segment is contiguous-by-construction this should be a no-op, but it is
    cheap insurance against a manifest edited by hand.
    """
    frames = [
        f
        for f in manifest.frames
        if segment.start <= f.frame_index <= segment.end and f.bbox is not None
    ]
    return sorted(frames, key=lambda f: f.frame_index)


def contiguity_gaps(frames: Sequence[Any]) -> list[tuple[int, int]]:
    """Gaps in an otherwise contiguous frame_index sequence, as (after, before)."""
    gaps: list[tuple[int, int]] = []
    for a, b in zip(frames, frames[1:]):
        if b.frame_index != a.frame_index + 1:
            gaps.append((a.frame_index, b.frame_index))
    return gaps
