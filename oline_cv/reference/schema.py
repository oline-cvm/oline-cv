"""Versioned expert-annotation schema for reference OL clips.

An annotation records what a knowledgeable football person says about a rep.
It is authored by hand, never derived from pose data, and never converted into
a numeric score — ``quality`` stays a neutral categorical verdict so that a
later study can test whether CV measurements actually track expert judgement.

Only ``schema_version``, ``rep_id``, ``video_file`` and ``expert.quality`` are
required. Position, technique and side may be omitted or left ``"unknown"``
when the annotator genuinely cannot tell from the film; an unreadable rep is
better recorded as unknown than guessed at.

Enum values are validated against ``oline_cv.benchmark.schema`` so the
annotation vocabulary and the benchmark vocabulary can never drift apart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from oline_cv.benchmark.schema import (
    PLAY_TYPES,
    POSITIONS,
    SIDES,
    normalize_play_type,
    normalize_position,
    normalize_side,
    normalize_technique,
)

ANNOTATION_SCHEMA_VERSION = "1.0"

# The expert's verdict on whether this rep belongs in reference footage.
# "good" does not mean the player is good; it means this rep is usable as a
# reference example. A future NFL player still has bad reps in high school.
QUALITY_VALUES: tuple[str, ...] = ("good", "not_good", "uncertain")

REQUIRED_TOP_LEVEL: tuple[str, ...] = ("schema_version", "rep_id", "video_file")

# Common competition levels. Free-form is allowed; this is only a hint for the
# CLI, because level is context, not a validated label.
COMMON_LEVELS: tuple[str, ...] = ("high_school", "college", "nfl", "unknown")


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and value.strip() != ""


def _is_unknown(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == "unknown"


def validate_annotation(
    obj: Any,
    *,
    rep_id: str | None = None,
) -> list[str]:
    """Return a list of human-readable problems. Empty list means valid.

    ``rep_id`` is the id implied by the filename; when given, the annotation's
    own ``rep_id`` must match it so a misnamed file cannot silently attach an
    expert's notes to the wrong clip.
    """
    errors: list[str] = []

    if not isinstance(obj, dict):
        return ["annotation is not a JSON object"]

    for key in REQUIRED_TOP_LEVEL:
        if not _nonempty_str(obj.get(key)):
            errors.append(f"missing or empty required field: {key}")

    version = obj.get("schema_version")
    if _nonempty_str(version) and version != ANNOTATION_SCHEMA_VERSION:
        errors.append(
            f"unsupported schema_version {version!r} "
            f"(this build understands {ANNOTATION_SCHEMA_VERSION!r})"
        )

    own_id = obj.get("rep_id")
    if rep_id is not None and _nonempty_str(own_id) and own_id != rep_id:
        errors.append(f"rep_id {own_id!r} does not match its filename ({rep_id!r})")

    video_file = obj.get("video_file")
    if _nonempty_str(video_file) and _nonempty_str(own_id):
        if Path(str(video_file)).stem != own_id:
            errors.append(
                f"video_file {video_file!r} does not match rep_id {own_id!r}"
            )

    player = obj.get("player")
    if player is not None:
        if not isinstance(player, dict):
            errors.append("player must be an object")
        else:
            position = player.get("position")
            if _nonempty_str(position) and not _is_unknown(position):
                if normalize_position(position) == "unknown":
                    errors.append(
                        f"invalid position {position!r} (expected one of {list(POSITIONS)})"
                    )

    play = obj.get("play")
    if play is not None:
        if not isinstance(play, dict):
            errors.append("play must be an object")
        else:
            play_type = play.get("play_type")
            if _nonempty_str(play_type) and not _is_unknown(play_type):
                if str(play_type).strip().lower() not in PLAY_TYPES:
                    errors.append(
                        f"invalid play_type {play_type!r} "
                        f"(expected one of {[*PLAY_TYPES, 'unknown']})"
                    )
            side = play.get("side")
            if _nonempty_str(side) and not _is_unknown(side):
                if normalize_side(side) == "unknown":
                    errors.append(
                        f"invalid side {side!r} (expected one of {list(SIDES)})"
                    )
            technique = play.get("technique")
            if technique is not None and not isinstance(technique, str):
                errors.append("technique must be a string")

    expert = obj.get("expert")
    if expert is None:
        errors.append("missing required field: expert.quality")
    elif not isinstance(expert, dict):
        errors.append("expert must be an object")
    else:
        quality = expert.get("quality")
        if not _nonempty_str(quality):
            errors.append("missing required field: expert.quality")
        elif str(quality).strip().lower() not in QUALITY_VALUES:
            errors.append(
                f"invalid expert.quality {quality!r} (expected one of {list(QUALITY_VALUES)})"
            )
        notes = expert.get("notes")
        if notes is not None:
            if not isinstance(notes, list) or not all(
                isinstance(n, str) for n in notes
            ):
                errors.append("expert.notes must be a list of strings")

    return errors


def make_annotation(
    *,
    rep_id: str,
    video_file: str,
    position: str | None = None,
    level: str | None = None,
    play_type: str | None = None,
    technique: str | None = None,
    side: str | None = None,
    quality: str,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    """Build an annotation dict. Unspecified football context stays "unknown".

    Notes are stored verbatim; nothing rewrites, summarises or interprets them.
    """
    quality_clean = str(quality).strip().lower()
    if quality_clean not in QUALITY_VALUES:
        raise ValueError(
            f"quality must be one of {list(QUALITY_VALUES)}, got {quality!r}"
        )

    return {
        "schema_version": ANNOTATION_SCHEMA_VERSION,
        "rep_id": rep_id,
        "video_file": video_file,
        "player": {
            "level": (level or "unknown").strip() or "unknown",
            "position": normalize_position(position),
        },
        "play": {
            # play_type has a real default ("pass") in the benchmark vocabulary,
            # so only normalise when the annotator actually supplied one.
            "play_type": normalize_play_type(play_type) if play_type else "unknown",
            "technique": normalize_technique(technique),
            "side": normalize_side(side),
        },
        "expert": {
            "quality": quality_clean,
            "notes": list(notes or []),
        },
    }


def annotation_quality(annotation: dict[str, Any]) -> str:
    """The expert verdict, lowercased. "uncertain" when absent."""
    quality = ((annotation or {}).get("expert") or {}).get("quality")
    if not _nonempty_str(quality):
        return "uncertain"
    return str(quality).strip().lower()


def annotation_context(annotation: dict[str, Any]) -> dict[str, str]:
    """Normalised football context used for benchmark-eligibility filtering."""
    player = (annotation or {}).get("player") or {}
    play = (annotation or {}).get("play") or {}
    raw_play_type = play.get("play_type")
    return {
        "position": normalize_position(player.get("position")),
        # Keep "unknown" distinct from "pass" so an unannotated rep is never
        # silently treated as a pass set.
        "play_type": (
            normalize_play_type(raw_play_type) if _nonempty_str(raw_play_type)
            and not _is_unknown(raw_play_type) else "unknown"
        ),
        "technique": normalize_technique(play.get("technique")),
        "side": normalize_side(play.get("side")),
    }
