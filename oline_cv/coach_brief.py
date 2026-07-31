"""Coach-facing brief from analysis numbers (shared by dashboard PDF + API)."""

from __future__ import annotations

from typing import Any


def build_coach_brief(r: dict[str, Any]) -> dict[str, Any]:
    flags = set(r.get("coach_language") or [])
    mods = r.get("modules") or {}
    fix: list[dict[str, str]] = []
    keep: list[dict[str, str]] = []
    play = "run" if r.get("play_type") == "run" else "pass"

    ms = r.get("reaction_time_ms")
    if r.get("late_off_the_ball") or "late_off_the_ball" in flags or (ms is not None and ms > 250):
        fix.append(
            {
                "title": "Get off the ball faster",
                "detail": (
                    f"First move came ~{int(round(ms))} ms after snap — push the first step on the ball."
                    if ms is not None
                    else "Looks late off the snap. Cue: move on the ball, not the defender."
                ),
            }
        )
    elif ms is not None and ms <= 180:
        keep.append(
            {
                "title": "Quick off the ball",
                "detail": f"First move in ~{int(round(ms))} ms — that tempo is a win. Keep it.",
            }
        )
    elif ms is not None:
        keep.append(
            {
                "title": "Acceptable get-off",
                "detail": f"First move ~{int(round(ms))} ms. Solid — chase a tick quicker next rep.",
            }
        )

    if r.get("initiated_by") == "hip":
        fix.append(
            {
                "title": "Lead with the feet, not the hips",
                "detail": "Hips moved before the feet. Teach a clean first step so the body doesn’t leak early.",
            }
        )
    elif r.get("initiated_by") == "foot":
        keep.append(
            {
                "title": "Foot-first start",
                "detail": "Feet fired first — good sequence off the snap.",
            }
        )

    posture = str(r.get("posture_classification") or "")
    if "bender" in posture or "waist_bender" in flags:
        fix.append(
            {
                "title": "Stay out of the waist bend",
                "detail": "Leaning at the waist. Cue: bend at the knees/ankles, keep the chest over the toes.",
            }
        )
    elif "balanced" in posture:
        keep.append(
            {
                "title": "Balanced posture",
                "detail": "Pad level and torso look controlled through the set.",
            }
        )

    foot = mods.get("footwork") or {}
    if foot.get("overset") or "overset" in flags:
        fix.append(
            {
                "title": "Don’t overset",
                "detail": "Set got too deep/wide. Shorten the second step — stay square to the rush lane.",
            }
        )

    base = r.get("mean_base_width")
    if "narrow_base" in flags or (base is not None and base < 0.35):
        fix.append(
            {
                "title": "Widen the base",
                "detail": "Feet got tight. Cue: athletic base — feel pressure on the inside of both feet.",
            }
        )
    elif base is not None and base >= 0.45:
        keep.append(
            {
                "title": "Good base width",
                "detail": "Feet stayed under the body with room to redirect.",
            }
        )

    mirror = r.get("lateral_match")
    if mirror is not None and mirror < 0.35:
        fix.append(
            {
                "title": "Mirror the rusher better",
                "detail": "Lateral match to the defender was soft. Stay attached — shuffle with their hips.",
            }
        )
    elif mirror is not None and mirror >= 0.55:
        keep.append(
            {
                "title": "Strong mirror",
                "detail": "Moved with the rusher laterally — that keeps the pocket clean.",
            }
        )

    give = r.get("anchor_give")
    if give is not None and give > 0.18:
        fix.append(
            {
                "title": "Anchor — stop giving ground",
                "detail": "Hips slid back after contact. Cue: drop the hips, stay connected, don’t catch high.",
            }
        )
    elif give is not None and give <= 0.1:
        keep.append(
            {
                "title": "Firm anchor",
                "detail": "Held ground after contact — pocket stayed put.",
            }
        )

    punch = r.get("punch_ms")
    if punch is not None and punch > 400:
        fix.append(
            {
                "title": "Get hands on sooner",
                "detail": f"Punch landed late (~{int(round(punch))} ms). Strike on arrival — don’t let them into your chest.",
            }
        )
    elif punch is not None and punch <= 280:
        keep.append(
            {
                "title": "Quick hands",
                "detail": f"Hands got there in ~{int(round(punch))} ms — keep striking on time.",
            }
        )

    engage = r.get("engagement_ms")
    if "early_disengage" in flags or (engage is not None and engage < 400 and play == "pass"):
        fix.append(
            {
                "title": "Finish the block longer",
                "detail": "Came off too early. Stay attached through the whistle — drive feet after contact.",
            }
        )
    elif engage is not None and engage >= 700:
        keep.append(
            {
                "title": "Sustained engagement",
                "detail": "Stayed on the block — that’s how pockets hold up.",
            }
        )

    if len(fix) >= 3:
        verdict = "Needs work — pick one cue and re-run"
        summary = f"Top fix: {fix[0]['title'].lower()}"
    elif not fix and keep:
        verdict = "Clean snap — build on what’s working"
        summary = keep[0]["title"]
    elif len(fix) == 1:
        verdict = "One clear coaching point"
        summary = fix[0]["title"]
    else:
        verdict = "Solid rep — chase one detail next"
        summary = "Pass-pro snapshot" if play == "pass" else "Run-block snapshot"

    return {
        "verdict": verdict,
        "summary": summary,
        "fix": fix[:4],
        "keep": keep[:3],
        "play": play,
    }
