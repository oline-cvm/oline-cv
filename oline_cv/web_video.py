"""Browser-friendly H.264 re-encode for overlay playback."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def _run_ffmpeg(ffmpeg: str, src: Path, dst: Path) -> bool:
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = " | ".join(err[-3:]) if err else "unknown ffmpeg error"
        log.warning("ffmpeg web encode failed (%s): %s", ffmpeg, tail)
        return False
    return dst.exists() and dst.stat().st_size > 1000


def ensure_web_mp4(src_path: str, dst_path: str | None = None) -> str:
    """Re-encode overlay to yuv420p H.264 so browsers can play it.

    OpenCV writes mp4v/FMP4 which most browsers cannot play. Falls back to the
    source path if ffmpeg/imageio is unavailable (preview will be blank).
    """
    src = Path(src_path)
    if not src.exists():
        return src_path
    dst = Path(dst_path) if dst_path else src.with_name(src.stem + "_web.mp4")

    # Already have a web encode
    if dst.exists() and dst.stat().st_size > 1000 and dst.resolve() != src.resolve():
        return str(dst)

    # Try imageio-ffmpeg
    try:
        import imageio_ffmpeg

        if _run_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe(), src, dst):
            return str(dst)
    except Exception as exc:
        log.warning("imageio-ffmpeg unavailable for web encode: %s", exc)

    # Try system ffmpeg
    try:
        if _run_ffmpeg("ffmpeg", src, dst):
            return str(dst)
    except Exception as exc:
        log.warning("system ffmpeg unavailable for web encode: %s", exc)

    log.warning(
        "Returning raw overlay %s — browsers often cannot play OpenCV mp4v. "
        "Install imageio-ffmpeg or ffmpeg for H.264 preview.",
        src.name,
    )
    return str(src)
