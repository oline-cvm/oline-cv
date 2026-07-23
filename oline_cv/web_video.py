"""Browser-friendly H.264 re-encode for overlay playback."""

from __future__ import annotations

from pathlib import Path


def ensure_web_mp4(src_path: str, dst_path: str | None = None) -> str:
    """Re-encode overlay to yuv420p H.264 so browsers can play it.

    Falls back to the source path if ffmpeg/imageio is unavailable.
    """
    src = Path(src_path)
    if not src.exists():
        return src_path
    dst = Path(dst_path) if dst_path else src.with_name(src.stem + "_web.mp4")

    # Try imageio-ffmpeg
    try:
        import imageio_ffmpeg
        import subprocess

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
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
        subprocess.run(cmd, check=True, capture_output=True)
        if dst.exists() and dst.stat().st_size > 1000:
            return str(dst)
    except Exception:
        pass

    # Try system ffmpeg
    try:
        import subprocess

        cmd = [
            "ffmpeg",
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
        subprocess.run(cmd, check=True, capture_output=True)
        if dst.exists() and dst.stat().st_size > 1000:
            return str(dst)
    except Exception:
        pass

    return str(src)
