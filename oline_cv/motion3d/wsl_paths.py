"""Windows <-> WSL path translation.

The Windows application never imports WHAM; it hands paths to a Linux process.
Every path crossing that boundary goes through here so the conversion is in one
tested place instead of scattered f-strings.

    C:\\Users\\rishb\\proj\\a.mp4   <->  /mnt/c/Users/rishb/proj/a.mp4

Rules:
- Drive letters are lowercased (WSL mounts /mnt/c, not /mnt/C).
- Backslashes become forward slashes.
- Paths that are already POSIX are passed through unchanged, so a caller can
  hand us a native Linux path (e.g. the WHAM repo root) without special-casing.
"""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

_WIN_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$", re.DOTALL)
_WSL_MOUNT = re.compile(r"^/mnt/([a-zA-Z])/(.*)$", re.DOTALL)


def is_windows_path(path: str) -> bool:
    return bool(_WIN_DRIVE.match(str(path)))


def windows_to_wsl(path: str | Path) -> str:
    """Convert a Windows path to its /mnt/<drive> WSL equivalent.

    POSIX-looking input is returned unchanged (already a Linux path).
    """
    s = str(path)
    m = _WIN_DRIVE.match(s)
    if not m:
        if s.startswith("\\\\"):
            raise ValueError(f"UNC paths are not supported by the WSL bridge: {s!r}")
        return s.replace("\\", "/")
    drive, rest = m.group(1).lower(), m.group(2)
    rest = rest.replace("\\", "/")
    rest = str(PurePosixPath(rest)) if rest else ""
    if rest in (".", ""):
        return f"/mnt/{drive}"
    return f"/mnt/{drive}/{rest}"


def wsl_to_windows(path: str) -> str:
    """Convert a /mnt/<drive>/... WSL path back to Windows form."""
    s = str(path)
    m = _WSL_MOUNT.match(s)
    if not m:
        return s
    drive, rest = m.group(1).upper(), m.group(2)
    return f"{drive}:\\" + rest.replace("/", "\\")


def quote_posix(path: str) -> str:
    """Single-quote a POSIX path for safe embedding in a bash -lc command."""
    return "'" + str(path).replace("'", "'\\''") + "'"
