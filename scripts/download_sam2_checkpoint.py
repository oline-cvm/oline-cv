"""Download the official SAM 2.1 tiny checkpoint into models/sam2/.

Source: https://github.com/facebookresearch/sam2 (Apache-2.0)
URL:    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
DEST = Path(__file__).resolve().parent.parent / "models" / "sam2" / "sam2.1_hiera_tiny.pt"


def main() -> int:
    DEST.parent.mkdir(parents=True, exist_ok=True)
    if DEST.is_file() and DEST.stat().st_size > 100_000_000:
        print(f"already present: {DEST} ({DEST.stat().st_size} bytes)")
        return 0
    print(f"downloading {URL}")
    print(f"        to {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    print(f"saved {DEST.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
