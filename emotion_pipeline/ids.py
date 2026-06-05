import hashlib
import re
from pathlib import Path
from typing import Optional

# YouTube IDs are exactly 11 chars from [A-Za-z0-9_-], wrapped in square brackets.
_YT_ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]")


def parse_video_id(filename: str) -> Optional[str]:
    """Extract the 11-char YouTube id from a filename, or None if not present."""
    m = _YT_ID_RE.search(filename)
    return m.group(1) if m else None


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 and return the hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
