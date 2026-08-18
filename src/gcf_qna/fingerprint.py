"""Content fingerprint used to key the page cache — dependency-free.

Lives outside extraction/ deliberately: the grounding layer (app image,
no pymupdf/aiohttp) needs it to locate a PDF's cache directory.
"""
from __future__ import annotations

import hashlib
import os
import zlib
from pathlib import Path


def fingerprint(path: Path) -> str:
    """Cheap, collision-safe-enough PDF identity: size + head/tail content."""
    st = path.stat()
    h = zlib.crc32(path.name.encode())
    with path.open("rb") as f:
        head = f.read(1 << 20)
        if st.st_size > (1 << 21):
            f.seek(-(1 << 20), os.SEEK_END)
            tail = f.read(1 << 20)
        else:
            tail = b""
    d = hashlib.sha256()
    d.update(str(st.st_size).encode())
    d.update(head)
    d.update(tail)
    return f"{d.hexdigest()[:32]}_{h:08x}"
