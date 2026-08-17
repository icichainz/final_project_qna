"""Render grounded citations as annotated page images.

Server-side highlighting: draw a Grounding's rects onto the cached page JPEG
(green = matched text lines, blue = table regions) and persist the result
under data/cache/highlights/, keyed by (doc, page, rect-set hash) so repeated
citations never redraw.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from gcf_qna import config
from gcf_qna.rag.ground import Grounding

HIGHLIGHT_DIR = config.DATA_DIR / "cache" / "highlights"

GREEN = (26, 158, 90)
BLUE = (36, 98, 199)


def annotated_page(g: Grounding) -> Optional[Path]:
    """Return a JPEG with the grounding's rects drawn; None without an image.

    A page-level grounding (no rects) returns the plain cached page, so the
    viewer always has something honest to show.
    """
    if g.image is None or not Path(g.image).exists():
        return None
    if not g.rects:
        return Path(g.image)

    key = hashlib.sha1(
        json.dumps([g.doc_id, g.page, g.kind, g.rects]).encode()
    ).hexdigest()[:16]
    out = HIGHLIGHT_DIR / f"{g.doc_id[:40]}_p{g.page:04d}_{key}.jpg"
    if out.exists():
        return out

    from PIL import Image, ImageDraw
    img = Image.open(g.image).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")
    color = BLUE if g.kind == "table" else GREEN
    for x0, y0, x1, y1 in g.rects:
        dr.rectangle((x0, y0, x1, y1), outline=color, width=3,
                     fill=(color[0], color[1], color[2], 40))
    HIGHLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.jpg")
    img.save(tmp, quality=88)
    tmp.replace(out)
    return out
