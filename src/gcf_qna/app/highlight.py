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


def _clip(rect, w: int, h: int):
    """Clip one rect to the image, or None when nothing of it lands inside.

    Rects reach here already corrected for page rotation (rag.ground normalizes
    sidecars on load), but a detector rect can still spill a hair past the page
    edge, and PIL raises on a box whose x1 < x0. Clip, then drop the empties.
    """
    x0, y0, x1, y1 = (float(v) for v in rect[:4])
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, x1 = max(0.0, min(x0, w - 1.0)), max(0.0, min(x1, w - 1.0))
    y0, y1 = max(0.0, min(y0, h - 1.0)), max(0.0, min(y1, h - 1.0))
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


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
    for rect in g.rects:
        box = _clip(rect, img.width, img.height)
        if box is None:
            continue
        dr.rectangle(box, outline=color, width=3,
                     fill=(color[0], color[1], color[2], 40))
    HIGHLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.jpg")
    img.save(tmp, quality=88)
    tmp.replace(out)
    return out
