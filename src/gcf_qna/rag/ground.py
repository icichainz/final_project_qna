"""Ground RAG chunks to page regions using the page-cache boxes sidecars.

The chunk text is VLM-written markdown; the boxes text is the PDF's embedded
text layer. They never match verbatim (the VLM reflows tables, drops running
headers, normalizes punctuation), so grounding is fuzzy:

  * per text line: fraction of the line's tokens present in the chunk;
    lines above MIN_LINE_COVER become highlight rects
  * table-dominant chunks (markdown pipe rows) ground to the page's detected
    table region(s) instead of lines
  * overall confidence below MIN_CONFIDENCE degrades to a page-level citation
    (page with no rects) — a citation that is right beats a highlight that is
    wrong

All rects are pixel coordinates on the cached page JPEG.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from gcf_qna import config

_WORD_RE = re.compile(r"[a-z0-9]+")

MIN_LINE_COVER = 0.6   # fraction of a line's tokens that must appear in the chunk
MIN_LINE_HITS = 2      # ...and this many of them in absolute terms
MIN_CONFIDENCE = 0.3   # chunk-token coverage below this -> page-level citation
SHORT_CHUNK_TOKENS = 8  # below this a chunk needs an exact line match to stay line-level


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(unicodedata.normalize("NFKD", text.lower()))


# ------------------------------------------------------------- rotation ----

# Rect kinds recorded in *unrotated* page space, and therefore needing the
# transform below. Verified against all 205 rotated pages in the corpus by
# re-extracting each page after pymupdf's remove_rotation():
#     lines / words  <- page.get_text("dict")                  unrotated
#     figures        <- get_image_rects(), cluster_drawings()   unrotated
#     tables         <- page.find_tables()                      ALREADY rendered
# find_tables is the odd one out -- it reports display coordinates -- so table
# rects must be left exactly as they are.
_ROTATED_SPACE_KEYS = ("lines", "words", "figures")


def rotate_rect(bbox: List[float], rotation: int,
                img_w: float, img_h: float) -> List[float]:
    """Map one [x0,y0,x1,y1] from unrotated-page pixels to rendered-image pixels.

    The renderer draws the page *after* applying /Rotate, so a 90-degree page
    whose unrotated pixel box is Wu x Hu comes out as an image of Wi=Hu, Hi=Wu.
    Reading the size off the recorded image dims keeps this exact and free of
    the zoom rounding stored in the sidecar:

        90 :  (x, y) -> (Hu - y, x)      = (Wi - y, x)
        180:  (x, y) -> (Wu - x, Hu - y) = (Wi - x, Hi - y)
        270:  (x, y) -> (y, Wu - x)      = (y, Hi - x)

    Corners swap under rotation, so the result is re-ordered into x0<=x1, y0<=y1.
    Matches pymupdf's page.rotation_matrix to within 0.0 px on the real corpus.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox[:4])
    if rotation == 90:
        r = (img_w - y1, x0, img_w - y0, x1)
    elif rotation == 180:
        r = (img_w - x1, img_h - y1, img_w - x0, img_h - y0)
    elif rotation == 270:
        r = (y0, img_h - x1, y1, img_h - x0)
    else:
        r = (x0, y0, x1, y1)
    return [round(min(r[0], r[2]), 1), round(min(r[1], r[3]), 1),
            round(max(r[0], r[2]), 1), round(max(r[1], r[3]), 1)]


def normalize_boxes(boxes: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Put every rect in a boxes sidecar into rendered-image pixel space.

    Extraction writes text/figure rects in unrotated page space but the cached
    JPEG is rendered rotated, so on a rotated page the two disagree and rects
    land transposed or off-image. Fixing it here rather than in the extractor
    heals every sidecar already on disk without re-extracting the corpus.

    Returns a shallow copy with 'rotation' zeroed, so a second call is a no-op.
    """
    if not boxes:
        return boxes
    rot = int(boxes.get("rotation") or 0) % 360
    img_w, img_h = boxes.get("w"), boxes.get("h")
    if rot not in (90, 180, 270) or not img_w or not img_h:
        return boxes
    out = dict(boxes)
    for key in _ROTATED_SPACE_KEYS:
        items = boxes.get(key)
        if not items:
            continue
        out[key] = [dict(it, bbox=rotate_rect(it["bbox"], rot, img_w, img_h))
                    if it.get("bbox") else it for it in items]
    out["rotation"] = 0
    out["rotation_applied"] = rot
    return out


def cache_dir_for(pdf_name: str) -> Optional[Path]:
    """Locate a PDF's page-cache directory by content fingerprint."""
    from gcf_qna.fingerprint import fingerprint
    pdf = config.RAW_PDF_DIR / pdf_name
    if not pdf.exists():
        return None
    d = Path(config.PAGE_CACHE_DIR) / fingerprint(pdf)
    return d if (d / "metadata.json").exists() else None


def load_page_assets(doc_id: str, page: int,
                     cache_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return {'image': Path|None, 'boxes': dict|None} for one page of a doc."""
    cdir = cache_dir or cache_dir_for(f"{doc_id}.pdf")
    if cdir is None:
        return None
    meta = json.loads((cdir / "metadata.json").read_text(encoding="utf-8"))
    for e in meta.get("pages", []):
        if e.get("n") == page:
            boxes = None
            if e.get("boxes") and (cdir / e["boxes"]).exists():
                # Single choke point for sidecars: everything downstream of here
                # sees rects in the coordinate space of the cached JPEG.
                boxes = normalize_boxes(
                    json.loads((cdir / e["boxes"]).read_text(encoding="utf-8")))
            image = cdir / e["img"] if e.get("img") else None
            return {"image": image, "boxes": boxes}
    return None


@dataclass
class Grounding:
    doc_id: str
    page: int
    rects: List[List[float]] = field(default_factory=list)  # [x0,y0,x1,y1] px
    kind: str = "lines"        # lines | table | page
    confidence: float = 0.0
    image: Optional[Path] = None


def ground_chunk(chunk: Dict[str, Any],
                 cache_dir: Optional[Path] = None) -> Optional[Grounding]:
    """Map one retrieved chunk {'doc_id','page','text'} to page regions."""
    doc_id, page = chunk.get("doc_id"), chunk.get("page")
    if not doc_id or not page:
        return None                      # page 0/None: pre-page-aware index
    assets = load_page_assets(doc_id, page, cache_dir)
    if not assets:
        return None
    boxes, image = assets["boxes"], assets["image"]
    if not boxes:
        return Grounding(doc_id, page, kind="page", image=image)
    text = chunk.get("text", "")

    # Table-dominant chunk -> the detected table region(s) on that page.
    pipe_rows = sum(1 for ln in text.splitlines() if ln.lstrip().startswith("|"))
    if pipe_rows >= 3 and boxes.get("tables"):
        return Grounding(doc_id, page, [t["bbox"] for t in boxes["tables"]],
                         kind="table", confidence=0.9, image=image)

    chunk_tokens = _tokens(text)
    chunk_set = set(chunk_tokens)
    rects: List[List[float]] = []
    covered = 0
    exact = 0
    for ln in boxes.get("lines", []):
        ltoks = _tokens(ln.get("text", ""))
        if not ltoks:
            continue
        hit = sum(1 for t in ltoks if t in chunk_set)
        # Two hurdles, not one. A fraction alone makes every single-token line --
        # a page number, a list marker, a stray '5' -- a 100% match for any chunk
        # containing that token, and enough of those drive confidence to 1.0
        # while the highlight sits on a footer. The absolute floor is the
        # constraint that single-token coincidences must never reach
        # MIN_CONFIDENCE: a 1-token line can no longer qualify at all, while a
        # 2-token line still can, but only when the chunk covers both tokens.
        # For lines of 3+ tokens MIN_LINE_COVER already implies >= 2 hits, so
        # this only bites where it should.
        if hit < MIN_LINE_HITS or hit / len(ltoks) < MIN_LINE_COVER:
            continue
        rects.append(ln["bbox"])
        covered += hit
        if hit == len(ltoks):
            exact += 1

    confidence = round(covered / len(chunk_tokens), 3) if chunk_tokens else 0.0
    # A short chunk clears MIN_CONFIDENCE on very little evidence, so it has to
    # clear a second bar: at least one matched line the chunk covers completely.
    # Otherwise cite the page and draw nothing -- right beats precise-and-wrong.
    thin = len(chunk_tokens) < SHORT_CHUNK_TOKENS and not exact
    if not rects or confidence < MIN_CONFIDENCE or thin:
        return Grounding(doc_id, page, kind="page", confidence=confidence, image=image)
    return Grounding(doc_id, page, rects, kind="lines",
                     confidence=min(confidence, 1.0), image=image)
