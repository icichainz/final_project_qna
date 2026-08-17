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
MIN_CONFIDENCE = 0.3   # chunk-token coverage below this -> page-level citation


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(unicodedata.normalize("NFKD", text.lower()))


def cache_dir_for(pdf_name: str) -> Optional[Path]:
    """Locate a PDF's page-cache directory by content fingerprint."""
    from gcf_qna.extraction.vlm import _fingerprint   # lazy: vlm pulls aiohttp
    pdf = config.RAW_PDF_DIR / pdf_name
    if not pdf.exists():
        return None
    d = Path(config.PAGE_CACHE_DIR) / _fingerprint(pdf)
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
                boxes = json.loads((cdir / e["boxes"]).read_text(encoding="utf-8"))
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
    for ln in boxes.get("lines", []):
        ltoks = _tokens(ln["text"])
        if not ltoks:
            continue
        hit = sum(1 for t in ltoks if t in chunk_set)
        if hit / len(ltoks) >= MIN_LINE_COVER:
            rects.append(ln["bbox"])
            covered += hit

    confidence = round(covered / len(chunk_tokens), 3) if chunk_tokens else 0.0
    if confidence < MIN_CONFIDENCE:
        return Grounding(doc_id, page, kind="page", confidence=confidence, image=image)
    return Grounding(doc_id, page, rects, kind="lines",
                     confidence=min(confidence, 1.0), image=image)
