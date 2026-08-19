"""Load extracted documents (markdown / plain text) for indexing.

Also owns the chunk RECORD: the page-level, section-aware splitter below and
the two-text schema every consumer downstream relies on.

    source_text     the page markdown, verbatim. What the model is shown and
                    what ground.py matches against the PDF's line boxes.
    retrieval_text  source_text with the section path prepended. What gets
                    embedded and lexically indexed. NEVER shown, never grounded.

`source_text` is stored under the historical key "text", so an index written by
this module still loads in code that predates the schema, and grounding — which
reads chunk["text"] — keeps seeing the untouched page text. Both accessors
below tolerate either spelling, and an old index that has neither new field.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from gcf_qna.rag.chunk import chunk_text

TEXT_SUFFIXES = {".md", ".txt"}


def iter_documents(source_dir: Path) -> Iterator[Tuple[str, str]]:
    """Yield (doc_id, text) for every text file under source_dir.

    Hidden directories are skipped — that excludes the VLM pipeline's
    .pages/ sidecars and .stale/ quarantine, which would otherwise index
    every page twice. doc_id is the path relative to source_dir, no suffix.
    """
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"No such extraction dir: {source_dir}")
    for f in sorted(source_dir.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel = f.relative_to(source_dir)
        if any(part.startswith(".") for part in rel.parts[:-1]):
            continue
        text = f.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            yield str(rel.with_suffix("")), text


# The merge separator the extraction pipeline writes: "**Page N**" fenced by ---
# The tail is a single \n, not \n+: a greedy tail swallows the blank line before
# the *next* marker when a page body is empty, so that marker stops matching and
# its content gets mis-attributed to the empty page. Bodies are .strip()ed below,
# so consuming only one newline costs nothing.
_PAGE_MARK_RE = re.compile(r"(?:^|\n+)---\n\*\*Page (\d+)\*\*\n---\n")


def split_pages(text: str) -> List[Tuple[int, str]]:
    """Split merged extraction markdown into (page_no, body) pairs.

    Falls back to a single (0, text) pair when no page markers exist, so
    documents from other extractors still index (page 0 = unknown).
    """
    parts = _PAGE_MARK_RE.split(text)
    out: List[Tuple[int, str]] = []
    for i in range(1, len(parts) - 1, 2):
        body = parts[i + 1].strip()
        if body:
            out.append((int(parts[i]), body))
    if not out:
        body = text.strip()
        if body:
            out.append((0, body))
    return out


# --------------------------------------------------------------- sections ---

# ATX heading: '## A.8 Total GCF funding requested'. The VLM writes these for
# every template heading it sees, so '#'-tracking is the general mechanism.
_ATX_RE = re.compile(r"^ {0,3}(#{1,6})\s+(\S.*?)\s*#*\s*$")
# A bold-only line counts as a heading ONLY when it prints a section number:
# bold is also used for ordinary emphasis, and 'nearly half of children' is
# not a section. '**A.8. Total GCF funding requested**' is.
_BOLD_RE = re.compile(r"^ {0,3}\*\*\s*(\S.*?)\s*\*\*[.:]?\s*$")

# Running headers/footers the extractor keeps ('GREEN CLIMATE FUND FUNDING
# PROPOSAL V2.0 | PAGE 3 OF 15'). They carry no section identity, and taking
# them for headings would prefix a whole page with boilerplate.
_RUNNING_RE = re.compile(
    r"green\s+climate\s+fund|funding\s+proposal\s+v\.?\s?\d"
    r"|page\s+\d+\s+of\s+\d+|^\s*page\s+\d+\s*$|^\s*[\d.]+\s*$", re.I)

_DOTTED = r"[A-H]\.\s?\d{1,2}(?:\.\s?\d{1,2}){0,2}"
# The printed section id at the head of a title, in the four spellings the
# corpus uses. The bare-letter form ('A PROJECT/PROGRAMME SUMMARY') demands an
# upper-case run behind it, otherwise 'A funding proposal titled ...' would
# register as section A.
_SEC_ID_RE = re.compile(
    rf"^\s*\**\s*(?:(?P<dotted>{_DOTTED})(?=[\s.):]|$)"
    r"|section\s+(?P<sec>[A-H])(?=[\s:.)]|$)"
    r"|(?P<annex>annexe?\s+(?:[IVXL]{1,5}|\d{1,2}))(?=[\s.:)]|$)"
    r"|(?P<bare>[A-H])\s+(?=(?-i:[A-Z][A-Z/&,'’ -]{3,})))", re.I)

MAX_SECTION_PATH = 150   # chars; deeper ancestry is dropped from the front


def section_id(title: str) -> Optional[str]:
    """The section id a heading prints ('A.8', 'B', 'Annex II'), or None."""
    m = _SEC_ID_RE.match(title or "")
    if not m:
        return None
    if m.group("dotted"):
        return re.sub(r"\s+", "", m.group("dotted")).rstrip(".").upper()
    if m.group("sec"):
        return m.group("sec").upper()
    if m.group("annex"):
        return "ANNEX " + re.sub(r"\s+", " ", m.group("annex")).split(None, 1)[1].upper()
    return m.group("bare").upper()


def _id_depth(sid: str) -> int:
    return 1 if sid.startswith("ANNEX") else sid.count(".") + 1


def _clean_title(raw: str) -> str:
    t = re.sub(r"\*+", "", raw or "").strip()
    t = re.sub(r"\s+", " ", t).strip(" .:;-–—")
    return t


@dataclass
class _Head:
    level: int
    title: str
    sid: Optional[str]
    depth: Optional[int]


class SectionTracker:
    """Heading stack for one document, in reading order.

    Two hierarchies are in play and the printed one wins: the VLM's '#' depth
    drifts between pages (A.8 comes back as '##' and A.9 as '###' on the very
    same page), while the printed ids never lie about nesting — A.9 is a
    sibling of A.8 and a child of A. Headings without an id fall back to '#'
    depth. Across a page break only id-bearing headings survive, because a
    template section really does continue onto the next page while 'Summary'
    at the top of page 3 says nothing about page 4.
    """

    def __init__(self) -> None:
        self.stack: List[_Head] = []

    # -- page boundary -------------------------------------------------------
    def page_break(self) -> None:
        self.stack = [h for h in self.stack if h.sid]

    def reset(self) -> None:
        self.stack = []

    # -- feeding -------------------------------------------------------------
    def heading(self, line: str) -> bool:
        """Consume one line; True when it was a heading (the stack moved)."""
        m = _ATX_RE.match(line)
        if m:
            level, raw = len(m.group(1)), m.group(2)
        else:
            m = _BOLD_RE.match(line)
            if not m:
                return False
            raw = m.group(1)
            if not section_id(raw):
                return False
            level = 0                      # level comes from the id below
        title = _clean_title(raw)
        if not title:
            return False
        sid = section_id(title)
        if not sid and _RUNNING_RE.search(title):
            # a running header at the top of a page ends the carried section:
            # page 9 is not still inside 'A.19' just because page 8 ended there
            self.stack = []
            return True
        depth = _id_depth(sid) if sid else None
        if depth is not None:
            level = level or depth
            while self.stack:
                top = self.stack[-1]
                if top.depth is not None:
                    # siblings and deeper ids pop; a parent survives only when
                    # its id really is this id's prefix (A -> A.9, not A.8 -> A.9)
                    if top.depth >= depth or not sid.startswith(top.sid):
                        self.stack.pop()
                        continue
                elif top.level >= level:
                    self.stack.pop()
                    continue
                break
        else:
            while self.stack and self.stack[-1].level >= level:
                self.stack.pop()
        self.stack.append(_Head(level, title, sid, depth))
        return True

    # -- reading -------------------------------------------------------------
    @property
    def path(self) -> Optional[str]:
        if not self.stack:
            return None
        titles = [h.title for h in self.stack]
        while len(titles) > 1 and len(" > ".join(titles)) > MAX_SECTION_PATH:
            titles.pop(0)                  # keep the most specific ancestry
        return " > ".join(titles)[:MAX_SECTION_PATH]


# ---------------------------------------------------------------- chunking ---

@dataclass
class PageChunk:
    text: str                              # source text: page markdown, verbatim
    section_path: Optional[str] = None
    kind: str = "text"                     # text | table


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _segments(body: str, tracker: SectionTracker):
    """(section_path, kind, text) runs: heading changes and table edges break."""
    buf: List[str] = []
    kind = "text"
    path = tracker.path
    out = []

    def flush():
        nonlocal buf, kind
        if buf and any(ln.strip() for ln in buf):
            out.append((path, kind, "\n".join(buf).strip("\n")))
        buf = []

    for line in body.splitlines():
        if tracker.heading(line):
            flush()
            path = tracker.path
            kind = "text"
            buf.append(line)               # the heading line stays in the page text
            continue
        table = _is_table_line(line)
        if table and kind != "table":
            flush()
            path = tracker.path
            kind = "table"
        elif not table and kind == "table" and line.strip():
            flush()
            path = tracker.path
            kind = "text"
        elif not table and kind == "table":
            continue                       # blank line inside/after a table
        buf.append(line)
    flush()
    return out


def _kind(text: str) -> str:
    """'table' when pipe rows carry the chunk — the same majority rule
    ground.py uses to decide between line rects and table rects."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "table" if lines and sum(map(_is_table_line, lines)) * 2 >= len(lines) else "text"


def chunk_page(body: str, size: int = 1000, overlap: int = 200,
               tracker: Optional[SectionTracker] = None) -> List[PageChunk]:
    """Split one page into section-tagged chunks.

    Packs whole segments up to `size` instead of cutting at a character offset,
    so a chunk starts at a heading whenever one is near. Only two things break
    the packing: a heading and a table. A table that fits in 2x size is never
    split — a markdown table cut in half loses its header row and stops being
    answerable — and it keeps the nearest heading as its section path.
    """
    tracker = tracker or SectionTracker()
    out: List[PageChunk] = []
    buf: List[str] = []
    buf_path: Optional[str] = None
    buf_len = 0

    def flush():
        nonlocal buf, buf_path, buf_len
        if buf:
            text = "\n\n".join(buf).strip()
            if text:
                out.append(PageChunk(text, buf_path, _kind(text)))
        buf, buf_path, buf_len = [], None, 0

    for path, kind, text in _segments(body, tracker):
        atomic = kind == "table" and len(text) <= 2 * size
        if len(text) > size and not atomic:
            flush()
            # Nothing is repeated across the pieces (no re-emitted header row):
            # a chunk that is not verbatim page text stops grounding cleanly.
            for piece in chunk_text(text, size, overlap):
                out.append(PageChunk(piece, path, _kind(piece)))
            continue
        if buf and buf_len + len(text) + 2 > size:
            flush()
        if buf_path is None:
            # first *named* segment wins: a chunk that opens with a running
            # header or with the tail of an unheaded page still belongs to the
            # first section it actually contains
            buf_path = path
        buf.append(text)
        buf_len += len(text) + 2
    flush()
    return out


def chunk_document(text: str, size: int = 1000, overlap: int = 200
                   ) -> Iterator[Tuple[int, PageChunk]]:
    """(page_no, PageChunk) for a whole merged extraction file."""
    tracker = SectionTracker()
    for page_no, body in split_pages(text):
        tracker.page_break()
        for chunk in chunk_page(body, size, overlap, tracker):
            yield page_no, chunk


# ------------------------------------------------------------ chunk record ---

def make_record(doc_id: str, page: int, chunk: PageChunk,
                section_prefix: bool = False) -> Dict[str, Any]:
    """One chunks.jsonl row.

    `section_prefix` prepends the section path to the embedded text. It is OFF
    by default because it was measured to HURT this embedder
    (all-mpnet-base-v2), badly: re-embedding three proposals both ways and
    ranking the gold evidence page inside its own document gave

        question                     source text   + section path
        FP173 total GCF funding          2              14
        'fp 173 ... how much gcf $$$'    6              56
        FP151 funding (French)          10              16
        FP274 consistency (2 pages)    4, 7            1, 11

    A generic heading ('PROJECT/PROGRAMME SUMMARY') is a large fraction of a
    1000-character chunk, and it pulls every chunk of a section toward the same
    point. The path is still STORED: it drives same-section expansion at query
    time, and the A/B is one build flag away for the multilingual embedder the
    plan wants to try next.
    """
    rec: Dict[str, Any] = {"doc_id": doc_id, "page": page, "text": chunk.text}
    if chunk.section_path:
        rec["section_path"] = chunk.section_path
        if section_prefix:
            rec["retrieval_text"] = f"{chunk.section_path}\n\n{chunk.text}"
    if chunk.kind == "table":
        rec["kind"] = "table"
    return rec


def source_text(chunk: Dict[str, Any]) -> str:
    """What the model and the grounder see: page text with no retrieval prefix."""
    return chunk.get("source_text") or chunk.get("text") or ""


def retrieval_text(chunk: Dict[str, Any]) -> str:
    """What WAS embedded and lexically indexed for this chunk.

    Never derives a prefix a build did not apply: the string returned here has
    to be the one the vector was made from, or reranking and BM25 would score
    text the index does not contain.
    """
    return chunk.get("retrieval_text") or source_text(chunk)
