"""Load extracted documents (markdown / plain text) for indexing."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, List, Tuple

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
