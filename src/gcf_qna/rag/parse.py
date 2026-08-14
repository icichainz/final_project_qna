"""Load extracted documents (markdown / plain text) for indexing."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

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
