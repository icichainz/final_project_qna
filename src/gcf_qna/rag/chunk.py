"""Split document text into overlapping chunks.

Same contract as the RecursiveCharacterTextSplitter the old code used
(prefer paragraph breaks, then lines, then sentences, then a hard cut),
without the langchain dependency.
"""
from __future__ import annotations

from typing import List

SEPARATORS = ["\n\n", "\n", ". ", " "]


def chunk_text(text: str, size: int = 1000, overlap: int = 200) -> List[str]:
    if size <= 0:
        raise ValueError("size must be positive")
    overlap = max(0, min(overlap, size // 2))
    chunks: List[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            for sep in SEPARATORS:
                cut = window.rfind(sep)
                if cut > size // 2:  # only break past the midpoint
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks
