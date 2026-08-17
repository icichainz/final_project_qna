"""Query the index.

Filters FAISS's -1 padding: when the index holds fewer vectors than top_k,
FAISS pads ids with -1, and Python's metadata[-1] silently returns the LAST
chunk — the old codebase fed that unrelated passage to the LLM as context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from gcf_qna.rag.embed import Embedder


@dataclass
class Hit:
    text: str
    doc_id: str
    score: float
    page: Optional[int] = None   # 1-based; None/0 = unknown (pre-page-aware index)


def _doc_match(doc_id: str, wanted: str) -> bool:
    """Forgiving doc match: the LLM may cite a truncated or partial id."""
    a, b = doc_id.lower(), wanted.lower()
    return a == b or a.startswith(b) or b in a


class Retriever:
    def __init__(self, index, chunks: List[Dict[str, Any]], embedder: Embedder):
        self.index = index
        self.chunks = chunks
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5,
               doc_filter: Optional[str] = None) -> List[Hit]:
        """Top-k chunks for a query; doc_filter restricts hits to one document.

        Scoped search widens the FAISS pass and post-filters: inside a single
        document "budget" has no 186k-chunk competition, so annex tables that
        lose the global similarity contest win the local one.
        """
        import numpy as np
        q = np.asarray(self.embedder.encode([query]), dtype="float32")

        def _pass(k: int) -> List[Hit]:
            scores, ids = self.index.search(q, k)
            out: List[Hit] = []
            for score, i in zip(scores[0], ids[0]):
                if i < 0 or i >= len(self.chunks):
                    continue
                c = self.chunks[i]
                if doc_filter is not None and not _doc_match(c.get("doc_id", ""), doc_filter):
                    continue
                out.append(Hit(text=c["text"], doc_id=c.get("doc_id", "?"),
                               score=float(score), page=c.get("page") or None))
                if len(out) >= top_k:
                    break
            return out

        if doc_filter is None:
            return _pass(top_k)
        hits = _pass(min(max(200, top_k * 40), self.index.ntotal))
        if len(hits) < top_k:
            # generic queries may not surface the target doc in any global
            # top-200 — scan the whole index; a flat scan is ~100 ms here
            hits = _pass(self.index.ntotal)
        return hits
