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


class Retriever:
    def __init__(self, index, chunks: List[Dict[str, Any]], embedder: Embedder):
        self.index = index
        self.chunks = chunks
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> List[Hit]:
        import numpy as np
        q = self.embedder.encode([query])
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), top_k)
        hits: List[Hit] = []
        for score, i in zip(scores[0], ids[0]):
            if i < 0 or i >= len(self.chunks):
                continue
            c = self.chunks[i]
            hits.append(Hit(text=c["text"], doc_id=c.get("doc_id", "?"),
                            score=float(score), page=c.get("page") or None))
        return hits
