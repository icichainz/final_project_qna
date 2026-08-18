"""Query the index.

Filters FAISS's -1 padding: when the index holds fewer vectors than top_k,
FAISS pads ids with -1, and Python's metadata[-1] silently returns the LAST
chunk — the old codebase fed that unrelated passage to the LLM as context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from gcf_qna import config
from gcf_qna.rag.embed import Embedder
from gcf_qna.rag.lexical import LexicalIndex


def rrf(ranked_lists: List[List[int]], k: int = 60,
        weights: Optional[List[float]] = None) -> Dict[int, float]:
    """Reciprocal Rank Fusion: rank-only merging of incomparable score scales.

    Weights bias whole lists: for identifier queries the dense list is pure
    noise (embeddings are blind to FP codes), so unweighted fusion interleaves
    noise into the top-5. Doubling the lexical weight there restores precision.
    """
    scores: Dict[int, float] = {}
    for li, ranking in enumerate(ranked_lists):
        w = weights[li] if weights else 1.0
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return scores


# Tokens where lexical search is near-authoritative: proposal + board codes.
_IDENTIFIER_RE = re.compile(r"\b(fp\s?\d{2,3}|b\.\d{2}|add\.\d{2})\b")


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
    def __init__(self, index, chunks: List[Dict[str, Any]], embedder: Embedder,
                 index_dir: Optional[Any] = None):
        self.index = index
        self.chunks = chunks
        self.embedder = embedder
        self.hybrid_enabled = False
        if config.HYBRID and index_dir is not None:
            try:
                self.lexical = LexicalIndex(index_dir)
                self.lexical.ensure(chunks)
                self.hybrid_enabled = True
            except Exception as e:   # lexical is an enhancement, never a blocker
                print(f"lexical index unavailable, dense-only: {e}", flush=True)

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
            if not self.hybrid_enabled:
                return _pass(top_k)
            # hybrid: fuse dense and lexical rankings by reciprocal rank
            n = config.CANDIDATES_PER_RETRIEVER
            dense_scores, dense_ids = self.index.search(q, n)
            dense_rank = [i for i in dense_ids[0] if 0 <= i < len(self.chunks)]
            # Identifier queries: restrict the lexical search to the id tokens
            # alone. With the full query, common words ("accredited entity")
            # drag topical wrong-doc chunks into BM25's tail, where dual-list
            # membership out-sums the right doc's lexical-only head.
            from gcf_qna.rag.lexical import tokenize
            id_toks = sorted({t.replace(".", "") for t in tokenize(query)
                              if re.fullmatch(r"fp\d{2,3}|b\.?\d{2}|add\.?\d{2}", t)})
            lex_query = " ".join(id_toks) if id_toks else query
            lex_rank = self.lexical.search(lex_query, n)
            if id_toks and lex_rank:
                # Two-stage for identifier queries: the lexical head IDENTIFIES
                # the document(s); a doc-scoped dense search then ranks chunks
                # semantically WITHIN each. Without this, id-tagged chunks tie
                # in BM25 and the right document's pages are picked ~randomly
                # (observed: FP274's financing question drew pp. 56-187, missed
                # the financing section, answered "no figure stated").
                target_docs: List[str] = []
                for i in lex_rank[:20]:
                    d = self.chunks[i].get("doc_id", "")
                    if d and d not in target_docs:
                        target_docs.append(d)
                    if len(target_docs) >= 3:
                        break
                per = max(3, top_k // max(1, len(target_docs)))
                routed: List[Hit] = []
                seen = set()
                for d in target_docs:
                    for h in self.search(query, per, doc_filter=d):
                        key = (h.doc_id, h.page, h.text[:80])
                        if key not in seen:
                            seen.add(key)
                            routed.append(h)
                if routed:
                    return routed[:top_k]
            weights = [1.0, 2.0] if id_toks else [1.0, 1.0]
            fused = sorted(rrf([dense_rank, lex_rank], config.RRF_K, weights).items(),
                           key=lambda kv: kv[1], reverse=True)[:top_k]
            hits = []
            for idx, score in fused:
                c = self.chunks[idx]
                hits.append(Hit(text=c["text"], doc_id=c.get("doc_id", "?"),
                                score=float(score), page=c.get("page") or None))
            return hits
        hits = _pass(min(max(200, top_k * 40), self.index.ntotal))
        if len(hits) < top_k:
            # generic queries may not surface the target doc in any global
            # top-200 — scan the whole index; a flat scan is ~100 ms here
            hits = _pass(self.index.ntotal)
        return hits
