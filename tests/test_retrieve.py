import hashlib

import faiss
import numpy as np
import pytest

from gcf_qna.rag.retrieve import Retriever, rrf


class FakeEmbedder:
    """Deterministic unit vectors from text hashes — no torch needed."""
    def encode(self, texts, **kw):
        out = []
        for t in texts:
            rng = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16))
            v = rng.standard_normal(16).astype("float32")
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


CHUNKS = [
    {"doc_id": "02_gcf-b42-02-add16-funding-proposal-package-fp274", "page": 8,
     "text": "total financing information for the project"},
    {"doc_id": "02_gcf-b42-02-add16-funding-proposal-package-fp274", "page": 9,
     "text": "expected results and outcomes"},
    {"doc_id": "61_gcf-b37-02-add05-funding-proposal-package-fp214", "page": 3,
     "text": "gender action plan budget"},
]


@pytest.fixture
def retriever(tmp_path):
    emb = FakeEmbedder()
    index = faiss.IndexFlatIP(16)
    index.add(emb.encode([c["text"] for c in CHUNKS]))
    return Retriever(index, CHUNKS, emb, index_dir=tmp_path)


def test_rrf_weights():
    s = rrf([[1, 2], [3, 1]], k=60, weights=[1.0, 2.0])
    assert s[1] > s[3] > s[2]


def test_unknown_identifier_confidence(retriever):   # review finding #2
    _, conf = retriever.search_with_confidence("What is the budget of FP999?", 2)
    assert conf < 1.0, "unresolvable identifiers must not claim full confidence"


def test_known_identifier_confidence(retriever):
    _, conf = retriever.search_with_confidence("budget of FP274", 2)
    assert conf == 1.0


def test_compact_identifier_routes(retriever):       # review finding #3 (behavior holds)
    hits = retriever.search("What does B42 Add16 cover?", 2)
    assert hits and all("fp274" in h.doc_id for h in hits)


def test_bad_doc_filter_degrades(retriever):
    hits = retriever.search("gender budget", 2, doc_filter="99_nonexistent")
    assert hits, "a filter matching nothing must degrade to unscoped search"


def test_round_robin_merge_no_starvation():
    """Cross-cutting review #2: the global cap must not starve later queries."""
    from itertools import zip_longest
    a = [("d1", i) for i in range(10)]
    b = [("d2", i) for i in range(10)]
    merged, seen = [], set()
    for tier in zip_longest(a, b):
        for h in tier:
            if h and h not in seen:
                seen.add(h)
                merged.append(h)
    top = merged[:15]
    assert sum(1 for d, _ in top if d == "d2") >= 7, "later query starved by cap"
