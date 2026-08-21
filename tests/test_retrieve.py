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


def test_confidence_is_read_off_the_query_not_the_original(retriever):
    """The weak-signal guard is a statement about the DOCUMENT match, and the
    rewrite is the text that carries the identifier. A vague original must not
    be able to talk a resolved identifier down (or an unresolved one up)."""
    _, with_original = retriever.search_with_confidence(
        "budget of FP274", 2, None, "how much was it again?")
    _, alone = retriever.search_with_confidence("budget of FP274", 2)
    assert with_original == alone == 1.0
    _, unknown = retriever.search_with_confidence(
        "budget of FP999", 2, None, "budget of FP274")
    assert unknown < 1.0, "the original must not vouch for an unresolved id"


def test_a_single_probe_is_the_plain_scoped_call(retriever):
    """_scoped_probes is the two-stage split's second stage. With one probe it
    must BE _scoped — that identity is what makes every caller who passes no
    original byte-identical to before."""
    import numpy as np
    qv = np.asarray(retriever.embedder.encode(["total financing"]),
                    dtype="float32")
    doc = "02_gcf-b42-02-add16-funding-proposal-package-fp274"
    assert retriever._scoped_probes([qv], doc, 5) == retriever._scoped(qv, doc, 5)


def test_the_original_never_reaches_an_unrouted_hybrid_query(retriever):
    """No doc_filter and no identifier: nothing has chosen a document yet, so
    the second probe has no document to rank inside and must not vote."""
    plain = retriever.search("gender action plan budget", 2)
    probed = retriever.search("gender action plan budget", 2,
                              original="total financing information")
    assert [(h.doc_id, h.page, h.score) for h in plain] == \
           [(h.doc_id, h.page, h.score) for h in probed]
