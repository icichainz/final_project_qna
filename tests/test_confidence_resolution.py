"""Exact-match confidence must require EVERY identifier to resolve.

FTS5 MATCH is OR-joined, so the old `lexical.search(" ".join(id_toks), 1)`
check let one live token vouch for a dead one: 'GCF/B.42/02/Add.99' pinned
conf=1.0 off b42 alone, two-stage routing served other B.42 addenda, and the
app's weak-signal note stayed silent while the model answered about a
document that does not exist.
"""
import hashlib

import faiss
import numpy as np
import pytest

from gcf_qna import config
from gcf_qna.rag.retrieve import Retriever


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
    {"doc_id": "03_gcf-b42-02-add11-funding-proposal-package-fp269", "page": 4,
     "text": "funding decision and board consideration"},
    {"doc_id": "61_gcf-b37-02-add05-funding-proposal-package-fp214", "page": 3,
     "text": "gender action plan budget"},
]


@pytest.fixture
def retriever(tmp_path):
    emb = FakeEmbedder()
    index = faiss.IndexFlatIP(16)
    index.add(emb.encode([c["text"] for c in CHUNKS]))
    r = Retriever(index, CHUNKS, emb, index_dir=tmp_path)
    assert r.hybrid_enabled, "these tests need the lexical index"
    return r


def test_all_identifiers_resolve_keeps_full_confidence(retriever):
    for q in ("budget of FP274", "What does B.42/02/Add.16 cover?",
              "Compare FP274 and FP269"):
        _, conf = retriever.search_with_confidence(q, 2)
        assert conf == 1.0, q


def test_partially_resolvable_identifier_loses_full_confidence(retriever):
    """b42 resolves, add99 does not — the live token must not vouch for it."""
    _, conf = retriever.search_with_confidence(
        "Summarize the funding decision in GCF/B.42/02/Add.99", 2)
    assert conf < 1.0
    _, conf = retriever.search_with_confidence("Compare FP274 and FP999", 2)
    assert conf < 1.0


def test_or_semantics_would_have_passed(retriever):
    """Pins the mechanism: the OR-joined check the fix replaced still hits."""
    assert retriever.lexical.search("add99 b42", 1), "OR match no longer fires"
    assert not retriever.lexical.search("add99", 1)
    assert retriever.lexical.search("b42", 1)


def test_no_identifier_resolves_degrades_confidence(retriever):
    for q in ("What is the budget of FP999?", "details of B.99/02/Add.99"):
        _, conf = retriever.search_with_confidence(q, 2)
        assert conf < 1.0, q


def test_degraded_confidence_trips_weak_signal(retriever):
    """The app flags weak_signal at conf < MIN_DENSE_SCORE; the fake embedder's
    random 16-d vectors sit far below it, so the note fires for the bug query."""
    _, conf = retriever.search_with_confidence(
        "Summarize the funding decision in GCF/B.42/02/Add.99", 2)
    assert conf < config.MIN_DENSE_SCORE


def test_dense_confidence_unaffected_without_identifiers(retriever):
    _, conf = retriever.search_with_confidence("gender action plan budget", 2)
    assert 0.0 < conf <= 1.0
