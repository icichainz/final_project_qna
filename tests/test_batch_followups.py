"""Follow-up batch from the gpt-5.2 production battery:

A. Year/board aggregates answered from the registry (retrieval never surfaces
   all of a year's proposals, so excerpt-scoped notes made the model refuse).
B. FP zero-padding: 'FP86' must reach the corpus's one zero-padded doc (fp086).
C. Citation-bracket parsing (pages belong to the nearest preceding doc id) and
   explicit in-message language requests beating wordlist statistics.
"""
import hashlib

import faiss
import numpy as np
import pytest

from gcf_qna.rag import registry
from gcf_qna.rag.lexical import fp_variants
from gcf_qna.rag.retrieve import Hit, Retriever, _doc_match


# --- B: fp zero-padding ----------------------------------------------------

def test_fp_variants():
    assert fp_variants("fp86") == ["fp086", "fp86"]
    assert fp_variants("fp086") == ["fp086", "fp86"]
    assert fp_variants("fp274") == ["fp274"]
    assert fp_variants("b42") == ["b42"]      # non-FP tokens pass through


def test_doc_match_zero_padding():
    doc = "189_12-status-approved-fps-adding-host-countries-respect-fp086-gcf-b37-07"
    assert _doc_match(doc, "fp86")
    assert _doc_match(doc, "fp086")
    assert not _doc_match("02_gcf-b42-02-add16-funding-proposal-package-fp274", "fp86")


class FakeEmbedder:
    def encode(self, texts, **kw):
        out = []
        for t in texts:
            rng = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16))
            v = rng.standard_normal(16).astype("float32")
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


PADDED_CHUNKS = [
    {"doc_id": "189_12-status-approved-fps-fp086-gcf-b37-07", "page": 1,
     "text": "green cities facility host countries FP086"},
    {"doc_id": "02_gcf-b42-02-add16-funding-proposal-package-fp274", "page": 8,
     "text": "total financing information for the project"},
]


@pytest.fixture
def padded_retriever(tmp_path):
    emb = FakeEmbedder()
    index = faiss.IndexFlatIP(16)
    index.add(emb.encode([c["text"] for c in PADDED_CHUNKS]))
    return Retriever(index, PADDED_CHUNKS, emb, index_dir=tmp_path)


def test_unpadded_fp_resolves_and_routes(padded_retriever):
    hits, conf = padded_retriever.search_with_confidence("What is FP86 about?", 3)
    assert conf == 1.0                       # variant group resolves
    assert any("fp086" in h.doc_id for h in hits)


def test_unresolvable_fp_still_degrades(padded_retriever):
    _, conf = padded_retriever.search_with_confidence("What is FP999 about?", 3)
    assert conf < 1.0


# --- A: registry-powered year/board notes ----------------------------------

FAKE_REGISTRY = {
    "124_gcf-b27-02-add11": {"fp": 151, "year": 2020, "board": 27,
                             "gcf_financing": "18.5 M USD"},
    "123_gcf-b27-02-add12": {"fp": 152, "year": 2020, "board": 27,
                             "gcf_financing": "150 M USD"},
    "02_gcf-b42-02-add16-funding-proposal-package-fp274":
        {"fp": 274, "year": 2025, "board": 42},
}


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(registry, "_cache", FAKE_REGISTRY)
    yield


def test_year_note_lists_registry_rows(fake_registry):
    from gcf_qna.app.chainlit_app import _year_assist
    hits, note = _year_assist("Which proposals were approved in 2020?", [])
    assert "FP151 (18.5 M USD GCF)" in note and "FP152 (150 M USD GCF)" in note
    assert "2 proposals" in note
    assert "authoritative" in note           # the model may answer from it


def test_year_note_registry_unavailable(monkeypatch):
    monkeypatch.setattr(registry, "load",
                        lambda: (_ for _ in ()).throw(FileNotFoundError()))
    from gcf_qna.app.chainlit_app import _year_assist
    _, note = _year_assist("proposals from 2020?", [])
    assert "state this limit" in note        # never a definitive claim
    assert "no registered proposals" not in note


def test_board_range_note():
    from gcf_qna.app.chainlit_app import _board_range_note
    note = _board_range_note("Does the corpus contain proposals from B.44?")
    assert "B.44 is not in this corpus" in note
    assert "B.11 (2015) through B.43 (2025)" in note
    assert _board_range_note("What was decided at B.42?") is None
    assert _board_range_note("no board mentioned") is None


# --- C: citation-bracket attribution + explicit language -------------------

HITS = [Hit(text="", doc_id="102_gcf-b30-02-add05", score=1.0, page=5),
        Hit(text="", doc_id="103_gcf-b30-03-add04", score=1.0, page=6)]


def test_chained_bracket_pages_attributed_correctly():
    from gcf_qna.app.chainlit_app import _invalid_citations
    ans = "FP173 asks more [102_gcf-b30-02-add05, p. 5; 103_gcf-b30-03-add04, p. 6]."
    assert _invalid_citations(ans, HITS) == []   # was: false-flagged 102 p.6


def test_invented_page_still_flagged():
    from gcf_qna.app.chainlit_app import _invalid_citations
    ans = "See [102_gcf-b30-02-add05, p. 5; 103_gcf-b30-03-add04, p. 99]."
    bad = _invalid_citations(ans, HITS)
    assert len(bad) == 1 and bad[0].startswith("103_") and "p.99" in bad[0]


def test_unretrieved_doc_pages_skipped():
    from gcf_qna.app.chainlit_app import _invalid_citations
    ans = "Registry says [999_some-other-doc, p. 12]."
    assert _invalid_citations(ans, HITS) == []


def test_explicit_language_request_wins():
    from gcf_qna.app.chainlit_app import _detect_lang
    assert _detect_lang("Now back to the first one — which country is it in, "
                        "and présente ta réponse en français.") == "French"
    assert _detect_lang("Réponds en anglais s'il te plaît, quel est le budget ?") == "English"
    assert _detect_lang("Which country is FP172 in?") == "English"
    assert _detect_lang("Quel est le financement total ?") == "French"
