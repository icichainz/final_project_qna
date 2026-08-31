"""The section probe: the ask side of `probe_pages(sections=...)`.

The fetch side has existed since the conflict-probe pass and sat dormant
because nothing wired an ask into it; the rebuilt default index then gave
every schema-2 chunk a `section_path`. The measured defect this cures
(release-19 arm 1, `l1x-sec-c2-fp126`, page_rate 0.0): 'What does section
C.2 of FP126 say?' shares almost no vocabulary with the C.2 heading or its
financing table, so similarity search returns cover-page chunks and page 40
never surfaces. Asking for the section by its printed id is the cure, under
the conflict probe's honesty rules: the fence decides WHEN, `probe_pages`
decides WHAT, and every failure is an empty supplement, never a search.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcf_qna.app import chainlit_app as app  # noqa: E402

GOLD_Q = "What does section C.2 of FP126 say?"
GOLD_DOC = "149_gcf-b25-02-add03"


# ------------------------------------------------------------- the fence ---
def test_the_gold_question_asks_for_the_gold_document():
    assert app._section_probe_asks(GOLD_Q) == [(GOLD_DOC, ["C.2"])]


def test_the_same_ask_in_french_resolves_identically():
    """'section' is the same word in both corpus languages."""
    q = "Que dit la section C.2 de la FP126 ?"
    assert app._section_probe_asks(q) == [(GOLD_DOC, ["C.2"])]


@pytest.mark.parametrize("q,reason", [
    ("What does C.2 of FP126 say?",
     "a bare code never fires - the word 'section' is the trigger"),
    ("What does section C.2 say?",
     "no document to scope to"),
    ("Compare section C.2 of FP126 and FP127",
     "two documents is the comparison path's territory"),
    ("What was approved at B.42?",
     "board codes belong to the board paths"),
    ("What is the total GCF funding requested in FP151?",
     "an ordinary identifier question"),
])
def test_non_section_questions_do_not_fire(q, reason):
    assert app._section_probe_asks(q) == [], reason


def test_a_section_the_document_lacks_is_an_ask_that_fetches_nothing():
    """'C.20' fires as an ask (the fence cannot know the corpus) but
    `_section_hit` guarantees 'C.20' never matches a 'C.2' path — the probe
    comes back empty instead of degrading into a search."""
    asks = app._section_probe_asks("What does section C.20 of FP126 cover?")
    assert asks == [(GOLD_DOC, ["C.20"])]


def test_two_codes_are_kept_in_question_order_and_capped():
    q = ("In FP126, what do section B.1 and section C.2 and section D.3 "
         "each cover?")
    assert app._section_probe_asks(q) == [(GOLD_DOC, ["B.1", "C.2"])]


# ------------------------------------------------------------- the probe ---
class _Hit:
    def __init__(self, doc, page, text="x", score=0.0):
        self.doc_id, self.page, self.text, self.score = doc, page, text, score


class _FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def probe_pages(self, doc_id, pages=(), k=4, query=None, sections=()):
        self.calls.append({"doc": doc_id, "pages": list(pages),
                           "sections": list(sections), "k": k, "query": query})
        return self.hits[:k]


def test_the_probe_resolves_the_id_to_pages_then_fetches_the_pages_whole():
    """Stage 1 asks by id alone (the fence's codes, no pages, no query);
    stage 2 re-asks by the pages stage 1 found, so chunks the tracker filed
    under a promoted table-header heading — the release-20 residue — ride
    along with the section they are printed in."""
    r = _FakeRetriever([_Hit(GOLD_DOC, 40)])
    got = app._section_probe(r, GOLD_Q, [], GOLD_Q)
    assert [h.page for h in got] == [40]
    first, second = r.calls
    assert first["sections"] == ["C.2"] and first["pages"] == []
    assert first["query"] is None, "page discovery is reading-order, not cosine"
    assert second["pages"] == [40] and second["query"] == GOLD_Q


def test_a_sectionless_index_yields_no_second_ask_and_no_probe():
    r = _FakeRetriever([])
    assert app._section_probe(r, GOLD_Q, [], GOLD_Q) == []
    assert len(r.calls) == 1, "no pages found -> nothing to fetch whole"


def test_pages_this_turn_already_holds_are_not_fetched_twice():
    r = _FakeRetriever([_Hit(GOLD_DOC, 40), _Hit(GOLD_DOC, 41)])
    got = app._section_probe(r, GOLD_Q, [_Hit(GOLD_DOC, 40)], GOLD_Q)
    assert [h.page for h in got] == [41]


def test_the_probe_is_capped_at_the_probe_budget():
    r = _FakeRetriever([_Hit(GOLD_DOC, p) for p in range(40, 50)])
    got = app._section_probe(r, GOLD_Q, [], GOLD_Q)
    assert len(got) <= app._MAX_PROBE_HITS


def test_no_retriever_and_a_raising_retriever_both_mean_no_probe():
    assert app._section_probe(None, GOLD_Q, [], None) == []

    class Boom:
        def probe_pages(self, *a, **k):
            raise RuntimeError("index gone")
    assert app._section_probe(Boom(), GOLD_Q, [], None) == []


def test_a_non_section_question_never_reaches_the_retriever():
    r = _FakeRetriever([_Hit(GOLD_DOC, 40)])
    assert app._section_probe(r, "What is FP126 about?", [], None) == []
    assert r.calls == []


# ----------------------------------------------------------- the context ---
def test_a_section_hit_is_labelled_not_scored():
    ranked = _Hit(GOLD_DOC, 2, "ranked", 0.44)
    sec = _Hit(GOLD_DOC, 40, "the table", 0.27)
    block = app._context_block([sec, ranked], (), (sec,))
    assert "(the asked-for section — fetched by section id, not ranked)" in block
    assert "(score 0.44)" in block
    assert "(score 0.27)" not in block


def test_the_conflict_label_is_untouched():
    conf = _Hit(GOLD_DOC, 5, "conflict page")
    block = app._context_block([conf], (conf,), ())
    assert "(registry conflict page — fetched by page, not ranked)" in block


# ------------------------------------------------------------- the wiring ---
def test_app_and_harness_wire_the_same_probe_at_the_same_point():
    """Parity by source: both callers run the app's own `_section_probe`
    after the conflict probe and hand BOTH supplements to `_context_block`,
    so a release record's excerpts are the excerpts production ships."""
    app_src = (ROOT / "src" / "gcf_qna" / "app" / "chainlit_app.py").read_text()
    ev_src = (ROOT / "scripts" / "eval_answers.py").read_text()
    assert "_section_probe)(\n        retriever, message.content, hits" \
        in app_src
    assert "app._section_probe(" in ev_src
    for src in (app_src, ev_src):
        assert "_context_block(hits, probe_hits, section_hits)" in src


# -------------------------------------------- the recorded failing shape ---
def test_the_gold_section_is_fetchable_from_the_live_index_by_id():
    """The release-19 miss, replayed against the real index without the
    embedder: a page-less, query-less probe must return the C.2 chunks in
    reading order — the financing-by-component table the gold case pins to
    page 40. Skipped when the index is absent (a fresh clone)."""
    idx = ROOT / "data" / "index" / "default"
    if not (idx / "index.faiss").exists():
        pytest.skip("default index not built")
    from gcf_qna.rag.index import load_index
    from gcf_qna.rag.retrieve import Retriever
    index, chunks, _ = load_index(idx)
    r = Retriever(index, chunks, None, index_dir=idx)
    got = r.probe_pages(GOLD_DOC, sections=["C.2"], k=4)
    assert got and all(h.doc_id == GOLD_DOC for h in got)
    assert 40 in {h.page for h in got}
    assert any("Financing by Component" in h.text for h in got)
