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


# ==========================================================================
# H10 residual — identifier_tokens over the registry's own FP forms
# ==========================================================================
# tests/test_registry_resolver.py::test_the_consumer_that_keeps_its_own_pattern
# RECORDED this function as the half of H10 that was left unfixed: the forms
# that put a word between 'FP' and the number bound everywhere else (registry
# note, FP-miss guard, doc resolution) and nowhere here, so those turns got no
# per-document BM25 head. That test now documents a miss that no longer
# happens; the coordinator flips it. These are the tests that pin the fix.
from gcf_qna.rag.retrieve import identifier_tokens  # noqa: E402


@pytest.mark.parametrize("form", [
    # the tokenizer already glued these back together
    "FP220", "FP 220", "FP-220", "FP#220", "FP.220",
    # ...and these it could not: a word, or an over-padded number
    "FP no. 220", "FP no 220", "FP number 220", "FP nos. 220",
    "proposal 220", "proposal #220", "proposal no. 220",
    "funding proposal 220", "funding proposal number 220",
    "proposition 220", "proposition n° 220", "proposition numéro 220",
])
def test_every_fp_form_the_registry_binds_now_routes(form):
    assert identifier_tokens(f"What is {form} about?") == ["fp220"]


@pytest.mark.parametrize("q,why", [
    ("What happened to proposal 2020?", "a year, not an id"),
    ("How many proposals from 2020?", "a year behind a plural noun"),
    ("Does it cover 220 countries?", "a count in front of a noun"),
    ("What is 220?", "a bare number identifies nothing"),
    ("proposals from fp2023", "the four-digit trap"),
    ("Look at proposal 2 in the list above", "an enumeration, not an id"),
    ("Compare FPs 12 and 74", "'FPs' is the plural ask: binding one of two "
                              "would hard-scope retrieval to the wrong half"),
    ("Compare proposals 220 and 203", "same, in prose"),
])
def test_the_widening_binds_none_of_the_adversarial_negatives(q, why):
    assert identifier_tokens(q) == [], why


def test_an_addendum_number_is_not_an_fp_number():
    """The negative that separates the two readers: 'Add.220' is a token the
    tokenizer keeps and the FP pattern must never claim."""
    assert identifier_tokens("What does Add.220 contain?") == []
    assert identifier_tokens("What does GCF/B.42/02/Add.16 cover?") == \
        ["add16", "b42"]


def test_the_over_padded_form_binds_to_one_unpadded_token():
    """'FP 0086' joins into a four-digit 'fp0086' the token pattern rejects.
    fp_variants padding-folds downstream, so ONE token is the whole fix."""
    assert identifier_tokens("Give me the details of FP 0086.") == ["fp86"]
    assert identifier_tokens("Give me the details of FP0086.") == ["fp86"]


def test_a_number_the_token_scan_already_found_is_not_emitted_twice():
    """'FP086' must stay ONE identifier: a second token for the same document
    widens _target_docs' routing limit and makes search_with_confidence demand
    two lexical resolutions where the corpus offers one spelling."""
    assert identifier_tokens("What is FP086 about?") == ["fp086"]
    assert identifier_tokens("Compare FP086 and proposal 220.") == \
        ["fp086", "fp220"]


def test_a_broken_registry_leaves_the_token_scan_alone(monkeypatch):
    """Registry-as-an-enhancement, the rule _registry_docs already follows:
    a registry that cannot be read costs the new forms, never the old ones."""
    from gcf_qna.rag import registry

    class Boom:
        def findall(self, _):
            raise RuntimeError("registry unreadable")

    monkeypatch.setattr(registry, "FP_RE", Boom())
    assert identifier_tokens("What is FP220 about?") == ["fp220"]
    assert identifier_tokens("Tell me about proposal 220.") == []


def test_a_widened_form_reaches_the_per_document_head(retriever):
    """The point of the widening: identifier routing, not just the note. The
    prose form must land on the document the compact form lands on."""
    prose = retriever.search("Tell me about funding proposal number 274.", 2)
    compact = retriever.search("Tell me about FP274.", 2)
    assert prose and all("fp274" in h.doc_id for h in prose)
    assert [h.doc_id for h in prose] == [h.doc_id for h in compact]


def test_a_widened_form_resolves_confidence_the_same_way(retriever):
    _, prose = retriever.search_with_confidence("budget of proposal 274", 2)
    _, compact = retriever.search_with_confidence("budget of FP274", 2)
    assert prose == compact == 1.0


# ==========================================================================
# Phase 2 — probe_pages, the doc-scoped supplementary probe
# ==========================================================================
DOC = "25_gcf-b40-02-add13-funding-proposal-package-fp251"
OTHER = "61_gcf-b37-02-add05-funding-proposal-package-fp214"
PROBE_CHUNKS = [
    {"doc_id": DOC, "page": 5, "section_path": "A SUMMARY > A.10 Grant",
     "text": "cover page: USD 30 million requested from the Fund"},
    {"doc_id": DOC, "page": 6, "section_path": "A SUMMARY > A.10 Grant",
     "text": "the grant component is USD 40 million"},
    {"doc_id": DOC, "page": 6, "section_path": "A SUMMARY > A.10 Grant",
     "text": "the grant component is USD 40 million"},          # exact reprint
    {"doc_id": DOC, "page": 6, "section_path": "A SUMMARY > A.10 Grant",
     "text": "co-financing rows for page six"},
    {"doc_id": DOC, "page": 42,
     "section_path": "C PROJECT DETAILS > C.2 Financing by component",
     "text": "component one: sustainable land management"},
    {"doc_id": DOC, "page": 7, "text": "page seven prose, no section path"},
    {"doc_id": OTHER, "page": 6, "text": "another document's page six"},
]


@pytest.fixture
def probe(tmp_path):
    emb = FakeEmbedder()
    index = faiss.IndexFlatIP(16)
    index.add(emb.encode([c["text"] for c in PROBE_CHUNKS]))
    return Retriever(index, PROBE_CHUNKS, emb, index_dir=tmp_path)


def test_probe_pages_returns_the_asked_for_pages(probe):
    hits = probe.probe_pages(DOC, [5, 6], k=4)
    assert {h.page for h in hits} == {5, 6}
    assert all(h.doc_id == DOC for h in hits)


def test_probe_pages_serves_every_asked_page_before_any_page_twice(probe):
    """The spread IS the request: a two-page conflict whose first page holds
    three chunks must not spend all the slots on that page."""
    hits = probe.probe_pages(DOC, [5, 6], k=2)
    assert [h.page for h in hits] == [5, 6]


def test_probe_pages_spread_survives_page_diversity_off(probe, monkeypatch):
    """PAGE_DIVERSITY tunes the SEARCH pipeline. Turning it off must not be
    able to collapse a page probe onto one page."""
    monkeypatch.setenv("PAGE_DIVERSITY", "0")
    assert [h.page for h in probe.probe_pages(DOC, [5, 6], k=2)] == [5, 6]


def test_probe_pages_never_degrades_to_another_document(probe):
    """search's doc_filter degrades to an unscoped search when it matches
    nothing — right for a primary query, wrong for a supplement: pages from
    another document would be cited as this document's."""
    assert probe.probe_pages("99_nonexistent", [5, 6], k=4) == []
    assert probe.search("page six", 2, doc_filter="99_nonexistent"), \
        "the contrast: search DOES degrade, deliberately"


def test_probe_pages_invents_no_page_and_asks_for_no_ranking(probe):
    assert probe.probe_pages(DOC, [999], k=4) == []
    assert probe.probe_pages(DOC, [], k=4) == []
    assert probe.probe_pages(DOC, [5, 6], k=0) == []


def test_probe_pages_only_ever_returns_the_pages_it_was_given(probe):
    hits = probe.probe_pages(DOC, [42], k=4)
    assert [h.page for h in hits] == [42]
    assert "sustainable land management" in hits[0].text


def test_probe_pages_collapses_a_reprinted_passage(probe):
    """The same dedup key search uses: one page's identical passage twice
    cannot spend two of the probe's slots."""
    hits = probe.probe_pages(DOC, [6], k=4)
    assert len(hits) == 2
    assert len({h.text for h in hits}) == 2


def test_probe_pages_query_orders_within_a_page_and_never_across_pages(probe):
    """`query` decides WHICH chunk of a requested page comes first. It cannot
    add a page, drop a page, or reach another document."""
    plain = probe.probe_pages(DOC, [5, 6], k=3)
    ranked = probe.probe_pages(DOC, [5, 6], k=3,
                               query="co-financing rows for page six")
    assert {h.page for h in plain} == {h.page for h in ranked} == {5, 6}
    first_six = lambda hs: next(h.text for h in hs if h.page == 6)
    assert first_six(plain) == "the grant component is USD 40 million"
    assert first_six(ranked) == "co-financing rows for page six"


def test_probe_pages_scores_are_cosines_only_when_a_query_asked_for_them(probe):
    assert all(h.score == 0.0 for h in probe.probe_pages(DOC, [5, 6], k=3))
    assert any(h.score > 0.0 for h in
               probe.probe_pages(DOC, [5, 6], k=3, query="grant component"))


def test_probe_pages_takes_a_section_hint(probe):
    """The registry's conflict candidates carry a section as well as a page
    ('rule:A.7', 'C.1'), and Phase 5c's presence checks read section_path."""
    hits = probe.probe_pages(DOC, [], k=4, sections=["C.2"])
    assert [h.page for h in hits] == [42]
    assert probe.probe_pages(DOC, [], k=4, sections=["C"]) == hits
    assert probe.probe_pages(DOC, [], k=4, sections=["C.20"]) == []


def test_a_section_hint_is_inert_where_the_build_stored_no_sections(probe):
    """Every chunk of data/index/default has section_path None. A section
    hint there must return nothing rather than something wrong — which is
    exactly what the sectioned-index A/B is for."""
    assert probe.probe_pages(OTHER, [], k=4, sections=["C.2"]) == []
    assert [h.page for h in probe.probe_pages(OTHER, [6], k=4)] == [6]


def test_probe_pages_unions_pages_and_sections(probe):
    hits = probe.probe_pages(DOC, [5], k=4, sections=["C.2"])
    assert sorted(h.page for h in hits) == [5, 42]


def test_probe_pages_hits_carry_source_text_and_their_own_provenance(probe):
    """A supplementary hit is a Hit like any other: the answer prompt and
    ground.py have to be able to cite it."""
    h = probe.probe_pages(DOC, [42], k=1)[0]
    assert h.doc_id == DOC and h.page == 42
    assert h.section_path == "C PROJECT DETAILS > C.2 Financing by component"
    assert h.chunk_index is not None
    assert h.text == PROBE_CHUNKS[h.chunk_index]["text"]


def test_probe_pages_reads_a_page_hint_in_any_spelling(probe):
    """Hints arrive as ints from registry.conflicts(), as strings from a note
    line, and as whatever a caller typed."""
    assert [h.page for h in probe.probe_pages(DOC, ["42"], k=2)] == [42]
    assert [h.page for h in probe.probe_pages(DOC, [42, None, "x"], k=2)] == [42]


def test_single_digit_proposals_bind_for_the_first_time(retriever):
    """A side effect worth pinning: 'fp\\d{2,3}' needs two digits, so FP1-FP9
    were never routed either — measured on the live index, 'Tell me about
    FP1.' did not return that document at k=10 at all, and now it ranks 1.
    The prose arm stays two-digit on purpose ('proposal 2' is an
    enumeration), so those numbers bind by marker only."""
    assert identifier_tokens("Tell me about FP1.") == ["fp1"]
    assert identifier_tokens("Tell me about FP 7.") == ["fp7"]
    assert identifier_tokens("Look at proposal 2 in the list above") == []
