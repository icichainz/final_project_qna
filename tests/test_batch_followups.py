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
    # ... and an EMPTY v2: the year note now prefers the v2 canonical figure,
    # so a v1-only fixture that left _cache_v2 alone would quietly read the
    # real data/registry_v2.json and print the corpus's money for these
    # stems. Empty means "v2 knows nothing here" -> the v1 fallback, which is
    # what these cases are pinning.
    monkeypatch.setattr(registry, "_cache_v2", {})
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


# --- A2: the year note's money, and the sum it must refuse (F11/P4) --------
#
# Measured: the note printed v1's `gcf_financing` raw. For FP153 that string
# is "28,654 million USD" — a print whose mantissa and scale word cannot both
# be true — and asked to total the 2020 note the model answered $29.0B
# against a truth near $1.36B (21x); an unprompted 2020-vs-2021 comparison
# came out backwards and the verifier passed it. Two changes answer that: the
# figure now comes from the v2 canonical fact (which marks a print it could
# not parse), and the note forbids the summation outright.

def _cand(raw, value, page=5, section="A.8", status="canonical"):
    return {"raw": raw, "value": value, "currency": "USD", "unit": None,
            "page": page, "section": section, "status": status}


V1_MONEY = {
    "124_gcf-b27-02-add11": {"fp": 151, "year": 2020, "board": 27,
                             "gcf_financing": "18.5 M USD"},
    # the OCR-garbled print, exactly as schema 1 stored it
    "126_gcf-b27-02-add14": {"fp": 153, "year": 2020, "board": 27,
                             "gcf_financing": "28,654 million USD"},
    # schema 1 never read this one; v2 did
    "125_gcf-b27-02-add10-rev01": {"fp": 150, "year": 2020, "board": 27},
}


@pytest.fixture
def year_money(monkeypatch):
    monkeypatch.setattr(registry, "_cache", V1_MONEY)
    monkeypatch.setattr(registry, "_cache_v2", {
        "124_gcf-b27-02-add11": {"fp": 151, "facts": {
            # v2 disagrees with the v1 text: v2 is the one that gets printed
            "gcf_funding_requested": [_cand("18,500,000 USD", 18_500_000.0)]}},
        "126_gcf-b27-02-add14": {"fp": 153, "facts": {
            "gcf_funding_requested": [_cand("28,654 million USD", None)]}},
        "125_gcf-b27-02-add10-rev01": {"fp": 150, "facts": {
            "gcf_funding_requested": [_cand("256.48 million USD", 256_480_000.0)]}},
    })
    yield


def test_year_note_money_is_the_v2_canonical_print(year_money):
    from gcf_qna.app.chainlit_app import _year_assist
    _, note = _year_assist("Which proposals were approved in 2020?", [])
    assert "FP151 (18,500,000 USD GCF)" in note      # v2 print, not v1's text
    assert "18.5 M USD" not in note
    assert "FP150 (256.48 million USD GCF)" in note  # v2 fills a v1 gap
    # printed as printed: no float reformat, no computed number anywhere
    assert "18500000" not in note and "1.85e" not in note


def test_a_print_v2_could_not_parse_is_quoted_and_flagged(year_money):
    """FP153's single unusable print is the whole of the 21x error."""
    from gcf_qna.app.chainlit_app import _year_assist
    _, note = _year_assist("Which proposals were approved in 2020?", [])
    assert 'FP153 ("28,654 million USD" GCF, unit as printed is ambiguous)' in note
    # the same words registry._money_bit uses, so one figure reads one way
    assert "unit as printed is ambiguous" in note


def test_year_note_falls_back_to_v1_when_v2_has_no_canonical(monkeypatch):
    """'Stated somewhere but not in a template section' is not a canonical
    fact: the note keeps the v1 string rather than promoting a candidate."""
    monkeypatch.setattr(registry, "_cache", V1_MONEY)
    monkeypatch.setattr(registry, "_cache_v2", {
        "124_gcf-b27-02-add11": {"fp": 151, "facts": {
            "gcf_funding_requested": [
                _cand("99 M USD", 99_000_000.0, status="supporting")]}}})
    from gcf_qna.app.chainlit_app import _year_assist
    _, note = _year_assist("Which proposals were approved in 2020?", [])
    assert "FP151 (18.5 M USD GCF)" in note
    assert "99 M USD" not in note


def test_year_note_survives_an_unreadable_v2(monkeypatch, fake_registry):
    """The year note is the answer to a year question; a broken v2 file may
    cost it the better figure, never the note."""
    monkeypatch.setattr(registry, "load_v2",
                        lambda: (_ for _ in ()).throw(ValueError("bad json")))
    from gcf_qna.app.chainlit_app import _year_assist
    _, note = _year_assist("Which proposals were approved in 2020?", [])
    assert "FP151 (18.5 M USD GCF)" in note          # v1 text, unharmed


@pytest.mark.parametrize("q", [
    "Which proposals were approved in 2020?",        # detailed, prints money
    "Which proposals were approved since 2018?",     # wide span, counts only
])
def test_the_year_note_forbids_the_sum(q, fake_registry):
    from gcf_qna.app.chainlit_app import _year_assist, _NO_SUM_RULE
    _, note = _year_assist(q, [])
    assert _NO_SUM_RULE in note
    assert "MUST NOT be summed" in note and "refusing the sum" in note


def test_the_no_sum_rule_publishes_no_page_and_no_document():
    """The note-page scope readers credit a cited page to a document named on
    the SAME line; the year note is one line, so a sentence that parsed as a
    doc id or a '(p.N,' pointer would hand the model a page it never saw."""
    import re

    from gcf_qna.app import chainlit_app as app
    from gcf_qna.rag import verify
    assert app._note_pages([app._NO_SUM_RULE]) == set()
    assert verify.note_page_scopes(app._NO_SUM_RULE) == []
    assert not re.search(r"[\[(][0-9]{1,3}_", app._NO_SUM_RULE)
    assert not re.search(r"\(p\.\d", app._NO_SUM_RULE)


# --- A3: 'B.<n>' is a board code OR a template heading (H5/P6) -------------

@pytest.mark.parametrize("q,why", [
    ("What does section B.3 of FP172 say?", "the recorded P6 question"),
    ("Which section B.3 commitments does FP172 list?", "'section' anywhere"),
    ("Que dit la rubrique B.3 de FP172 ?", "the French heading word"),
    ("What does § B.3 cover?", "the section sign"),
    ("What does B.2(a) of FP172 report?", "a paragraph letter: never a board"),
    ("What is in B.3?", "unframed and ambiguous - no definitive denial"),
    ("Which board approved the B.2(a) figure for FP172?",
     "a board word does not turn a lettered heading into a meeting"),
])
def test_low_numbers_do_not_get_the_out_of_range_note(q, why):
    from gcf_qna.app.chainlit_app import _board_range_note
    assert _board_range_note(q) is None, why


@pytest.mark.parametrize("q", [
    "What was approved at B.3?",
    "Which proposals did the board approve at B.3?",
    "Qu'est-ce qui a été approuvé lors de la réunion B.3 ?",
])
def test_an_explicitly_claimed_low_board_is_still_denied(q):
    """B.1-B.10 are real GCF meetings (2012-2015) and genuinely outside this
    corpus, so a question that frames one AS a meeting keeps the definitive
    note - the guard removes the false positives, not the true ones."""
    from gcf_qna.app.chainlit_app import _board_range_note
    note = _board_range_note(q)
    assert note is not None and "B.3 is not in this corpus" in note


def test_out_of_range_boards_are_unchanged_above_the_section_range():
    """No heading is numbered above 10, so nothing above it is ambiguous -
    including next to the word 'section'."""
    from gcf_qna.app.chainlit_app import _board_range_note
    for q in ("Which funding proposals were approved at B.44?",
              "What does section B.44 say?"):
        assert "B.44 is not in this corpus" in _board_range_note(q)
    assert _board_range_note("What was decided at B.30?") is None
    assert _board_range_note("What does document GCF/B.42/02/Add.16 contain?") is None


def test_several_codes_read_independently():
    from gcf_qna.app.chainlit_app import _board_range_note
    assert "B.9" not in _board_range_note("What about B.9 and B.44?")
    both = _board_range_note("Which proposals were approved at B.9 and B.44?")
    assert "B.9, B.44 are not in this corpus" in both     # numeric order, plural


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
