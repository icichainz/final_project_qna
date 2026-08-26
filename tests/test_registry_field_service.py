"""Arity-one field service, non-money conflicts, and extraction honesty.

Campaign Phase 1, the three mechanism items that share one line of output:
`registry._fmt`. The measured gap was an ASYMMETRY, not an absence — the same
fact, asked about one document, was reachable only through retrieval, while
asked about two it came back as a page- and section-cited matrix cell:

    "Compare the implementation periods of FP220 and FP203"
        -> FP220 | implementation_period | 12 years (p.5, A.11) | stated
    "What is the implementation period of FP220?"
        -> the registry line, which printed title, entity, countries and two
           money figures, and no implementation period at all

The matrix served 15 fields; the line served 5. What is pinned here:

A. the service. A single-identifier turn appends the fields the question asks
   for, from registry v2's own candidates, with v2's own '(p.N, SECTION)'.
B. its three guards: only asked fields, absent fields stay silent, and a field
   the document contradicts itself on is left to `_conflict_lines`.
C. the ARITY PAIR — the same field asked at one identifier and at two comes
   back with the same value and the same citation, which is the exit gate of
   the phase and the reason `fields_for` is reused rather than reimplemented.
D. the caps and their markers: a cut list says it was cut and carries its true
   count; a clipped value carries the clip.
E. conflicts beyond money: the four non-money conflicts the corpus holds now
   warn, in text formatting, while every money line stays byte-identical.
F. extraction honesty: `suspect` and `llm_fallback` documents say so, in a
   marker that publishes NO page and NO document to `_note_pages` /
   `note_page_scopes` — an inert marker cannot become an invented citation.
"""
import json
import re
from pathlib import Path

import pytest

from gcf_qna.rag import planner, registry
from gcf_qna.rag import verify as V

ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# synthetic corpus (the registry._cache / _cache_v2 pattern of test_planner)
# --------------------------------------------------------------------------

FP300 = "300_gcf-b40-02-add01"
FP301 = "301_gcf-b40-02-add02"
FP302 = "302_gcf-b40-02-add03"


def _text(raw, page, section, status="canonical"):
    return {"raw": raw, "value": None, "currency": None, "unit": None,
            "page": page, "section": section, "status": status}


def _num(raw, value, page, section, status="canonical", unit=None, cur=None):
    return {"raw": raw, "value": value, "currency": cur, "unit": unit,
            "page": page, "section": section, "status": status}


REG1 = {
    FP300: {"fp": 300, "title": "Coastal Resilience Facility",
            "accredited_entity": "IFAD", "countries": ["Kenya"],
            "board": 40, "year": 2024},
    FP301: {"fp": 301, "title": "Contradictory Document",
            "accredited_entity": "UNDP", "countries": ["Sudan"],
            "board": 40, "year": 2024},
    FP302: {"fp": 302, "title": "Sparse Document",
            "accredited_entity": "KfW", "countries": ["Fiji"],
            "board": 40, "year": 2024},
}

LONG_EE = ("The project activities will be implemented through the implementing "
           "partner and other responsible parties, namely the Department of "
           "Forests and Soil Management Authority, the Ministry of Energy and "
           "Environment and its sub-organizations, and the Department of "
           "Forestry and Wildlife Conservation")

REG2 = {
    FP300: {"fp": 300, "facts": {
        "title": [_text("Coastal Resilience Facility", 3, "A.1.1")],
        "accredited_entity": [_text("IFAD", 3, "A.1.5")],
        "gcf_funding_requested": [_num("50,000,000 USD", 5e7, 5, "A.8", cur="USD")],
        "total_financing": [_num("200,000,000 USD", 2e8, 5, "A.7", cur="USD")],
        "implementation_period": [_num("12 years", 12.0, 5, "A.11", unit="years")],
        "executing_entity": [_text("Ministry of Water", 8, "A.20"),
                             _text("Ministry of Water", 3, "A.1.6", "supporting")],
        "national_designated_authority": [_text("Office of Climate Change", 3,
                                                "rule:A.1.4")],
        "co_financing": [_num("150,000,000 USD", 1.5e8, 13, "C.1(b)", cur="USD")],
        "instruments": [_text("Grant", 8, "A.10 Grant"),
                        _text("Loan", 8, "A.10 Loan", "supporting")],
        "financial_instruments": [_num("$5 million USD", 5e6, 8, "A.10 Grant",
                                       unit="million", cur="USD"),
                                  _num("$46 million USD", 4.6e7, 8, "A.10 Loan",
                                       "supporting", "million", "USD")],
        "ess_category": [_text("B", 6, "A.13")],
    }, "coverage": {"llm_fallback": False}},
    FP301: {"fp": 301, "facts": {
        "implementation_period": [_num("5 years, 0 months", 5.0, 5, "A.11",
                                       unit="years"),
                                  _num("25 years", 25.0, 5, "A.11", "conflicting",
                                       "years")],
        "total_financing": [_num("41,185,114 USD", 41185114.0, 5, "A.7", cur="USD"),
                            _num("25,645,114 USD", 25645114.0, 5, "A.7",
                                 "conflicting", cur="USD")],
    }, "coverage": {"llm_fallback": False}},
    FP302: {"fp": 302, "facts": {
        "executing_entity": [_text(LONG_EE, 6, "A.20")],
    }, "coverage": {"llm_fallback": True, "suspect": "gcf>total"}},
}


@pytest.fixture
def reg(monkeypatch):
    monkeypatch.setattr(registry, "_cache", REG1)
    monkeypatch.setattr(registry, "_cache_v2", REG2)
    return registry


_TRAILER_RE = re.compile(r"\s*\[[^\]]+, cover pages\]\s*$")


def _seg(note, label):
    """The one ';'-separated segment of a note line headed by `label`.

    The '[stem, cover pages]' trailer belongs to the LINE, not to the last
    field on it, so it is dropped here rather than asserted around.
    """
    for line in (note or "").splitlines():
        for part in line.split(";"):
            part = _TRAILER_RE.sub("", part.strip()).strip()
            if part.lower().startswith(label.lower()):
                return part
    return None


# --------------------------------------------------------------------------
# A. the service
# --------------------------------------------------------------------------

def test_a_single_identifier_serves_the_field_the_question_asks_for(reg):
    note = registry.registry_note("What is the implementation period of FP300?")
    assert _seg(note, "implementation period") == \
        "implementation period: 12 years (p.5, A.11)"


def test_the_served_value_and_pointer_come_from_registry_v2(reg):
    """Not a re-reading of anything: the segment is the canonical candidate."""
    canon = registry.canonical(FP300, "implementation_period")
    note = registry.registry_note("How long is FP300's implementation period?")
    assert canon["raw"] in note
    assert f"(p.{canon['page']}, {canon['section']})" in note


def test_the_three_newly_served_fields_reach_the_line(reg):
    ee = registry.registry_note("Who is the executing entity of FP300?")
    assert _seg(ee, "executing entity") == \
        "executing entity: Ministry of Water (p.8, A.20)"
    nda = registry.registry_note("What is the national designated authority for FP300?")
    assert _seg(nda, "national designated authority") == \
        "national designated authority: Office of Climate Change (p.3, A.1.4)"
    fi = registry.registry_note("Which financial instruments does FP300 use?")
    assert "financial instruments: $5 million USD (p.8, A.10 Grant); " \
           "financial instruments: $46 million USD (p.8, A.10 Loan)" in fi


def test_the_section_is_what_says_which_instrument_an_amount_belongs_to(reg):
    """A.10 prices each instrument on ONE page; the row is the section."""
    note = registry.registry_note("What are the financial instruments of FP300?")
    assert "$5 million USD (p.8, A.10 Grant)" in note
    assert "$46 million USD (p.8, A.10 Loan)" in note


def test_a_served_list_repeats_its_label_so_every_value_is_readable(reg):
    """NOT `_list_bit`'s count-first shape. `verify._field_lines` reads the
    value of a field as the first amount AFTER its label, so a leading '(2)'
    publishes the COUNT as the field's value — measured: a claim correctly
    stating the loan's $46 million came back as a contradiction of '2'."""
    from gcf_qna.rag import verify as _V
    note = registry.registry_note("What are the financial instruments of FP300?")
    line = note.splitlines()[0]
    rx = dict(_V._FIELD_RES)["financial_instruments"]
    read = [_V._value_after(seg, at)[0].value
            for seg, at in _V._field_lines(line, rx)]
    assert read == [5_000_000.0, 46_000_000.0]     # the values, not the count


def test_a_money_field_the_line_never_printed_is_now_servable(reg):
    note = registry.registry_note("How much co-financing does FP300 report?")
    assert _seg(note, "co-financing") == "co-financing: 150,000,000 USD (p.13, C.1(b))"


# --------------------------------------------------------------------------
# B. the guards
# --------------------------------------------------------------------------

def test_only_the_asked_field_is_served(reg):
    """No kitchen-sink line: this document states seven servable fields."""
    note = registry.registry_note("What is the implementation period of FP300?")
    assert "implementation period:" in note
    for absent in ("executing entity:", "national designated authority:",
                   "co-financing:", "financial instruments", "ESS category:"):
        assert absent not in note


def test_a_question_with_no_field_word_serves_nothing(reg):
    """`fields_for`'s default set is the four the line already prints."""
    note = registry.registry_note("Tell me about FP300.")
    assert "implementation period:" not in note
    assert note == "Registry — " + registry._fmt({"doc_id": FP300, **REG1[FP300]})


def test_an_absent_field_appends_nothing_rather_than_asserting_absence(reg):
    """A note that wrote 'not stated' from an EMPTY CANDIDATE LIST would be
    asserting a fact the registry does not hold — nobody read FP302 for its
    ESS category and found nothing; the extractor simply has no candidate.

    STILL TRUE, and now it is the negative twin of section G rather than the
    whole rule. Section G serves the absence where the registry holds it AS
    DATA — a ratified `meta.confirmed_absence` with the pages that were read.
    The two are one rule stated once: the line says only what the store knows,
    and 'not printed' is something a store can know."""
    note = registry.registry_note("What is the ESS category of FP302?")
    assert "ESS category" not in note
    assert "not stated" not in note
    assert not registry._absences(FP302)


def test_a_contradicted_field_is_left_to_the_conflict_line(reg):
    """Printing the canonical figure as the value would be the silent choice
    between two disagreeing prints that the conflict machinery refuses."""
    note = registry.registry_note("What is the implementation period of FP301?")
    assert "implementation period: 5 years" not in note
    assert "implementation_period is printed as 5 years, 0 months (p.5, A.11); " \
           "also as 25 years (p.5, A.11)" in note


def test_the_five_fields_the_line_already_prints_are_never_served_twice(reg):
    note = registry.registry_note(
        "What is the title, accredited entity, countries, GCF funding requested "
        "and total financing of FP300?")
    for once in ("accredited entity:", "countries (", "GCF funding requested:",
                 "total financing:"):
        assert note.count(once) == 1


def test_two_identifiers_serve_no_fields_because_the_matrix_does(reg):
    """The exact complement of `planner.detect`: one mechanism per turn."""
    note = registry.registry_note(
        "Compare the implementation periods of FP300 and FP301.")
    assert "implementation period: 12 years" not in note
    matrix = planner.plan_and_render(
        "Compare the implementation periods of FP300 and FP301.")
    assert "FP300 | implementation_period | 12 years (p.5, A.11) | stated" in matrix


def test_one_document_under_two_names_is_served_once(reg):
    """'FP300 and GCF/B.40/02/Add.01' is ONE document — one arity, and the
    fact belongs on one of the two lines, not on both."""
    note = registry.registry_note(
        "What is the implementation period of FP300, i.e. GCF/B.40/02/Add.01?")
    assert note.count("implementation period: 12 years") == 1


def test_the_apposition_that_names_a_document_is_not_a_field_ask(reg):
    """Measured on the gold set: 'FP152, the ... equity proposal' asks for the
    accredited entity, and 'FP259, the Pacific tuna adaptation programme' for
    the total financing. `fields_for` is calibrated for a matrix, where a spare
    column costs a line; here a spare segment is a fact asserted about a
    document nobody asked about."""
    note = registry.registry_note(
        "Which accredited entity is behind FP300, the coastal equity facility?")
    assert "instruments" not in note
    # the ask itself still lands when it is the question rather than the label
    asked = registry.registry_note("Which instruments does FP300 use?")
    assert "instruments: Grant (p.8, A.10 Grant); instruments: Loan " \
           "(p.8, A.10 Loan)" in asked


def test_the_apposition_guard_only_eats_a_span_that_follows_an_identifier(reg):
    assert registry._drop_document_apposition(
        "What is the implementation period of FP300?") == \
        "What is the implementation period of FP300?"
    assert registry._drop_document_apposition(
        "Which entity runs FP300, the coastal facility?") == \
        "Which entity runs FP300?"
    # no identifier in front: the span is the question, and it stays
    keep = "Which proposals, the ones approved in 2020, name UNDP?"
    assert registry._drop_document_apposition(keep) == keep


def test_a_missing_document_serves_nothing_and_still_says_so(reg):
    note = registry.registry_note("What is the implementation period of FP999?")
    assert "NOT FOUND in the 273-document corpus registry" in note
    assert "implementation period:" not in note


# --------------------------------------------------------------------------
# C. the arity pair — the phase's exit gate
# --------------------------------------------------------------------------

def test_the_arity_pair_returns_the_same_value_and_the_same_citation(reg):
    single = registry.registry_note("What is the implementation period of FP300?")
    seg = _seg(single, "implementation period")
    pair = planner.plan_and_render(
        "Compare the implementation periods of FP300 and FP301.")
    row = next(l for l in pair.splitlines()
               if l.startswith("FP300 | implementation_period"))
    assert "12 years" in seg and "12 years" in row
    assert "(p.5, A.11)" in seg and "(p.5, A.11)" in row


@pytest.mark.parametrize("field,label", [
    ("implementation_period", "implementation period"),
    ("co_financing", "co-financing"),
    ("executing_entity", "executing entity"),
])
def test_every_served_field_cites_the_cell_the_matrix_would_cite(reg, field, label):
    """One value and one citation per fact, whichever arity asked for it."""
    cell = planner._registry_cell(
        planner.DocRef(label="FP300", kind="fp", doc_id=FP300, fp=300), field)
    note = registry.registry_note(f"FP300: what is its {label}?")
    seg = _seg(note, label)
    assert seg, f"{field} was not served"
    assert registry._where({"page": cell.page, "section": cell.section}) in seg


# --------------------------------------------------------------------------
# D. the caps, and the markers that announce them
# --------------------------------------------------------------------------

def test_a_cut_list_says_it_was_cut_and_carries_the_true_count(reg, monkeypatch):
    """The cap is set above the longest list the corpus holds, so it is a
    backstop rather than a working cap — and when it bites it says so."""
    names = ["Grant", "Loan", "Equity", "Guarantee", "Results-based", "Other",
             "Second loss"]
    assert len(names) > registry._MAX_FIELD_VALUES
    many = dict(REG2[FP300])
    many["facts"] = dict(many["facts"], instruments=[
        _text(n, 8, f"A.10 {n}", "canonical" if i == 0 else "supporting")
        for i, n in enumerate(names)])
    monkeypatch.setattr(registry, "_cache_v2", {**REG2, FP300: many})
    note = registry.registry_note("Which instruments does FP300 use?")
    assert note.count("instruments: ") == registry._MAX_FIELD_VALUES + 1
    assert (f"instruments: not every value is listed — "
            f"{registry._MAX_FIELD_VALUES} of {len(names)} shown above, "
            "list truncated") in note
    assert "Second loss" not in note


def test_the_cap_never_bites_on_the_corpus_as_it_stands(reg):
    note = registry.registry_note("Which instruments does FP300 use?")
    assert "list truncated" not in note
    assert note.count("instruments: ") == 2


def test_a_paragraph_length_value_is_clipped_with_the_marker(reg):
    seg = _seg(registry.registry_note("Who is the executing entity of FP302?"),
               "executing entity")
    assert "…" in seg
    assert len(seg) < len(LONG_EE)
    assert seg.endswith("(p.6, A.20)")


def test_a_value_that_is_the_template_talking_is_not_served(reg, monkeypatch):
    """`verify._field_lines` skips a WHOLE LINE carrying an instruction phrase,
    and a registry line is one line per document — so serving 'Indicate the
    number of years...' as a value would take that document's money segments
    off the checker with it. 14 such candidates over 11 documents."""
    from gcf_qna.rag import verify as _V
    patched = dict(REG2[FP300])
    patched["facts"] = dict(patched["facts"], implementation_period=[
        _text("Indicate the number of years the project is expected to last",
              6, "A.11")])
    monkeypatch.setattr(registry, "_cache_v2", {**REG2, FP300: patched})
    note = registry.registry_note("What is the implementation period of FP300?")
    assert "implementation period:" not in note
    assert not _V._INSTRUCTION_RE.search(note)


def test_a_second_print_is_served_when_the_first_is_the_template_talking(
        reg, monkeypatch):
    patched = dict(REG2[FP300])
    patched["facts"] = dict(patched["facts"], implementation_period=[
        _text("Indicate the number of years the project will last", 6, "A.11"),
        _num("12 years", 12.0, 7, "A.11", "supporting", unit="years")])
    monkeypatch.setattr(registry, "_cache_v2", {**REG2, FP300: patched})
    note = registry.registry_note("What is the implementation period of FP300?")
    assert _seg(note, "implementation period") == \
        "implementation period: 12 years (p.7, A.11)"


def test_the_line_is_one_line_however_many_fields_are_served(reg):
    note = registry.registry_note(
        "For FP300, give the implementation period, the executing entity, the "
        "co-financing and the ESS category.")
    first = note.splitlines()[0]
    for label in ("implementation period:", "executing entity:", "co-financing:",
                  "ESS category:"):
        assert label in first


# --------------------------------------------------------------------------
# E. conflicts beyond money
# --------------------------------------------------------------------------

def test_a_non_money_conflict_warns_and_says_values_not_figures(reg):
    (money, text) = registry._conflict_lines({"doc_id": FP301, **REG1[FP301]})
    assert money.endswith("— report both figures with their pages.")
    assert "implementation_period is printed as 5 years, 0 months (p.5, A.11); " \
           "also as 25 years (p.5, A.11) — report both values with their pages." \
           in text


def test_money_conflict_lines_keep_money_formatting(reg):
    (money, _) = registry._conflict_lines({"doc_id": FP301, **REG1[FP301]})
    assert money == (
        f"Registry — CONFLICT in this document ({FP301}): total_financing is "
        "printed as 41,185,114 USD (p.5, A.7); also as 25,645,114 USD (p.5, A.7) "
        "— report both figures with their pages.")


def test_a_conflicting_text_print_is_clipped_like_a_served_one(reg, monkeypatch):
    patched = dict(REG2[FP302])
    patched["facts"] = dict(patched["facts"], executing_entity=[
        _text(LONG_EE, 6, "A.20"),
        _text(LONG_EE.replace("Ministry of Energy", "Ministry of Finance"), 9,
              "A.20", "conflicting")])
    monkeypatch.setattr(registry, "_cache_v2", {**REG2, FP302: patched})
    (line,) = registry._conflict_lines({"doc_id": FP302, **REG1[FP302]})
    assert "…" in line
    assert LONG_EE not in line


def test_the_conflict_cap_still_names_every_field_it_held_back(reg, monkeypatch):
    patched = dict(REG2[FP301])
    patched["facts"] = dict(
        patched["facts"],
        gcf_funding_requested=[_num("1 USD", 1.0, 5, "A.8", cur="USD"),
                               _num("2 USD", 2.0, 6, "A.8", "conflicting", cur="USD")],
        co_financing=[_num("3 USD", 3.0, 7, "C.1(b)", cur="USD"),
                      _num("4 USD", 4.0, 8, "C.1(b)", "conflicting", cur="USD")])
    monkeypatch.setattr(registry, "_cache_v2", {**REG2, FP301: patched})
    lines = registry._conflict_lines({"doc_id": FP301, **REG1[FP301]})
    assert len(lines) == registry._MAX_CONFLICT_LINES + 1
    assert lines[-1].endswith(
        "2 further fields (co_financing, implementation_period) also print "
        "disagreeing prints, not listed above — list truncated.")


def test_a_money_only_cap_line_is_unchanged(reg, monkeypatch):
    """FP65 is the one document in the corpus that reaches the cap today, and
    its held line must not gain a word."""
    patched = dict(REG2[FP301])
    patched["facts"] = dict(
        {k: v for k, v in patched["facts"].items() if k != "implementation_period"},
        gcf_funding_requested=[_num("1 USD", 1.0, 5, "A.8", cur="USD"),
                               _num("2 USD", 2.0, 6, "A.8", "conflicting", cur="USD")],
        co_financing=[_num("3 USD", 3.0, 7, "C.1(b)", cur="USD"),
                      _num("4 USD", 4.0, 8, "C.1(b)", "conflicting", cur="USD")])
    monkeypatch.setattr(registry, "_cache_v2", {**REG2, FP301: patched})
    lines = registry._conflict_lines({"doc_id": FP301, **REG1[FP301]})
    assert lines[-1] == (
        f"Registry — CONFLICT in this document ({FP301}): 1 further field "
        "(co_financing) also prints disagreeing figures, not listed above — "
        "list truncated.")


def test_the_conflict_order_is_derived_so_a_new_field_warns_without_an_edit(reg):
    order = registry._conflict_field_order(
        {"beneficiaries_direct": [], "gcf_funding_requested": [], "some_new_field": []})
    assert order == ["gcf_funding_requested", "beneficiaries_direct", "some_new_field"]


# --------------------------------------------------------------------------
# F. extraction honesty
# --------------------------------------------------------------------------

def test_a_flagged_document_says_so_on_its_own_line(reg):
    note = registry.registry_note("Tell me about FP302.")
    assert "extraction flagged (llm_fallback):" in note
    assert "extraction flagged (gcf>total):" in note
    assert "verify" in note


def test_a_clean_document_says_nothing(reg):
    assert "extraction flagged" not in registry.registry_note("Tell me about FP300.")


def test_the_flag_does_not_need_a_question(reg):
    """`chainlit_app._extend_registry_note` prints a line for a document this
    TURN resolved to and has no question to pass: a flagged extraction is
    flagged on every line it appears on."""
    assert "extraction flagged" in registry._fmt({"doc_id": FP302, **REG1[FP302]})


def test_the_marker_publishes_no_page_and_no_document(reg):
    """An inert marker cannot become the invented citation `_meta_page` and the
    page-less listings are built to prevent. Pinned against BOTH readers, which
    are the same two regexes byte for byte."""
    from gcf_qna.app import chainlit_app as app
    line = "Registry — " + registry._fmt({"doc_id": FP302, **REG1[FP302]})
    flag = next(p for p in line.split(";") if "extraction flagged" in p)
    assert V.note_page_scopes(flag) == []
    assert app._note_pages([flag]) == set()


def test_the_served_pointer_IS_creditable(reg):
    """The other direction, and the point of serving the field at all: the page
    beside a served value is a page the app's citation gate credits."""
    from gcf_qna.app import chainlit_app as app
    note = registry.registry_note("What is the implementation period of FP300?")
    assert (FP300, 5) in app._note_pages([note])
    assert (V.note_scope_doc(FP300), 5) in V.note_page_scopes(note.splitlines()[0])


# --------------------------------------------------------------------------
# the real corpus — the cases the plan names, on the data it names them for
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    if not (ROOT / "data" / "registry.json").exists():
        pytest.skip("built corpus not present")
    registry.load()
    registry.load_v2()
    return registry


def test_gold_case_l2x_arity_implperiod_single_is_now_served(real):
    """The gold case the phase pins: it passed via retrieval, and the note it
    reached the model with carried no implementation period at all."""
    note = real.registry_note("What is the implementation period of FP220?")
    assert "implementation period: 12 years (p.5, A.11)" in note
    doc = "55_gcf-b37-02-add11-funding-proposal-package-fp220"
    from gcf_qna.app import chainlit_app as app
    assert (doc, 5) in app._note_pages([note])


def test_the_real_arity_pair_agrees(real):
    single = real.registry_note("What is the implementation period of FP220?")
    pair = planner.plan_and_render(
        "Compare the implementation periods of FP220 and FP203.")
    row = next(l for l in pair.splitlines()
               if l.startswith("FP220 | implementation_period"))
    assert "12 years" in single and "12 years" in row
    canon = real.canonical("55_gcf-b37-02-add11-funding-proposal-package-fp220",
                           "implementation_period")
    cite = real._where(canon)
    assert cite in _seg(single, "implementation period") and cite in row


@pytest.mark.parametrize("fp,field,label", [
    (202, "beneficiaries_direct", "direct beneficiaries"),
])
def test_the_non_money_conflicts_the_corpus_holds_now_warn(real, fp, field, label):
    """Enumerated from data/registry_v2.json: conflicting non-money candidates
    that no mechanism looked at until `_conflict_lines` widened past money.

    The census was four, in three documents. Phase 3's adjudication (ratified
    by the owner 2026-08-26) refuted three of them as extraction artifacts and
    the builder corrects them from data/registry_corrections.json: FP139's
    '25 years' is the A.12 lifespan under a shifted label, and FP240's two
    'outcomes' were financing figures bled in under A.5/A.6 (money against
    tonnes, not two readings of one outcome). FP202's stands — 81,551 direct
    beneficiaries under A.6 against 1,251,769 under A.7, both on page 6 — so
    it is the case that keeps the mechanism honest here, and it is stable
    across the rebuild that lands the corrections.
    """
    row = real.by_fp(fp)
    lines = real._conflict_lines(row)
    assert any(f"{field} is printed as" in l and l.endswith(
        "report both values with their pages.") for l in lines)


def test_a_field_the_corpus_contradicts_is_not_also_served(real):
    """The served line must not pick one of two disagreeing prints. FP202 asks
    for direct beneficiaries: the conflict warns and the value is withheld,
    while the field's uncontradicted twin is served as usual."""
    note = real.registry_note("How many direct beneficiaries does FP202 have?")
    assert "direct beneficiaries: 81,551" not in note
    assert "beneficiaries_direct is printed as" in note
    assert "indirect beneficiaries: 147,039" in note


#: The two documents whose ``llm_fallback`` flag the builder's reuse path
#: dropped, and which the SHIPPED registry therefore publishes with no
#: extraction caveat on them. Both are two-page board notices: the
#: deterministic parser finds no financing fact, the model is called, and
#: nothing it returns survives verification, so the previous build flagged
#: them with zero llm candidates to carry — and the reuse path used to set the
#: flag only in the branch that had a candidate.
#:
#: The defect is FIXED in ``scripts/build_registry_v2.carry_forward_llm``
#: (the flag is now carried from the reuse seed independent of candidate
#: survival, pinned by
#: ``test_registry_v2.py::test_the_reuse_path_keeps_the_flag_of_a_call_that_returned_nothing``),
#: and a rebuild on the fixed builder restores both: measured on a scratch
#: build seeded from the pre-loss registry, llm_fallback 17 -> 19, nothing
#: else moved. The shipped ``data/registry_v2.json`` is a build product and is
#: not hand-edited, so the restoration lands with the next registry rebuild
#: and this pin moves 19 -> 21 THEN, loudly, in one traced step.
FLAG_PENDING_A_REBUILD = ("193_gcf-b22-10-add01-rev01", "196_gcf-b19-22-add21-rev01")


def test_the_flagged_documents_are_the_ones_the_registry_flags(real):
    """THE CENSUS, TRACED. Every count here has a reason on the record:

      2  suspect (`gcf>total`), disjoint from the fallback set —
         `201_gcf-b19-22-add16-rev01` (22.50 million USD requested against
         $2 million total, the long-standing one) and
         `27_gcf-b40-02-add11-funding-proposal-package-fp249`, which is NEW
         and is the system working: ratified correction C129 reads FP249's
         A.8 as 'USD 29.25 million' where the store had a single-digit
         misread, and 29.25 > the document's own A.7 total of USD 29 million.
         The detector is reporting a real anomaly the correction surfaced; it
         is not a regression in the detector.

     17  llm_fallback, i.e. documents whose values came from the fallback
         extraction and which say so on their line. NINETEEN calls were made;
         two of the nineteen are `FLAG_PENDING_A_REBUILD` above, whose flag
         the reuse defect dropped out of the shipped file and whose
         restoration is a rebuild away.

     19  flagged in total, and `registry._extraction_flags` publishes exactly
         that set — which is the property this test exists for: the caveat a
         reader sees on a line is the caveat the store recorded, with no
         second opinion in between.
    """
    v2 = json.loads((ROOT / "data" / "registry_v2.json").read_text())["documents"]
    suspect = {s for s, r in v2.items() if (r.get("coverage") or {}).get("suspect")}
    fallback = {s for s, r in v2.items()
                if (r.get("coverage") or {}).get("llm_fallback")}
    flagged = suspect | fallback
    said = {s for s in real.load() if real._extraction_flags(s)}
    assert said == flagged
    assert not suspect & fallback, sorted(suspect & fallback)
    assert len(suspect) == 2, sorted(suspect)
    assert len(fallback) == 17, sorted(fallback)
    assert len(flagged) == 19
    # named, not counted: the two the reuse defect unflagged are still
    # unflagged in the shipped build, and nothing else is missing from it
    assert set(FLAG_PENDING_A_REBUILD).isdisjoint(flagged)


# --------------------------------------------------------------------------
# G. confirmed absences: the fact that a document does NOT print a field
# --------------------------------------------------------------------------
# Section B's second guard ("an absent field appends NOTHING ... recording a
# CONFIRMED absence is Phase 3's data work") has been half-superseded. Phase 3
# did the work: `data/registry_absences.json` ratifies 51 absences, 48 of them
# published as `documents[doc].meta.confirmed_absence`, each with the pages
# that were read. Nothing in `src/` had ever opened the key — asked how much
# GCF funding FP273 requests, the line printed title, entity, countries, board
# and a fallback marker and said NOTHING about financing. Silence is not an
# answer to an ask, and it is indistinguishable from a gap in retrieval.
#
# The guard survives where it belongs: silence is still the answer when the
# store records NO ratified absence (`test_an_absent_field_appends_nothing...`
# above is that test, and it is unchanged). What changes is the case where the
# registry holds the absence AS DATA.

FP310 = "310_gcf-b41-02-add01"      # ratified absence, nothing printed for it
FP311 = "311_gcf-b41-02-add02"      # ratified absence AND a print: the tension

ABS_REG1 = {
    FP310: {"fp": 310, "title": "Results-Based Payment Programme",
            "accredited_entity": "FMO", "countries": ["Papua New Guinea"],
            "board": 41, "year": 2025},
    FP311: {"fp": 311, "title": "Corrected Document", "accredited_entity": "UNDP",
            "countries": ["Brazil"], "gcf_financing": "USD 96,452,228",
            "board": 41, "year": 2025},
}
_ABS = {"pages_checked": [1, 16], "evidence": "no A.7/A.8/B.2 block exists",
        "group": "REDD+ RBP financing", "ratified": "owner, 2026-08-26"}
ABS_REG2 = {
    FP310: {"fp": 310, "facts": {}, "coverage": {"llm_fallback": False},
            "meta": {"confirmed_absence": {"gcf_funding_requested": dict(_ABS),
                                           "total_financing": dict(_ABS)}}},
    FP311: {"fp": 311, "facts": {}, "coverage": {"llm_fallback": False},
            "meta": {"confirmed_absence": {"gcf_funding_requested": dict(_ABS)}}},
}


@pytest.fixture
def absent(monkeypatch):
    monkeypatch.setattr(registry, "_cache", ABS_REG1)
    monkeypatch.setattr(registry, "_cache_v2", ABS_REG2)
    return registry


def test_a_ratified_absence_is_served_as_a_fact(absent):
    note = absent.registry_note("How much GCF funding does FP310 request?")
    assert _seg(note, "GCF funding requested") == (
        "GCF funding requested: NOT PRINTED ANYWHERE IN THIS DOCUMENT — a "
        "CONFIRMED ABSENCE the registry has ratified, not a gap in retrieval. "
        "The registry read pages 1-16 of this document. State that the GCF "
        "funding requested is not printed anywhere in this document, and never "
        "carry one over from another document")


def test_only_the_asked_absence_is_served(absent):
    """FP310 is ratified absent on BOTH money fields; the ask names one. Same
    discipline as the field service: a kitchen-sink line is noise, not
    evidence, and this line is what the prompt calls authoritative."""
    note = absent.registry_note("How much GCF funding does FP310 request?")
    assert note.count("NOT PRINTED ANYWHERE IN THIS DOCUMENT") == 1
    assert _seg(note, "total financing") is None
    both = absent.registry_note(
        "What are the GCF funding requested and total financing of FP310?")
    assert both.count("NOT PRINTED ANYWHERE IN THIS DOCUMENT") == 2


def test_a_question_naming_no_field_serves_no_absence(absent):
    assert "CONFIRMED ABSENCE" not in absent.registry_note("What is FP310?")


def test_an_absence_is_never_published_over_a_print(absent):
    """THE TENSION, and the ruling this pass takes on it. `175_gcf-b22-10-add02`
    (FP100) records gcf_funding_requested as a fact-layer absence while its top
    level carries the ratified correction 'USD 96,452,228'. Both statements are
    true of different layers; a note that made both would be on both sides of
    the question the turn actually asked. `data/registry_absences.json` says an
    absence is never published over a print the build holds, and the gold case
    `w2a-rbp-fp100-gcf` asserts the corrected figure — so the print wins and
    the absence stays silent."""
    note = absent.registry_note("How much GCF financing does FP311 request?")
    assert "GCF financing (as printed): USD 96,452,228" in note
    assert "CONFIRMED ABSENCE" not in note


def test_two_identifiers_serve_no_absence_because_the_matrix_does(absent):
    """Same arity rule as the field service: an absence is a statement about
    one document, and a two-document turn is the matrix's, with its own
    'missing' vocabulary."""
    note = absent.registry_note(
        "Compare the GCF funding requested of FP310 and FP311.")
    assert "CONFIRMED ABSENCE" not in note


def test_the_absence_publishes_no_page_and_no_value(absent):
    """Two readers, two hazards.

    `_note_pages` / `note_page_scopes` would turn a '(p.N)' here into a citable
    scope on a line that ends '[stem, cover pages]' — and a page that was READ
    AND FOUND EMPTY is the last page an answer should be invited to cite. So
    the span that was read prints as 'pages 1-16'.

    `verify._field_conflict` reads the first amount after a field label as that
    field's value; a page number sitting there is a contradiction waiting to be
    manufactured out of '1'. So no digit follows the label inside
    `_value_after`'s 80-character window.
    """
    from gcf_qna.app import chainlit_app as app
    note = absent.registry_note("How much GCF funding does FP310 request?")
    assert app._note_pages([note]) == set()
    assert V.note_page_scopes(note) == []
    assert not V._INSTRUCTION_RE.search(note)
    rx = dict(V._FIELD_RES)["gcf_financing"]
    segs = list(V._field_lines(note, rx))
    assert segs and all(V._value_after(seg, at) == [] for seg, at in segs)


def test_a_malformed_or_missing_absence_record_leaves_the_line_alone(absent,
                                                                    monkeypatch):
    """Same never-break contract as `_v2_facts` and `_v2_meta`: the key is
    additive and optional, and absent, partial or the wrong type must leave the
    line byte-identical to the one the store alone produced."""
    good = absent.registry_note("How much GCF funding does FP310 request?")
    for broken in ({"meta": None}, {"meta": {"confirmed_absence": []}},
                   {"meta": {"confirmed_absence": {"gcf_funding_requested": 7}}},
                   {}):
        monkeypatch.setattr(registry, "_cache_v2",
                            {**ABS_REG2, FP310: {"fp": 310, "facts": {},
                                                 "coverage": {}, **broken}})
        note = absent.registry_note("How much GCF funding does FP310 request?")
        assert "CONFIRMED ABSENCE" not in note
        assert note.startswith('Registry — FP310: "Results-Based Payment')
    monkeypatch.setattr(registry, "_cache_v2", ABS_REG2)
    assert absent.registry_note(
        "How much GCF funding does FP310 request?") == good


def test_a_pages_checked_the_readers_could_not_credit_prints_nothing(absent,
                                                                     monkeypatch):
    """`_meta_page`'s discipline, applied to the span: a page outside 1..999,
    a bool, or a non-list prints no span at all rather than a nonsense one."""
    for bad in ([], [0], [1000], True, "1-16", [None, "x"]):
        monkeypatch.setattr(registry, "_cache_v2", {**ABS_REG2, FP310: {
            "fp": 310, "facts": {}, "coverage": {},
            "meta": {"confirmed_absence": {
                "gcf_funding_requested": {**_ABS, "pages_checked": bad}}}}})
        note = absent.registry_note("How much GCF funding does FP310 request?")
        assert "CONFIRMED ABSENCE" in note and "The registry read" not in note


def test_one_page_checked_reads_as_one_page(absent, monkeypatch):
    monkeypatch.setattr(registry, "_cache_v2", {**ABS_REG2, FP310: {
        "fp": 310, "facts": {}, "coverage": {},
        "meta": {"confirmed_absence": {
            "gcf_funding_requested": {**_ABS, "pages_checked": [4]}}}}})
    assert "The registry read page 4 of this document." in \
        absent.registry_note("How much GCF funding does FP310 request?")


def test_the_served_absence_is_a_negative_the_verifier_supports(absent):
    """THE POINT, and the reason the wording is what it is.

    Ruling 3 — support an uncited claim because a computed note confirmed the
    absence — was implemented and DELETED: it passed the rider 'and the total
    co-financing is USD 18.5 million' on the strength of a negative beside it.
    So a negative verifies here the way every other claim does: it cites the
    document, the citation resolves to this line at document scope, and the
    matchers read the line. Both halves are pinned — the cited absence-as-fact
    verifies, the uncited one with a figure attached does not.
    """
    note = absent.registry_note("How much GCF funding does FP310 request?")
    ev = V.build_evidence([], note)
    cited = (f"The GCF funding requested is not printed anywhere in this "
             f"document — the registry has ratified this as a confirmed "
             f"absence for **FP310** “Results-Based Payment Programme”, whose "
             f"accredited entity is FMO. [{FP310}, cover pages]")
    verdicts = V.classify_deterministic(V.extract_claims(cited), ev)
    assert verdicts and [v.status for v in verdicts] == \
        [V.SUPPORTED] * len(verdicts)
    assert all(v.claim.citations for v in verdicts)

    rider = ("The GCF funding requested is not printed anywhere in this "
             "document, and the total co-financing is USD 18.5 million.")
    assert [v.status for v in
            V.classify_deterministic(V.extract_claims(rider), ev)] == \
        [V.UNSUPPORTED]


def test_a_figure_invented_for_the_absent_field_still_fails(absent):
    """The failure the note exists to stop, and the note does not excuse it:
    the line says the figure is not printed, so a figure cited to the line is
    not in the evidence."""
    note = absent.registry_note("How much GCF funding does FP310 request?")
    ev = V.build_evidence([], note)
    guess = (f"**FP310** requests USD 18,500,000 of GCF financing. "
             f"[{FP310}, cover pages]")
    assert [v.status for v in
            V.classify_deterministic(V.extract_claims(guess), ev)] == \
        [V.UNSUPPORTED]


# --- the real corpus: 51 ratified absences, 48 published, one reader --------

FP273 = "03_gcf-b42-02-add15-funding-proposal-package-fp273"


def test_the_reader_reaches_every_published_absence(real):
    """`_absences` is the ONLY reader in `src/`; before this pass there was
    none, and only `scripts/build_registry_v2.py` wrote the key. What it reads
    is what the build published, document for document and field for field."""
    v2 = json.loads((ROOT / "data" / "registry_v2.json").read_text())["documents"]
    published = {s: set((r.get("meta") or {}).get("confirmed_absence") or {})
                 for s, r in v2.items()
                 if (r.get("meta") or {}).get("confirmed_absence")}
    assert {s: set(real._absences(s)) for s in published} == published
    # Re-pinned 2026-08-26: 48 -> 46 field-absences over the same 16 documents.
    # Both losses are the serving wave's two RBP add-candidate rows doing
    # exactly what they were ratified to do — "stop a ratified absence and a
    # ratified top-level print from both being live at once":
    #   * C73 FP100 [175_gcf-b22-10-add02] adds 'GCF RBP: 96,452,228', so
    #     gcf_funding_requested is no longer a confirmed absence there;
    #   * C74 FP142 [133_gcf-b27-02-add02] adds 'Total Budget | $82,000,000',
    #     same field, same effect.
    # The DOCUMENT count is unchanged at 16 — each of the two keeps its
    # total_financing absence — and the field vocabulary below is unchanged.
    assert len(published) == 16 and sum(len(v) for v in published.values()) == 46
    assert set().union(*published.values()) == {
        "title", "countries", "accredited_entity",
        "gcf_funding_requested", "total_financing"}


def test_the_ratification_file_and_the_build_agree_on_the_count(real):
    """51 ratified, 48 published. The gap is the file's own arithmetic — one
    row superseded by the FP273 'Implementing entity' ruling, and absences the
    build refuses to publish over a print it holds — and it is recorded, not
    inferred."""
    doc = json.loads((ROOT / "data" / "registry_absences.json").read_text())
    assert doc["count"] == 51 and doc["superseded"] == 1
    assert doc["ratified"].startswith("owner, ")


def test_gold_case_w2a_rbp_fp273_absence_is_now_served(real):
    """The case that named the gap. Its notes said the runtime 'does not' serve
    the absence — verified at authoring time by calling `registry_note` and
    getting silence about financing — and that the case must keep passing when
    the runtime is wired to it. Both halves, here."""
    note = real.registry_note("How much GCF funding does FP273 request?")
    seg = _seg(note, "GCF funding requested")
    assert seg is not None and seg.startswith(
        "GCF funding requested: NOT PRINTED ANYWHERE IN THIS DOCUMENT")
    assert "The registry read pages 1-16 of this document." in seg
    assert note.rstrip().endswith(f"[{FP273}, cover pages]")


def test_the_fp273_absence_answer_still_scores_the_gold_case(real):
    """The honest-scoping answer the case expects today, and the stronger
    absence-as-fact answer the served note makes available: both satisfy the
    same scope alternation, and the second one VERIFIES claim by claim."""
    gold = ROOT / "scripts" / "answer_gold.jsonl"
    case = next(c for c in (json.loads(ln) for ln in
                            gold.read_text().splitlines() if ln.strip())
                if c["id"] == "w2a-rbp-fp273-absence")
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from eval_answers import score_answer

    note = real.registry_note(case["question"])
    ev = V.build_evidence([], note)
    scoped = (f"The funding proposal package for **FP273** does not state a "
              f"GCF funding figure: no GCF financing amount is printed in this "
              f"document. [{FP273}, cover pages]")
    served = (f"**FP273** — “Papua New Guinea REDD+ RBP for results period "
              f"2014-2016” (FMO, Papua New Guinea). There is no GCF funding "
              f"requested figure: it is not printed anywhere in this document, "
              f"which the registry records as a confirmed absence. "
              f"[{FP273}, cover pages]")
    for answer in (scoped, served):
        assert score_answer(case, answer, [], [note])["pass"], answer[:60]
    verdicts = V.classify_deterministic(V.extract_claims(served), ev)
    assert verdicts and [v.status for v in verdicts] == [V.SUPPORTED] * len(verdicts)


def test_no_other_gold_question_gains_or_loses_a_note_line(real):
    """The sweep, as a test. Every gold question's note is recomputed with the
    absence reader disabled and with it on; exactly one differs, and it differs
    only by the segment this pass added."""
    gold = ROOT / "scripts" / "answer_gold.jsonl"
    cases = [json.loads(ln) for ln in gold.read_text().splitlines() if ln.strip()]
    served = {c["id"]: real.registry_note(c["question"]) or "" for c in cases}
    real._absence_bits, keep = (lambda d, q, p: []), real._absence_bits
    try:
        silent = {c["id"]: real.registry_note(c["question"]) or "" for c in cases}
    finally:
        real._absence_bits = keep
    changed = [i for i in served if served[i] != silent[i]]
    assert changed == ["w2a-rbp-fp273-absence"]
    assert served[changed[0]].replace(
        _seg(served[changed[0]], "GCF funding requested") + "; ", "") == \
        silent[changed[0]]
