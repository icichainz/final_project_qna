"""Deterministic comparison planning (plan step 4).

Detection, resolution and cell filling run on a synthetic registry (the
registry._cache / _cache_v2 pattern of test_registry_v2.py) so the unit tests
never depend on the built corpus. The gate tests at the bottom run against the
REAL data files when they exist: FP254/FP228/FP248 must return all three rows
and refuse a cross-currency ranking.
"""
import re

import pytest

from gcf_qna.rag import planner, registry
from gcf_qna.rag.planner import (build_matrix, detect, fields_for, plan_and_render,
                                 render)

# --------------------------------------------------------------------------
# synthetic corpus
# --------------------------------------------------------------------------

REG1 = {
    "22_gcf-b40-02-add16-rev01-funding-proposal-package-fp254": {
        "fp": 254, "title": "Scaling Resilient Water Infrastructure",
        "accredited_entity": "IFC", "countries": ["Fiji"], "board": 40, "year": 2024},
    "48_gcf-b38-02-add08-funding-proposal-package-fp228": {
        "fp": 228, "title": "Cambodian Climate Financing Facility",
        "accredited_entity": "UNDP", "countries": ["Cambodia"], "board": 38, "year": 2024},
    "28_gcf-b40-02-add10-rev01-funding-proposal-package-fp248": {
        "fp": 248, "title": "Land-based Mitigation in West Kalimantan",
        "accredited_entity": "KfW", "countries": ["Indonesia"], "board": 40, "year": 2024},
    "02_gcf-b42-02-add16-funding-proposal-package-fp274": {
        "fp": 274, "title": "BRACE", "accredited_entity": "Save the Children Australia",
        "countries": ["Zambia"], "board": 42, "year": 2025},
}


def _money(raw, value, cur, page, section, status, unit=None):
    return {"raw": raw, "value": value, "currency": cur, "unit": unit,
            "page": page, "section": section, "status": status}


def _text(raw, page, section, status="canonical"):
    return {"raw": raw, "value": None, "currency": None, "unit": None,
            "page": page, "section": section, "status": status}


REG2 = {
    "22_gcf-b40-02-add16-rev01-funding-proposal-package-fp254": {
        "fp": 254, "facts": {
            "title": [_text("Scaling Resilient Water Infrastructure", 3, "A.1.1")],
            "accredited_entity": [_text("IFC", 3, "A.1.5")],
            "total_financing": [_money("$1,262,000,000 USD", 1262000000.0, "USD", 5,
                                       "A.7", "canonical")],
            "gcf_funding_requested": [_money("USD 258 million (USD", 258000000.0, "USD",
                                             108, "rule:A.8", "canonical", "million")],
        }},
    "48_gcf-b38-02-add08-funding-proposal-package-fp228": {
        "fp": 228, "facts": {
            "title": [_text("Cambodian Climate Financing Facility", 4, "A.1.1")],
            "total_financing": [_money("108.96 | million USD", 108960000.0, "USD", 56,
                                       "rule:A.7", "canonical", "million")],
            "gcf_funding_requested": [_money("50 million USD", 50000000.0, "USD", 56,
                                             "rule:C.1(a)", "canonical", "million")],
            "instruments": [_text("Grant", 9, "A.10 Grant"),
                            _text("Loan", 9, "A.10 Loan", "supporting")],
        }},
    "28_gcf-b40-02-add10-rev01-funding-proposal-package-fp248": {
        "fp": 248, "facts": {
            "title": [_text("Land-based Mitigation in West Kalimantan", 3, "A.1.1")],
            "total_financing": [_money("100,194,751 Eur", 100194751.0, "EUR", 5,
                                       "A.7", "canonical")],
            "gcf_funding_requested": [
                _money("59,484,751 Eur", 59484751.0, "EUR", 5, "A.8", "canonical"),
                _money("EUR 150,300,751", 150300751.0, "EUR", 51, "rule:C.1(a)",
                       "conflicting")],
        }},
    "02_gcf-b42-02-add16-funding-proposal-package-fp274": {
        "fp": 274, "facts": {
            "title": [_text("BRACE", 3, "A.1.1")],
            "gcf_funding_requested": [
                _money("40,511,264 USD", 40511264.0, "USD", 7, "A.8", "canonical"),
                _money("49,751,264", 49751264.0, None, 8, "A.10 Grant", "conflicting"),
                _money("40,751,254", 40751254.0, "USD", 40, "rule:C.1(a)", "conflicting")],
            # only a supporting candidate: no template section elected a canonical
            "ess_category": [_text("B", 12, "llm", "supporting")],
        }},
}


@pytest.fixture
def reg(monkeypatch):
    monkeypatch.setattr(registry, "_cache", REG1)
    monkeypatch.setattr(registry, "_cache_v2", REG2)
    return registry


@pytest.fixture
def no_registry(monkeypatch):
    """Neither registry file exists: nothing resolves, nothing is called missing."""
    monkeypatch.setattr(registry, "_cache", {})
    monkeypatch.setattr(registry, "_cache_v2", {})
    return registry


class FakeHit:
    def __init__(self, text, doc_id, score=0.5, page=None):
        self.text, self.doc_id, self.score, self.page = text, doc_id, score, page


class FakeRetriever:
    """Public contract only: search(query, top_k, doc_filter) -> [Hit]."""

    def __init__(self, hits_by_doc=None, unscoped_leak=None):
        self.hits_by_doc = hits_by_doc or {}
        self.unscoped_leak = unscoped_leak      # what retrieve.py returns when a
        self.calls = []                         # doc_filter matches nothing

    def search(self, query, top_k=5, doc_filter=None):
        self.calls.append((query, top_k, doc_filter))
        for stem, hits in self.hits_by_doc.items():
            if doc_filter and (doc_filter in stem or stem in doc_filter):
                return hits[:top_k]
        return ([self.unscoped_leak] if self.unscoped_leak else [])[:top_k]


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def test_two_fp_ids_with_a_field_word_fire(reg):
    plan = detect("Compare the GCF funding requested by FP254 and FP248")
    assert plan is not None
    assert [d.label for d in plan.docs] == ["FP254", "FP248"]
    assert plan.fields == ["gcf_funding_requested"]
    assert not plan.default_fields


def test_three_ids_slash_written_keep_question_order(reg):
    plan = detect("FP254/FP228/FP248 — which one costs the most?")
    assert [d.label for d in plan.docs] == ["FP254", "FP228", "FP248"]
    assert plan.fields == ["total_financing", "gcf_funding_requested"]
    assert len(plan.jobs) == 6


def test_two_ids_without_any_field_word_use_the_default_set(reg):
    plan = detect("Compare FP254 and FP274")
    assert plan.default_fields
    assert set(plan.fields) == set(planner.DEFAULT_FIELDS)
    assert plan.fields == ["title", "accredited_entity",
                           "total_financing", "gcf_funding_requested"]


def test_board_codes_resolve_through_the_registry(reg):
    plan = detect("Compare GCF/B.42/02/Add.16 and GCF/B.38/02/Add.08 financing")
    assert [d.label for d in plan.docs] == ["GCF/B.42/02/Add.16", "GCF/B.38/02/Add.08"]
    assert [d.fp for d in plan.docs] == [274, 228]
    assert plan.docs[0].doc_id.endswith("fp274")


def test_an_fp_and_a_board_code_naming_one_document_are_one_row(reg):
    plan = detect("Compare FP274 with GCF/B.42/02/Add.16 and FP248")
    assert [d.label for d in plan.docs] == ["FP274", "FP248"]
    assert plan.docs[0].aliases == ["GCF/B.42/02/Add.16"]


@pytest.mark.parametrize("q,fields", [
    ("Quel est le financement total du FP254 et du FP248 ?", ["total_financing"]),
    ("Comparez le financement du FVC pour FP254 et FP248", ["gcf_funding_requested"]),
    ("Quelle est l'entité accréditée du FP254 et du FP274 ?", ["accredited_entity"]),
    ("Quels pays sont couverts par FP254 et FP248 ?", ["countries"]),
    ("Combien de bénéficiaires pour FP254 et FP248 ?",
     ["beneficiaries_direct", "beneficiaries_indirect"]),
    ("Quelle est la période de mise en œuvre du FP254 et du FP248 ?",
     ["implementation_period"]),
    ("Comparez le cofinancement du FP254 et du FP248", ["co_financing"]),
])
def test_french_field_keywords(reg, q, fields):
    plan = detect(q)
    assert plan is not None and plan.fields == fields


@pytest.mark.parametrize("q,fields", [
    ("accredited entity of FP254 and FP248", ["accredited_entity"]),
    ("Which instruments do FP254 and FP228 use?", ["instruments"]),
    ("ESS category of FP254 vs FP248", ["ess_category"]),
    ("implementation period of FP254 and FP248", ["implementation_period"]),
    ("What is the total cost of FP254 and FP248?", ["total_financing"]),
    ("co-financing of FP254 and FP248", ["co_financing"]),
])
def test_english_field_keywords(reg, q, fields):
    assert detect(q).fields == fields


@pytest.mark.parametrize("q", [
    "What is the GCF funding requested by FP254?",         # one identifier
    "Compare the two proposals we discussed",              # no identifier
    "Compare the financing of these projects",             # 'compare' without ids
    "Which proposals finance mangrove restoration?",       # pure semantic
    "Summarize GCF/B.42/02/Add.16",                        # one board code
    "",
])
def test_single_id_and_semantic_questions_stay_with_the_conductor(reg, q):
    assert detect(q) is None


def test_a_year_is_not_an_identifier(reg):
    assert detect("proposals from fp2023 and 2024") is None


def test_fields_for_is_independent_of_identifiers():
    assert fields_for("total financing")[0] == ["total_financing"]
    assert fields_for("hello")[1] is True          # default set


# --------------------------------------------------------------------------
# jobs / missing documents
# --------------------------------------------------------------------------

def test_unresolvable_id_becomes_a_missing_document_row(reg):
    plan = detect("Compare the financing of FP254 and FP999")
    assert [d.label for d in plan.docs] == ["FP254", "FP999"]
    assert plan.docs[1].missing and plan.docs[1].doc_id is None
    missing_jobs = [j for j in plan.jobs if j.missing_document]
    assert len(missing_jobs) == 1 and missing_jobs[0].field is None

    m = build_matrix(plan)
    cell = m.row("FP999")[0]
    assert cell.status == "missing-document"
    out = render(m)
    assert "FP999" in out and "MISSING DOCUMENT" in out
    # the resolvable document is still fully answered
    assert m.cell("FP254", "total_financing").status == "stated"


def test_unknown_board_code_is_a_missing_document_row(reg):
    plan = detect("Compare GCF/B.42/02/Add.99 and FP254")
    assert plan.docs[0].missing
    assert build_matrix(plan).row("GCF/B.42/02/Add.99")[0].status == "missing-document"


def test_every_job_becomes_exactly_one_cell(reg):
    plan = detect("Compare FP254, FP228 and FP248")
    m = build_matrix(plan)
    assert len(m.cells) == len(plan.jobs)
    assert [(c.doc.label, c.field) for c in m.cells] == \
           [(j.doc.label, j.field) for j in plan.jobs]


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------

def test_canonical_cell_carries_value_currency_page_and_section(reg):
    m = build_matrix(detect("total financing of FP254 and FP248"))
    c = m.cell("FP254", "total_financing")
    assert (c.status, c.value, c.currency, c.page, c.section) == \
           ("stated", 1262000000.0, "USD", 5, "A.7")
    assert c.raw == "$1,262,000,000 USD"


def test_conflicting_candidates_are_attached_and_both_values_render(reg):
    m = build_matrix(detect("GCF funding of FP274 and FP254"))
    c = m.cell("FP274", "gcf_funding_requested")
    assert c.status == "contradictory"
    assert {x["page"] for x in c.conflicts} == {8, 40}
    out = render(m)
    assert "USD 40,511,264" in out and "49,751,264" in out and "40,751,254" in out
    assert out.count("CONFLICT in the same document") == 2


def test_supporting_only_candidate_is_used_and_flagged(reg):
    m = build_matrix(detect("ESS category of FP274 and FP254"))
    c = m.cell("FP274", "ess_category")
    assert c.status == "stated" and c.raw == "B" and c.note
    assert "no template-section candidate" in render(m)


def test_list_field_keeps_every_instrument(reg):
    m = build_matrix(detect("instruments of FP228 and FP248"))
    c = m.cell("FP228", "instruments")
    assert c.status == "stated" and [x["raw"] for x in c.extras] == ["Loan"]
    assert "Grant, Loan" in render(m)


def test_retrieval_fallback_fills_a_cell_the_registry_misses(reg):
    stem = "22_gcf-b40-02-add16-rev01-funding-proposal-package-fp254"
    r = FakeRetriever({stem: [FakeHit("ESS category: B, medium risk", stem, 0.71, 44)]})
    m = build_matrix(detect("ESS category of FP254 and FP274"), r)
    c = m.cell("FP254", "ess_category")
    assert c.status == "retrieved" and c.page == 44 and "medium risk" in c.text
    assert r.calls[0][1] == 3 and r.calls[0][2] == stem       # k=3, doc-scoped
    assert "retrieved from this document only" in render(m)


def test_a_hit_from_another_document_is_never_used(reg):
    """Retriever.search() degrades to an unscoped search when a filter matches
    nothing — a cell must not quote a different proposal's page."""
    other = "02_gcf-b42-02-add16-funding-proposal-package-fp274"
    r = FakeRetriever({}, unscoped_leak=FakeHit("Category C", other, 0.9, 3))
    m = build_matrix(detect("ESS category of FP254 and FP248"), r)
    c = m.cell("FP254", "ess_category")
    assert c.status == "missing" and c.text is None
    assert "Category C" not in render(m)


def test_retrieval_failure_degrades_to_missing(reg):
    class Boom:
        def search(self, *a, **kw):
            raise RuntimeError("index closed")
    m = build_matrix(detect("ESS category of FP254 and FP248"), Boom())
    assert m.cell("FP254", "ess_category").status == "missing"
    assert "retrieval failed" in render(m)


def test_no_retriever_gives_a_registry_only_matrix(reg):
    m = build_matrix(detect("ESS category of FP254 and FP248"), None)
    assert {c.status for c in m.cells} == {"missing"}
    assert "no registry fact and no retriever" in render(m)


# --------------------------------------------------------------------------
# comparability
# --------------------------------------------------------------------------

def test_mixed_currencies_are_not_comparable(reg):
    m = build_matrix(detect("total financing of FP254 and FP248"))
    v = m.comparable("total_financing")
    assert v.ok is False and not v
    assert "EUR vs USD" in v.reason and "no conversion rule" in v.reason
    assert "FP254 USD" in v.reason and "FP248 EUR" in v.reason
    assert "NOT COMPARABLE" in render(m)


def test_same_currency_and_unit_is_comparable(reg):
    m = build_matrix(detect("total financing of FP254 and FP228"))
    v = m.comparable("total_financing")
    assert v.ok and bool(v)
    assert "COMPARABLE" in render(m)


def test_a_document_stating_no_figure_blocks_the_ranking(reg):
    """FP274 has no total_financing fact: the other two must not be ranked
    'the largest' with a document silently left out."""
    m = build_matrix(detect("total financing of FP254 and FP274"))
    v = m.comparable("total_financing")
    assert not v.ok
    assert "FP274 states no comparable figure (missing)" in v.reason
    assert "would drop it from the answer" in v.reason


def test_a_column_with_no_figure_at_all_says_so(reg):
    m = build_matrix(detect("co-financing of FP254 and FP228"))
    v = m.comparable("co_financing")
    assert not v.ok and "no document states a parsed figure" in v.reason


def test_a_contradiction_blocks_the_ranking(reg):
    m = build_matrix(detect("GCF funding of FP274 and FP254"))
    v = m.comparable("gcf_funding_requested")
    assert not v.ok and "conflicting figures" in v.reason


def test_text_fields_are_never_ranked(reg):
    m = build_matrix(detect("Compare FP254 and FP228"))
    v = m.comparable("title")
    assert not v.ok and "text field" in v.reason


def test_currency_not_printed_is_not_assumed(reg, monkeypatch):
    facts = dict(REG2["48_gcf-b38-02-add08-funding-proposal-package-fp228"]["facts"])
    facts["total_financing"] = [_money("108,960,000", 108960000.0, None, 56,
                                       "A.7", "canonical")]
    patched = dict(REG2)
    patched["48_gcf-b38-02-add08-funding-proposal-package-fp228"] = {"fp": 228,
                                                                    "facts": facts}
    monkeypatch.setattr(registry, "_cache_v2", patched)
    m = build_matrix(detect("total financing of FP254 and FP228"))
    v = m.comparable("total_financing")
    assert not v.ok and "currency not printed for FP228" in v.reason
    assert "(currency not printed)" in render(m)


# --------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------

def test_render_is_deterministic(reg):
    q = "Compare the financing of FP254, FP228 and FP248"
    a = plan_and_render(q)
    b = plan_and_render(q)
    assert a == b


def test_render_marks_every_cell(reg):
    plan = detect("Compare FP254, FP228 and FP999")
    out = render(build_matrix(plan))
    for job in plan.jobs:
        pattern = re.escape(f"{job.doc.label} | {job.field or '*'} |")
        assert re.search(pattern, out), f"cell not rendered: {job.doc.label}/{job.field}"
    # one status word closes every cell line, and there are exactly as many
    # such lines as jobs — no cell dropped, none invented
    body = [ln for ln in out.splitlines()
            if " | " in ln and ln.rsplit(" | ", 1)[1] in planner.STATUSES]
    assert len(body) == len(plan.jobs)


def test_render_orders_documents_by_the_question_then_fields(reg):
    out = render(build_matrix(detect("Compare FP248, FP254 and FP228")))
    assert out.index("FP248 |") < out.index("FP254 |") < out.index("FP228 |")
    row = out[out.index("FP248 ->"):out.index("FP254 ->")]
    assert row.index("| title |") < row.index("| accredited_entity |") \
        < row.index("| total_financing |") < row.index("| gcf_funding_requested |")


def test_render_states_the_page_and_never_invents_one(reg):
    out = render(build_matrix(detect("total financing of FP254 and FP248")))
    assert "(p.5, A.7)" in out
    assert "rule A.8" not in out                     # not a field of this plan
    m = build_matrix(detect("GCF funding of FP254 and FP248"))
    assert "(p.108, rule A.8)" in render(m)          # section id not printed on p.108


def test_plan_and_render_returns_none_when_the_planner_does_not_fire(reg):
    assert plan_and_render("What does FP254 finance?") is None


# --------------------------------------------------------------------------
# degradation
# --------------------------------------------------------------------------

def test_without_registry_v2_every_cell_takes_the_retrieval_route(monkeypatch, reg):
    monkeypatch.setattr(registry, "_cache_v2", {})
    stems = list(REG1)
    r = FakeRetriever({s: [FakeHit(f"financing text of {s}", s, 0.6, 7)] for s in stems})
    m = build_matrix(detect("total financing of FP254 and FP248"), r)
    assert {c.status for c in m.cells} == {"retrieved"}
    assert len(r.calls) == 2 and all(c[2] for c in r.calls)


def test_without_any_registry_ids_are_unresolved_but_not_declared_missing(no_registry):
    """A closed-world 'this does not exist' needs the closed world loaded."""
    plan = detect("total financing of FP254 and FP248")
    assert plan is not None
    assert all(not d.missing and d.doc_id is None for d in plan.docs)
    assert [d.scope for d in plan.docs] == ["fp254", "fp248"]
    r = FakeRetriever({"22_gcf-b40-02-add16-rev01-funding-proposal-package-fp254":
                       [FakeHit("USD 1,262,000,000 total", "…fp254", 0.6, 5)]})
    m = build_matrix(plan, r)
    assert [c.status for c in m.cells] == ["retrieved", "missing"]
    assert "UNRESOLVED" in render(m)


def test_registry_v2_present_but_empty_for_one_document(reg):
    m = build_matrix(detect("Compare FP254 and FP274"), None)
    # FP274 has no accredited_entity fact in v2 and no retriever is available
    assert m.cell("FP274", "accredited_entity").status == "missing"
    assert m.cell("FP254", "accredited_entity").status == "stated"


# --------------------------------------------------------------------------
# the three fields Phase 1 added, and the invariants that keep the map whole
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question,want", [
    ("Who is the executing entity of FP254 and FP248?", "executing_entity"),
    ("Quelle est l'entité d'exécution de FP254 et FP248 ?", "executing_entity"),
    ("Which executing agency runs FP254 versus FP248?", "executing_entity"),
    ("What is the national designated authority for FP254 and FP248?",
     "national_designated_authority"),
    ("Compare the NDA of FP254 and FP248", "national_designated_authority"),
    ("Quelle est l'autorité nationale désignée de FP254 et FP248 ?",
     "national_designated_authority"),
    ("Compare the financial instruments of FP254 and FP248",
     "financial_instruments"),
    ("Quels instruments financiers pour FP254 et FP248 ?", "financial_instruments"),
])
def test_the_three_added_fields_are_detected(question, want):
    """193 / 101 / 70 documents state them, and nothing could ask for them."""
    fields, used_default = fields_for(question)
    assert want in fields and not used_default


def test_the_bare_instrument_word_still_asks_only_for_the_names():
    """'a grant' is a title word in this corpus, not a request for the A.10
    amount column — which is exactly why the amounts get their own rule."""
    fields, _ = fields_for("Which instruments does the equity proposal use?")
    assert fields == ["instruments"]
    # ('financial' also carries the generic money word 'financ', which has
    # always pulled in the two money fields — that arm is untouched here.)
    fields, _ = fields_for("What financial instruments are requested?")
    assert fields[-2:] == ["instruments", "financial_instruments"]


def test_every_field_of_every_rule_is_orderable_and_answerable():
    """Three maps, one vocabulary: a field that a rule can ask for must have a
    column position and a doc-scoped query, or a matrix cell it fills has no
    place to print and no way to fall back to retrieval."""
    ruled = {f for _, fs in planner._FIELD_RULES for f in fs}
    assert ruled <= set(planner.FIELD_ORDER)
    assert set(planner.FIELD_ORDER) == set(planner._FIELD_QUERIES)
    assert set(planner.DEFAULT_FIELDS) <= set(planner.FIELD_ORDER)
    assert planner.LIST_FIELDS <= set(planner.FIELD_ORDER)


def test_the_added_fields_were_inserted_not_reordered():
    """`fields_for` reads its output order off FIELD_ORDER, so a question
    asking for the same fields must still produce a byte-identical matrix."""
    before = ("title", "countries", "accredited_entity", "project_size",
              "total_financing", "gcf_funding_requested", "co_financing",
              "instruments", "implementation_period", "lifespan", "ess_category",
              "mitigation_outcome", "adaptation_outcome",
              "beneficiaries_direct", "beneficiaries_indirect")
    assert tuple(f for f in planner.FIELD_ORDER if f in before) == before


def test_the_amount_column_of_a10_is_a_list_cell_with_the_instrument_in_its_section(
        reg, monkeypatch):
    patched = dict(REG2)
    patched["48_gcf-b38-02-add08-funding-proposal-package-fp228"] = {
        "fp": 228, "facts": dict(
            REG2["48_gcf-b38-02-add08-funding-proposal-package-fp228"]["facts"],
            financial_instruments=[
                _money("$5 million USD", 5e6, "USD", 9, "A.10 Grant", "canonical",
                       "million"),
                _money("$46 million USD", 4.6e7, "USD", 9, "A.10 Loan",
                       "supporting", "million")])}
    monkeypatch.setattr(registry, "_cache_v2", patched)
    m = build_matrix(detect("financial instruments of FP228 and FP254"))
    cell = m.cell("FP228", "financial_instruments")
    assert cell.status == "stated" and cell.raw == "$5 million USD"
    assert [x["section"] for x in cell.extras] == ["A.10 Loan"]
    out = render(m)
    assert "$5 million USD, $46 million USD" in out
    assert 'also: "$46 million USD" (p.9, A.10 Loan)' not in out   # same page
    # a list of amounts is not one figure: no ranking is even offered
    assert "financial_instruments: COMPARABLE" not in out


# --------------------------------------------------------------------------
# GATE (real data files)
# --------------------------------------------------------------------------

_real = pytest.mark.skipif(not (registry.load() and registry.load_v2()),
                           reason="needs data/registry.json + data/registry_v2.json")


@_real
def test_gate_three_documents_all_rows_no_starvation():
    out = plan_and_render("Compare the financing of FP254, FP228 and FP248")
    assert out is not None
    for fp in ("FP254", "FP228", "FP248"):
        for f in ("total_financing", "gcf_funding_requested"):
            assert re.search(re.escape(f"{fp} | {f} |"), out), f"{fp}/{f} starved"
    assert "missing-document 0" in out


@_real
def test_gate_refuses_cross_currency_ranking():
    m = build_matrix(detect("Compare the financing of FP254, FP228 and FP248"))
    for f in ("total_financing", "gcf_funding_requested"):
        v = m.comparable(f)
        assert not v.ok and "no conversion rule" in v.reason
        assert "EUR vs USD" in v.reason


@_real
def test_gate_fp214_vs_fp274_currency_and_conflicts():
    m = build_matrix(detect("Compare the GCF financing of FP214 and FP274"))
    a = m.cell("FP214", "gcf_funding_requested")
    b = m.cell("FP274", "gcf_funding_requested")
    assert (a.currency, a.page) == ("EUR", 114) and a.status == "stated"
    assert (b.currency, b.page) == ("USD", 7) and b.status == "contradictory"
    # RE-PINNED 2026-08-26 (corpus cure), {8, 40} -> {40}. The note below says
    # "C83 corrects the p.8 figure too, but in `financial_instruments`, so the
    # gcf_funding_requested conflict candidate on that page still reads
    # 49,751,264 and the conflict does not dissolve". That was true while the
    # correction was the only thing that knew: the CORPUS page still printed
    # the misread, so the parser still minted a rival from it. The cure
    # re-extracted p.8 and it now prints '- [x] Grant: 40,751,264' — the
    # ratified figure C83 named, which the independent pymupdf extraction of
    # that page prints too. With no rival digits on p.8 there is no candidate
    # to conflict, and one of the document's two internal disagreements is
    # genuinely gone rather than hidden. p.40 still disagrees (40,751,254
    # against the canonical 40,751,264), so the cell stays `contradictory` and
    # the field stays non-comparable — both still asserted here.
    assert {x["page"] for x in b.conflicts} == {40}
    out = render(m)
    assert "EUR 38.17 million (p.114, rule C.1(a))" in out
    # Re-pinned 2026-08-26 (cross-check round): C103 corrects FP274's canonical
    # GCF request on p.7 from '40,511,264 USD' to '40,751,264 USD' — the
    # independent pymupdf extraction of the cited page prints
    # 'A.8 / B.2(b) — "40,751,264 _____ USD"' and the qwen markdown the store
    # cited misread it. The section prints 'corrected' rather than 'A.8'
    # because the ratified row carries no section of its own.
    # Everything the ROW is about is unchanged and still asserted above: the
    # cell is still USD/p.7/contradictory, and the two conflicting pages are
    # still {8, 40}. C83 corrects the p.8 figure too, but in
    # `financial_instruments`, so the gcf_funding_requested conflict candidate
    # on that page still reads 49,751,264 and the conflict does not dissolve.
    assert "USD 40,751,264 (p.7, corrected)" in out
    assert not m.comparable("gcf_funding_requested").ok
