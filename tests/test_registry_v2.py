"""Schema-2 registry: deterministic template parsing and the v2 accessors.

The parser tests run on synthetic markdown shaped like the real corpus (page
markers, GCF template headings, VLM noise) so they document the heading
variants the builder supports. The accessor tests use a synthetic v2 dict via
registry._cache_v2, the parallel of the v1 registry._cache pattern.
"""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from gcf_qna.rag import registry


def _build_module():
    p = Path(__file__).resolve().parents[1] / "scripts" / "build_registry_v2.py"
    spec = importlib.util.spec_from_file_location("build_registry_v2", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _build_module()


def page(n: int, body: str) -> str:
    return f"---\n**Page {n}**\n---\n{body}\n\n"


def facts_of(text: str):
    return B.build_document("99_gcf-b42-02-add16-funding-proposal-package-fp274", text)["facts"]


def canon(facts, field):
    return next((c for c in facts.get(field, []) if c["status"] == "canonical"), None)


# --------------------------------------------------------------------------
# number normalization
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("49,751,264", 49751264),          # US thousands grouping
    ("21,128,224", 21128224),
    ("1,234", 1234),                   # single 3-digit group after ',' is grouping
    ("358.26", 358.26),                # US decimal point
    ("46,10", 46.10),                  # OCR of a decimal printed with a comma
    ("44,709782", 44.709782),          # ... and of a long decimal
    ("544.0", 544.0),
    ("28654", 28654),
    ("1 234 567", 1234567),            # space-grouped
])
def test_to_number(raw, expected):
    assert B.to_number(raw) == pytest.approx(expected)


def test_unparseable_raw_keeps_value_null():
    assert B.to_number("Enter amount") is None
    assert B.to_number("USD") is None
    assert B.read_amount("Enter amount | million USD ($)") is None


def test_million_equals_full_digits():
    """'23.6 million USD' and '23,600,000 USD' are the same value."""
    a = B.read_amount("23.6 million USD")
    b = B.read_amount("23,600,000 USD")
    assert a["value"] == b["value"] == 23_600_000
    assert a["unit"] == "million" and b["unit"] is None
    assert B._same_value(a, b)


def test_unit_word_clashing_with_the_figure_suppresses_the_value():
    """'68.780 | billion USD ($)' cannot be 68.78 billion when the page's own
    rows sum to 68.78 million, and it cannot be silently rescaled either: the
    raw is published with NO value rather than a number contradicting the print.
    (The GCF template prints 'million USD ($)' as a currency-COLUMN label.)"""
    got = B.read_amount("40,751,254 | million USD ($)")
    assert got["value"] is None and got["unit"] is None
    assert got["raw"] == "40,751,254 | million USD" and got["currency"] == "USD"
    assert B.read_amount("68.780 | billion USD ($)")["value"] is None
    # the same words below the ceiling are a real unit, not a clash
    assert B.read_amount("46.10 million USD")["value"] == 46_100_000


def test_currency_normalization():
    assert B.read_amount("$46,10 | million USD ($)")["currency"] == "USD"
    assert B.read_amount("44,709782 | million euro (€)")["currency"] == "EUR"
    assert B.read_amount("219,981,104 Eur")["currency"] == "EUR"
    assert B.read_amount("49,751,264")["currency"] is None      # not stated -> null


def test_placeholders_percentages_and_years_are_not_amounts():
    assert B.read_amount("Enter number %") is None
    assert B.read_amount("10%") is None
    assert B.read_amount("5 years / 60 months") is None
    assert B.read_amount("2024") is None                        # bare year
    assert B.read_amount("(G = JAH30)") is None                 # digits glued to a word


# --------------------------------------------------------------------------
# heading variants + page attribution
# --------------------------------------------------------------------------

MODERN = (
    page(1, "# Consideration of funding proposals - Addendum XVI\n\n"
            "a) A funding proposal titled \"Building the Climate Resilience of Children\";")
    + page(3, "#### Project/Programme title:\nBuilding the Climate Resilience of Children\n\n"
              "#### Country(ies):\nZambia, South Sudan, Togo\n\n"
              "#### Accredited Entity:\nSave the Children Australia")
    + page(7, "## A.7 Total financing (GCF + co-finance)**\n46,737,340 USD\n\n"
              "## A.8 Total GCF funding requested\n40,511,264 USD\n\n"
              "### A.9 Project size\nSmall (Up to USD 50 million)")
    + page(8, "## A.10. Financial instruments (a) requested for the GCF funding\n"
              "- Grant: 49,751,264\n- Loan: Enter number\n\n"
              "## A.11. Implementation period\n- 5 years / 60 months\n\n"
              "## A.12. Total lifespan\n- 10 years\n\n"
              "## A.14. ESS category\n- Refer to the AE's safeguard policy\n- C")
    + page(40, "## C.1 Total Financing\n\n**(a) Received GCF funding (i + ii + iii)**\n\n"
               "Total amount | Currency\n-------------|---------\n"
               "40,751,254Enter amount | million USD ($)")
)


def test_modern_template_block():
    f = facts_of(MODERN)
    assert canon(f, "title")["raw"].startswith("Building the Climate Resilience")
    assert canon(f, "title")["page"] == 3
    assert canon(f, "countries")["raw"] == "Zambia, South Sudan, Togo"
    assert canon(f, "accredited_entity")["raw"] == "Save the Children Australia"
    assert canon(f, "project_size")["raw"] == "Small (Up to USD 50 million)"
    assert canon(f, "ess_category")["raw"] == "C"
    assert canon(f, "implementation_period")["value"] == 60
    assert canon(f, "implementation_period")["unit"] == "months"
    assert canon(f, "lifespan")["value"] == 10 and canon(f, "lifespan")["unit"] == "years"
    tf = canon(f, "total_financing")
    assert (tf["value"], tf["page"], tf["section"]) == (46_737_340, 7, "A.7")


def test_page_attribution_follows_the_page_markers():
    f = facts_of(MODERN)
    pages = {(c["section"], c["page"]) for c in f["gcf_funding_requested"]}
    assert ("A.8", 7) in pages          # template section
    assert ("A.10 Grant", 8) in pages   # instrument line, next page
    assert ("rule:C.1(a)", 40) in pages  # section C, 30+ pages later


def test_old_template_a1x_and_b2_headings():
    text = (
        page(5, "## A.1 Brief Project / Programme Information\n\n"
                "### A.1.1 Project / programme title\nFiji Urban Water Supply Project\n\n"
                "### A.1.3 Country of operation\nRepublic of Fiji\n\n"
                "### A.1.5 Accredited entity\nAsian Development Bank")
        + page(13, "### B.2. Project Financing Information\n\n"
                   "#### (a) Total project financing\n"
                   "| Financial Instrument | Amount | Currency | Tenor |\n|---|---|---|---|\n"
                   "| (a) = (b) + (c) | 44,709782 | million euro (£) | Options |\n\n"
                   "#### (b) GCF financing to recipient\n"
                   "| (b) = (i) + (ii) | 23,709782 | million euro (£) | Options |")
    )
    f = facts_of(text)
    assert canon(f, "title")["raw"] == "Fiji Urban Water Supply Project"
    assert canon(f, "countries")["raw"] == "Republic of Fiji"
    assert canon(f, "accredited_entity")["raw"] == "Asian Development Bank"
    tf = canon(f, "total_financing")
    assert tf["value"] == pytest.approx(44_709_782) and tf["currency"] == "EUR"
    assert tf["section"] == "rule:B.2(a)" and tf["page"] == 13   # page printed no number
    gcf = canon(f, "gcf_funding_requested")
    assert gcf["value"] == pytest.approx(23_709_782) and gcf["page"] == 13


def test_heading_number_drift_is_recorded_not_corrected():
    """VLM output prints 'A7' with the GCF request inside it (FP269). The label
    decides the field; the printed number is kept as provenance."""
    text = page(5, "### A7. Total financing (GCF + co-financing)\n"
                   "- **Total GCF funding requested**: 358.26 million USD\n"
                   "- **Multi-country proposals**: 190.00 million USD")
    f = facts_of(text)
    gcf = canon(f, "gcf_funding_requested")
    assert gcf["value"] == 358_260_000 and gcf["unit"] == "million"
    # both printed figures survive, the second as a conflict
    assert {c["value"] for c in f["gcf_funding_requested"]} == {358_260_000, 190_000_000}
    assert [c["status"] for c in f["total_financing"] if c["value"] == 190_000_000] \
        == ["conflicting"]
    assert f["total_financing"][0]["section"] == "A7"


def test_same_line_and_table_row_labels():
    text = page(3, "Programme Title: Papua New Guinea REDD+ RBP\n\n"
                   "Country: Papua New Guinea\n\n"
                   "| Accredited Entity | Food and Agriculture Organization |")
    f = facts_of(text)
    assert canon(f, "title")["raw"] == "Papua New Guinea REDD+ RBP"
    assert canon(f, "countries")["raw"] == "Papua New Guinea"
    assert canon(f, "accredited_entity")["raw"].startswith("Food and Agriculture")


def test_prose_mentions_are_not_field_labels():
    text = page(9, "The Accredited Entity shall report annually to the Board and the "
                   "total GCF funding requested was discussed at length.\n")
    f = facts_of(text)
    assert "accredited_entity" not in f


def test_beneficiaries_direct_and_indirect():
    text = page(7, "## A.6 Expected adaptation outcome (Core indicator 2: direct and "
                   "indirect beneficiaries reached)\n808,150 direct  \n8,040,844 indirect")
    f = facts_of(text)
    assert canon(f, "beneficiaries_direct")["value"] == 808_150
    assert canon(f, "beneficiaries_indirect")["value"] == 8_040_844


def test_percentage_shares_are_not_beneficiary_counts():
    text = page(5, "### A6. Expected adaptation benefits\n"
                   "13.1% (direct benefits) + 8.5% (indirect benefits)")
    f = facts_of(text)
    assert "beneficiaries_direct" not in f and "beneficiaries_indirect" not in f


# --------------------------------------------------------------------------
# conflict detection
# --------------------------------------------------------------------------

def test_conflict_detection_keeps_every_candidate_with_page_and_raw():
    f = facts_of(MODERN)
    by_page = {c["page"]: c for c in f["gcf_funding_requested"]}
    assert by_page[7]["status"] == "canonical" and by_page[7]["value"] == 40_511_264
    assert by_page[8]["status"] == "conflicting" and by_page[8]["raw"] == "49,751,264"
    assert by_page[40]["status"] == "conflicting" and by_page[40]["value"] == 40_751_254
    assert by_page[40]["raw"] == "40,751,254"          # exact source text, no invention


def test_rounded_restatement_supports_it_does_not_conflict():
    """'$26.74 million' can only be as precise as 10 000, so it agrees with
    26,736,295 — printed precision, not a fixed percentage, sets the tolerance."""
    text = (page(8, "## A.8 Total GCF funding requested\n26,736,295 USD")
            + page(66, "#### (a) Requested GCF funding\n$26.74 | million USD"))
    f = facts_of(text)
    assert [c["status"] for c in f["gcf_funding_requested"]] == ["canonical", "supporting"]


def test_full_precision_mismatch_is_a_conflict():
    text = (page(8, "## A.8 Total GCF funding requested\n40,511,264 USD")
            + page(40, "#### (a) Requested GCF funding\n40,751,254 USD"))
    f = facts_of(text)
    assert [c["status"] for c in f["gcf_funding_requested"]] == ["canonical", "conflicting"]


def test_incompatible_currency_is_never_ranked_as_a_conflict():
    text = (page(8, "## A.8 Total GCF funding requested\n40,511,264 USD")
            + page(40, "#### (a) Requested GCF funding\n40,000,000 EUR"))
    f = facts_of(text)
    assert [c["status"] for c in f["gcf_funding_requested"]] == ["canonical", "supporting"]


def test_years_and_months_are_not_compared_across_units():
    a = {"value": 5, "currency": None, "unit": "years"}
    b = {"value": 60, "currency": None, "unit": "months"}
    assert not B._compatible(a, b)


def test_instrument_breakdown_is_not_a_contradiction():
    """A.10 lists one amount per instrument: a grant beside a loan is the
    breakdown of the request, not two readings of one field."""
    text = page(8, "## A.10 Financial instruments requested for the GCF funding\n"
                   "- Grant: 20,000,000\n- Loan: 30,000,000\n- Equity: Enter number")
    f = facts_of(text)
    assert {c["section"] for c in f["financial_instruments"]} == {"A.10 Grant", "A.10 Loan"}
    assert not [c for c in f["financial_instruments"] if c["status"] == "conflicting"]
    # ... and with several instruments no single one is promoted to the request
    assert "gcf_funding_requested" not in f


def test_page_break_truncation_is_not_a_second_reading():
    text = page(7, "### A.5 Expected mitigation outcomes\n601,550 tCO2e\n\n"
                   "### A.5 Expected mitigation outcomes\n60")
    f = facts_of(text)
    assert [c["value"] for c in f["mitigation_outcome"]] == [601_550]


def test_section_is_only_claimed_when_the_page_prints_it():
    """A label found on page 55 of a package does not make page 55 'B.2(b)'."""
    text = (page(9, "#### (a) Total project financing\n- 122,564 | USD")
            + page(55, "### B.2 GCF financing to recipient\n- US$ 17,346,000"))
    f = facts_of(text)
    assert canon(f, "total_financing")["section"] == "rule:B.2(a)"   # not printed
    assert canon(f, "gcf_funding_requested")["section"] == "B.2"     # printed on the page


def test_co_financing_line_never_becomes_the_gcf_request():
    """A.8's window may list the co-finance split before the total (113_add09):
    the co-financing line must not be read as the GCF request."""
    text = page(5, "- **A.7.1 Total financing (GCF + co-finance):** 143,327,000 USD\n"
                   "- **A.8.1 Total GCF funding requested:**\n"
                   "  - Co-financing: 60,477,000 USD\n"
                   "  - Total: 82,849,000 USD")
    f = facts_of(text)
    c = canon(f, "gcf_funding_requested")
    assert c["value"] == 82_849_000 and c["raw"].startswith("82,849,000")
    assert 60_477_000 not in [x["value"] for x in f["gcf_funding_requested"]]
    # ... and the A.7 window stops at the A.8 label instead of eating its lines
    assert [x["value"] for x in f["total_financing"]] == [143_327_000]


def test_component_far_below_the_canonical_total_is_not_a_conflict():
    """FP271-shaped: portfolio targets and component lines are smaller parts of
    the total, not competing readings of it."""
    text = (page(5, "## A.7 Total financing (GCF + co-finance)\n993,000,000 USD")
            + page(60, "#### (a) Total project financing\n200,000,000 USD"))
    f = facts_of(text)
    assert [c["status"] for c in f["total_financing"]] == ["canonical", "supporting"]


def test_ambiguous_scale_still_detects_a_mismatched_mantissa():
    """Both prints are unscalable ('28,654 million USD'), but 26,654 != 28,654
    whatever the true scale is — the conflict survives the value suppression."""
    text = (page(5, "### A.8 Total GCF funding requested\n28,654 million USD")
            + page(48, "| (b) Requested GCF amount | 26,654 million USD |"))
    f = facts_of(text)
    got = f["gcf_funding_requested"]
    assert [c["value"] for c in got] == [None, None]
    assert [c["status"] for c in got] == ["canonical", "conflicting"]


def test_abbreviated_scale_words_need_an_adjacent_currency():
    """'28 M USD' is 28 million (FP151's real A.7 print); '5 m of pipe' is not
    money at all, and neither is a bare figure that only shares a row with USD."""
    assert B.read_amount("28 M USD")["value"] == 28_000_000
    assert B.read_amount("USD$ 500 M (Choose an item)")["value"] == 500_000_000
    assert B.read_amount("2.5 MM USD")["value"] == 2_500_000
    assert B.read_amount("5 m of pipe") is None
    assert B.read_amount("proposals, 9 out of 17: 190.00 million USD")["value"] \
        == 190_000_000                      # the '9' is prose, not a $9 amount


def test_unbindable_scale_word_suppresses_the_value():
    """FP151 p45: '| 32,500 plus | million USD ($) |'. The row prints a scale
    word that cannot bind to the figure, and binding it would contradict it —
    so no number is published. When only a template PLACEHOLDER separates the
    two, the currency column is unfilled boilerplate and the figure stands."""
    assert B.read_amount("32,500 plus | million USD ($)")["value"] is None
    assert B.read_amount("32,500 plus | million USD ($)")["raw"] == "32,500"
    assert B.read_amount("40,751,254Enter amount | million USD ($)")["value"] == 40_751_254


def test_figure_too_small_for_its_label_keeps_the_print_not_the_number():
    """'999.9 USD' under 'A.7 Total financing' is a real print with an unstated
    scale; a $999.90 total would be a fiction. Bare furniture still drops."""
    got = B.read_amount("999.9 USD")
    assert got["value"] is None and got["raw"] == "999.9 USD"
    assert B.read_amount("30") is None


def test_prose_hit_is_not_canonical_when_the_template_heading_exists():
    """FP254-shaped: the A.8 table prints 58,000,000 and page 108 prose says
    'USD 258 million'. The template section wins; the prose stays a candidate."""
    text = (page(5, "A.7 Total financing (GCF + co-financiers) | $1,262,000,000 USD |\n\n"
                    "A.8 Total GCF funding | $58,000,000 USD |")
            + page(108, "### Project details:\n- Total target financing: USD 1.26 billion;\n"
                        "- GCF funding requested: USD 258 million (USD 250 million loan)"))
    f = facts_of(text)
    c = canon(f, "gcf_funding_requested")
    assert (c["value"], c["page"], c["section"]) == (58_000_000, 5, "A.8")
    assert [x["status"] for x in f["gcf_funding_requested"] if x["page"] == 108] \
        == ["conflicting"]


def test_empty_template_section_yields_no_canonical_rather_than_prose():
    """When the document prints the A.8 heading but nothing parses there, a
    far-away prose match must not be promoted to the document's stated value."""
    text = (page(5, "### A.8 Total GCF funding requested\nEnter amount")
            + page(90, "- GCF funding requested: USD 258 million (USD 250 million loan)"))
    f = facts_of(text)
    assert canon(f, "gcf_funding_requested") is None
    assert [c["status"] for c in f["gcf_funding_requested"]] == ["supporting"]
    assert f["gcf_funding_requested"][0]["section"] == "rule:A.8"


def test_gcf_above_total_is_flagged_not_published_silently():
    text = (page(5, "## A.7 Total financing (GCF + co-finance)\n30,000,000 USD\n\n"
                    "## A.8 Total GCF funding requested\n40,000,000 USD"))
    doc = B.build_document("99_gcf-b42-02-add16-x", text)
    assert doc["coverage"]["suspect"] == "gcf>total"
    ok = B.build_document("99_gcf-b42-02-add16-x", page(
        5, "## A.7 Total financing (GCF + co-finance)\n50,000,000 USD\n\n"
           "## A.8 Total GCF funding requested\n40,000,000 USD"))
    assert "suspect" not in ok["coverage"]


def test_board_code_comes_from_the_document_id():
    f = facts_of(MODERN)
    assert canon(f, "board_code")["raw"] == "GCF/B.42/02/Add.16"
    assert canon(f, "board_code")["section"] == "doc_id"


def test_candidate_schema_is_stable():
    keys = {"raw", "value", "currency", "unit", "page", "section", "status"}
    for cands in facts_of(MODERN).values():
        for c in cands:
            assert set(c) == keys
            assert c["status"] in {"canonical", "supporting", "conflicting"}
            assert isinstance(c["page"], int)


# --------------------------------------------------------------------------
# accessors (synthetic v2 registry)
# --------------------------------------------------------------------------

V2 = {
    "02_gcf-b42-02-add16-funding-proposal-package-fp274": {
        "fp": 274, "title": "BRACE", "board": 42, "year": 2025,
        "facts": {
            "gcf_funding_requested": [
                {"raw": "40,511,264 USD", "value": 40511264.0, "currency": "USD",
                 "unit": None, "page": 7, "section": "A.8", "status": "canonical"},
                {"raw": "49,751,264", "value": 49751264.0, "currency": None, "unit": None,
                 "page": 8, "section": "A.10 Grant", "status": "conflicting"},
                {"raw": "40,751,254", "value": 40751254.0, "currency": "USD", "unit": None,
                 "page": 40, "section": "C.1(a)", "status": "conflicting"},
            ],
            "total_financing": [
                {"raw": "46,737,340 USD", "value": 46737340.0, "currency": "USD",
                 "unit": None, "page": 7, "section": "A.7", "status": "canonical"},
            ],
        },
        "coverage": {"era": "A5-A14 block (FP template v2/v3)", "llm_fallback": False},
    },
    "03_gcf-b42-02-add15-funding-proposal-package-fp273": {
        "fp": 273, "title": "PNG REDD+ RBP", "board": 42, "year": 2025,
        "facts": {"title": [{"raw": "PNG REDD+ RBP", "value": None, "currency": None,
                             "unit": None, "page": 3, "section": "llm",
                             "status": "supporting"}]},
        "coverage": {"era": "unrecognized template", "llm_fallback": True},
    },
    "266_gcf-b11-04-add08": {
        "fp": 8, "title": "Fiji Urban Water Supply", "board": 11, "year": 2015,
        "facts": {"title": [{"raw": "Fiji Urban Water Supply", "value": None,
                             "currency": None, "unit": None, "page": 5,
                             "section": "A.1.1", "status": "canonical"}]},
        "coverage": {"era": "A.1.x block (FP template v1)", "llm_fallback": False},
    },
}


@pytest.fixture
def reg2(monkeypatch):
    monkeypatch.setattr(registry, "_cache_v2", V2)
    return registry


def test_facts_by_fp_number_and_by_stem(reg2):
    assert reg2.facts(274)["total_financing"][0]["value"] == 46737340.0
    assert reg2.facts("FP274") == reg2.facts(274)
    assert reg2.facts("02_gcf-b42-02-add16-funding-proposal-package-fp274") == reg2.facts(274)
    assert reg2.facts(999) == {}


def test_canonical_returns_the_template_section_candidate(reg2):
    c = reg2.canonical(274, "gcf_funding_requested")
    assert (c["value"], c["page"], c["section"]) == (40511264.0, 7, "A.8")
    assert reg2.canonical(274, "ess_category") is None       # field not extracted
    assert reg2.canonical(999, "total_financing") is None


def test_conflicts_lists_only_conflicting_fields(reg2):
    got = reg2.conflicts(274)
    assert set(got) == {"gcf_funding_requested"}
    assert {c["page"] for c in got["gcf_funding_requested"]} == {8, 40}
    assert reg2.conflicts(8) == {}


def test_v2_accessors_do_not_touch_the_v1_cache(monkeypatch):
    """Another agent's code calls by_fp/by_year mid-flight: v2 must not leak."""
    monkeypatch.setattr(registry, "_cache", {})
    monkeypatch.setattr(registry, "_cache_v2", V2)
    assert registry.facts(274)                     # v2 answers
    assert registry.by_fp(274) is None             # v1 stays empty
    assert registry.by_year(2025) == []
    assert registry.registry_note("What is FP274?") is None


def test_missing_v2_file_is_an_empty_registry(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "_cache_v2", None)
    monkeypatch.setattr(registry.config, "DATA_DIR", tmp_path)
    try:
        assert registry.load_v2() == {}
        assert registry.facts(274) == {} and registry.conflicts(274) == {}
    finally:
        registry._cache_v2 = None


# --------------------------------------------------------------------------
# v1 identifier-parsing regressions (zero-padded and hyphenated FP ids)
# --------------------------------------------------------------------------

REG1 = {
    "266_gcf-b11-04-add08": {"fp": 8, "title": "Urban Water Supply and Wastewater, Fiji",
                             "accredited_entity": "ADB", "countries": ["Fiji"],
                             "board": 11, "year": 2015},
    "180_gcf-b21-10-add14": {"fp": 86, "title": "Green Cities Facility",
                             "accredited_entity": "EBRD", "countries": ["Albania"],
                             "board": 21, "year": 2018},
    "60_gcf-b37-02-add08-funding-proposal-package-fp220": {
        "fp": 220, "title": "Blue Action Fund", "accredited_entity": "KfW",
        "countries": ["Kenya"], "board": 37, "year": 2023},
}


@pytest.fixture
def reg1(monkeypatch):
    monkeypatch.setattr(registry, "_cache", REG1)
    return registry


def test_zero_padded_fp_id_resolves_to_the_right_proposal(reg1):
    resolved, missing = reg1.resolve_fps("Summarize FP0086")
    assert [r["fp"] for r in resolved] == [86] and not missing
    note = reg1.registry_note("Summarize FP0086")
    assert "Green Cities Facility" in note
    assert "Urban Water Supply" not in note            # used to resolve to FP8


def test_hyphenated_fp_id_reaches_the_registry(reg1):
    resolved, _ = reg1.resolve_fps("What does FP-220 finance?")
    assert [r["fp"] for r in resolved] == [220]
    assert "Blue Action Fund" in reg1.registry_note("What does FP-220 finance?")


def test_single_digit_and_spaced_ids_still_work(reg1):
    assert [r["fp"] for r in reg1.resolve_fps("FP8 and FP 86")[0]] == [8, 86]
    # zero padding and the bare number are the same identifier, not two
    assert len(reg1.resolve_fps("Compare FP086 with FP86")[0]) == 1


def test_a_four_digit_number_is_not_an_fp_id(reg1):
    resolved, missing = reg1.resolve_fps("proposals from fp2023")
    assert not resolved and not missing


def test_stem_suffixes_keep_their_fp(reg2):
    """4 corpus stems end '-fp272_0' / '-fp203_1': '\\b' finds no boundary
    between '2' and '_', so the FP used to be lost."""
    assert registry.FP_RE.search("04_gcf-b42-02-add14-funding-proposal-package-fp272_0") \
        .group(1) == "272"
    assert registry.FP_RE is registry._FP_RE          # app imports the public name
    assert reg2.facts("03_gcf-b42-02-add15-funding-proposal-package-fp273_0") == \
        reg2.facts(273)


def test_package_preference_is_case_folded(monkeypatch):
    """The package doc is preferred over a mention doc even when the stem spells
    the id in capitals ('..._Funding_proposal_package_for_FP203')."""
    monkeypatch.setattr(registry, "_cache", {
        "12_status-approved-fps-adding-host-countries": {"fp": 203, "title": "status doc"},
        "72_GCF_B.35_02_Add.05_Funding_proposal_package_for_FP203": {"fp": 203,
                                                                    "title": "package"},
    })
    assert registry.by_fp(203)["title"] == "package"


# ==========================================================================
# Phase 3 — template VARIANTS: the fallback pass and the era families
#
# These tests are written against the corpus the recognizers were written
# for. Each variant test names the real extracted document whose markdown
# motivated the rule and asserts the field AND its page provenance, so a
# rule that stops reading that document fails here rather than silently
# shrinking coverage. The fixtures are read-only: nothing under data/ is
# written, and the tests skip when the extraction directory is absent.
# ==========================================================================

EXTRACTED = Path(__file__).resolve().parents[1] / "data" / "extracted" / "vlm" / "qwen_qwen2.5-vl-7b"
REG_V2 = Path(__file__).resolve().parents[1] / "data" / "registry_v2.json"

needs_corpus = pytest.mark.skipif(not EXTRACTED.is_dir(),
                                  reason="extracted corpus absent")


#: Documents whose pages the Phase 3 VLM re-extraction is replacing (the
#: 28-page worklist, in flight 2026-08-26). A pin below is a claim about a
#: RULE reading a specific extraction of a page; while that extraction is
#: being rewritten the pin is suspended rather than failed, and the test says
#: what the document now reads so the census can be re-taken when the pass
#: lands. Nothing outside this set is ever skipped.
REEXTRACTING = {
    "136_gcf-b26-02-add11", "150_gcf-b25-02-add02", "167_gcf-b23-02-add02",
    "201_gcf-b19-22-add16-rev01", "206_gcf-b19-22-add11", "208_gcf-b19-22-add09",
    "219_gcf-b18-04-add11", "221_gcf-b18-04-add09", "226_gcf-b18-04-add04",
    "234_gcf-b16-07-add04", "238_gcf-b15-13-add10", "251_gcf-b14-07-add06",
    "253_gcf-b14-07-add04", "258_gcf-b13-16-add08", "262_gcf-b13-16-add04",
    "263_gcf-b13-16-add03", "266_gcf-b11-04-add08", "268_gcf-b11-04-add06",
    "32_gcf-b40-02-add06-funding-proposal-package-fp244",
    "50_gcf-b38-02-add06-funding-proposal-package-fp226",
    "78_gcf-b34-02-add07-rev01",
}


def _pin_or_skip(stem: str, field: str, c, ok: bool) -> None:
    if ok:
        return
    if stem in REEXTRACTING:
        pytest.skip(f"{stem} is on the Phase 3 re-extraction worklist; its "
                    f"{field} now reads {c!r} — re-take this pin when the pass lands")
    raise AssertionError(f"{stem}: {field} reads {c!r}")


def _doc(stem: str, fallback: bool = True) -> dict:
    p = EXTRACTED / f"{stem}.md"
    if not p.exists():
        pytest.skip(f"{stem}.md absent")
    return B.build_document(stem, p.read_text(encoding="utf-8", errors="replace"),
                            fallback=fallback)


# --------------------------------------------------------------------------
# the additive discipline, stated as a test
# --------------------------------------------------------------------------

def test_a_fallback_rule_never_touches_a_field_the_strict_pass_read():
    """The whole safety argument in one assertion: the second pass is consulted
    per field, and only for a field the strict rules left empty."""
    pages = B.split_pages(page(5, "## A.7 Total financing (GCF + co-finance)\n"
                                  "30,000,000 USD\n\n"
                                  "## Total requested (i+ii+iii)\n99,000,000 USD"))
    strict = B.extract_candidates(pages, fallback=False)
    both = B.extract_candidates(pages, fallback=True)
    assert both["total_financing"] == strict["total_financing"]
    # ... and the field the strict pass could not read IS filled by the variant
    assert "gcf_funding_requested" not in strict
    assert both["gcf_funding_requested"][0]["value"] == 99_000_000


@needs_corpus
@pytest.mark.parametrize("stem", [
    "02_gcf-b42-02-add16-funding-proposal-package-fp274",   # modern, both fields
    "266_gcf-b11-04-add08",                                 # v1
    "241_gcf-b15-13-add07",                                 # v1, gained by fallback
])
def test_the_strict_pass_output_is_untouched_by_the_fallback_pass(stem):
    """Every field the strict pass produced comes back byte-identical."""
    strict = _doc(stem, fallback=False)["facts"]
    both = _doc(stem, fallback=True)["facts"]
    for field, cands in strict.items():
        assert both[field] == cands, field


# --------------------------------------------------------------------------
# variant recognizers, each against the document it was written for
# --------------------------------------------------------------------------

@needs_corpus
@pytest.mark.parametrize("stem,field,page_no,value,raw_bit", [
    # 'Total requested' — the v1 B.2(b) block's own total row, in the spellings
    # the corpus prints. The strict B.2(b) rule opens on 'Requested GCF amount'
    # but its window closes before this row.
    ("241_gcf-b15-13-add07", "gcf_funding_requested", 9, None, "24,140"),
    ("148_gcf-b25-02-add04", "gcf_funding_requested", 12, 26_574_567.0, "26,574,567"),
    ("226_gcf-b18-04-add04", "gcf_funding_requested", 9, 9_983_521.0, "9,983,521"),
    ("178_gcf-b21-10-add19", "gcf_funding_requested", 10, 15_500_000.0, "15.5 million"),
    ("225_gcf-b18-04-add05", "gcf_funding_requested", 7, 26_500_000.0, "26.5 million"),
    ("247_gcf-b14-07-add10", "gcf_funding_requested", 16, 132_000_000.0, "132"),
    # B.2(b) label spellings: 'Required GCF amount', 'Disaggregated GCF amount',
    # 'GCF financing to recipient' behind a bare 'B.' enumerator.
    ("182_gcf-b21-10-add12", "gcf_funding_requested", 50, 24_300_000.0, "24.3 M EUR"),
    ("270_gcf-b11-04-add04", "gcf_funding_requested", 8, 40_000_000.0, "40.0 million"),
    ("156_gcf-b24-02-add05", "gcf_funding_requested", 13, 23_709_782.0, "23,709782"),
    ("188_gcf-b21-10-add06", "gcf_funding_requested", 8, 22_000_000.0, "22 million"),
    # A.8 behind a mangled enumerator / behind the 'e.g.' guard / renamed
    # PIN RE-TAKEN 2026-08-26 (corpus cure), and this row has changed KIND, so
    # it is written out rather than edited in place. It used to read '180
    # million' behind a mangled 'A8' enumerator on p.8 — the only witness in
    # the corpus for that spelling. Ratified correction C105 had already shown
    # the digits were a misread (independent p.8: 'A.8 / B.2(b) — "150,000,000
    # USD"'), and the cure re-extracted the page: it now prints a clean
    # '## A.8. Total GCF funding requested / 150,000,000 USD'. So BOTH halves
    # of what this row witnessed were extraction artifacts — the digits and the
    # mangled enumerator — and the STRICT pass now reads the field (verified:
    # `build_document(..., fallback=False)` returns the same candidate). The
    # row is kept because the pin is still the corpus's own reading of a real
    # page, but it no longer witnesses the variant recognizer, and the ratified
    # canonical it now agrees with is unchanged at 150,000,000. Exactly the
    # precedent of the FP233 row at the bottom of this list.
    ("08_gcf-b42-02-add10-funding-proposal-package-fp268",
     "gcf_funding_requested", 8, 150_000_000.0, "150,000,000"),
    ("46_gcf-b38-02-add10-funding-proposal-package-fp230",
     "gcf_funding_requested", 5, 32_800_000.0, "32.8 million"),
    ("125_gcf-b27-02-add10-rev01", "gcf_funding_requested", 5, 256_480_000.0, "256.48"),
    ("19_gcf-b41-02-add05-funding-proposal-package-fp257",
     "gcf_funding_requested", 5, 75_623_754.0, "75,623,754"),
    # total_financing: 'Total project finance' (no -ing) and the emphasis-split
    # figure.
    ("249_gcf-b14-07-add08-rev01", "total_financing", 11, 1_538_500_000.0, "1538.5"),
    ("188_gcf-b21-10-add06", "total_financing", 8, 37_600_000.0, "37.6 million"),
    # PIN RE-TAKEN 2026-08-26 (serving-wave session). This row used to read
    # 69,830,370 under 'A7 Total funding required (GCF + co-financing)' and was
    # the FALLBACK_RULES entry's only witness in the corpus. The cross-extractor
    # check proved 69,830,370 is printed on no page of the PDF, and the page
    # re-extraction brought back a standard cover: 'A.7. Total financing (GCF +
    # co-finance) 79,690,370 USD'. So the "template variant" that rule was
    # written for was an extraction artifact, not a spelling any GCF cover uses
    # — no document in the corpus prints 'Total funding required' any more. The
    # rule is left in place (it is consulted only for a field the strict pass
    # leaves empty, so it can cost nothing) and this pin now records the
    # STRICT A.7 reading the document really has.
    ("43_gcf-b39-02-add08-funding-proposal-package-fp233",
     "total_financing", 5, 79_690_370.0, "79,690,370"),
])
def test_variant_recognizer_reads_the_document_it_was_written_for(
        stem, field, page_no, value, raw_bit):
    c = canon(_doc(stem)["facts"], field)
    _pin_or_skip(stem, field, c, bool(c) and c["page"] == page_no
                 and c["value"] == value and raw_bit in c["raw"])
    assert c is not None, f"{stem}: no canonical {field}"
    assert c["page"] == page_no
    assert c["value"] == value
    assert raw_bit in c["raw"]


@needs_corpus
@pytest.mark.parametrize("stem,field,page_no,text_bit", [
    # 'Program title' (one m) is the REDD+ RBP cover and the v1 'A.1.1 Project
    # / program title'; 'Projects/Programme title' and 'Project (programme)
    # title' are extraction spellings of the same label.
    ("100_gcf-b30-02-add07", "title", 3, "Upper Athi River"),
    ("175_gcf-b22-10-add02", "title", 3, "Amazon biome"),
    ("131_gcf-b27-02-add04", "title", 3, "Costa Rica REDD-plus"),
    ("165_gcf-b23-02-add04", "title", 3, "Ecuador REDD-plus"),
    ("271_gcf-b11-04-add03", "title", 5, "resilience of ecosystems"),
    ("247_gcf-b14-07-add10", "title", 6, "Universal Green Energy Access"),
    ("70_gcf-b35-02-add07-rev01_0", "title", 3, "Infrastructure Climate Resilient Fund"),
    # 'Country/cities', 'Country/countries', 'A.1.2 Country location'
    ("43_gcf-b39-02-add08-funding-proposal-package-fp233", "countries", 3, "Tajikistan"),
    ("68_gcf-b36-02-add02-rev01", "countries", 3, "Pakistan"),
    ("233_gcf-b16-07-add05", "countries", 5, "Morocco"),
    # 'Accredited Entities' (plural), 'Accrediting Entity' (extraction spelling)
    ("08_gcf-b42-02-add10-funding-proposal-package-fp268",
     "accredited_entity", 3, "Food and Agriculture Organization"),
    ("144_gcf-b26-02-add03", "accredited_entity", 3, "International Union for Conservation"),
])
def test_variant_text_recognizer_reads_its_document(stem, field, page_no, text_bit):
    c = canon(_doc(stem)["facts"], field)
    _pin_or_skip(stem, field, c, bool(c) and c["page"] == page_no
                 and text_bit in c["raw"])
    assert c is not None, f"{stem}: no canonical {field}"
    assert c["page"] == page_no
    assert text_bit in c["raw"]


# --------------------------------------------------------------------------
# the guards that keep the relaxed reading honest
# --------------------------------------------------------------------------

def test_loose_mode_stops_at_the_co_financing_table_header():
    """'Total requested (a)+(b)+(c)... | Senior Loans | 99,596,000 | ADB' is the
    ADB loan under the CO-FINANCING table, not the GCF request. Only the
    co-financing and GCF-to-AE tables carry a 'Name of Institution' column, so
    the window ends there and the block yields nothing rather than a lie."""
    text = page(12, "#### Total requested (a) + (b) + (c)\n"
                    "| Financial Instrument | Amount | Currency | Name of Institutions |\n"
                    "|---|---|---|---|\n"
                    "| Senior Loans | 99,596,000 | USD | ADB |")
    assert "gcf_funding_requested" not in B.build_document("99_x", text)["facts"]


def test_loose_mode_binds_a_unit_word_across_markdown_emphasis():
    """'| **37.6** million USD ($) |' is 37.6 million: emphasis is markup, and
    with it in the way the figure reads as sub-$10k table furniture and is
    dropped."""
    text = page(8, "| Total project financing* | **37.6** million USD ($) |")
    c = canon(B.build_document("99_x", text)["facts"], "total_financing")
    assert c["value"] == 37_600_000 and c["raw"] == "37.6 million USD"


def test_loose_mode_e_g_guard_needs_a_real_dot():
    """The shipped guard is dot-optional, so 'by the GCF' reads as 'e.g.' and
    the A.8 figure behind it is dropped (FP230). A genuine 'e.g.' is still
    rejected."""
    assert B.read_amounts("by the GCF\n\n32.8 million Euros", 1, loose=True)
    assert not B.read_amounts("e.g. 32.8 million Euros", 1, loose=True)
    assert not B.read_amounts("by the GCF\n\n32.8 million Euros", 1)      # shipped guard


def test_loose_mode_rejects_a_per_unit_cost():
    """'US$ 1,358.0/tonne' under 'Total Programme financing' is a unit cost."""
    assert not B.read_amounts("| US$ 1,358.0/tonne |", 1, loose=True)
    assert B.read_amounts("| US$ 1,538.5 million |", 1, loose=True)


def test_a_total_window_does_not_swallow_the_gcf_request_line():
    """FP233 prints the request under the A.7 TOTAL label. The two rows are
    different fields, and the request is not a second reading of the total."""
    text = page(5, "## A7 Total funding required (GCF + co-financing)\n\n"
                   "- Total amount requested: 69,830,370 USD\n"
                   "- Total GCF funding requested: 39,000,000 USD\n")
    cands = B.build_document("99_x", text)["facts"]["total_financing"]
    assert [c["value"] for c in cands] == [69_830_370]


# --------------------------------------------------------------------------
# era families
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,era", [
    ("## A.7 Total financing (GCF + co-finance)\n30,000,000 USD",
     "A5-A14 block (FP template v2/v3)"),
    ("#### A.1.1 Project / programme title\nX\n#### A.1.5 Accredited entity\nY",
     "A.1.x block (FP template v1)"),
    # the RBP pilot is its own template: no A.7, no A.8, and a cover field no
    # other template has
    ("## REDD-plus results based payments\n**Programme Title:** X",
     "REDD+ RBP block (RBP pilot template v1.0)"),
    ("**REDD-plus entity/focal point:** Ms X\n**Accredited Entity:** UNDP",
     "REDD+ RBP block (RBP pilot template v1.0)"),
    # numbering the extraction invented, labels it did not
    ("### A.9 Project size\nMedium\n\nA.1. Implementation period\n7 years\n\n"
     "A.2. Total lifespan\n20 years",
     "A5-A14 block (FP template v2/v3, variant numbering)"),
    ("#### A3.1 Project (programme) title\nX\n#### A3.5 Accredited entity\nY\n"
     "##### A1.7 Project size category\nSmall",
     "A.1.x block (FP template v1, variant numbering)"),
    ("# Consideration of funding proposals - Addendum I\n\n"
     "The funding proposal of FP082 has been withdrawn from the twenty-second "
     "meeting of the Board by the accredited entity.",
     "board notice (not a proposal template)"),
    ("## Funding Proposal Summary for FP005\n\n# Brief Programme Information",
     "funding proposal summary (pre-template, B.11 era)"),
    ("Nothing here names a template at all.", "unrecognized template"),
])
def test_era_families(text, era):
    assert B.era_of(text) == era


def test_era_families_are_only_ever_a_split_of_unrecognized():
    """A document the two shipped patterns name keeps that name: the new
    branches are consulted after them, never instead of them."""
    modern = ("## A.7 Total financing (GCF + co-finance)\n30,000,000 USD\n"
              "## REDD-plus results based payments\n"
              "### A.9 Project size\nA.1. Implementation period\nA.2. Total lifespan")
    assert B.era_of(modern) == "A5-A14 block (FP template v2/v3)"
    old = ("#### A.1.5 Accredited entity\nY\n"
           "## REDD+ results based payments\nhas been withdrawn by the accredited entity")
    assert B.era_of(old) == "A.1.x block (FP template v1)"


@needs_corpus
@pytest.mark.parametrize("stem,era", [
    ("145_gcf-b26-02-add02", "REDD+ RBP block (RBP pilot template v1.0)"),
    ("155_gcf-b24-02-add06", "REDD+ RBP block (RBP pilot template v1.0)"),
    ("113_gcf-b28-02-add09", "A5-A14 block (FP template v2/v3, variant numbering)"),
    ("61_gcf-b37-02-add05-funding-proposal-package-fp214",
     "A5-A14 block (FP template v2/v3, variant numbering)"),
    ("264_gcf-b13-16-add02", "A.1.x block (FP template v1, variant numbering)"),
    ("193_gcf-b22-10-add01-rev01", "board notice (not a proposal template)"),
    ("268_gcf-b11-04-add06", "funding proposal summary (pre-template, B.11 era)"),
    # FP274: qwen deterministically renders the cover WITHOUT the A.n
    # prefixes and the A.8 row (byte-identical across the original
    # extraction, the corpus cure, and a retry), while the independent
    # pymupdf extraction proves the PDF prints 'A.7. ... A.8. ...'.
    # A model-rendering limitation, not corpus damage: the era honestly
    # reads variant-numbering for the corpus AS SERVED, and the ratified
    # corrections protect every canonical regardless (C83 carried).
    ("02_gcf-b42-02-add16-funding-proposal-package-fp274",
     "A5-A14 block (FP template v2/v3, variant numbering)"),
    ("266_gcf-b11-04-add08", "A.1.x block (FP template v1)"),
])
def test_era_family_of_a_real_document(stem, era):
    assert _doc(stem, fallback=False)["coverage"]["era"] == era


@needs_corpus
def test_no_document_in_the_corpus_is_left_without_a_template_family():
    """The Phase 3 gate for the recognizer half: 'unrecognized template' is a
    hole in the parser, and after the variant families there is none."""
    stems = sorted(p.stem for p in EXTRACTED.glob("*.md"))
    assert stems, "no extracted documents"
    unknown = [s for s in stems
               if B.era_of((EXTRACTED / f"{s}.md").read_text(encoding="utf-8",
                                                             errors="replace"))
               == "unrecognized template"]
    assert unknown == []


# --------------------------------------------------------------------------
# H19 watch guard
# --------------------------------------------------------------------------

def _fp_index(documents: dict) -> dict:
    idx = {}
    for stem, row in documents.items():
        idx.setdefault(row.get("fp"), []).append(stem)
    return idx


@pytest.mark.skipif(not REG_V2.exists(), reason="registry_v2.json absent")
def test_h19_exactly_one_document_per_fp():
    """H19 watch guard. Everything downstream — the FP resolver, the registry
    note, `facts_by_fp`, every single-identifier gold case — assumes an FP
    number names ONE document. The corpus satisfies that today by accident of
    what was downloaded, not by construction. The day a second package for the
    same FP is added (a Rev.02, a re-submission, a second addendum), this test
    fails and the assumption gets revisited BEFORE the resolver starts picking
    one of two documents silently."""
    docs = json.loads(REG_V2.read_text(encoding="utf-8"))["documents"]
    dupes = {fp: stems for fp, stems in _fp_index(docs).items()
             if fp is not None and len(stems) > 1}
    assert dupes == {}, (
        "a second document now shares an FP number; the single-document "
        f"assumption no longer holds: {dupes}")
    missing = sorted(s for s, r in docs.items() if r.get("fp") is None)
    assert missing == [], f"documents with no FP number: {missing}"


def test_h19_guard_fires_on_a_second_document_for_one_fp():
    """The guard's own alarm, so it cannot rot into a tautology."""
    two = {"a_fp274": {"fp": 274}, "b_fp274_rev02": {"fp": 274}, "c_fp100": {"fp": 100}}
    dupes = {fp: s for fp, s in _fp_index(two).items() if fp is not None and len(s) > 1}
    assert dupes == {274: ["a_fp274", "b_fp274_rev02"]}


# ==========================================================================
# Phase 3 — the ratified data decisions of 2026-08-26
#
# Two data files, both consumed by the builder so that data/registry_v2.json
# is never hand-edited: 58 corrections the adjudication proved wrong, and 51
# (document, field) pairs confirmed ABSENT plus the corpus-level finding that
# no document prints its own GCF Board approval date.
#
# The corpus is being re-extracted page by page while these run, so nothing
# below asserts the content of a page on that worklist: the mechanism tests
# are synthetic, and the two corpus tests name documents no re-extraction
# touches.
# ==========================================================================

CORRECTIONS = Path(__file__).resolve().parents[1] / "data" / "registry_corrections.json"
ABSENCES = Path(__file__).resolve().parents[1] / "data" / "registry_absences.json"

needs_decisions = pytest.mark.skipif(
    not (CORRECTIONS.exists() and ABSENCES.exists()),
    reason="ratified decision files absent")


def _cand(raw, value=None, page=5, section="A.7", status="canonical", **kw):
    return {"raw": raw, "value": value, "currency": "USD", "unit": None,
            "page": page, "section": section, "status": status, **kw}


def _entry(**kw):
    base = {"id": "T01", "doc_id": "D", "field": "total_financing",
            "layer": "fact-canonical", "action": "correct-to",
            "wrong": {"raw": "1,000 USD", "page": 5},
            "corrected": {"raw": "1,000,000 USD", "value": 1_000_000.0,
                          "currency": "USD", "unit": None, "page": 40,
                          "section": None, "quote": 'p40: "Total: 1,000,000"'},
            "ratified": "owner, 2026-08-26"}
    base.update(kw)
    return base


# --------------------------------------------------------------------------
# the two new date fields
# --------------------------------------------------------------------------

def test_a_date_printed_beside_its_label_is_read():
    facts = facts_of(page(3, "## Date of first submission: 2019/07/04\n"
                             "## Date of current submission: 2021/02/17"))
    got = {c["raw"] for c in facts["date_of_submission"]}
    assert got == {"2019/07/04", "2021/02/17"}


def test_the_printed_label_rides_along_in_the_section():
    """Two submission dates under two labels: without the label the store
    would hold two dates and no way to say which is which."""
    facts = facts_of(page(3, "## Date of first submission: 2019/07/04\n"
                             "## Date of current submission: 2021/02/17"))
    by_raw = {c["raw"]: c["section"] for c in facts["date_of_submission"]}
    assert by_raw["2019/07/04"].endswith("Date of first submission")
    assert by_raw["2021/02/17"].endswith("Date of current submission")


def test_a_date_on_the_line_below_its_label_is_read():
    """The v1 cover prints 'A.1.9 Date of submission' and the date beneath it."""
    facts = facts_of(page(4, "#### A.1.9 Date of submission\n\n17 February 2020\n\n"
                             "#### A.1.10 Project contact details"))
    c = canon(facts, "date_of_submission")
    assert c["raw"] == "17 February 2020" and c["page"] == 4


def test_a_date_is_never_given_a_value():
    """'[2020/13/05]' has no month 13. The print is the fact; parsing it into a
    real date would be inventing one."""
    facts = facts_of(page(3, "| Date of first submission: | [2020/13/05] |"))
    c = canon(facts, "date_of_submission")
    assert c["raw"] == "[2020/13/05]" and c["value"] is None and c["unit"] is None


def test_not_applicable_under_a_date_label_is_not_a_date():
    facts = facts_of(page(3, "| Expected approval from accredited entity's Board "
                             "(if applicable) | N/A |"))
    assert "ae_board_approval_date" not in facts


def test_an_empty_date_cell_never_borrows_the_next_row_s_date():
    """The defect this guard exists for: an empty AE-board cell followed by the
    implementation-start row published a START date as an approval date."""
    facts = facts_of(page(3,
        "| Expected approval from accredited entity's Board (if applicable) | TBD |\n"
        "|---|---|\n"
        "| Expected financial close (if applicable) | TBD |\n"
        "| Estimated implementation start and end date | Start: 01/12/2020<br>End: 31/12/2026 |"))
    assert "ae_board_approval_date" not in facts


def test_a_date_about_the_GCF_board_is_not_the_AE_s_own_approval():
    """H9: no document prints its own GCF Board approval date, so a sentence
    under A.13 that talks about the GCF Board's decision is about something
    else entirely and may not be filed as the AE's internal approval."""
    facts = facts_of(page(6, "A.13. Expected date of AE internal approval\n\n"
                             "IDB approval of the program will follow GCF board "
                             "approval on 15/03/2022"))
    assert "ae_board_approval_date" not in facts


def test_a_date_buried_in_the_template_s_explanatory_sentence_is_quoted_alone():
    """The A.13 explanation runs long and ends with the date; clipping the line
    would publish a fact whose own raw does not contain the date."""
    facts = facts_of(page(6, "A.13. Expected date of AE internal approval\n\n"
                             "This is the date that the Accredited Entity "
                             "observed/achieved to be able to implement the "
                             "project/ activity (if applicable): 6/15/2020"))
    assert canon(facts, "ae_board_approval_date")["raw"] == "6/15/2020"


def test_the_date_fields_are_not_reachable_by_an_approval_shaped_question():
    """The whole point of naming them honestly. Neither field is this
    proposal's GCF Board approval date — H9 found 0 of 273 documents printing
    one — so neither may be served by the field service or the planner until
    somebody decides what an approval-shaped ask should do. This test fails the
    moment either is wired up."""
    from gcf_qna.rag import planner
    new = {"date_of_submission", "ae_board_approval_date"}
    assert not (new & set(registry._SERVED_LABELS))
    assert not (new & set(getattr(planner, "FIELD_ORDER", ())))
    assert not (new & set(getattr(planner, "FIELD_KEYWORDS", {}) or {}))


# --------------------------------------------------------------------------
# the third pass cannot touch what the first two read
# --------------------------------------------------------------------------

def test_the_extra_pass_never_touches_a_field_the_other_passes_read():
    pages = B.split_pages(page(5, "## A.1.5 Accredited entity: Some Bank\n"
                                  "## A.1.5 Implementing entity: Other Body\n"
                                  "## Date of first submission: 2019/07/04"))
    without = B.extract_candidates(pages, era="A.1.x block (FP template v1)",
                                   extra_rules=False)
    with_ = B.extract_candidates(pages, era="A.1.x block (FP template v1)")
    assert with_["accredited_entity"] == without["accredited_entity"]
    assert "date_of_submission" not in without


def test_the_implementing_entity_slot_is_read_only_in_the_v1_eras():
    """OWNER RATIFICATION 2026-08-26: in the earliest template the slot that
    later reads 'Accredited entity' is printed 'Implementing entity'. That is a
    decision about what the field MEANS, so it is era-gated to the v1 families
    and never renames a modern document's entity."""
    pages = B.split_pages(page(5, "## A.1.5 Implementing entity: PROFONANPE"))
    v1 = B.extract_candidates(pages, era="A.1.x block (FP template v1)")
    modern = B.extract_candidates(pages, era="A5-A14 block (FP template v2/v3)")
    assert v1["accredited_entity"][0]["raw"] == "PROFONANPE"
    assert "accredited_entity" not in modern


def test_the_implementing_entity_mapping_keeps_the_printed_label():
    pages = B.split_pages(page(5, "## A.1.5 Implementing entity: PROFONANPE"))
    c = B.extract_candidates(pages, era="A.1.x block (FP template v1)")["accredited_entity"][0]
    assert c["section"] == "A.1.5 Implementing entity"


@needs_corpus
def test_the_one_document_the_implementing_entity_ratification_is_for():
    """273_gcf-b11-04-add01 is the only document in the corpus that turns on
    it: 'A.1.5 Implementing entity' on p.5, and no accredited entity anywhere
    else. Its page is not on the re-extraction worklist."""
    doc = _doc("273_gcf-b11-04-add01")
    c = canon(doc["facts"], "accredited_entity")
    assert c is not None and c["page"] == 5
    assert c["section"].endswith("Implementing entity")
    assert "accredited_entity" not in doc["coverage"]["core_missing"]
    assert doc["meta"]["mapped_labels"][0]["printed_label"] == "Implementing entity"


# --------------------------------------------------------------------------
# corrections
# --------------------------------------------------------------------------

def test_a_correction_overrides_the_candidate_and_keeps_where_it_came_from():
    facts = {"total_financing": [_cand("1,000 USD", 1000.0)]}
    dec = B.Decisions([_entry(doc_id="D")])
    recs = B.apply_fact_corrections("D", facts, dec)
    c = facts["total_financing"][0]
    assert c["value"] == 1_000_000.0 and c["page"] == 40 and c["status"] == "canonical"
    assert c["corrected"] is True
    assert c["corrected_from"] == {"raw": "1,000 USD", "value": 1000.0,
                                   "currency": "USD", "unit": None, "page": 5,
                                   "section": "A.7", "status": "canonical"}
    assert recs[0]["quote"].startswith("p40:") and recs[0]["ratified"] == "owner, 2026-08-26"


def test_a_corrected_candidate_does_not_claim_a_section_the_page_may_not_print():
    facts = {"total_financing": [_cand("1,000 USD", 1000.0)]}
    B.apply_fact_corrections("D", facts, B.Decisions([_entry(doc_id="D")]))
    assert facts["total_financing"][0]["section"] == "corrected"


def test_a_value_fix_keeps_the_print_and_moves_only_the_number():
    facts = {"total_financing": [_cand("$141.390 thousand USD", 141_390.0, page=7,
                                       section="rule:B.2(a)")]}
    dec = B.Decisions([_entry(action="value-fix", doc_id="D",
                              wrong={"raw": "$141.390 thousand USD", "page": 7},
                              corrected={"raw": "$141.390 thousand USD",
                                         "value": 141_390_000.0, "currency": "USD",
                                         "unit": "thousand", "page": 7,
                                         "section": "rule:B.2(a)", "quote": "p7"})])
    B.apply_fact_corrections("D", facts, dec)
    c = facts["total_financing"][0]
    assert (c["raw"], c["page"], c["section"]) == ("$141.390 thousand USD", 7, "rule:B.2(a)")
    assert c["value"] == 141_390_000.0 and c["corrected"] is True


def test_a_reclassified_candidate_moves_to_the_field_it_belongs_to():
    """FP139 in miniature: a label-shifted duplicate A-block put the GCF request
    under A.7. Moving it fills the missing field AND dissolves the conflict."""
    facts = {"total_financing": [_cand("41,185,114 USD", 41_185_114.0),
                                 _cand("25,645,114 USD", 25_645_114.0,
                                       status="conflicting")]}
    dec = B.Decisions([_entry(action="reclassify", doc_id="D", layer="fact-conflicting",
                              to_field="gcf_funding_requested", to_status="canonical",
                              wrong={"raw": "25,645,114 USD", "page": 5},
                              corrected={"raw": "25,645,114 USD", "value": 25_645_114.0,
                                         "currency": "USD", "unit": None, "page": 5,
                                         "section": None, "quote": "p32 C.1"})])
    B.apply_fact_corrections("D", facts, dec)
    assert [c["status"] for c in facts["total_financing"]] == ["canonical"]
    moved = facts["gcf_funding_requested"][0]
    assert moved["status"] == "canonical" and moved["value"] == 25_645_114.0
    assert moved["reclassified_from"] == "total_financing"


def test_a_reclassification_does_not_displace_a_different_reading():
    """It outranks an extracted candidate that prints the SAME figure, and only
    that: a canonical reading a different number keeps its place, loudly, and
    the ratified print lands beside it as the disagreement it is."""
    facts = {"total_financing": [_cand("25,645,114 USD", 25_645_114.0,
                                       status="conflicting")],
             "gcf_funding_requested": [_cand("9,000,000 USD", 9_000_000.0, page=32)]}
    dec = B.Decisions([_entry(action="reclassify", doc_id="D", layer="fact-conflicting",
                              to_field="gcf_funding_requested", to_status="canonical",
                              wrong={"raw": "25,645,114 USD", "page": 5},
                              corrected={"raw": "25,645,114 USD", "value": 25_645_114.0,
                                         "currency": "USD", "unit": None, "page": 5,
                                         "section": None, "quote": "q"})])
    B.apply_fact_corrections("D", facts, dec)
    kept = {c["raw"]: c["status"] for c in facts["gcf_funding_requested"]}
    assert kept == {"9,000,000 USD": "canonical", "25,645,114 USD": "conflicting"}
    assert dec.alarms and "filed as supporting" in dec.alarms[0]


def test_a_promotion_elects_the_page_s_own_print_and_drops_the_wrong_one():
    """FP240: an A.5 money row was the 'mitigation outcome'; the tonnage on
    p.61 was filed as the conflicting one. Money against tonnes is not two
    readings of an outcome, so the wrong print is dropped, not demoted."""
    facts = {"mitigation_outcome": [_cand("$214,219 million USD", 214_219.0, section="A.5"),
                                    _cand("1,639,681 | 327,936", 1_639_681.0, page=61,
                                          section="rule:A.5", status="conflicting")]}
    dec = B.Decisions([_entry(action="promote", doc_id="D", field="mitigation_outcome",
                              wrong={"raw": "$214,219 million USD", "page": 5},
                              promote={"raw": "1,639,681 | 327,936", "page": 61},
                              corrected={"raw": "1,639,681 | 327,936", "value": 1_639_681.0,
                                         "currency": None, "unit": None, "page": 61,
                                         "section": "rule:A.5", "quote": "p61"})])
    B.apply_fact_corrections("D", facts, dec)
    assert [(c["raw"], c["status"]) for c in facts["mitigation_outcome"]] == [
        ("1,639,681 | 327,936", "canonical")]
    assert facts["mitigation_outcome"][0]["corrected"] is True


def test_a_confirmed_absence_correction_removes_the_field_entirely():
    facts = {"title": [_cand("Consideration of funding proposals - Addendum I",
                             page=1, section="llm", status="supporting")]}
    dec = B.Decisions([_entry(action="confirm-absence", doc_id="D", field="title",
                              layer="fact-supporting", corrected=None,
                              wrong={"raw": "Consideration of funding proposals - "
                                            "Addendum I", "page": 1})])
    B.apply_fact_corrections("D", facts, dec)
    assert "title" not in facts


def test_a_row_with_no_defensible_replacement_loses_canonical_rather_than_lying():
    """Two of the 58 are 'WRONG, and no print in the document settles it'. The
    print stays and says it is disputed; nothing canonical is asserted."""
    facts = {"gcf_funding_requested": [_cand("$2,793,000 USD", 2_793_000.0, page=6)]}
    dec = B.Decisions([_entry(action="re-extract", doc_id="D",
                              field="gcf_funding_requested", corrected=None,
                              adjudication_note="candidates 2,793,000 / 12,793,000",
                              wrong={"raw": "$2,793,000 USD", "page": 6})])
    B.apply_fact_corrections("D", facts, dec)
    c = facts["gcf_funding_requested"][0]
    assert c["status"] == "supporting" and c["disputed"] is True
    assert B._canon_of(facts, "gcf_funding_requested") is None


def test_a_correction_whose_target_has_moved_is_not_applied_and_is_shouted_about():
    """The corpus is re-extracted underneath the build. A ratified correction
    that no longer matches any candidate must never be WRITTEN ONTO whatever is
    there instead.

    RE-TAKEN 2026-08-26 (corpus-cure round). The safety property above is
    unchanged and still asserted: the candidate the row does not name keeps its
    own print and never acquires `corrected`. What changed is the row's other
    half. Reporting the row NOT APPLIED used to mean DISCARDING the ratified
    figure, and after the cure that cost 27 ratified figures across 25 fields —
    the cure leaving the store worse off than the corrections alone had. The
    figure is now carried forward as its own candidate, marked `carried_forward`
    so no reader mistakes it for a print the parser found, and the disagreement
    is alarmed as loudly as the miss it replaces."""
    facts = {"total_financing": [_cand("2,000 USD", 2000.0)]}
    dec = B.Decisions([_entry(doc_id="D")])
    B.apply_fact_corrections("D", facts, dec)
    # the candidate the row does NOT name is untouched — the whole safety property
    assert facts["total_financing"][0]["raw"] == "2,000 USD"
    assert "corrected" not in facts["total_financing"][0]
    # ... and it is no longer what the store publishes, because the ratified
    # figure it disagrees with is now here beside it
    assert facts["total_financing"][0]["status"] == "supporting"
    carried = facts["total_financing"][1]
    assert carried["value"] == 1_000_000.0 and carried["status"] == "canonical"
    assert carried["carried_forward"] is True and carried["corrected_from"] is None
    # not applied, not lost: its own ledger, and still an alarm
    assert not dec.applied and not dec.unapplied
    assert [c["id"] for c in dec.carried] == ["T01"]
    assert any("CARRIED FORWARD" in a and "2,000 USD" in a for a in dec.alarms)


def test_a_top_level_correction_rewrites_the_flat_field():
    row = {"gcf_financing": "USD 150 million", "facts": {}}
    dec = B.Decisions([_entry(doc_id="D", field="gcf_financing", layer="top-level",
                              wrong={"raw": "USD 150 million", "page": 6},
                              corrected={"raw": "USD 96,452,228", "value": 96_452_228.0,
                                         "currency": "USD", "unit": None, "page": 103,
                                         "section": None, "quote": "p103"})])
    recs = B.apply_top_level_corrections("D", row, dec)
    assert row["gcf_financing"] == "USD 96,452,228"
    assert recs[0]["from"] == "USD 150 million"


def test_a_pending_reextraction_prefers_the_fresh_page_when_the_two_agree():
    """FP226's cover was garbled and is being re-extracted. When the fresh parse
    says what the correction says, the PAGE is published, not the table."""
    row = {"gcf_financing": "78.32 million Euros",
           "facts": {"gcf_funding_requested": [
               {"raw": "40.79 million Eur", "value": 40_790_000.0, "currency": "EUR",
                "unit": "million", "page": 5, "section": "A.8", "status": "canonical"}]}}
    dec = B.Decisions([_entry(doc_id="D", field="gcf_financing", layer="top-level",
                              pending_reextraction=True,
                              wrong={"raw": "78.32 million Euros", "page": 5},
                              corrected={"raw": "EUR 40.79 million", "value": 40_790_000.0,
                                         "currency": "EUR", "unit": "million", "page": 138,
                                         "section": None, "quote": "p138"})])
    recs = B.apply_top_level_corrections("D", row, dec)
    assert row["gcf_financing"] == "40.79 million Eur"
    assert "re-extraction agrees" in recs[0]["resolved_by"]
    assert not dec.alarms


def test_a_pending_reextraction_that_disagrees_is_flagged_loudly():
    row = {"gcf_financing": "78.32 million Euros",
           "facts": {"gcf_funding_requested": [
               {"raw": "12.00 million Eur", "value": 12_000_000.0, "currency": "EUR",
                "unit": "million", "page": 5, "section": "A.8", "status": "canonical"}]}}
    dec = B.Decisions([_entry(doc_id="D", field="gcf_financing", layer="top-level",
                              pending_reextraction=True,
                              wrong={"raw": "78.32 million Euros", "page": 5},
                              corrected={"raw": "EUR 40.79 million", "value": 40_790_000.0,
                                         "currency": "EUR", "unit": "million", "page": 138,
                                         "section": None, "quote": "p138"})])
    recs = B.apply_top_level_corrections("D", row, dec)
    assert row["gcf_financing"] == "EUR 40.79 million"         # the ratified value stands
    assert any("PENDING RE-EXTRACTION DISAGREES" in a for a in dec.alarms)
    assert recs[0]["reextraction_disagreement"]["value"] == 12_000_000.0


# --------------------------------------------------------------------------
# confirmed absences
# --------------------------------------------------------------------------

def test_a_confirmed_absence_is_published_with_the_pages_that_were_read():
    dec = B.Decisions(absences=[{"doc_id": "D", "field": "total_financing",
                                 "pages_checked": [1, 16], "evidence": "no A.7 block",
                                 "group": "REDD+ RBP financing", "status": "ratified",
                                 "ratified": "owner, 2026-08-26"}])
    got = B.absence_meta("D", {}, dec)
    assert got["total_financing"]["pages_checked"] == [1, 16]
    assert got["total_financing"]["ratified"] == "owner, 2026-08-26"


def test_an_absence_is_never_published_over_a_value_the_build_found():
    """Absence-as-fact is a claim about the document. If the parser DID read the
    field, the claim is wrong and the loud path is the only honest one."""
    dec = B.Decisions(absences=[{"doc_id": "D", "field": "total_financing",
                                 "pages_checked": [1], "evidence": "e",
                                 "status": "ratified"}])
    got = B.absence_meta("D", {"total_financing": [_cand("1,000,000 USD", 1e6)]}, dec)
    assert got == {}
    assert any("ABSENCE CONTRADICTED" in a for a in dec.alarms)


def test_a_superseded_absence_is_kept_in_the_file_and_not_published():
    """The one row where two ratifications meet: 273_gcf-b11-04-add01's
    accredited entity was ratified ABSENT and then ratified as the 'Implementing
    entity' slot. The row stays on the record, marked, and nothing is served."""
    dec = B.Decisions(absences=[{"doc_id": "D", "field": "accredited_entity",
                                 "pages_checked": [5, 6], "evidence": "e",
                                 "status": "superseded",
                                 "superseded_by": {"decision": "ratification (3)"}}])
    assert B.absence_meta("D", {}, dec) == {}
    assert dec.absences_skipped[0]["field"] == "accredited_entity"
    assert not dec.alarms


# --------------------------------------------------------------------------
# the files themselves
# --------------------------------------------------------------------------

@needs_decisions
def test_the_ratified_files_say_what_they_are():
    corr = json.loads(CORRECTIONS.read_text(encoding="utf-8"))
    absc = json.loads(ABSENCES.read_text(encoding="utf-8"))
    # 58 wrong + 4 phase-3 riders + the 12 serving-wave rows
    # + the 117 cross-check-session rows (the 114 STORE-WRONG rows of the
    # cross-extractor census, two further add-candidate rows where the ratified
    # action names a second printed row, and the one AMBIGUOUS row adjudicated
    # wrong either way with no ratified replacement)
    assert corr["count"] == len(corr["corrections"]) == 191
    xc = [e for e in corr["corrections"]
          if e["ratified"] == "owner, 2026-08-26 (cross-check session)"]
    assert len(xc) == 117
    assert Counter(e["action"] for e in xc) == {
        "correct-to": 112, "add-candidate": 3, "drop-candidate": 1, "re-extract": 1}
    # the eight STORE-RIGHT rows of that census were ARM false positives and get
    # no correction at all — the arm's two defects were fixed instead
    assert not [e for e in xc if e["row_ref"]["verdict"] == "STORE-RIGHT"]
    riders = [e for e in corr["corrections"] if "rider" in e["row_ref"]]
    assert len(riders) == 4
    assert {e["ratified"] for e in riders} == {"owner, 2026-08-26 (rider session)"}
    assert {e["row_ref"]["verdict"] for e in riders} == {"CONFIRMED"}
    # each session names the adjudication it came from, and nothing else does
    assert [s["session"] for s in corr["sources"]] == [
        "owner, 2026-08-26", "owner, 2026-08-26 (serving-wave session)",
        "owner, 2026-08-26 (cross-check session)"]
    assert absc["count"] == len(absc["absences"]) == 51
    assert absc["superseded"] == 1                     # see the test above
    assert len(absc["corpus_level"]) == 1
    assert absc["corpus_level"][0]["field"] == "gcf_board_approval_date"
    assert absc["corpus_level"][0]["evidence"]["true_positives_after_reading"] == 0
    for e in corr["corrections"] + absc["absences"] + absc["corpus_level"]:
        assert e["ratified"].startswith("owner, 2026-08-26")
    assert set(corr["sessions"]) == {e["ratified"] for e in corr["corrections"]}


@needs_decisions
def test_every_correction_carries_its_adjudication_row_and_its_quote():
    for e in json.loads(CORRECTIONS.read_text(encoding="utf-8"))["corrections"]:
        assert e["row_ref"]["pointer"].startswith("rows[")
        assert e["row_ref"]["file"].endswith(("phase3_adjudication.json",
                                              "serving_wave_adjudication.json",
                                              "cross_check_adjudication.json"))
        assert e["action"] in {"correct-to", "value-fix", "reclassify", "promote",
                               "confirm-absence", "re-extract", "add-candidate",
                               "drop-candidate"}
        if e["action"] == "drop-candidate":
            # the one action that deletes a print and puts nothing in its place:
            # it has to carry the ground it was ratified on and the proof
            assert e["wrong"]["raw"] is not None, e["id"]
            assert e["corrected"] is None, e["id"]
            assert e["dropped"]["ground"] in B.DROP_GROUNDS, e["id"]
            assert e["dropped"]["evidence"] and e["dropped"]["quote"], e["id"]
            assert e["dropped"]["searched"], e["id"]
            if e["dropped"]["ground"] == "label-bleed":
                assert e["dropped"]["belongs_to"], e["id"]
                assert e["dropped"]["belongs_to_raw"], e["id"]
        elif e["action"] == "add-candidate":
            # nothing is wrong on an add row: it carries a print, not a fix
            assert e["wrong"] is None, e["id"]
            assert e["corrected"] is None, e["id"]
            assert e["add"]["quote"] and e["add"]["page"], e["id"]
            assert e["add"]["status"] in {"supporting", "conflicting"}, e["id"]
            if e["add"].get("derived"):
                assert e["add"]["derived_from"], e["id"]
        else:
            assert e["wrong"]["raw"] is not None, e["id"]
            if e["action"] in {"confirm-absence", "re-extract"}:
                assert e["corrected"] is None
            else:
                assert e["corrected"]["quote"], e["id"]
                assert e["corrected"]["page"], e["id"]


@needs_decisions
@needs_corpus
def test_every_ratified_decision_names_a_document_that_exists():
    stems = {p.stem for p in EXTRACTED.glob("*.md")}
    corr = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["corrections"]
    absc = json.loads(ABSENCES.read_text(encoding="utf-8"))["absences"]
    unknown = sorted({e["doc_id"] for e in corr + absc} - stems)
    assert unknown == []


@needs_decisions
def test_the_two_files_agree_about_the_three_absences_they_share():
    """Three of the 58 corrections say 'drop the value, the field is absent'.
    Each must be recorded as an absence too, or the store would drop a value
    and say nothing about why."""
    corr = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["corrections"]
    absc = json.loads(ABSENCES.read_text(encoding="utf-8"))["absences"]
    pairs = {(a["doc_id"], a["field"]) for a in absc}
    dropped = [(e["doc_id"], e["field"]) for e in corr if e["action"] == "confirm-absence"]
    assert len(dropped) == 3
    assert set(dropped) <= pairs


@needs_decisions
def test_the_candidate_schema_is_stable_and_the_extra_keys_are_opt_in():
    """The correction markers ride on corrected candidates only; every other
    candidate keeps exactly the seven published keys."""
    keys = {"raw", "value", "currency", "unit", "page", "section", "status"}
    extra = {"corrected", "corrected_from", "reclassified_from", "disputed", "dispute",
             "added", "derived", "derived_from", "cross_check"}
    facts = {"total_financing": [_cand("1,000 USD", 1000.0), _cand("9 USD", 9.0,
                                                                   status="supporting")]}
    B.apply_fact_corrections("D", facts, B.Decisions([_entry(doc_id="D")]))
    for c in facts["total_financing"]:
        assert set(c) <= keys | extra
        assert set(c) >= keys
    assert set(facts["total_financing"][1]) == keys


# --------------------------------------------------------------------------
# the statuses a correction leaves behind
# --------------------------------------------------------------------------

def test_a_conflict_with_a_refuted_figure_stops_being_a_conflict():
    """FP169's page-46 print was 'conflicting' because it disagreed with the
    A.7 figure the adjudication then refuted — and the correction ADOPTED that
    very print. Left alone the note would have warned that 19,710,637
    contradicts 19,710,637."""
    facts = {"total_financing": [_cand("$15,716,621 USD", 15_716_621.0),
                                 _cand("19,710,637", 19_710_637.0, page=46,
                                       section="A.7", status="conflicting")]}
    dec = B.Decisions([_entry(doc_id="D",
                              wrong={"raw": "$15,716,621 USD", "page": 5},
                              corrected={"raw": "19,710,637 USD", "value": 19_710_637.0,
                                         "currency": "USD", "unit": None, "page": 46,
                                         "section": None, "quote": "p46"})])
    B.apply_fact_corrections("D", facts, dec)
    assert [c["status"] for c in facts["total_financing"]] == ["canonical", "supporting"]


def test_a_print_that_disagrees_with_the_CORRECTED_figure_still_conflicts():
    """The other direction: re-marking is not a way of quieting conflicts."""
    facts = {"total_financing": [_cand("$15,716,621 USD", 15_716_621.0),
                                 _cand("22,000,000 USD", 22_000_000.0, page=46,
                                       section="C.1", status="supporting")]}
    dec = B.Decisions([_entry(doc_id="D",
                              wrong={"raw": "$15,716,621 USD", "page": 5},
                              corrected={"raw": "19,710,637 USD", "value": 19_710_637.0,
                                         "currency": "USD", "unit": None, "page": 46,
                                         "section": None, "quote": "p46"})])
    B.apply_fact_corrections("D", facts, dec)
    assert facts["total_financing"][1]["status"] == "conflicting"


def test_re_marking_keeps_its_hands_off_when_it_cannot_compare():
    """No canonical to compare with — the two 're-extract' rows leave the field
    without one — means no re-marking at all, rather than a guess."""
    cands = [_cand("$2,793,000 USD", 2_793_000.0, status="supporting"),
             _cand("$12,793,000 USD", 12_793_000.0, page=7, status="conflicting")]
    B.remark_conflicts("gcf_funding_requested", cands)
    assert [c["status"] for c in cands] == ["supporting", "conflicting"]


# --------------------------------------------------------------------------
# add-candidate: the riders that ADD a print rather than fix one
# --------------------------------------------------------------------------

def _add_entry(**add):
    base = {"raw": "(a) Total programme financing (millions) | US$158",
            "value": 158_000_000.0, "currency": "USD", "unit": "million",
            "page": 54, "section": "added", "status": "conflicting",
            "quote": 'p54: "| (a) Total programme financing (millions) | US$158 |"'}
    base.update(add)
    return _entry(action="add-candidate", doc_id="D", wrong=None, corrected=None,
                  add=base)


def test_an_add_candidate_row_records_a_print_the_store_was_not_carrying():
    """FP048's rider: the p.54 print is a second reading of the same field, so
    the conflict machinery must be able to see it."""
    facts = {"total_financing": [_cand("$150 million USD", 150_000_000.0, page=11,
                                       section="rule:B.2(a)")]}
    dec = B.Decisions([_add_entry()])
    recs = B.apply_fact_corrections("D", facts, dec)
    added = facts["total_financing"][1]
    assert added["value"] == 158_000_000.0 and added["page"] == 54
    assert added["added"] is True and "corrected" not in added
    assert recs[0]["from"] is None and recs[0]["to"]["raw"] == added["raw"]
    assert not dec.alarms


def test_an_added_print_is_judged_by_the_same_conflict_rules_as_any_other():
    """The row asks for 'conflicting' and the rules agree — 158M against a
    canonical 150M is past the precision either print claims. The status is not
    taken on trust: it is recomputed and only then published."""
    facts = {"total_financing": [_cand("$150 million USD", 150_000_000.0, page=11,
                                       section="rule:B.2(a)")]}
    B.apply_fact_corrections("D", facts, B.Decisions([_add_entry()]))
    assert facts["total_financing"][1]["status"] == "conflicting"


def test_a_ratified_status_the_conflict_rules_disagree_with_is_said_out_loud():
    """Same row against a canonical that AGREES with the added print: the rules
    make it supporting, and the disagreement with the ratified row is reported
    rather than resolved silently in either direction."""
    facts = {"total_financing": [_cand("$158 million USD", 158_000_000.0, page=11,
                                       section="rule:B.2(a)")]}
    dec = B.Decisions([_add_entry()])
    recs = B.apply_fact_corrections("D", facts, dec)
    assert facts["total_financing"][1]["status"] == "supporting"
    assert recs[0]["status_after_remark"] == "supporting"
    assert any("the row asks for status 'conflicting'" in a for a in dec.alarms)


def test_a_derived_added_value_quotes_the_operands_and_names_the_sum_a_sum():
    """FP042's rider. 96 MEUR is printed on no page: it is 20 + 76, both printed
    on p.55. The raw carries the two operands and says 'sum', the candidate says
    `derived`, and the arithmetic is written down — the store never publishes a
    figure as a print when no page prints it."""
    facts = {"total_financing": [_cand("76 | million eur", 76_000_000.0, page=11,
                                       section="rule:B.2(a)", currency="EUR")]}
    dec = B.Decisions([_add_entry(
        raw="Total amount of co-financing = 76 MEUR; Fund’s investment = 20 MEUR "
            "(sum: 96 MEUR)",
        value=96_000_000.0, currency="EUR", page=55, derived=True,
        derived_from="20 + 76 = 96 MEUR, both operands printed on p.55",
        quote='p55: "Total amount of co-financing = 76 MEUR"')])
    B.apply_fact_corrections("D", facts, dec)
    c = facts["total_financing"][1]
    assert c["derived"] is True and "20 + 76" in c["derived_from"]
    assert "96" not in c["raw"].replace("(sum: 96 MEUR)", "")   # only as a named sum
    assert c["status"] == "conflicting" and c["value"] == 96_000_000.0


def test_an_add_candidate_row_never_doubles_a_print_the_build_already_reads():
    """The day the parser learns to read that page, the ratified row must not
    add a second copy of the same candidate."""
    facts = {"total_financing": [
        _cand("$150 million USD", 150_000_000.0, page=11, section="rule:B.2(a)"),
        _cand("(a) Total programme financing (millions) | US$158", 158_000_000.0,
              page=54, section="rule:B.2(a)", status="conflicting")]}
    dec = B.Decisions([_add_entry()])
    B.apply_fact_corrections("D", facts, dec)
    assert len(facts["total_financing"]) == 2
    assert dec.unapplied and "already in this build" in dec.unapplied[0]["why"]


@needs_decisions
def test_the_four_riders_are_the_four_the_owner_ratified():
    corr = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["corrections"]
    riders = {(e["fp"], e["field"], e["action"]) for e in corr
              if "rider" in e["row_ref"]}
    assert riders == {
        ("FP240", "total_financing", "value-fix"),          # set 221,219,000
        ("FP115", "gcf_funding_requested", "value-fix"),    # set 60,000,000
        ("FP48", "total_financing", "add-candidate"),       # the p.54 US$158 print
        ("FP42", "total_financing", "add-candidate"),       # the 96-MEUR implication
    }


# --------------------------------------------------------------------------
# the serving half: a corrected flat field must reach the note line
#
# title / entity / countries and the two money FALLBACKS are read from
# registry.json (v1) by `registry._fmt`. The corrections are applied by the
# BUILDER into registry_v2.json and v1 is not rewritten, so without a serving
# preference a rebuild would store the corrected figure and keep serving the
# refuted one — the one outcome the correction pass exists to prevent.
# --------------------------------------------------------------------------

def _v2_row(**kw):
    row = {"fp": 999, "facts": {}, "coverage": {"llm_fallback": False}}
    row.update(kw)
    return {"zz_doc_fp999": row}


def _v1_row(**kw):
    row = {"fp": 999, "title": "v1 title", "accredited_entity": "v1 entity",
           "board": 40, "year": 2024}
    row.update(kw)
    return {"zz_doc_fp999": row}


def _correction(field, to, page=31):
    return {"id": "C52", "field": field, "layer": "top-level", "action": "correct-to",
            "from": "the refuted print", "to": to, "page_of_quote": page,
            "quote": "p31", "ratified": "owner, 2026-08-26"}


@pytest.fixture
def served(monkeypatch):
    def _go(v1, v2):
        monkeypatch.setattr(registry, "_cache", v1)
        monkeypatch.setattr(registry, "_cache_v2", v2)
        return registry.registry_note("What is the total financing of FP999?")
    return _go


def test_a_ratified_top_level_correction_is_what_gets_served(served):
    """FP245 in miniature: v1 holds the A.7 total under 'GCF financing'; the
    ratified correction moved it to the grants row, and the line must say so."""
    note = served(_v1_row(gcf_financing="35,107,775 USD"),
                  _v2_row(gcf_financing="USD 27,995,786",
                          meta={"corrections": [
                              _correction("gcf_financing", "USD 27,995,786")]}))
    assert "GCF financing (as printed): USD 27,995,786" in note
    assert "35,107,775" not in note


def test_a_ratified_absence_takes_the_bit_off_the_line_entirely(served):
    """FP086: 'EUR 100 million' is a GCF commitment ceiling, not the programme's
    total financing, and the document never states one. Confirming the absence
    has to REMOVE the claim, not restate it."""
    note = served(_v1_row(total_financing="EUR 100 million"),
                  _v2_row(total_financing=None,
                          meta={"corrections": [
                              _correction("total_financing", None, page=2)]}))
    assert "total financing" not in note
    assert "EUR 100 million" not in note


def test_a_corrected_text_field_cites_the_page_the_correction_names(served):
    """meta_provenance matched its quote against the value v1 holds — the
    refuted one — so a corrected title keeps the correction's own page, not the
    pointer to a page whose quote says something else."""
    note = served(_v1_row(title="Argentina REDD-plus RBP for results period 2014-2019"),
                  _v2_row(title="Argentina REDD-plus RBP for results period 2014-2016",
                          meta_provenance={"title": {"page": 9, "quote": "old"}},
                          meta={"corrections": [
                              _correction("title",
                                          "Argentina REDD-plus RBP for results "
                                          "period 2014-2016", page=3)]}))
    assert '"Argentina REDD-plus RBP for results period 2014-2016" (p.3)' in note
    assert "2014-2019" not in note and "(p.9)" not in note


def test_v1_still_wins_where_no_correction_names_the_field(served):
    """Narrow on purpose. The two files differ for many reasons that are not
    ratified decisions; only a correction for exactly this (document, field)
    changes what is served."""
    note = served(_v1_row(title="v1 title"),
                  _v2_row(title="a different v2 title", meta={"corrections": [
                      _correction("gcf_financing", "USD 1")]}))
    assert '"v1 title"' in note and "a different v2 title" not in note


def test_a_registry_without_the_meta_block_serves_exactly_what_it_always_did(served):
    """The never-break contract: today's shipped registry_v2.json carries no
    `meta` at all, so the line must come out byte-identical until the rebuild
    lands."""
    v1 = _v1_row(title="v1 title", gcf_financing="35,107,775 USD")
    plain = served(v1, _v2_row())
    with_meta_key_but_no_corrections = served(v1, _v2_row(meta={"ratified": "x"}))
    assert plain == with_meta_key_but_no_corrections
    assert "35,107,775 USD" in plain


@pytest.mark.parametrize("meta", [None, {}, [], "corrections", {"corrections": "x"},
                                  {"corrections": [None, 7, {"layer": "top-level"}]}])
def test_a_malformed_meta_block_never_breaks_the_line(served, meta):
    note = served(_v1_row(gcf_financing="35,107,775 USD"), _v2_row(meta=meta))
    assert "GCF financing (as printed): 35,107,775 USD" in note


# --------------------------------------------------------------------------
# drop-candidate: the action that DELETES a print
#
# Ratified 2026-08-26 (serving-wave session) for four rows the wave proved the
# PDF does not support. It is the only action that leaves the store holding
# less than the extraction found and offers nothing in its place, so the row
# has to prove its ground before the builder will run it.
# --------------------------------------------------------------------------

def _drop_entry(wrong=None, **dropped):
    base = {"page": 73, "ground": "printed-nowhere",
            "searched": ["pymupdf", "pdfplumber"],
            "evidence": "zero hits for 143,507 on any page in either extraction",
            "quote": "p73 (independent): the E.2.2 table prints no such row"}
    base.update(dropped)
    return _entry(action="drop-candidate", doc_id="D", layer="fact-conflicting",
                  wrong=wrong or {"raw": "143,507 million", "page": 73},
                  corrected=None, dropped=base)


def test_a_fabricated_print_is_dropped_and_the_search_that_proved_it_recorded():
    """FP162's page-73 row. No extraction of the PDF prints 143,507, so there is
    no figure to correct it TO — the candidate is not a reading of the document
    at all."""
    facts = {"total_financing": [_cand("143,327 million USD", 143_327_000.0),
                                 _cand("143,507 million", None, page=73,
                                       section="rule:B.2(a)", status="conflicting")]}
    dec = B.Decisions([_drop_entry()])
    recs = B.apply_fact_corrections("D", facts, dec)
    assert [c["raw"] for c in facts["total_financing"]] == ["143,327 million USD"]
    assert recs[0]["to"] is None and recs[0]["ground"] == "printed-nowhere"
    assert recs[0]["searched"] == ["pymupdf", "pdfplumber"]
    assert "zero hits" in recs[0]["evidence"]
    assert recs[0]["quote"].startswith("p73")
    assert not dec.alarms


def test_dropping_the_only_rival_reading_dissolves_the_conflict():
    """The point of the four drops: two documents were published as
    contradicting themselves on the strength of a row the PDF never printed."""
    facts = {"total_financing": [_cand("118.08 | million eur", 118_080_000.0, page=114),
                                 _cand("EUR 80.59 million", 80_590_000.0, page=137,
                                       status="conflicting")]}
    dec = B.Decisions([_drop_entry(
        wrong={"raw": "EUR 80.59 million", "page": 137}, page=137,
        evidence="80.59 appears on no page of the PDF",
        quote="p137 (independent), Table 24: 118.08")])
    B.apply_fact_corrections("D", facts, dec)
    assert [c["status"] for c in facts["total_financing"]] == ["canonical"]
    assert not dec.alarms


def test_a_drop_that_empties_a_field_removes_the_field():
    facts = {"total_financing": [_cand("143,507 million", None, page=73,
                                       status="conflicting")]}
    B.apply_fact_corrections("D", facts, B.Decisions([_drop_entry()]))
    assert "total_financing" not in facts


def test_a_drop_with_no_ratified_ground_is_refused_and_shouted_about():
    """A drop is destructive, so 'the adjudication said so' is not enough: the
    row names WHY, out of a closed vocabulary, or it does not run."""
    facts = {"total_financing": [_cand("143,507 million", None, page=73,
                                       status="conflicting")]}
    dec = B.Decisions([_drop_entry(ground="looks wrong")])
    B.apply_fact_corrections("D", facts, dec)
    assert len(facts["total_financing"]) == 1
    assert dec.unapplied and "ratified ground" in dec.unapplied[0]["why"]
    assert any("NOT APPLIED" in a for a in dec.alarms)


def test_a_label_bleed_drop_needs_the_field_that_owns_the_print_to_hold_it():
    """FP176's A.9 bleed: 'USD 250 Million' is the project-size CEILING read as
    a GCF request. Dropping it is safe only because project_size carries the
    whole print — so the builder checks that before deleting anything."""
    bleed = {"page": 5, "ground": "label-bleed", "belongs_to": "project_size",
             "belongs_to_raw": "Medium (Up to USD 250 Million)",
             "searched": ["pymupdf"], "evidence": "the 250 is the size band's ceiling",
             "quote": 'p5: "A.9. Project size | Medium (Up to USD 250 Million)"'}
    entry = _entry(action="drop-candidate", doc_id="D", field="gcf_funding_requested",
                   layer="fact-supporting", corrected=None,
                   wrong={"raw": "USD 250 Million", "page": 5}, dropped=bleed)
    gcf = [_cand("€30,138,772 Eur", 30_138_772.0),
           _cand("USD 250 Million", 250_000_000.0, status="supporting")]
    size = [_cand("Medium (Up to USD 250 Million)", None, section="rule:A.9")]

    held = {"gcf_funding_requested": list(gcf), "project_size": size}
    B.apply_fact_corrections("D", held, B.Decisions([entry]))
    assert [c["raw"] for c in held["gcf_funding_requested"]] == ["€30,138,772 Eur"]

    # ... and with nothing holding the print, the drop would lose it: refused
    orphan = {"gcf_funding_requested": list(gcf)}
    dec = B.Decisions([entry])
    B.apply_fact_corrections("D", orphan, dec)
    assert len(orphan["gcf_funding_requested"]) == 2
    assert dec.unapplied and "would be lost, not moved" in dec.unapplied[0]["why"]


@needs_decisions
def test_the_twelve_serving_wave_rows_are_the_ones_the_owner_ratified():
    corr = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["corrections"]
    wave = [e for e in corr
            if e["ratified"] == "owner, 2026-08-26 (serving-wave session)"]
    assert {(e["fp"], e["field"], e["action"]) for e in wave} == {
        # the four misfiled / misread canonical totals
        ("FP260", "total_financing", "correct-to"),          # -> 83,811,581
        ("FP204", "total_financing", "correct-to"),          # -> 1,119,000,000
        ("FP261", "total_financing", "correct-to"),          # -> 391.43 million
        ("FP233", "total_financing", "correct-to"),          # -> 79,690,370
        # FP162's two unparsed covers
        ("FP162", "total_financing", "value-fix"),           # -> 143,327,000
        ("FP162", "gcf_funding_requested", "value-fix"),     # -> 82,849,900
        # the four drops
        ("FP162", "gcf_funding_requested", "drop-candidate"),   # fabricated $423M
        ("FP162", "total_financing", "drop-candidate"),         # fabricated 143,507M
        ("FP214", "total_financing", "drop-candidate"),         # fabricated 80.59M
        ("FP176", "gcf_funding_requested", "drop-candidate"),   # A.9 bleed
        # the two RBP prints that stop an absence and a top-level from both
        # being live at once
        ("FP100", "gcf_funding_requested", "add-candidate"),
        ("FP142", "gcf_funding_requested", "add-candidate"),
    }
    assert len(wave) == 12
    assert {e["row_ref"]["file"] for e in wave} == {
        "scratchpad/serving_wave_adjudication.json"}


@needs_decisions
def test_the_two_rbp_add_candidates_are_for_documents_the_absence_file_names():
    """The add-candidate rows only do their job because an absence row exists
    for the same (document, field): the print is what withholds it."""
    corr = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["corrections"]
    absc = json.loads(ABSENCES.read_text(encoding="utf-8"))
    pairs = {(a["doc_id"], a["field"]) for a in absc["absences"]}
    adds = [(e["doc_id"], e["field"]) for e in corr
            if e["action"] == "add-candidate"
            and e["ratified"].endswith("(serving-wave session)")]
    assert len(adds) == 2 and set(adds) <= pairs
    # and the ratified precedence is written down where a reader of an absence
    # will look for it
    prec = absc["serving_precedence"]
    assert prec["order"] == ["fact-canonical", "top-level-as-printed",
                             "confirmed-absence"]
    assert prec["ratified"] == "owner, 2026-08-26 (serving-wave session)"


def test_a_supporting_print_withholds_an_absence_quietly_and_says_why():
    """The FP100/FP142 shape. A CANONICAL print against a ratified absence is a
    flat contradiction and gets shouted about; a SUPPORTING one means the
    absence was true about the template and the document prints the figure
    somewhere else — withheld, with the reason, and no alarm."""
    dec = B.Decisions(absences=[{"doc_id": "D", "field": "gcf_funding_requested",
                                 "pages_checked": [1, 16], "evidence": "no cover field",
                                 "status": "ratified"}])
    facts = {"gcf_funding_requested": [_cand("GCF RBP: 96,452,228", 96_452_228.0,
                                             page=103, section="AE-fee budget (Table 19)",
                                             status="supporting")]}
    assert B.absence_meta("D", facts, dec) == {}
    assert not dec.alarms                       # nothing contradicts anything
    why = dec.absences_skipped[0]["why"]
    assert "absence not published over a print" in why and "p.103" in why


# ==========================================================================
# the cross-extractor verification arm
#
# OWNER RATIFICATION 2026-08-26 (serving-wave session): adopted as a STANDING
# arm of the build. Everything else in this file reads one rendering of the
# corpus — the qwen2.5-vl-7b markdown — and phase 3's literal pass proved every
# registry raw is printed on the markdown page it cites. That proof is silent
# about the MARKDOWN being wrong, because it checks the markdown against
# itself. This arm reads the figure again out of a text extraction made by a
# different tool and says so when the two disagree.
#
# The fixtures below are the eight pages the serving wave proved fabricated.
# ==========================================================================

INDEPENDENT = Path(__file__).resolve().parents[1] / "data" / "extracted" / "pymupdf"

needs_independent = pytest.mark.skipif(
    not INDEPENDENT.is_dir(), reason="independent extraction absent")


def indep(pages: dict) -> Path:
    """An in-memory stand-in for data/extracted/pymupdf/<doc>.txt."""
    return "".join(f"=== PAGE {n} ===\n{b}\n" for n, b in sorted(pages.items()))


@pytest.fixture
def indep_dir(tmp_path):
    def _write(doc_id: str, pages: dict):
        (tmp_path / f"{doc_id}.txt").write_text(indep(pages), encoding="utf-8")
        return tmp_path
    return _write


@pytest.mark.parametrize("tok,key", [
    ("49,944,050", "49944050"),        # US grouping
    ("49.944.050", "49944050"),        # European grouping — the same figure
    ("49944050", "49944050"),          # no grouping at all
    ("0049", "49"),                    # leading zeros are not a disagreement
    ("391.43", "39143"),
    ("5", None),                       # a lone digit confirms itself against
    ("7", None),                       # anything, so it is never a key
])
def test_a_figure_keys_to_its_digits_whatever_the_separators(tok, key):
    assert B._figure_key(tok) == key


def test_the_page_side_joins_space_separated_thousands_and_the_raw_side_never_does():
    """PDF text layers print '1 234 567' and break numbers over line ends. The
    page may be read both ways so a real print is not called missing; the
    CANDIDATE may not, or joining could invent the figure it is meant to check."""
    assert "1234567" in B.figure_keys("total 1 234 567 USD", spaced=True)
    assert "1234567" not in B.figure_keys("total 1 234 567 USD")


def _pages(body, page=5):
    return {page: body + "\n" + "filler text " * 20}


def test_a_figure_the_independent_extraction_prints_is_confirmed(indep_dir):
    root = indep_dir("D", _pages("A.7. Total financing 49,944,050 USD"))
    facts = {"total_financing": [_cand("49,944,050 USD", 49_944_050.0)]}
    block, counts = B.cross_check_meta("D", facts, root=root)
    assert block is None                       # nothing to say
    assert counts == {"confirmed-print": 1}
    assert "cross_check" not in facts["total_financing"][0]


def test_the_same_amount_in_another_notation_is_not_a_disagreement(indep_dir):
    """The store normalizes 'USD 40,000,000' where the PDF prints '$ 40
    million'. Same figure, different notation — read with the builder's own
    amount reader rather than called a fabrication."""
    root = indep_dir("D", _pages("(b) Requested GCF amount $ 40 million", page=58))
    facts = {"gcf_funding_requested": [_cand("USD 40,000,000", 40_000_000.0, page=58)]}
    block, counts = B.cross_check_meta("D", facts, root=root)
    assert block is None and counts == {"confirmed-value": 1}


def test_a_digit_misread_is_flagged_with_the_figure_the_pdf_actually_prints(indep_dir):
    """FP233's cover in miniature: the markdown prints 69,830,370 and the PDF
    prints 79,690,370. Nothing is corrected — the flag says the two readings
    disagree and names the figure the other extractor saw."""
    root = indep_dir("D", _pages("A.7. Total financing 79,690,370 USD"))
    facts = {"total_financing": [_cand("69,830,370 USD", 69_830_370.0)]}
    block, counts = B.cross_check_meta("D", facts, root=root)
    assert counts == {"not-in-document": 1}
    assert facts["total_financing"][0]["cross_check"] == "not-in-document"
    assert facts["total_financing"][0]["value"] == 69_830_370.0   # NOT corrected
    flag = block["flagged"][0]
    assert flag["figure"] == "69830370"
    assert "79690370" in flag["independent_page_prints"]
    assert block["ratified"] == "owner, 2026-08-26 (serving-wave session)"


def test_a_figure_printed_elsewhere_in_the_document_is_a_weaker_flag(indep_dir):
    """Not the same finding: the figure exists, the page attribution does not."""
    root = indep_dir("D", {5: "A.7. Total financing 12,345,678 USD " + "x " * 40,
                           9: "the programme's 99,999,999 USD envelope " + "x " * 40})
    facts = {"total_financing": [_cand("99,999,999 USD", 99_999_999.0, page=5)]}
    block, counts = B.cross_check_meta("D", facts, root=root)
    assert counts == {"not-on-cited-page": 1}
    assert block["flagged"][0]["verdict"] == "not-on-cited-page"


def test_a_page_with_no_readable_text_layer_is_unknown_not_absent(indep_dir):
    """A scanned page extracts to nothing. 'The other tool saw nothing at all'
    is not evidence that the VLM invented the figure."""
    root = indep_dir("D", {5: "  "})
    facts = {"total_financing": [_cand("49,944,050 USD", 49_944_050.0)]}
    block, counts = B.cross_check_meta("D", facts, root=root)
    assert block is None and counts == {"no-independent-page": 1}


def test_a_document_with_no_independent_extraction_is_counted_not_guessed_at(tmp_path):
    facts = {"total_financing": [_cand("49,944,050 USD", 49_944_050.0)]}
    block, counts = B.cross_check_meta("D", facts, root=tmp_path)
    assert block is None and counts == {"no-independent-extraction": 1}
    assert "cross_check" not in facts["total_financing"][0]


def test_only_canonical_money_facts_are_in_scope(indep_dir):
    """The ratification scopes the arm to canonical money: a supporting print
    the parser found on some narrative page is not the store's answer to
    anything, and flagging it would bury the flags that matter."""
    root = indep_dir("D", _pages("A.7. Total financing 49,944,050 USD"))
    facts = {"total_financing": [_cand("49,944,050 USD", 49_944_050.0),
                                 _cand("11,111,111 USD", 11_111_111.0,
                                       status="supporting")],
             "title": [_cand("A title with 12,345,678 in it", None,
                             status="canonical")]}
    block, counts = B.cross_check_meta("D", facts, root=root)
    assert block is None and counts == {"confirmed-print": 1}


def test_a_flag_is_withdrawn_when_a_rebuild_confirms_the_figure(indep_dir):
    """The arm runs on every build, so a candidate that was flagged and has
    since been corrected must not keep a stale marker."""
    facts = {"total_financing": [_cand("69,830,370 USD", 69_830_370.0,
                                       cross_check="not-in-document")]}
    root = indep_dir("D", _pages("A.7. Total financing 69,830,370 USD"))
    block, counts = B.cross_check_meta("D", facts, root=root)
    assert block is None and counts == {"confirmed-print": 1}
    assert "cross_check" not in facts["total_financing"][0]


def test_a_raw_with_no_multi_digit_figure_is_reported_as_uncheckable(indep_dir):
    root = indep_dir("D", _pages("A.10 Grant $5 million"))
    facts = {"financial_instruments": [_cand("$5 million USD", None)]}
    block, counts = B.cross_check_meta("D", facts, root=root)
    assert block is None and counts == {"no-figure": 1}


# --- the eight fabricated pages, against the real corpus -------------------

@needs_corpus
@needs_independent
@pytest.mark.parametrize("stem,field,raw,page,value", [
    # the five the serving wave proved fabricated AND that a money fact cites
    ("15_gcf-b41-02-add09-rev01-funding-proposal-package-fp261",
     "total_financing", "$381.43 million USD", 32, 381_430_000.0),   # 391.43 misread
    ("43_gcf-b39-02-add08-funding-proposal-package-fp233",
     "total_financing", "69,830,370 USD", 5, 69_830_370.0),          # 79,690,370 misread
    ("113_gcf-b28-02-add09",
     "gcf_funding_requested", "$423 million USD", 64, 423_000_000.0),  # invented cell
    ("113_gcf-b28-02-add09",
     "total_financing", "143,507 million", 73, None),                # invented row
    ("61_gcf-b37-02-add05-funding-proposal-package-fp214",
     "total_financing", "EUR 80.59 million", 137, 80_590_000.0),     # invented table
])
def test_the_arm_flags_the_pages_the_wave_proved_fabricated(stem, field, raw, page, value):
    """These five rows are exactly what the arm exists to catch, and it catches
    them from the corpus as it stands — no adjudication, no model call."""
    pages = B.independent_pages(stem)
    assert pages, stem
    cand = {"raw": raw, "value": value, "currency": None, "unit": None,
            "page": page, "section": "x", "status": "canonical"}
    verdict, detail = B.cross_check_candidate(cand, pages, B.figure_keys(
        "\n".join(pages.values()), spaced=True))
    assert verdict in B.CROSS_CHECK_FLAGS, (stem, verdict, detail)


@needs_corpus
@needs_independent
@pytest.mark.parametrize("stem,field,raw,page,value", [
    # FP260's p31 and FP204's p162: the GCF row filed as the total. The FIGURE
    # is printed exactly there — the defect is which label it sits under
    ("16_gcf-b41-02-add08-funding-proposal-package-fp260",
     "total_financing", "25,000,000 USD", 31, 25_000_000.0),
    ("71_gcf-b35-02-add06",
     "total_financing", "160 | million USD", 162, 160_000_000.0),
    # FP176's A.9 bleed: 'USD 250 Million' is printed on p5, as a size band
    ("99_gcf-b30-02-add08",
     "gcf_funding_requested", "USD 250 Million", 5, 250_000_000.0),
])
def test_the_arm_does_not_flag_a_misfiling_because_the_figure_is_printed(
        stem, field, raw, page, value):
    """The arm answers one question — 'does the other extractor print this
    figure on this page?' — and the four misfiled rows of the same wave answer
    it YES. A wrong FIELD is a different defect and needs a different check;
    reporting these would only make the fabrication flags harder to find."""
    pages = B.independent_pages(stem)
    assert pages, stem
    cand = {"raw": raw, "value": value, "currency": None, "unit": None,
            "page": page, "section": "x", "status": "canonical"}
    verdict, _ = B.cross_check_candidate(cand, pages, None)
    assert verdict.startswith("confirmed"), (stem, verdict)


# --- the two arm defects the cross-check census found, and their eight
# --- named false positives ------------------------------------------------

@pytest.mark.parametrize("text,closed", [
    ("27 054 | million USD", "27054 | million USD"),   # FP68's own raw
    ("55 000 000 USD", "55000000 USD"),
    ("96 953 million USD", "96953 million USD"),
    ("1,234 5,678", "1,234 5,678"),        # two figures, NOT one: never joined
    ("49,944,050 USD", "49,944,050 USD"),  # nothing to close
    ("page 5 of 99", "page 5 of 99"),
])
def test_a_candidates_space_grouped_thousands_close_and_nothing_else_does(text, closed):
    """Defect (a). The page side may join any run of digits and spaces because
    a text layer breaks numbers anywhere; the candidate side may only close a
    well-formed thousands grouping, or joining could invent the very figure the
    check is meant to test."""
    assert B.unsplit_thousands(text) == closed


@pytest.mark.parametrize("cand_key,page_key,hit", [
    ("152500000", "4152500000", True),    # FP223: marker 4 glued in front
    ("880000000", "8800000007", True),    # FP190: marker 7 glued behind
    ("5800", "580010", True),             # FP65:  marker 10 glued behind
    ("790", "7900", False),               # a TRAILING ZERO is a magnitude, not
    ("4994405", "49944050", False),       # a marker: 79.0/79.00, 4,994,405/49,944,050
    ("12345", "123456789", False),        # more than a marker's worth of digits
    ("49", "4917", False),                # never chew a key below three digits
])
def test_only_a_footnote_marker_sized_glue_is_forgiven_on_the_page_side(
        cand_key, page_key, hit):
    """Defect (b). Strip 1-2 digits off ONE end, require the remainder to equal
    the candidate's key EXACTLY, and require the stripped digits to look like a
    superscript (1-99, never 0-led). No substring test, no edit distance."""
    assert bool(B.glued_footnote_key({cand_key}, {page_key})) is hit


# The eight canonical money facts the 2026-08-26 cross-check census adjudicated
# STORE-RIGHT: the store is right and the ARM was the misreader. Four are the
# two ratified defects and must not flag again. Four are not: the census names
# them as classes the arm cannot see at all, and they are pinned here as they
# stand so that fixing them is a visible decision rather than a drift.
_FALSE_POSITIVES = [
    # (stem, field, raw, page, value, still_flags, why)
    ("207_gcf-b19-22-add10", "gcf_funding_requested", "27 054 | million USD", 11,
     None, False, "defect (a): the raw's own '27 054' keyed as the fragment '54'"),
    ("52_gcf-b37-02-add14-funding-proposal-package-fp223", "gcf_funding_requested",
     "152,500,000 USD", 5, 152_500_000.0, False,
     "defect (b): the text layer prints '4152,500,000' — marker 4 glued in front"),
    ("85_gcf-b33-02-add04_0", "total_financing", "880,000,000 USD", 5,
     880_000_000.0, False,
     "defect (b): the text layer prints '880,000,0007' — marker 7 glued behind"),
    ("210_gcf-b19-22-add07", "co_financing", "$580.0 million USD", 14,
     580_000_000.0, False,
     "defect (b): the text layer prints '580.010' — marker 10 glued behind"),
    ("169_gcf-b22-10-add10-rev01", "co_financing", "USD 437 million", 14,
     437_000_000.0, True,
     "NOT a defect: 437 is the SUM of the page's four co-financing rows "
     "(260+58+77+42) and is printed nowhere as one token. An arm keyed on "
     "figures cannot see a derived figure"),
    ("220_gcf-b18-04-add10-rev01", "co_financing", "USD 74.1 million", 9,
     74_100_000.0, True,
     "NOT a defect: 74.1 is derived too — (a) 118.6 - (b) 44.5"),
    ("228_gcf-b18-04-add02", "gcf_funding_requested", "USD 110,000,000", 9,
     110_000_000.0, True,
     "NOT a defect: the page prints '110' with 'million USD ($)' on the NEXT "
     "line, so the value check cannot bind the scale word to the figure"),
    ("164_gcf-b23-02-add05", "total_financing", "79.0 | million USD", 12,
     79_000_000.0, True,
     "NOT a defect: '79.0' against the page's '79.00'. Keys carry no decimal "
     "point, so forgiving this would forgive 4,994,405 against 49,944,050"),
]


@needs_corpus
@needs_independent
@pytest.mark.parametrize(
    "stem,field,raw,page,value,still_flags,why",
    _FALSE_POSITIVES, ids=[f"{r[0].split('_')[0]}-{r[1][:4]}" for r in _FALSE_POSITIVES])
def test_the_census_false_positives_land_where_the_adjudication_put_them(
        stem, field, raw, page, value, still_flags, why):
    pages = B.independent_pages(stem)
    assert pages, stem
    cand = {"raw": raw, "value": value, "currency": None, "unit": None,
            "page": page, "section": "x", "status": "canonical"}
    verdict, _ = B.cross_check_candidate(
        cand, pages, B.figure_keys("\n".join(pages.values()), spaced=True))
    if still_flags:
        assert verdict in B.CROSS_CHECK_FLAGS, (stem, verdict, why)
    else:
        assert verdict.startswith("confirmed"), (stem, verdict, why)


@needs_corpus
@needs_independent
def test_the_defect_fixes_did_not_blind_the_arm_to_a_real_misread():
    """The split-number fix must make the arm read the candidate's figure, not
    stop checking it: FP236's '96 953 million USD' closes to 96953, which the
    page still does not print — it prints 90.953. Sensitivity, not amnesty."""
    stem = "40_gcf-b39-02-add11-funding-proposal-package-fp236"
    pages = B.independent_pages(stem)
    if not pages:                                   # pragma: no cover
        pytest.skip(f"{stem} has no independent extraction")
    cand = {"raw": "96 953 million USD", "value": None, "currency": "USD",
            "unit": None, "page": 5, "section": "x", "status": "canonical"}
    verdict, detail = B.cross_check_candidate(
        cand, pages, B.figure_keys("\n".join(pages.values()), spaced=True))
    assert verdict in B.CROSS_CHECK_FLAGS, (verdict, detail)
    assert detail["figure"] == "96953"               # the figure, not '96' or '953'


@needs_corpus
@needs_independent
def test_every_document_in_the_corpus_has_an_independent_extraction_to_check():
    """The arm is only a standing guarantee while both renderings cover the
    whole corpus. If a document loses its independent extraction the census
    says so out loud rather than reporting it clean."""
    vlm = {p.stem for p in EXTRACTED.glob("*.md")}
    independent = {p.stem for p in INDEPENDENT.glob("*.txt")}
    assert not (vlm - independent - {"status"}), sorted(vlm - independent)[:5]


# --------------------------------------------------------------------------
# a ratified row and a page re-extraction fixing the same defect
#
# The serving-wave adjudication pairs them on purpose ("correct-to 79,690,370;
# re-extract p5"), and the re-extraction ran the same day. Five of its six
# affected rows found their target gone because the fresh page reads the figure
# correctly — which must not look like a ratified decision that failed.
# --------------------------------------------------------------------------

def _reex(**kw):
    base = {"ran": "2026-08-26 (serving-wave session)",
            "model": "qwen/qwen2.5-vl-7b", "pages": [5]}
    base.update(kw)
    return base


def test_a_reextracted_page_that_now_reads_the_ratified_figure_is_not_an_alarm():
    """FP233: the row said 'correct 69,830,370 to 79,690,370 and re-extract p5'.
    The fresh p5 prints 79,690,370, so the wrong candidate is gone and the right
    one is canonical — the outcome the row asked for, reached the other way."""
    facts = {"total_financing": [_cand("79,690,370 USD", 79_690_370.0)]}
    dec = B.Decisions([_entry(doc_id="D", reextracted=_reex(),
                              wrong={"raw": "69,830,370 USD", "page": 5},
                              corrected={"raw": "USD 79,690,370", "value": 79_690_370.0,
                                         "currency": "USD", "unit": None, "page": 94,
                                         "section": None, "quote": "p94"})])
    recs = B.apply_fact_corrections("D", facts, dec)
    assert not dec.unapplied and not dec.alarms
    assert dec.applied == ["T01"]
    assert "79,690,370" in recs[0]["resolved_by_reextraction"]
    # and nothing was touched: the fresh page's own print stands as it was read
    assert facts["total_financing"][0] == _cand("79,690,370 USD", 79_690_370.0)


def test_a_drop_row_is_settled_when_the_reextraction_removed_the_fabrication():
    """FP162's p73 and FP214's p137. The row exists to delete a print the PDF
    does not contain; the fresh page not printing it is exactly that outcome."""
    facts = {"total_financing": [_cand("143,327 million USD", 143_327_000.0)]}
    dec = B.Decisions([_drop_entry() | {"reextracted": _reex(pages=[73])}])
    recs = B.apply_fact_corrections("D", facts, dec)
    assert not dec.unapplied and not dec.alarms
    assert "ratified to drop" in recs[0]["resolved_by_reextraction"]


def test_a_reextracted_page_that_still_reads_wrong_does_not_settle_the_row():
    """FP260's shape, and the reason this is not a blanket amnesty: the fresh
    p31 came back honest (the C.1 '(a)' label restored) and the parser STILL
    elected the GCF row as the total. The ratified figure has not landed, so
    the row is NOT settled and the disagreement is shouted about.

    RE-TAKEN 2026-08-26 (corpus-cure round): the shout is now CARRIED FORWARD
    rather than NOT APPLIED, and the ratified figure takes the canonical it was
    ratified to hold instead of being dropped. A re-extraction that disagrees
    with the owner does not get to overrule the owner in silence."""
    facts = {"total_financing": [_cand("25,000,000 | USD", 25_000_000.0, page=31,
                                       section="C.1")]}
    dec = B.Decisions([_entry(doc_id="D", reextracted=_reex(pages=[31, 32]),
                              wrong={"raw": "25,000,000 USD", "page": 31},
                              corrected={"raw": "83,811,581 USD", "value": 83_811_581.0,
                                         "currency": "USD", "unit": None, "page": 7,
                                         "section": None, "quote": "p7"})])
    entry = dec.by_doc["D"][0]
    B.apply_fact_corrections("D", facts, dec)
    # NOT settled — the fresh canonical is not the ratified figure — and the
    # row is therefore carried rather than dropped
    assert B.reextraction_settled(entry, {"total_financing": [
        _cand("25,000,000 | USD", 25_000_000.0, page=31, section="C.1")]}) is None
    assert not dec.unapplied
    assert [c["id"] for c in dec.carried] == ["T01"]
    assert any("CARRIED FORWARD" in a and "25,000,000" in a for a in dec.alarms)
    assert B._canon_of(facts, "total_financing")["value"] == 83_811_581.0


def test_an_undeclared_reextraction_settles_a_row_on_its_outcome():
    """RE-TAKEN 2026-08-26 (corpus-cure round), and this one REVERSES.

    It used to read `..._gets_no_benefit_of_the_doubt`: a row had to have said
    in advance (`reextracted`) that its page was going for re-extraction, and a
    target that vanished for any other reason was an alarm. The corpus cure
    re-extracted 95 pages across a hundred-odd ratified rows and annotated none
    of them, so every one of those rows arrived undeclared — 80 of them landing
    exactly the ratified figure and being shouted about anyway, which buried the
    25 rows that had genuinely lost theirs under 111 identical alarms.

    A declaration was never the evidence. The fresh page reading the ratified
    figure is, and that is what is checked now."""
    facts = {"total_financing": [_cand("79,690,370 USD", 79_690_370.0)]}
    dec = B.Decisions([_entry(doc_id="D",
                              wrong={"raw": "69,830,370 USD", "page": 5},
                              corrected={"raw": "USD 79,690,370", "value": 79_690_370.0,
                                         "currency": "USD", "unit": None, "page": 94,
                                         "section": None, "quote": "p94"})])
    recs = B.apply_fact_corrections("D", facts, dec)
    assert not dec.unapplied and not dec.carried and not dec.alarms
    assert "79,690,370" in recs[0]["resolved_by_reextraction"]
    # and nothing was touched: the fresh page's own print stands as it was read
    assert facts["total_financing"] == [_cand("79,690,370 USD", 79_690_370.0)]


def test_a_near_miss_never_counts_as_the_ratified_figure():
    """THE ROW FP274 COST. Settlement used to be tested with `_agree`, whose
    0.5% band exists so two prints of one reading ('40.15 million' /
    '40,150,000') recognise each other. Ten dollars either side of forty
    million is inside that band — so the cure leaving p.40's '40,751,254'
    standing against a ratified 40,751,264 read as 'the re-extracted page reads
    the ratified figure', and a single-digit misread the cross-extractor arm
    independently flags `not-in-document` would have superseded the owner.

    A supersession claim is a claim about digits, so it is tested on digits."""
    facts = {"gcf_funding_requested": [_cand("40,751,254", 40_751_254.0, page=40)]}
    entry = _entry(doc_id="D", field="gcf_funding_requested",
                   wrong={"raw": "40,511,264 USD", "page": 7},
                   corrected={"raw": "40,751,264 USD", "value": 40_751_264.0,
                              "currency": "USD", "unit": None, "page": 7,
                              "section": None, "quote": "p7 A.8"})
    assert B.reextraction_settled(entry, facts) is None
    assert B._agree(40_751_254.0, 40_751_264.0)          # the old test said yes
    assert not B._same_figure(40_751_254.0, 40_751_264.0)
    dec = B.Decisions([entry])
    B.apply_fact_corrections("D", facts, dec)
    assert [c["id"] for c in dec.carried] == ["T01"]
    assert B._canon_of(facts, "gcf_funding_requested")["value"] == 40_751_264.0


def test_a_ratified_figure_the_parse_holds_but_does_not_publish_is_promoted():
    """'The wrong one is gone' is still not 'the right one is published'.

    RE-TAKEN 2026-08-26 (corpus-cure round). The row is still NOT settled — a
    field whose canonical is not the ratified figure has settled nothing — but
    the outcome is no longer to drop the figure and shout. The parse already
    holds it; what it got wrong is which candidate to elect, so the candidate is
    promoted and the election is what gets said out loud. Three real rows have
    this shape (C104/FP270, C124, C127), each with the ratified print sitting
    unelected behind a worse one."""
    facts = {"total_financing": [_cand("79,690,370 USD", 79_690_370.0,
                                       status="supporting")]}
    dec = B.Decisions([_entry(doc_id="D", reextracted=_reex(),
                              wrong={"raw": "69,830,370 USD", "page": 5},
                              corrected={"raw": "USD 79,690,370", "value": 79_690_370.0,
                                         "currency": "USD", "unit": None, "page": 94,
                                         "section": None, "quote": "p94"})])
    recs = B.apply_fact_corrections("D", facts, dec)
    assert not dec.unapplied
    assert [c["id"] for c in dec.carried] == ["T01"]
    assert recs[0]["carried_forward"] == "promoted"
    # promoted in place: the page print it was read from is what stands
    assert facts["total_financing"] == [_cand("79,690,370 USD", 79_690_370.0)]


def test_a_superseded_link_in_a_correction_chain_is_never_carried_forward():
    """106_gcf-b30-02-add01's request was corrected TWICE: phase 3 moved it to
    18,591,556, and the cross-check round then read the PDF itself and moved it
    again to 16,591,556 — the later row's `wrong` block quoting the earlier
    row's output verbatim, section 'corrected' and all.

    While both targets existed the chain resolved itself in order. Once the
    cured page reads the FINAL figure directly, BOTH targets are gone, and
    carrying the intermediate row forward would reinstate a reading the owner
    has since superseded over the one they ratified last. Four doc/field pairs
    in the ledger have this shape."""
    facts = {"gcf": [_cand("16,591,556 USD", 16_591_556.0)]}
    first = _entry(id="C02", field="gcf",
                   wrong={"raw": "$34,585,556 USD", "page": 5},
                   corrected={"raw": "18,591,556 USD", "value": 18_591_556.0,
                              "currency": "USD", "unit": None, "page": 46,
                              "section": None, "quote": "p46"})
    later = _entry(id="C108", field="gcf",
                   wrong={"raw": "18,591,556 USD", "value": 18_591_556.0, "page": 46},
                   corrected={"raw": "16,591,556 USD", "value": 16_591_556.0,
                              "currency": "USD", "unit": None, "page": 46,
                              "section": None, "quote": "independent p46"})
    dec = B.Decisions([first, later])
    recs = B.apply_fact_corrections("D", facts, dec)
    assert B._superseded_link(first, dec) and not B._superseded_link(later, dec)
    # the store keeps the figure ratified LAST, and the dead link says nothing
    assert [c["raw"] for c in facts["gcf"]] == ["16,591,556 USD"]
    assert not dec.carried and not dec.unapplied and not dec.alarms
    assert "18,591,556" in recs[0]["superseded_by_a_later_row"]


@needs_decisions
def test_the_reextracted_rows_name_the_run_that_produced_them():
    """Every row that claims a re-extraction has to say which pages went, with
    what, and what came back — the same standard the corrections themselves are
    held to."""
    corr = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["corrections"]
    reex = [e for e in corr if e.get("reextracted")]
    # C47/C48/C77 are the cross-check session's own p9 re-extraction of
    # 177_gcf-b21-10-add20: two rows of the FIRST session whose targets that page
    # carried, and the row that supersedes one of them
    assert {e["id"] for e in reex} == {"C63", "C65", "C66", "C69", "C70", "C71",
                                       "C47", "C48", "C77"}
    for e in reex:
        r = e["reextracted"]
        assert r["model"] == "qwen/qwen2.5-vl-7b", e["id"]
        assert r["pages"] and r["backups"], e["id"]
        assert r["ran"].startswith("2026-08-26"), e["id"]
        # what came back is recorded per page, and no page came back garbled
        for page, got in r["outcome"].items():
            assert got["garble_present"] == [], (e["id"], page)
    # the one row the re-extraction MOVED rather than settled keeps the target
    # it was ratified against, beside the one it now points at
    moved = next(e for e in corr if e["id"] == "C63")
    assert moved["wrong_before_reextraction"]["raw"] == "25,000,000 USD"
    assert moved["wrong"]["raw"] != moved["wrong_before_reextraction"]["raw"]
    assert "re_pointed" in moved["reextracted"]


# --------------------------------------------------------------------------
# reproducibility, and the reuse path that must not swallow a ratified row
# --------------------------------------------------------------------------

def test_the_arm_reports_the_same_nearest_figures_every_run():
    """The nearest-figure list is chosen out of a SET, so ties broken by
    iteration order would make the same corpus build to different bytes on
    different runs. The build is a data product; it has to be reproducible."""
    keys = {"79690370", "69830371", "69830379", "40690370", "39000000"}
    first = B._nearest_figures("69830370", keys)
    assert first == B._nearest_figures("69830370", set(reversed(sorted(keys))))
    assert first == B._nearest_figures("69830370", keys)
    assert first[0] in ("69830371", "69830379")     # both one edit away
    assert first == sorted(first[:2]) + first[2:]   # ... and the tie is ordered


def test_a_carried_forward_llm_candidate_comes_back_uncorrected():
    """The reuse path saves model calls; it must not carry a ratified
    correction forward. A candidate reused WITH its correction baked in is one
    no correction row can find, so reusing the shipped registry would make its
    own ratified rows stop landing, quietly, one rebuild at a time."""
    shipped = {"raw": "19,710,637 USD", "value": 19_710_637.0, "currency": "USD",
               "unit": None, "page": 46, "section": "corrected", "status": "supporting",
               "corrected": True,
               "corrected_from": {"raw": "**Total:** $222,000", "value": 222_000.0,
                                  "currency": "USD", "unit": None, "page": 8,
                                  "section": "llm", "status": "supporting"}}
    back = B.uncorrected(shipped)
    assert back == shipped["corrected_from"]
    assert "corrected" not in back and back["section"] == "llm"
    # an untouched candidate is carried through unchanged, and copied not aliased
    plain = {"raw": "x", "section": "llm", "status": "supporting"}
    assert B.uncorrected(plain) == plain and B.uncorrected(plain) is not plain


#: The two documents whose ``llm_fallback`` flag the reuse path used to drop.
#: Both are two-page board notices with no A.x/B.2/C.1 template block: the
#: deterministic parser finds nothing, the model is called, and the model
#: returns nothing that survives the ``raw not in page`` verification. A fresh
#: build flags them (the call happened); the reuse path only set the flag
#: inside the branch that had a candidate to carry, so the flag disappeared on
#: the first rebuild and could never come back — a stem already present in the
#: seed is never re-queued for a call either. The registry then published both
#: lines with no extraction caveat on them at all.
CALLED_AND_EMPTY = ("193_gcf-b22-10-add01-rev01", "196_gcf-b19-22-add21-rev01")


def _seed_row(**coverage):
    return {"facts": {}, "coverage": {"era": "board notice (not a proposal "
                                      "template)", "pages": 2, "fields": 1,
                                      **coverage}}


def test_the_reuse_path_keeps_the_flag_of_a_call_that_returned_nothing():
    """THE FLAG IS A CALL RECORD, NOT A CANDIDATE COUNT.

    ``llm_fallback()`` sets it on every document it SENDS to the model, before
    it knows whether anything verifiable comes back; ``registry.
    _extraction_flags`` publishes it as 'the values on this line came from a
    fallback extraction'. A reuse rebuild that conditions the flag on a
    surviving candidate therefore un-says something the build said, for
    exactly the documents where the caveat matters most: the ones the model
    could not read either.

    Both real documents are named, and both directions are pinned — the flag
    survives with no candidates, and a document the previous build never
    called does not acquire one.
    """
    paths = [Path(f"{stem}.md") for stem in CALLED_AND_EMPTY]
    paths.append(Path("never_called.md"))
    docs = {p.stem: {"facts": {}, "coverage": {"llm_fallback": False}}
            for p in paths}
    previous = {stem: _seed_row(llm_fallback=True) for stem in CALLED_AND_EMPTY}
    previous["never_called"] = _seed_row(llm_fallback=False)

    todo, reused = B.carry_forward_llm(paths, docs, previous, empty=paths)

    for stem in CALLED_AND_EMPTY:
        assert docs[stem]["coverage"]["llm_fallback"] is True, stem
    assert docs["never_called"]["coverage"]["llm_fallback"] is False
    # no candidate was carried, so nothing counts as reused ...
    assert reused == 0
    # ... and no call is re-spent: every stem is already in the seed
    assert todo == []


def test_the_reuse_path_still_carries_candidates_and_still_queues_new_work():
    """The other two arms of the same function, so the flag fix cannot be
    mistaken for a relaxation of either: a seed candidate is merged (and the
    flag comes with it), and a document that is empty and unknown to the seed
    is queued for a call."""
    carried = Path("188_gcf-b21-10-add06.md")
    fresh = Path("999_new-and-empty.md")
    docs = {p.stem: {"facts": {}, "coverage": {"llm_fallback": False}}
            for p in (carried, fresh)}
    previous = {carried.stem: {"coverage": {"llm_fallback": True}, "facts": {
        "title": [{"raw": "A Title", "section": "llm", "status": "supporting"}]}}}

    todo, reused = B.carry_forward_llm([carried, fresh], docs, previous,
                                       empty=[carried, fresh])

    assert reused == 1
    assert docs[carried.stem]["coverage"]["llm_fallback"] is True
    assert docs[carried.stem]["facts"]["title"][0]["raw"] == "A Title"
    assert docs[carried.stem]["coverage"]["fields"] == 1
    assert todo == [fresh]


def test_a_currency_between_figure_and_scale_binds_the_scale():
    """FP155 p.8 prints '25 USD million' (cross-check stop #3): the currency
    sits between the figure and its scale word, so the adjacent <unit> group
    never binds and the canonical went unvalued - polluting the 2021 totals.
    The binder is narrow: only a currency token may separate them, and the
    clash guard still wins ('28,654 USD million' stays raw-without-value)."""
    import build_registry_v2 as b
    assert b.read_amount("25 USD million")["value"] == 25_000_000
    assert b.read_amount("25 USD")["value"] is None
    assert b.read_amount("28,654 USD million")["value"] is None
    assert b.read_amount("25 million USD")["value"] == 25_000_000
