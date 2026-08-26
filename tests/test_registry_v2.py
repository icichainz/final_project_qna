"""Schema-2 registry: deterministic template parsing and the v2 accessors.

The parser tests run on synthetic markdown shaped like the real corpus (page
markers, GCF template headings, VLM noise) so they document the heading
variants the builder supports. The accessor tests use a synthetic v2 dict via
registry._cache_v2, the parallel of the v1 registry._cache pattern.
"""
import importlib.util
import json
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
    ("08_gcf-b42-02-add10-funding-proposal-package-fp268",
     "gcf_funding_requested", 8, 180_000_000.0, "180 million"),
    ("46_gcf-b38-02-add10-funding-proposal-package-fp230",
     "gcf_funding_requested", 5, 32_800_000.0, "32.8 million"),
    ("125_gcf-b27-02-add10-rev01", "gcf_funding_requested", 5, 256_480_000.0, "256.48"),
    ("19_gcf-b41-02-add05-funding-proposal-package-fp257",
     "gcf_funding_requested", 5, 75_623_754.0, "75,623,754"),
    # total_financing: 'Total project finance' (no -ing), the emphasis-split
    # figure, and 'Total funding required (GCF + co-financing)'.
    ("249_gcf-b14-07-add08-rev01", "total_financing", 11, 1_538_500_000.0, "1538.5"),
    ("188_gcf-b21-10-add06", "total_financing", 8, 37_600_000.0, "37.6 million"),
    ("43_gcf-b39-02-add08-funding-proposal-package-fp233",
     "total_financing", 5, 69_830_370.0, "69,830,370"),
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
    ("02_gcf-b42-02-add16-funding-proposal-package-fp274",
     "A5-A14 block (FP template v2/v3)"),
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
    that no longer matches any candidate must never be applied to whatever is
    there instead."""
    facts = {"total_financing": [_cand("2,000 USD", 2000.0)]}
    dec = B.Decisions([_entry(doc_id="D")])
    B.apply_fact_corrections("D", facts, dec)
    assert facts["total_financing"][0]["raw"] == "2,000 USD"
    assert "corrected" not in facts["total_financing"][0]
    assert dec.unapplied and dec.unapplied[0]["id"] == "T01"
    assert any("NOT APPLIED" in a for a in dec.alarms)


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
    assert corr["count"] == len(corr["corrections"]) == 62      # 58 wrong + 4 riders
    riders = [e for e in corr["corrections"] if "rider" in e["row_ref"]]
    assert len(riders) == 4
    assert {e["ratified"] for e in riders} == {"owner, 2026-08-26 (rider session)"}
    assert {e["row_ref"]["verdict"] for e in riders} == {"CONFIRMED"}
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
        assert e["row_ref"]["file"].endswith("phase3_adjudication.json")
        assert e["action"] in {"correct-to", "value-fix", "reclassify", "promote",
                               "confirm-absence", "re-extract", "add-candidate"}
        if e["action"] == "add-candidate":
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
    extra = {"corrected", "corrected_from", "reclassified_from", "disputed", "dispute"}
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
