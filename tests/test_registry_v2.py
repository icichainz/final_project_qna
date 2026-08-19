"""Schema-2 registry: deterministic template parsing and the v2 accessors.

The parser tests run on synthetic markdown shaped like the real corpus (page
markers, GCF template headings, VLM noise) so they document the heading
variants the builder supports. The accessor tests use a synthetic v2 dict via
registry._cache_v2, the parallel of the v1 registry._cache pattern.
"""
import importlib.util
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
