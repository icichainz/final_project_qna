"""Claim verification (plan step 5). The verifier is a PURE DETECTOR.

No test here touches the network: the LLM layer — one batched judge call, and
only that — is exercised through fake clients, and the degradation tests
remove OPENAI_API_KEY outright. The fixture evidence set is deliberately tiny
— four keys — so every verdict in this file can be checked by reading the
fixture.

WHAT IS NOT TESTED HERE ANY MORE, and why the file is shorter than its
history. `verify.repair` and its adoption gates (the 817abdb probes, the
substance floor, the language gate, carry-off, the sampling pin ON THE
REWRITE) were deleted at eac4c94 together with the pass they guarded: when
repair was finally allowed to act, 3 of 5 adopted rewrites had deleted
verified evidence, and the cause was structural rather than a missing gate.
Every one of those tests went with its code. Where such a test ALSO pinned
something the detector still does, that half was kept and is marked at the
place it now lives:

  * the sampling pin, the degradation ladder and the call budget now ride on
    the JUDGE call, which is the only call this module makes;
  * `test_a_block_made_only_of_lead_ins_states_nothing` keeps the extraction
    property; `test_the_report_both_licence_is_not_a_licence_to_invent` keeps
    the report-both calibration; `test_a_page_only_repoint_of_a_wrong_figure_
    is_still_contradicted` keeps the `_conflict_before_support` finding.

A test that can only fail if a rewrite exists is not evidence about a
detector, so none was kept as a monument.
"""
import importlib.util
import json
import re
import pathlib

import pytest

from gcf_qna.rag import verify as V

DOC = "124_gcf-b27-02-add11"          # FP151 package
DOC2 = "123_gcf-b27-02-add12"         # FP152 package

REGISTRY_LINE = (
    'Registry — FP151: "Technical Assistance (TA) Facility for the Global '
    'Subnational Climate Fund"; accredited entity: International Union for '
    'Conservation of Nature and Natural Resources (IUCN); countries: Angola, '
    'Benin, Kenya; GCF financing (as printed): 18.5 M USD; total financing '
    f'(as printed): 28 M USD; board B.27, 2020 [{DOC}, cover pages]')

YEAR_NOTE = ("Note (computed from the corpus registry, which is complete — treat "
             "it as authoritative): 2020 — 30 proposals; 2014 — no board meeting, "
             "no registered proposals.")


@pytest.fixture
def evidence():
    return {
        (DOC, None): REGISTRY_LINE,
        (DOC, 45): ("### (a) Requested GCF funding (Total amount)\n"
                    "| (vi) Grants | 18,500,000 | 7 | |"),
        (DOC2, 5): ("Total GCF funding requested: USD 150 million\n"
                    "Accredited entity: Pegasus Capital Advisors LP"),
        V.NOTES_KEY: YEAR_NOTE,
    }


# ---------------------------------------------------------------------------
# number normalization
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tok,want", [
    ("49,751,264", 49751264.0),
    ("18,500,000", 18500000.0),
    ("46,10", 46.1),            # VLM-mangled decimal comma
    ("18,5", 18.5),             # French decimal comma
    ("358.26", 358.26),
    ("1 234 567", 1234567.0),   # space-grouped thousands
])
def test_to_number(tok, want):
    assert V.to_number(tok) == pytest.approx(want)


def test_unit_word_that_contradicts_its_mantissa_has_no_value():
    """'28,654 million USD' would be USD 28.7bn — above any GCF proposal ever
    approved. The scale is unknown, so only the mantissa stays comparable."""
    a = V.amounts("GCF financing (as printed): 28,654 million USD")[0]
    assert a.value is None and a.clash
    assert a.bare == pytest.approx(28654)
    other = V.amounts("Requested GCF amount 26,654 million USD")[0]
    assert not V.amount_matches(a, other)


def test_amount_matches_is_currency_aware_and_rounding_tolerant():
    eur = V.amounts("EUR 87 million")[0]
    usd = V.amounts("USD 87 million")[0]
    assert not V.amount_matches(eur, usd)
    assert V.amount_matches(V.amounts("USD 18.5 million")[0],
                            V.amounts("18,500,000")[0])
    assert not V.amount_matches(V.amounts("USD 18.5 million")[0],
                               V.amounts("25,000,000")[0])


def test_identifier_digits_are_not_amounts():
    """'GCF/B.27/02/Add.12' is one pointer, not three figures."""
    assert V.amounts("Document GCF/B.27/02/Add.12 is the funding proposal") == []


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------

def test_chained_citation_attaches_pages_to_the_nearest_preceding_doc():
    cits = V.parse_citations(f"figures [{DOC}, p. 5; {DOC2}, p. 6]")
    assert [(c.doc, c.page) for c in cits] == [(DOC, 5), (DOC2, 6)]


def test_cover_pages_is_registry_scope_with_no_page():
    (c,) = V.parse_citations(f"IUCN implements it [{DOC}, cover pages]")
    assert (c.doc, c.page, c.kind) == (DOC, None, "cover")


def test_note_bracket_without_a_document_is_a_note_citation():
    (c,) = V.parse_citations("[Registry — 30 funding-proposal documents from 2020]")
    assert c.doc is None and c.kind == "note"


def test_markdown_brackets_are_not_citations():
    assert V.parse_citations("see [the table] below and [1]") == []


def test_cited_sources_lists_pairs_in_order_without_duplicates():
    answer = (f"one [{DOC}, cover pages] two [{DOC}, p. 45] three "
              f"[{DOC}, p. 45] four [{DOC2}, p. 5]")
    assert V.cited_sources(answer) == [(DOC, None), (DOC, 45), (DOC2, 5)]


# ---------------------------------------------------------------------------
# claim extraction
# ---------------------------------------------------------------------------

def test_money_claim_carries_its_amount_and_citations():
    answer = (f"FP151 requests **USD 18.5 million** in GCF funding (a **grant**) "
              f"[{DOC}, cover pages; {DOC}, p. 45].")
    (claim,) = V.extract_claims(answer)
    assert claim.kind == "money"
    assert [a.value for a in claim.amounts] == [18500000.0]
    assert [(c.doc, c.page) for c in claim.citations] == [(DOC, None), (DOC, 45)]
    assert claim.required


def test_entity_claim_keeps_acronym_and_long_form_as_one_entity():
    answer = (f"FP151 is implemented by the accredited entity **International "
              f"Union for Conservation of Nature and Natural Resources (IUCN)**. "
              f"[{DOC}, cover pages]")
    (claim,) = V.extract_claims(answer)
    assert claim.kind == "entity"
    variants = claim.entities[0]
    assert "IUCN" in variants
    assert any("International Union" in v for v in variants)


def test_country_list_becomes_one_entity_per_country():
    answer = f"It covers **Angola, Benin and Kenya** [{DOC}, cover pages]."
    (claim,) = V.extract_claims(answer)
    names = [vs[0] for vs in claim.entities]
    assert {"Angola", "Benin", "Kenya"} <= set(names)


def test_french_decimal_comma_claim():
    answer = (f"FP151 demande **18,5 millions USD** de financement du FVC "
              f"[{DOC}, cover pages].")
    (claim,) = V.extract_claims(answer)
    assert claim.amounts[0].value == pytest.approx(18_500_000)


def test_bullets_and_table_rows_are_claim_units():
    answer = (f"- **FP086:** GCF financing **EUR 87 million** [{DOC}, cover pages]\n"
              f"| FP152 | USD 150 million | [{DOC2}, p. 5] |")
    claims = V.extract_claims(answer)
    assert [c.unit_kind for c in claims] == ["bullet", "table-row"]


def test_refusals_hedges_and_summaries_are_not_claims():
    answer = ("Retrieval did not surface the FP151 and FP152 funding proposal "
              "documents, so I can't compare their GCF funding.\n"
              "The excerpts do not explain why the country lists differ.\n"
              "In summary, two documents are involved.\n"
              "If you want, I can compare them once retrieval surfaces the pages.")
    assert V.extract_claims(answer) == []


def test_uncited_factual_sentence_is_still_a_claim():
    (claim,) = V.extract_claims("The total GCF funding requested is USD 150 million.")
    assert claim.kind == "money" and not claim.cited


def test_sentences_inherit_the_citation_of_their_paragraph():
    """Trailing-citation style: one bracket at the end of a paragraph covers
    the sentences before it. Attributing per sentence made every lead sentence
    an 'uncited claim' — 4 of the 22 gold-passing recorded answers."""
    answer = (f"FP086 is the **Green Cities Facility**. It is implemented by "
              f"the **European Bank for Reconstruction and Development** "
              f"[{DOC}, p. 45].")
    first, second = V.extract_claims(answer)
    assert [(c.doc, c.page) for c in first.citations] == [(DOC, 45)]
    assert first.inherited and not second.inherited


def test_citations_are_not_inherited_across_paragraphs_or_bullets():
    answer = (f"FP086 is the **Green Cities Facility**.\n\n"
              f"FP152 is the **Global Fund** [{DOC}, p. 45].")
    first, _ = V.extract_claims(answer)
    assert first.citations == [] and not first.inherited

    bullets = (f"- **FP086:** EUR 87 million [{DOC}, p. 45]\n"
               f"- **FP220:** USD 50,000,000")
    _, second = V.extract_claims(bullets)
    assert second.citations == []          # one bullet's source is not another's


def test_a_bulleted_list_borrows_the_citation_line_under_it():
    """A country list cites once, on its own line under the last bullet. Each
    bullet may borrow THAT (it is the list's source) — which is not the same
    as one bullet borrowing another bullet's inline citation."""
    answer = ("The Board approves the following host countries:\n"
              "- **Kazakhstan**\n"
              "- **Moldova**\n"
              f"[{DOC}, p. 4]")
    claims = V.extract_claims(answer)
    assert [cl.entities[0][0] for cl in claims] == ["Kazakhstan", "Moldova"]
    assert all([(c.doc, c.page) for c in cl.citations] == [(DOC, 4)]
               for cl in claims)
    assert all(cl.inherited for cl in claims)


def test_ellipsis_inside_a_sentence_does_not_orphan_its_citation():
    answer = ("The corpus holds **30 funding proposals from 2020** "
              f"(FP124 … FP153) [{DOC}, cover pages].")
    (claim,) = V.extract_claims(answer)
    assert [(c.doc, c.page) for c in claim.citations] == [(DOC, None)]
    assert not claim.inherited             # its own citation, not a borrowed one


def test_hedged_sentence_with_a_cited_figure_is_still_a_claim():
    """The glue filter must not swallow the figure it wraps."""
    answer = f"In summary, FP151 requests **USD 18,500,000** [{DOC}, p. 45]."
    (claim,) = V.extract_claims(answer)
    assert claim.kind == "money"
    assert claim.amounts[0].value == pytest.approx(18_500_000)


def test_hedges_without_a_cited_figure_are_still_dropped():
    assert V.extract_claims("In summary, two documents are involved.") == []


def test_identifier_dressed_as_a_name_is_not_an_entity():
    ents = [vs[0] for vs in V.entities("Funding Proposal FP173 sets up the "
                                       "**Amazon Bioeconomy Fund**")]
    assert "Amazon Bioeconomy Fund" in ents
    assert not any("FP173" in e for e in ents)


def test_single_word_cut_out_of_a_longer_name_is_dropped():
    ents = [vs[0] for vs in V.entities(
        'the **Amazon Bioeconomy Fund: Unlocking private capital** initiative')]
    assert "Unlocking" not in ents


def test_malformed_citation_never_crashes_extraction():
    answer = ("There are **30 funding proposals from 2020** in the corpus "
              "[Registry metadata line in prompt; computed registry note].")
    (claim,) = V.extract_claims(answer)
    assert claim.kind == "number"
    assert all(c.doc is None for c in claim.citations)


# ---------------------------------------------------------------------------
# HEADING FORM — adjudication ruling 6, and the escape closing it
#
# THE DEFECT: `extract_claims` minted markdown colon lead-ins and heading-form
# units as required fact-bearing claims. Twelve of the 71 adjudicated rows are
# that shape, they are most of the gold arm's remaining false positives, they
# inflate the claim-gate denominators, and the re-A/B at 9925a2c measured ~7
# of 22 repair rejections blocking on them with the judge answering "no claim
# text provided" — inherited failures no rewrite can clear.
#
# THE HARD SIDE IS THE FALSE-NEGATIVE ONE. Ruling 6 says the shape wins even
# when the unit carries a checkable proposition, so a rule that simply deletes
# the unit hands an attacker a place to hide a figure. Every test below pins
# ONE dimension, and the pairs vary the dimension rather than the example:
# same content with and without the form, same form with and without content.
# ---------------------------------------------------------------------------

LEAD_EV = {
    (DOC, None): REGISTRY_LINE,
    (DOC, 45): ("### (a) Requested GCF funding (Total amount)\n"
                "| (vi) Grants | 18,500,000 | 7 | |"),
}


def _one(answer):
    claims = V.extract_claims(answer)
    assert len(claims) == 1, [c.text for c in claims]
    return claims[0]


# ------------------------------------------------- the form test itself ----
@pytest.mark.parametrize("text,shape", [
    ("**Accredited entity**", "bold-label"),
    ("**Accredited entity:**", "bold-label"),
    ("__Financing__", "bold-label"),
    ("## Financing", "markdown-heading"),
    ("###### Financing", "markdown-heading"),
    ("The financing terms are as follows:", "colon-lead-in"),
    ("Key figures:", "colon-lead-in"),
    ("Les valeurs sont les suivantes :", "colon-lead-in"),
    (f"What the excerpts show [{DOC}, p. 45]:", "colon-lead-in"),
])
def test_the_shapes_rulings_1_2_and_6_name_are_recognised(text, shape):
    assert V.heading_form(text) == shape


@pytest.mark.parametrize("text", [
    # bold emphasis INSIDE prose is not a label
    "The accredited entity is **Pegasus Capital Advisors LP**.",
    "**IUCN** and **Pegasus** are both accredited.",
    # TWO SPANS, and the unit ends on the second one's marker. A label is ONE
    # span; a greedy `**.+**` swallows the prose between them and reads this
    # whole line as a label, which would delete both names from verification.
    "**IUCN** and **Pegasus Capital Advisors LP**",
    "**Angola**, **Benin** and **Kenya**",
    # a colon INSIDE a sentence, before a quotation, is not a lead-in
    'The cover page states: "total financing (as printed): 28 M USD".',
    # a bold span with a trailing comma is a value in a list, not a label
    "**GCF grant: USD 21.127 million**,",
    "",
    "   ",
])
def test_ordinary_prose_is_not_heading_form(text):
    assert V.heading_form(text) is None


def test_a_line_of_two_bold_spans_is_prose_and_keeps_its_names(no_registry):
    """The behavioural half of the two-span case: both names stay checked."""
    answer = ("**IUCN** and **Wakanda Development Bank**\n\n"
              f"- **USD 18,500,000** in GCF funding [{DOC}, p. 45]\n")
    claims = V.extract_claims(answer)
    assert len(claims) == 2, [c.text for c in claims]
    assert claims[0].text == "**IUCN** and **Wakanda Development Bank**"
    assert V.lead_ins(answer) == []


def test_a_bullet_or_table_row_is_never_a_lead_in():
    """DIMENSION: the unit kind, with the text held constant.

    `- **USD 50,000,000** [doc, p. 5]` is a bold-only span the moment its
    citation is stripped. `_units` already treats a list item as an
    independent statement, and reading one as a heading would delete the
    figure it states — the single most expensive thing this change could do.
    """
    line = "**USD 50,000,000**"
    assert V.heading_form(line, "sentence") == "bold-label"
    assert V.heading_form(line, "bullet") is None
    assert V.heading_form(line, "table-row") is None
    (claim,) = V.extract_claims(
        f"Figures:\n\n- **USD 18,500,000** [{DOC}, p. 45]\n")
    assert claim.kind == "money" and claim.unit_kind == "bullet"


# ------------------------------------------------ direction 1: it drops ----
def test_a_contentless_lead_in_produces_no_claim():
    answer = (f"The financing terms are as follows:\n\n"
              f"- **USD 18,500,000** in GCF funding [{DOC}, p. 45]\n")
    claims = V.extract_claims(answer)
    assert [c.text for c in claims] == [f"**USD 18,500,000** in GCF funding [{DOC}, p. 45]"]
    (li,) = V.lead_ins(answer)
    assert li.shape == "colon-lead-in" and li.carried_to == 0


def test_a_bold_label_over_a_block_produces_no_claim():
    answer = (f"**Accredited entity**\n"
              f"- **USD 18,500,000** in GCF funding [{DOC}, p. 45]\n")
    assert [c.text for c in V.extract_claims(answer)] == \
        [f"**USD 18,500,000** in GCF funding [{DOC}, p. 45]"]


def test_the_same_sentence_without_the_colon_is_still_a_claim(no_registry):
    """DIMENSION: the FORM, with the content held constant. The colon is the
    only difference between these two answers."""
    body = "FP151 was submitted by **Pegasus Capital Advisors LP**"
    dropped = V.extract_claims(f"{body}:\n\n- **USD 18,500,000** [{DOC}, p. 45]\n")
    kept = V.extract_claims(f"{body}.\n\n- **USD 18,500,000** [{DOC}, p. 45]\n")
    assert len(dropped) == 1 and len(kept) == 2
    assert kept[0].text.startswith("FP151 was submitted")


# --------------------------------- direction 2: its content is verified ----
def test_a_figure_hidden_in_a_lead_in_is_still_checked(no_registry):
    """THE ESCAPE, closed. A fabricated figure inside heading form must not
    walk out of verification with the heading."""
    answer = (f"**FP151 (total financing: USD 999 million)**:\n\n"
              f"- **USD 18,500,000** in GCF funding [{DOC}, p. 45]\n")
    claim = _one(answer)
    assert any(a.raw.startswith("999") or "999" in a.raw
               for a in claim.lead_in_amounts), claim.lead_in_amounts
    (v,) = V.classify(V.extract_claims(answer), LEAD_EV, use_llm=False)
    assert v.status == V.UNSUPPORTED, (v.status, v.reason)
    assert "999" in v.reason


def test_the_same_block_without_the_fabrication_verifies(no_registry):
    """The positive control for the test above: the gate is not vacuous and
    it is not always-on."""
    answer = (f"**FP151 (total financing: USD 28 million)**:\n\n"
              f"- **USD 18,500,000** in GCF funding [{DOC}, p. 45]\n")
    (v,) = V.classify(V.extract_claims(answer), LEAD_EV, use_llm=False)
    assert v.status == V.SUPPORTED, (v.status, v.reason)


def test_a_name_hidden_in_a_lead_in_is_still_checked(no_registry):
    """BOTH DIMENSIONS, not one. A single-dimension merge that carried only
    figures would let a fabricated NAME ride a heading over a figures-only
    block straight past the verifier."""
    answer = (f"**FP151 — Wakanda Development Bank**\n\n"
              f"- **USD 18,500,000** in GCF funding [{DOC}, p. 45]\n")
    claim = _one(answer)
    assert [vs[0] for vs in claim.lead_in_entities] == ["Wakanda Development Bank"]
    (v,) = V.classify(V.extract_claims(answer), LEAD_EV, use_llm=False)
    assert v.status == V.UNSUPPORTED and "Wakanda" in v.reason


def test_a_lead_in_naming_something_the_document_prints_still_verifies(no_registry):
    answer = (f"**FP151 — International Union for Conservation of Nature**\n\n"
              f"- **USD 18,500,000** in GCF funding [{DOC}, p. 45]\n")
    (v,) = V.classify(V.extract_claims(answer), LEAD_EV, use_llm=False)
    assert v.status == V.SUPPORTED, (v.status, v.reason)


def test_a_trailing_lead_in_with_content_is_verified_where_it_stands(no_registry):
    """Nothing follows it, so nothing completes its predicate and nothing can
    take its content: ruling 6's rationale does not reach it and it stays a
    claim rather than becoming a hiding place."""
    answer = (f"- **USD 18,500,000** in GCF funding [{DOC}, p. 45]\n\n"
              f"**FP151 — Wakanda Development Bank**")
    claims = V.extract_claims(answer)
    assert len(claims) == 2, [c.text for c in claims]
    assert V.lead_ins(answer) == []
    assert claims[1].text == "**FP151 — Wakanda Development Bank**"
    verdicts = V.classify(claims, LEAD_EV, use_llm=False)
    assert verdicts[1].status == V.UNSUPPORTED, verdicts[1].reason


def test_a_trailing_contentless_lead_in_adds_no_failure(no_registry):
    """The other half of the invariant: with nothing checkable to lose, a
    trailing lead-in simply disappears rather than becoming a failure."""
    answer = (f"- **USD 18,500,000** in GCF funding [{DOC}, p. 45]\n\n"
              f"**Notes:**")
    assert len(V.extract_claims(answer)) == 1
    res = V.verify_answer(answer, LEAD_EV, use_llm=False)
    assert res.status == "verified", (res.status, [v.reason for v in res.verdicts])


# ------------------------------------------------- what is NOT carried -----
def test_a_document_id_inside_a_label_is_not_carried_as_a_figure(no_registry):
    """MEASURED, not guessed: `**FP172 (103_gcf-b30-03-add04)**` yields the
    bare 'amounts' 103 and 03 out of a document identifier, and demanding the
    evidence print '103' would be a fabricated check, not a preserved one.
    Only money-like figures cross."""
    answer = (f"**FP151 ({DOC})**\n\n"
              f"- **USD 18,500,000** in GCF funding [{DOC}, p. 45]\n")
    claim = _one(answer)
    assert claim.lead_in_amounts == []
    (v,) = V.classify(V.extract_claims(answer), LEAD_EV, use_llm=False)
    assert v.status == V.SUPPORTED, (v.status, v.reason)


def test_content_the_unit_below_restates_is_not_carried_twice(no_registry):
    """Ruling 6's own rationale — the predicate is completed by the unit below
    it — so a term that unit already states is checked as its own."""
    answer = (f"**Total GCF funding: USD 18,500,000**\n\n"
              f"- **USD 18,500,000** in GCF funding [{DOC}, p. 45]\n")
    claim = _one(answer)
    assert claim.lead_in_amounts == []
    assert claim.lead_ins and claim.lead_ins[0].startswith("**Total GCF")


def test_a_lead_in_never_preempts_a_contradiction(no_registry):
    """PLACEMENT, pinned as its own dimension. The carried check gates SUPPORT
    and nothing else. Checking it earlier cost the contradicted arm a row
    (34/34 -> 33/34): the claim stopped reaching the conflict scan and came
    back UNSUPPORTED on the heading's name instead of CONTRADICTED on its own
    figure."""
    ev = {(DOC, 5): "GCF funding requested: 40,751,254 USD",
          (DOC, 48): "GCF funding requested: 38,000,000 USD",
          (DOC, 9): "C.1 — see the financing tables in section B."}
    answer = (f"**FP274 — Wakanda Development Bank**\n\n"
              f"- The **GCF funding requested** is **USD 40,751,254** [{DOC}, p. 9].\n")
    claim = _one(answer)
    assert claim.lead_in_entities
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.CONTRADICTED, (v.status, v.reason)
    assert "38,000,000" in v.reason


def test_the_carried_scope_is_the_document_not_the_page(no_registry):
    """MEASURED. Checking the carried content against the CITED PAGE newly
    flagged two claims the recorded release passed — a lead-in names what the
    block is about, and the honest question is whether the DOCUMENT prints it.
    Here the registry cover key names IUCN and the cited page does not."""
    answer = (f"**FP151 — International Union for Conservation of Nature**\n\n"
              f"- **USD 18,500,000** in GCF funding [{DOC}, p. 45]\n")
    claim = _one(answer)
    assert "iucn" not in V.norm_text(LEAD_EV[(DOC, 45)])
    ok_page, _ = V._check_lead_in(claim, LEAD_EV[(DOC, 45)])
    ok_doc, _ = V._check_lead_in(claim, "\n".join(LEAD_EV.values()))
    assert ok_page is False and ok_doc is True


# ------------------------------------------------------- glue, not form ----
def test_the_conversational_offer_is_glue_and_not_a_claim():
    """`claim-2bca31faa865e0d00c91c737`, adjudicated `not_a_claim`. It ends in
    a full stop and asserts nothing; the form test is the wrong place for it
    and the hedge list is the right one."""
    offer = ("If you meant a specific corpus among these (e.g., “RSF corpus” "
             "vs “Offshore Fund corpus”), tell me which one and I can answer "
             "precisely to the extent the excerpts include the relevant "
             "Board-meeting field.")
    assert V.heading_form(offer) is None
    assert V.extract_claims(offer) == []


def test_the_hedge_list_still_lets_a_cited_figure_through(no_registry):
    """The escape clause `claim_kind` already had, re-pinned against the new
    alternation: a hedge that still states a cited figure stays a claim."""
    answer = (f"I can answer that FP151 requests **USD 18,500,000** "
              f"[{DOC}, p. 45].")
    (claim,) = V.extract_claims(answer)
    assert claim.kind == "money"


# --------------------------------------------- one walk, two entry points --
def test_lead_ins_and_extract_claims_cannot_disagree():
    """They are produced by the same pass. A second reading of the answer
    would be a second opinion about which units ruling 6 removed, and a tool
    reconciling a recording against the tree would then be told two stories."""
    answer = ("**FP151 — the package**\n\n"
              f"- **USD 18,500,000** [{DOC}, p. 45]\n\n"
              "Key figures:\n\n"
              f"- **USD 28 million** total [{DOC}, cover pages]\n")
    claims = V.extract_claims(answer)
    lis = V.lead_ins(answer)
    assert [li.shape for li in lis] == ["bold-label", "colon-lead-in"]
    for li in lis:
        assert li.carried_to is not None
        assert li.text in claims[li.carried_to].lead_ins
        assert li.claim is not None and li.claim.text == li.text


# ------------------------------------- a block of nothing but lead-ins -----
def test_a_block_made_only_of_lead_ins_states_nothing():
    """THE DETECTOR HALF of what was `test_a_rewrite_made_only_of_lead_ins_is
    _not_substance`. That test asked what the deleted substance floor did with
    a page of headings; the property underneath it is ruling 6's and survives
    the floor: three colon lead-ins in a row are three lead-ins and zero
    claims, however many checkable-looking words they contain.

    Kept because it is the one place a MULTI-unit block of pure form is
    checked — the neighbouring tests each pin a single lead-in — and because
    an extraction that started minting claims here would put a fact-shaped
    verdict on text that states nothing."""
    block = ("The figures are as follows:\n\nDetails below:\n\n"
             "Summary of the above:")
    assert V.extract_claims(block) == []
    assert [li.shape for li in V.lead_ins(block)] == ["colon-lead-in"] * 3


# ---------------------------------------------------------------------------
# evidence assembly
# ---------------------------------------------------------------------------

class _Hit:
    def __init__(self, doc_id, page, text):
        self.doc_id, self.page, self.text = doc_id, page, text


def test_build_evidence_keys_hits_by_doc_and_page_and_notes_by_document():
    ev = V.build_evidence([_Hit(DOC, 45, "grants 18,500,000"),
                           _Hit(DOC, None, "cover text")],
                          [REGISTRY_LINE, YEAR_NOTE])
    assert (DOC, 45) in ev
    assert V.NOTES_KEY in ev and "2014" in ev[V.NOTES_KEY]
    # the registry line names its document, so it also becomes that document's
    # cover-page scope — which is exactly what '[doc, cover pages]' cites
    assert "18.5 M USD" in ev[(DOC, None)]


def test_build_evidence_attributes_matrix_rows_to_their_page():
    block = (f"FP151 -> {DOC} | \"TA Facility\"\n"
             f"FP151 | gcf_funding_requested | USD 18.5 million (p.5, A.8) | stated")
    ev = V.build_evidence([], block)
    assert "18.5 million" in ev[(DOC, 5)]


# ---------------------------------------------------------------------------
# deterministic classification
# ---------------------------------------------------------------------------

def test_supported_money_claim(evidence):
    answer = f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, cover pages]."
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.status == V.SUPPORTED and v.source == "deterministic"


def test_supported_across_number_formats(evidence):
    """'18,500,000' on the cited page and 'USD 18.5 million' in the answer are
    one fact; the French '18,5 millions USD' is the same fact again."""
    for text in ("**USD 18.5 million**", "**18,500,000 USD**", "**18,5 millions USD**"):
        answer = f"FP151 requests {text} in GCF funding [{DOC}, p. 45]."
        (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
        assert v.status == V.SUPPORTED, text


def test_contradicted_when_the_cited_field_prints_another_value(evidence):
    answer = f"FP151 requests **USD 25 million** in GCF funding [{DOC}, cover pages]."
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.status == V.CONTRADICTED
    assert "18.5 M USD" in v.reason


def test_unsupported_when_the_cited_page_was_never_retrieved(evidence):
    answer = f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, p. 99]."
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.status == V.UNSUPPORTED
    assert "never retrieved" in v.reason
    # certain already: an invented page needs no judge
    assert v.plausible is False


def test_unsupported_when_the_claim_carries_no_citation(evidence):
    answer = "The total GCF funding requested is USD 150 million."
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.status == V.UNSUPPORTED and "no citation" in v.reason
    # the figure does exist in the evidence, so this one is worth adjudicating
    assert v.plausible is True


def test_unsupported_entity_names_the_missing_name(evidence):
    answer = f"The accredited entity is **Pegasus Capital Advisors LP** [{DOC}, cover pages]."
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.status == V.UNSUPPORTED
    assert "Pegasus" in v.reason


def test_page_mismatch_is_supported_but_flagged(evidence, no_registry):
    """A real figure attached to the wrong page of the right document is a
    citation defect, not an invention — and must not read the same.

    'total financing 28 M USD' is printed on the registry cover line, not on
    the cited p.45.

    `no_registry` was added when the promotion branch started running the
    conflict gate, and it is not a workaround: the corpus registry files a
    real CONFLICT for this document's total_financing (28 M USD p.5 vs
    $720,000,000 p.60), so with the registry live this sentence states one
    side of a known conflict and is CONTRADICTED — as the SAME sentence citing
    '[doc, cover pages]' already was on every earlier revision. That
    asymmetry was the defect, and it is pinned in
    `test_a_page_repoint_cannot_escape_a_registry_recorded_conflict` below.
    What this test owns is the other half: with nothing else disagreeing, the
    wrong page is a caution and not a failure.
    """
    answer = f"Total financing is **28 M USD** [{DOC}, p. 45]."
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.status == V.SUPPORTED
    assert "citation-page-mismatch" in v.flags
    res = V.RepairResult(answer, "verified", [v], answer)
    assert res.cautions == [v]              # shown as a caution, not a failure


def test_page_only_citation_is_broken_not_a_note_citation(evidence):
    """'[p. 5]' names no document. Letting it land on the computed notes made
    every page-less page reference verify against a year note."""
    answer = "FP151 requests **USD 18.5 million** in GCF funding [p. 5]."
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.status == V.UNSUPPORTED
    assert v.scope == [] and any("invalid-citation" in f for f in v.flags)


def test_note_level_citation_resolves_to_the_computed_notes(evidence):
    answer = ("There are **30 funding proposals from 2020** in the corpus "
              "[computed registry note in the context].")
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.status == V.SUPPORTED
    assert v.scope == [V.NOTES_KEY]


def test_known_document_conflict_is_contradicted_even_when_the_page_is_absent(
        monkeypatch, evidence):
    """FP153's second figure lives on p.48, which retrieval need not surface.
    The fact registry scanned every page, so the conflict is still catchable.
    """
    monkeypatch.setattr(
        "gcf_qna.rag.registry.facts",
        lambda doc: {"gcf_funding_requested": [
            {"raw": "28,654 million USD", "value": None, "page": 5,
             "status": "canonical"},
            {"raw": "26,654 million USD", "value": None, "page": 48,
             "status": "conflicting"}]})
    ev = {(DOC, None): "GCF financing (as printed): 28,654 million USD"}
    answer = f"FP153 requests **28,654 million USD** in GCF funding [{DOC}, cover pages]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.CONTRADICTED
    assert "26,654 million USD" in v.reason and "p.48" in v.reason
    assert "known-document-conflict" in v.flags


def test_registry_conflict_check_can_be_switched_off(monkeypatch):
    monkeypatch.setattr("gcf_qna.rag.registry.facts",
                        lambda doc: {"gcf_funding_requested": [
                            {"raw": "28,654 million USD", "page": 5, "status": "canonical"},
                            {"raw": "26,654 million USD", "page": 48,
                             "status": "conflicting"}]})
    ev = {(DOC, None): "GCF financing (as printed): 28,654 million USD"}
    answer = f"FP153 requests **28,654 million USD** in GCF funding [{DOC}, cover pages]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False,
                      registry_conflicts=False)
    assert v.status == V.SUPPORTED


def test_registry_backed_figure_on_a_page_that_was_not_retrieved(
        monkeypatch, evidence):
    """Reporting BOTH sides of a known conflict — what the answer prompt asks
    for — cites a page this turn may not hold. That must verify, or no
    conflict-aware answer could ever pass."""
    monkeypatch.setattr(
        "gcf_qna.rag.registry.facts",
        lambda doc: {"gcf_funding_requested": [
            {"raw": "28,654 million USD", "page": 5, "status": "canonical"},
            {"raw": "26,654 million USD", "page": 48, "status": "conflicting"}]})
    ev = {(DOC, None): "GCF financing (as printed): 28,654 million USD"}
    answer = (f"FP153's cover page states **28,654 million USD** "
              f"[{DOC}, cover pages], while p. 48 states **26,654 million USD** "
              f"[{DOC}, p. 48].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED
    assert "registry-backed-page-not-retrieved" in v.flags
    assert "p.48" in v.reason


def test_registry_backed_name_on_a_page_that_was_not_retrieved(monkeypatch):
    """Same rescue for text facts: the cited document IS the Ecuador REDD+
    proposal, even when the ten passages this turn holds never print the
    country name."""
    monkeypatch.setattr("gcf_qna.rag.registry.load",
                        lambda: {DOC: {"title": "Ecuador REDD+ RBP for results "
                                                "period 2014",
                                       "countries": ["Ecuador"]}})
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: {})
    ev = {(DOC, 12): "The working group met to review the results period."}
    answer = f"It reports REDD+ working-group meetings in **Ecuador** [{DOC}, p. 12]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED
    assert "registry-backed-page-not-retrieved" in v.flags


def test_a_name_absent_from_evidence_and_registry_stays_unsupported(monkeypatch):
    monkeypatch.setattr("gcf_qna.rag.registry.load", lambda: {DOC: {}})
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: {})
    ev = {(DOC, 12): "The working group met to review the results period."}
    answer = f"It reports meetings in **Uzbekistan** [{DOC}, p. 12]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.UNSUPPORTED


def test_a_figure_absent_from_evidence_and_registry_stays_unsupported(
        monkeypatch, evidence):
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: {})
    answer = f"FP151 requests **USD 61 million** [{DOC}, cover pages]."
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.failed


def test_classification_never_raises_on_garbage():
    weird = "[[[ **USD** ]] 1,,2.3.4 [doc, p. ] — 2020 [" * 3
    V.classify(V.extract_claims(weird), {}, use_llm=False)


# ---------------------------------------------------------------------------
# the adversarial set: three defects the verifier must catch, one it must not
# ---------------------------------------------------------------------------

ADVERSARIAL = {
    "invented-page": (f"FP151 requests **USD 18.5 million** in GCF funding "
                      f"[{DOC}, p. 99].", V.UNSUPPORTED),
    "wrong-figure": (f"FP151 requests **USD 25 million** in GCF funding "
                     f"[{DOC}, cover pages].", V.CONTRADICTED),
    "uncited-figure": ("The total GCF funding requested is USD 150 million.",
                       V.UNSUPPORTED),
    "french-decimal-comma": (f"FP151 demande **18,5 millions USD** de "
                             f"financement du FVC [{DOC}, cover pages].",
                             V.SUPPORTED),
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_adversarial_answers(name, evidence):
    answer, expected = ADVERSARIAL[name]
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.status == expected, f"{name}: {v.reason}"


def test_french_decimal_comma_against_english_million():
    """'18,5 millions USD' and '18.5 million' are one figure in two locales;
    a verifier that reads the comma as a thousands group would call the
    French answer a 185x overstatement."""
    ev = {(DOC, 5): "Total GCF funding requested: 18.5 million USD"}
    answer = f"FP151 demande **18,5 millions USD** de financement [{DOC}, p. 5]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED


# ---------------------------------------------------------------------------
# LLM layer, mocked
# ---------------------------------------------------------------------------

class FakeClient:
    """Minimal stand-in for the OpenAI client: replies in order, records calls."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                content = outer.replies.pop(0) if outer.replies else ""
                if isinstance(content, Exception):
                    raise content
                msg = type("M", (), {"content": content})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def test_adjudication_merges_judge_verdicts_into_the_deterministic_ones(evidence):
    # separate paragraphs: the first claim must stay uncited (citations are
    # inherited within a paragraph, not across one)
    answer = ("The total GCF funding requested is USD 150 million.\n\n"
              f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, p. 45].")
    claims = V.extract_claims(answer)
    client = FakeClient(json.dumps({"verdicts": [
        {"id": 0, "status": "supported", "reason": "page 5 states it"}]}))
    verdicts = V.classify(claims, evidence, client=client)
    assert len(client.calls) == 1                     # ONE batched call
    assert verdicts[0].status == V.SUPPORTED and verdicts[0].source == "llm"
    assert verdicts[1].status == V.SUPPORTED and verdicts[1].source == "deterministic"


def test_judge_receives_the_passage_that_carries_the_value(evidence):
    """An uncited claim has no scope. Handing the judge '(no evidence held)'
    can only get the deterministic verdict rubber-stamped, so it is sent the
    passage where the value actually appears."""
    answer = "The total GCF funding requested is USD 150 million."
    client = FakeClient(json.dumps({"verdicts": []}))
    V.classify(V.extract_claims(answer), evidence, client=client)
    sent = client.calls[0]["messages"][-1]["content"]
    assert "Total GCF funding requested: USD 150 million" in sent
    assert "no evidence held" not in sent


def test_adjudication_is_skipped_when_nothing_is_plausible(evidence):
    answer = f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, p. 99]."
    client = FakeClient(json.dumps({"verdicts": []}))
    verdicts = V.classify(V.extract_claims(answer), evidence, client=client)
    assert client.calls == []                          # a certain verdict is free
    assert verdicts[0].status == V.UNSUPPORTED


def test_malformed_judge_reply_keeps_the_deterministic_verdicts(evidence):
    answer = "The total GCF funding requested is USD 150 million."
    client = FakeClient("I think claim 0 is fine, honestly")
    (v,) = V.classify(V.extract_claims(answer), evidence, client=client)
    assert v.status == V.UNSUPPORTED and v.source == "deterministic"


def test_judge_call_failure_is_not_an_answer_failure(evidence):
    answer = "The total GCF funding requested is USD 150 million."
    client = FakeClient(RuntimeError("502 upstream"))
    (v,) = V.classify(V.extract_claims(answer), evidence, client=client)
    assert v.status == V.UNSUPPORTED


# ---------------------------------------------------------------------------
# the statuses a detector can return
#
# `verified` / `partial` / `abstain` / `unverified-llm`, and nothing else:
# `repaired` was the fifth and it went with the pass that produced it. Each
# test below passes a client, so `_status_for` sees an LLM as available and
# the distinction being made is about the VERDICTS, not about the key.
# ---------------------------------------------------------------------------


# NOTE: nine repair tests lived here (adoption, invented sources, the
# figure-swap, re-attribution, gutting, the preamble strip, the independent
# switches, deletion-as-repair) and one carry-on test before them. They tested
# `verify.repair`, which no longer exists. None of them asserted anything
# about extraction, classification or the judge that the tests around them do
# not already assert, so nothing was salvaged from them.


def test_abstain_when_every_fact_bearing_claim_failed(evidence):
    answer = (f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 99].\n"
              f"Its accredited entity is **Pegasus Capital Advisors LP** "
              f"[{DOC}, cover pages].")
    client = FakeClient(json.dumps({"verdicts": []}))   # the judge moves nothing
    res = V.verify_answer(answer, evidence, client=client)
    assert res.status == "abstain"
    assert len(res.failures) == 2
    assert res.answer == answer and not res.repaired   # abstain shows the text


def test_partial_when_one_of_two_claims_stays_unsupported(evidence):
    answer = (f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, p. 45].\n"
              f"FP151 also received **USD 40 million** in co-financing [{DOC}, p. 99].")
    client = FakeClient(json.dumps({"verdicts": []}))  # the judge moves nothing
    res = V.verify_answer(answer, evidence, client=client)
    assert res.status == "partial"
    assert res.answer == answer and not res.repaired   # partial ships as written
    assert len(res.unsupported) == 1
    assert res.counts()[V.SUPPORTED] == 1


def test_at_most_one_llm_call_per_answer(evidence):
    """Was `test_at_most_two_llm_calls_per_answer` (1 adjudicate + 1 repair).
    The budget is now ONE, and the test is kept because the budget is the
    thing worth pinning: a second call appearing here would mean something in
    this module started talking to a model behind the operator's back."""
    answer = ("The total GCF funding requested is USD 150 million.\n\n"
              f"FP151 requests **USD 25 million** in GCF funding [{DOC}, cover pages].")
    client = FakeClient(json.dumps({"verdicts": [{"id": 0, "status": "unsupported",
                                                 "reason": "not stated"}]}),
                        "a second reply that must never be asked for")
    res = V.verify_answer(answer, evidence, client=client)
    assert len(client.calls) == 1                      # the judge, and nothing else
    assert res.answer == answer


def test_the_judge_token_budget_is_sized_to_the_batch(evidence):
    """THE DETECTOR HALF of `test_repair_token_budget_is_sized_to_the_answer`.
    The sizing rule and the 'the model id comes from config, not a literal'
    pin were about the calls this module makes; one call is left, so they ride
    on it."""
    answer = ("The total GCF funding requested is USD 150 million.\n\n"
              "The accredited entity is Pegasus Capital Advisors LP.")
    client = FakeClient(json.dumps({"verdicts": []}))
    verdicts = V.classify_deterministic(V.extract_claims(answer), evidence)
    todo = [v for v in verdicts if v.status == V.UNSUPPORTED and v.plausible]
    assert len(todo) == 2, [(v.claim.text, v.status, v.plausible) for v in verdicts]
    V.adjudicate(verdicts, evidence, client=client)
    assert client.calls[0]["max_completion_tokens"] == 400 * len(todo) + 800
    assert client.calls[0]["model"]                     # config.CHAT_MODEL, not hardcoded


# ---------------------------------------------------------------------------
# degradation
# ---------------------------------------------------------------------------

def test_no_api_key_means_deterministic_verdicts_and_the_answer_as_written(
        monkeypatch, evidence):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer = f"FP151 requests **USD 25 million** in GCF funding [{DOC}, cover pages]."
    res = V.verify_answer(answer, evidence)
    assert res.status == "unverified-llm"
    assert res.answer == answer and not res.repaired
    assert res.contradicted and res.verdicts[0].source == "deterministic"


def test_no_api_key_still_reports_a_clean_answer_as_verified(monkeypatch, evidence):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer = f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, cover pages]."
    res = V.verify_answer(answer, evidence)
    assert res.status == "verified" and res.ok


def test_empty_evidence_flags_everything_without_crashing(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer = f"FP151 requests **USD 18.5 million** [{DOC}, p. 5]."
    res = V.verify_answer(answer, {})
    assert res.status == "unverified-llm" and res.unsupported


def test_sources_of_a_verified_answer_are_what_it_cites(evidence):
    answer = (f"FP151 requests **USD 18.5 million** in GCF funding "
              f"[{DOC}, cover pages; {DOC}, p. 45].")
    res = V.verify_answer(answer, evidence, use_llm=False)
    assert res.status == "verified"
    assert res.sources == [(DOC, None), (DOC, 45)]

# ===========================================================================
# Wave 2 — calibration against data/eval/release_release-1-adjudicated.jsonl
#
# What survives here is the subset that an ablation could show earning
# adjudicated rows, minus everything that two adversarial reviews showed
# promoting fabricated content. Five relaxations were implemented and then
# DELETED rather than patched:
#
#   ruling 7 (retrieval-scoped negatives)  excused a name BECAUSE it was
#       absent from every held key, and absence is what a fabrication looks
#       like — the condition selected for what it meant to exclude.  4 rows.
#   ruling 3 (registry-confirmed absence)  supported an uncited claim whose
#       rider appeared anywhere in the held set, any document, any field.
#       Attributing a rider needs a document; an uncited claim names none.  4 rows.
#   acronym pairing                        a two-way substring test let words
#       be appended to a real registry expansion, defeating all 51 indexed
#       acronyms.  1 row.
#   registry_settled                       treated registry SILENCE about a
#       figure as a ruling that it was compatible.  0 rows.
#   containment dedup                      once the acronym machinery was
#       gone it swallowed the only check on an invented expansion.  1 row.
#   board tokens from the cited document id  satisfied the token and left the
#       predicate unverified.  0 rows.
#
# Every test below states the dimension its adversarial twin varies. A test
# that varies some other dimension passes against code that is broken along
# the one that matters, which is how three rounds of this work went wrong.
# ===========================================================================

FAKE = "Wakanda Development Bank"

CONFLICT_FACTS = {"gcf_funding_requested": [
    {"raw": "40,511,264 USD", "page": 7, "status": "canonical"},
    {"raw": "49,751,264", "page": 8, "status": "conflicting"}]}

CONFLICT_EV = {(DOC, None): "GCF funding requested: 40,511,264 USD (p.7, A.8)",
               (DOC, 8): "A.10 Financial instruments (a) requested for the GCF "
                         "funding - Grant: 49,751,264"}


@pytest.fixture
def no_registry(monkeypatch):
    """No registry at all, so a verdict can only come from held evidence."""
    monkeypatch.setattr("gcf_qna.rag.registry.load", lambda: {})
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: {})


def _both_sides():
    return (f"- **USD 40,511,264** is the GCF funding requested (p.7, A.8) "
            f"[{DOC}, cover pages].\n"
            f"- **USD 49,751,264** (p.8, A.10 “Grant”) [{DOC}, p. 8].")


# ---------------------------------------------------------------------------
# ruling 5 — a doc-level bracket is satisfied by any held key of that document
#   authorising rows: claim-2270588f, claim-b854edf7, claim-c502764f,
#   claim-db773f2e, claim-fd05177c
# ---------------------------------------------------------------------------

def test_doc_level_citation_is_satisfied_by_another_held_key_of_that_document(
        no_registry):
    """RULING 5. The bracket names the document; the figure sits on p.40 of
    that same held document, and the claim's own prose says so."""
    ev = {(DOC, None): "Registry — FP274: GCF funding requested: 40,511,264 USD",
          (DOC, 40): "C.1 Total Financing (a) Received GCF funding … 40,751,254"}
    answer = f"**USD 40,751,254** appears in C.1(a) on p.40 [{DOC}]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED


def test_a_coarse_citation_is_reported_as_coarse(no_registry):
    """Ruling 5 forgives the coarse bracket; it does not hide it. Folding the
    widening into the strict scope made strict == wide, which dropped this
    caution from 14 of the 66 recorded answers AND switched off the
    intra-document conflict detector below."""
    ev = {(DOC, None): "Registry — FP274.",
          (DOC, 40): "C.1 (a) Received GCF funding … 40,751,254"}
    answer = f"**USD 40,751,254** appears in C.1(a) [{DOC}]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED
    assert "citation-page-mismatch" in v.flags


def test_a_doc_level_bracket_still_surfaces_an_intra_document_conflict(
        no_registry):
    """DIMENSION: whether the keys are tested one at a time. p.5 agrees and
    p.48 does not, and `_field_conflict` stops at the first page that agrees —
    so concatenating the keys never reaches the disagreeing one. CONTRADICTED,
    not merely flagged: a non-empty flag list would be satisfied by the
    citation-page-mismatch above and would pin nothing."""
    ev = {(DOC, None): "Registry — FP151.",
          (DOC, 5): "GCF funding requested: 40,751,254 USD",
          (DOC, 48): "GCF funding requested: 38,000,000 USD"}
    answer = f"The **GCF funding requested** is **USD 40,751,254** [{DOC}]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.CONTRADICTED, (v.status, v.reason, v.flags)
    assert "conflict-elsewhere-in-document" in v.flags


def test_doc_level_citation_to_a_document_that_does_not_entail_it_still_fails(
        no_registry):
    """DIMENSION: which document holds the figure. Same claim, same bracket
    shape, same figure."""
    ev = {(DOC, None): "Registry — FP274: GCF funding requested: 40,511,264 USD",
          (DOC, 40): "C.1 Total Financing (a) Received GCF funding … 40,511,264",
          (DOC2, 9): "C.1 Total Financing (a) Received GCF funding … 40,751,254"}
    answer = f"**USD 40,751,254** appears in C.1(a) on p.40 [{DOC}]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "40,751,254" in v.reason


def test_doc_level_widening_does_not_reach_a_document_never_cited(no_registry):
    ev = {(DOC, None): "Registry — FP151: GCF funding requested: 18.5 M USD",
          (DOC2, 5): "Total GCF funding requested: USD 150 million"}
    answer = f"FP151 requests **USD 150 million** in GCF funding [{DOC}]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed


# ---------------------------------------------------------------------------
# the citation-repointing escape, and the one gate that closes it
#
# THE INVARIANT: a verdict may not answer SUPPORTED on a scope it has not
# conflict-tested.  Support is decided in three places — the strictly cited
# keys, the rest of a page-less bracket's document (ruling 5), and, when
# neither holds the figure, the rest of a PAGED bracket's document (or the
# registry) — and the conflict test used to run in the first place only, plus
# ruling 5.  So the SAME false sentence changed verdict with its citation:
#
#   'FP151 requests USD 28 million in GCF funding [doc, cover pages]'
#       -> CONTRADICTED (the cover line prints 18.5 M USD for GCF financing
#          and 28 M USD for TOTAL financing: the answer transposed the fields)
#   'FP151 requests USD 28 million in GCF funding [doc, p. 45]'
#       -> SUPPORTED, 'value found in the cited document, but not on the cited
#          page', one caution and nothing else — on a page that prints
#          18,500,000 and never prints 28 million.
#
# Same figure, same document, a strictly WORSE citation, a passing verdict.
# While the repair pass existed, a rewrite whose only diff was the page number
# was adopted as a correction — but that was the consequence, not the defect.
# The defect is in VERIFICATION, it was live under VERIFY=1 with repair off,
# and deleting repair does not touch it: the tests below are the fix and they
# run on the detector alone.
#
# The tests below fix the direction (a repoint may only make a verdict worse,
# never better) and then pin, one dimension each, everything the gate must NOT
# do — because the cheapest way to close this is to over-close it.
# ---------------------------------------------------------------------------

#: the cover line prints BOTH of FP151's figures, under their own labels —
#: each one heading its own `;`-separated segment, which is the only way
#: `_field_lines` reads a label as the field being stated rather than as prose
_REPOINT_EV = {
    (DOC, None): ('Registry — FP151: "TA Facility"; '
                  "GCF financing (as printed): 18.5 M USD; "
                  "total financing (as printed): 28 M USD"),
    (DOC, 45): ("### (a) Requested GCF funding (Total amount)\n"
                "| (vi) Grants | 18,500,000 | 7 | |"),
    (DOC, 60): "B.2 (a) programme cost table — no financing total is stated here",
}


@pytest.mark.parametrize("cite", ["cover pages", "p. 45", "p. 60", None])
def test_no_citation_of_this_document_makes_the_transposed_figure_pass(
        no_registry, cite):
    """DIMENSION: the CITATION, with the figure held constant — the axis a
    figure-level check cannot see.

    The claim states the document's TOTAL-financing figure under the
    GCF-financing label.  Every way of pointing at the document that holds
    both figures must fail: the page that prints the rival, a page that prints
    neither, and the whole document.  p.45 is the recorded escape.
    """
    bracket = f"[{DOC}, {cite}]" if cite else f"[{DOC}]"
    answer = f"FP151 requests **USD 28 million** in GCF funding {bracket}."
    (v,) = V.classify(V.extract_claims(answer), _REPOINT_EV, use_llm=False)
    assert v.status == V.CONTRADICTED, (cite, v.status, v.reason, v.flags)
    assert "18.5 M USD" in v.reason


def test_the_widened_branch_is_scanned_per_key_not_as_one_blob(no_registry):
    """DIMENSION: how the widened keys are handed to the detector — the exact
    shape the previous attempt at this call site died on.

    `_scoped_field_conflict(c, evidence, strict + wide_only)` passed the UNION,
    and `_field_conflict` stops at the first page that AGREES, so the agreeing
    page inside the merged blob suppressed the disagreeing one: the reported
    row closed and the vector stayed open.  Here p.5 agrees with the claim and
    p.48 does not, and the claim cites a page that holds neither.
    """
    ev = {(DOC, 5): "GCF funding requested: 40,751,254 USD",
          (DOC, 48): "GCF funding requested: 38,000,000 USD",
          (DOC, 9): "C.1 — see the financing tables in section B."}
    answer = f"The **GCF funding requested** is **USD 40,751,254** [{DOC}, p. 9]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.CONTRADICTED, (v.status, v.reason)
    assert "38,000,000" in v.reason
    assert "conflict-elsewhere-in-document" in v.flags


def test_the_widened_branch_still_reports_a_wrong_page_as_a_wrong_page(
        no_registry):
    """FALSE-POSITIVE SIDE.  A coarse or wrong-page citation to a document
    that agrees with itself is a citation defect and must stay one: SUPPORTED,
    cautioned, not CONTRADICTED.  The gate may only fire on the claim's OWN
    field label."""
    ev = {(DOC, 5): "GCF funding requested: 40,751,254 USD",
          (DOC, 48): "Co-financing: 38,000,000 USD",
          (DOC, 9): "C.1 — see the financing tables in section B."}
    answer = f"The **GCF funding requested** is **USD 40,751,254** [{DOC}, p. 9]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED, (v.status, v.reason)
    assert v.flags == ["citation-page-mismatch"]


def test_a_page_repoint_cannot_escape_a_registry_recorded_conflict(monkeypatch):
    """DIMENSION: the source of the conflict — held page text above, the fact
    registry here.  `registry_conflict` is part of the same gate, so it too
    used to be skipped by the widened branch: the identical sentence was
    CONTRADICTED at '[doc, cover pages]' and SUPPORTED at '[doc, p. 8]'."""
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: CONFLICT_FACTS)
    ev = dict(CONFLICT_EV)
    ev[(DOC, 3)] = "A.1 — this page states no financing figure at all."
    for cite in ("cover pages", "p. 3"):
        answer = (f"- **USD 40,511,264** is the GCF funding requested "
                  f"[{DOC}, {cite}].")
        (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
        assert v.status == V.CONTRADICTED, (cite, v.status, v.reason)
        assert "known-document-conflict" in v.flags


def test_the_report_both_licence_survives_the_widened_branch(monkeypatch):
    """FALSE-POSITIVE SIDE, and the largest one: 23 adjudicated rows.

    The registry note says 'report both figures with their pages', an answer
    that obeys prints one figure per bullet, and a bullet may perfectly well
    reach the widened branch — its figure is on a page the bracket does not
    name.  The licence is ANSWER-scoped (`_reported_elsewhere` -> `also`), so
    it has to survive the gate on every branch, not only on the strict one.
    """
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: CONFLICT_FACTS)
    ev = dict(CONFLICT_EV)
    ev[(DOC, 3)] = "A.1 — this page states no financing figure at all."
    answer = (f"- **USD 40,511,264** is the GCF funding requested (p.7, A.8) "
              f"[{DOC}, p. 3].\n"
              f"- **USD 49,751,264** (p.8, A.10 “Grant”) [{DOC}, p. 8].")
    verdicts = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert len(verdicts) == 2
    assert all(v.status == V.SUPPORTED for v in verdicts), \
        [(v.status, v.reason) for v in verdicts]
    assert "citation-page-mismatch" in verdicts[0].flags


def test_a_registry_settled_rival_is_still_deferred_to_on_the_widened_branch(
        monkeypatch):
    """FALSE-POSITIVE SIDE.  `registry_ruled_compatible` is the one thing that
    may outrank a page-level disagreement — the registry read both prints and
    filed the rival 'supporting' (row `id-fp152-financing`: a per-project
    tranche far below the programme total).  It is applied per rival inside
    `_key_conflict`, so widening the scope must not widen past it."""
    facts = {"total_financing": [
        {"raw": "720 M USD", "page": 5, "status": "canonical"},
        {"raw": "$100,000,000", "page": 55, "status": "supporting"}]}
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: facts)
    ev = {(DOC, 5): "A7. Total financing (SCF + co-finance) 720 M USD",
          (DOC, 55): "(a) Total project financing: $100,000,000",
          (DOC, 3): "A.1 — this page states no financing figure at all."}
    answer = f"**Total financing** is **720 M USD** [{DOC}, p. 3]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED, (v.status, v.reason)
    assert "citation-page-mismatch" in v.flags


def test_the_registry_backed_branch_runs_the_same_gate(monkeypatch):
    """THE SIBLING BRANCH.  `registry-backed-page-not-retrieved` answers
    SUPPORTED from the registry for a figure no held key prints, and it used
    to be reached with only the cited page tested — the same hole one branch
    over.  Closing one and leaving the other is a relocation, not a fix."""
    facts = {"gcf_funding_requested": [
        {"raw": "26,736,295 USD", "page": 61, "status": "canonical"}]}
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: facts)
    ev = {(DOC, 3): "A.1 — this page states no financing figure at all.",
          (DOC, 48): "GCF funding requested: 38,000,000 USD"}
    answer = (f"The **GCF funding requested** is **USD 26,736,295** "
              f"[{DOC}, p. 3].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.CONTRADICTED, (v.status, v.reason, v.flags)
    assert "38,000,000" in v.reason
    # the branch that ALMOST supported it is still named on the verdict
    assert "registry-backed-page-not-retrieved" in v.flags
    # and with nothing disagreeing it is still SUPPORTED, as it must be
    (ok,) = V.classify(V.extract_claims(answer),
                       {(DOC, 3): ev[(DOC, 3)]}, use_llm=False)
    assert ok.status == V.SUPPORTED
    assert "registry-backed-page-not-retrieved" in ok.flags


def test_the_widened_conflict_scan_is_the_scan_cross_page_conflicts_names(
        no_registry):
    """The new scan is a CROSS-PAGE scan and is gated by the switch that
    exists to name cross-page scans — otherwise `cross_page_conflicts=False`
    would mean two different things on two branches, and the ablation that
    measures that switch would under-report."""
    ev = {(DOC, 5): "GCF funding requested: 40,751,254 USD",
          (DOC, 48): "GCF funding requested: 38,000,000 USD",
          (DOC, 9): "C.1 — see the financing tables in section B."}
    answer = f"The **GCF funding requested** is **USD 40,751,254** [{DOC}, p. 9]."
    claims = V.extract_claims(answer)
    (on,) = V.classify_deterministic(claims, ev)
    (off,) = V.classify_deterministic(claims, ev, cross_page_conflicts=False)
    assert on.status == V.CONTRADICTED and off.status == V.SUPPORTED


def test_a_widened_verdict_never_claims_a_scope_it_did_not_test(no_registry):
    """The invariant itself, asserted structurally rather than by example: for
    every SUPPORTED verdict, `_conflict_before_support` re-run over the keys
    the verdict published as its own scope must still find nothing.  A branch
    that widened its scope without widening its test fails here whatever
    fixture it was added for."""
    ev = {(DOC, None): "Registry — FP151: GCF financing (as printed): 18.5 M USD",
          (DOC, 45): "| (vi) Grants | 18,500,000 | 7 | |",
          (DOC, 48): "GCF funding requested: 38,000,000 USD",
          (DOC2, 5): "Total GCF funding requested: USD 150 million"}
    for answer in (f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, p. 45].",
                   f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, p. 47].",
                   f"FP152 requests **USD 150 million** in GCF funding [{DOC2}]."):
        for v in V.classify(V.extract_claims(answer), ev, use_llm=False):
            if v.status != V.SUPPORTED:
                continue
            assert V._conflict_before_support(
                v.claim, ev, v.scope, v.scope, []) is None, (answer, v.reason)


# ---------------------------------------------------------------------------
# 'report both figures' — a conflict the ANSWER itself resolves
#   the largest contributor: 23 adjudicated rows, including claim-03d4cab1,
#   claim-0a498b0a, claim-0c2cdfab, claim-0c5bcfd1, claim-0ceca63e,
#   claim-11d3a178, claim-1cdbc791, claim-218f0773, claim-2bdc51a5,
#   claim-4650af24, claim-4b104f74, claim-5b351da8, claim-60c200b4,
#   claim-6e77b89f, claim-79a9c71a, claim-bdadca96, claim-ca1c1388,
#   claim-d43027b0, claim-d9a64d9b, claim-e33e0ea0, claim-ea234d3b,
#   claim-ee76069f, claim-f8289306
#
# It has four gates. Each has a test below whose ONLY job is to fail when
# that gate is removed; the widening harness in
# test_every_gate_of_the_report_both_suppression_is_pinned proves they do.
# ---------------------------------------------------------------------------

def test_reporting_both_sides_of_a_known_conflict_is_not_a_contradiction(
        monkeypatch):
    """The registry note says, verbatim, 'report both figures with their
    pages'. An answer that obeys prints one figure per bullet, and the
    obeying bullets carry a SECTION rather than a repeated field label."""
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: CONFLICT_FACTS)
    verdicts = V.classify(V.extract_claims(_both_sides()), CONFLICT_EV, use_llm=False)
    assert len(verdicts) == 2
    assert all(v.status == V.SUPPORTED for v in verdicts), \
        [(v.status, v.reason) for v in verdicts]


def test_reporting_only_one_side_of_a_known_conflict_is_still_contradicted(
        monkeypatch):
    """DIMENSION: whether the other side is reported at all."""
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: CONFLICT_FACTS)
    answer = (f"- **USD 40,511,264** is the GCF funding requested (p.7, A.8) "
              f"[{DOC}, cover pages].")
    (v,) = V.classify(V.extract_claims(answer), CONFLICT_EV, use_llm=False)
    assert v.status == V.CONTRADICTED
    assert "known-document-conflict" in v.flags


#: gate 3a — the sibling must speak about the SAME DOCUMENT.
def test_gate_same_document(monkeypatch):
    """An answer comparing two proposals states many figures; one of them
    equalling this document's other side is not this document's conflict
    being reported."""
    monkeypatch.setattr("gcf_qna.rag.registry.facts",
                        lambda doc: CONFLICT_FACTS if doc == DOC else {})
    ev = {(DOC, None): "GCF funding requested: 40,511,264 USD (p.7, A.8)",
          (DOC2, 5): "A.8 Total GCF funding requested 49,751,264 USD"}
    answer = (f"- FP274's **GCF funding requested** is **USD 40,511,264** "
              f"[{DOC}, cover pages].\n"
              f"- FP152's **GCF funding requested** is **USD 49,751,264** "
              f"[{DOC2}, p. 5].")
    verdicts = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert verdicts[0].status == V.CONTRADICTED
    assert verdicts[1].status == V.SUPPORTED


#: gate 3b — a sibling claiming a DIFFERENT field is not reporting this one.
GATE_3B_FACTS = {"gcf_funding_requested": [
    {"raw": "40,751,254 USD", "page": 5, "status": "canonical"},
    {"raw": "38,000,000 USD", "page": 5, "status": "conflicting"}]}
GATE_3B_EV = {(DOC, 5): "GCF funding requested: 40,751,254 USD\n"
                        "Total financing: 38,000,000 USD"}
GATE_3B_ANSWER = (
    f"- The **GCF funding requested** is **USD 38,000,000** [{DOC}, p. 5].\n"
    f"- The **total financing** is **USD 40,751,254** [{DOC}, p. 5].")


def test_gate_different_field(monkeypatch):
    """DIMENSION: the FIELD LABEL, and nothing else — same document, same two
    figures, same bullets, same registry, both figures registry-recorded so
    the registry gate cannot be what rejects. The labels are swapped onto each
    other's value, and two transposed figures must not license each other.

    Both labels resolve to a REAL field (`gcf_financing`, `total_financing`).
    An earlier version of this test used 'total co-financing', which
    `claim_field` cannot see at all — `norm_text` strips the hyphen — so the
    field gate did no work in it and the test passed either way."""
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: GATE_3B_FACTS)
    assert V.claim_field("The **GCF funding requested** is x") == "gcf_financing"
    assert V.claim_field("The **total financing** is x") == "total_financing"
    verdicts = V.classify(V.extract_claims(GATE_3B_ANSWER), GATE_3B_EV, use_llm=False)
    assert verdicts[0].status == V.CONTRADICTED, \
        [(v.status, v.reason) for v in verdicts]


#: gate 3c — a claim whose chained brackets name several documents attributes
#: its figure to none of them in particular.
GATE_3C_EV = {(DOC, 5): "GCF funding requested: 40,751,254 USD",
              (DOC2, 7): "GCF funding requested: 12,000,000 USD"}
GATE_3C_ANSWER = (
    f"- The **GCF funding requested** is **USD 38,000,000** [{DOC}, p. 5].\n"
    f"- The **GCF funding requested** figures are **USD 40,751,254** and "
    f"**USD 12,000,000** [{DOC}, p. 5; {DOC2}, p. 7].")


def test_gate_single_document_attribution(monkeypatch):
    """DIMENSION: how many documents the sibling's bracket names. The sibling
    states this document's other side, with a real field label and a
    registry-recorded value — everything gate 3b and gate 3e ask for — but its
    chained bracket names two documents, so the figure is attributed to
    neither in particular and cannot stand as this document's other side."""
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: GATE_3B_FACTS)
    verdicts = V.classify(V.extract_claims(GATE_3C_ANSWER), GATE_3C_EV, use_llm=False)
    assert verdicts[0].status == V.CONTRADICTED, \
        [(v.status, v.reason) for v in verdicts]


#: gate 3e — the licence for 'report both' is the registry note that says so.
def test_gate_registry_records_the_other_side(no_registry):
    """DIMENSION: whether the registry records the sibling's figure for this
    field. Here the answer mislabels the document's other figure as leverage;
    the registry knows nothing of it, so it is not the conflict's other
    side."""
    ev = {(DOC, None): "Registry — FP151.",
          (DOC, 5): "GCF funding requested: 40,751,254 USD",
          (DOC, 48): "GCF funding requested: 38,000,000 USD"}
    answer = (f"- The **GCF funding requested** is **USD 40,751,254** [{DOC}, p. 5].\n"
              f"- Roughly **USD 38,000,000** was leveraged from partners [{DOC}, p. 48].")
    verdicts = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert any(v.failed for v in verdicts), [(v.status, v.reason) for v in verdicts]


# ---------------------------------------------------------------------------
# gate 3f — per-key conflict: what the registry filed about the RIVAL print
#
# ONE DIMENSION, three values. Everything else is byte-identical across the
# three tests: same document, same two pages, same evidence text, same answer,
# same canonical registry row. Only the status the registry gives the p.99
# print moves — supporting / conflicting / not recorded at all — because that
# is the only thing `registry_ruled_compatible` reads.
#
# The numbers are the Wave-0b Blocker-2 probe verbatim: a document whose
# registry knows 26,736,295 while p.99 prints 999,111,222 went CONTRADICTED ->
# SUPPORTED with empty flags under the deleted `registry_settled`, because it
# asked only about the CLAIM's figure and took the registry's SILENCE about
# the rival for a ruling. Row 3 below is that probe and it must stay red.
# ---------------------------------------------------------------------------

GATE_3F_EV = {(DOC, 8): "A7. Total financing: 26,736,295 USD",
              (DOC, 99): "A7. Total financing: 999,111,222 USD"}
#: the registry's elected reading, identical in every row below
_CANON = {"raw": "26,736,295 USD", "page": 8, "status": "canonical"}
_RIVAL = {"raw": "999,111,222 USD", "page": 99}


def _gate_3f(monkeypatch, rival_status, cite_page=8, ev=None,
             figure="USD 26,736,295"):
    """The same turn, with the registry filing the rival print as asked.

    ``cite_page`` selects the BRANCH of `classify_deterministic` that reaches
    `_key_conflict`: p.8 is the page that prints the claim's own figure, so the
    claim verifies and the conflict is looked for elsewhere; p.99 does not, so
    the claim takes the not-found branch, which runs its own conflict check and
    never consults `registry_conflict`. Both branches must be pinned — the
    first version of these tests covered only the first, and widening the gate
    to defer on a `conflicting` rival then failed nothing at all, because
    `registry_conflict` re-raised the same verdict behind its back.
    """
    cands = [dict(_CANON)] + ([dict(_RIVAL, status=rival_status)]
                              if rival_status else [])
    monkeypatch.setattr("gcf_qna.rag.registry.facts",
                        lambda doc: {"total_financing": cands})
    answer = f"The **total financing** is **{figure}** [{DOC}, p. {cite_page}]."
    (v,) = V.classify(V.extract_claims(answer), ev or GATE_3F_EV, use_llm=False)
    return v


def test_gate_3f_a_rival_the_registry_filed_supporting_is_settled(monkeypatch):
    """VALUE 1 — `supporting`. The registry read both prints and ruled that the
    rival is not a second reading of this field (its own build rule: a figure
    far below the canonical total is a component or a tranche, and supporting
    is "never an assertion of conflict"). That ruling, and only that ruling,
    outranks the page-level disagreement."""
    v = _gate_3f(monkeypatch, "supporting")
    assert v.status == V.SUPPORTED, (v.status, v.reason)
    assert "conflict-elsewhere-in-document" not in v.flags


def test_gate_3f_a_rival_the_registry_filed_conflicting_still_contradicts(
        monkeypatch):
    """VALUE 2 — `conflicting`. Same registry, same lookup, opposite ruling:
    the registry says these ARE two readings of one field. The flag is
    asserted, not just the status: widen the gate to defer on a `conflicting`
    rival and the row stays CONTRADICTED — `registry_conflict` re-raises it
    from behind the gate — but as `known-document-conflict`, losing the page
    that prints the other figure. Status alone cannot see that, which is why
    the first version of this test pinned nothing."""
    v = _gate_3f(monkeypatch, "conflicting")
    assert v.status == V.CONTRADICTED, (v.status, v.reason)
    assert v.flags == ["conflict-elsewhere-in-document"], v.flags
    assert "999,111,222" in v.reason


def test_gate_3f_registry_silence_about_the_rival_never_suppresses(monkeypatch):
    """VALUE 3 — not recorded. THE BLOCKER-2 PROBE. The registry knows the
    claim's figure and knows nothing whatever about p.99's. Silence is not a
    ruling that the two are compatible; the verdict must stay CONTRADICTED and
    must still carry the flag that says where the other print is."""
    v = _gate_3f(monkeypatch, None)
    assert v.status == V.CONTRADICTED, (v.status, v.reason)
    assert "999,111,222" in v.reason
    assert "conflict-elsewhere-in-document" in v.flags


def test_gate_3f_supporting_settles_the_not_found_branch_too(monkeypatch):
    """VALUE 1, other branch: the answer cites p.99, which does not print its
    figure, so the claim reaches the not-found branch. The rival is settled, so
    the page is simply the wrong page — reported as such."""
    v = _gate_3f(monkeypatch, "supporting", cite_page=99)
    assert v.status == V.SUPPORTED, (v.status, v.reason)
    assert "citation-page-mismatch" in v.flags


def test_gate_3f_conflicting_contradicts_where_the_registry_path_never_runs(
        monkeypatch):
    """VALUE 2, other branch — THE ONE THAT MAKES CLAUSE 3 LOAD-BEARING.

    `registry_conflict` is only consulted after a claim verifies, so on the
    not-found branch nothing re-raises the conflict behind the gate. Widen the
    gate to defer on a `conflicting` rival and this row goes CONTRADICTED ->
    SUPPORTED with `citation-page-mismatch`. It must stay red, and for the
    per-key reason rather than the registry's."""
    v = _gate_3f(monkeypatch, "conflicting", cite_page=99)
    assert v.status == V.CONTRADICTED, (v.status, v.reason)
    assert "known-document-conflict" not in v.flags
    assert "999,111,222" in v.reason


def test_gate_3f_silence_contradicts_on_the_not_found_branch_too(monkeypatch):
    """VALUE 3, other branch. Silence suppresses nothing here either."""
    v = _gate_3f(monkeypatch, None, cite_page=99)
    assert v.status == V.CONTRADICTED, (v.status, v.reason)


def test_gate_3f_the_answer_must_be_stating_the_registrys_own_reading(
        monkeypatch):
    """THE CLAUSE THAT KEEPS THE PAIR FROM INVERTING.

    The rival is the `supporting` p.99 print, exactly as in value 1 — but the
    answer states a figure the registry records nowhere. The registry elected
    26,736,295 and never blessed 555,777,999, so it has no ruling to defer to
    and the disagreement stands. Drop the canonical clause and this row goes
    SUPPORTED: any figure at all would inherit a tranche's licence."""
    ev = {(DOC, 42): "A7. Total financing: 555,777,999 USD",
          (DOC, 99): "A7. Total financing: 999,111,222 USD"}
    v = _gate_3f(monkeypatch, "supporting", cite_page=42, ev=ev,
                 figure="USD 555,777,999")
    assert v.status == V.CONTRADICTED, (v.status, v.reason)
    assert "999,111,222" in v.reason


def test_gate_3f_a_settled_print_does_not_excuse_the_next_one_on_its_page(
        monkeypatch):
    """The deference is per RIVAL, not per page. p.99 prints the settled
    999,111,222 and then an unrecorded 777,777,777 under the same label; the
    second is nobody's ruling and must still be reported. Turn the skip into
    an early return and this row goes SUPPORTED."""
    ev = {(DOC, 8): "A7. Total financing: 26,736,295 USD",
          (DOC, 99): "A7. Total financing: 999,111,222 USD\n"
                     "A7. Total financing: 777,777,777 USD"}
    v = _gate_3f(monkeypatch, "supporting", ev=ev)
    assert v.status == V.CONTRADICTED, (v.status, v.reason)
    assert "777,777,777" in v.reason


def test_gate_3f_is_off_when_registry_conflicts_is_off(monkeypatch):
    """`registry_conflicts=False` means 'decide from held evidence alone'. A
    registry-derived SUPPRESSION is registry influence exactly as a
    registry-derived contradiction is, so it must be off in that mode too —
    otherwise the flag stops meaning what it says and a caller that opted out
    of the registry silently keeps one of its rulings."""
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: {
        "total_financing": [dict(_CANON), dict(_RIVAL, status="supporting")]})
    claims = V.extract_claims(
        f"The **total financing** is **USD 26,736,295** [{DOC}, p. 8].")
    (on,) = V.classify_deterministic(claims, GATE_3F_EV)
    (off,) = V.classify_deterministic(claims, GATE_3F_EV, registry_conflicts=False)
    assert (on.status, off.status) == (V.SUPPORTED, V.CONTRADICTED)


def test_the_real_fp152_row_is_not_contradicted_by_its_own_annex():
    """The live regression, against the REAL registry and the REAL corpus text
    — no monkeypatch, so this row fails if `data/registry_v2.json` stops
    filing the p.55 print `supporting`.

    FP152 prints 'A7. Total financing (SCF + co-finance) 720 M USD' on p.5 and
    '(a) Total project financing: $100,000,000' on p.55, inside an E.2.2
    cost-per-tonne calculation whose next line reads '(b) Expected GCF
    contribution: $75,000,000' — half the programme's own 150 M USD, which is
    what makes the block a per-project template row rather than the programme
    total. Same words, different scope."""
    ev = {(DOC2, 5): "## A7. Total financing (SCF + co-finance) 720 M USD\n"
                     "## A8. Total GCF funding requested 150 M USD",
          (DOC2, 55): "### E.2.2. Estimated cost per t CO2-eq, defined as total "
                      "investment required to achieve the mitigation\n"
                      "(a) Total project financing: $100,000,000\n"
                      "(b) Expected GCF contribution: $75,000,000"}
    answer = (f"- **Total financing (SCF + co-finance):** **720 M USD** "
              f"[{DOC2}, p.5 (A7)]")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED, (v.status, v.reason)


def test_a_figure_no_evidence_prints_is_unsupported_even_when_both_are_reported(
        monkeypatch):
    """Two bullets cannot license each other. Suppression removes the
    CONTRADICTED verdict only; every figure still has to be printed."""
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: CONFLICT_FACTS)
    answer = _both_sides() + (
        f"\n- **USD 77,777,777** is the GCF funding requested on p.8 too "
        f"[{DOC}, p. 8].")
    verdicts = V.classify(V.extract_claims(answer), CONFLICT_EV, use_llm=False)
    assert [v.status for v in verdicts[:2]] == [V.SUPPORTED, V.SUPPORTED]
    assert verdicts[2].failed and "77,777,777" in verdicts[2].reason


def test_an_invented_second_figure_does_not_count_as_reporting_both(monkeypatch):
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: CONFLICT_FACTS)
    answer = (f"- **USD 40,511,264** is the GCF funding requested [{DOC}, cover pages].\n"
              f"- **USD 77,777,777** is the GCF funding requested [{DOC}, p. 8].")
    verdicts = V.classify(V.extract_claims(answer), CONFLICT_EV, use_llm=False)
    assert verdicts[0].status == V.CONTRADICTED
    assert verdicts[1].failed


def test_registry_silence_about_a_figure_is_not_a_ruling(monkeypatch):
    """A `registry_settled` escape once deferred to the registry whenever it
    recorded the claim's figure and had NOT marked the rival 'conflicting'.
    Silence is not a ruling: the rival may simply never have been scanned."""
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: {
        "gcf_funding_requested": [{"raw": "26,736,295", "page": 5,
                                   "status": "canonical"}]})
    ev = {(DOC, 5): "GCF funding requested: 26,736,295",
          (DOC, 99): "GCF funding requested: 999,111,222"}
    answer = f"The **GCF funding requested** is **26,736,295** [{DOC}, p. 5]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.CONTRADICTED and v.flags


# ---------------------------------------------------------------------------
# the relaxations that were DELETED — pinned so they cannot come back
# ---------------------------------------------------------------------------

NOT_FOUND_NOTE = ("Registry — FP999: NOT FOUND in the 273-document corpus "
                  "registry. Do not infer details for it from other documents.")


def test_a_retrieval_scoped_negative_is_not_support(no_registry):
    """RULING 7, deleted. The name is absent from every held key, which is
    what the ruling required — and what a fabrication looks like."""
    ev = {(DOC, 50): "Thematic track record: metros in Nagpur and Cairo."}
    answer = ("None of the retrieved excerpts mention a **metro system in "
              "Antarctica**.")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed


def test_a_negation_never_excuses_a_name_in_the_same_sentence(no_registry):
    """The shape the clause splitter was built for, and could not hold: with
    the punctuation stripped before splitting, 'Antarctica; it is a Wakanda
    Development Bank project' was ONE clause and the fabricated entity
    shipped as verified."""
    ev = {(DOC, 50): "Thematic track record: metros in Nagpur and Cairo."}
    answer = (f"None of the retrieved excerpts mention **Antarctica**; it is "
              f"a **{FAKE}** project [{DOC}, p. 50].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "Wakanda" in v.reason


def test_a_registry_confirmed_absence_is_not_support(no_registry):
    """RULING 3, deleted. An uncited factual claim has no citation, and that
    is the verdict — whatever a computed note says about the identifier."""
    ev = {V.NOTES_KEY: NOT_FOUND_NOTE}
    answer = "FP999 does not exist in this corpus (273-document registry)."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed


@pytest.mark.parametrize("rider,ev_extra", [
    ("the total co-financing is **USD 18.5 million**", {}),
    ("the accredited entity of FP151 is **IUCN**",
     {(DOC2, 9): "accredited entity: IUCN"}),
    (f"the accredited entity is **{FAKE}**", {}),
])
def test_an_absence_never_carries_a_rider(no_registry, rider, ev_extra):
    """DIMENSION: where the rider's value lives. Ruling 3's guard varied
    PRESENCE — it used a value present nowhere — so it never tested
    attribution. Each rider here is present in the held set under the wrong
    field, in the wrong document, or not at all."""
    ev = {V.NOTES_KEY: NOT_FOUND_NOTE,
          (DOC, 5): "A.8 Total GCF funding requested 18.5 M USD", **ev_extra}
    answer = f"FP999 does not exist in this corpus, and {rider}."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed


@pytest.mark.parametrize("expansion", [
    "Asian Development Bank of Wakanda",         # a real expansion, extended
    "Asian Development Bank",                    # the real expansion
    "Association of Displaced Bankers",          # invented, initials agree
])
def test_an_acronym_never_vouches_for_a_spelled_out_name(no_registry, expansion):
    """Acronym pairing, deleted. A two-way substring test against a registry
    index let words be APPENDED to a real expansion and stay a substring,
    defeating all 51 indexed acronyms. A page printing 'ADB' does not say
    what ADB stands for, so the spelled-out form is checked as printed."""
    ev = {(DOC, None): "Registry — FP: accredited entity: ADB"}
    answer = f"The accredited entity is **ADB ({expansion})** [{DOC}, cover pages]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed


def test_a_board_token_is_not_satisfied_by_the_cited_document_id(no_registry):
    """The board-token widening, deleted. A year/board claim is checked by its
    token and nothing else, so satisfying the token from the CITATION left the
    predicate unverified."""
    ev = {(DOC, 4): "Guidance on proposal formatting and the GCF Disclosure "
                    "Policy."}
    answer = (f"The retrieved **GCF/B.27/02/Add.11** was withdrawn by the "
              f"Board [{DOC}, p. 4].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed


def test_a_board_token_the_evidence_prints_is_supported(no_registry):
    ev = {(DOC, None): "Registry — FP151: … board B.27, 2020"}
    answer = f"FP151 was considered at **B.27** in **2020** [{DOC}, cover pages]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED


# ---------------------------------------------------------------------------
# the corpus's comma decimal mark
#   authorising row: claim-c83fbe25
# ---------------------------------------------------------------------------

COFIN = {(DOC, 140): "Co-financing: USD 28,025 million sourced as follows: "
                     "- Grant funding of USD 26,958 million from APFC; and "
                     "- Grant funding of USD 1,066 million from local government"}


def test_a_decimal_comma_with_a_scale_word_matches_the_same_figure(no_registry):
    """'USD 28,025 million' cannot be 2.8e10; its only other reading is the
    corpus's decimal comma, which is what the answer prints as 28.025."""
    answer = (f"**co-financing: USD 28.025 million** (including **USD 26.958 "
              f"million** from APFC and **USD 1.066 million** from local "
              f"government) [{DOC}, p. 140].")
    (v,) = V.classify(V.extract_claims(answer), COFIN, use_llm=False)
    assert v.status == V.SUPPORTED


def test_the_decimal_comma_reading_does_not_move_the_predicate(no_registry):
    """DIMENSION: the FIELD the figure is attributed to, not its digits. The
    figure is right and re-readable; the label it is filed under is not the
    one the page prints it beneath."""
    ev = {(DOC, 140): "Co-financing: USD 28,025 million sourced as follows\n"
                      "GCF funding requested: USD 21,127 million"}
    answer = (f"The **GCF funding requested** is **USD 28.025 million** "
              f"[{DOC}, p. 140].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed


def test_the_decimal_comma_reading_does_not_make_two_figures_equal(no_registry):
    ev = {(DOC, 141): "Co-financing: USD 28,025 million sourced as follows"}
    answer = f"**co-financing: USD 26.958 million** [{DOC}, p. 141]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed


def test_a_text_that_never_demonstrates_the_convention_does_not_get_it():
    """'USD 1,066 million' alone is a plausible 1.066 billion, and nothing in
    its text says otherwise. The alternative reading is taken from evidence,
    not assumed."""
    (lone,) = V.amounts("Grant funding of USD 1,066 million from local government")
    assert lone.value == pytest.approx(1.066e9) and lone.alt is None
    assert not V.amount_matches(lone, V.amounts("USD 1.066 million")[0])
    (spread,) = [a for a in V.amounts(
        "Co-financing: USD 28,025 million … of USD 1,066 million")
        if a.num == "1,066"]
    assert spread.alt == pytest.approx(1.066e6)
    assert V.amount_matches(spread, V.amounts("USD 1.066 million")[0])


def test_two_self_contradictory_scales_still_disagree():
    """The pin that must not move: this is the FP153 conflict."""
    a = V.amounts("GCF financing (as printed): 28,654 million USD")[0]
    b = V.amounts("Requested GCF amount 26,654 million USD")[0]
    assert not V.amount_matches(a, b)
    assert V.amount_matches(a, V.amounts("GCF: 28,654 million USD")[0])


# ---------------------------------------------------------------------------
# entity extraction — each pinned END TO END, through a verdict
#   authorising rows: claim-87ad9dbb (_all_generic), claim-e79ef060 (_trim_run)
# ---------------------------------------------------------------------------

def test_a_form_field_label_is_not_checked_as_a_name(no_registry):
    """'**Funding proposal ID / name:** FP86 — "Green Cities Facility"' bolds
    the LABEL. No page prints the label as the answer spells it."""
    ev = {(DOC, None): 'Registry — FP86: "Green Cities Facility"'}
    answer = (f'**Funding proposal ID / name:** **FP86 — "Green Cities '
              f'Facility"** [{DOC}, cover pages]')
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED


def test_a_name_that_merely_contains_a_generic_word_is_still_checked(
        no_registry):
    """DIMENSION: whether EVERY word is furniture. One generic word does not
    make a name into a label, and the name must still be found."""
    ev = {(DOC, None): 'Registry — FP86: "Green Cities Facility"'}
    answer = f"The **Climate Investment Fund Name** was approved [{DOC}]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "Climate Investment Fund Name" in v.reason


def test_a_capitalized_run_stops_at_its_last_capital(no_registry):
    """'one with Pegasus and one with IUCN' — the connective pulled the
    sentence's grammar into the name, and no page prints 'Pegasus and one'."""
    ev = {(DOC, 143): "implemented through two independent funded agreements: "
                      "one between Pegasus and GCF and another between IUCN "
                      "and GCF"}
    answer = (f"It describes two agreements—one with **Pegasus** and one with "
              f"**IUCN** [{DOC}, p. 143].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED


def test_a_connective_inside_a_real_institution_name_is_still_checked(
        no_registry):
    """DIMENSION: whether the run really continues after the connective. The
    trailing words are capitalized, so they belong to the name and must be
    found."""
    ev = {(DOC, 5): "The entity is the European Bank."}
    answer = (f"The entity is the **European Bank for Reconstruction and "
              f"Development** [{DOC}, p. 5].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "Reconstruction" in v.reason


def test_two_independent_names_are_both_checked(no_registry):
    ev = {(DOC, 5): "The programme operates in Kenya."}
    answer = f"The programme covers **Kenya** and **Uganda** [{DOC}, p. 5]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "Uganda" in v.reason


# ---------------------------------------------------------------------------
# the report-both licence, at the three answers that separate it
#
# THE DETECTOR HALF of `test_repair_gates_hold_against_the_report_both_
# relaxation` (817abdb). That test drove the three answers below through the
# repair pass and asked which rewrite was adopted; the calibration it was
# really pinning is a CLASSIFICATION one — an answer that reports both sides
# of a registry-known conflict verifies, one that reports a single side is
# contradicted, and one that reports a second figure no evidence prints is not
# rescued by looking like the first. Stated directly, it needs no rewrite.
# ---------------------------------------------------------------------------

def test_the_report_both_licence_is_not_a_licence_to_invent(monkeypatch):
    """Three answers, one evidence set, one registry row. The only thing that
    varies is what the answer says, which is the point: the licence is granted
    by the FIGURES being the ones the corpus prints, never by the shape of an
    answer that lists two of them."""
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: CONFLICT_FACTS)

    one_side = (f"FP274's **GCF funding requested** is **USD 40,511,264** "
                f"[{DOC}, cover pages].")
    assert V.verify_answer(one_side, CONFLICT_EV, use_llm=False).failures

    both = V.verify_answer(_both_sides(), CONFLICT_EV, use_llm=False)
    assert both.status == "verified" and not both.failures, both.verdicts

    invented = V.verify_answer(
        f"- **USD 40,511,264** [{DOC}, cover pages].\n"
        f"- **USD 77,777,777** [{DOC}, p. 8].", CONFLICT_EV, use_llm=False)
    assert invented.failures, "an invented second figure bought the licence"
    assert any("77,777,777" in v.reason for v in invented.failures), \
        [v.reason for v in invented.failures]


# ---------------------------------------------------------------------------
# the corpus registry the headline numbers depend on
# ---------------------------------------------------------------------------

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"
REGISTRY_FILES = ("registry.json", "registry_v2.json")


def test_the_registry_files_the_verifier_reads_are_present_and_tracked():
    """The conflict machinery reads data/registry.json and
    data/registry_v2.json. They were GIT-IGNORED, so a clean checkout scored a
    DIFFERENT system — gold precision 25.0% instead of 34.3%, and one
    held-correct claim newly flagged — with nothing to say it had happened.

    This fails loudly instead. If it is red, the tree is missing the corpus
    registry and every number derived from the verifier is measuring
    something else."""
    missing = [n for n in REGISTRY_FILES if not (DATA / n).exists()]
    assert not missing, (
        f"data/{{{', '.join(missing)}}} is absent. The verifier's conflict and "
        f"registry checks silently degrade without it and the suite would be "
        f"scoring a different system. Restore it from git (it is tracked via a "
        f"negative .gitignore pattern) and verify with "
        f"`sha256sum -c data/eval/CHECKSUMS.sha256`.")


def test_the_registry_files_are_covered_by_the_integrity_anchor():
    """Tracked is not enough — the anchor has to cover them, or a silent edit
    changes every headline number without failing anything."""
    anchor = (DATA / "eval" / "CHECKSUMS.sha256")
    if not anchor.exists():
        pytest.skip("checksum anchor not present")
    text = anchor.read_text()
    for name in REGISTRY_FILES:
        assert f"data/{name}" in text, f"data/{name} is not in {anchor}"


def test_the_registry_is_actually_loadable_and_non_empty():
    """A present-but-empty registry degrades exactly like an absent one."""
    from gcf_qna.rag import registry
    rows = registry.load()
    assert rows and len(rows) > 100, f"registry.load() returned {len(rows or {})} rows"


# ---------------------------------------------------------------------------
# the gold, and the seeded claims the gold cannot contain
# ---------------------------------------------------------------------------

GOLD = pathlib.Path(__file__).resolve().parents[1] / "data" / "eval"
SCORER = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "score_verifier.py"


def _scorer():
    if not SCORER.exists():
        pytest.skip("scripts/score_verifier.py not present")
    spec = importlib.util.spec_from_file_location("score_verifier", SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scored():
    for name in ("release_release-1-adjudicated.jsonl",
                 "release_release-1-evidence.jsonl", "release_release-1.jsonl"):
        if not (GOLD / name).exists():
            pytest.skip(f"{name} not present")
    sv = _scorer()
    rows, problems, resolved = sv.build_rows(
        sv.read_jsonl(GOLD / "release_release-1-adjudicated.jsonl"),
        sv.read_jsonl(GOLD / "release_release-1-evidence.jsonl"),
        sv.read_jsonl(GOLD / "release_release-1.jsonl"))
    res = sv.score(rows)
    res["resolved_by_extraction"] = resolved
    return sv, res, problems


#: The 14 adjudicated false positives Wave 2 deliberately does NOT clear.
#: Each was cleared at some point by a relaxation that a review then showed
#: promoting fabricated content; the relaxation was DELETED and the row left
#: flagged. This is a trade, not an oversight — see the module header.
KNOWN_UNCLEARED = {
    # board token inside a document identifier (the widening also licensed an
    # unverified predicate: 'GCF/B.27/02/Add.11 was withdrawn by the Board')
    "claim-0edfd39b9d09178723c29b1e", "claim-8dae35b9948cc2e32bd729ea",
    "claim-dbe325c0f9c5eb56cdd26cc3",
    # ruling 7, retrieval-scoped negatives (excused a name BECAUSE it was
    # absent from every held key)
    "claim-26b99fbf113f3cdb349272fb", "claim-58ccad0129a5219991e4fa4f",
    "claim-a5142ecad0be97f11f917d19", "claim-ea1c4101763608facc2fb2f9",
    # ruling 3, registry-confirmed absence (rider attributable to no document)
    "claim-229a5ced3090d6cc23c529ab", "claim-64176ae457904fd23b0edcf9",
    "claim-954578b2b365c11cd36a8e42", "claim-e9874a0de7e486f0a756f424",
    # acronym pairing (a two-way substring defeated all 51 indexed acronyms)
    "claim-67c782a8899254303657be34", "claim-7d7cbf5650cdee55e83b5678",
    # containment dedup, deleted with the acronym machinery it depended on
    "claim-a59cc16ed03d732e4b14a48f",
}

#: Two of the fourteen are uncited negatives whose subject appears nowhere in
#: the held evidence, so `plausible` is False and the judge is never asked.
#: That is PARENT behaviour for an uncited claim, not something Wave 2 did —
#: recorded here so the gap is named rather than assumed away.
NOT_JUDGE_REACHABLE = {
    "claim-a5142ecad0be97f11f917d19", "claim-ea1c4101763608facc2fb2f9",
}


#: THE ONE ADJUDICATED TRUE FAILURE THE NOTE-PAGE SCOPE OVERTURNS, named
#: rather than counted, and the only row in this file that costs a Wave-1
#: `verifier_correct: true` its flag. IT NEEDS AN OWNER'S SIGNATURE, because a
#: test cannot decide that a human adjudication is superseded.
#:
#:   claim-e0be9178189ce5084dfc7164, case `fr-cmp-currency`, label
#:   `wrong_citation`, verifier_correct true.
#:
#: The claim carries FOUR brackets. Three were exact; the fourth cites
#: '02_…fp274, p.7' for 40,511,264 USD, and p.7 was never retrieved. The
#: reviewer's own note ends: "The 40,511,264 figure was available in the held
#: cover-pages registry entry, so ruling 4 applies: cited page never held
#: while the fact sat elsewhere in held evidence. NOTE THE GENERATION COPIED
#: '(p.7, A.8)' STRAIGHT OUT OF THE REGISTRY NOTE."
#:
#: That last sentence is this change's whole subject. The registry note prints
#: 'GCF funding requested: 40,511,264 USD (p.7, A.8)'; the answer prompt at
#: HEAD instructs the model to cite the page a registry row prints; the app's
#: `_invalid_citations` and the harness's `score_answer` both count that page
#: as a legal target. Under the note-page scope the citation is COMPLIANCE,
#: and it is structurally identical to release-3's conf-fp153-gcf ('(p.48,
#: B.2(b))') and fr-fp172-nepal ('(p.76, B.2(b))'), which this change exists
#: to stop failing. There is no predicate that keeps this row flagged and lets
#: those two pass: the label was applied under ruling 4 as it stood in Wave 1,
#: before the rule that governs it existed.
#:
#: It is an EXACT set for the same reason `KNOWN_UNCLEARED` is: it fails when
#: a second adjudicated true failure stops being flagged just as loudly as it
#: fails when this one starts again.
#:
#: RATIFIED by the owner, 2026-08-25 (ruling 8 in
#: docs/wave1-adjudication-review.md): in the application as shipped, citing
#: the page a registry line prints is the prompt-mandated behaviour, so a
#: gold label that punishes it would make the instrument contradict the spec
#: it measures. The label stands as history; the rule it was judged under
#: has been deliberately superseded. Any second row still requires its own
#: ratification.
SUPERSEDED_BY_NOTE_PAGE_SCOPE = {"claim-e0be9178189ce5084dfc7164"}


def test_no_adjudicated_true_failure_stops_being_flagged():
    """The regression gate. Twelve rows were adjudicated 'the verifier was
    right'; if calibration silences one of them, the calibration is wrong.

    TEN of the twelve are still scored here. The other two left the join with
    the lead-in wave and are NOT silenced — they are named row by row in
    `RESOLVED_BY_EXTRACTION`, asserted by
    `test_the_two_rows_the_gold_still_calls_genuine_defects_are_named`, and
    their content is still verified through the claim below them. This test
    reads the rows that DO join; that test reads the ones that do not, so the
    pair still covers all twelve.

    ONE of the ten is no longer flagged, and it is not silenced either: it is
    `SUPERSEDED_BY_NOTE_PAGE_SCOPE`, written out above with the reviewer's own
    note, because the citation it was labelled `wrong_citation` for is the
    exact shape the note-page scope makes legal."""
    _sv, res, _p = _scored()
    missed = {r["row_id"] for r in res["rows"]
              if r["arm"] == "gold" and r["should_flag"] and not r["flagged"]}
    assert missed == SUPERSEDED_BY_NOTE_PAGE_SCOPE
    scored_true = sum(1 for r in res["rows"]
                      if r["arm"] == "gold" and r["should_flag"])
    resolved_true = sum(1 for r in res["resolved_by_extraction"]
                        if r["should_flag"])
    assert scored_true == 10 and resolved_true == 2, (scored_true, resolved_true)


def test_the_adjudicated_false_positives_are_cleared_or_named():
    """An EXACT-SET assertion, deliberately: it fails when a row stops being
    cleared AND when one starts, so neither a regression nor an unlogged
    widening can pass silently. This is the test that caught three separate
    widenings."""
    _sv, res, _p = _scored()
    still = {r["row_id"] for r in res["rows"]
             if r["arm"] == "gold" and r["label"] == "verifier_false_positive"
             and r["flagged"]}
    assert still == KNOWN_UNCLEARED


def test_the_uncleared_rows_still_reach_the_judge():
    """Not clearing them is only acceptable because the live path still asks
    the judge: an unsupported-but-plausible claim is exactly the residue
    `adjudicate` is given. A row that is neither cleared nor judge-reachable
    would be silently lost."""
    _sv, res, _p = _scored()
    unreachable = {r["row_id"] for r in res["rows"]
                   if r["row_id"] in KNOWN_UNCLEARED
                   and not (r["status"] == V.UNSUPPORTED and r["plausible"])}
    assert unreachable == NOT_JUDGE_REACHABLE, unreachable


def test_no_claim_the_recorded_release_passed_becomes_a_failure():
    """The held-correct arm. NOTE this test is permissiveness-FRIENDLY — a
    more permissive verifier passes it more easily — so it is never read
    alone; the fabricated arm below is its adversary."""
    _sv, res, _p = _scored()
    noise = [(r["row_id"], r["reason"]) for r in res["rows"]
             if r["arm"] == "held-correct" and r["flagged"]]
    assert len(noise) <= 1, noise


def test_fabricated_claims_are_seeded_and_are_still_caught():
    """The arm the adjudicated gold structurally cannot contain, and the one
    that is permissiveness-HOSTILE: every row is a claim the release passed,
    mutated until it is FALSE about the evidence its own turn held.

    The bound moved 37 -> 41 when `_ABSENCE_SHAPE` was widened. That is not a
    regression in the verifier: the old pattern matched NOTHING in this corpus,
    so the `rider-and` / `rider-semicolon` shapes the module docstring
    advertised were never once seeded. Widening it seeded four of them and the
    verifier misses all four — a pre-existing hole the arm could not see, now
    counted. `test_the_rider_shapes_are_actually_seeded` pins that they exist.
    """
    _sv, res, _p = _scored()
    fab = res["arms"]["fabricated"]
    assert res["arm_sizes"]["fabricated"] >= 40
    assert fab["counts"]["fp"] == 0
    assert fab["counts"]["fn"] <= 41, fab["false_negatives"]


def test_the_rider_shapes_are_actually_seeded():
    """COUNTED, NOT ASSERTED. `fabricate`'s docstring has always described a
    rider bolted onto a true negative; for the whole life of the scorer it
    produced zero of them, and nobody noticed because nothing counted them."""
    _sv, res, _p = _scored()
    shapes = res["fabrication_kinds"]
    assert shapes.get("rider-and", 0) >= 1 and shapes.get("rider-semicolon", 0) >= 1
    assert res["absence_shaped"]["fabricated"] >= 2


#: Repaired rows this tree still flags — an EXACT set, for the same reason
#: KNOWN_UNCLEARED is one: it fails when a row starts being cleared as well as
#: when one stops, so neither a regression nor an unlogged widening passes.
#:
#: * the two `abs-antarctica` rows are the ABSENCE region. Rulings 3 and 7 were
#:   deliberately deleted (see `classify_deterministic`), so a closed-world
#:   negative has no supporting branch left. These two rows are the measured
#:   price of that decision and the only place in the whole instrument where it
#:   is visible.
#: * `agg-corpus-boards` and `fr-disc-thai-rice` carry a name `verify.entities`
#:   extracts and the seed builder's `_marked_names` does not ('RSF corpus',
#:   'CSA, Thailande'). The mutation may not use `verify.entities` — it is
#:   under test — so the repair moved a citation without seeing that term.
#:   Instrument limitation, named rather than hidden.
#: * `cmp-fp172-fp173-rank` is repaired to a figure its cited scope really
#:   prints, and the REGISTRY records a conflicting figure for that document
#:   and field. The document contradicts itself; flagging it is correct.
#:
#: `rep-citation-agg-corpus-boards-90d9520d7b` LEFT THE ARM with the lead-in
#: wave: it was built from the conversational offer 'tell me which one and I
#: can answer precisely', which `_GLUE_RE` now drops (adjudicated
#: `claim-2bca31faa865e0d00c91c737`, label `not_a_claim`). The arm is drawn
#: from the claims the release FLAGGED, so a unit that stops being a claim
#: stops being repairable — a population change, not a verdict change, and
#: the four rows that remain are the four it always had.
REPAIRED_STILL_FLAGGED = {
    "rep-absence-abs-antarctica-096905d75a",
    "rep-absence-abs-antarctica-b67fa5942a",
    "rep-citation-fr-disc-thai-rice-3aec1712f7",
    "rep-figure-cmp-fp172-fp173-rank-fa227d25b3",
}


def test_repaired_claims_are_seeded_and_are_cleared_or_named():
    """The fourth arm: claims the release FLAGGED, made TRUE about their own
    turn's evidence. It is permissiveness-FRIENDLY like held-correct, but
    unlike held-correct it covers the population where over-strictness lives —
    the rows that were flagged in the first place."""
    _sv, res, _p = _scored()
    # THE BOUND MOVED 20 -> 12 WITH THE LEAD-IN WAVE, and it is a population
    # change with a name. The arm repairs claims the release FLAGGED; 14 of
    # those 71 units no longer produce a claim at all (ruling 6 heading forms
    # plus one glue offer), so the 14 `rep-citation-*` rows built by pointing
    # a bracket at an uncited lead-in have no claim left to repair. Every row
    # that leaves is listed by `resolved-by-extraction`, and the rows that
    # stay are scored exactly as before — `REPAIRED_STILL_FLAGGED` is an
    # exact set and it lost only the member whose carrier unit was dropped.
    assert res["arm_sizes"]["repaired"] >= 12, res["arm_sizes"]
    still = {r["row_id"] for r in res["rows"]
             if r["arm"] == "repaired" and r["flagged"]}
    assert still == REPAIRED_STILL_FLAGGED


def test_the_repaired_arm_reaches_the_absence_region():
    """The region the reviewer showed was in NO arm. Counted from the rows, not
    claimed in a docstring: the previous seed set contained zero absence-shaped
    rows while asserting it contained some."""
    sv, res, _p = _scored()
    assert res["absence_shaped"]["repaired"] >= 2, res["absence_shaped"]
    flagged, _trunc = sv.release_failures(
        sv.read_jsonl(GOLD / "release_release-1.jsonl"))
    census = sv.absence_census(
        sv.read_jsonl(GOLD / "release_release-1-evidence.jsonl"), flagged)
    # 71 -> 57: `absence_census` counts the release's flagged claims that
    # extraction STILL mints, and 14 of the 71 units are now dropped
    # (`resolved-by-extraction`). The number is pinned exactly, not loosened
    # to an inequality, so a further drift in extraction still fails here.
    assert census["flagged"] == 57, census
    assert len(res["resolved_by_extraction"]) == 71 - 57, res["resolved_by_extraction"]
    assert census["candidates"] >= 8, census


def test_every_repaired_row_carries_a_structural_validity_statement():
    """`should_flag = False` is a claim about the world, so the row has to say
    what makes it true — and the statement is checked from the evidence text,
    never from a verdict."""
    _sv, res, _p = _scored()
    for r in res["rows"]:
        if r["arm"] == "repaired":
            assert r["should_flag"] is False
            assert r["validity"] and len(r["validity"]) > 20, r


def test_the_seed_set_does_not_depend_on_the_verifier_it_scores():
    """A seed set drawn from 'whatever the current code supports' hands a
    permissive verifier a smaller and easier test. Source-level half: neither
    mutation may name a verdict at all."""
    sv, res, _p = _scored()
    src = SCORER.read_text()
    for fn in ("def fabricate(", "def repair(", "def contradict("):
        body = src[src.index(fn):src.index("# ---", src.index(fn))]
        code = "\n".join(ln for ln in body.splitlines()
                         if ln.strip() and not ln.lstrip().startswith("#"))
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        assert "classify" not in code and "verdict" not in code \
            and ".status" not in code, fn
    for row in res["rows"]:
        if row["arm"] == "fabricated":
            assert row["should_flag"] is True


def test_the_seed_set_is_bit_identical_under_a_forced_verdict(monkeypatch):
    """The behavioural half, and the stronger one: run the whole seed
    construction with `classify_deterministic` replaced by a function that
    returns all-SUPPORTED, and again by one that returns all-CONTRADICTED, and
    the seed set must be bit-identical to the real tree's — AND the replacement
    must never be called at all.

    A source scan can be defeated by an indirection; this cannot. It is what
    makes `--baseline` meaningful across two trees, and it is the property the
    printed `seed set sha256` claims."""
    for name in ("release_release-1-adjudicated.jsonl",
                 "release_release-1-evidence.jsonl", "release_release-1.jsonl"):
        if not (GOLD / name).exists():
            pytest.skip(f"{name} not present")
    sv = _scorer()
    args = (sv.read_jsonl(GOLD / "release_release-1-adjudicated.jsonl"),
            sv.read_jsonl(GOLD / "release_release-1-evidence.jsonl"),
            sv.read_jsonl(GOLD / "release_release-1.jsonl"))
    base = sv.seed_digest(sv.build_rows(*args)[0])
    assert len(base) == 64
    for forced in (V.SUPPORTED, V.CONTRADICTED):
        calls = []

        def fake(claims, evidence, *a, _s=forced, _c=calls, **k):
            _c.append(1)
            return [V.Verdict(c, _s, "forced", [], flags=[]) for c in claims]

        monkeypatch.setattr(V, "classify_deterministic", fake)
        assert sv.seed_digest(sv.build_rows(*args)[0]) == base, forced
        assert calls == [], f"seed construction consulted the verifier ({forced})"
        monkeypatch.undo()


def test_the_judge_bound_counters_are_pinned():
    """`flagged` collapses CONTRADICTED and UNSUPPORTED; the production path
    does not. An UNSUPPORTED-and-plausible verdict is exactly the residue
    `adjudicate` is handed, so with `VERIFY_LLM=1` a clearing judge ships it,
    while a CONTRADICTED one is never shown to the judge at all.

    This is not theoretical. Forcing the registry deference to True moved two
    fabricated rows CONTRADICTED -> UNSUPPORTED and NOTHING else in the whole
    instrument: TP/FP/FN/TN flat on all four arms, zero flag flips, zero answer
    statuses changed under any judge bound. `fn_clearing_judge` 104 -> 106 is
    the only number that sees that relaxation, so it is pinned like a gate."""
    _sv, res, _p = _scored()
    fab = res["arms"]["fabricated"]["counts"]
    assert fab["fn_clearing_judge"] <= 104, \
        res["arms"]["fabricated"]["escapes_clearing_judge"]
    assert fab["fp_contradicted"] == 0
    assert res["arms"]["held-correct"]["counts"]["fn_clearing_judge"] == 0
    assert res["arms"]["held-correct"]["counts"]["fp_contradicted"] == 0
    # a repaired row the verifier CONTRADICTS cannot be rescued by any judge;
    # exactly one exists and it is named in REPAIRED_STILL_FLAGGED
    assert res["arms"]["repaired"]["counts"]["fp_contradicted"] <= 1, \
        res["arms"]["repaired"]["contradicted_but_should_not_be"]


#: THE GOLD JOIN AFTER RULING 6, as an EXACT set for the same reason
#: `KNOWN_UNCLEARED` is one: it fails when a row stops joining as well as when
#: one starts. Fourteen of the 71 adjudicated units no longer yield a claim.
#: Twelve are the taxonomy's own `not_a_claim` rows. TWO ARE NOT, and they are
#: the whole reason this set is written out row by row rather than counted:
#:
#:   claim-66bc581a  'FP153 ("Mongolian Green Finance Corporation") has
#:                   **inconsistent figures** in the retrieved document:'
#:   claim-e7bb6639  'Among the retrieved excerpts, the Global Subnational
#:                   Climate Fund (...) is described as being formed by **two
#:                   funding proposals** submitted by two accredited entities:'
#:
#: Both are labelled `missing_citation`, `verifier_correct: true`, and both
#: reviewer notes give the SAME reason: 'it ends in a colon but does assert
#: something checkable, so the "asserts nothing checkable" test fails'. That
#: is ruling 1's test, and ruling 6 — added after Wave 1 labelling and
#: owner-approved — reverses it in exactly these words: a colon lead-in is
#: `not_a_claim` EVEN IF it carries a checkable proposition. `claim-8b23b13e`
#: ('FP267 ("Eco-DRR") shows **conflicting figures** ... :') is the identical
#: shape, was one of the three sampled inter-rater disagreements, and WAS
#: re-resolved to `not_a_claim`. These two were not sampled, so the frozen
#: gold still carries the pre-ruling-6 label. The file may not be edited here;
#: the discrepancy is recorded instead, and their content is not lost — the
#: scorer's `content_carried` shows both units' names still verified through
#: the claim below them.
RESOLVED_BY_EXTRACTION = {
    "claim-2bca31faa865e0d00c91c737",     # glue: 'I can answer precisely'
    "claim-2c56455366f7137b7b551677",     # colon lead-in
    "claim-39c728e38ac9ebd7ba9693b2",
    "claim-66bc581a6917e396a5ece535",     # label missing_citation — ruling 6
    "claim-8b23b13e6e1697cdf60d6223",
    "claim-8d4ecc0eaebb42248dc60dd3",     # bold label
    "claim-9e6f9809aa4a7bff02717747",
    "claim-a5f28680f4a78ad61acc7a84",     # bold label
    "claim-a72ac18e5f06bed1b1b6dbc2",
    "claim-c4a42b13ce733ad3a8d996ba",
    "claim-cea13d50e7d89d041051d4f9",
    "claim-cf409d05679be27dc480f80f",     # bold label
    "claim-e6d195710f1b15b70b3e96f7",
    "claim-e7bb663933e0226ecc371617",     # label missing_citation — ruling 6
}


def test_every_scored_row_joins_exactly_one_extracted_claim():
    _sv, res, problems = _scored()
    assert res["unmatched"] == [] and problems == []


def test_the_rows_that_stop_joining_are_exactly_the_resolved_ones():
    """No gold row may leave the join without being named.

    `unmatched` is the regression channel and it is empty; this is the other
    half — the rows that legitimately left, as an exact set. A row that stops
    joining for any reason other than 'its unit produced no claim' lands in
    `unmatched` above, and one that leaves for that reason has to be listed
    here or this fails."""
    _sv, res, _p = _scored()
    got = {r["row_id"] for r in res["resolved_by_extraction"]}
    assert got == RESOLVED_BY_EXTRACTION, (got - RESOLVED_BY_EXTRACTION,
                                           RESOLVED_BY_EXTRACTION - got)


def test_the_two_rows_the_gold_still_calls_genuine_defects_are_named():
    """The −2 side of ruling 6's ledger, asserted rather than narrated.

    Twelve rows were adjudicated 'the verifier was right'. Two of them are
    colon lead-ins whose label predates ruling 6, so implementing that ruling
    removes them from the join. The trade is only acceptable because it is
    visible and because their content is still checked — both are asserted
    here."""
    _sv, res, _p = _scored()
    off = {r["row_id"]: r for r in res["resolved_by_extraction"]
           if r["label"] != "not_a_claim"}
    assert set(off) == {"claim-66bc581a6917e396a5ece535",
                        "claim-e7bb663933e0226ecc371617"}, sorted(off)
    for r in off.values():
        assert r["should_flag"] is True, r
        assert r["shape"] == "colon-lead-in", r
        # the unit's names did not stop being verified: they ride on the claim
        # below and are checked against the scope THAT claim cites
        assert r["content_carried"], r
        assert r["content_lost"] == [], r


def test_no_gold_true_content_is_lost_by_the_resolution():
    """The content ledger, over every resolved row.

    A row may resolve only when nothing the verifier was RIGHT about stops
    being checked. Exactly one row loses terms — `claim-2bca31fa`, the
    conversational offer, `verifier_correct: false` — so what it stops
    checking is content the adjudication already ruled was never a claim."""
    _sv, res, _p = _scored()
    lost = [r for r in res["resolved_by_extraction"] if r["content_lost"]]
    assert [r["row_id"] for r in lost] == ["claim-2bca31faa865e0d00c91c737"], lost
    for r in lost:
        assert r["should_flag"] is False, r


# ---------------------------------------------------------------------------
# the contradiction arm, and the instrument hole it closes
#
# FINDING F1.  The merge review of 4a04d32 replaced `verify._field_conflict`
# with `return None` — deleting every evidence-text contradiction the verifier
# can emit — re-ran `scripts/score_verifier.py`, and every printed digit was
# identical: gold 30.8%, overall 73.1%, recall 68.0%, four matrices flat to the
# count, caution census flat.  The unit suite caught it; the instrument that
# will certify Wave 4's repair A/B did not.  Two causes, and a test for each:
#
#   (a) `Replay.cautions()` counted flags on SUPPORTED claims only, and two of
#       the verifier's flags are emitted ONLY on a CONTRADICTED verdict.
#   (b) no arm held a claim that MUST come back CONTRADICTED, and the matrices
#       count `flagged`, which a lost contradiction usually does not change.
# ---------------------------------------------------------------------------

#: The contradiction-arm rows this tree does NOT get right — an EXACT set,
#: for the same reason KNOWN_UNCLEARED and REPAIRED_STILL_FLAGGED are exact:
#: it fails when a row starts passing as well as when one stops.  It is now
#: EMPTY, and the row it used to hold is why the set is kept rather than
#: deleted with its last member.
#:
#: `con-elsewhere-id-fp152-financing-b580f69e0a`: `123_gcf-b27-02-add12` p.5
#: prints, verbatim, '## A8. Total GCF funding requested\n150 M USD', and the
#: row states 150 M USD for total_financing while the same document's registry
#: line prints '720 M USD' for that field.  Two things had to be true for it to
#: come back SUPPORTED, and both were: `verify.amounts` does not yield
#: '150 M USD' from that page (the page also carries '- Equity: 150 MUSD', and
#: what it yields is `('150', 150.0, 'USD', None)`), so the claim missed its
#: strict scope; and the same-document fallback branch that then supported it —
#: 'value found in the cited document, but not on the cited page' — ran NO
#: conflict check at all before returning SUPPORTED.  The instrument found that
#: second half independently of the attack suite, and named it here as a gap in
#: the code under test rather than as the arm being wrong.  The gap is closed:
#: every SUPPORTED exit now goes through `_conflict_before_support`, the row
#: comes back CONTRADICTED on the cross-page scan, and the arm is 34/34.
CONTRADICTION_ARM_MISSES: set = set()

#: Every shape the arm must actually reach.  COUNTED, NOT ASSERTED — the same
#: discipline `test_the_rider_shapes_are_actually_seeded` exists for, after a
#: docstring advertised absence shapes the seed set never once produced.
CONTRADICTION_SHAPES = {"same-key", "transposed", "wrong-page", "elsewhere"}


def test_the_contradiction_arm_is_seeded_and_reaches_every_shape():
    """The arm the other four structurally cannot contain: a claim whose CITED
    key prints a different value under the claim's OWN field label."""
    _sv, res, _p = _scored()
    assert res["arm_sizes"].get("contradicted", 0) >= 30, res["arm_sizes"]
    shapes = res["contradicted_by_shape"]
    assert set(shapes) == CONTRADICTION_SHAPES, sorted(shapes)
    for shape in CONTRADICTION_SHAPES:
        assert shapes[shape]["rows"] >= 1, (shape, shapes)
    # the two shapes whose loss PROMOTES rather than degrades: the claim's own
    # figure verifies against the page it cites, so the field conflict is the
    # only thing standing between it and SUPPORTED
    assert shapes["transposed"]["rows"] + shapes["elsewhere"]["rows"] >= 15, shapes


def test_every_contradiction_row_must_contradict_and_says_why():
    _sv, res, _p = _scored()
    rows = [r for r in res["rows"] if r["arm"] == "contradicted"]
    for r in rows:
        assert r["should_flag"] is True and r["must_contradict"] is True, r
        assert r["validity"] and len(r["validity"]) > 30, r
        assert r["field"] in ("gcf_financing", "total_financing", "co_financing"), r


def test_the_contradiction_arm_is_structurally_valid():
    """RE-DERIVED FROM THE EVIDENCE TEXT, not trusted from the row.

    For every row: the rival really is printed under that field's own label on
    a key of the cited document, the claim does not state the rival itself, and
    the claim's field is the one the row says it is. Nothing here reads a
    verdict — it is the same certification the row was built under, run again
    from the recorded evidence.
    """
    sv, res, _p = _scored()
    rows, _problems, _resolved = sv.build_rows(
        sv.read_jsonl(GOLD / "release_release-1-adjudicated.jsonl"),
        sv.read_jsonl(GOLD / "release_release-1-evidence.jsonl"),
        sv.read_jsonl(GOLD / "release_release-1.jsonl"))
    con = [r for r in rows if r["arm"] == "contradicted"]
    assert len(con) >= 30
    for row in con:
        field, rival = row["field"], row["rival"]
        cited = (row["cited"][0], row["cited"][1])
        assert sv._seed_field_of(row["claim_text"]) == field, row["row_id"]
        # the rival is really PRINTED, under that field's own label, on a key
        # of the cited document — the same key for every shape but `elsewhere`,
        # where it is another key of the same document by construction
        printed = {k: got[field][0] for k, t in row["evidence"].items()
                   if field in (got := sv._seed_printed_fields(t))}
        assert printed, f"{row['row_id']}: no key prints {field}"
        assert any(k[0] == cited[0] and sv._digits(v) == sv._digits(rival)
                   for k, v in printed.items()), (row["row_id"], rival, printed)
        if row["why"] != "elsewhere":
            assert sv._digits(printed.get(cited, "")) == sv._digits(rival), row
        else:
            assert field not in sv._seed_printed_fields(
                row["evidence"].get(cited, "")), row["row_id"]
        # and the claim itself does not state the rival its cited scope gives:
        # a claim that reports both sides is the instructed behaviour, not a
        # contradiction, and the verifier is right not to flag it
        mine = {sv._digits(s) for s in
                sv._seed_runs(V._strip_citations(row["claim_text"]))}
        assert sv._digits(rival) not in mine, (row["row_id"], rival)
        # nor does any sibling claim of the same answer that names that document
        others = [c for c in V.extract_claims(row["answer"])
                  if c.text != row["claim_text"]]
        assert sv._digits(rival) not in sv._sibling_digits(
            V.extract_claims(row["answer"]),
            next(c for c in V.extract_claims(row["answer"])
                 if c.text == row["claim_text"]), cited[0]), row["row_id"]
        assert others is not None


def test_the_contradiction_counters_are_pinned():
    """The digits the reviewer's experiment has to move, pinned like gates."""
    _sv, res, _p = _scored()
    con = res["arms"]["contradicted"]["counts"]
    assert con["must_contradict"] == res["arm_sizes"]["contradicted"]
    assert con["tp_contradicted"] >= 33, res["arms"]["contradicted"]
    assert con["fp"] == 0 and con["fp_contradicted"] == 0
    lost = {r["row_id"] for r in res["rows"]
            if r.get("must_contradict") and r["status"] != V.CONTRADICTED}
    assert lost == CONTRADICTION_ARM_MISSES, lost


def _rescore_with(monkeypatch, name, replacement):
    """Rebuild AND rescore the whole seed set with one verifier symbol
    replaced.  The seed set must not move — that is what makes the two scores
    comparable — so the digest is asserted first."""
    sv = _scorer()
    args = (sv.read_jsonl(GOLD / "release_release-1-adjudicated.jsonl"),
            sv.read_jsonl(GOLD / "release_release-1-evidence.jsonl"),
            sv.read_jsonl(GOLD / "release_release-1.jsonl"))
    before_rows = sv.build_rows(*args)[0]
    before = sv.score(before_rows)
    digest = sv.seed_digest(before_rows)
    monkeypatch.setattr(V, name, replacement)
    after_rows = sv.build_rows(*args)[0]
    assert sv.seed_digest(after_rows) == digest, name
    return sv, before, sv.score(after_rows)


def test_deleting_the_field_conflict_path_is_now_DETECTED(monkeypatch):
    """FINDING F1, RUN AS A TEST.  This is the reviewer's exact experiment —
    `_field_conflict` replaced by `return None` — and the assertion is that the
    SCORER now sees it.  Before the contradiction arm existed, every one of
    these numbers was identical across the two trees.

    Measured on this tree: the contradicted arm goes TP 33 -> 12, FN 1 -> 22,
    recall 97.1% -> 35.3%, TP|contra 33 -> 0, and 22 rows are PROMOTED to a
    passing verdict — a false negative in the region a repair pass fires on.
    """
    sv, before, after = _rescore_with(
        monkeypatch, "_field_conflict", lambda *a, **k: None)
    b = before["arms"]["contradicted"]["counts"]
    a = after["arms"]["contradicted"]["counts"]
    assert b["tp_contradicted"] >= 33 and a["tp_contradicted"] == 0
    assert a["fn"] >= 20 and a["fn"] > b["fn"]
    assert after["arms"]["contradicted"]["recall"] < 0.5 <= \
        before["arms"]["contradicted"]["recall"]
    assert a["contradiction_promoted"] >= 20
    # and the overall table moves too, which is what a reviewer reads first
    assert after["overall"]["recall"] < before["overall"]["recall"] - 0.10


def test_a_contradiction_that_only_DEGRADES_is_also_detected(monkeypatch):
    """The subtler half.  Collapsing CONTRADICTED onto UNSUPPORTED leaves every
    cell of every matrix exactly where it was — `flagged` is true either way —
    so nothing that counts `flagged` can see it.  It is not cosmetic: with
    `VERIFY_LLM=1` an UNSUPPORTED-and-plausible claim is the residue the judge
    is handed and may clear, while a CONTRADICTED one is never shown to it.
    The status counters and the flag census are what see this."""
    real = V.classify_deterministic

    def no_contradictions(claims, evidence, *a, **k):
        """The code stops EMITTING contradictions; the constants are untouched.
        Patching `verify.CONTRADICTED` itself would prove nothing — the scorer
        reads the same symbol, so the two sides would move together."""
        return [V.Verdict(v.claim, V.UNSUPPORTED, v.reason, v.scope,
                          source=v.source, flags=list(v.flags),
                          plausible=v.plausible)
                if v.status == V.CONTRADICTED else v
                for v in real(claims, evidence, *a, **k)]

    sv, before, after = _rescore_with(
        monkeypatch, "classify_deterministic", no_contradictions)
    for arm in sv.ARMS:
        b = before["arms"][arm]["counts"]
        a = after["arms"][arm]["counts"]
        assert (b["tp"], b["fp"], b["fn"], b["tn"]) == \
            (a["tp"], a["fp"], a["fn"], a["tn"]), arm      # nothing moves here
    assert before["overall"]["counts"]["tp_contradicted"] >= 33
    assert after["arms"]["contradicted"]["counts"]["tp_contradicted"] == 0
    moved = {k for k in set(before["flag_census"]) | set(after["flag_census"])
             if before["flag_census"].get(k) != after["flag_census"].get(k)}
    assert any(k.startswith("contradicted:") for k in moved), moved


def test_the_caution_census_no_longer_counts_supported_claims_only():
    """CAUSE (a) OF FINDING F1.  `conflict-elsewhere-in-document` and
    `known-document-conflict` are appended only on branches that return
    CONTRADICTED, so a census filtered to SUPPORTED could never count them,
    never see them appear and never see them go.  It is now taken over every
    verdict and tagged with the status it sits on."""
    sv, res, _p = _scored()
    census = res["flag_census"]
    assert census.get("contradicted:conflict-elsewhere-in-document", 0) >= 10, census
    assert census.get("contradicted:known-document-conflict", 0) >= 1, census
    # the same status tagging on the live-path census
    ev = sv.read_jsonl(GOLD / "release_release-1-evidence.jsonl")
    rep = sv.Replay(ev[0].get("answer") or "", sv.evidence_of(ev[0]))
    assert all(":" in c for c in rep.cautions())
    assert all(":" not in c for c in rep.user_cautions())


def test_the_recorded_answers_alone_cannot_witness_the_contradiction_path():
    """A MEASURED LIMIT, recorded rather than assumed away.  The 66 recorded
    answers produce no CONTRADICTED verdict at all, so the live-path block of
    the report — statuses and cautions over those answers — is structurally
    incapable of witnessing this path however it is filtered.  That is why the
    flag census is also taken over the SEEDED rows."""
    sv, _res, _p = _scored()
    ev = sv.read_jsonl(GOLD / "release_release-1-evidence.jsonl")
    states = sv.answer_state(ev)
    assert len(states) == 66
    assert sum(s["statuses"].get(V.CONTRADICTED, 0) for s in states.values()) == 0
    assert not any(c.startswith("contradicted:")
                   for s in states.values() for c in s["cautions"])


@pytest.mark.parametrize("what,patch", [
    # the registry deference, forced to defer to every rival print
    ("registry-defers-always",
     ("registry_ruled_compatible", lambda *a, **k: True)),
    # conflict detection stops knowing what field a claim is about
    ("conflict-ignores-field", ("_FIELD_RES", [])),
    # the per-key rival scan is gutted while _field_conflict itself is intact
    ("key-conflict-no-rivals",
     ("_key_conflict", lambda *a, **k: (None, None))),
])
def test_three_more_contradiction_path_ablations_are_detected(monkeypatch,
                                                              what, patch):
    """Three relaxations of my own devising, each on a different symbol of the
    path.  All three were invisible to the four-arm scorer; all three now
    collapse the contradicted arm.  `field-label-anywhere` (the OVER-strict
    direction) is covered by its own test below."""
    sv, before, after = _rescore_with(monkeypatch, patch[0], patch[1])
    b = before["arms"]["contradicted"]["counts"]
    a = after["arms"]["contradicted"]["counts"]
    assert b["tp_contradicted"] >= 33, what
    assert a["tp_contradicted"] == 0, what
    assert a["contradiction_lost"] == a["must_contradict"], what
    assert a["contradiction_promoted"] >= 20, what


def test_the_cross_page_branch_has_its_own_detector(monkeypatch):
    """`cross_page_conflicts` guards ONE of the three `_key_conflict` calls, so
    a change that only turns it off must move less than deleting the path —
    and it must still move.  The `elsewhere` shape is the only rows that reach
    it, and its `conflict-elsewhere-in-document` census entry empties."""
    sv = _scorer()
    args = (sv.read_jsonl(GOLD / "release_release-1-adjudicated.jsonl"),
            sv.read_jsonl(GOLD / "release_release-1-evidence.jsonl"),
            sv.read_jsonl(GOLD / "release_release-1.jsonl"))
    rows = sv.build_rows(*args)[0]
    before = sv.score(rows)
    real = V.classify_deterministic
    monkeypatch.setattr(V, "classify_deterministic",
                        lambda c, e, cross_page_conflicts=True, **k:
                        real(c, e, cross_page_conflicts=False, **k))
    after = sv.score(sv.build_rows(*args)[0])
    b = before["arms"]["contradicted"]["counts"]
    a = after["arms"]["contradicted"]["counts"]
    assert 0 < a["contradiction_lost"] < a["must_contradict"], (b, a)
    assert a["tp_contradicted"] < b["tp_contradicted"]
    assert before["flag_census"].get(
        "contradicted:conflict-elsewhere-in-document", 0) >= 10
    assert after["flag_census"].get(
        "contradicted:conflict-elsewhere-in-document", 0) == 0


def test_over_strictness_in_the_conflict_path_is_detected_too(monkeypatch):
    """The arm must not only catch DELETIONS.  Opening `_FIELD_PREFIX_OK` so a
    field label buried in prose counts as the field being stated turns prose
    into contradictions: measured 32 -> 41 false positives overall, and the
    `FP|contradicted` counter — a claim no judge can rescue — 1 -> 10."""
    import re as _re
    sv, before, after = _rescore_with(
        monkeypatch, "_FIELD_PREFIX_OK", _re.compile(r"^[\s\S]*$"))
    assert after["overall"]["counts"]["fp"] > before["overall"]["counts"]["fp"]
    assert after["overall"]["counts"]["fp_contradicted"] >= \
        before["overall"]["counts"]["fp_contradicted"] + 5
    assert after["arms"]["held-correct"]["counts"]["fp"] > 0


#: Contradiction-path changes the instrument STILL cannot see, measured and
#: named rather than left to be discovered by the next review.
#:
#: * `first-value-only` — `_field_conflict` reads the first TWO amounts after a
#:   label when deciding 'the field agrees somewhere'; narrowing that to one
#:   moves nothing anywhere, because no row in any arm has a field label whose
#:   SECOND printed amount is the one its claim states.
#: * `also-reported-unfiltered` — dropping the `registry_records` filter from
#:   the 'report both figures' licence.  To move a row, an answer would need a
#:   sibling claim stating the rival AND that rival to be absent from the
#:   registry for that document and field.  Measured over the whole recorded
#:   corpus: 48 (claim, key) candidate pairs, of which 47 have a registry-
#:   recorded rival and the one that does not has no sibling stating it.  The
#:   shape has ZERO instances to seed, so this is a corpus limit, not a
#:   generator one, and no amount of arm-building closes it.
INSTRUMENT_STILL_BLIND = ("first-value-only", "also-reported-unfiltered")


def test_the_arm_records_which_evidence_text_it_cannot_reach():
    """A MEASURED LIMIT, pinned so it cannot quietly persist.

    Every rival this arm contradicts a claim with is printed on a DOCUMENT-LEVEL
    evidence key (the registry line `build_evidence` attaches to `(doc, None)`),
    never on a numbered page.  That is a property of the recorded evidence, not
    of the generator: across all 66 turns only six numbered pages print a field
    label heading its own segment with a money value at all, and none of the
    six pairs with a claim that names that field (four are `co_financing`,
    which no recorded claim names, one is FP152 p.55 whose rival the registry
    ruled compatible, one is a case with no matching claim).

    Which key the rival sits on does not change which branch of the verifier
    runs — the conflict scan reads the evidence dict identically — but the arm
    may not claim coverage it does not have.  If a future evidence set makes a
    page-level rival seedable, this test fails and the claim gets updated."""
    _sv, res, _p = _scored()
    assert res["rival_key_kinds"] == {
        "document-level key": res["arm_sizes"]["contradicted"]}, \
        res["rival_key_kinds"]


def test_the_first_value_narrowing_is_still_invisible(monkeypatch):
    """Named, not buried. This asserts the KNOWN blindness so that a future
    change which happens to make it visible fails here and gets recorded,
    instead of the blind spot quietly persisting in a comment."""
    sv = _scorer()
    args = (sv.read_jsonl(GOLD / "release_release-1-adjudicated.jsonl"),
            sv.read_jsonl(GOLD / "release_release-1-evidence.jsonl"),
            sv.read_jsonl(GOLD / "release_release-1.jsonl"))
    before = sv.score(sv.build_rows(*args)[0])
    real = V._value_after
    monkeypatch.setattr(V, "_value_after",
                        lambda line, at, window=80: real(line, at, window)[:1])
    after = sv.score(sv.build_rows(*args)[0])
    same = all(before["arms"][arm]["counts"] == after["arms"][arm]["counts"]
               for arm in sv.ARMS)
    assert same, "the first-value narrowing became visible — update " \
                 "INSTRUMENT_STILL_BLIND and this test"


# ===========================================================================
# ENTITY-MATCHER ARTIFACTS — the five classes the judge audit isolated
#
# Source: `data/eval/judge_audit_parity-baseline.jsonl` (28 promotions) and
# `data/eval/judge_audit_adjudication_parity-baseline.jsonl` (11 human
# adjudications). Every relaxation below names the row id it exists for, and
# every one carries an adversarial twin that varies THE DIMENSION THE
# RELAXATION TURNS ON — not a neighbouring one. Six previous reviews were
# defeated by a test that varied something else, so the dimension is named in
# each docstring and the mutation that must break it is stated with it.
#
#   class 1  acronym <- printed expansion      cid-fp0086-padded
#   class 2  possessive / orthography          id-fp203-objective
#   class 3  composite gluing (extraction)     disc-subnational-pair
#   class 4  denied term (extraction)          abs-antarctica
#   class 5  cross-lingual exonym              fr-disc-thai-rice (2 rows)
# ===========================================================================

ESMP_PAGE = ("Updated supporting documents for restructuring paper:\n"
             "- Environmental and Social Impact Assessment (ESIA) or "
             "Environmental and Social Management Plan (if applicable)\n"
             "- Appraisal Report or Due Diligence")


# ------------------------------------------------------- class 1: acronyms --
def test_an_acronym_is_satisfied_by_the_expansion_the_page_prints(no_registry):
    """CLASS 1, row `cid-fp0086-padded`. The answer compresses a phrase the
    cited page prints IN FULL. The initialism is a function OF the evidence:
    to satisfy it the page has to print all four words, in order, adjacent."""
    ev = {(DOC, 39): ESMP_PAGE}
    answer = f"The list includes **ESIA/ESMP (if applicable)** [{DOC}, p. 39]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED, (v.status, v.reason)


def test_an_acronym_whose_letters_are_reordered_is_not_an_expansion(no_registry):
    """DIMENSION: the letter sequence. Same page, same words, one transposition
    — 'ESPM'. Mutating the rule to compare letter SETS passes this."""
    ev = {(DOC, 39): ESMP_PAGE}
    answer = f"The list includes **ESIA/ESPM (if applicable)** [{DOC}, p. 39]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "ESPM" in v.reason, (v.status, v.reason)


def test_an_expansion_may_not_have_extra_words_between_its_initials(no_registry):
    """DIMENSION: adjacency. The page prints E...S...M...P in order, but with
    two content words wedged in, so it is not the expansion of ESMP — it is a
    different phrase that happens to contain those initials. Mutating the gap
    from 'connectives only' to 'any words' passes this."""
    ev = {(DOC, 39): "- Environmental Impact Assessment and Social "
                     "Management Plan (if applicable)"}
    answer = f"The list includes **ESMP (if applicable)** [{DOC}, p. 39]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "ESMP" in v.reason, (v.status, v.reason)


def test_a_two_letter_initialism_is_not_resolved_by_a_capitalised_pair(no_registry):
    """DIMENSION: how many letters must agree. Two-letter forms ('AE', 'EE')
    are the corpus's commonest abbreviations AND collide with any capitalised
    pair; 'Andean Ecosystems' does not say 'accredited entity'. Mutating the
    minimum length from 3 to 2 passes this."""
    ev = {(DOC, 5): "The Andean Ecosystems programme, page 5."}
    answer = f"The **AE** manages the programme [{DOC}, p. 5]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "AE" in v.reason, (v.status, v.reason)


@pytest.mark.parametrize("expansion", [
    "International Fund for Agricultural Development",   # the true gloss
    "International Federation of Arctic Drillers",       # score_verifier's
])
def test_the_expansion_direction_stays_shut(no_registry, expansion):
    """DIMENSION: the direction. Row `id-fp220-entity` is this shape and is
    DELIBERATELY NOT FIXED. A page printing 'IFAD' does not say what IFAD
    stands for, and the matcher cannot tell the true gloss from the fabricated
    one — only that neither is on the page. This is the direction
    `score_verifier.FAKE_EXPANSIONS` mutates; making the rule two-way to close
    `id-fp220-entity` passes the first row here and the second with it."""
    ev = {(DOC, None): "Registry — FP220: accredited entity: IFAD"}
    answer = f"FP220 is implemented by **IFAD ({expansion})** [{DOC}, cover pages]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed, (v.status, v.reason)


# ---------------------------------------------------- class 2: possessives --
def test_a_possessive_is_the_same_name(no_registry):
    """CLASS 2, row `id-fp203-objective`. The extractor lifted `Colombia’s`
    (U+2019) out of 'to support Colombia’s climate goals'; the cited cover page
    prints `Colombia`."""
    ev = {(DOC, None): "Registry — FP203: Heritage Colombia (HECO); "
                       "countries: Colombia"}
    answer = (f"It is a proposal focused on **sustainably managed landscapes** "
              f"to support Colombia’s climate goals [{DOC}, cover pages].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED, (v.status, v.reason)


@pytest.mark.parametrize("apostrophe", ["'", "’", "‘", "ʼ"])
def test_the_apostrophe_shape_does_not_decide_the_verdict(no_registry, apostrophe):
    """CLASS 2, orthography half. Which of the apostrophe codepoints the model
    happened to emit must not change a verdict."""
    ev = {(DOC, None): "Registry — FP203: countries: Colombia"}
    answer = (f"The proposal supports Colombia{apostrophe}s climate goals "
              f"[{DOC}, cover pages].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED, (apostrophe, v.status, v.reason)


def test_a_bare_trailing_s_is_not_a_possessive(no_registry):
    """DIMENSION: the apostrophe. 'Andes' ends in s and is not a possessive;
    de-inflecting it to 'Ande' would then match 'Andean' under the substring
    test. Mutating the rule to strip a bare trailing 's' passes this."""
    ev = {(DOC, 50): "Financed through the Andean Development Corporation."}
    answer = f"The programme covers the **Andes** [{DOC}, p. 50]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "Andes" in v.reason, (v.status, v.reason)


# ------------------------------------------------------ class 3: composites --
GLUE_PAGE = ("Pegasus Capital Advisors LP is the accredited entity. "
             "The Global Subnational Climate Fund (SoFC Global) is the "
             "vehicle.")


def test_a_glued_pair_of_attested_names_is_not_a_third_name(no_registry):
    """CLASS 3, row `disc-subnational-pair`. `_CAPRUN_RE` ran two bolded names
    together across 'for the' and invented `Pegasus Capital Advisors for the
    Global Subnational Climate Fund`, which no page prints."""
    ev = {(DOC, 76): GLUE_PAGE}
    answer = (f"A funding proposal submitted by **Pegasus Capital Advisors** "
              f"for the **Global Subnational Climate Fund (SoFC Global)** "
              f"[{DOC}, p. 76].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED, (v.status, v.reason)


def test_a_glued_pair_needs_both_halves_attested(no_registry):
    """DIMENSION: how many halves must stand on their own. Here the right half
    is not bolded, so it is not a candidate of its own and the glued run is the
    ONLY thing carrying those words — dropping it would delete the check
    outright. Mutating the rule to require only the left half passes this."""
    ev = {(DOC, 76): GLUE_PAGE}
    answer = (f"A funding proposal submitted by **Pegasus Capital Advisors** "
              f"for the Wakanda Development Bank [{DOC}, p. 76].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "Wakanda" in v.reason, (v.status, v.reason)


def test_dropping_a_glued_pair_deletes_no_check(no_registry):
    """DIMENSION: what survives the drop. This is the whole safety argument
    for class 3 stated as a test — the composite goes only because both halves
    remain candidates in their own right, so a fabricated half is still
    reported by name. Mutating `_drop_glued` to remove the halves too, or to
    match a half by containment in a longer name, passes this."""
    ev = {(DOC, 76): GLUE_PAGE}
    answer = (f"A funding proposal submitted by **Pegasus Capital Advisors** "
              f"for the **Wakanda Development Bank** [{DOC}, p. 76].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "Wakanda Development Bank" in v.reason, (v.status, v.reason)


# --------------------------------------------------- class 4: denied terms --
def test_a_bound_negative_prefix_is_not_an_assertion(no_registry):
    """CLASS 4, row `abs-antarctica`. 'other non-Antarctica infrastructure
    activities' DENIES Antarctica; the extractor was reading the denial as an
    assertion, and Antarctica's absence from the corpus is the case's point.

    Note where this operates: EXTRACTION. It removes a term the answer does
    not assert; it never excuses a term the evidence does not contain."""
    ev = {(DOC, 50): "Electric bus replacement in Argentina and Costa Rica."}
    answer = (f"The excerpts discuss electric buses in **Argentina** "
              f"[{DOC}, p. 50], and other non-Antarctica infrastructure "
              f"activities.")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED, (v.status, v.reason)


def test_a_bound_negative_prefix_does_not_excuse_its_neighbours(no_registry):
    """DIMENSION: scope. The two rejected rounds excused terms by PROXIMITY —
    a character window and a clause split — so a fabricated name standing
    beside a negator shipped verified. A bound prefix attaches to one word and
    to nothing else. Mutating the rule to a window, a clause or a sentence
    passes this."""
    ev = {(DOC, 50): "Electric bus replacement in Argentina and Costa Rica."}
    answer = (f"The excerpts discuss other non-Antarctica activities run by "
              f"the **Wakanda Development Bank** [{DOC}, p. 50].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "Wakanda" in v.reason, (v.status, v.reason)


def test_a_positive_printing_defeats_the_negative_prefix(no_registry):
    """DIMENSION: every printing, not the nearest one. The unit asserts
    Antarctica outright and denies it in a rider; the assertion stands and is
    checked. Mutating the rule to drop a term as soon as ONE printing is bound
    passes this."""
    ev = {(DOC, 50): "Electric bus replacement in Argentina and Costa Rica."}
    answer = (f"**Antarctica** is covered [{DOC}, p. 50], alongside other "
              f"non-Antarctica work.")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "Antarctica" in v.reason, (v.status, v.reason)


# ---------------------------------------------------- class 5: exonyms ------
THAI_PAGE = ("No objection letter issued by the national designated "
             "authority(ies) or focal point(s) of Thailand, for the project "
             "Thai Rice.")


def test_a_corpus_exonym_is_the_same_country(no_registry):
    """CLASS 5, rows `fr-disc-thai-rice` (both). A French answer over English
    pages: `Thaïlande` and `Autorité` are printed forms of `Thailand` and
    `authority`, and the matcher is a literal substring test."""
    ev = {(DOC, 202): THAI_PAGE}
    answer = (f"La lettre de non-objection de l’**Autorité** nationale "
              f"désignée de la **Thaïlande** [{DOC}, p. 202].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED, (v.status, v.reason)


def test_a_cross_lingual_variant_is_an_exact_table_hit(no_registry):
    """DIMENSION: exact key. The table is a closed list of printed forms, not
    a similarity. 'Taïwan' deaccents to something close to 'Thailand' and is a
    DIFFERENT place. Mutating the lookup to a prefix, a fuzzy or an
    edit-distance match passes this."""
    ev = {(DOC, 202): THAI_PAGE}
    answer = f"Le projet soutient la riziculture à **Taïwan** [{DOC}, p. 202]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "Ta" in v.reason, (v.status, v.reason)


def test_a_french_word_inside_a_longer_name_is_not_translated(no_registry):
    """DIMENSION: whole candidate vs its words. Written as a word-for-word
    substitution first, this cleared adjudicated defect
    `claim-6c4788ddf1da438d7049706e` (`missing_retrieval_evidence`, case
    fr-disc-thai-rice): rewriting the French words inside 'Ministry of
    Agriculture and Cooperatives (MOAC) – Thaïlande' produced a string the
    cited page prints, and an answer that names the WRONG proposal verified.
    Translating a whole name is a spelling change; translating a fragment of a
    longer assertion rebuilds the assertion."""
    ev = {(DOC, 6): "Ministry of Agriculture and Cooperatives (MOAC), "
                    "Thailand, is the executing entity."}
    answer = (f"Le projet est porté par le **Ministry of Agriculture and "
              f"Cooperatives (MOAC) – Thaïlande** [{DOC}, p. 6].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed, (v.status, v.reason)


def test_the_exonym_table_names_only_countries_this_corpus_records():
    """The table's admission rule, as a test. Every English value is either a
    country the corpus registry records or the one institutional cognate an
    audited row needs, so the table cannot introduce a referent the corpus
    never had."""
    from gcf_qna.rag import registry
    known = set()
    for row in (registry.load() or {}).values():
        got = row.get("countries") or row.get("country") or []
        if isinstance(got, str):
            got = got.split(",")
        for c in got:
            known.add(V.norm_text(str(c)))
    if not known:
        pytest.skip("registry not present in this checkout")
    unknown = sorted(v for v in set(V._FR_EN_NAMES.values()) - {"authority"}
                     if not any(v in k or k in v for k in known))
    assert unknown == [], unknown


def test_a_glued_half_is_not_covered_by_containment(no_registry):
    """DIMENSION: whole name vs substring. The right half here is 'Global
    Subnational Climate Fund OF WAKANDA' — a longer string than the attested
    name it contains. Accepting a half because some shorter extracted name
    sits inside it drops the composite AND the only candidate carrying
    'Wakanda', so the fabricated tail is never checked by anything. Mutating
    the membership test from equality to containment passes this."""
    ev = {(DOC, 76): GLUE_PAGE}
    answer = (f"The **Global Subnational Climate Fund (SoFC Global)** received "
              f"a proposal from **Pegasus Capital Advisors** for the Global "
              f"Subnational Climate Fund of Wakanda [{DOC}, p. 76].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.failed and "Wakanda" in v.reason, (v.status, v.reason)


@pytest.mark.parametrize("mark", ["'", "’", "ʼ", "ʹ", "‛", "′"])
def test_an_exotic_apostrophe_inside_a_name_still_matches(no_registry, mark):
    """CLASS 2, the orthography half, pinned where it actually bites: an
    apostrophe INSIDE a name, where no possessive rule can rescue it.
    'Côte d’Ivoire' is a corpus country and the answer model emits the mark
    from whichever font it was trained on. Deleting the fold in ``norm_text``
    passes none of these."""
    ev = {(DOC, None): "Registry — FP: countries: Côte d'Ivoire, Ghana"}
    answer = f"The programme covers **Côte d{mark}Ivoire** [{DOC}, cover pages]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED, (mark, v.status, v.reason)


# ===========================================================================
# Wave 4's first blocking prerequisite: the model call this module makes is a
# PINNED sample, and an endpoint that will not be pinned costs the parameter
# and never the turn.
#
# Wave 4 had three prerequisites and a carry-off decision; the other three
# were properties of the repair rewrite (its language, its size, whether a
# pre-repair judge ruling could certify it) and went with it at eac4c94. This
# one is not: the JUDGE is still an LLM call, it still decides verdicts a user
# sees, and an unpinned judge is still a verdict nobody can re-derive. Every
# test below was written against `verify.repair` as the vehicle — repair was
# the second call and the one with a rewrite to compare — and each is kept
# with the judge in its place. The DIMENSION each varies is unchanged: the
# individual parameter, which parameter is refused, how the refusal is
# expressed, whether the failure names a parameter at all.
# ===========================================================================

@pytest.fixture(autouse=True)
def _forget_endpoint_capabilities():
    """`_SAMPLING_UNSUPPORTED` is remembered for the PROCESS, so one test that
    exercises the degradation path could unpin every later one. Reset around
    each test in this module."""
    V._reset_sampling_support()
    yield
    V._reset_sampling_support()


class RejectingClient(FakeClient):
    """A client that refuses named parameters the way a real endpoint does.

    `shape` picks HOW the refusal is expressed, because that is the only thing
    a caller can key on and the three shapes are all real:
      'sdk'     openai-python's BadRequestError, which carries `.param` and
                `.status_code` (measured on this deployment: param='seed',
                code='invalid_type')
      'message' a 4xx whose body only mentions the parameter in prose, which
                is what a self-hosted OpenAI-compatible server returns
      'typeerror' a client whose `create()` does not accept the keyword at all
    """

    def __init__(self, *replies, refuses=(), shape="sdk"):
        super().__init__(*replies)
        self.refuses, self.shape = set(refuses), shape
        outer, inner = self, self.chat.completions

        class _Completions:
            def create(self, **kw):
                bad = sorted(p for p in outer.refuses if p in kw)
                if bad:
                    outer.calls.append(kw)
                    raise outer._refusal(bad[0])
                return inner.create(**kw)

        self.chat = type("Chat", (), {"completions": _Completions()})()

    def _refusal(self, param):
        if self.shape == "typeerror":
            return TypeError(
                f"Completions.create() got an unexpected keyword argument "
                f"'{param}'")
        msg = (f"Error code: 400 - {{'error': {{'message': \"Unsupported "
               f"parameter: '{param}' is not supported with this model.\", "
               f"'type': 'invalid_request_error', 'param': '{param}', "
               f"'code': 'unsupported_parameter'}}}}")
        exc = RuntimeError(msg)
        if self.shape == "sdk":
            exc.status_code, exc.param, exc.code = 400, param, "unsupported_parameter"
        return exc


def _pinned_calls(client):
    """The requests that carried a system prompt this module owns."""
    owns = (V.ADJUDICATE_PROMPT,)
    return [c for c in client.calls
            if any(m.get("content") in owns for m in c["messages"])]


#: An answer whose FIRST claim is uncited-but-plausible, so the judge is
#: actually called. Every test in this section needs a real judge call and
#: nothing else, so they all use this one and vary only the endpoint.
JUDGED_ANSWER = ("The total GCF funding requested is USD 150 million.\n\n"
                 f"FP151 requests **USD 18.5 million** in GCF funding "
                 f"[{DOC}, p. 45].")
JUDGE_REPLY = json.dumps({"verdicts": [{"id": 0, "status": "supported",
                                        "reason": "p.5 states it"}]})


def _judged(client, evidence):
    """Run the judge over `JUDGED_ANSWER` and hand back its verdicts."""
    return V.adjudicate(
        V.classify_deterministic(V.extract_claims(JUDGED_ANSWER), evidence),
        evidence, client=client)


@pytest.mark.parametrize("param,want", [("temperature", 0),
                                        ("seed", V.SAMPLING_SEED)])
def test_every_call_this_module_makes_carries_the_pin(evidence, param, want):
    """DIMENSION: the individual pinned parameter. Wave 4's finding is not
    'the rewrite is random', it is that NEITHER parameter was sent on ANY call
    this module made — so the test varies the parameter, and dropping just one
    of the two still fails."""
    client = FakeClient(JUDGE_REPLY)
    V.verify_answer(JUDGED_ANSWER, evidence, client=client)
    roles = [c["messages"][0]["content"] for c in client.calls]
    assert roles == [V.ADJUDICATE_PROMPT], roles
    for call in client.calls:
        assert call.get(param) == want, (call.get(param), call["messages"][0])


def test_two_judgements_of_the_same_inputs_send_byte_identical_requests(evidence):
    """The property the pin buys, at the layer a test can see without a
    network: same answer + same evidence -> same request. Whether the SERVER
    then returns the same verdicts is an endpoint property and is measured
    separately; what this file can guarantee is that we stopped asking it a
    different question every time.

    Was `test_two_repairs_of_the_same_inputs_send_byte_identical_requests`.
    The rewrite it compared is gone; the request-level property is the half
    that was ever testable here, and it is the judge's now."""
    payloads = []
    for _ in range(2):
        c = FakeClient(JUDGE_REPLY)
        _judged(c, evidence)
        payloads.append(json.dumps(c.calls[0], sort_keys=True))
    assert payloads[0] == payloads[1]
    assert '"seed": %d' % V.SAMPLING_SEED in payloads[0]
    assert '"temperature": 0' in payloads[0]


@pytest.mark.parametrize("shape", ["sdk", "message", "typeerror"])
@pytest.mark.parametrize("refused,kept", [("seed", "temperature"),
                                          ("temperature", "seed")])
def test_a_rejected_parameter_degrades_the_call_and_keeps_the_other_pin(
        evidence, shape, refused, kept):
    """DIMENSION: which parameter the endpoint refuses, crossed with how it
    says so. Some OpenAI-compatible servers 400 on `seed`; some reasoning
    models 400 on `temperature`. Either must cost the parameter, never the
    turn and never the OTHER parameter."""
    client = RejectingClient(JUDGE_REPLY, refuses=(refused,), shape=shape)
    verdicts = _judged(client, evidence)
    assert verdicts[0].status == V.SUPPORTED and verdicts[0].source == "llm"
    assert refused in V._SAMPLING_UNSUPPORTED
    assert kept not in V._SAMPLING_UNSUPPORTED
    assert client.calls[-1].get(kept) is not None        # the other pin held
    assert refused not in client.calls[-1]


@pytest.mark.parametrize("boom", [
    RuntimeError("502 Bad Gateway"),
    RuntimeError("Error code: 429 - rate limit reached for gpt-5.2"),
    TimeoutError("read timed out"),
    RuntimeError("Error code: 400 - {'error': {'message': \"Invalid "
                 "'messages': too many items\", 'param': 'messages'}}"),
])
def test_an_ordinary_failure_may_not_unpin_the_module(evidence, boom):
    """THE WIDENING DIRECTION, and the one that matters. A retry that fires on
    any exception would silently drop the pin on the first flaky call and
    never restore it — the module would go back to unpinned sampling with
    nothing in the record saying when. Only a failure that NAMES a parameter
    we sent is allowed to drop it."""
    client = FakeClient(boom)
    verdicts = _judged(client, evidence)
    assert V._SAMPLING_UNSUPPORTED == set()
    assert len(client.calls) == 1                        # no retry was taken
    # the turn survived: the deterministic verdicts stand, unmoved
    assert [v.source for v in verdicts] == ["deterministic"] * len(verdicts)
    assert verdicts[0].status == V.UNSUPPORTED


def test_an_endpoint_that_refuses_everything_still_terminates(evidence):
    """The retry is bounded by the number of pinned parameters, so a server
    that refuses both cannot turn one judge call into an unbounded call
    loop."""
    client = RejectingClient(JUDGE_REPLY, refuses=("seed", "temperature"))
    verdicts = _judged(client, evidence)
    assert len(client.calls) == len(V._PINNED_SAMPLING) + 1 == 3
    assert verdicts[0].status == V.SUPPORTED and verdicts[0].source == "llm"
    assert V._SAMPLING_UNSUPPORTED == {"seed", "temperature"}


def test_a_refusal_is_remembered_so_it_is_paid_for_once(evidence):
    """A server that does not support `seed` does not start supporting it
    mid-process; re-probing it every turn is latency and cost with no
    information."""
    first = RejectingClient(JUDGE_REPLY, refuses=("seed",))
    _judged(first, evidence)
    second = RejectingClient(JUDGE_REPLY, refuses=("seed",))
    _judged(second, evidence)
    assert len(first.calls) == 2 and len(second.calls) == 1
    assert "seed" not in second.calls[0]
    assert second.calls[0]["temperature"] == V.SAMPLING_TEMPERATURE


def test_verify_only_ever_calls_the_judge(evidence):
    """The answer-generation call is not this module's and is not pinned by
    this change. Pinned here as a structural fact rather than a promise: every
    request verify.py issues carries its own single system prompt — and, since
    eac4c94, there is only one prompt left for it to carry.

    A second reply is loaded into the client on purpose. If any path in this
    module ever grows a second call, it gets that reply and this goes red."""
    client = FakeClient(json.dumps({"verdicts": []}),
                        "a second reply that must never be asked for")
    V.verify_answer(JUDGED_ANSWER, evidence, client=client)
    assert len(client.calls) == 1
    assert _pinned_calls(client) == client.calls


# ---------------------------------------------------------------------------
# (b) the language gate on the adoption path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# (c) the minimum-substance floor
# ---------------------------------------------------------------------------


# NOTE: three sections stood here — (b) the language gate, (c) the minimum
# substance floor, (d) carry-off — 18 tests over the gates that decided
# whether a REWRITE could be adopted. There is no rewrite and no adoption
# decision, so there is nothing left for them to be about. The finding they
# encode is in the module docstring of `verify.py`, which is where a reader
# asking "why is there no repair here?" will look.


# ---------------------------------------------------------------------------
# reproducibility BELOW the model layer: the deterministic classifier must not
# depend on the process's string hash seed
# ---------------------------------------------------------------------------

_HASHSEED_PROBE = '''
import json, sys
sys.path.insert(0, sys.argv[1])
from gcf_qna.rag import verify as V
from gcf_qna.rag import registry as R
R.load = lambda: {}
R.facts = lambda d: {}
A, B = "124_gcf-b27-02-add11", "124_gcf-b27-02-add12"
ev = {(A, 5): "Total GCF funding requested: USD 18.5 million",
      (B, 5): "Total GCF funding requested: USD 150 million"}
ans = "FP151 requests **USD 18.5 million** in GCF funding [124_gcf-b27-02-add1, p. 5]."
(v,) = V.classify_deterministic(V.extract_claims(ans), ev)
print(json.dumps({"status": v.status, "scope": [list(k) for k in v.scope]}))
'''


def test_the_deterministic_layer_is_the_same_under_every_hash_seed():
    """DIMENSION: PYTHONHASHSEED — the only thing that varies between these
    eight processes.

    Wave 4 attributed the 18-27% adoption flip to unpinned sampling. Part of
    it was never in the model at all: both callers of `_resolve_doc` build
    their candidate pool as a SET comprehension over the evidence keys, a
    truncated document id is routinely a prefix of two held documents
    (`...add11` / `...add12`, everywhere in this corpus), and "the first
    prefix match" out of a set is whatever the hash seed makes it. Measured on
    HEAD with the LLM removed entirely and byte-identical inputs:

        seeds 0,1,2 -> supported     seeds 3..7 -> contradicted

    Same answer, same evidence, opposite adoption decision. Pinning the model
    calls does not touch this, and a canary sampling it would be sampling the
    interpreter."""
    import os
    import subprocess
    import sys
    src = str(pathlib.Path(__file__).resolve().parents[1] / "src")
    seen = {}
    for seed in [str(i) for i in range(8)]:
        env = dict(os.environ, PYTHONHASHSEED=seed, PRELOAD="0")
        r = subprocess.run([sys.executable, "-c", _HASHSEED_PROBE, src],
                           capture_output=True, text=True, env=env, timeout=120)
        assert r.returncode == 0, (seed, r.stderr[-800:])
        seen.setdefault(r.stdout.strip(), []).append(seed)
    assert len(seen) == 1, seen


@pytest.mark.parametrize("container", ["list", "reversed", "set", "frozenset",
                                       "tuple", "generator"])
def test_resolve_doc_depends_on_the_contents_and_not_the_container(container):
    """The property, not an example: the same candidates in any order — and in
    any container, since both call sites pass a set comprehension — give the
    same resolution. Written over the container dimension because that is what
    the defect turns on; a test that varied the document ids instead would
    have passed on every revision of this file."""
    docs = [DOC2, DOC, "199_gcf-b40-02-add09"]
    made = {"list": docs, "reversed": list(reversed(docs)), "set": set(docs),
            "frozenset": frozenset(docs), "tuple": tuple(docs),
            "generator": (d for d in reversed(docs))}[container]
    assert V._resolve_doc("124_gcf-b27-02-add1", made) == DOC
    assert V._resolve_doc(DOC2, list(docs)) == DOC2        # exact still wins


# NOTE: two tests stood here over `_unpinned_note` — "a rewrite the endpoint
# would not let us pin says so in its notes", and its permissive twin. The
# helper existed for REWRITES and only `repair()` ever called it, so it went
# with them.
#
# A GAP IS BEING RECORDED HERE, NOT CLOSED. Nothing now puts a pin refusal
# into a turn's record: an endpoint that 400s on `seed` still costs the JUDGE
# call its pin, `_complete` still degrades and still prints one line, and the
# turn's notes stay empty. That was already true before eac4c94 on the shipped
# path (production ran VERIFY_REPAIR=0, so `_unpinned_note` was never reached
# in production), which is exactly why closing it here was not allowed: adding
# a judge-side note would be a NEW observable, and this change had to be
# byte-identical on the live path over both recorded runs. Disclosing an
# unpinned judge sample is a real improvement and it is a SEPARATE one.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# the hole this wave closed, kept as the executable statement it was written
# as. It carried `xfail(reason="OPEN: wave0b N2 / wave0c findings 2 and 7 ...
# retire the marker when it lands")` from 52152af until the
# `_conflict_before_support` design pass; the marker is retired here because
# the pass landed, and the row is now a live pin rather than a record of a
# known defect. An xfail that has started passing is a test nobody is reading.
# ---------------------------------------------------------------------------

def test_a_page_only_repoint_of_a_wrong_figure_is_still_contradicted(no_registry):
    """FP151's cover line prints 18.5 M USD as the GCF financing and 28 M USD
    as the TOTAL. An answer that says the GCF figure is 28 million, cited to
    the cover pages, is correctly CONTRADICTED. Moving the SAME wrong figure's
    bracket to p. 45 — a page that prints 18,500,000 and never prints 28
    million — USED TO make it SUPPORTED with a `citation-page-mismatch`
    caution: zero factual change, a strictly worse citation, and a warning
    replaced by a green status.

    Reproduced on this fixture at 52152af and untouched by any of the wave-4
    repair gates — none of them looked at what a rewrite did to a claim's
    SCOPE, which is why it had to be fixed in VERIFICATION and why it is still
    fixed now that there are no gates and no rewrite. Both citations are
    CONTRADICTED (`_conflict_before_support`); the repoint buys nothing.

    THE ASSERTION CHANGED WITH THE MODULE. It used to end on
    `res.repair_rejected` — the rewrite being refused. What that was standing
    in for is asserted directly here: the repointed sentence is contradicted
    on its own."""
    ev = {(DOC, None): ('Registry — FP151: "Technical Assistance (TA) '
                        'Facility"; GCF financing (as printed): 18.5 M USD; '
                        'total financing (as printed): 28 M USD; board B.27, '
                        f'2020 [{DOC}, cover pages]'),
          (DOC, 45): ("### (a) Requested GCF funding (Total amount)\n"
                      "| (vi) Grants | 18,500,000 | 7 | |")}
    wrong = f"FP151 requests **USD 28 million** in GCF funding [{DOC}, cover pages]."
    moved = f"FP151 requests **USD 28 million** in GCF funding [{DOC}, p. 45]."
    # THE PREMISE IS LOAD-BEARING, and it is asserted first for a reason: an
    # xfail passes on ANY exception, so a fixture that never reaches the
    # subject assertion produces a green run that documents nothing. An
    # earlier version of this test carried a shortened registry line on which
    # the cover-pages citation already verified SUPPORTED — the premise failed,
    # the hole below was never exercised, and the marker hid it. If this line
    # ever goes red the fixture has stopped reproducing and the row is telling
    # you nothing about the hole.
    assert V.classify_deterministic(V.extract_claims(wrong), ev)[0].status == \
        V.CONTRADICTED, "fixture no longer reproduces the conflict"
    (v,) = V.classify_deterministic(V.extract_claims(moved), ev)
    assert v.status == V.CONTRADICTED, (v.status, v.reason, v.flags)
    # a client is passed, and never called (use_llm=False): `_status_for`
    # reads it only to tell 'abstain' from 'no judge was available'
    client = FakeClient()
    res = V.verify_answer(moved, ev, client=client, use_llm=False)
    assert res.status == "abstain" and res.answer == moved
    assert client.calls == []


# ---------------------------------------------------------------------------
# NOTE-PAGE SCOPE
#
# The computed notes print their own provenance — 'GCF funding requested:
# 21,128,224USD (p.6, A.8)', '; also as 49,151,817 USD (p.76, B.2(b))' — and
# the answer prompt tells the model to cite the page a row prints.  TWO of the
# three instruments already agreed that such a page is a legal citation
# target: `chainlit_app._note_pages` feeds `_invalid_citations`, and
# `eval_answers.score_answer` passes the same set.  THE VERIFIER DID NOT.
# `build_evidence` keyed a main registry line at `(doc, None)` only (the
# '[stem, cover pages]' branch returns before the page branch is reached) and
# a conflict line at its FIRST pointer only (`_MATRIX_PAGE_RE.search`), so a
# claim citing the page the note itself had printed resolved to nothing and
# came back UNSUPPORTED, 'cited evidence was never retrieved'.
#
# It cost release-3 two rows, both of them the SECOND HALF of a conflict
# report the registry note had ORDERED the answer to write:
#
#   conf-fp153-gcf   '[122_gcf-b27-02-add13, p. 48]'  <- '; also as 26,654
#                    million USD (p.48, B.2(b))'
#   fr-fp172-nepal   '[103_gcf-b30-03-add04, p. 76]'  <- '; also as
#                    49,151,817 USD (p.76, B.2(b))'
#
# The fixtures below are those two rows' own note blocks, byte for byte.
#
# WHAT THE RULE IS, exactly, because the widening is the dangerous part: a
# cited (doc, page) additionally resolves to a NOTE LINE iff that line names
# that document AND prints '(p.<page>,' or '(p.<page>)' — the same per-line
# attribution `_note_pages` uses.  It is then read by the same matchers at the
# same strictness as any other scope.  A page no note line printed for that
# document keeps failing exactly as before, and the tests that pin THAT are
# the ones to be afraid of losing.
# ---------------------------------------------------------------------------

NOTE_DOC = "103_gcf-b30-03-add04"        # FP172 package, release-3 fr-fp172-nepal
NOTE_DOC2 = "122_gcf-b27-02-add13"       # FP153 package, release-3 conf-fp153-gcf

#: fr-fp172-nepal's registry note, verbatim. The main line prints p.6 twice;
#: the conflict line prints p.6 AND p.76.
FP172_NOTE = (
    'Registry — FP172: "Mitigating GHG emission through modern, efficient and '
    'climate-friendly clean cooking solutions (CCS)"; accredited entity: '
    'Alternative Energy Promotion Centre, Ministry of Energy, Water Resources '
    'and Irrigation, Government of Nepal.; countries: Nepal; GCF funding '
    'requested: 21,128,224USD (p.6, A.8); total financing: 49,151,817 USD '
    f'(p.6, A.7); board B.30, 2021 [{NOTE_DOC}, cover pages]\n'
    f'Registry — CONFLICT in this document ({NOTE_DOC}): gcf_funding_requested '
    'is printed as 21,128,224USD (p.6, A.8); also as 49,151,817 USD '
    '(p.76, B.2(b)) — report both figures with their pages.')

#: conf-fp153-gcf's registry note, verbatim.
FP153_NOTE = (
    'Registry — FP153: "Mongolian Green Finance Corporation"; accredited '
    'entity: XacBank LLC; countries: Mongolia; GCF funding requested: '
    '"28,654 million USD" (p.5, A.8) (unit as printed is ambiguous); total '
    'financing: "49,654 million USD" (p.5, A.7) (unit as printed is '
    f'ambiguous); board B.27, 2020 [{NOTE_DOC2}, cover pages]\n'
    f'Registry — CONFLICT in this document ({NOTE_DOC2}): '
    'gcf_funding_requested is printed as 28,654 million USD (p.5, A.8); also '
    'as 26,654 million USD (p.48, B.2(b)) — report both figures with their '
    'pages.')

#: cmp-fp151-fp152-gcf's registry note: TWO documents in ONE block, which is
#: the only fixture that can tell 'per line' from 'per block'. 124_… prints
#: p.5 and p.60; 123_… prints p.5 and nothing else.
TWO_DOC_NOTE = (
    'Registry — FP151: "Technical Assistance (TA) Facility for the Global '
    'Subnational Climate Fund (SfCfT Global - Equity: submitted separately)"; '
    'accredited entity: International Union for Conservation of Nature and '
    'Natural Resources (IUCN); countries: Angola, Benin, Botswana, Burkina '
    'Faso, Burundi; GCF funding requested: 18.5 M USD (p.5, A.8); total '
    f'financing: 28 M USD (p.5, A.7); board B.27, 2020 [{DOC}, cover pages]\n'
    f'Registry — CONFLICT in this document ({DOC}): total_financing is '
    'printed as 28 M USD (p.5, A.7); also as $720000000 (p.60, B.2(a)) — '
    'report both figures with their pages.\n'
    'Registry — FP152: "Global Sub-national Climate Fund (GCF Global) – '
    'Equity"; accredited entity: Pegasus Capital Advisors LP (Pegasus); '
    'countries: Angola, Burkina Faso, Cameroon, Côte d\'Ivoire, Democratic '
    'Republic of the Congo; GCF funding requested: 150 M USD (p.5, A8); total '
    f'financing: 720 M USD (p.5, A7); board B.27, 2020 [{DOC2}, cover pages]\n'
    'Registry — the identifiers above resolve to DIFFERENT documents. Never '
    'merge them or treat them as the same proposal.')


# --- the keys ---------------------------------------------------------------

def test_a_note_line_publishes_every_page_it_prints_for_its_own_document():
    """Both halves of the measured gap, in one fixture.

    p.6 comes from a MAIN registry line, which the '[stem, cover pages]'
    branch used to consume before any page was read; p.76 is a conflict line's
    SECOND pointer, which `_MATRIX_PAGE_RE.search` used to stop short of. The
    keys are namespaced, and that is asserted rather than assumed: filing them
    at `(doc, page)` would hand every document-wide scan a key it never had.
    """
    ev = V.build_evidence([], [FP172_NOTE])
    scopes = V.note_scopes(ev)
    assert (V.note_scope_doc(NOTE_DOC), 6) in scopes
    assert (V.note_scope_doc(NOTE_DOC), 76) in scopes
    assert "49,151,817" in scopes[(V.note_scope_doc(NOTE_DOC), 76)]
    # the pages are NOT filed as the document's own keys …
    assert (NOTE_DOC, 76) not in ev
    # … and no key at all was added: `claims.evidence_keys` is a recorded
    # artifact and the release backfill asserts it reconstructs exactly
    assert set(ev) == set(V.build_evidence([], [FP172_NOTE]))
    assert not any(V.is_notes_doc(k[0]) and k[1] is not None for k in ev)
    # … and everything the old keying produced is still produced
    assert (NOTE_DOC, None) in ev and "21,128,224USD" in ev[(NOTE_DOC, None)]
    assert (NOTE_DOC, 6) in ev          # the conflict line's first pointer
    assert V.NOTES_KEY in ev


def test_a_note_page_is_attributed_per_line_not_per_block():
    """The rule `_note_pages` applies, and the one an easier reading gets
    wrong: p.60 is printed on 124_…'s conflict line, and 123_… sits in the
    same block three lines down. Per BLOCK it would inherit p.60."""
    scopes = V.note_scopes(V.build_evidence([], [TWO_DOC_NOTE]))
    assert (V.note_scope_doc(DOC), 60) in scopes
    assert (V.note_scope_doc(DOC2), 60) not in scopes
    assert (V.note_scope_doc(DOC2), 5) in scopes


# --- the targeted effect ----------------------------------------------------

def test_the_page_a_note_printed_is_a_scope_the_claim_may_cite():
    """conf-fp153-gcf, the recorded answer verbatim.

    The second claim reports the note's OWN second figure at the note's OWN
    page — compliance with the instruction the same note carries ('report both
    figures with their pages') — and the release scored it UNSUPPORTED, 'cited
    evidence was never retrieved: 122_gcf-b27-02-add13, p.48'.

    The registry is REAL here (no `no_registry`), because the row is: the two
    claims report two figures for one field, so the conflict gate fires on the
    second and is answered by the licence, which the corpus registry is what
    grants. The note scope decides only whether the claim can be READ.
    """
    ev = V.build_evidence([], [FP153_NOTE])
    answer = ('The registry metadata for FP153 (“Mongolian Green Finance '
              'Corporation”) shows **a GCF funding request of “28,654 million '
              f'USD”** (unit as printed is ambiguous). [{NOTE_DOC2}, p. 5]\n\n'
              'However, the same document also prints **“26,654 million USD”** '
              'as the GCF funding requested, creating an internal '
              f'inconsistency. [{NOTE_DOC2}, p. 48]')
    first, second = V.classify_deterministic(V.extract_claims(answer), ev)
    assert first.status == V.SUPPORTED, (first.status, first.reason)
    assert second.status == V.SUPPORTED, (second.status, second.reason)
    assert second.scope == [(V.note_scope_doc(NOTE_DOC2), 48)]
    # and it is not smuggled in as a coarse citation: no page-mismatch caution
    assert "citation-page-mismatch" not in second.flags


def test_the_french_half_of_a_conflict_report_verifies_on_its_printed_page():
    """fr-fp172-nepal, the recorded answer verbatim — narrow no-break spaces,
    French prose, and an English note line. BOTH claims must pass: the first
    on a retrieved page, the second on the page only the note printed.

    NO `no_registry` HERE, and that is the row's own shape rather than a
    convenience: the second claim reports the document's OTHER figure for the
    field its first claim already reported, so the conflict gate fires on it
    and is answered by the 'report both figures with their pages' licence —
    which needs the corpus registry to confirm that BOTH prints are recorded
    for this document and field. Take the registry away and this row is
    CONTRADICTED, which is `test_a_note_scoped_support_is_still_conflict_tested`
    one screen down. The note scope is what makes the claim readable at all;
    every gate behind it still runs.
    """
    hits = [_Hit(NOTE_DOC, 6, "A.8 Financement du FVC demandé 21,128,224 USD")]
    ev = V.build_evidence(hits, [FP172_NOTE])
    answer = (
        "Le FP172 (« Mitigating GHG emission through modern, efficient and "
        "climate-friendly clean cooking solutions (CCS) ») indique un "
        "**financement GCF demandé de 21 128 224 USD** à la section "
        f"A.8. [{NOTE_DOC}, p. 6]  \n\nCependant, le même document présente "
        "aussi **49 151 817 USD** comme « GCF funding "
        "requested » à la section B.2(b), ce qui est signalé comme un "
        f"conflit dans le registre. [{NOTE_DOC}, p. 76]")
    first, second = V.classify_deterministic(V.extract_claims(answer), ev)
    assert first.status == V.SUPPORTED, (first.status, first.reason)
    assert first.scope == [(NOTE_DOC, 6)], "a retrieved page keeps its own key"
    assert second.status == V.SUPPORTED, (second.status, second.reason)
    assert second.scope == [(V.note_scope_doc(NOTE_DOC), 76)]


@pytest.mark.parametrize("claim,cite,want", [
    # the note line prints '18,5 M USD'; the answer may spell it either way
    ("**18,5 millions USD**", 5, V.SUPPORTED),
    ("**18.5 million USD**", 5, V.SUPPORTED),
    ("**USD 18,500,000**", 5, V.SUPPORTED),
    # … and a decimal-comma reading of a different number is still a different
    # number, on the note scope exactly as on a page
    ("**19,5 millions USD**", 5, V.UNSUPPORTED),
])
def test_a_decimal_comma_note_line_is_matched_by_the_same_number_reader(
        no_registry, claim, cite, want):
    """French answers print '18,5 millions USD' for the corpus's '18.5 M USD'
    and the harness records French rows against English note lines. The note
    scope must go through `amounts`/`amount_matches` like everything else —
    'same matchers, same strictness' has to be true in both directions."""
    note = ("Registry — FP151: financement du FVC demandé : 18,5 M USD "
            f"(p.5, A.8); board B.27, 2020 [{DOC}, cover pages]")
    ev = V.build_evidence([], [note])
    answer = f"FP151 demande {claim} de financement du FVC [{DOC}, p. 5]."
    (v,) = V.classify_deterministic(V.extract_claims(answer), ev)
    assert v.status == want, (v.status, v.reason, v.flags)


def test_a_claim_the_note_line_does_not_state_still_fails_on_its_own_page(
        no_registry):
    """SCOPE RESOLUTION ONLY. The citation now resolves; the CONTENT check is
    untouched, so a claim the line does not print fails as it always did — and
    it fails as 'not found in the cited evidence', never as 'never retrieved',
    because the evidence was found and read."""
    ev = V.build_evidence([], [FP172_NOTE])
    answer = (f"The document also prints **USD 88,000,000** as the GCF funding "
              f"requested. [{NOTE_DOC}, p. 76]")
    (v,) = V.classify_deterministic(V.extract_claims(answer), ev)
    assert v.status == V.UNSUPPORTED, (v.status, v.reason)
    assert "never retrieved" not in v.reason
    assert "88,000,000" in v.reason


# --- the attacks ------------------------------------------------------------

@pytest.mark.parametrize("page", [75, 77, 7, 760, 6_1])
def test_a_page_no_note_line_printed_is_still_never_retrieved(
        no_registry, page):
    """THE ATTACK THE WIDENING INVITES: p.76 is printed, so try p.75.

    An invented page that reads as precision is the failure ruling 5 exists to
    avoid pushing generation toward, and it is the one the fabricated arm
    watches. The verdict text matters as much as the verdict: 'never
    retrieved' is what the harness and the app both key off.
    """
    ev = V.build_evidence([], [FP172_NOTE])
    answer = (f"The document also prints **49,151,817 USD** as the GCF funding "
              f"requested. [{NOTE_DOC}, p. {page}]")
    (v,) = V.classify_deterministic(V.extract_claims(answer), ev)
    assert v.status == V.UNSUPPORTED, (page, v.status, v.reason)
    assert "cited evidence was never retrieved" in v.reason
    assert f"invalid-citation:{NOTE_DOC}, p.{page}" in v.flags


def test_a_page_printed_for_one_document_cannot_be_cited_for_its_neighbour(
        no_registry):
    """THE ATTACK ACROSS THE BLOCK. 124_… prints p.60 and 123_… does not, and
    both lines sit in one note. The figure is real, the document is real, the
    pairing is invented — and 'the note block printed p.60 somewhere' must not
    be enough."""
    ev = V.build_evidence([], [TWO_DOC_NOTE])
    answer = f"FP152's total financing is **$720000000** [{DOC2}, p. 60]."
    (v,) = V.classify_deterministic(V.extract_claims(answer), ev)
    assert v.status == V.UNSUPPORTED, (v.status, v.reason)
    assert "cited evidence was never retrieved" in v.reason
    # the same sentence pointed at the document whose line prints p.60 resolves
    ok = f"FP151's total financing is also printed as **$720000000** [{DOC}, p. 60]."
    (w,) = V.classify_deterministic(V.extract_claims(ok), ev)
    assert w.scope == [(V.note_scope_doc(DOC), 60)]


@pytest.mark.parametrize("value", ["49,151,818", "50,000,000", "4,915,181",
                                   "49.151.817"])
def test_a_value_that_contradicts_the_note_line_is_never_supported(
        no_registry, value):
    """THE ATTACK ON THE CONTENT SIDE: ride the new scope in with a figure the
    line refutes — one digit off, an order of magnitude off, a re-punctuated
    reading. CONTRADICTED or UNSUPPORTED, never SUPPORTED."""
    ev = V.build_evidence([], [FP172_NOTE])
    answer = (f"The **GCF funding requested** is also printed as **{value} "
              f"USD**. [{NOTE_DOC}, p. 76]")
    (v,) = V.classify_deterministic(V.extract_claims(answer), ev)
    assert v.status != V.SUPPORTED, (value, v.status, v.reason)
    assert v.failed


def test_a_note_scoped_support_is_still_conflict_tested(no_registry):
    """THE GATE, on the new scope. `_conflict_before_support` is the ONE gate
    between a verified claim and SUPPORTED, and a fourth way to become
    SUPPORTED is exactly the shape that historically walked around it: the
    note line carries the figure, and the document's own p.48 prints a
    different one under the claim's own field label."""
    hits = [_Hit(NOTE_DOC, 48, "GCF funding requested: 38,000,000 USD")]
    ev = V.build_evidence(hits, [FP172_NOTE])
    answer = (f"The **GCF funding requested** is **USD 49,151,817** "
              f"[{NOTE_DOC}, p. 76].")
    (v,) = V.classify_deterministic(V.extract_claims(answer), ev)
    assert v.status == V.CONTRADICTED, (v.status, v.reason, v.flags)
    assert "38,000,000" in v.reason
    assert "conflict-elsewhere-in-document" in v.flags
    # and the switch that names cross-page scans still names this one
    (off,) = V.classify_deterministic(V.extract_claims(answer), ev,
                                      cross_page_conflicts=False)
    assert off.status == V.SUPPORTED


def test_a_note_scoped_verdict_never_claims_a_scope_it_did_not_test(
        no_registry):
    """The structural invariant of
    `test_a_widened_verdict_never_claims_a_scope_it_did_not_test`, re-asserted
    over the scope this change adds."""
    hits = [_Hit(NOTE_DOC, 6, "A.8 GCF funding requested 21,128,224 USD")]
    ev = V.build_evidence(hits, [FP172_NOTE])
    for answer in (f"The GCF funding requested is **21,128,224 USD** [{NOTE_DOC}, p. 6].",
                   f"It is also printed as **49,151,817 USD** [{NOTE_DOC}, p. 76].",
                   f"It is printed as **49,151,817 USD** [{NOTE_DOC}, cover pages]."):
        for v in V.classify(V.extract_claims(answer), ev, use_llm=False):
            if v.status != V.SUPPORTED:
                continue
            assert V._conflict_before_support(
                v.claim, ev, v.scope, v.scope, []) is None, (answer, v.reason)


def test_every_supported_exit_of_the_deterministic_pass_runs_the_gate_first():
    """THE PIN, structurally: not 'these fixtures pass the gate' but 'no
    SUPPORTED verdict can be constructed in `classify_deterministic` before
    `_conflict_before_support` has run in its own block chain'.

    A fifth support branch added for a fifth kind of scope fails here on the
    day it is written, which is the only version of this test that survives
    the next widening.
    """
    import ast
    import inspect
    fn = next(n for n in ast.walk(ast.parse(inspect.getsource(V)))
              if isinstance(n, ast.FunctionDef)
              and n.name == "classify_deterministic")

    def gate_call(node):
        return any(isinstance(n, ast.Call)
                   and getattr(n.func, "id", "") == "_conflict_before_support"
                   for n in ast.walk(node))

    def supported(node):
        return [n for n in ast.walk(node)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") == "Verdict"
                and len(n.args) >= 2 and getattr(n.args[1], "id", "") == "SUPPORTED"]

    seen, ungated = [], []

    def walk(body, gated):
        for stmt in body:
            subs = [b for name in ("body", "orelse", "finalbody")
                    for b in [getattr(stmt, name, None)] if isinstance(b, list)]
            nested = {id(n) for sub in subs for s in sub for n in supported(s)}
            own = [n for n in supported(stmt) if id(n) not in nested]
            for n in own:
                seen.append(n.lineno)
                if not gated:
                    ungated.append(n.lineno)
            for sub in subs:
                walk(sub, gated)
            if gate_call(stmt):
                gated = True

    walk(fn.body, False)
    assert seen, "found no SUPPORTED verdict at all — the pin has gone blind"
    assert len(seen) >= 2, f"only {len(seen)} SUPPORTED exits found; expected 2"
    assert not ungated, (
        f"SUPPORTED constructed at line(s) {ungated} of verify.py without "
        f"`_conflict_before_support` having run first")


# --- the things that must NOT have moved ------------------------------------

def test_a_retrieved_page_is_never_given_the_notes_text(no_registry):
    """NARROWNESS. The lookup fires only when the cited (doc, page) is held by
    nothing else, so a page that WAS retrieved is judged on its own text —
    the note line does not join it. Otherwise this stops being a scope fix and
    becomes a content widening on every page a registry line happens to name.
    """
    main_line = FP172_NOTE.splitlines()[0]      # no conflict line: p.6 is
    hits = [_Hit(NOTE_DOC, 6, "A.8 — the requested amount table is on p.7.")]
    ev = V.build_evidence(hits, [main_line])    # published by the note only
    assert (V.note_scope_doc(NOTE_DOC), 6) in V.note_scopes(ev), \
        "fixture: p.6 is note-printed"
    answer = (f"The GCF funding requested is **21,128,224 USD** "
              f"[{NOTE_DOC}, p. 6].")
    (v,) = V.classify_deterministic(V.extract_claims(answer), ev)
    assert v.scope[0] == (NOTE_DOC, 6), "the cited page is the retrieved one"
    assert not any(V.is_notes_doc(k[0]) for k in v.scope), v.scope
    # the value is in the note line and therefore in the document's COVER
    # scope, so this is the page-mismatch verdict the widened branch already
    # emitted before this change — NOT 'value found in the cited evidence',
    # which is what a note line joined onto p.6's own text would have produced
    assert "not on the cited page" in v.reason, (v.status, v.reason)
    assert "citation-page-mismatch" in v.flags


def test_a_note_scope_is_not_a_document(no_registry):
    """The pseudo-document must stay invisible to everything that asks 'which
    document is this claim about?'. If a note scope counted as a second
    document, `_reported_elsewhere` would stop attributing the claim (it needs
    EXACTLY one) and the 'report both figures' licence — 23 adjudicated rows —
    would die on every note-scoped claim."""
    assert V.is_notes_doc(V.NOTES_DOC)
    assert V.is_notes_doc(V.note_scope_doc(NOTE_DOC))
    assert not V.is_notes_doc(NOTE_DOC)
    ev = V.build_evidence([], [FP172_NOTE])
    # ruling 5's coarse scope is the document's OWN keys and gains nothing
    claim = V.extract_claims(f"It prints **49,151,817 USD** [{NOTE_DOC}].")[0]
    strict, wide, bad, r5 = V._scopes(claim, ev)
    assert not any(V.is_notes_doc(k[0]) for k in strict + wide + r5), \
        (strict, wide, r5)


def test_the_judge_is_shown_a_note_scope_by_name(no_registry):
    """The snippet the adjudication prompt carries has to say WHERE the text
    came from; '__notes__:103_…' would be the pseudo-document leaking into a
    model prompt."""
    ev = V.build_evidence([], [FP172_NOTE])
    snippet = V._evidence_snippet(ev, [(V.note_scope_doc(NOTE_DOC), 76)])
    assert "__notes__" not in snippet
    assert f"computed notes for {NOTE_DOC}, p.76" in snippet
