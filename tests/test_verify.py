"""Claim verification and repair (plan step 5).

No test here touches the network: the LLM layer is exercised through fake
clients, and the degradation tests remove OPENAI_API_KEY outright. The
fixture evidence set is deliberately tiny — four keys — so every verdict in
this file can be checked by reading the fixture.
"""
import inspect
import json

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


def test_page_mismatch_is_supported_but_flagged(evidence):
    """A real figure attached to the wrong page of the right document is a
    citation defect, not an invention — and must not read the same.

    'total financing 28 M USD' is printed on the registry cover line, not on
    the cited p.45.
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


def test_registry_not_found_supports_the_exact_negative_existence_claim():
    note = ("Registry — FP999: NOT FOUND in the 273-document corpus registry. "
            "Do not infer details for it from other documents.")
    ev = V.build_evidence([], note)
    answer = "FP999 does not exist in this corpus [registry note in context]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED


def test_registry_not_found_for_another_id_cannot_support_existence_claim():
    ev = V.build_evidence([], "Registry — FP998: NOT FOUND in the corpus registry.")
    answer = "FP999 does not exist in this corpus [registry note in context]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.UNSUPPORTED


def test_uncited_note_backed_claim_is_grounded_but_not_citation_complete():
    note = "Registry — FP999: NOT FOUND in the 273-document corpus registry."
    ev = V.build_evidence([], note)
    (v,) = V.classify(V.extract_claims("FP999 does not exist in this corpus."),
                      ev, use_llm=False)
    assert v.status == V.UNSUPPORTED
    assert V.GROUNDED_FLAG in v.flags
    assert "no citation" in v.reason
    # ONE flag carries groundedness. A second name for the same condition is
    # a fact with two spellings, and the two drift.
    assert "grounded-without-citation" not in v.flags
    assert [f for f in v.flags if "grounded" in f] == []


def test_unknown_board_code_uses_not_found_semantics_not_board_mentions():
    note = "Registry — GCF/B.42/02/Add.99: NOT FOUND in the corpus registry."
    ev = V.build_evidence([], note)
    answer = ("GCF/B.42/02/Add.99 does not exist in this corpus "
              "[registry note in context].")
    (claim,) = V.extract_claims(answer)
    assert claim.kind == "existence"
    (v,) = V.classify([claim], ev, use_llm=False)
    assert v.status == V.SUPPORTED


def test_registered_explicit_alias_can_back_a_full_name(monkeypatch):
    full = "International Union for Conservation of Nature and Natural Resources"
    monkeypatch.setattr("gcf_qna.rag.registry.load", lambda: {
        DOC: {"accredited_entity": f"{full} (IUCN)"}})
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: {})
    ev = {(DOC, 5): "Accredited entity: IUCN"}
    answer = f"The accredited entity is **{full} (IUCN)** [{DOC}, p. 5]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED
    assert "registry-backed-page-not-retrieved" in v.flags


def test_invented_acronym_expansion_is_not_accepted(monkeypatch):
    actual = "International Union for Conservation of Nature and Natural Resources"
    monkeypatch.setattr("gcf_qna.rag.registry.load", lambda: {
        DOC: {"accredited_entity": f"{actual} (IUCN)"}})
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: {})
    ev = {(DOC, 5): "Accredited entity: IUCN"}
    answer = (f"The accredited entity is **Invented Universal Climate Network "
              f"(IUCN)** [{DOC}, p. 5].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.UNSUPPORTED


def test_arbitrary_initials_are_not_derived_from_a_full_name(monkeypatch):
    monkeypatch.setattr("gcf_qna.rag.registry.load", lambda: {DOC: {}})
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: {})
    ev = {(DOC, 5): "Accredited entity: Imaginary Finance Corporation"}
    answer = f"The accredited entity is **IFC** [{DOC}, p. 5]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.UNSUPPORTED


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
    """Reporting BOTH sides of a known conflict — what the prompt asks for and
    what the repair pass produces — cites a page this turn may not hold. That
    must verify, or no conflict answer could ever pass."""
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


def _money_conflict_registry(monkeypatch):
    facts = {
        DOC: {"gcf_funding_requested": [
            {"raw": "USD 10 million", "page": 5, "status": "canonical"},
            {"raw": "USD 20 million", "page": 48, "status": "conflicting"}]},
        DOC2: {"gcf_funding_requested": [
            {"raw": "USD 20 million", "page": 5, "status": "canonical"}]},
    }
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: facts.get(doc, {}))


def test_unrelated_same_document_number_does_not_suppress_conflict(monkeypatch):
    _money_conflict_registry(monkeypatch)
    ev = {(DOC, None): ("GCF financing: USD 10 million; "
                        "total financing: USD 20 million")}
    answer = (f"GCF funding is **USD 10 million** and total financing is "
              f"**USD 20 million** [{DOC}, cover pages].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.CONTRADICTED
    assert "known-document-conflict" in v.flags


def test_cross_document_number_does_not_suppress_conflict(monkeypatch):
    _money_conflict_registry(monkeypatch)
    ev = {
        (DOC, None): "GCF financing: USD 10 million",
        (DOC2, None): "GCF financing: USD 20 million",
    }
    answer = (f"FP151 requests **USD 10 million** in GCF funding "
              f"[{DOC}, cover pages], while FP152 requests **USD 20 million** "
              f"in GCF funding [{DOC2}, cover pages].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.CONTRADICTED
    assert "known-document-conflict" in v.flags


@pytest.mark.parametrize("counterpart", ["EUR 20 million", "USD 20 billion"])
def test_incompatible_counterpart_does_not_suppress_conflict(
        monkeypatch, counterpart):
    _money_conflict_registry(monkeypatch)
    ev = {(DOC, None): f"GCF financing: USD 10 million; note: {counterpart}"}
    answer = (f"FP151 requests **USD 10 million** in GCF funding; another "
              f"printed figure is **{counterpart}** [{DOC}, cover pages].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.CONTRADICTED
    assert "known-document-conflict" in v.flags


def test_same_document_same_field_compatible_counterpart_suppresses_conflict(
        monkeypatch):
    _money_conflict_registry(monkeypatch)
    ev = {(DOC, None): "GCF financing: USD 10 million"}
    answer = (f"FP151 requests **USD 10 million** in GCF funding "
              f"[{DOC}, cover pages], while the same document prints "
              f"**USD 20 million** [{DOC}, p. 48].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED
    assert "registry-backed-page-not-retrieved" in v.flags


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
# repair
# ---------------------------------------------------------------------------

def test_repair_fixes_a_contradicted_value_and_reverifies(evidence):
    answer = f"FP151 requests **USD 25 million** in GCF funding [{DOC}, cover pages]."
    fixed = f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, cover pages]."
    client = FakeClient(fixed)
    res = V.verify_answer(answer, evidence, client=client)
    assert res.status == "repaired" and res.repaired
    assert res.answer == fixed
    assert not res.failures


def test_repair_is_rejected_when_it_invents_a_source(evidence):
    """Rule 4 of the repair prompt is enforced in python, not by asking nicely."""
    answer = f"FP151 requests **USD 25 million** in GCF funding [{DOC}, cover pages]."
    invented = (f"FP151 requests **USD 18.5 million** in GCF funding "
                f"[{DOC}, p. 512; 999_gcf-b99-99-add99, p. 3].")
    client = FakeClient(invented)
    res = V.verify_answer(answer, evidence, client=client)
    assert res.repair_rejected and not res.repaired
    assert res.answer == answer                        # the original, flagged
    assert res.status == "abstain"                     # its only claim failed
    assert any("999_gcf-b99-99-add99" in n for n in res.notes)
    # the invented PAGE of a real document is caught by the same check
    assert any(f"{DOC}, p.512" in n for n in res.notes)


def test_repair_that_swaps_one_wrong_figure_for_another_is_rejected(evidence):
    """The reviewer's repro: 58M is wrong, the model 'fixes' it to 61M, which
    is equally absent from the evidence. Shipping that is worse than flagging
    the original, because it looks corrected."""
    answer = f"FP151 requests **USD 58 million** in GCF funding [{DOC}, cover pages]."
    client = FakeClient(
        f"FP151 requests **USD 61 million** in GCF funding [{DOC}, cover pages].")
    res = V.verify_answer(answer, evidence, client=client)
    assert res.repair_rejected and not res.repaired
    assert res.answer == answer                        # the original, flagged
    assert "still fail verification" in " ".join(res.notes)


def test_repair_that_re_attributes_a_claim_to_another_document_is_rejected(evidence):
    """Moving the claim onto a different retrieved document's figure verifies
    cleanly — and is an invented attribution, not a repair."""
    answer = f"FP151 requests **USD 58 million** in GCF funding [{DOC}, p. 45]."
    client = FakeClient(
        f"FP151 requests **USD 150 million** in GCF funding [{DOC2}, p. 5].")
    res = V.verify_answer(answer, evidence, client=client, use_llm=False)
    assert res.repair_rejected and res.answer == answer
    assert any("not shown to the repair pass" in n for n in res.notes)


def test_repair_that_guts_a_mostly_correct_answer_is_rejected(evidence):
    """A bare refusal is not a repair of an answer that had supported facts."""
    answer = (f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, p. 45].\n\n"
              f"Its accredited entity is **Pegasus Capital Advisors LP** "
              f"[{DOC}, cover pages].")
    client = FakeClient("The retrieved excerpts do not state FP151's funding.")
    res = V.verify_answer(answer, evidence, client=client, use_llm=False)
    assert res.repair_rejected and res.answer == answer
    assert any("removed every supported factual claim" in n for n in res.notes)
    assert res.status == "partial"


def test_introduced_source_check_is_exact_not_prefix(evidence):
    """182 of the 273 corpus ids are <= 24 characters: a prefix match would
    wave through any suffix appended to one of them."""
    answer = f"FP151 requests **USD 58 million** in GCF funding [{DOC}, p. 45]."
    client = FakeClient(f"FP151 requests **USD 18,500,000** "
                        f"[{DOC}-annex-volume-2, p. 45].")
    res = V.verify_answer(answer, evidence, client=client, use_llm=False)
    assert res.repair_rejected and res.answer == answer
    assert any(f"{DOC}-annex-volume-2" in n for n in res.notes)


def test_repair_output_preamble_is_stripped(evidence):
    answer = f"FP151 requests **USD 25 million** in GCF funding [{DOC}, cover pages]."
    fixed = f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, cover pages]."
    client = FakeClient("Sure! Here is the repaired answer:\n\n" + fixed)
    res = V.verify_answer(answer, evidence, client=client)
    assert res.repaired and res.answer == fixed


def test_repair_runs_with_the_judge_switched_off(evidence):
    """use_llm and allow_repair are independent switches: VERIFY_LLM=0 turns
    off the judge, not the repair pass."""
    answer = f"FP151 requests **USD 25 million** in GCF funding [{DOC}, cover pages]."
    fixed = f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, cover pages]."
    client = FakeClient(fixed)
    res = V.verify_answer(answer, evidence, client=client, use_llm=False)
    assert len(client.calls) == 1                       # repair only, no judge
    assert "You repair an answer" in client.calls[0]["messages"][0]["content"]
    assert res.status == "repaired" and res.answer == fixed


def test_judge_verdicts_survive_the_post_repair_recheck(evidence):
    """The recheck is deterministic-only; without carrying the judge's rulings
    a cleared paraphrase comes back unsupported and sinks a good repair."""
    answer = ("The accredited entity is **Pegasus Capital Advisors LP**.\n\n"
              f"FP151 requests **USD 25 million** in GCF funding [{DOC}, cover pages].")
    fixed = ("The accredited entity is **Pegasus Capital Advisors LP**.\n\n"
             f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, cover pages].")
    client = FakeClient(
        json.dumps({"verdicts": [{"id": 0, "status": "supported",
                                  "reason": "the cited page names it"}]}),
        fixed)
    res = V.verify_answer(answer, evidence, client=client)
    assert res.status == "repaired" and res.answer == fixed
    carried = [v for v in res.verdicts if v.source == "llm"]
    assert len(carried) == 1 and carried[0].status == V.SUPPORTED


def test_repair_may_delete_the_claim_entirely(evidence):
    """Removing an unsupportable claim is a valid repair: what is left states
    no fact, so nothing is left to contradict the evidence."""
    answer = f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 99]."
    client = FakeClient("The retrieved excerpts do not state FP151's GCF funding.")
    res = V.verify_answer(answer, evidence, client=client)
    assert res.status == "repaired" and res.answer.startswith("The retrieved")
    assert V.extract_claims(res.answer) == []


def test_abstain_when_every_fact_bearing_claim_failed(evidence):
    answer = (f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 99].\n"
              f"Its accredited entity is **Pegasus Capital Advisors LP** "
              f"[{DOC}, cover pages].")
    client = FakeClient(answer)                        # the model changed nothing
    res = V.verify_answer(answer, evidence, client=client)
    assert res.status == "abstain"
    assert len(res.failures) == 2


def test_partial_when_one_of_two_claims_stays_unsupported(evidence):
    answer = (f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, p. 45].\n"
              f"FP151 also received **USD 40 million** in co-financing [{DOC}, p. 99].")
    client = FakeClient(answer)                        # the model changed nothing
    res = V.verify_answer(answer, evidence, client=client)
    assert res.status == "partial"
    assert res.repair_rejected                         # no improvement, no adoption
    assert len(res.unsupported) == 1
    assert res.counts()[V.SUPPORTED] == 1


def test_at_most_two_llm_calls_per_answer(evidence):
    answer = ("The total GCF funding requested is USD 150 million.\n\n"
              f"FP151 requests **USD 25 million** in GCF funding [{DOC}, cover pages].")
    client = FakeClient(json.dumps({"verdicts": [{"id": 0, "status": "unsupported",
                                                 "reason": "not stated"}]}),
                        f"FP151 requests **USD 18.5 million** [{DOC}, cover pages].")
    V.verify_answer(answer, evidence, client=client)
    assert len(client.calls) == 2                      # 1 adjudicate + 1 repair


def test_repair_token_budget_is_sized_to_the_answer(evidence):
    answer = f"FP151 requests **USD 25 million** in GCF funding [{DOC}, cover pages]."
    client = FakeClient(answer)
    V.repair(answer, V.classify(V.extract_claims(answer), evidence, use_llm=False),
             evidence, client=client)
    assert client.calls[0]["max_completion_tokens"] >= 900
    assert client.calls[0]["model"]                     # config.CHAT_MODEL, not hardcoded


# ---------------------------------------------------------------------------
# degradation
# ---------------------------------------------------------------------------

def test_no_api_key_means_deterministic_verdicts_and_no_repair(monkeypatch, evidence):
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


# ---------------------------------------------------------------------------
# Wave 0b rework — the regressions docs/wave0-review-verdict.md §2 reproduced
#
# Every test below fails on the held wave (148a7cc) and passes after the fix.
# Each matcher change is pinned in BOTH directions — an adversarial negative
# beside the permissive regression — because the held suite pinned only the
# strict side, which is exactly why three defeated gates survived 575 green
# tests. Where a fix restores parent behaviour, the parent's own guard is
# re-asserted beside it so the restoration cannot go one step too far.
# ---------------------------------------------------------------------------

FP172 = "103_gcf-b30-03-add04"


def _no_registry(monkeypatch):
    """Isolate the evidence path: no registry rescue, no registry conflicts."""
    monkeypatch.setattr("gcf_qna.rag.registry.load", lambda: {})
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: {})


def _facts(monkeypatch, mapping):
    monkeypatch.setattr("gcf_qna.rag.registry.facts",
                        lambda doc: mapping.get(doc, {}))


# --- verdict §2 #1 — the two-field sentence -------------------------------

@pytest.mark.parametrize("text,field,want", [
    # the value printed AFTER its label — the shape that always worked
    ("GCF financing (as printed): 18.5 M USD; total financing: 28 M USD",
     "gcf_financing", "18.5"),
    # the value printed BEFORE its label: the window started at the label, so
    # every figure to its left was dropped and the list came back EMPTY
    ("FP151 requests USD 28,000,000 as GCF financing, out of a total "
     "financing of USD 49,000,000.", "gcf_financing", "28,000,000"),
    ("USD 28,000,000 total financing and USD 18,500,000 GCF financing",
     "gcf_financing", "18,500,000"),
    ("USD 28,000,000 total financing and USD 18,500,000 GCF financing",
     "total_financing", "28,000,000"),
])
def test_a_field_window_never_loses_the_figure_its_label_states(text, field, want):
    got = [a.raw for a in V._field_context_amounts(text, field)]
    # an empty list reads as 'nothing to check' at every call site
    assert got, f"{field} of {text!r} produced no stated figure"
    assert any(want in raw for raw in got), got


def test_a_field_window_does_not_swallow_the_field_beside_it():
    """The permissive half: widening the window must not make every figure on
    the line the field's own value, or a genuine disagreement is suppressed."""
    line = "GCF financing (as printed): 18.5 M USD; total financing: 28 M USD"
    got = [a.raw for a in V._field_context_amounts(line, "gcf_financing")]
    assert got == ["18.5 M USD"]


def test_a_two_field_sentence_still_reports_its_own_field(evidence):
    """verdict §2 #1: a repair that reports the TOTAL financing figure as the
    GCF figure. The registry line prints 18.5 M USD for GCF and 28 M USD for
    total; naming both fields must not disable the check."""
    answer = (f"FP151 requests **USD 28,000,000** as GCF financing, out of a "
              f"total financing of USD 28,000,000 [{DOC}, cover pages].")
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.status == V.CONTRADICTED
    assert "18.5 M USD" in v.reason


def test_a_two_field_sentence_that_agrees_with_both_fields_is_supported(evidence):
    """...and the same sentence with the RIGHT figures still verifies: the
    fix must fail closed, not fail always."""
    answer = (f"FP151 requests **USD 18.5 million** as GCF financing, out of a "
              f"total financing of **USD 28 million** [{DOC}, cover pages].")
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.status == V.SUPPORTED


def test_repair_reporting_the_total_as_the_gcf_figure_is_rejected(evidence):
    """The gate, end to end: the swapped sentence re-verifies and is adopted
    only if the contradiction test short-circuits."""
    answer = f"FP151 requests **USD 61 million** as GCF financing [{DOC}, cover pages]."
    swapped = (f"FP151 requests **USD 28,000,000** as GCF financing, out of a "
               f"total financing of USD 28,000,000 [{DOC}, cover pages].")
    res = V.verify_answer(answer, evidence, client=FakeClient(swapped))
    assert res.repair_rejected and not res.repaired
    assert res.answer == answer
    assert "still fail verification" in res.notes[0]


def test_repair_correcting_the_gcf_figure_is_still_adopted(evidence):
    """The permissive half of the same gate: a real correction must land."""
    answer = f"FP151 requests **USD 61 million** as GCF financing [{DOC}, cover pages]."
    fixed = (f"FP151 requests **USD 18.5 million** as GCF financing, out of a "
             f"total financing of **USD 28 million** [{DOC}, cover pages].")
    res = V.verify_answer(answer, evidence, client=FakeClient(fixed))
    assert res.repaired and res.answer == fixed


def test_a_known_document_conflict_survives_a_two_field_sentence(monkeypatch):
    """classify_deterministic gated the registry check on the same empty list
    (verdict §2 #1, second half)."""
    _facts(monkeypatch, {DOC: {"gcf_funding_requested": [
        {"raw": "USD 21,128,224", "page": 6, "status": "canonical"},
        {"raw": "USD 49,151,817", "page": 76, "status": "conflicting"}]}})
    ev = {(DOC, None): ("GCF funding requested: USD 21,128,224; "
                        "total financing: USD 49,151,817")}
    answer = (f"FP172 requests **USD 21,128,224** as GCF financing, out of a "
              f"total financing of USD 49,151,817 [{DOC}, cover pages].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.CONTRADICTED
    assert "known-document-conflict" in v.flags


def test_registry_backing_is_not_lost_on_a_two_field_sentence(monkeypatch):
    """verdict finding 11 — the same empty local view stripped real registry
    backing off any sentence naming two fields, a live false positive."""
    _facts(monkeypatch, {DOC: {"gcf_funding_requested": [
        {"raw": "USD 26,654,000", "page": 48, "status": "canonical"}]}})
    ev = {(DOC, None): "total financing (as printed): 28 M USD"}
    answer = (f"FP151 requests **USD 26,654,000** as GCF financing, out of a "
              f"total financing of **28 M USD** [{DOC}, cover pages].")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED
    assert "registry-backed-page-not-retrieved" in v.flags


def test_registry_backing_stays_scoped_to_the_clause_that_cites_the_document():
    """The wave's genuine tightening, kept: with one bracket per document a
    registry row may back only the figures that bracket attributes to it."""
    answer = (f"FP151 requests **USD 10 million** [{DOC}, cover pages], while "
              f"FP152 requests **USD 20 million** [{DOC2}, cover pages].")
    (claim,) = V.extract_claims(answer)
    assert V.registry_backed(DOC, claim, V.amounts("USD 20 million")) is None


# --- verdict §2 #2 — the chained citation bracket -------------------------

def _fp172_conflict(monkeypatch):
    _facts(monkeypatch, {FP172: {"gcf_funding_requested": [
        {"raw": "21,128,224 USD", "page": 6, "status": "canonical"},
        {"raw": "49,151,817 USD", "page": 76, "status": "conflicting"}]}})
    return {(FP172, 6): "A.8. Total GCF funding requested: 21,128,224 USD",
            (FP172, 76): "Requested GCF amount: 49,151,817 USD",
            (DOC, 5): "Accredited entity: IUCN"}


@pytest.mark.parametrize("bracket", [
    f"[{FP172}, p. 6]",                        # as recorded
    f"[{FP172}, p. 6; {DOC}, p. 5]",           # one document wider
])
def test_widening_a_bracket_cannot_turn_a_contradiction_into_support(
        monkeypatch, bracket):
    """verdict §2 #2, the recorded FP172 sentence: 21,128,224 on p.6 against
    49,151,817 on p.76. _citation_context returned '' for EVERY document of a
    chained bracket, so the conflict test walked past all of them — the same
    sentence flipped to SUPPORTED with an empty flags list."""
    ev = _fp172_conflict(monkeypatch)
    answer = (f"- **USD 21,128,224** — listed in the project summary as "
              f"“A.8. Total GCF funding requested” (p.6) {bracket}.")
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.CONTRADICTED, f"{bracket}: {v.reason}"
    assert "49,151,817" in v.reason


def test_an_inherited_block_citation_is_checked_against_what_it_names(monkeypatch):
    """A claim whose citations were inherited prints no bracket of its own, so
    the clause-carving loop found nothing and returned '' — the same hole,
    reached without touching a bracket at all."""
    ev = _fp172_conflict(monkeypatch)
    answer = (f"- **USD 21,128,224** — the A.8 Total GCF funding requested figure\n"
              f"[{FP172}, p. 6; {DOC}, p. 5]")
    (claim,) = V.extract_claims(answer)
    assert claim.inherited and len(claim.citations) == 2
    (v,) = V.classify([claim], ev, use_llm=False)
    assert v.status == V.CONTRADICTED


def test_repair_cannot_clear_a_conflict_by_widening_a_bracket(monkeypatch):
    """End to end, with both documents already cited by the original so the
    introduced-source gate cannot be what catches it."""
    ev = _fp172_conflict(monkeypatch)
    answer = (f"- **USD 99,999,999** — A.8 Total GCF funding requested "
              f"[{FP172}, p. 6].\n"
              f"- The accredited entity is IUCN [{DOC}, p. 5].")
    widened = (f"- **USD 21,128,224** — A.8 Total GCF funding requested "
               f"[{FP172}, p. 6; {DOC}, p. 5].\n"
               f"- The accredited entity is IUCN [{DOC}, p. 5].")
    res = V.repair(answer, V.classify_deterministic(V.extract_claims(answer), ev),
                   ev, client=FakeClient(widened))
    assert res.repair_rejected and not res.repaired
    assert res.answer == answer


def test_single_document_brackets_still_scope_their_own_clause():
    """The permissive half: per-bracket scoping is what stops one document's
    figure satisfying a claim about another, and it must survive the fix."""
    answer = (f"FP151 requests **USD 10 million** [{DOC}, cover pages], while "
              f"FP152 requests **USD 20 million** [{DOC2}, cover pages].")
    (claim,) = V.extract_claims(answer)
    first, second = V._citation_context(claim, DOC), V._citation_context(claim, DOC2)
    assert "10 million" in first and "20 million" not in first
    assert "20 million" in second and "10 million" not in second
    # a document the claim never cites is still given nothing
    assert V._citation_context(claim, "999_never-cited") == ""


def test_a_chained_bracket_offers_its_clause_to_every_document_it_names():
    answer = f"FP151 requests **USD 10 million** [{DOC}, p. 5; {DOC2}, p. 5]."
    (claim,) = V.extract_claims(answer)
    assert all("10 million" in V._citation_context(claim, d) for d in (DOC, DOC2))
    assert V._citation_context(claim, "999_never-cited") == ""


# --- verdict §2 #3 — the entity matcher -----------------------------------

# The four variants that flipped supported -> unsupported on the recorded
# release-1 traffic, each with the registry line the turn actually held.
RECORDED_ENTITIES = {
    "id-fp173-entity": (
        "accredited entity: Inter-American Development Bank; "
        "countries: Brazil, Colombia, Ecuador",
        "**Inter-American Development Bank (IDB)**"),
    "cid-fp0086-padded": (
        "accredited entity: European Bank for Reconstruction and Development; "
        "countries: Moldova, Tajikistan",
        "**European Bank for Reconstruction and Development (EBRD)**"),
    "bc-b30-03-add04": (
        "accredited entity: Alternative Energy Promotion Centre, Ministry of "
        "Energy, Water Resources and Irrigation, Government of Nepal.",
        "**Alternative Energy Promotion Centre (AEPC)**"),
    "disc-subnational-pair": (
        "Funding proposal submitted by International Union for Conservation "
        "of Nature and Natural Resources (IUCN) for the TA Facility",
        "**Funding proposal submitted by the International Union for "
        "Conservation of Nature (IUCN)**"),
}


@pytest.mark.parametrize("case", sorted(RECORDED_ENTITIES))
def test_recorded_entity_rows_verify_against_the_evidence_they_cite(
        monkeypatch, case):
    """The corpus prints an accredited entity's full name and rarely the
    parenthetical an answer adds. Requiring the printed PAIR took four correct
    recorded answers to unsupported and id-fp173-entity to ABSTAIN."""
    _no_registry(monkeypatch)
    line, named = RECORDED_ENTITIES[case]
    ev = {(DOC, None): line}
    (v,) = V.classify(V.extract_claims(f"The entity is {named} [{DOC}, cover pages]."),
                      ev, use_llm=False)
    assert v.status == V.SUPPORTED, f"{case}: {v.reason}"


def test_the_anti_gutting_gate_can_still_run_on_a_correct_entity_answer(monkeypatch):
    """_supported_required dropping to 0 is not just a wrong number: it is the
    precondition of the gate that stops a bare refusal replacing a mostly
    correct answer, so the gate silently stopped running."""
    _no_registry(monkeypatch)
    ev = {(DOC, None): "accredited entity: Inter-American Development Bank"}
    answer = (f"FP173 is implemented by the **Inter-American Development Bank "
              f"(IDB)** [{DOC}, cover pages].")
    verdicts = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert V._supported_required(verdicts) == 1
    assert V._status_for(verdicts, True, False) == "verified"


@pytest.mark.parametrize("variants,text,present", [
    # the long form is printed; the parenthetical is the answer's own shorthand
    (["Inter-American Development Bank (IDB)", "Inter-American Development Bank",
      "IDB"], "accredited entity: Inter-American Development Bank", True),
    # ADVERSARIAL: the acronym alone can never carry an invented expansion
    (["Invented Universal Climate Network (IUCN)",
      "Invented Universal Climate Network", "IUCN"],
     "Accredited entity: IUCN", False),
    # the acronym alone IS enough when the source itself attests the alias
    (["Funding proposal submitted by the International Union for Conservation "
      "of Nature (IUCN)",
      "Funding proposal submitted by the International Union for Conservation "
      "of Nature", "IUCN"],
     "submitted by International Union for Conservation of Nature and Natural "
     "Resources (IUCN)", True),
    # ADVERSARIAL: this source says IDB is a DIFFERENT bank. The pairing is
    # wrong, and the other bank being named on the page does not make it right.
    (["Inter-American Development Bank (IDB)", "Inter-American Development Bank",
      "IDB"],
     "Islamic Development Bank (IDB) is the accredited entity. The "
     "Inter-American Development Bank co-finances the programme.", False),
    # ...while the source's OWN pairing still verifies against that same text
    (["Islamic Development Bank (IDB)", "Islamic Development Bank", "IDB"],
     "Islamic Development Bank (IDB) is the accredited entity. The "
     "Inter-American Development Bank co-finances the programme.", True),
    # ADVERSARIAL: initials are never derived from a full name
    (["IFC"], "Accredited entity: Imaginary Finance Corporation", False),
])
def test_entity_present_pairs(variants, text, present):
    assert V._entity_present(variants, text) is present


@pytest.mark.parametrize("a,b,same", [
    ("Funding proposal submitted by the International Union for Conservation "
     "of Nature",
     "International Union for Conservation of Nature and Natural Resources", True),
    ("Green Climate Fund", "Green Climate Fund", True),
    ("Inter-American Development Bank", "Islamic Development Bank", False),
    ("Invented Universal Climate Network",
     "International Union for Conservation of Nature and Natural Resources", False),
    ("World Bank", "Inter-American Development Bank", False),
    ("", "Inter-American Development Bank", False),
])
def test_same_entity_pairs(a, b, same):
    assert V._same_entity(a, b) is same


def test_explicit_aliases_are_read_never_manufactured():
    """A one-word long form is not an alias authority: 'Profonanpe (PRO)' does
    not license PRO for anything, and accepting it is the mutation that makes
    the whole alias rule vacuous."""
    assert V._explicit_aliases("Profonanpe (PRO) manages the fund") == []
    assert V._explicit_aliases(
        "the International Union for Conservation of Nature (IUCN) is the AE"
    ) == [("international union for conservation of nature", "iucn")]


@pytest.mark.parametrize("hay,needle,want", [
    # ADVERSARIAL: two different ministries
    ("The Ministry of Environmental Protection", "Ministry of Environment", False),
    ("the Ministry of Environment of Peru", "Ministry of Environment", True),
    # the corpus prints acronyms possessively and slash-joined far more often
    # than bare, and norm_text keeps both marks
    ("PROFONANPE.", "PROFONANPE", True),
    ("the PROFONANPE's board met", "PROFONANPE", True),
    ("an IUCN/GCF joint programme", "IUCN", True),
    ("an IUCN/GCF joint programme", "GCF", True),
    ("PROFONANPEX", "PROFONANPE", False),
    ("XPROFONANPE", "PROFONANPE", False),
])
def test_contains_norm_word_boundaries(hay, needle, want):
    assert V._contains_norm(V.norm_text(hay), V.norm_text(needle)) is want


def test_a_longer_ministry_name_does_not_verify_a_shorter_one(monkeypatch):
    _no_registry(monkeypatch)
    ev = {(DOC, 5): "The executing entity is the Ministry of Environmental Protection."}
    answer = f"The executing entity is the **Ministry of Environment** [{DOC}, p. 5]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.UNSUPPORTED


def test_a_possessive_occurrence_in_the_page_supports_the_name(monkeypatch):
    _no_registry(monkeypatch)
    ev = {(DOC, 5): "PROFONANPE's board approved the disbursement."}
    answer = f"The executing entity is **PROFONANPE** [{DOC}, p. 5]."
    (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
    assert v.status == V.SUPPORTED


# --- verdict §2 #5 — groundedness over all held evidence ------------------

def test_groundedness_is_measured_over_all_held_evidence_not_the_citation(evidence):
    """The flag was set only on the uncited branch, so for a CITED claim
    groundedness collapsed onto citation support — precisely the miscitation
    case the two metrics exist to tell apart. USD 150 million is printed on
    DOC2 p.5; this claim cites DOC p.45."""
    answer = f"The total GCF funding requested is **USD 150 million** [{DOC}, p. 45]."
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.status == V.UNSUPPORTED
    assert V.GROUNDED_FLAG in v.flags


def test_deleting_a_citation_cannot_raise_groundedness(evidence):
    """The metric was non-monotone: the same sentence counted as grounded only
    once its citation was removed."""
    cited = f"The total GCF funding requested is **USD 150 million** [{DOC}, p. 45]."
    uncited = "The total GCF funding requested is **USD 150 million**."
    (a,) = V.classify(V.extract_claims(cited), evidence, use_llm=False)
    (b,) = V.classify(V.extract_claims(uncited), evidence, use_llm=False)
    assert V.GROUNDED_FLAG in b.flags
    assert V.GROUNDED_FLAG in a.flags


def test_a_claim_no_held_evidence_states_is_not_grounded(evidence):
    """The negative direction: the rescoping changes which evidence a claim is
    read against, never what counts as a match."""
    answer = f"FP151 requests **USD 61 million** in GCF funding [{DOC}, p. 45]."
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert v.failed
    assert V.GROUNDED_FLAG not in v.flags


@pytest.mark.parametrize("answer", [
    "The total GCF funding requested is **USD 150 million**.",
    f"The total GCF funding requested is **USD 150 million** [{DOC}, p. 45].",
])
def test_the_groundedness_flag_never_changes_a_status(answer, evidence):
    (v,) = V.classify(V.extract_claims(answer), evidence, use_llm=False)
    assert V.GROUNDED_FLAG in v.flags
    assert v.status == V.UNSUPPORTED


def test_the_groundedness_flag_is_not_a_user_facing_caution(evidence):
    """It now rides on nearly every correct claim, and warning the reader
    about all of them is the same as warning them about none."""
    ok = f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, p. 45]."
    (v,) = V.classify(V.extract_claims(ok), evidence, use_llm=False)
    assert v.status == V.SUPPORTED and V.GROUNDED_FLAG in v.flags
    assert V.RepairResult(ok, "verified", [v], ok).cautions == []
    # a flag that really is a caution still reaches the reader
    mismatch = f"Total financing is **28 M USD** [{DOC}, p. 45]."
    (w,) = V.classify(V.extract_claims(mismatch), evidence, use_llm=False)
    assert "citation-page-mismatch" in w.flags
    assert V.RepairResult(mismatch, "verified", [w], mismatch).cautions == [w]


def test_groundedness_carries_one_name_only():
    """Two flags under one condition are one fact with two spellings; the
    moment one is updated they disagree."""
    assert "grounded-without-citation" not in inspect.getsource(V)


# --- verdict finding 7 — the judge promoting an UNCITED claim -------------

def test_a_judge_promotion_of_an_uncited_claim_is_recorded(evidence):
    """The judge only promotes — it is never asked about a supported claim —
    so a ruling on a claim citing nothing turns 'the reader was given no
    pointer' into SUPPORTED, which the legacy support rate then absorbs under
    the same key as a properly cited claim. It has to be visible on the
    verdict, not inferable from a mode field two files away."""
    answer = "The total GCF funding requested is USD 150 million."
    client = FakeClient(json.dumps({"verdicts": [
        {"id": 0, "status": "supported", "reason": "p.5 states it"}]}))
    (v,) = V.classify(V.extract_claims(answer), evidence, client=client)
    assert v.status == V.SUPPORTED and v.source == "llm"
    assert v.claim.cited is False
    assert V.JUDGE_UNCITED_FLAG in v.flags


def test_a_judge_promotion_of_a_cited_claim_carries_no_uncited_flag(evidence):
    answer = f"The accredited entity is **Pegasus Capital Advisors LP** [{DOC}, cover pages]."
    client = FakeClient(json.dumps({"verdicts": [
        {"id": 0, "status": "supported", "reason": "the cover line names it"}]}))
    (v,) = V.classify(V.extract_claims(answer), evidence, client=client)
    assert v.status == V.SUPPORTED and v.source == "llm"
    assert v.claim.cited is True
    assert V.JUDGE_UNCITED_FLAG not in v.flags


# --- ordered fix 44 — the gate probe set, in the suite --------------------

def test_no_probe_shape_hides_the_fp172_conflict(monkeypatch):
    """The gate probe set, as one table: every way of dressing the same
    disagreeing sentence must still read CONTRADICTED. Zero violations is the
    precondition for flipping VERIFY_REPAIR.

    The citation-shape rows must land on CONTRADICTED, naming the counter
    figure. The two-field rows also state 49,151,817, which the cited page
    does not print, so they fail as UNSUPPORTED instead — a different verdict,
    the same thing the repair gate reads. Neither may reach SUPPORTED.
    """
    ev = _fp172_conflict(monkeypatch)
    body = "**USD 21,128,224** — A.8 Total GCF funding requested"
    probes = {
        "single-document bracket":
            (f"- {body} [{FP172}, p. 6].", V.CONTRADICTED),
        "chained bracket":
            (f"- {body} [{FP172}, p. 6; {DOC}, p. 5].", V.CONTRADICTED),
        "chained bracket, other order":
            (f"- {body} [{DOC}, p. 5; {FP172}, p. 6].", V.CONTRADICTED),
        "inherited block citation":
            (f"- {body}\n[{FP172}, p. 6; {DOC}, p. 5]", V.CONTRADICTED),
        "two field labels":
            (f"- {body}, out of a total financing of USD 49,151,817 "
             f"[{FP172}, p. 6].", V.UNSUPPORTED),
        "both at once":
            (f"- {body}, out of a total financing of USD 49,151,817 "
             f"[{FP172}, p. 6; {DOC}, p. 5].", V.UNSUPPORTED),
    }
    violations = []
    for name, (answer, want) in probes.items():
        (v,) = V.classify(V.extract_claims(answer), ev, use_llm=False)
        if not v.failed or v.status != want:
            violations.append(f"{name}: {v.status} ({v.reason[:70]})")
    assert violations == []
