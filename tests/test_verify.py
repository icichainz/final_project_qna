"""Claim verification and repair (plan step 5).

No test here touches the network: the LLM layer is exercised through fake
clients, and the degradation tests remove OPENAI_API_KEY outright. The
fixture evidence set is deliberately tiny — four keys — so every verdict in
this file can be checked by reading the fixture.
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
# the repair gates of 817abdb, re-proved against the calibrated matcher
# ---------------------------------------------------------------------------

def test_repair_gates_hold_against_the_report_both_relaxation(monkeypatch):
    """A repair that reports both sides of a known conflict is adopted; one
    that reports a figure no evidence prints is not — even though the answer
    'reports both'."""
    monkeypatch.setattr("gcf_qna.rag.registry.facts", lambda doc: CONFLICT_FACTS)
    answer = (f"FP274's **GCF funding requested** is **USD 40,511,264** "
              f"[{DOC}, cover pages].")
    assert V.verify_answer(answer, CONFLICT_EV, use_llm=False,
                           allow_repair=False).failures        # the premise

    good = V.verify_answer(answer, CONFLICT_EV,
                           client=FakeClient(_both_sides()), use_llm=False)
    assert good.repaired and not good.failures

    bad = V.verify_answer(answer, CONFLICT_EV, client=FakeClient(
        f"- **USD 40,511,264** [{DOC}, cover pages].\n"
        f"- **USD 77,777,777** [{DOC}, p. 8]."), use_llm=False)
    assert bad.repair_rejected and bad.answer == answer
    assert "still fail verification" in " ".join(bad.notes)


def test_repair_may_not_introduce_a_source_via_a_doc_level_bracket(no_registry):
    """Ruling 5 widened what a doc-level bracket RESOLVES to; it did not widen
    what a repair may cite."""
    ev = {(DOC, None): "Registry — FP151: GCF funding requested: 18.5 M USD",
          (DOC, 45): "| (vi) Grants | 18,500,000 |"}
    answer = f"FP151 requests **USD 58 million** [{DOC}, cover pages]."
    res = V.verify_answer(answer, ev, client=FakeClient(
        "FP151 requests **USD 18.5 million** [999_gcf-b99-99-add99]."),
        use_llm=False)
    assert res.repair_rejected and res.answer == answer
    assert any("999_gcf-b99-99-add99" in n for n in res.notes)


def test_repair_anti_gutting_still_holds(no_registry):
    ev = {(DOC, None): "Registry — FP151: GCF funding requested: 18.5 M USD",
          (DOC, 45): "| (vi) Grants | 18,500,000 |"}
    answer = (f"FP151 requests **USD 18.5 million** [{DOC}, p. 45].\n\n"
              f"FP151 also lists **USD 61 million** somewhere [{DOC}, p. 45].")
    res = V.verify_answer(answer, ev, client=FakeClient(
        "The retrieved excerpts do not state FP151's funding."), use_llm=False)
    assert res.repair_rejected and res.answer == answer
    assert any("removed every supported factual claim" in n for n in res.notes)


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
    rows, problems = sv.build_rows(
        sv.read_jsonl(GOLD / "release_release-1-adjudicated.jsonl"),
        sv.read_jsonl(GOLD / "release_release-1-evidence.jsonl"),
        sv.read_jsonl(GOLD / "release_release-1.jsonl"))
    return sv, sv.score(rows), problems


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


def test_no_adjudicated_true_failure_stops_being_flagged():
    """The regression gate. Twelve rows were adjudicated 'the verifier was
    right'; if calibration silences one of them, the calibration is wrong."""
    _sv, res, _p = _scored()
    missed = [r["row_id"] for r in res["rows"]
              if r["arm"] == "gold" and r["should_flag"] and not r["flagged"]]
    assert missed == []


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
    mutated until it is FALSE about the evidence its own turn held."""
    _sv, res, _p = _scored()
    fab = res["arms"]["fabricated"]
    assert res["arm_sizes"]["fabricated"] >= 40
    assert fab["counts"]["fp"] == 0
    assert fab["counts"]["fn"] <= 37, fab["false_negatives"]


def test_the_seed_set_does_not_depend_on_the_verifier_it_scores():
    """A seed set drawn from 'whatever the current code supports' hands a
    permissive verifier a smaller and easier test."""
    sv, res, _p = _scored()
    src = SCORER.read_text()
    body = src[src.index("def fabricate("):src.index("# ---", src.index("def fabricate("))]
    code = "\n".join(ln for ln in body.splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#"))
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    assert "classify" not in code and "verdict" not in code and ".status" not in code
    for row in res["rows"]:
        if row["arm"] == "fabricated":
            assert row["should_flag"] is True


def test_every_scored_row_joins_exactly_one_extracted_claim():
    _sv, res, problems = _scored()
    assert res["unmatched"] == [] and problems == []
