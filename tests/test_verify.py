"""Claim verification and repair (plan step 5).

No test here touches the network: the LLM layer is exercised through fake
clients, and the degradation tests remove OPENAI_API_KEY outright. The
fixture evidence set is deliberately tiny — four keys — so every verdict in
this file can be checked by reading the fixture.
"""
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
    answer = ("The total GCF funding requested is USD 150 million.\n"
              f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, p. 45].")
    claims = V.extract_claims(answer)
    client = FakeClient(json.dumps({"verdicts": [
        {"id": 0, "status": "supported", "reason": "page 5 states it"}]}))
    verdicts = V.classify(claims, evidence, client=client)
    assert len(client.calls) == 1                     # ONE batched call
    assert verdicts[0].status == V.SUPPORTED and verdicts[0].source == "llm"
    assert verdicts[1].status == V.SUPPORTED and verdicts[1].source == "deterministic"


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
    assert len(res.unsupported) == 1
    assert res.counts()[V.SUPPORTED] == 1


def test_at_most_two_llm_calls_per_answer(evidence):
    answer = ("The total GCF funding requested is USD 150 million.\n"
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
