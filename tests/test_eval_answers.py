"""Self-tests for the answer-level evaluation harness (scripts/eval_answers.py).

These test the HARNESS, not the RAG system: fixture parsing and validation,
the scoring path driven by a fake retriever, and the matchers. No API calls,
no FAISS index — everything here runs in milliseconds.
"""
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_answers as ev  # noqa: E402
from gcf_qna.rag import registry  # noqa: E402
from gcf_qna.rag.retrieve import Hit  # noqa: E402

FIXTURE = ROOT / "scripts" / "answer_gold.jsonl"


# ---------------------------------------------------------------- fixtures --
@pytest.fixture(scope="module")
def cases():
    return ev.load_cases(FIXTURE)


def _write(tmp_path, *rows):
    p = tmp_path / "cases.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


def _case(**kw):
    row = {"id": "x", "class": "identifier", "lang": "en", "question": "q?",
           "expect": {"behavior": "answer", "docs": [], "pages": [],
                      "must_contain": [], "must_not_contain": [],
                      "fields": [], "notes": ""}}
    row.update(kw)
    return row


# ------------------------------------------------------------ fixture file --
def test_fixture_parses_and_is_well_formed(cases):
    assert 50 <= len(cases) <= 80, "the plan calls for 50-80 answer cases"
    assert len({c["id"] for c in cases}) == len(cases)
    for c in cases:
        assert c["expect"]["behavior"] in {"answer", "conflict", "abstain"}
        assert c["lang"] in ("en", "fr")
        assert c["question"].strip()
        assert all(isinstance(p, int) for p in c["expect"]["pages"])


def test_fixture_covers_every_planned_class(cases):
    got = {c["class"] for c in cases}
    assert got == set(ev.CLASS_ORDER), got.symmetric_difference(set(ev.CLASS_ORDER))


def test_fixture_labelled_docs_exist_in_the_registry(cases):
    docs = json.loads((ROOT / "data" / "registry.json").read_text(encoding="utf-8"))
    known = set(docs["documents"])
    for c in cases:
        for d in c["expect"]["docs"]:
            assert d in known, f"{c['id']}: unknown document stem {d!r}"


def test_fixture_has_the_named_followup_regression(cases):
    c = next(c for c in cases if c["id"] == "fu-fp220-after-unrelated")
    prior = " ".join(t["content"] for t in c["turns"])
    assert "FP254" in prior and "FP248" in prior
    assert "FP220" in c["question"]


def test_abstain_cases_label_no_documents(cases):
    for c in cases:
        if c["expect"]["behavior"] == "abstain":
            assert not c["expect"]["docs"], c["id"]


# ------------------------------------------------------------- validation ---
def test_load_cases_rejects_missing_key(tmp_path):
    row = _case()
    del row["question"]
    with pytest.raises(ValueError, match="missing 'question'"):
        ev.load_cases(_write(tmp_path, row))


def test_load_cases_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate id"):
        ev.load_cases(_write(tmp_path, _case(), _case()))


def test_load_cases_rejects_unknown_behavior(tmp_path):
    row = _case()
    row["expect"]["behavior"] = "shrug"
    with pytest.raises(ValueError, match="behavior"):
        ev.load_cases(_write(tmp_path, row))


def test_load_cases_rejects_bad_language(tmp_path):
    with pytest.raises(ValueError, match="lang"):
        ev.load_cases(_write(tmp_path, _case(lang="de")))


def test_load_cases_fills_defaults_and_coerces_pages(tmp_path):
    row = {"id": "y", "class": "identifier", "lang": "en", "question": "q?",
           "expect": {"pages": ["8", 40]}}
    got = ev.load_cases(_write(tmp_path, row))[0]
    assert got["expect"]["behavior"] == "answer"
    assert got["expect"]["pages"] == [8, 40]
    assert got["expect"]["must_contain"] == [] and got["turns"] == []


# ---------------------------------------------------------------- matchers --
@pytest.mark.parametrize("pattern,text,want", [
    ("IFAD", "implemented by ifad", True),                     # case-insensitive
    ("IFAD", "implemented by IFC", False),
    (r"re:49[,.\s]?751[,.\s]?264", "USD 49 751 264", True),    # FR digit spacing
    (r"re:49[,.\s]?751[,.\s]?264", "USD 49,751,264", True),
    (r"re:^never", "never at the start", True),
    (r"re:EUR|euro", "the amount is in Euros", True),
    (r"re:\bIFC\b", "IFAD is the entity", False),
])
def test_matches(pattern, text, want):
    assert ev.matches(pattern, text) is want


def test_matches_handles_empty_answer():
    assert ev.matches("anything", "") is False
    assert ev.matches("re:.+", None) is False


def test_regex_spans_newlines():
    assert ev.matches(r"re:18\.5.*USD", "18.5 M\nUSD grant")


# ---------------------------------------------------------------- behavior --
def test_abstention_is_detected_in_the_head_only():
    assert ev.looks_abstained("FP999 does not exist in this corpus.")
    assert ev.looks_abstained("FP999 n'existe pas dans le corpus.")
    tail = ("FP220 requests 50,000,000 USD from the GCF [55_gcf-b37, p. 5]. " * 6
            + "The excerpts do not contain the disbursement schedule.")
    assert not ev.looks_abstained(tail), "a late caveat is not an abstention"


@pytest.mark.parametrize("answer,want", [
    # a bare 'aucun' is ordinary French, not a refusal
    ("Il n'y a aucun doute : le FP151 demande 18,5 millions USD.", False),
    ("Aucun doute possible, l'entité accréditée est l'IUCN.", False),
    # phrase-bound refusals still register
    ("Aucune proposition ne correspond à cette description.", True),
    ("Aucun document du corpus ne mentionne ce projet.", True),
    ("Aucune information sur ce point dans les extraits.", True),
    ("Le FP999 n'existe pas dans ce corpus.", True),
    ("Je n'ai pas trouvé de proposition correspondante.", True),
])
def test_french_abstention_is_phrase_bound(answer, want):
    assert ev.looks_abstained(answer) is want


def test_french_positive_answer_keeps_behavior_answer():
    assert ev.behavior_ok(
        "answer", "Il n'y a aucun doute : le FP151 demande 18,5 millions USD.")


@pytest.mark.parametrize("expected,answer,want", [
    ("abstain", "FP999 does not exist in this corpus.", True),
    ("abstain", "FP999 funds solar in Kenya.", False),
    ("answer", "FP220 is implemented by IFAD.", True),
    ("answer", "The corpus does not contain FP220.", False),
    ("conflict", "p.8 states 49,751,264 while p.40 states 40,751,254 — the "
                 "document is inconsistent.", True),
    ("conflict", "FP274 requests 49,751,264 USD.", False),
])
def test_behavior_ok(expected, answer, want):
    assert ev.behavior_ok(expected, answer) is want


@pytest.mark.parametrize("answer,want", [
    # keyword without a conflict SHAPE: an ordinary comparison
    ("FP151 and FP152 have different accredited entities.", False),
    ("The two proposals differ in scope and instrument.", False),
    ("FP151 (IUCN) [124_gcf-b27-02-add11, p. 5] and FP152 (Pegasus) "
     "[123_gcf-b27-02-add12, p. 5] have different accredited entities.", False),
    # keyword + two values in one sentence
    ("The document is inconsistent: 49,751,264 in section A.10 versus "
     "40,751,254 in the funding table.", True),
    # keyword + two different pages close together
    ("The figures disagree between p. 8 and p. 40.", True),
    # a value pair with no keyword is not a conflict claim either
    ("FP274 requests 49,751,264 USD of 100,194,751 USD total financing.", False),
])
def test_conflict_needs_keyword_and_shape(answer, want):
    assert ev.looks_conflicted(answer) is want


def test_conflict_shape_helpers():
    assert ev._two_values_in_a_sentence("49,751,264 versus 40,751,254")
    assert not ev._two_values_in_a_sentence("only 49,751,264 here")
    assert ev._two_pages_close_together("between p. 8 and p. 40")
    assert not ev._two_pages_close_together("[a, p. 5] and [b, p. 5]"), \
        "the same page twice is not a page-vs-page contradiction"
    far = "p. 8 " + "x" * 200 + " p. 40"
    assert not ev._two_pages_close_together(far)


def test_language_ok_uses_the_app_heuristic():
    assert ev.language_ok("fr", "Le financement du GCF est de 18,5 millions USD.")
    assert not ev.language_ok("fr", "The GCF financing is 18.5 million USD.")
    assert ev.language_ok("en", "The GCF financing is 18.5 million USD.")


# --------------------------------------------------------------- doc match --
@pytest.mark.parametrize("hit,expected,want", [
    ("124_gcf-b27-02-add11", "124_gcf-b27-02-add11", True),
    ("04_gcf-b42-02-add14-funding-proposal-package-fp272_0",
     "04_gcf-b42-02-add14-funding-proposal-package-fp272", True),   # index shard
    ("123_gcf-b27-02-add12", "124_gcf-b27-02-add11", False),
    ("", "124_gcf-b27-02-add11", False),
])
def test_doc_eq(hit, expected, want):
    assert ev.doc_eq(hit, expected) is want


def test_multi_identifier_proxy():
    assert ev.multi_identifier("Compare FP151 and FP152.")
    assert ev.multi_identifier("GCF/B.42/02/Add.16 versus GCF/B.37/02/Add.11?")
    assert not ev.multi_identifier("What is the GCF funding for FP151?")
    assert not ev.multi_identifier("Which proposals were approved in 2020?")


@pytest.mark.parametrize("text,want", [
    ("What is FP86 about?", {"86"}),
    ("Give me the details of FP0086.", {"86"}),          # zero-padded
    ("FP-220 accredited entity??", {"220"}),             # hyphenated
    ("What is FP 86 about?", {"86"}),                    # spaced
    ("FP86 and FP086 and FP-86", {"86"}),                # one identifier
    ("Compare FP151 and FP152.", {"151", "152"}),
    ("Which proposals were approved in 2020?", set()),
    ("fp2023 is a year, not a proposal", set()),         # must not match
])
def test_fp_ids_normalizes_every_id_form(text, want):
    assert ev.fp_ids(text) == want


def test_fp_pattern_is_borrowed_from_the_registry_not_copied():
    """A stale private copy is exactly how this drifted before."""
    assert ev._FP_TOKEN_RE is getattr(registry, "_FP_RE")


# ------------------------------------------------------- fake-retriever run --
class FakeRetriever:
    """Canned hits keyed by query substring — the scoring path without FAISS."""

    def __init__(self, table):
        self.table = table
        self.hybrid_enabled = True

    def search(self, query, top_k=10, doc_filter=None):
        for key, hits in self.table.items():
            if key.lower() in query.lower():
                return hits[:top_k]
        return []

    def search_with_confidence(self, query, top_k=10, doc_filter=None):
        return self.search(query, top_k), 1.0


def _hits(*pairs):
    # pages are ints in the real index (rag.parse.split_pages), matched here
    return [Hit(text=f"excerpt from {d} p{p}", doc_id=d, score=0.9, page=p)
            for d, p in pairs]


def _hits2(doc, page, text):
    """One hit carrying real passage text — what claim support verifies against."""
    return [Hit(text=text, doc_id=doc, score=0.9, page=page)]


FP151 = "124_gcf-b27-02-add11"
FP152 = "123_gcf-b27-02-add12"


def test_score_retrieval_through_a_fake_retriever():
    retriever = FakeRetriever({
        "FP151": _hits((FP151, 5), (FP151, 45), ("999_other", 1)),
    })
    case = _case(id="r1", question="What is the GCF funding for FP151?",
                 expect={"behavior": "answer", "docs": [FP151], "pages": [5],
                         "must_contain": [], "must_not_contain": [],
                         "fields": [], "notes": ""})
    hits, conf = retriever.search_with_confidence(case["question"], 10)
    r = ev.score_retrieval(case, hits)
    assert r["rank"] == 1 and r["r5"] and r["r10"] and r["cover10"]
    assert r["pages_hit"] == 1 and r["page_rate"] == 1.0
    assert ev.retrieval_score(r) == 1.0


def test_score_retrieval_misses_are_reported():
    retriever = FakeRetriever({"FP151": _hits(("999_other", 1)) * 6})
    case = _case(expect={"behavior": "answer", "docs": [FP151], "pages": [5],
                         "must_contain": [], "must_not_contain": [],
                         "fields": [], "notes": ""})
    r = ev.score_retrieval(case, retriever.search("FP151", 10))
    assert r["rank"] is None and r["r5"] is False and r["cover10"] is False
    assert r["page_rate"] == 0.0
    assert ev.retrieval_score(r) == 0.0


def test_recall_at_5_is_rank_sensitive():
    hits = _hits(*[("999_other", i) for i in range(1, 8)]) + _hits((FP151, 5))
    case = _case(expect={"behavior": "answer", "docs": [FP151], "pages": [],
                         "must_contain": [], "must_not_contain": [],
                         "fields": [], "notes": ""})
    r = ev.score_retrieval(case, hits)
    assert r["rank"] == 8 and r["r5"] is False and r["r10"] is True


def test_coverage_requires_every_expected_document():
    case = _case(expect={"behavior": "answer", "docs": [FP151, FP152], "pages": [],
                         "must_contain": [], "must_not_contain": [],
                         "fields": [], "notes": ""})
    one = ev.score_retrieval(case, _hits((FP151, 5)))
    both = ev.score_retrieval(case, _hits((FP151, 5), (FP152, 5)))
    assert one["r5"] is True and one["cover10"] is False
    assert both["cover10"] is True


def test_page_hits_only_count_inside_the_expected_documents():
    case = _case(expect={"behavior": "answer", "docs": [FP151], "pages": [8, 40],
                         "must_contain": [], "must_not_contain": [],
                         "fields": [], "notes": ""})
    r = ev.score_retrieval(case, _hits((FP151, 8), ("999_other", 40)))
    assert r["pages_hit"] == 1 and r["page_rate"] == 0.5


def test_cases_without_document_labels_are_not_scored_for_recall():
    case = _case(expect={"behavior": "answer", "docs": [], "pages": [],
                         "must_contain": [], "must_not_contain": [],
                         "fields": [], "notes": ""})
    r = ev.score_retrieval(case, _hits(("999_other", 1)))
    assert r["r5"] is None and r["cover10"] is None
    assert ev.retrieval_score(r) == 1.0


# ------------------------------------------------------------ answer score --
def test_score_answer_all_checks_pass():
    case = _case(lang="en",
                 expect={"behavior": "conflict", "docs": [], "pages": [],
                         "must_contain": [r"re:49[,.\s]?751[,.\s]?264", "40,751,254"],
                         "must_not_contain": [r"re:\bIFC\b"],
                         "fields": [], "notes": ""})
    answer = ("FP274 states two different GCF amounts: 49,751,264 on p. 8 and "
              "40,751,254 on p. 40 "
              "[02_gcf-b42-02-add16-funding-proposal-package-fp274, p. 8].")
    hits = _hits(("02_gcf-b42-02-add16-funding-proposal-package-fp274", 8),
                 ("02_gcf-b42-02-add16-funding-proposal-package-fp274", 40))
    ch = ev.score_answer(case, answer, hits)
    assert ch["pass"] and ch["score"] == 1.0
    assert ch["behavior"] and ch["language"] and ch["citations"]


def test_score_answer_partial_credit_is_granular():
    case = _case(expect={"behavior": "conflict", "docs": [], "pages": [],
                         "must_contain": ["49,751,264", "40,751,254"],
                         "must_not_contain": [], "fields": [], "notes": ""})
    ch = ev.score_answer(case, "FP274 requests 49,751,264 USD.", [])
    # behavior fails, one of two must_contain fails, lang + cites pass -> 3/5
    assert ch["pass"] is False
    assert ch["score"] == pytest.approx(3 / 5)


def test_score_answer_flags_forbidden_strings():
    case = _case(expect={"behavior": "answer", "docs": [], "pages": [],
                         "must_contain": [], "must_not_contain": [r"re:\bIFC\b"],
                         "fields": [], "notes": ""})
    ch = ev.score_answer(case, "FP220 is implemented by IFC.", [])
    assert ch["must_not_contain"][r"re:\bIFC\b"] is False
    assert ch["pass"] is False


def test_score_answer_detects_invented_pages():
    case = _case(expect={"behavior": "answer", "docs": [], "pages": [],
                         "must_contain": [], "must_not_contain": [],
                         "fields": [], "notes": ""})
    hits = _hits((FP151, 5))
    ok = ev.score_answer(case, f"18.5 M USD [{FP151}, p. 5].", hits)
    bad = ev.score_answer(case, f"18.5 M USD [{FP151}, p. 35].", hits)
    assert ok["citations"] is True
    assert bad["citations"] is False and bad["bad_citations"]


def test_score_answer_language_failure_for_french_case():
    case = _case(lang="fr",
                 expect={"behavior": "answer", "docs": [], "pages": [],
                         "must_contain": [], "must_not_contain": [],
                         "fields": [], "notes": ""})
    assert ev.score_answer(case, "The GCF financing is 18.5 million USD.", [])["language"] is False
    assert ev.score_answer(case, "Le financement du GCF est de 18,5 M USD.", [])["language"] is True


# --------------------------------------------------------------- selection --
def test_select_by_ids(cases):
    got = ev.select(cases, ids="conf-fp274-gcf,abs-fp999")
    assert [c["id"] for c in got] == ["conf-fp274-gcf", "abs-fp999"]


def test_select_by_ids_rejects_unknown(cases):
    with pytest.raises(SystemExit):
        ev.select(cases, ids="nope")


def test_sample_is_stratified_and_deterministic(cases):
    got = ev.select(cases, sample=12)
    assert len(got) == 12
    classes = {c["class"] for c in got}
    for required in ("conflict", "abstain", "french"):
        assert required in classes, f"a 12-case sample must include a {required} case"
    assert [c["id"] for c in got] == [c["id"] for c in ev.select(cases, sample=12)]


def test_sample_larger_than_corpus_returns_everything(cases):
    assert len(ev.select(cases, sample=10_000)) == len(cases)


def test_select_by_class(cases):
    got = ev.select(cases, classes="conflict")
    assert got and {c["class"] for c in got} == {"conflict"}


# ------------------------------------------------------------ guard + I/O ---
def test_fp_guard_mirrors_the_app_short_circuit():
    from gcf_qna.app import chainlit_app as app
    fake = types.SimpleNamespace(app=app)
    assert "FP999" in ev.Pipeline.fp_guard(fake, "What does FP999 fund?")
    assert "n'existe pas" in ev.Pipeline.fp_guard(fake, "Que finance le FP999 ?")
    assert ev.Pipeline.fp_guard(fake, "Which entity implements FP220?") is None


def _plan_pipe(**kw):
    from gcf_qna.app import chainlit_app as app
    return types.SimpleNamespace(app=app, raw_retrieval=False,
                                 scope_single_id=False, **kw)


Q152 = "Which accredited entity implements FP152?"


def test_plan_flows_through_the_app_guards_unchanged_by_default():
    """Production shape since step 1: a cold single-FP question is
    pre-scoped by _rescope_items and resolved to the authoritative stem
    (B.27 filenames carry no FP number, so the registry does the mapping)."""
    items = ev.Pipeline.plan(_plan_pipe(), Q152)
    assert items == [{"q": Q152, "doc": "123_gcf-b27-02-add12"}]


def test_plan_scoped_tag_is_registry_resolved():
    """The tag machinery under test: _rescope_items keeps a tag the message
    itself names, then _resolve_doc_tags maps 'fp152' onto the real stem —
    B.27 filenames carry no FP number, so the plain token matches nothing."""
    pipe = types.SimpleNamespace(app=_plan_pipe().app, raw_retrieval=False,
                                 scope_single_id=True)
    items = ev.Pipeline.plan(pipe, Q152)
    assert items[0]["doc"] == "123_gcf-b27-02-add12"


def test_raw_retrieval_bypasses_the_guards():
    pipe = types.SimpleNamespace(app=_plan_pipe().app, raw_retrieval=True,
                                 scope_single_id=True)
    assert ev.Pipeline.plan(pipe, Q152)[0]["doc"] == "fp152"


def test_tables_render(capsys, cases):
    rows = []
    for cid, cls, score in (("a", "identifier", 1.0), ("b", "conflict", 0.5)):
        rows.append({
            "id": cid, "class": cls, "lang": "en", "guard": False,
            "expect": {"behavior": "conflict"},
            "retrieval": {"docs_expected": 1, "r5": True, "r10": True,
                          "cover10": True, "pages_expected": 1, "page_rate": score,
                          "pages_hit": 1, "rank": 1},
            "score": score,
            "checks": {"behavior": score == 1.0, "must_contain": {"x": True},
                       "must_not_contain": {}, "language": True, "citations": True,
                       "bad_citations": [], "score": score, "pass": score == 1.0},
        })
    ev.print_retrieval_table(rows, skipped=[{"id": "fu-x"}])
    ev.print_answer_table(rows)
    ev._print_failures(rows)
    out = capsys.readouterr().out
    assert "TOTAL" in out and "skipped" in out and "failing cases" in out


def test_record_and_compare_round_trip(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    base = [{"id": "a", "class": "identifier", "score": 0.5},
            {"id": "b", "class": "conflict", "score": 1.0}]
    after = [{"id": "a", "class": "identifier", "score": 1.0},
             {"id": "b", "class": "conflict", "score": 0.5}]
    pa = ev.record(base, "a")
    pb = ev.record(after, "b")
    assert pa.name == "answers_baseline_a.jsonl"
    ev.run_compare(pa, pb)
    out = capsys.readouterr().out
    assert "better" in out and "worse" in out
    assert "better a" in re.sub(r"\s+", " ", out)


# ==========================================================================
# required-field coverage
# ==========================================================================
def _v2(monkeypatch, table, fps=None):
    """Stand in for registry v2: {doc_id: {v2_field: [candidate, ...]}}.

    Stubbed rather than read from data/registry_v2.json so the scorer is
    tested against a fixed corpus of facts, and so a concurrent rebuild of the
    registry cannot turn these assertions red.
    """
    monkeypatch.setattr(ev.registry, "facts", lambda d: table.get(d, {}))
    monkeypatch.setattr(
        ev.registry, "canonical",
        lambda d, f: next((c for c in table.get(d, {}).get(f, [])
                           if c.get("status") == "canonical"), None))
    monkeypatch.setattr(ev, "_doc_fp", lambda d: (fps or {}).get(d))


def _cand(raw, **kw):
    return {"raw": raw, "value": kw.get("value"), "currency": kw.get("currency"),
            "unit": kw.get("unit"), "page": kw.get("page", 5),
            "section": kw.get("section", "A.8"),
            "status": kw.get("status", "canonical")}


def _fields_case(docs, fields, **kw):
    return _case(expect={"behavior": "answer", "docs": docs, "pages": [],
                         "must_contain": [], "must_not_contain": [],
                         "fields": fields, "notes": ""}, **kw)


def test_field_cell_is_stated_when_the_answer_reprints_the_registry_value(monkeypatch):
    _v2(monkeypatch, {FP151: {"gcf_funding_requested": [_cand("18.5 M USD")]}})
    case = _fields_case([FP151], ["gcf_financing"])
    got = ev.score_fields(case, f"FP151 requests USD 18,500,000 [{FP151}, p. 5].")
    assert got["cells"][0]["status"] == "stated"
    assert got["coverage"] == 1.0 and got["n_unscorable"] == 0


@pytest.mark.parametrize("answer,want", [
    ("FP151 requests USD 18.5 million.", "stated"),        # unit word
    ("FP151 requests 18,500,000 USD.", "stated"),          # fully printed
    ("Le FP151 demande 18,5 millions USD.", "stated"),     # French comma decimal
    ("FP151 requests EUR 18.5 million.", "missed"),        # currency is a fact
    ("FP151 requests USD 21.1 million.", "missed"),        # a different figure
])
def test_field_amount_matching_reuses_the_verifier(monkeypatch, answer, want):
    """The matcher must be verify's, not a second implementation: same
    separator rules, same unit words, same cross-currency guard."""
    _v2(monkeypatch, {FP151: {"gcf_funding_requested": [_cand("18.5 M USD")]}})
    case = _fields_case([FP151], ["gcf_financing"])
    assert ev.score_fields(case, answer)["cells"][0]["status"] == want


def test_field_text_match_accepts_the_acronym_and_the_full_name(monkeypatch):
    _v2(monkeypatch, {FP151: {"accredited_entity": [
        _cand("International Union for Conservation of Nature and "
              "Natural Resources (IUCN)")]}})
    case = _fields_case([FP151], ["accredited_entity"])
    for answer in ("The accredited entity is IUCN.",
                   "Implemented by the International Union for Conservation of "
                   "Nature and Natural Resources."):
        assert ev.score_fields(case, answer)["cells"][0]["status"] == "stated"
    assert ev.score_fields(case, "The accredited entity is IFAD.")[
        "cells"][0]["status"] == "missed"


def test_field_list_candidate_matches_any_of_its_items(monkeypatch):
    _v2(monkeypatch, {FP151: {"countries": [_cand("Africa: Angola; Benin; Kenya")]}})
    case = _fields_case([FP151], ["countries"])
    assert ev.score_fields(case, "It covers Kenya and Benin.")[
        "cells"][0]["status"] == "stated"
    assert ev.score_fields(case, "It covers Fiji.")["cells"][0]["status"] == "missed"


@pytest.mark.parametrize("answer", [
    "The GCF amount for FP151 is not stated in the retrieved excerpts.",
    "The excerpts do not state the GCF funding for FP151.",
    "FP151: GCF financing — not specified.",
    "Le montant du financement GCF du FP151 n'est pas précisé dans les extraits.",
    "Les extraits ne mentionnent pas le financement GCF du FP151.",
])
def test_a_field_explicitly_marked_missing_counts_as_covered(monkeypatch, answer):
    _v2(monkeypatch, {FP151: {"gcf_funding_requested": [_cand("18.5 M USD")]}})
    case = _fields_case([FP151], ["gcf_financing"])
    got = ev.score_fields(case, answer)
    assert got["cells"][0]["status"] == "marked-missing"
    assert got["coverage"] == 1.0 and got["n_marked_missing"] == 1


def test_saying_nothing_about_a_field_is_not_marking_it_missing(monkeypatch):
    _v2(monkeypatch, {FP151: {"gcf_funding_requested": [_cand("18.5 M USD")]}})
    case = _fields_case([FP151], ["gcf_financing"])
    got = ev.score_fields(case, "FP151 is a technical assistance facility.")
    assert got["cells"][0]["status"] == "missed" and got["coverage"] == 0.0


def test_a_field_the_registry_never_recorded_is_unscorable_never_a_pass(monkeypatch):
    _v2(monkeypatch, {FP151: {"gcf_funding_requested": [_cand("18.5 M USD")]}})
    case = _fields_case([FP151], ["gcf_financing", "co_financing"])
    got = ev.score_fields(case, f"FP151 requests 18.5 M USD [{FP151}, p. 5].")
    cells = {c["field"]: c["status"] for c in got["cells"]}
    assert cells["co_financing"] == "unscorable"
    assert got["n_scorable"] == 1 and got["n_covered"] == 1
    assert got["coverage"] == 1.0, "an unscorable cell must not dilute coverage"
    assert got["n_unscorable"] == 1

    only = ev.score_fields(_fields_case([FP151], ["co_financing"]), "anything")
    assert only["coverage"] is None and only["n_scorable"] == 0


def test_field_coverage_is_scored_per_document_not_per_answer(monkeypatch):
    """Both proposals request the same amount here: crediting FP152's cell
    from the sentence about FP151 is exactly the failure this scoping
    prevents."""
    _v2(monkeypatch,
        {FP151: {"gcf_funding_requested": [_cand("18.5 M USD")]},
         FP152: {"gcf_funding_requested": [_cand("18.5 M USD")]}},
        fps={FP151: 151, FP152: 152})
    case = _fields_case([FP151, FP152], ["gcf_financing"])
    answer = (f"FP151 requests USD 18.5 million [{FP151}, p. 5].\n\n"
              f"For FP152 the GCF amount is not stated in the excerpts.")
    cells = {c["doc"]: c for c in ev.score_fields(case, answer)["cells"]}
    assert cells[FP151]["status"] == "stated"
    assert cells[FP152]["status"] == "marked-missing"
    assert cells[FP151]["scoped"] == cells[FP152]["scoped"] == "document"


def test_field_scope_falls_back_to_the_whole_answer_when_unattributable(monkeypatch):
    _v2(monkeypatch, {FP151: {"gcf_funding_requested": [_cand("18.5 M USD")]}},
        fps={FP151: 151})
    case = _fields_case([FP151], ["gcf_financing"])
    got = ev.score_fields(case, "The requested amount is USD 18.5 million.")
    assert got["cells"][0]["scoped"] == "answer"
    assert got["cells"][0]["status"] == "stated"


def test_cases_without_a_field_contract_are_not_scored():
    assert ev.score_fields(_fields_case([FP151], []), "anything") is None
    assert ev.score_fields(_fields_case([], ["gcf_financing"]), "anything") is None


def test_field_label_maps_onto_the_registry_v2_field_name():
    assert ev.FIELD_TO_V2["gcf_financing"] == "gcf_funding_requested"
    assert ev.FIELD_TO_V2["accredited_entity"] == "accredited_entity"
    # borrowed from the verifier, not re-declared: one mapping, one owner
    assert all(ev.FIELD_TO_V2[k] == v for k, v in ev._V2_FROM_VERIFY.items())


# ==========================================================================
# claim support
# ==========================================================================
def _verdict(status, text="a claim", kind="money", reason="because",
             citations=None):
    return types.SimpleNamespace(
        status=status, reason=reason, source="deterministic",
        claim=types.SimpleNamespace(text=text, kind=kind,
                                    citations=list(citations or [])))


def test_score_claims_wires_extract_build_and_classify(monkeypatch):
    """The harness must classify against the evidence IT assembled — the same
    hits and notes it put in the prompt — not against a fresh retrieval."""
    seen = {}

    class Stub:
        SUPPORTED, CONTRADICTED, UNSUPPORTED = (
            "supported", "contradicted", "unsupported")

        @staticmethod
        def extract_claims(answer):
            seen["answer"] = answer
            return ["c1", "c2", "c3", "c4"]

        @staticmethod
        def build_evidence(hits, notes):
            seen["evidence_args"] = (hits, notes)
            return {("d1", 5): "text", ("__notes__", None): "note"}

        @staticmethod
        def classify_deterministic(claims, evidence):
            seen["classify_args"] = (claims, evidence)
            return [_verdict("supported", citations=["c"]),
                    _verdict("supported", citations=["c"]),
                    _verdict("contradicted", "bad", reason="p.40 says 40,751,254",
                             citations=["c"]),
                    _verdict("unsupported", "loose")]

        @staticmethod
        def _text_of(evidence, keys):
            seen["text_of_keys"] = list(keys)
            return "\n".join(evidence[k] for k in keys if k in evidence)

        @staticmethod
        def _verify_against(claim, text):
            seen.setdefault("grounded_against", []).append(text)
            return (claim in ("c1", "c2", "c4"), "")

    monkeypatch.setattr(ev, "verify", Stub)
    hits = _hits((FP151, 5))
    got = ev.score_claims("the answer", hits, ["registry note", None, "year note"])

    assert seen["answer"] == "the answer"
    assert seen["evidence_args"][0] is hits
    assert seen["evidence_args"][1] == ["registry note", "year note"], \
        "empty notes are dropped, the rest are passed through in order"
    assert seen["classify_args"][0] == ["c1", "c2", "c3", "c4"]
    assert got["claims"] == 4 and got["supported"] == 2
    assert got["contradicted"] == 1 and got["unsupported"] == 1
    assert got["support_rate"] == 0.5
    assert got["evidence_keys"] == ["d1|5", "__notes__|-"]
    assert [f["status"] for f in got["failures"]] == ["contradicted", "unsupported"]
    assert "40,751,254" in got["failures"][0]["reason"]
    # groundedness is scored over the UNION of the turn's evidence, not the
    # claim's cited scope: every claim sees both blocks in one blob
    assert seen["text_of_keys"] == [("d1", 5), ("__notes__", None)]
    assert set(seen["grounded_against"]) == {"text\nnote"}
    assert got["grounded"] == 3 and got["groundedness_rate"] == 0.75
    # c4 is grounded and still UNSUPPORTED: the evidence carries it, the claim
    # does not point at it. That separation is the whole reason for the split.
    assert got["failures"][1]["grounded"] is True
    assert got["citation_supported"] == 2 and got["citation_completeness_rate"] == 0.5
    assert got["cited"] == 3 and got["citation_presence_rate"] == 0.75
    assert got["n_failures"] == 2


def test_score_claims_on_an_answer_with_no_claims(monkeypatch):
    class Stub:
        SUPPORTED, CONTRADICTED, UNSUPPORTED = (
            "supported", "contradicted", "unsupported")
        extract_claims = staticmethod(lambda a: [])
        build_evidence = staticmethod(lambda h, n: {})
        classify_deterministic = staticmethod(lambda c, e: [])
        _text_of = staticmethod(lambda e, k: "")
        _verify_against = staticmethod(lambda c, t: (False, ""))

    monkeypatch.setattr(ev, "verify", Stub)
    got = ev.score_claims("FP999 does not exist in this corpus.", [], [])
    assert got["claims"] == 0 and got["support_rate"] is None
    assert got["groundedness_rate"] is None
    assert got["citation_completeness_rate"] is None
    assert got["citation_presence_rate"] is None
    assert got["n_failures"] == 0


def test_claim_support_is_not_the_old_page_presence_check():
    """A fabricated figure attached to a page that WAS retrieved passes the
    citation check and must still fail claim support."""
    hits = _hits2(FP151, 5, "A.8 Total GCF funding requested: 18.5 M USD")
    good = ev.score_claims(f"FP151 requests 18.5 million USD [{FP151}, p. 5].",
                           hits, [])
    bad = ev.score_claims(f"FP151 requests 99.9 million USD [{FP151}, p. 5].",
                          hits, [])
    case = _case(expect={"behavior": "answer", "docs": [], "pages": [],
                         "must_contain": [], "must_not_contain": [],
                         "fields": [], "notes": ""})
    assert ev.score_answer(case, f"FP151 requests 99.9 million USD [{FP151}, p. 5].",
                           hits)["citations"] is True
    assert good["supported"] == 1 and good["support_rate"] == 1.0
    assert bad["supported"] == 0 and bad["unsupported"] == 1


def test_claim_support_sees_the_notes_the_prompt_carried():
    """A note-level claim verifies against the computed registry note the
    harness itself prepended — evidence the retrieved passages do not carry.
    Pass the notes and it is supported; drop them and the same sentence cites
    something the turn never held."""
    note = ("Registry — 30 funding-proposal documents from 2020 in the corpus: "
            "FP151 \"TA Facility\"; FP152 \"Equity\"")
    answer = ("The corpus holds 30 funding-proposal documents from 2020 "
              "[registry note in your context].")
    hits = _hits2(FP151, 5, "unrelated passage text")
    with_note = ev.score_claims(answer, hits, [note])
    without = ev.score_claims(answer, hits, [])
    assert with_note["supported"] == 1
    assert "__notes__|-" in with_note["evidence_keys"]
    assert without["supported"] == 0 and without["unsupported"] == 1


# ==========================================================================
# latency / tokens / cost
# ==========================================================================
@pytest.mark.parametrize("values,q,want", [
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.5, 5),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.95, 10),
    ([3.0], 0.95, 3.0),
    ([2.0, 1.0], 0.5, 1.0),
    ([], 0.5, None),
])
def test_percentile_is_nearest_rank(values, q, want):
    assert ev.percentile(values, q) == want


def test_usage_totals_aggregates_only_rows_that_called_the_model():
    rows = [
        {"usage": {"latency_s": 2.0, "prompt_tokens": 1000,
                   "completion_tokens": 100, "total_tokens": 1100}},
        {"usage": {"latency_s": 6.0, "prompt_tokens": 3000,
                   "completion_tokens": 200}},
        {"usage": {}},                       # guard short-circuit: no call
        {"error": "boom"},                   # errored case: no call
    ]
    got = ev.usage_totals(rows)
    assert got["calls"] == 2
    assert got["p50"] == 2.0 and got["p95"] == 6.0 and got["max"] == 6.0
    assert got["prompt_tokens"] == 4000 and got["completion_tokens"] == 300
    assert got["total_tokens"] == 4300


def test_cost_comes_from_the_single_rate_constant():
    rows = [{"usage": {"latency_s": 1.0, "prompt_tokens": 1_000_000,
                       "completion_tokens": 1_000_000}}]
    got = ev.usage_totals(rows)
    want = (1_000_000 * ev.TOKEN_COST_USD["prompt"]
            + 1_000_000 * ev.TOKEN_COST_USD["completion"])
    assert got["cost_usd"] == pytest.approx(round(want, 4))
    assert set(ev.TOKEN_COST_USD) == {"prompt", "completion"}


# ==========================================================================
# release report
# ==========================================================================
def _release_row(cid, cls, *, score=1.0, fields=None, claims=None, usage=None,
                 error=None):
    if error:
        return {"id": cid, "class": cls, "lang": "en", "expect": {"behavior": "answer"},
                "score": 0.0, "error": error, "guard": False}
    return {
        "id": cid, "class": cls, "lang": "en", "guard": False, "score": score,
        "expect": {"behavior": "answer"},
        "retrieval": {"docs_expected": 1, "r5": True, "r10": True, "cover10": True,
                      "pages_expected": 0, "page_rate": None, "pages_hit": None,
                      "rank": 1},
        "checks": {"behavior": True, "must_contain": {"x": score == 1.0},
                   "must_not_contain": {}, "language": True, "citations": True,
                   "bad_citations": [], "score": score, "pass": score == 1.0},
        "fields": fields, "claims": claims,
        "usage": usage or {"latency_s": 3.0, "prompt_tokens": 5000,
                           "completion_tokens": 500, "total_tokens": 5500},
    }


FIELDS_OK = {"cells": [{"doc": FP151, "field": "gcf_financing",
                        "v2_field": "gcf_funding_requested", "scoped": "document",
                        "status": "stated"}],
             "n_cells": 1, "n_scorable": 1, "n_stated": 1, "n_marked_missing": 0,
             "n_missed": 0, "n_unscorable": 0, "n_covered": 1, "coverage": 1.0}
FIELDS_BAD = {"cells": [{"doc": FP152, "field": "gcf_financing",
                         "v2_field": "gcf_funding_requested", "scoped": "answer",
                         "status": "missed", "expected": ["150 M USD"]},
                        {"doc": FP152, "field": "title", "v2_field": "title",
                         "scoped": "answer", "status": "unscorable",
                         "why": "no candidate"}],
              "n_cells": 2, "n_scorable": 1, "n_stated": 0, "n_marked_missing": 0,
              "n_missed": 1, "n_unscorable": 1, "n_covered": 0, "coverage": 0.0}
CLAIMS_OK = {"claims": 4, "supported": 4, "contradicted": 0, "unsupported": 0,
             "support_rate": 1.0, "evidence_keys": ["d|5"], "failures": []}
CLAIMS_BAD = {"claims": 4, "supported": 2, "contradicted": 1, "unsupported": 1,
              "support_rate": 0.5, "evidence_keys": ["d|5"],
              "failures": [{"status": "unsupported", "kind": "money",
                            "text": "t", "reason": "not in the cited evidence"}]}


def test_release_table_renders_every_aggregate(capsys):
    rows = [_release_row("a", "identifier", fields=FIELDS_OK, claims=CLAIMS_OK),
            _release_row("b", "comparison", score=0.5, fields=FIELDS_BAD,
                         claims=CLAIMS_BAD),
            _release_row("c", "abstain", error="RuntimeError: chat call failed")]
    summary = ev.print_release_table(rows, cases_total=3)
    out = capsys.readouterr().out
    assert "RELEASE REPORT" in out and "3 cases" in out and "1 errored" in out
    for section in ("FIELD COVERAGE", "CLAIM SUPPORT", "LATENCY / COST",
                    "ERRORED CASES"):
        assert section in out
    assert "estimated cost" in out and "ESTIMATED rates" in out
    assert "RuntimeError: chat call failed" in out
    assert summary["fields"] == {"cells": 3, "scorable": 2, "stated": 1,
                                 "marked_missing": 0, "missed": 1, "unscorable": 1}
    assert summary["claims"]["total"] == 8 and summary["claims"]["supported"] == 6
    assert summary["claims"]["rate"] == 0.75
    assert summary["usage"]["calls"] == 2          # the errored case made no call
    assert summary["errors"] == ["c"]


def test_release_table_survives_a_scorer_that_failed(capsys):
    """A scorer stub ({'error': ...}) must be skipped, not crash the report."""
    rows = [_release_row("a", "identifier", fields={"error": "boom"},
                         claims={"error": "boom"}),
            _release_row("b", "identifier", fields=FIELDS_OK, claims=CLAIMS_OK)]
    summary = ev.print_release_table(rows)
    assert summary["fields"]["cells"] == 1 and summary["claims"]["total"] == 4
    assert "RELEASE REPORT" in capsys.readouterr().out


def test_release_records_under_its_own_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    assert ev.record([], "release-1", prefix="release_").name == "release_release-1.jsonl"
    assert ev.record([], "x").name == "answers_baseline_x.jsonl"


def test_compare_tolerates_old_records_missing_the_new_keys(tmp_path, monkeypatch,
                                                            capsys):
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    old = [{"id": "a", "class": "identifier", "score": 0.5}]     # pre-release
    new = [_release_row("a", "identifier", fields=FIELDS_OK, claims=CLAIMS_OK)]
    ev.run_compare(ev.record(old, "old"), ev.record(new, "new"))
    out = capsys.readouterr().out
    assert "TOTAL" in out and "better" in out
    assert "field coverage" not in out, \
        "a metric only one side carries cannot be diffed"


def test_compare_diffs_the_new_metrics_when_both_runs_carry_them(capsys, tmp_path,
                                                                 monkeypatch):
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    a = [_release_row("a", "identifier", score=0.5, fields=FIELDS_BAD,
                      claims=CLAIMS_BAD)]
    b = [_release_row("a", "identifier", fields=FIELDS_OK, claims=CLAIMS_OK)]
    ev.run_compare(ev.record(a, "a"), ev.record(b, "b"))
    out = re.sub(r"\s+", " ", capsys.readouterr().out)
    assert "field coverage 1 cases 0.0% -> 100.0%" in out
    assert "claim support 1 cases 50.0% -> 100.0%" in out


# ==========================================================================
# the recording is the measurement — overwrite protection
#
# data/eval/release_release-1.jsonl is the only copy of the 66-answer release
# run every later --compare is measured against. It is not regenerable: a
# re-run calls the model again and is a different sample. These pin that the
# harness cannot destroy one by accident, and that nothing about recording a
# NEW label changed.
# ==========================================================================
ANCHOR = [{"id": "a", "class": "identifier", "score": 1.0}]
LATER = [{"id": "a", "class": "identifier", "score": 0.0}]


def test_record_refuses_to_overwrite_an_existing_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    out = ev.record(ANCHOR, "release-1", prefix="release_")
    before = out.read_text(encoding="utf-8")

    with pytest.raises(SystemExit) as e:
        ev.record(LATER, "release-1", prefix="release_")

    assert "refusing to overwrite" in str(e.value)
    assert str(out) in str(e.value), "the message must name the file it saved"
    assert "--force-record" in str(e.value), "and how to mean it"
    assert out.read_text(encoding="utf-8") == before, "the anchor was clobbered"


def test_force_record_allows_the_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    out = ev.record(ANCHOR, "release-1", prefix="release_")
    again = ev.record(LATER, "release-1", prefix="release_", force=True)
    assert again == out
    assert json.loads(out.read_text(encoding="utf-8").splitlines()[0])["score"] == 0.0


@pytest.mark.parametrize("prefix, name", [
    ("answers_baseline_", "answers_baseline_fresh.jsonl"),
    ("release_", "release_fresh.jsonl"),
])
def test_a_fresh_label_still_records_without_the_flag(tmp_path, monkeypatch,
                                                     prefix, name):
    """Backward compatibility: the guard only ever fires on an existing path."""
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    out = ev.record(ANCHOR, "fresh", prefix=prefix)
    assert out.name == name
    assert json.loads(out.read_text(encoding="utf-8"))["id"] == "a"


def test_record_creates_the_eval_dir_when_it_is_missing(tmp_path, monkeypatch):
    """The existence check must not be what makes a first-ever run fail."""
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path / "eval")
    assert ev.record(ANCHOR, "first").exists()


def test_record_path_agrees_with_where_record_writes(tmp_path, monkeypatch):
    """One target computation, so the pre-flight cannot check a different
    file from the one the run would later write."""
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    assert ev.record_path("x", "release_") == ev.record(ANCHOR, "x", "release_")


def test_cli_refuses_the_clobber_before_spending_the_run(tmp_path, monkeypatch):
    """The refusal has to land BEFORE the model calls: refusing only at write
    time would burn 66 answers and then throw them away."""
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    (tmp_path / "release_release-1.jsonl").write_text("anchor\n", encoding="utf-8")
    ran = []
    monkeypatch.setattr(ev, "run_eval", lambda *a, **kw: ran.append(1) or [])

    with pytest.raises(SystemExit) as e:
        ev.main(["--release", "--record", "release-1"])

    assert "refusing to overwrite" in str(e.value)
    assert not ran, "the run must not start when its recording is doomed"
    assert (tmp_path / "release_release-1.jsonl").read_text(encoding="utf-8") \
        == "anchor\n"


def test_cli_force_record_overwrites(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    target = tmp_path / "release_release-1.jsonl"
    target.write_text("anchor\n", encoding="utf-8")
    monkeypatch.setattr(ev, "run_eval", lambda *a, **kw: list(LATER))

    assert ev.main(["--release", "--record", "release-1", "--force-record"]) == 0
    assert json.loads(target.read_text(encoding="utf-8"))["score"] == 0.0
    assert "recorded ->" in capsys.readouterr().out


def test_cli_records_a_fresh_label_with_no_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    monkeypatch.setattr(ev, "run_eval", lambda *a, **kw: list(ANCHOR))
    assert ev.main(["--answers", "--record", "brand-new"]) == 0
    assert (tmp_path / "answers_baseline_brand-new.jsonl").exists()


# ==========================================================================
# Wave 3 — production parity
#
# Every switch below is pinned in BOTH directions: a flag whose off-path is
# untested is a flag that can be silently wrong in exactly the configuration
# the cheap per-commit run uses. Nothing here calls an API — the model client
# is a fake that records what it was asked for, which is also how the
# zero-API guarantees are tested rather than asserted.
# ==========================================================================
class _Resp:
    """An OpenAI chat response, only the fields the harness reads."""

    def __init__(self, content, model="gpt-5.2-2025-12-11", pt=11, ct=7):
        self.choices = [types.SimpleNamespace(
            message=types.SimpleNamespace(content=content))]
        self.model = model
        self.system_fingerprint = "fp_test"
        self.usage = types.SimpleNamespace(prompt_tokens=pt, completion_tokens=ct,
                                           total_tokens=pt + ct)


class FakeClient:
    """Records every request; replies from a queue, last reply repeating."""

    def __init__(self, replies=("an answer",)):
        self.replies = list(replies)
        self.calls = []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kw):
        self.calls.append(kw)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return _Resp(reply)


def _args(**kw):
    base = dict(answers=True, release=False, history_mode="isolated",
                temperature=0.0, seed=7, k=10, verifier_mode="deterministic",
                production_planner=False, conductor=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


class FakePipe:
    def __init__(self, out, verifier_mode="deterministic"):
        self.out = out
        self.verifier_mode = verifier_mode

    def run(self, question, turns=()):
        self.seen = (question, list(turns))
        return self.out

    def parity(self):
        return {"verifier_mode": self.verifier_mode}


def _answer_out(**kw):
    """A non-guard `Pipeline.run` result, ready for _run_case."""
    system, context, question = "SYSTEM", "Context excerpts:\nx", "q?"
    out = {"guard": False, "chat": False, "guard_answer": None,
           "hits": [], "confidence": 0.9, "weak": False,
           "plan": [{"q": question, "doc": None}], "decomposed": False,
           "system": system, "context": context, "refs_note": None,
           "user": f"Context excerpts:\n{context}\n\nQuestion: {question}",
           "calls": [],
           "messages": [{"role": "system", "content": system},
                        {"role": "user", "content": "ONE USER TURN"}],
           "notes": {"registry": None, "year": None, "board": None,
                     "matrix": None}}
    out.update(kw)
    return out


# ------------------------------------------------------- F10: run pinning ---
def test_every_call_carries_the_pinned_temperature_and_seed():
    c = FakeClient()
    ev.ask_model(c, "sys", (), "user", pins=ev._pinning(0.0, 7))
    assert c.calls[0]["temperature"] == 0.0 and c.calls[0]["seed"] == 7


def test_pinning_can_be_dropped_and_then_nothing_is_sent():
    assert ev._pinning(None, None) == {}
    c = FakeClient()
    ev.ask_model(c, "sys", (), "user", pins=ev._pinning(None, None))
    assert "temperature" not in c.calls[0] and "seed" not in c.calls[0]


def test_ask_model_records_the_served_snapshot_not_the_alias():
    """The record used to carry `gpt-5.2` — the name we ASKED for. A snapshot
    rotation would then move every number with nothing in the file to show it."""
    _, meta = ev.ask_model(FakeClient(), "sys", (), "user")
    assert meta["model"] == ev.config.CHAT_MODEL
    assert meta["snapshot"] == "gpt-5.2-2025-12-11"
    assert meta["system_fingerprint"] == "fp_test"


def test_ask_model_uses_a_supplied_messages_array_verbatim():
    c = FakeClient()
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    ev.ask_model(c, "ignored", [{"role": "user", "content": "history"}],
                 "ignored", messages=msgs)
    assert c.calls[0]["messages"] == msgs


# ------------------------------------------------- run metadata / artifacts --
def test_run_meta_pins_the_verifier_blob_and_the_artifacts(monkeypatch):
    monkeypatch.setattr(ev, "_sha256_file",
                        lambda p, missing=None: f"sha:{Path(p).name}")
    m = ev.run_meta(_args())
    assert m["verify_blob_sha"] == "sha:verify.py"
    assert m["artifacts"]["index_faiss_sha256"] == "sha:index.faiss"
    assert m["artifacts"]["registry_v2_sha256"] == "sha:registry_v2.json"
    assert m["harness"]["seed"] == 7 and m["harness"]["temperature"] == 0.0


def test_sha256_of_a_missing_file_is_the_missing_marker(tmp_path):
    assert ev._sha256_file(tmp_path / "nope.bin", missing="gone") == "gone"
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    assert ev._sha256_file(f) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


# ------------------------------------------- the metric split (definition) ---
def test_the_four_compared_metric_keys_are_distinct():
    assert len(set(ev.METRIC_KEYS)) == len(ev.METRIC_KEYS) == 4


def test_groundedness_separates_a_miscitation_from_a_fabrication():
    """The whole reason for the split. Same evidence, same matcher:

    * the figure IS in the turn's evidence but the answer points at the wrong
      document — grounded, not citation-complete;
    * the figure is nowhere — neither.

    Scoped to the citation alone (verify.py@HEAD `_verify_against`) the two are
    indistinguishable, which is what makes the split carry information.

    Across DOCUMENTS, not across pages: verify's `_scopes` already widens a
    citation to the whole document it names, so a wrong-page citation inside
    the right document verifies at HEAD and is not a miscitation this metric
    should separate. The one it must separate is evidence held under a
    document the answer never cited.
    """
    doc_a, doc_b = "999_doc-a", "998_doc-b"
    hits = [Hit(text="Cover page. Project title and country.",
                doc_id=doc_a, score=0.9, page=5),
            Hit(text="A.8 Total funding requested: 18.5 million USD",
                doc_id=doc_b, score=0.9, page=45)]
    mis = ev.score_claims(
        f"The programme requests 18.5 million USD [{doc_a}, p. 5].", hits, [])
    fab = ev.score_claims(
        f"The programme requests 99.9 million USD [{doc_a}, p. 5].", hits, [])
    assert mis["citation_supported"] == 0 and mis["grounded"] == 1
    assert fab["citation_supported"] == 0 and fab["grounded"] == 0
    assert mis["cited"] == 1 and mis["citation_presence_rate"] == 1.0
    assert mis["groundedness_rate"] == 1.0 and mis["citation_completeness_rate"] == 0.0


def test_the_deterministic_scoped_identity_is_enforced_not_hoped_for(monkeypatch):
    """`supported == citation_supported` holds in deterministic mode because an
    uncited claim is UNSUPPORTED. If a change ever breaks that, the harness
    must stop rather than print two numbers that no longer mean what they say."""
    class Stub:
        SUPPORTED, CONTRADICTED, UNSUPPORTED = (
            "supported", "contradicted", "unsupported")
        extract_claims = staticmethod(lambda a: ["c1"])
        build_evidence = staticmethod(lambda h, n: {})
        classify_deterministic = staticmethod(
            lambda c, e: [_verdict("supported")])       # supported, uncited
        _text_of = staticmethod(lambda e, k: "")
        _verify_against = staticmethod(lambda c, t: (False, ""))

    monkeypatch.setattr(ev, "verify", Stub)
    with pytest.raises(AssertionError, match="scoped identity"):
        ev.score_claims("a", [], [])


# ------------------------------------------------ gap 1: prescope metadata ---
def test_prescope_is_counted_by_observation_not_asserted():
    """The parity block claims production's single-FP pre-scoping runs here.
    The claim is worth nothing unless something counted it, so the app's own
    function is wrapped and the tally is what the record carries."""
    from gcf_qna.app import chainlit_app as app
    ev._instrument_prescope(app)
    before = dict(ev._PRESCOPE_STATS)
    items = app._rescope_items([{"q": Q152, "doc": None}], Q152, [])
    assert ev._PRESCOPE_STATS["calls"] == before["calls"] + 1
    assert ev._PRESCOPE_STATS["tagged"] == before["tagged"] + 1
    assert items[0]["doc"], "the tag the counter says it added"
    assert ev._PRESCOPE_STATS["wrapped"] is True


def test_instrumenting_prescope_twice_does_not_double_count():
    from gcf_qna.app import chainlit_app as app
    ev._instrument_prescope(app)
    ev._instrument_prescope(app)
    before = ev._PRESCOPE_STATS["calls"]
    app._rescope_items([{"q": Q152, "doc": None}], Q152, [])
    assert ev._PRESCOPE_STATS["calls"] == before + 1


_FULL_PARITY = {
    "production_single_id_prescope": True, "comparison_flag": "decomposed",
    "abstain_keeps_original": True, "guard_verification_skipped": True,
    "answer_history_isolation": True, "planner": {"enabled": True},
    "conductor": {"enabled": True}, "verifier_mode": "production",
    "usage_accounts_judge_and_repair": True,
}
_DEPLOYED = {"sha": "abc1234", "switches": {
    "CONDUCTOR": "1", "PLANNER": "1", "VERIFY": "1", "VERIFY_LLM": "1",
    "INDEX_NAME": "default"}}


def _deployed(**over):
    d = {"sha": _DEPLOYED["sha"],
         "switches": dict(_DEPLOYED["switches"], CHAT_MODEL=ev.config.CHAT_MODEL)}
    d["switches"].update(over.pop("switches", {}))
    d.update(over)
    return d


def test_parity_level_names_every_gap_still_open():
    level, _ = ev._parity_level({"planner": {}, "conductor": {}}, _deployed(),
                                "abc1234deadbeef")
    assert level.startswith("partial: ")
    for gap in ("gap1-prescope", "gap2-comparison-flag", "gap3-abstain",
                "gap4-guard", "gap5-history-isolation", "gap6-planner",
                "gap7-conductor", "gap8-verifier-config",
                "gap9-usage-accounting"):
        assert gap in level


def test_parity_level_is_full_only_when_the_deployment_matches():
    assert ev._parity_level(_FULL_PARITY, _deployed(), "abc1234deadbeef")[0] == "full"


def test_a_drifted_switch_blocks_full_and_is_named():
    level, drift = ev._parity_level(_FULL_PARITY,
                                    _deployed(switches={"PLANNER": "0"}),
                                    "abc1234deadbeef")
    assert level == "partial: deployment-drift"
    assert any("PLANNER" in d for d in drift)


def test_an_unknown_deployment_blocks_full():
    level, _ = ev._parity_level(_FULL_PARITY, _deployed(sha=None),
                                "abc1234deadbeef")
    assert level == "partial: unverified-deployment"


def test_a_different_app_source_blocks_full():
    level, drift = ev._parity_level(_FULL_PARITY, _deployed(), "9999999deadbeef")
    assert level == "partial: deployment-drift"
    assert any("app source" in d for d in drift)


def test_a_later_commit_that_did_not_touch_src_is_not_a_drift(monkeypatch):
    """The image is built `COPY src/ src/`. A docs-only commit past the
    deployed sha runs the same application, and calling that a drift would
    make `level=full` unreachable for a reason that is not about the code."""
    monkeypatch.setattr(ev, "app_source_matches", lambda sha: True)
    deployed = _deployed()
    level, drift = ev._parity_level(_FULL_PARITY, deployed, "9999999deadbeef")
    assert level == "full" and drift == []
    assert any("byte-identical" in n for n in deployed.get("notes", []))
    # one `deployed` dict is graded once per case; the note must not pile up
    for _ in range(5):
        ev._parity_level(_FULL_PARITY, deployed, "9999999deadbeef")
    assert sum("byte-identical" in n for n in deployed["notes"]) == 1


def test_the_real_tree_is_checked_against_the_real_deployed_sha():
    """Not a mock: the deployed row in docs/DEPLOYED.md names a sha, and this
    asserts the harness is importing that application's source.

    The tree legitimately moves ahead of production between deploys, and that
    is development, not a regression — so an ahead-of-deploy tree SKIPS with
    the drift named. The negative half still runs either way: a sha that
    matches nothing must never report parity.
    """
    deployed = ev.deployment_fingerprint()
    assert deployed["sha"], "no deployed sha on record"
    assert ev.app_source_matches("0000000") is False
    if ev.app_source_matches(deployed["sha"]) is not True:
        pytest.skip(
            f"tree has moved past deployed sha {deployed['sha']} — a release "
            "run would measure code production is not serving. Deploy, or "
            "record the run as non-parity."
        )


def test_deployment_fingerprint_prefers_the_captured_file(tmp_path):
    f = tmp_path / "fp.txt"
    f.write_text("PLANNER=1\nVERIFY=1\nfp-gcf:abc1234\nlocal-artifacts\n",
                 encoding="utf-8")
    got = ev.deployment_fingerprint(f)
    assert got["sha"] == "abc1234" and got["switches"]["PLANNER"] == "1"
    assert got["remote_artifacts_verified"] is True


def test_deployment_fingerprint_falls_back_to_the_tracked_deploy_log(tmp_path):
    got = ev.deployment_fingerprint(tmp_path / "absent.txt")
    assert got["remote_artifacts_verified"] is False
    assert "DEPLOYED.md" in got["source"]
    assert got["notes"] and "not performed" in got["notes"][0]
    # the switches the .env does not carry come from the code defaults
    assert set(got["switches"]) >= {"CONDUCTOR", "PLANNER", "VERIFY",
                                    "VERIFY_LLM",
                                    "INDEX_NAME", "CHAT_MODEL"}


# ------------------------------------------------- gap 2: comparison flag ---
def _flag_pipe(flag):
    return types.SimpleNamespace(comparison_flag=flag)


def test_comparison_block_follows_the_plan_not_the_question():
    one = [{"q": "x", "doc": None}]
    two = [{"q": "x", "doc": "a"}, {"q": "y", "doc": "b"}]
    q = "Compare FP151 and FP152."
    pipe = _flag_pipe("decomposed")
    assert ev.Pipeline._decomposed(pipe, one, q) is False
    assert ev.Pipeline._decomposed(pipe, two, "anything") is True
    assert ev.Pipeline._decomposed(pipe, one, q, plan=object()) is True


def test_the_retired_proxy_is_still_selectable_and_disagrees():
    one = [{"q": "x", "doc": None}]
    q = "Compare FP151 and FP152."
    assert ev.Pipeline._decomposed(_flag_pipe("proxy"), one, q) is True
    assert ev.Pipeline._decomposed(_flag_pipe("decomposed"), one, q) is False
    assert ev.Pipeline._decomposed(_flag_pipe("off"),
                                   [{"q": "a"}, {"q": "b"}], q) is False


# --------------------------------------------------------- gap 3: abstain ---
def test_abstain_keeps_the_original_body():
    res = types.SimpleNamespace(status="abstain", answer="")
    assert ev.final_answer("ORIGINAL", res) == ("ORIGINAL", "abstain-original")


def test_a_verifier_returning_nothing_falls_back_to_the_original():
    assert ev.final_answer("ORIGINAL", None) == ("ORIGINAL", "model")
    empty = types.SimpleNamespace(status="partial", answer="")
    assert ev.final_answer("ORIGINAL", empty) == ("ORIGINAL", "model")
    none = types.SimpleNamespace(status="partial", answer=None)
    assert ev.final_answer("ORIGINAL", none) == ("ORIGINAL", "model")
    no_answer_attr = types.SimpleNamespace(status="partial")
    assert ev.final_answer("ORIGINAL", no_answer_attr) == ("ORIGINAL", "model")


def test_a_verifier_result_can_never_replace_the_answer():
    """The repair pathway is gone: even a result carrying a different answer
    text must not reach the display (pure detector)."""
    for status in ("verified", "partial", "unverified-llm"):
        res = types.SimpleNamespace(status=status, answer="FIXED")
        assert ev.final_answer("ORIGINAL", res) == ("ORIGINAL", "model")


# ---------------------------------------------------- gap 4: guard answers ---
GUARD_ANSWER = "FP999 does not exist in this corpus (273-document registry)."


def test_guard_answers_are_not_verified_and_their_claims_are_published():
    out = _answer_out(guard=True, guard_answer=GUARD_ANSWER, hits=[],
                      system=None, user=None, messages=None,
                      notes={"registry": "Registry — FP999: NOT FOUND",
                             "year": None, "board": None, "matrix": None})
    rec = ev._run_case(FakePipe(out), None, _args(),
                       _case(id="abs-fp999", turns=[]))
    would = ev.score_claims(GUARD_ANSWER, [], ["Registry — FP999: NOT FOUND"])
    assert rec["claims"] is None, "a guard answer has no verdicts in production"
    assert rec["claims_skipped"]["claims_removed"] == would["claims"]
    assert rec["claims_skipped"]["supported_removed"] == would["supported"]
    assert "guard-answer" in rec["claims_skipped"]["reason"]


def test_a_chat_mode_turn_is_removed_from_the_denominator_too():
    out = _answer_out(chat=True)
    rec = ev._run_case(FakePipe(out), FakeClient(["chatty reply"]), _args(),
                       _case(id="chat", turns=[]))
    assert rec["claims"] is None and "chat-mode" in rec["claims_skipped"]["reason"]


def test_a_normal_turn_keeps_its_claims():
    rec = ev._run_case(FakePipe(_answer_out()), FakeClient(["plain text"]),
                       _args(), _case(turns=[]))
    assert rec["claims_skipped"] is None and isinstance(rec["claims"], dict)


# ------------------------------------------------ gap 5: history isolation ---
_TURNS = [{"role": "user", "content": "EARLIER QUESTION"},
          {"role": "assistant", "content": "EARLIER ANSWER with 58,000,000"}]


def test_history_isolation_sends_system_plus_one_user_turn():
    c = FakeClient()
    ev._run_case(FakePipe(_answer_out()), c, _args(history_mode="isolated"),
                 _case(turns=_TURNS))
    msgs = c.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "EARLIER ANSWER" not in json.dumps(msgs), \
        "a prior answer must never reach the factual call as evidence"


def test_prepend_mode_puts_the_conversation_back():
    c = FakeClient()
    ev._run_case(FakePipe(_answer_out()), c, _args(history_mode="prepend"),
                 _case(turns=_TURNS))
    msgs = c.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "EARLIER ANSWER" in json.dumps(msgs)


def test_the_resolved_refs_note_is_what_carries_the_referents():
    from gcf_qna.app import chainlit_app as app
    note = app._resolved_refs_note([{"q": "x", "doc": "123_gcf-b27-02-add12"}],
                                   "And that one?")
    assert note and "FP152" in note
    out = _answer_out(refs_note=note,
                      messages=app._answer_messages("S", "ctx", "And that one?",
                                                    note))
    c = FakeClient()
    rec = ev._run_case(FakePipe(out), c, _args(), _case(turns=_TURNS))
    assert note in c.calls[0]["messages"][1]["content"]
    assert rec["refs_note"] == note


# --------------------------------------------------------- gap 6: planner ---
def test_the_planner_is_off_unless_it_is_asked_for():
    pipe = types.SimpleNamespace(production_planner=False)
    assert ev.Pipeline.planner_plan(
        pipe, "Compare FP151 and FP152 GCF financing.") == (None, None)


def _planner_pipe(retriever=None):
    from gcf_qna.app import chainlit_app as app
    return types.SimpleNamespace(
        production_planner=True, app=app,
        retriever=retriever if retriever is not None else FakeRetriever({}),
        planner_stats={"detected": 0, "intent_ok": 0, "matrix_built": 0,
                       "matrix_failed": 0})


def test_the_intent_gate_vetoes_two_ids_that_ask_nothing():
    """`detect` fires on ANY message naming two documents. Prose that merely
    mentions two is not a request for a 2xN matrix, and the app's gate — not a
    copy of it — is what decides."""
    pipe = _planner_pipe()
    got = ev.Pipeline.planner_plan(
        pipe, "FP151 is interesting. Separately, FP152 was approved last year.")
    assert got == (None, None)
    assert pipe.planner_stats["detected"] == 1
    assert pipe.planner_stats["intent_ok"] == 0


def test_a_comparison_builds_a_matrix_block():
    pipe = _planner_pipe()
    plan, block = ev.Pipeline.planner_plan(
        pipe, "Compare the GCF financing of FP151 and FP152.")
    assert pipe.planner_stats["intent_ok"] == 1
    if plan is None:                      # no cell carried evidence
        assert pipe.planner_stats["matrix_failed"] == 1
    else:
        assert block and pipe.planner_stats["matrix_built"] == 1
        assert "FP151" in block or "FP152" in block


# ------------------------------------------------------- gap 7: conductor ---
def _cond_pipe(client, conductor=True):
    from gcf_qna.app import chainlit_app as app
    return types.SimpleNamespace(
        app=app, conductor=conductor, client=client, pins={"seed": 7},
        conductor_stats={"calls": 0, "fanned_out": 0, "chat": 0, "failed": 0})


def test_the_conductor_call_is_the_apps_call():
    from gcf_qna.app import chainlit_app as app
    reply = json.dumps({"mode": "retrieve", "queries": [
        {"q": "FP151 GCF financing", "doc": "fp151"},
        {"q": "FP152 GCF financing", "doc": "fp152"}]})
    c = FakeClient([reply])
    pipe = _cond_pipe(c)
    mode, items, meta = ev.Pipeline.conduct(
        pipe, "Compare FP151 and FP152.", ())
    kw = c.calls[0]
    assert kw["messages"][0]["content"] is app.CONDUCTOR_PROMPT
    assert kw["response_format"] == {"type": "json_object"}
    assert kw["max_completion_tokens"] == 300 and kw["seed"] == 7
    assert mode == "retrieve" and len(items) == 2
    assert meta["role"] == "conductor" and meta["snapshot"]
    assert pipe.conductor_stats["calls"] == 1
    assert pipe.conductor_stats["fanned_out"] == 1


def test_the_conductor_sees_the_conversation_the_answer_call_does_not():
    c = FakeClient([json.dumps({"mode": "retrieve", "queries": []})])
    ev.Pipeline.conduct(_cond_pipe(c), "And that one?", _TURNS)
    user = c.calls[0]["messages"][1]["content"]
    assert "EARLIER ANSWER" in user and "And that one?" in user


def test_the_conductor_output_goes_through_the_rewrite_guards():
    """A fabricated tag for a document the message never names is stripped by
    `_rescope_items`; the harness must not adopt the raw model output."""
    reply = json.dumps({"mode": "retrieve",
                        "queries": [{"q": "total financing", "doc": "02_fp999"}]})
    c = FakeClient([reply])
    _, items, _ = ev.Pipeline.conduct(_cond_pipe(c), "What does FP151 request?", ())
    assert items[0]["doc"] != "02_fp999"


def test_chat_mode_is_honoured():
    c = FakeClient([json.dumps({"mode": "chat", "queries": []})])
    pipe = _cond_pipe(c)
    mode, _, _ = ev.Pipeline.conduct(pipe, "what did you just say?", ())
    assert mode == "chat" and pipe.conductor_stats["chat"] == 1


def test_a_broken_conductor_reply_keeps_the_raw_message():
    c = FakeClient(["not json at all"])
    pipe = _cond_pipe(c)
    mode, items, _ = ev.Pipeline.conduct(pipe, "a question", ())
    assert mode == "retrieve" and items == [{"q": "a question", "doc": None}]
    assert pipe.conductor_stats["failed"] == 1


def test_the_conductor_off_path_makes_no_call_at_all():
    c = FakeClient()
    pipe = _cond_pipe(c, conductor=False)
    mode, items, meta = ev.Pipeline.conduct(pipe, "a question", _TURNS)
    assert c.calls == [] and meta is None
    assert (mode, items) == ("retrieve", [{"q": "a question", "doc": None}])


# -------------------------------------------------- gap 8: verifier config ---
def _fake_res(**kw):
    base = dict(status="verified", answer="an answer", verdicts=[],
                repaired=False, repair_rejected=False)
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_production_mode_calls_verify_answer_the_way_the_app_does(monkeypatch):
    seen = {}

    def fake(answer, evidence, client=None, use_llm=True, **kw):
        seen.update(answer=answer, client=client, use_llm=use_llm, kw=kw)
        return _fake_res(answer=answer)

    monkeypatch.setattr(ev.verify, "verify_answer", fake)
    ev.verify_production("an answer", [], [], client="CLIENT")
    assert seen["use_llm"] is True
    assert "allow_repair" not in seen["kw"], \
        "the repair pathway was removed; nothing may ask for it"
    assert seen["client"] == "CLIENT"


def test_there_is_no_repair_parameter_left(monkeypatch):
    """verify_production cannot ask for a rewrite: it forwards no repair
    argument whatever the ambient environment says, and the parameter itself
    is gone rather than ignored."""
    seen = {}

    def fake(answer, evidence, client=None, use_llm=True, **kw):
        seen.update(kw=kw)
        return _fake_res()

    monkeypatch.setattr(ev.verify, "verify_answer", fake)
    monkeypatch.setenv("VERIFY_REPAIR", "1")
    ev.verify_production("a", [], [])
    assert "allow_repair" not in seen["kw"]
    with pytest.raises(TypeError):
        ev.verify_production("a", [], [], allow_repair=True)


def test_deterministic_mode_makes_no_verification_call():
    """The zero-API guarantee, tested rather than asserted: one model call for
    the answer, and nothing else."""
    c = FakeClient()
    ev._run_case(FakePipe(_answer_out(), verifier_mode="deterministic"), c,
                 _args(), _case(turns=[]))
    assert len(c.calls) == 1


def test_the_judge_budget_is_recorded_per_case(monkeypatch):
    monkeypatch.setattr(ev.verify, "verify_answer",
                        lambda a, e, **kw: _fake_res(answer=a))
    monkeypatch.setattr(ev.verify, "extract_claims", lambda a: ["c"] * 20)
    monkeypatch.setattr(
        ev.verify, "classify_deterministic",
        lambda c, e: [types.SimpleNamespace(status=ev.verify.UNSUPPORTED,
                                            plausible=True)] * 20)
    _, _, judge = ev.verify_production("a", [], [])
    assert judge["judge_candidates"] == 20
    assert judge["judge_max_claims"] == ev.JUDGE_MAX_CLAIMS
    assert judge["judge_budget_exhausted"] == 1


def test_a_judge_budget_that_does_not_bind_reports_zero(monkeypatch):
    monkeypatch.setattr(ev.verify, "verify_answer",
                        lambda a, e, **kw: _fake_res(answer=a))
    monkeypatch.setattr(ev.verify, "extract_claims", lambda a: [])
    monkeypatch.setattr(ev.verify, "classify_deterministic", lambda c, e: [])
    _, _, judge = ev.verify_production("a", [], [])
    assert judge["judge_budget_exhausted"] == 0 and judge["judge_candidates"] == 0


# ------------------------------------------------- gap 9: usage accounting ---
def test_the_metered_client_books_the_judge():
    """The judge is the detector's only model call; a repair prompt cannot be
    booked because the prompt itself no longer exists."""
    sink = []
    inner = FakeClient(["{}"])
    m = ev.MeteredClient(inner, sink)
    m.chat.completions.create(model="m", messages=[
        {"role": "system", "content": ev.verify.ADJUDICATE_PROMPT},
        {"role": "user", "content": "claims"}])
    assert [c["role"] for c in sink] == ["judge"]
    assert all(c["prompt_tokens"] == 11 for c in sink)
    assert len(inner.calls) == 1, "the proxy must pass every call through"
    assert not hasattr(ev.verify, "REPAIR_PROMPT")


def test_an_unrecognised_verify_call_is_labelled_rather_than_lost():
    sink = []
    ev.MeteredClient(FakeClient(), sink).chat.completions.create(
        model="m", messages=[{"role": "system", "content": "something else"}])
    assert sink[0]["role"] == "verify-other"


def test_usage_totals_accounts_every_role_and_keeps_the_answer_latency():
    rows = [{"usage": ev.turn_usage([
        {"role": "conductor", "latency_s": 0.5, "prompt_tokens": 100,
         "completion_tokens": 10, "total_tokens": 110},
        {"role": "answer", "latency_s": 3.0, "prompt_tokens": 2000,
         "completion_tokens": 200, "total_tokens": 2200},
        {"role": "judge", "latency_s": 1.5, "prompt_tokens": 800,
         "completion_tokens": 80, "total_tokens": 880}])}]
    u = ev.usage_totals(rows)
    assert u["calls"] == 3
    assert u["p50"] == 5.0, "the turn cost 5s of model time, not 3"
    assert u["answer_p50"] == 3.0
    assert u["prompt_tokens"] == 2900 and u["completion_tokens"] == 290
    assert u["by_role"]["judge"]["calls"] == 1
    assert u["by_role"]["conductor"]["prompt_tokens"] == 100


def test_usage_totals_still_reads_a_record_from_before_the_accounting():
    """release-1 and the spread runs carry a flat usage block. A comparison
    that silently dropped them would compare a run against nothing."""
    u = ev.usage_totals([{"usage": {"latency_s": 3.4, "prompt_tokens": 100,
                                    "completion_tokens": 10}}])
    assert u["calls"] == 1 and u["p50"] == 3.4 and u["answer_p50"] == 3.4
    assert u["by_role"]["answer"]["calls"] == 1


def test_turn_usage_of_a_turn_that_called_nothing_is_empty():
    assert ev.turn_usage([]) == {} and ev.turn_usage([None]) == {}


# --------------------------------------------- F12 rescoring / compare gate --
def _rec_row(cid, answer, hits, **kw):
    row = {"id": cid, "class": "identifier", "lang": "en", "question": "q?",
           "answer": answer, "hits": hits, "notes_used": {}, "score": 1.0,
           "claims": {"claims": 1, "supported": 1, "support_rate": 1.0}}
    row.update(kw)
    return row


def test_rescore_recomputes_offline_and_keeps_the_recorded_block(tmp_path):
    hits = [{"doc": FP151, "page": 45, "score": 0.9,
             "text": "A.8 Total GCF funding requested: 18.5 M USD"}]
    src = tmp_path / "rec.jsonl"
    src.write_text(json.dumps(_rec_row(
        "a", f"FP151 requests 18.5 million USD [{FP151}, p. 45].", hits)) + "\n",
        encoding="utf-8")
    out = tmp_path / "out.jsonl"
    ev.run_rescore(src, out)
    got = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert got["rescored"]["api_calls"] == 0
    assert got["rescored"]["evidence_source"] == "record"
    assert got["claims_recorded"]["supported"] == 1
    assert got["claims"]["groundedness_rate"] is not None
    assert got["claims"]["citation_completeness_rate"] is not None


def test_rescore_needs_evidence_when_the_record_carries_none(tmp_path):
    src = tmp_path / "rec.jsonl"
    src.write_text(json.dumps(_rec_row(
        "a", "FP151 requests 18.5 million USD.",
        [{"doc": FP151, "page": 45, "score": 0.9}])) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="no backfilled evidence"):
        ev.run_rescore(src, tmp_path / "out.jsonl")


def test_rescore_refuses_a_backfill_whose_keys_do_not_match(tmp_path):
    src = tmp_path / "rec.jsonl"
    src.write_text(json.dumps(_rec_row(
        "a", "FP151 requests 18.5 million USD.",
        [{"doc": FP151, "page": 45, "score": 0.9}])) + "\n", encoding="utf-8")
    ev_file = tmp_path / "ev.jsonl"
    ev_file.write_text(json.dumps({
        "case_id": "a", "answer": "FP151 requests 18.5 million USD.",
        "evidence_keys_match": False, "evidence": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not match"):
        ev.run_rescore(src, tmp_path / "out.jsonl", evidence=ev_file)


def _two_runs(tmp_path, a_rate, b_rate, key="groundedness_rate"):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    for path, rate in ((a, a_rate), (b, b_rate)):
        rows = [{"id": "c1", "class": "identifier", "score": 1.0,
                 "claims": {"support_rate": 0.9, key: rate} if rate is not None
                 else {"support_rate": 0.9}}]
        path.write_text("".join(json.dumps(r) + "\n" for r in rows),
                        encoding="utf-8")
    return a, b


def test_require_metrics_reports_no_regression_and_the_delta(tmp_path, capsys):
    a, b = _two_runs(tmp_path, 0.80, 0.90)
    report = ev.run_compare(a, b, require_metrics=["groundedness_rate"])
    assert report["no_regression"] is True
    assert report["metrics"]["groundedness_rate"]["delta"] == pytest.approx(0.10)
    assert "no_regression" in capsys.readouterr().out


def test_require_metrics_fails_on_a_regression(tmp_path):
    a, b = _two_runs(tmp_path, 0.90, 0.80)
    assert ev.run_compare(a, b, require_metrics=["groundedness_rate"]
                          )["no_regression"] is False


def test_require_metrics_fails_on_a_metric_neither_run_carries(tmp_path):
    a, b = _two_runs(tmp_path, None, None)
    report = ev.run_compare(a, b, require_metrics=["groundedness_rate"])
    assert report["no_regression"] is False
    assert report["metrics"]["groundedness_rate"]["a"] is None
    assert report["missing"]


def test_compare_without_require_metrics_still_returns_none(tmp_path):
    a, b = _two_runs(tmp_path, 0.9, 0.8)
    assert ev.run_compare(a, b) is None


def test_the_cli_exits_1_when_a_required_metric_regresses(tmp_path):
    a, b = _two_runs(tmp_path, 0.90, 0.80)
    assert ev.main(["--compare", str(a), str(b),
                    "--require-metrics", "groundedness_rate"]) == 1


def test_the_cli_exits_0_when_it_improves(tmp_path):
    a, b = _two_runs(tmp_path, 0.80, 0.90)
    assert ev.main(["--compare", str(a), str(b),
                    "--require-metrics", "groundedness_rate"]) == 0


# ------------------------------------------------------- release defaults ----
def _capture_args(monkeypatch):
    """Drive main() far enough to see the resolved flags without hashing a
    750 MB index or calling anything."""
    seen = {}
    monkeypatch.setattr(ev, "run_eval",
                        lambda args, cases: (seen.update(vars(args)), [])[1])
    monkeypatch.setattr(ev, "_sha256_file", lambda p, missing=None: "sha")
    return seen


def test_release_turns_the_parity_switches_on_by_default(monkeypatch):
    seen = _capture_args(monkeypatch)
    ev.main(["--release", "--ids", "id-fp151-gcf"])
    assert seen["production_planner"] is True and seen["conductor"] is True
    assert seen["history_mode"] == "isolated"
    assert seen["comparison_flag"] == "decomposed"
    assert seen["verifier_mode"] == "deterministic", \
        "production verification stays opt-in: it costs a judge call per case"


def test_the_parity_switches_can_be_turned_off_under_release(monkeypatch):
    seen = _capture_args(monkeypatch)
    ev.main(["--release", "--no-conductor", "--no-production-planner",
             "--ids", "id-fp151-gcf"])
    assert seen["production_planner"] is False and seen["conductor"] is False


def test_a_non_release_run_leaves_them_off(monkeypatch):
    seen = _capture_args(monkeypatch)
    ev.main(["--answers", "--ids", "id-fp151-gcf"])
    assert seen["production_planner"] is False and seen["conductor"] is False


def test_a_zero_api_run_cannot_ask_for_a_model_call():
    with pytest.raises(SystemExit, match="--conductor needs"):
        ev.main(["--retrieval-only", "--conductor", "--ids", "id-fp151-gcf"])
    with pytest.raises(SystemExit, match="verifier-mode production needs"):
        ev.main(["--retrieval-only", "--verifier-mode", "production",
                 "--ids", "id-fp151-gcf"])
