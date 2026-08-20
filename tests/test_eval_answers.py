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
def _verdict(status, text="a claim", kind="money", reason="because"):
    return types.SimpleNamespace(
        status=status, reason=reason,
        claim=types.SimpleNamespace(text=text, kind=kind))


def _verdict2(status, *, cited, flags=(), text="a claim", kind="money",
              reason="because"):
    """A verdict shaped like rag.verify.Verdict: status, flags, and a claim
    that knows whether it carried a citation. The three claim metrics are
    functions of exactly these three things."""
    return types.SimpleNamespace(
        status=status, reason=reason, flags=list(flags),
        claim=types.SimpleNamespace(
            text=text, kind=kind, cited=bool(cited),
            citations=(["c"] if cited else [])))


class _StubVerify:
    """verify module stand-in returning fixed verdicts — lets a test state the
    (status, cited, flags) triple directly instead of reverse-engineering an
    answer that produces it."""

    SUPPORTED, CONTRADICTED, UNSUPPORTED = (
        "supported", "contradicted", "unsupported")

    def __init__(self, verdicts, evidence=None):
        self._verdicts = list(verdicts)
        self._evidence = {("d1", 5): "text"} if evidence is None else evidence

    def extract_claims(self, answer):
        return [v.claim for v in self._verdicts]

    def build_evidence(self, hits, notes):
        return self._evidence

    def classify_deterministic(self, claims, evidence):
        return self._verdicts


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
            return [_verdict("supported"), _verdict("supported"),
                    _verdict("contradicted", "bad", reason="p.40 says 40,751,254"),
                    _verdict("unsupported", "loose")]

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


def test_score_claims_on_an_answer_with_no_claims(monkeypatch):
    class Stub:
        SUPPORTED, CONTRADICTED, UNSUPPORTED = (
            "supported", "contradicted", "unsupported")
        extract_claims = staticmethod(lambda a: [])
        build_evidence = staticmethod(lambda h, n: {})
        classify_deterministic = staticmethod(lambda c, e: [])

    monkeypatch.setattr(ev, "verify", Stub)
    got = ev.score_claims("FP999 does not exist in this corpus.", [], [])
    assert got["claims"] == 0 and got["support_rate"] is None


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


def test_uncited_note_backed_claim_is_grounded_but_citation_incomplete():
    """Runs through the REAL rag.verify, so it also pins the coupling: this
    file reads groundedness off ev.GROUNDED_FLAGS, and if the verifier renames
    or stops emitting that flag the metric would silently read 0 here."""
    note = ("Registry - 30 funding-proposal documents from 2020 in the corpus: "
            "FP151 TA Facility")
    answer = "The corpus holds 30 funding-proposal documents from 2020."
    got = ev.score_claims(answer, _hits2(FP151, 5, "unrelated passage"), [note])
    assert got["claims"] == 1
    assert got["grounded"] == 1 and got["groundedness_rate"] == 1.0
    assert got["citation_complete"] == 0
    assert got["citation_completeness_rate"] == 0.0
    assert got["citation_present"] == 0 and got["citation_presence_rate"] == 0.0
    assert got["support_rate"] == 0.0, "the compatibility gate must not forgive it"


# --------------------------------------------------------------------------
# metric contract — citation completeness is 'cited AND supported'
# (docs/claim-support-execution-plan.md "Metric contract";
#  docs/wave0-review-verdict.md finding 4)
# --------------------------------------------------------------------------
_REAL_PASSAGE = "A.8 Total GCF funding requested: 18.5 M USD"
_NEVER_RETRIEVED = "999_gcf-b99-99-add99-does-not-exist"


def test_fabricated_citation_to_a_never_retrieved_doc_is_not_citation_complete():
    """Verdict finding 4, the blocker itself. The bracket names a document this
    turn never held, so the claim cites nothing that entails it. It has
    citation PRESENCE and neither completeness nor groundedness."""
    hits = _hits2(FP151, 5, _REAL_PASSAGE)
    got = ev.score_claims(
        f"FP151 requests 18.5 million USD [{_NEVER_RETRIEVED}, p. 1].", hits, [])
    assert got["claims"] == 1
    assert got["citation_complete"] == 0, \
        "a citation to a document that was never retrieved cannot complete a claim"
    assert got["citation_completeness_rate"] == 0.0
    assert got["supported"] == 0
    assert got["citation_present"] == 1 and got["citation_presence_rate"] == 1.0


def test_appending_a_bracket_cannot_raise_citation_completeness():
    """The gaming surface: '[anything, p.1]' must move presence and nothing
    else. Same sentence, same evidence, one appended bracket."""
    hits = _hits2(FP151, 5, _REAL_PASSAGE)
    sentence = "FP151 requests 99.9 million USD"
    bare = ev.score_claims(sentence + ".", hits, [])
    gamed = ev.score_claims(f"{sentence} [{_NEVER_RETRIEVED}, p. 1].", hits, [])
    assert bare["claims"] == gamed["claims"] == 1
    assert gamed["citation_completeness_rate"] <= bare["citation_completeness_rate"]
    assert gamed["citation_complete"] == bare["citation_complete"] == 0
    assert gamed["groundedness_rate"] <= bare["groundedness_rate"]
    assert gamed["citation_present"] == 1 and bare["citation_present"] == 0, \
        "presence is the only number a fabricated bracket may move"


def test_a_real_citation_to_evidence_that_states_the_claim_is_citation_complete():
    """The permissive direction. Tightening completeness must not make it
    vacuous: a correctly cited, correctly supported claim still counts, on all
    three metrics."""
    hits = _hits2(FP151, 5, _REAL_PASSAGE)
    got = ev.score_claims(f"FP151 requests 18.5 million USD [{FP151}, p. 5].",
                          hits, [])
    assert got["claims"] == 1 and got["supported"] == 1
    assert got["citation_complete"] == 1 and got["citation_completeness_rate"] == 1.0
    assert got["citation_present"] == 1 and got["citation_presence_rate"] == 1.0
    assert got["grounded"] == 1 and got["groundedness_rate"] == 1.0


def test_citation_presence_counts_the_bracket_completeness_does_not(monkeypatch):
    """The production-judge shape, verdict-level: a cited claim the judge left
    UNSUPPORTED, and an uncited claim it promoted to SUPPORTED. Completeness
    counts neither; presence counts the first; the legacy tally counts the
    second. Nothing may collapse the three onto one number."""
    monkeypatch.setattr(ev, "verify", _StubVerify([
        _verdict2("unsupported", cited=True),
        _verdict2("supported", cited=False),
    ]))
    got = ev.score_claims("answer", [], [])
    assert got["claims"] == 2
    assert got["supported"] == 1, "legacy verdict tally"
    assert got["citation_present"] == 1 and got["citation_presence"] == "1/2"
    assert got["citation_complete"] == 0, \
        "cited-but-unsupported and supported-but-uncited are both incomplete"
    assert got["citation_completeness"] == "0/2"


def test_groundedness_reads_the_all_evidence_flag_not_citation_state(monkeypatch):
    """Groundedness is 'entailed by ANY held evidence' — it comes off the
    verifier's public flag surface, so an UNSUPPORTED claim the verifier found
    elsewhere is grounded, and a claim with no such flag is not."""
    monkeypatch.setattr(ev, "verify", _StubVerify([
        _verdict2("unsupported", cited=True, flags=["value-present-elsewhere"]),
        _verdict2("unsupported", cited=True, flags=["invalid-citation:zzz"]),
    ]))
    got = ev.score_claims("answer", [], [])
    assert got["grounded"] == 1 and got["groundedness"] == "1/2"
    assert got["groundedness_rate"] == 0.5
    assert got["citation_complete"] == 0, "grounded is not citation-complete"


def test_groundedness_can_be_zero(monkeypatch):
    monkeypatch.setattr(ev, "verify", _StubVerify([
        _verdict2("unsupported", cited=True), _verdict2("contradicted", cited=True)]))
    got = ev.score_claims("answer", [], [])
    assert got["grounded"] == 0 and got["groundedness_rate"] == 0.0
    assert got["groundedness"] == "0/2"


def test_every_claim_metric_is_published_as_n_over_d(monkeypatch):
    monkeypatch.setattr(ev, "verify", _StubVerify([
        _verdict2("supported", cited=True),
        _verdict2("unsupported", cited=True, flags=["value-present-elsewhere"]),
        _verdict2("unsupported", cited=False),
    ]))
    got = ev.score_claims("answer", [], [])
    assert got["groundedness"] == "2/3"
    assert got["citation_completeness"] == "1/3"
    assert got["citation_presence"] == "2/3"
    assert ev.n_over_d(91, 165) == "91/165"


def test_n_over_d_survives_an_empty_denominator(monkeypatch):
    monkeypatch.setattr(ev, "verify", _StubVerify([]))
    got = ev.score_claims("", [], [])
    assert got["claims"] == 0
    assert got["groundedness"] == got["citation_completeness"] == "0/0"
    assert got["citation_presence"] == "0/0"
    assert got["groundedness_rate"] is None
    assert got["citation_completeness_rate"] is None
    assert got["citation_presence_rate"] is None


def test_the_duplicate_citation_support_key_is_gone(monkeypatch):
    """Binding decision 4: `citation_support_rate` was identical to the legacy
    `support_rate` and gave --compare two rows for one regression."""
    monkeypatch.setattr(ev, "verify", _StubVerify([_verdict2("supported", cited=True)]))
    got = ev.score_claims("answer", [], [])
    assert "citation_support_rate" not in got and "citation_supported" not in got
    keys = [k for _, block, k, _ in ev.COMPARED_METRICS if block == "claims"]
    assert "citation_support_rate" not in keys


def test_compared_metric_keys_are_pairwise_distinct():
    """The startup assertion runs at import; this pins that it is not vacuous."""
    assert ev._assert_distinct_metric_keys() is ev.COMPARED_METRICS
    names = [m[0] for m in ev.COMPARED_METRICS]
    keys = [(m[1], m[2]) for m in ev.COMPARED_METRICS]
    assert len(set(names)) == len(names) == 4
    assert len(set(keys)) == len(keys)


def test_distinctness_assertion_fires_when_two_metrics_share_a_record_key():
    """Re-adding the deleted duplicate under a new display name must abort at
    startup, not print two green deltas for one regression."""
    with pytest.raises(AssertionError) as e:
        ev._assert_distinct_metric_keys((
            ("groundedness", "claims", "groundedness_rate", True),
            ("citation_completeness", "claims", "citation_completeness_rate", True),
            ("citation_support", "claims", "citation_completeness_rate", True),
        ))
    assert "claims.citation_completeness_rate" in str(e.value)


def test_distinctness_assertion_fires_when_two_metrics_share_a_name():
    with pytest.raises(AssertionError) as e:
        ev._assert_distinct_metric_keys((
            ("citation_completeness", "claims", "citation_completeness_rate", True),
            ("citation_completeness", "claims", "citation_presence_rate", False),
        ))
    assert "citation_completeness" in str(e.value)


def test_gate_threshold_is_the_exact_integer_numerator():
    assert ev.gate_threshold(165) == 157        # ceil(0.95 * 165) = 156.75 -> 157
    assert ev.gate_threshold(100) == 95         # exact, no rounding up
    assert ev.gate_threshold(20) == 19
    assert ev.gate_threshold(0) == 0


def test_offline_verifier_default_cannot_use_client_or_repair(monkeypatch):
    seen = {}
    result = types.SimpleNamespace(answer="A", status="verified", verdicts=[])

    class Stub:
        @staticmethod
        def build_evidence(hits, notes):
            return {("d", 1): "evidence"}

        @staticmethod
        def verify_answer(answer, evidence, **kwargs):
            seen.update(kwargs)
            return result

    monkeypatch.setattr(ev, "verify", Stub)
    got, evidence = ev.run_offline_verifier("A", [], client=object())
    assert got is result and evidence == {("d", 1): "evidence"}
    assert seen == {"client": None, "use_llm": False, "allow_repair": False}


def test_offline_production_verifier_can_opt_into_repair(monkeypatch):
    seen = {}
    client = object()
    result = types.SimpleNamespace(answer="fixed", status="repaired", verdicts=[])

    class Stub:
        build_evidence = staticmethod(lambda hits, notes: {})

        @staticmethod
        def verify_answer(answer, evidence, **kwargs):
            seen.update(kwargs)
            return result

    monkeypatch.setattr(ev, "verify", Stub)
    got, _ = ev.run_offline_verifier(
        "wrong", [], client=client, mode="production", allow_repair=True)
    assert got is result
    assert seen == {"client": client, "use_llm": True, "allow_repair": True}


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
             "support_rate": 1.0, "evidence_keys": ["d|5"], "failures": [],
             "grounded": 4, "groundedness_rate": 1.0, "groundedness": "4/4",
             "citation_complete": 4, "citation_completeness_rate": 1.0,
             "citation_completeness": "4/4",
             "citation_present": 4, "citation_presence_rate": 1.0,
             "citation_presence": "4/4"}
# 4 claims, every one carrying a bracket (presence 4/4) but only 2 cited AND
# supported (completeness 2/4) — the gap the old metric hid.
CLAIMS_BAD = {"claims": 4, "supported": 2, "contradicted": 1, "unsupported": 1,
              "support_rate": 0.5, "evidence_keys": ["d|5"],
              "grounded": 3, "groundedness_rate": 0.75, "groundedness": "3/4",
              "citation_complete": 2, "citation_completeness_rate": 0.5,
              "citation_completeness": "2/4",
              "citation_present": 4, "citation_presence_rate": 1.0,
              "citation_presence": "4/4",
              "failures": [{"status": "unsupported", "kind": "money",
                            "text": "t", "reason": "not in the cited evidence"}]}
# A run recorded before the split: the three metric keys simply are not there.
CLAIMS_LEGACY = {"claims": 4, "supported": 4, "contradicted": 0, "unsupported": 0,
                 "support_rate": 1.0, "evidence_keys": ["d|5"], "failures": []}


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
    # The gated number is citation completeness (6/8), NOT presence (8/8).
    assert summary["claims"]["citation_complete"] == 6
    assert summary["claims"]["citation_present"] == 8
    assert summary["claims"]["grounded"] == 7
    assert summary["claims"]["rate"] == 0.75
    assert summary["usage"]["calls"] == 2          # the errored case made no call
    assert summary["errors"] == ["c"]


def test_release_report_publishes_all_three_metrics_as_n_over_d(capsys):
    rows = [_release_row("a", "identifier", fields=FIELDS_OK, claims=CLAIMS_OK),
            _release_row("b", "comparison", score=0.5, fields=FIELDS_BAD,
                         claims=CLAIMS_BAD)]
    summary = ev.print_release_table(rows)
    out = re.sub(r"[ \t]+", " ", capsys.readouterr().out)
    assert "groundedness : 7/8" in out
    assert "citation completeness : 6/8" in out
    assert "citation presence : 8/8" in out
    assert "(reported, never gated)" in out
    # ceil(0.95 * 8) = 8 -> 6/8 FAILs. The gate reads completeness, so the
    # 8/8 presence number can never carry it.
    assert "gate >= 95% = 8/8" in out and "FAIL" in out
    assert summary["claims"]["citation_completeness"] == "6/8"
    assert summary["claims"]["citation_presence"] == "8/8"
    assert summary["claims"]["groundedness"] == "7/8"
    assert summary["claims"]["gate_threshold"] == 8


def test_release_report_says_n_a_rather_than_borrowing_another_metric(capsys):
    """Verdict finding 21: `.get('grounded', c['supported'])` printed the
    verdict tally under the groundedness header, asserting 91 == 110 == 141.
    A pre-split record reports n/a."""
    rows = [_release_row("a", "identifier", fields=FIELDS_OK, claims=CLAIMS_LEGACY)]
    summary = ev.print_release_table(rows)
    out = re.sub(r"[ \t]+", " ", capsys.readouterr().out)
    assert "groundedness : n/a" in out
    assert "citation completeness : n/a" in out
    assert "groundedness : 4/4" not in out, \
        "the supported count must never be printed as groundedness"
    assert "identifier n/a/4" in out, \
        "the per-class breakdown says n/a too, never a fabricated 0/4"
    assert "identifier 0/4" not in out
    assert summary["claims"]["grounded"] is None
    assert summary["claims"]["citation_complete"] is None
    assert summary["claims"]["groundedness"] is None
    assert summary["claims"]["rate"] is None


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
    assert "field coverage 1 cases 0/1 -> 1/1 0.0% -> 100.0%" in out
    assert "citation completeness 1 cases 2/4 -> 4/4 50.0% -> 100.0%" in out
    assert "claim support" not in out, \
        "--compare reads citation completeness, not the legacy verdict tally"


def test_compare_pools_counts_and_says_so_when_the_denominator_moves(
        capsys, tmp_path, monkeypatch):
    """A rate whose denominator moved is not a comparison: deleting two
    unsupportable claims takes 2/4 to 2/2 and must not read as a clean win."""
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    shrunk = dict(CLAIMS_OK, claims=2, supported=2, grounded=2,
                  citation_complete=2, citation_present=2,
                  groundedness="2/2", citation_completeness="2/2",
                  citation_presence="2/2")
    a = [_release_row("a", "identifier", fields=FIELDS_OK, claims=CLAIMS_BAD)]
    b = [_release_row("a", "identifier", fields=FIELDS_OK, claims=shrunk)]
    ev.run_compare(ev.record(a, "a"), ev.record(b, "b"))
    out = re.sub(r"\s+", " ", capsys.readouterr().out)
    assert "citation completeness 1 cases 2/4 -> 2/2" in out
    assert "DENOMINATOR MOVED 4 -> 2" in out


def test_compare_skips_a_claim_metric_the_old_run_never_recorded(
        capsys, tmp_path, monkeypatch):
    """A pre-split record has no citation_complete. It is skipped, never
    back-filled from `support_rate` — the two are not the same number in
    production mode."""
    monkeypatch.setattr(ev, "EVAL_DIR", tmp_path)
    a = [_release_row("a", "identifier", fields=FIELDS_OK, claims=CLAIMS_LEGACY)]
    b = [_release_row("a", "identifier", fields=FIELDS_OK, claims=CLAIMS_OK)]
    ev.run_compare(ev.record(a, "a"), ev.record(b, "b"))
    out = re.sub(r"\s+", " ", capsys.readouterr().out)
    assert "citation completeness" not in out
    assert "field coverage 1 cases 1/1 -> 1/1" in out


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

# pipeline parity and verifier no-regression records
# ==========================================================================
def test_production_planner_path_builds_matrix_and_scoped_queries(monkeypatch):
    docs = [types.SimpleNamespace(scope="doc-a", missing=False, label="FP1"),
            types.SimpleNamespace(scope="doc-b", missing=False, label="FP2")]
    plan = types.SimpleNamespace(docs=docs)
    matrix = types.SimpleNamespace(cells=[types.SimpleNamespace(status="stated")])
    searches = []

    class Retriever:
        def search_with_confidence(self, query, top_k, doc_filter):
            searches.append((query, top_k, doc_filter))
            return _hits2(doc_filter, 5, f"evidence for {doc_filter}"), 0.9

    app = types.SimpleNamespace(
        _planner_intent=lambda question, candidate: True,
        _plan_query=lambda candidate, doc: f"{doc.label} financing",
        _year_assist=lambda question, hits: (hits, None),
        _board_range_note=lambda question: None,
        _detect_lang=lambda question: "English",
        _doc_label=lambda doc, page: f"{doc}, p. {page}",
        _rescope_items=lambda items, question, cited: items,
        _resolve_doc_tags=lambda items: items,
    )
    pipe = object.__new__(ev.Pipeline)
    pipe.app = app
    pipe.retriever = Retriever()
    pipe.top_k = 10
    pipe.comparison_proxy = True
    pipe.raw_retrieval = False
    pipe.scope_single_id = False
    pipe.production_planner = True
    pipe.fp_guard = lambda question: None

    monkeypatch.setattr(ev.planner, "detect", lambda question: plan)
    monkeypatch.setattr(ev.planner, "build_matrix", lambda candidate, retriever: matrix)
    monkeypatch.setattr(ev.planner, "render", lambda candidate: "EVIDENCE MATRIX")
    monkeypatch.setattr(ev.registry, "registry_note", lambda question: None)
    monkeypatch.setattr(ev, "assemble", lambda **kwargs: kwargs)

    out = pipe.run("Compare FP1 and FP2 financing")
    assert searches == [("FP1 financing", 5, "doc-a"),
                        ("FP2 financing", 5, "doc-b")]
    assert "EVIDENCE MATRIX" in out["user"]
    parity = out["pipeline_parity"]
    assert parity["level"] == "partial"
    assert parity["deterministic_planner"]["used"] is True
    assert parity["deterministic_planner"]["matrix_in_prompt"] is True
    assert parity["conductor"] == {"available": False, "used": False}
    assert parity["retrieval_helpers"]["production_single_id_prescope"] is False
    assert parity["answer_history_isolation"] is False


def test_case_record_keeps_original_and_discloses_followup_limitation(monkeypatch):
    class Pipe:
        def run(self, question):
            return {
                "guard": True, "guard_answer": "Original answer.", "hits": [],
                "weak": False, "plan": [], "notes": {},
                "pipeline_parity": {"level": "partial", "limitations": [],
                                    "conductor": {"available": False, "used": False}},
            }

    result = types.SimpleNamespace(
        answer="Returned answer.", status="repaired", verdicts=[], repaired=True,
        repair_rejected=False)
    monkeypatch.setattr(ev, "run_offline_verifier",
                        lambda *a, **kw: (result, {}))
    args = types.SimpleNamespace(answers=True, verifier_mode="production",
                                 verifier_repair=True)
    case = _case(turns=[{"role": "user", "content": "Earlier question"}])
    rec = ev._run_case(Pipe(), object(), args, case)
    assert rec["original_answer"] == "Original answer."
    assert rec["answer"] == "Returned answer."
    assert rec["verification"]["answer_changed"] is True
    assert rec["verification"]["comparison"]["no_regression"] is True
    assert "verifier judge and repair excluded" in rec["verification"]["usage_accounting"]
    assert rec["pipeline_parity"]["followup"] == {
        "fixture_has_history": True,
        "conductor_history_resolution": "unavailable",
    }
    assert any("follow-up history" in x
               for x in rec["pipeline_parity"]["limitations"])


def test_verifier_comparison_flags_metric_regressions():
    before = {"checks": {"pass": True, "score": 1.0},
              "fields": {"coverage": 1.0},
              "claims": {"groundedness_rate": 1.0,
                         "citation_completeness_rate": 1.0,
                         "citation_presence_rate": 1.0}}
    after = {"checks": {"pass": False, "score": 0.5},
             "fields": {"coverage": 0.5},
             "claims": {"groundedness_rate": 0.5,
                        "citation_completeness_rate": 0.25,
                        "citation_presence_rate": 1.0}}
    got = ev.compare_verifier_output(before, after)
    assert got["no_regression"] is False
    assert set(got["regressions"]) == {
        "answer_checks_pass_to_fail", "field_coverage_decreased",
        "groundedness_decreased", "citation_completeness_decreased",
    }
    assert got["deltas"]["answer_score"] == -0.5
    assert got["deltas"]["citation_completeness"] == -0.75
    assert "citation_support" not in got["deltas"], \
        "one regression, one row: the duplicate key is deleted"


def test_one_regression_produces_exactly_one_row():
    """Binding decision 4, stated as behaviour: a single dropped claim must
    count once. With citation_support_rate beside citation_completeness_rate it
    counted twice and printed four deltas where three exist."""
    before = {"checks": {"pass": True, "score": 1.0}, "fields": {"coverage": 1.0},
              "claims": {"groundedness_rate": 1.0,
                         "citation_completeness_rate": 1.0,
                         "citation_presence_rate": 1.0}}
    after = {"checks": {"pass": True, "score": 1.0}, "fields": {"coverage": 1.0},
             "claims": {"groundedness_rate": 1.0,
                        "citation_completeness_rate": 0.5,
                        "citation_presence_rate": 1.0}}
    got = ev.compare_verifier_output(before, after)
    assert got["regressions"] == ["citation_completeness_decreased"]
    assert len(got["deltas"]) == 1 + len(ev.COMPARED_METRICS)   # + answer_score


def test_citation_presence_is_reported_but_never_gated():
    """Presence is the metric a fabricated bracket moves. It is published as a
    delta and must never, in either direction, be a regression."""
    before = {"checks": {"pass": True, "score": 1.0}, "fields": {"coverage": 1.0},
              "claims": {"groundedness_rate": 1.0,
                         "citation_completeness_rate": 1.0,
                         "citation_presence_rate": 1.0}}
    after = {"checks": {"pass": True, "score": 1.0}, "fields": {"coverage": 1.0},
             "claims": {"groundedness_rate": 1.0,
                        "citation_completeness_rate": 1.0,
                        "citation_presence_rate": 0.25}}
    got = ev.compare_verifier_output(before, after)
    assert got["deltas"]["citation_presence"] == -0.75
    assert got["regressions"] == [] and got["no_regression"] is True
    gated = {name for name, _, _, g in ev.COMPARED_METRICS if g}
    assert "citation_presence" not in gated
    assert gated == {"field_coverage", "groundedness", "citation_completeness"}


def test_raw_retrieval_and_production_planner_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        ev.main(["--raw-retrieval", "--production-planner"])
