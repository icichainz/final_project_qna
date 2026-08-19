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
    """Production shape: the conductor emits no doc tag for a cold
    single-topic question, and neither guard may invent one."""
    items = ev.Pipeline.plan(_plan_pipe(), Q152)
    assert items == [{"q": Q152, "doc": None}]


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
