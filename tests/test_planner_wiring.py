"""Plan step 6 — the step-4 comparison planner, wired into the app behind its
own switch (config.PLANNER, default OFF).

The planner itself is tested in test_planner.py; what is tested here is the
WIRING, driven through the real main() with Chainlit I/O and the OpenAI client
faked out (the harness of test_step1_isolation.py):

A. the switch. PLANNER=0 must leave the turn byte-identical to today's
   conductor path — same conductor call, same sub-queries, same answer prompt.
B. the planner path. A two-id comparison skips the conductor LLM call
   entirely, retrieves one authoritative stem per document, and puts the
   rendered evidence matrix ABOVE the registry note and the excerpts, with
   MATRIX_BLOCK in the system prompt.
C. the intent gate (review finding F4). detect() fires on ANY message naming
   >= 2 documents, comparative or not — "FP254 is interesting. Separately,
   FP248 was approved last year." is prose, not a matrix question, and belongs
   to the conductor. The gate is in the wiring, not in planner.py.
D. degradation. A matrix that throws, or one whose every cell is empty, falls
   back to the conductor path lazily — never a user-facing crash.
"""
import asyncio
import json
import sys
import types

import pytest

from gcf_qna import config
from gcf_qna.app import chainlit_app as app
from gcf_qna.app.prompts import MATRIX_BLOCK, assemble
from gcf_qna.rag import planner, registry
from gcf_qna.rag.retrieve import Hit

FP228 = "48_gcf-b38-02-add08-funding-proposal-package-fp228"
FP248 = "28_gcf-b40-02-add10-rev01-funding-proposal-package-fp248"
FP254 = "22_gcf-b40-02-add16-rev01-funding-proposal-package-fp254"

REG1 = {
    FP254: {"fp": 254, "title": "Scaling Resilient Water Infrastructure",
            "accredited_entity": "IFC", "countries": ["Fiji"], "board": 40,
            "year": 2024, "gcf_financing": "USD 258 million"},
    FP228: {"fp": 228, "title": "Cambodian Climate Financing Facility",
            "accredited_entity": "UNDP", "countries": ["Cambodia"], "board": 38,
            "year": 2024, "gcf_financing": "50 million USD"},
    FP248: {"fp": 248, "title": "Land-based Mitigation in West Kalimantan",
            "accredited_entity": "KfW", "countries": ["Indonesia"], "board": 40,
            "year": 2024, "gcf_financing": "59,484,751 Eur"},
}


def _money(raw, value, cur, page, section, status="canonical", unit=None):
    return {"raw": raw, "value": value, "currency": cur, "unit": unit,
            "page": page, "section": section, "status": status}


def _text(raw, page, section, status="canonical"):
    return {"raw": raw, "value": None, "currency": None, "unit": None,
            "page": page, "section": section, "status": status}


REG2 = {
    FP254: {"fp": 254, "facts": {
        "title": [_text("Scaling Resilient Water Infrastructure", 3, "A.1.1")],
        "accredited_entity": [_text("IFC", 3, "A.1.5")],
        "total_financing": [_money("$1,262,000,000 USD", 1262000000.0, "USD", 5, "A.7")],
        "gcf_funding_requested": [_money("USD 258 million (USD", 258000000.0, "USD",
                                         108, "rule:A.8", unit="million")],
    }},
    FP228: {"fp": 228, "facts": {
        "title": [_text("Cambodian Climate Financing Facility", 4, "A.1.1")],
        "accredited_entity": [_text("UNDP", 4, "A.1.5")],
        "total_financing": [_money("108.96 | million USD", 108960000.0, "USD", 56,
                                   "rule:A.7", unit="million")],
        "gcf_funding_requested": [_money("50 million USD", 50000000.0, "USD", 56,
                                         "rule:C.1(a)", unit="million")],
    }},
    FP248: {"fp": 248, "facts": {
        "title": [_text("Land-based Mitigation in West Kalimantan", 3, "A.1.1")],
        "gcf_funding_requested": [
            _money("59,484,751 Eur", 59484751.0, "EUR", 5, "A.8"),
            _money("EUR 150,300,751", 150300751.0, "EUR", 51, "rule:C.1(a)",
                   "conflicting")],
    }},
}


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(registry, "_cache", REG1)
    monkeypatch.setattr(registry, "_cache_v2", REG2)
    yield


# ---------------------------------------------------------------------------
# harness (test_step1_isolation.py's, plus Retriever.search for the matrix)
# ---------------------------------------------------------------------------
class FakeMessage:
    sent: list = []

    def __init__(self, content="", elements=None):
        self.content = content
        self.elements = elements or []

    async def stream_token(self, token):
        self.content += token

    async def send(self):
        FakeMessage.sent.append(self.content)
        return self


class FakeStep:
    steps: list = []

    def __init__(self, name=None):
        self.name, self.output = name, None
        FakeStep.steps.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


class FakeRetriever:
    """Records the doc filter of both retrieval entry points: the app's
    search_with_confidence fan-out and the matrix's search()."""

    def __init__(self, hits, matrix_hits=None):
        self.hits = hits
        self.matrix_hits = matrix_hits if matrix_hits is not None else hits
        self.calls = []
        self.matrix_calls = []

    def search_with_confidence(self, query, top_k=10, doc_filter=None):
        self.calls.append({"q": query, "doc": doc_filter})
        return list(self.hits), 0.9

    def search(self, query, top_k=5, doc_filter=None):
        self.matrix_calls.append({"q": query, "doc": doc_filter})
        return list(self.matrix_hits)


class FakeOpenAI:
    def __init__(self, conductor_json=None, answer="ANSWER", **kw):
        self.conductor_json = conductor_json
        self.answer = answer
        self.calls = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(
            create=self._create))

    async def _create(self, **kw):
        self.calls.append(kw)
        if kw.get("stream"):
            return self._stream()
        payload = self.conductor_json or {"mode": "retrieve", "queries": []}
        return _completion(json.dumps(payload))

    async def _stream(self):
        for part in (self.answer,):
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(
                delta=types.SimpleNamespace(content=part))])

    @property
    def conductor_calls(self):
        return [c for c in self.calls if not c.get("stream")]

    @property
    def answer_call(self):
        return next(c for c in self.calls if c.get("stream"))


def _completion(text):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content=text))])


HITS = [Hit(text="The GCF is requested to provide USD 258 million.",
            doc_id=FP254, score=0.8, page=108)]


def _now(fn, *a, **kw):
    async def run():
        return fn(*a, **kw)
    return run()


@pytest.fixture
def app_env(monkeypatch, fake_registry):
    FakeMessage.sent = []
    FakeStep.steps = []
    monkeypatch.setattr(app.cl, "Message", FakeMessage)
    monkeypatch.setattr(app.cl, "Step", FakeStep)
    monkeypatch.setattr(app.cl, "make_async",
                        lambda fn: (lambda *a, **kw: _now(fn, *a, **kw)))
    monkeypatch.setattr(app, "ground_chunk", lambda *a, **kw: None)
    yield


def _run_main(monkeypatch, question, history, client, retriever=None):
    retriever = retriever or FakeRetriever(HITS)
    session = FakeSession(retriever=retriever, history=list(history))
    monkeypatch.setattr(app.cl, "user_session", session)
    monkeypatch.setitem(sys.modules, "openai",
                        types.SimpleNamespace(AsyncOpenAI=lambda **kw: client))
    asyncio.run(app.main(FakeMessage(content=question)))
    return session, retriever


def _context_of(client):
    """The context block the answer call actually received."""
    body = client.answer_call["messages"][1]["content"]
    return body.split("Context excerpts:\n", 1)[1]


Q_COMPARE = "Compare the GCF financing of FP254 and FP228"
# what the conductor answers for it today (PLANNER=0 keeps this path)
CONDUCTOR_FANOUT = {"mode": "retrieve", "queries": [
    {"q": "total GCF funding requested", "doc": "fp254"},
    {"q": "total GCF funding requested", "doc": "fp228"}]}


# ---------------------------------------------------------------------------
# A. the switch
# ---------------------------------------------------------------------------
def test_planner_is_off_by_default():
    """This deploy ships the conductor path; PLANNER=1 is opt-in."""
    assert config.PLANNER is False


def test_planner_off_goes_to_the_conductor_exactly_as_today(monkeypatch, app_env):
    monkeypatch.setattr(config, "PLANNER", False)
    client = FakeOpenAI(conductor_json=CONDUCTOR_FANOUT)
    _, retriever = _run_main(monkeypatch, Q_COMPARE, [], client)

    assert len(client.conductor_calls) == 1
    assert [c["doc"] for c in retriever.calls] == [FP254, FP228]
    assert retriever.matrix_calls == []          # the planner never ran
    context = _context_of(client)
    assert "EVIDENCE MATRIX" not in context
    assert MATRIX_BLOCK not in client.answer_call["messages"][0]["content"]


def test_planner_off_answer_prompt_is_unchanged_by_the_wiring(monkeypatch, app_env):
    """The gate for 'PLANNER=0 changes nothing': the whole answer call — system
    prompt, refs note, context, question — is identical whether the planner
    module is reachable or removed from the turn entirely."""
    monkeypatch.setattr(config, "PLANNER", False)
    prompts = []
    for detect in (planner.detect, lambda q: pytest.fail("detect() must not run")):
        monkeypatch.setattr(app.planner, "detect", detect)
        client = FakeOpenAI(conductor_json=CONDUCTOR_FANOUT)
        _run_main(monkeypatch, Q_COMPARE, [], client)
        prompts.append(client.answer_call["messages"])
    monkeypatch.setattr(app.planner, "detect", planner.detect)
    assert prompts[0] == prompts[1]


# ---------------------------------------------------------------------------
# B. the planner path
# ---------------------------------------------------------------------------
def test_planner_on_skips_the_conductor_and_ships_the_matrix(monkeypatch, app_env):
    monkeypatch.setattr(config, "PLANNER", True)
    client = FakeOpenAI(conductor_json=CONDUCTOR_FANOUT)
    _, retriever = _run_main(monkeypatch, Q_COMPARE, [], client)

    # no conductor LLM call at all
    assert client.conductor_calls == []
    # one authoritative stem per document, each with the English query the plan
    # resolved (the conductor that would have translated was skipped)
    assert retriever.calls == [
        {"q": "FP254 total GCF funding requested amount", "doc": FP254},
        {"q": "FP228 total GCF funding requested amount", "doc": FP228}]
    context = _context_of(client)
    assert "EVIDENCE MATRIX" in context
    assert context.index("EVIDENCE MATRIX") < context.index("Registry —")
    assert context.index("Registry —") < context.index("(score ")
    assert "FP254 | gcf_funding_requested" in context
    assert "FP228 | gcf_funding_requested" in context
    # the prompt block that makes the matrix binding
    system = client.answer_call["messages"][0]["content"]
    assert MATRIX_BLOCK in system
    assert "Address EVERY row of the matrix." in system
    # evidence isolation (plan step 1) is untouched
    assert [m["role"] for m in client.answer_call["messages"]] == ["system", "user"]


def test_scoped_queries_are_english_even_for_a_french_question(monkeypatch, app_env):
    """Review finding 14. The index is English and the planner path skips the
    conductor's translation, so the raw French message reached the retriever
    verbatim: measured on 'Comparez le financement de FP151 et FP152' over the
    real index, 2/10 excerpts carried a financing figure (both incidental
    tCO2-table numbers) against 4/10 — including both documents' actual
    financing lines — for the field queries below."""
    monkeypatch.setattr(config, "PLANNER", True)
    client = FakeOpenAI()
    _, retriever = _run_main(
        monkeypatch, "Comparez le financement du FP254 et du FP228", [], client)
    fields = "total financing GCF plus co-financing amount total GCF funding requested amount"
    assert retriever.calls == [{"q": f"FP254 {fields}", "doc": FP254},
                               {"q": f"FP228 {fields}", "doc": FP228}]
    for call in retriever.calls:
        assert "Comparez" not in call["q"] and "financement" not in call["q"]
    # the user's own wording still reaches the ANSWER model untouched
    assert "Comparez le financement du FP254 et du FP228" in \
        client.answer_call["messages"][1]["content"]


def test_plan_query_joins_the_planners_own_field_phrasings(fake_registry):
    """The English map is the planner's, not a second copy in the app."""
    plan = planner.detect("Compare FP254 and FP228")           # default fields
    assert plan.default_fields
    q = app._plan_query(plan, plan.docs[0])
    assert q.startswith("FP254 ")
    for field in plan.fields[:4]:
        assert planner._FIELD_QUERIES[field] in q


def test_plan_query_is_bounded_for_a_many_field_question(fake_registry):
    plan = planner.detect(
        "Compare the title, countries, accredited entity, instruments, "
        "ESS category and beneficiaries of FP254 and FP228")
    assert len(plan.fields) > 4
    q = app._plan_query(plan, plan.docs[0])
    assert planner._FIELD_QUERIES[plan.fields[4]] not in q      # capped at four


def test_planner_path_skips_the_rewrite_guards(monkeypatch, app_env):
    """_rescope_items/_resolve_doc_tags repair MODEL-written doc tags. The plan's
    stems come from the registry, so neither guard runs on this path."""
    monkeypatch.setattr(config, "PLANNER", True)
    called = []
    monkeypatch.setattr(app, "_rescope_items",
                        lambda *a, **k: called.append("rescope") or a[0])
    monkeypatch.setattr(app, "_resolve_doc_tags",
                        lambda items: called.append("resolve") or items)
    monkeypatch.setattr(app, "_prescope_single_fp",
                        lambda items, msg: called.append("prescope") or items)
    client = FakeOpenAI()
    _, retriever = _run_main(monkeypatch, Q_COMPARE, [], client)
    assert called == []
    assert [c["doc"] for c in retriever.calls] == [FP254, FP228]


def test_planner_uses_the_registry_before_retrieval(monkeypatch, app_env):
    """Registry v2 answers the cells, so the matrix needs no search of its own;
    the fan-out still runs, because the excerpts carry the prose."""
    monkeypatch.setattr(config, "PLANNER", True)
    client = FakeOpenAI()
    _, retriever = _run_main(monkeypatch, Q_COMPARE, [], client)
    assert retriever.matrix_calls == []
    assert len(retriever.calls) == 2
    assert "stated" in _context_of(client)


def test_planner_path_keeps_the_comparison_block(monkeypatch, app_env):
    monkeypatch.setattr(config, "PLANNER", True)
    client = FakeOpenAI()
    _run_main(monkeypatch, Q_COMPARE, [], client)
    system = client.answer_call["messages"][0]["content"]
    assert "Never rank or compare amounts in different currencies" in system


def test_a_conflicting_figure_reaches_the_context_with_both_pages(
        monkeypatch, app_env):
    monkeypatch.setattr(config, "PLANNER", True)
    client = FakeOpenAI()
    _run_main(monkeypatch, "Compare the GCF financing of FP254 and FP248", [], client)
    context = _context_of(client)
    assert "CONFLICT in the same document" in context
    assert "p.5" in context and "p.51" in context
    assert "NOT COMPARABLE" in context           # USD vs EUR, plus the conflict


def test_missing_document_row_survives_a_partly_resolving_question(
        monkeypatch, app_env):
    """One id resolves, so the all-missing abstention does not fire; the matrix
    carries FP999's refusal as a row instead of dropping the identifier."""
    monkeypatch.setattr(config, "PLANNER", True)
    client = FakeOpenAI()
    _, retriever = _run_main(monkeypatch, "Compare FP254 and FP999", [], client)
    assert client.conductor_calls == []
    context = _context_of(client)
    assert "FP999 | * | MISSING DOCUMENT" in context
    assert "missing-document" in context
    assert [c["doc"] for c in retriever.calls] == [FP254]     # never scoped to fp999


def test_all_missing_ids_still_abstain_before_the_matrix(monkeypatch, app_env):
    """The FP abstention short-circuit stays AHEAD of the matrix build: a
    question naming only unknown ids is refused, not answered from a matrix of
    two missing-document rows."""
    monkeypatch.setattr(config, "PLANNER", True)
    client = FakeOpenAI()
    _run_main(monkeypatch, "Compare FP998 and FP999", [], client)
    assert client.calls == []                    # no conductor, no answer call
    assert FakeMessage.sent == [
        "FP998, FP999 does not exist in this corpus (273-document registry)."]


# ---------------------------------------------------------------------------
# C. the intent gate (F4)
# ---------------------------------------------------------------------------
NON_COMPARATIVE = ("FP254 is interesting. Separately, FP248 was approved "
                   "last year.")


def test_two_id_prose_without_a_question_goes_to_the_conductor(monkeypatch, app_env):
    monkeypatch.setattr(config, "PLANNER", True)
    client = FakeOpenAI(conductor_json={"mode": "retrieve", "queries": [
        {"q": "FP254 and FP248 overview", "doc": None}]})
    _run_main(monkeypatch, NON_COMPARATIVE, [], client)
    assert len(client.conductor_calls) == 1
    assert "EVIDENCE MATRIX" not in _context_of(client)
    assert MATRIX_BLOCK not in client.answer_call["messages"][0]["content"]


@pytest.mark.parametrize("question", [
    "Compare the GCF financing of FP254 and FP228",       # comparison word
    "FP254 vs FP248",                                     # 'vs'
    "What is the difference between FP254 and FP248?",    # 'difference'
    "Comparez le financement du FP254 et du FP248",       # FR
    "Quelle est la différence entre FP254 et FP248 ?",    # FR, accented
    "Lequel de FP254 et FP248 est le plus grand ?",       # FR 'lequel'
    "accredited entity of FP254 and FP248",               # field keyword only
    "FP254 and FP248 implementation period",              # field keyword only
    "Which of FP254 and FP248 is bigger?",                # question, both ids
])
def test_intent_gate_fires(fake_registry, question):
    plan = planner.detect(question)
    assert plan is not None
    assert app._planner_intent(question, plan) is True


@pytest.mark.parametrize("question", [
    NON_COMPARATIVE,
    "FP254 is a water project. FP248 is a forestry project.",
    "I read FP254 yesterday and FP248 the day before.",
    "Please summarize FP254. Then summarize FP248.",
])
def test_intent_gate_holds_back_non_comparative_prose(fake_registry, question):
    plan = planner.detect(question)
    assert plan is not None                      # detect() alone would fire
    assert app._planner_intent(question, plan) is False


@pytest.mark.parametrize("question", [
    "What is the GCF funding requested by FP254?",   # one identifier
    "Compare their GCF funding.",                    # follow-up, no identifier
    "Which proposals finance mangrove restoration?",  # purely semantic
])
def test_single_id_and_followup_questions_never_reach_the_planner(
        monkeypatch, app_env, question):
    monkeypatch.setattr(config, "PLANNER", True)
    client = FakeOpenAI(conductor_json={"mode": "retrieve", "queries": [
        {"q": "gcf funding requested", "doc": None}]})
    _run_main(monkeypatch, question, [], client)
    assert len(client.conductor_calls) == 1
    assert "EVIDENCE MATRIX" not in _context_of(client)


# ---------------------------------------------------------------------------
# D. degradation
# ---------------------------------------------------------------------------
def test_build_matrix_failure_falls_back_to_the_conductor(monkeypatch, app_env):
    monkeypatch.setattr(config, "PLANNER", True)

    def _boom(*a, **kw):
        raise RuntimeError("registry_v2.json is corrupt")

    monkeypatch.setattr(app.planner, "build_matrix", _boom)
    client = FakeOpenAI(conductor_json=CONDUCTOR_FANOUT)
    _, retriever = _run_main(monkeypatch, Q_COMPARE, [], client)

    # the conductor call the planner path skipped happens lazily, here
    assert len(client.conductor_calls) == 1
    assert [c["doc"] for c in retriever.calls] == [FP254, FP228]
    context = _context_of(client)
    assert "EVIDENCE MATRIX" not in context
    assert MATRIX_BLOCK not in client.answer_call["messages"][0]["content"]
    # the failure is logged in a step, never shown to the user
    assert any("RuntimeError" in (s.output or "") for s in FakeStep.steps)
    assert not any("corrupt" in m for m in FakeMessage.sent)
    assert FakeMessage.sent[0] == "ANSWER"


def test_an_all_missing_matrix_falls_back_to_the_conductor(monkeypatch, app_env):
    """No registry v2 and nothing retrievable: every cell would say MISSING, so
    the matrix carries no evidence and the normal path is the honest one."""
    monkeypatch.setattr(config, "PLANNER", True)
    monkeypatch.setattr(registry, "_cache_v2", {})
    client = FakeOpenAI(conductor_json=CONDUCTOR_FANOUT)
    retriever = FakeRetriever(HITS, matrix_hits=[])
    _, retriever = _run_main(monkeypatch, Q_COMPARE, [], client, retriever)

    assert len(client.conductor_calls) == 1
    assert retriever.matrix_calls                 # it did try
    assert "EVIDENCE MATRIX" not in _context_of(client)


def test_fallback_without_a_conductor_still_answers(monkeypatch, app_env):
    """PLANNER=1, CONDUCTOR=0, matrix broken: the raw message is the only
    query and the turn completes."""
    monkeypatch.setattr(config, "PLANNER", True)
    monkeypatch.setattr(config, "CONDUCTOR", False)
    monkeypatch.setattr(app.planner, "build_matrix",
                        lambda *a, **kw: (_ for _ in ()).throw(ValueError("x")))
    client = FakeOpenAI()
    _, retriever = _run_main(monkeypatch, Q_COMPARE, [], client)
    assert client.conductor_calls == []
    assert [c["q"] for c in retriever.calls] == [Q_COMPARE]
    assert FakeMessage.sent[0] == "ANSWER"


def test_planner_survives_a_retriever_that_raises(monkeypatch, app_env):
    """build_matrix swallows retrieval errors per cell; the turn must not die
    with it when registry facts already answer the columns."""
    monkeypatch.setattr(config, "PLANNER", True)

    class Angry(FakeRetriever):
        def search(self, *a, **kw):
            raise IOError("index unavailable")

    client = FakeOpenAI()
    _run_main(monkeypatch, Q_COMPARE, [], client, Angry(HITS))
    assert "EVIDENCE MATRIX" in _context_of(client)


# ---------------------------------------------------------------------------
# prompt assembly
# ---------------------------------------------------------------------------
def test_assemble_ships_the_matrix_block_only_when_asked():
    assert MATRIX_BLOCK in assemble(matrix=True)
    assert MATRIX_BLOCK not in assemble()
    assert MATRIX_BLOCK not in assemble(year=True, registry=True, comparison=True)


def test_matrix_block_states_the_four_rules_it_exists_for():
    for rule in ("Address EVERY row", "missing-document", "CONFLICT",
                 "Never\nconvert between currencies"):
        assert rule in MATRIX_BLOCK
