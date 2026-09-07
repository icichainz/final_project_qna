"""Entity memory per thread — which documents the conversation is ABOUT.

Before this, the conductor was told the conversation's documents by
`_cited_docs`: a regex over the ANSWER TEXT of the last messages, each of
which reaches the conductor truncated to 1200 characters. A turn whose
citation fell past that cut, or whose answer cited nothing at all, left the
conductor with no referent for 'it' and `_resolve_doc` with no list to pin a
mangled tag against.

`pipeline.ConversationMemory` replaces the regex with the turn's OWN resolved
plan: identifiers the registry resolved out of the user's message ('asked'),
doc tags and sub-query ids that survived the rewrite guards ('resolved'), and
the citations the answer actually made ('cited', page-aware when the verifier
ran). Never prose read for facts.

What is pinned here:

A. the structure: dedupe, recency ordering, provenance precedence, the cap and
   the conductor line's own length budget.
B. resume: rebuilt from persisted metadata when it is there, from the replayed
   steps when it is not, and the metadata never gets to break a resume.
C. the wiring, through the real `main()`: the conductor prompt carries the
   memory line instead of the citation line, the guards resolve against the
   memory's doc ids, an empty memory falls back to `_cited_docs`, and a
   factual turn updates the memory from its own plan and answer.
D. parity: the eval harness builds the memory from a fixture's `turns` with
   the same function the app's resume path uses.
"""
import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_answers as ev  # noqa: E402
from gcf_qna import config, pipeline  # noqa: E402
from gcf_qna.app import chainlit_app as app  # noqa: E402
from gcf_qna.rag import registry  # noqa: E402
from gcf_qna.rag.retrieve import Hit  # noqa: E402

FP151 = "124_gcf-b27-02-add11"
FP152 = "123_gcf-b27-02-add12"
FP173 = "102_gcf-b30-02-add05"
FP220 = "55_gcf-b37-02-add11-funding-proposal-package-fp220"
FP248 = "28_gcf-b40-02-add10-rev01-funding-proposal-package-fp248"
FP254 = "22_gcf-b40-02-add16-rev01-funding-proposal-package-fp254"

FAKE_REGISTRY = {
    FP151: {"fp": 151, "board": 27, "year": 2020, "accredited_entity": "IUCN"},
    FP152: {"fp": 152, "board": 27, "year": 2020, "accredited_entity": "Pegasus"},
    FP173: {"fp": 173, "board": 30, "year": 2021, "accredited_entity": "IDB"},
    FP220: {"fp": 220, "board": 37, "year": 2023, "accredited_entity": "IFAD"},
    FP248: {"fp": 248, "board": 40, "year": 2024, "accredited_entity": "GIZ"},
    FP254: {"fp": 254, "board": 40, "year": 2024, "accredited_entity": "IFC"},
}


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(registry, "_cache", FAKE_REGISTRY)
    monkeypatch.setattr(registry, "_cache_v2", {})
    yield


# ---------------------------------------------------------------------------
# A. the structure
# ---------------------------------------------------------------------------
def test_an_entry_carries_the_registry_facts_not_a_parse_of_the_id(fake_registry):
    mem = pipeline.ConversationMemory()
    mem.update(asked=[FP151], turn=1)
    entry, = mem.entries
    # B.27-era stems print no FP number at all: the registry is the only source
    assert "151" not in FP151
    assert (entry.fp, entry.board, entry.year) == (151, 27, 2020)
    assert entry.label == "B.27, 2020"
    assert (entry.first_turn, entry.last_turn, entry.how) == (1, 1, "asked")


def test_a_document_the_registry_does_not_have_never_enters(fake_registry):
    """An invented tag is not a document the conversation was about."""
    mem = pipeline.ConversationMemory()
    mem.update(resolved=["02_fp999", "", None], cited=["not-a-doc"], turn=1)
    assert mem.entries == [] and not mem


def test_repeats_dedupe_and_move_to_the_recent_end(fake_registry):
    mem = pipeline.ConversationMemory()
    mem.update(asked=[FP151], turn=1)
    mem.update(asked=[FP220], turn=2)
    mem.update(cited=[FP151], turn=3)
    assert mem.doc_ids() == [FP220, FP151]          # oldest first, 'it' last
    entry = mem.get(FP151)
    assert (entry.first_turn, entry.last_turn) == (1, 3)
    assert len(mem) == 2


def test_the_strongest_provenance_wins_and_is_never_downgraded(fake_registry):
    mem = pipeline.ConversationMemory()
    mem.update(cited=[FP151], turn=1)
    assert mem.get(FP151).how == "cited"
    mem.update(resolved=[FP151], turn=2)
    assert mem.get(FP151).how == "resolved"
    mem.update(asked=[FP151], turn=3)
    assert mem.get(FP151).how == "asked"
    mem.update(cited=[FP151], turn=4)
    assert mem.get(FP151).how == "asked", "a citation must not demote an ask"


def test_one_turns_three_sources_land_strongest_first(fake_registry):
    """Within a turn: what the user named, then what the plan resolved, then
    what the answer cited."""
    mem = pipeline.ConversationMemory()
    mem.update(asked=[FP220], resolved=[FP248], cited=[FP254], turn=1)
    assert mem.doc_ids() == [FP220, FP248, FP254]
    assert [e.how for e in mem.entries] == ["asked", "resolved", "cited"]


def test_the_cap_drops_the_oldest_never_the_newest(fake_registry):
    mem = pipeline.ConversationMemory(max_docs=2)
    for i, doc in enumerate((FP151, FP152, FP220), start=1):
        mem.update(asked=[doc], turn=i)
    assert mem.doc_ids() == [FP152, FP220]


def test_the_default_cap_is_the_budget_the_citation_line_had(fake_registry):
    """`_cited_docs` shipped its last 12; the memory spends the same budget on
    better-chosen entries."""
    assert pipeline._MEMORY_MAX_DOCS == 12
    assert pipeline.ConversationMemory().max_docs == 12


def test_the_conductor_line_names_the_identifier_and_the_stem(fake_registry):
    mem = pipeline.ConversationMemory()
    mem.update(asked=[FP220], turn=1)
    mem.update(asked=[FP254], turn=2)
    assert mem.conductor_line() == (
        "Documents discussed so far, most recent last: "
        f"FP220 = {FP220}; FP254 = {FP254}")


def test_an_empty_memory_has_no_line(fake_registry):
    assert pipeline.ConversationMemory().conductor_line() is None


#: The conductor prompt is a length budget like the answer prompt
#: (tests/test_prompts.py): every block in it is paid for on every turn, and
#: this line is the only unbounded thing in it — twelve entries of a
#: 60-character stem is ~800 characters against a ~1,400-character prompt.
MAX_MEMORY_LINE_CHARS = 900


def test_the_conductor_line_stays_within_its_budget(fake_registry):
    """Over budget, the OLDEST entries pay: the referent of 'it' is at the
    recent end, so that is the end that must survive."""
    assert pipeline._MEMORY_LINE_CHARS == MAX_MEMORY_LINE_CHARS
    mem = pipeline.ConversationMemory()
    for i, doc in enumerate(sorted(FAKE_REGISTRY) * 3, start=1):
        mem.update(asked=[doc], turn=i)
    line = mem.conductor_line()
    assert len(line) <= MAX_MEMORY_LINE_CHARS
    assert line.endswith(mem.doc_ids()[-1])          # the newest is still there


# ---------------------------------------------------------------------------
# B. resume
# ---------------------------------------------------------------------------
RESUMED = [
    {"role": "user", "content": "What is the total GCF funding in FP151?"},
    {"role": "assistant",
     "content": f"FP151 requests 18.5 M USD [{FP151}, p. 5 — FP151, B.27, 2020]."},
    {"role": "user", "content": "And its accredited entity?"},
    {"role": "assistant", "content": f"IUCN [{FP151}, cover pages]."},
    {"role": "user", "content": "Compare it with FP220."},
    {"role": "assistant",
     "content": f"FP220 is implemented by IFAD [{FP220}, p. 5]."},
]


def test_a_resumed_thread_rebuilds_from_the_citations_it_can_still_read(fake_registry):
    mem = pipeline.ConversationMemory.from_history(RESUMED)
    assert mem.doc_ids() == [FP151, FP220]
    assert mem.get(FP151).how == "asked"            # the message named it
    assert mem.get(FP220).how == "asked"
    assert mem.turn == 3


def test_the_rebuild_survives_a_thread_that_cited_nothing(fake_registry):
    mem = pipeline.ConversationMemory.from_history(
        [{"role": "user", "content": "hello"},
         {"role": "assistant", "content": "hi there"}])
    assert not mem and mem.conductor_line() is None


def test_metadata_round_trips(fake_registry):
    mem = pipeline.ConversationMemory()
    mem.update(asked=[FP151], cited=[FP220], turn=4)
    back = pipeline.ConversationMemory.from_metadata(
        json.loads(json.dumps(mem.to_metadata())))
    assert back.doc_ids() == mem.doc_ids() and back.turn == 4
    assert [e.how for e in back.entries] == ["asked", "cited"]
    assert back.conductor_line() == mem.conductor_line()


@pytest.mark.parametrize("bad", [None, {}, {"docs": []}, {"docs": [{}]},
                                 {"docs": "not a list"}, "garbage"])
def test_unusable_metadata_hands_the_resume_back_to_the_rebuild(bad, fake_registry):
    """None means 'rebuild from the steps' — never a crashed resume."""
    got = pipeline.ConversationMemory.from_metadata(bad)
    assert got is None or not got


def test_the_app_rebuilds_from_the_steps_when_the_metadata_is_absent(
        monkeypatch, fake_registry):
    steps = [{"type": "user_message", "output": m["content"], "createdAt": f"{i}"}
             if m["role"] == "user" else
             {"type": "assistant_message", "output": m["content"], "createdAt": f"{i}"}
             for i, m in enumerate(RESUMED)]
    session = _FakeSession()
    monkeypatch.setattr(app.cl, "user_session", session)
    monkeypatch.setattr(app.cl, "make_async",
                        lambda fn: (lambda *a, **kw: _now(fn, *a, **kw)))
    monkeypatch.setattr(app, "get_retriever", lambda: "RETRIEVER")
    asyncio.run(app.on_resume({"steps": steps, "metadata": None}))
    assert session.get("memory").doc_ids() == [FP151, FP220]


def test_the_app_prefers_the_persisted_memory_over_the_rebuild(
        monkeypatch, fake_registry):
    """Cheap when it is there; the rebuild is what makes it optional."""
    mem = pipeline.ConversationMemory()
    mem.update(asked=[FP254], turn=7)
    session = _FakeSession()
    monkeypatch.setattr(app.cl, "user_session", session)
    monkeypatch.setattr(app.cl, "make_async",
                        lambda fn: (lambda *a, **kw: _now(fn, *a, **kw)))
    monkeypatch.setattr(app, "get_retriever", lambda: "RETRIEVER")
    asyncio.run(app.on_resume(
        {"steps": [], "metadata": json.dumps({app._MEMORY_KEY: mem.to_metadata()})}))
    got = session.get("memory")
    assert got.doc_ids() == [FP254] and got.turn == 7


# ---------------------------------------------------------------------------
# harness for the wiring half (the shape of tests/test_step1_isolation.py)
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

    async def update(self):
        return self


class FakeStep:
    def __init__(self, name=None):
        self.name, self.output = name, None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search_with_confidence(self, query, top_k=10, doc_filter=None,
                               original=None):
        self.calls.append({"q": query, "doc": doc_filter})
        return list(self.hits), 0.9


class FakeOpenAI:
    def __init__(self, conductor_json=None, answer="ANSWER"):
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
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=json.dumps(payload)))])

    async def _stream(self):
        yield types.SimpleNamespace(choices=[types.SimpleNamespace(
            delta=types.SimpleNamespace(content=self.answer))])

    @property
    def conductor_call(self):
        return next(c for c in self.calls if not c.get("stream"))


def _now(fn, *a, **kw):
    async def run():
        return fn(*a, **kw)
    return run()


@pytest.fixture
def app_env(monkeypatch, fake_registry):
    FakeMessage.sent = []
    monkeypatch.setattr(app.cl, "Message", FakeMessage)
    monkeypatch.setattr(app.cl, "Step", FakeStep)
    monkeypatch.setattr(app.cl, "make_async",
                        lambda fn: (lambda *a, **kw: _now(fn, *a, **kw)))
    monkeypatch.setattr(app, "ground_chunk", lambda *a, **kw: None)
    monkeypatch.setattr(config, "PLANNER", False)
    yield


HITS = [Hit(text="ARCAFIM is implemented by IFAD.", doc_id=FP220, score=0.8, page=5)]


def _run_main(monkeypatch, question, history, client, memory=None):
    retriever = FakeRetriever(HITS)
    session = _FakeSession(retriever=retriever, history=list(history),
                           memory=memory)
    monkeypatch.setattr(app.cl, "user_session", session)
    monkeypatch.setitem(sys.modules, "openai",
                        types.SimpleNamespace(AsyncOpenAI=lambda **kw: client))
    asyncio.run(app.main(FakeMessage(content=question)))
    return session, retriever


def _conductor_user(client):
    return client.conductor_call["messages"][1]["content"]


# ---------------------------------------------------------------------------
# C. the wiring
# ---------------------------------------------------------------------------
HISTORY = [
    {"role": "user", "content": "Which accredited entity implements FP220?"},
    {"role": "assistant", "content": f"IFAD [{FP220}, p. 5]."},
]


def test_the_conductor_prompt_carries_the_memory_line(monkeypatch, app_env):
    mem = pipeline.ConversationMemory()
    mem.update(asked=[FP220], turn=1)
    client = FakeOpenAI()
    _run_main(monkeypatch, "And its financing?", HISTORY, client, memory=mem)
    user = _conductor_user(client)
    assert f"Documents discussed so far, most recent last: FP220 = {FP220}" in user
    assert "Documents cited in conversation:" not in user


def test_an_empty_memory_falls_back_to_the_citation_regex(monkeypatch, app_env):
    """An old session, or a thread whose replay cited nothing: the behaviour
    that predates the memory is exactly what is left."""
    client = FakeOpenAI()
    _run_main(monkeypatch, "And its financing?", HISTORY, client,
              memory=pipeline.ConversationMemory())
    user = _conductor_user(client)
    assert f"Documents cited in conversation: {FP220}" in user
    assert "Documents discussed so far" not in user


# The comparison whose history half has no citation to be found in: the
# earlier answer states the figure and names no document (a truncated answer,
# or one that hedged), so `_cited_docs` reads nothing out of it.
_UNCITED_HISTORY = [
    {"role": "user", "content": "What is the total GCF financing of FP254?"},
    {"role": "assistant", "content": "It requests USD 58,000,000 from the GCF."},
]
_FANOUT = {"mode": "retrieve",
           "queries": [{"q": "total financing", "doc": "fp220"},
                       {"q": "total financing", "doc": "02_fp254"}]}


def test_the_memory_is_what_the_rewrite_guards_resolve_a_tag_against(
        monkeypatch, app_env):
    """The comparison's history half keeps its scope.

    'How does FP220 compare to it?' names ONE identifier and fans out to two,
    so the second tag is only defensible if it points at a document the
    conversation is known to be about. `_resolve_doc` decides that against
    `history_docs`, and those now come from the memory — which holds FP254
    from the turn that ASKED about it, whether or not the answer cited it.
    """
    mem = pipeline.ConversationMemory()
    mem.update(asked=[FP254], turn=1)
    client = FakeOpenAI(conductor_json=_FANOUT)
    _, retriever = _run_main(monkeypatch, "How does FP220 compare to it?",
                             _UNCITED_HISTORY, client, memory=mem)
    assert [c["doc"] for c in retriever.calls] == [FP220, FP254]
    assert [c["q"] for c in retriever.calls] == ["total financing"] * 2


def test_without_the_memory_that_half_of_the_comparison_loses_its_scope(
        monkeypatch, app_env):
    """The contrast, and the reason the memory exists: `_cited_docs` finds no
    document in an answer that cited none, so the history half is stripped and
    its sub-query is rewritten to the message's own identifier — both legs of
    the comparison then search FP220."""
    assert app._cited_docs(_UNCITED_HISTORY) == []
    client = FakeOpenAI(conductor_json=_FANOUT)
    _, retriever = _run_main(monkeypatch, "How does FP220 compare to it?",
                             _UNCITED_HISTORY, client,
                             memory=pipeline.ConversationMemory())
    assert [c["doc"] for c in retriever.calls] == [FP220, None]
    assert [c["q"] for c in retriever.calls] == ["total financing",
                                                 "total financing FP220"]


def test_a_factual_turn_updates_the_memory_from_its_own_plan_and_answer(
        monkeypatch, app_env):
    mem = pipeline.ConversationMemory()
    client = FakeOpenAI(answer=f"IFAD implements it [{FP220}, p. 5].")
    session, _ = _run_main(monkeypatch,
                           "Which accredited entity implements FP220?", [],
                           client, memory=mem)
    got = session.get("memory")
    assert got.doc_ids() == [FP220]
    assert got.get(FP220).how == "asked"            # the message named it


def test_a_chat_turn_leaves_the_memory_alone(monkeypatch, app_env):
    """Continuity is the answer there; no plan was resolved and no document
    was retrieved, so nothing about the conversation's documents changed."""
    mem = pipeline.ConversationMemory()
    mem.update(asked=[FP220], turn=1)
    client = FakeOpenAI(conductor_json={"mode": "chat", "queries": []},
                        answer="You asked about FP220.")
    session, _ = _run_main(monkeypatch, "what did you just say?", HISTORY,
                           client, memory=mem)
    assert session.get("memory").doc_ids() == [FP220]
    assert session.get("memory").turn == 1


def test_the_memory_survives_an_answer_that_cites_nothing(monkeypatch, app_env):
    """The shape the regex could not serve: a truncated or citation-less answer
    still leaves the conversation's document in the memory, because the PLAN
    resolved it."""
    mem = pipeline.ConversationMemory()
    client = FakeOpenAI(answer="The excerpts do not state it.")
    session, _ = _run_main(monkeypatch,
                           "Which accredited entity implements FP220?", [],
                           client, memory=mem)
    assert app._cited_docs([{"content": "The excerpts do not state it."}]) == []
    assert session.get("memory").doc_ids() == [FP220]


# ---------------------------------------------------------------------------
# D. parity with the eval harness
# ---------------------------------------------------------------------------
def test_the_harness_builds_the_memory_from_turns_the_way_the_app_does(
        monkeypatch, fake_registry):
    """A fixture's `turns` are all the harness has of a thread, so it rebuilds
    the memory with `ConversationMemory.from_history` — the same function the
    app's resume path uses when a thread's metadata is gone."""
    seen = {}

    class _Client:
        def __init__(self):
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create))

        def _create(self, **kw):
            seen["messages"] = kw["messages"]
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=json.dumps({"mode": "retrieve", "queries": []})))],
                usage=None, model="m")

    pipe = types.SimpleNamespace(
        app=app, conductor=True, client=_Client(), pins={},
        conductor_stats={"calls": 0, "fanned_out": 0, "chat": 0, "failed": 0})
    ev.Pipeline.conduct(pipe, "And its financing?", RESUMED)
    user = seen["messages"][1]["content"]
    expected = pipeline.ConversationMemory.from_history(RESUMED).conductor_line()
    assert expected and expected in user
