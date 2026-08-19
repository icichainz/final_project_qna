"""Plan step 1 — answer evidence isolated from conversation history — plus the
measured single-FP pre-scoping quick win.

A. Isolation. The factual answer call used to receive prior assistant answers
   in its messages array, so old claims (with their figures) arrived as
   pseudo-evidence: the plan's named regression is FP220 asked after unrelated
   FP254/FP248 turns, answered with IFC/GIZ instead of IFAD. The conductor
   keeps the whole conversation (it resolves 'it' / 'those' from it); the
   answer call gets the latest question, a short id-only note of resolved
   references, and THIS turn's excerpts. Chat-mode turns keep full history —
   there the conversation is the subject, not the evidence.

   These tests pin the MESSAGE-ASSEMBLY SHAPE, not model output: the gate
   ("changing the preceding conversation must not change a fully specified
   factual answer") is a property of the prompt we send, and that is
   deterministic and testable without an API call.

B. Pre-scoping. A message naming exactly one registered FP scopes its lone
   sub-query to that document (measured: r@5 88% -> 96%, page-hit 81% -> 94%,
   zero regressions). The narrow conditions are what keep the fan-out,
   board-code and unknown-id paths intact.
"""
import asyncio
import json
import sys
import types

import pytest

from gcf_qna import config
from gcf_qna.app import chainlit_app as app
from gcf_qna.rag import registry
from gcf_qna.rag.retrieve import Hit

FP151 = "124_gcf-b27-02-add11"
FP152 = "123_gcf-b27-02-add12"
FP220 = "55_gcf-b37-02-add11-funding-proposal-package-fp220"
FP248 = "28_gcf-b40-02-add10-rev01-funding-proposal-package-fp248"
FP254 = "22_gcf-b40-02-add16-rev01-funding-proposal-package-fp254"
FP274 = "02_gcf-b42-02-add16-funding-proposal-package-fp274"

FAKE_REGISTRY = {
    FP151: {"fp": 151, "board": 27, "year": 2020, "accredited_entity": "IUCN",
            "gcf_financing": "18.5 M USD"},
    FP152: {"fp": 152, "board": 27, "year": 2020, "accredited_entity": "Pegasus",
            "gcf_financing": "150 M USD"},
    FP220: {"fp": 220, "board": 37, "year": 2023, "accredited_entity": "IFAD",
            "gcf_financing": "50,000,000 USD"},
    FP248: {"fp": 248, "board": 40, "year": 2024, "accredited_entity": "GIZ"},
    FP254: {"fp": 254, "board": 40, "year": 2024, "accredited_entity": "IFC"},
    FP274: {"fp": 274, "board": 42, "year": 2025, "accredited_entity": "SC Australia"},
}


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(registry, "_cache", FAKE_REGISTRY)
    yield


# ---------------------------------------------------------------------------
# A. message assembly for the factual answer call
# ---------------------------------------------------------------------------
def test_answer_messages_carry_no_conversation():
    msgs = app._answer_messages("SYSTEM", "CONTEXT", "Which entity implements FP220?")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == "SYSTEM"
    assert "CONTEXT" in msgs[1]["content"] and "FP220" in msgs[1]["content"]


def test_answer_messages_place_the_refs_note_before_the_evidence():
    msgs = app._answer_messages("SYSTEM", "CONTEXT", "and its financing?",
                                "This question refers to: FP220 = " + FP220 + ".")
    body = msgs[1]["content"]
    assert body.index("refers to") < body.index("Context excerpts")
    assert body.endswith("Question: and its financing?")


def test_refs_note_is_built_from_ids_not_from_earlier_answer_text(fake_registry):
    """The tag is the conductor's resolution of 'those'; the note names the
    document, never anything the earlier answer claimed about it."""
    note = app._resolved_refs_note([{"q": "gcf financing", "doc": FP151},
                                    {"q": "gcf financing", "doc": FP152}],
                                   "Compare their GCF funding.")
    assert note.startswith("This question refers to: ")
    assert f"FP151 = {FP151}" in note and f"FP152 = {FP152}" in note
    assert "18.5" not in note and "IUCN" not in note   # no figures, no prose


def test_refs_note_adds_the_messages_own_ids_through_the_registry(fake_registry):
    note = app._resolved_refs_note([{"q": "accredited entity", "doc": None}],
                                   "Which accredited entity implements FP220?")
    assert note == ("This question refers to: FP220 = " + FP220 +
                    ". (Identifiers resolved by the system; the excerpts "
                    "below are the only evidence.)")


def test_refs_note_is_none_when_nothing_resolves(fake_registry):
    assert app._resolved_refs_note([{"q": "green cities", "doc": None}],
                                   "Which proposal funds green cities?") is None


def test_refs_note_survives_a_missing_registry(monkeypatch):
    monkeypatch.setattr(registry, "load",
                        lambda: (_ for _ in ()).throw(FileNotFoundError()))
    monkeypatch.setattr(registry, "resolve_fps",
                        lambda q: (_ for _ in ()).throw(FileNotFoundError()))
    note = app._resolved_refs_note([{"q": "financing", "doc": "fp152"}], "and FP152?")
    assert note.startswith("This question refers to: FP152.")   # tag is its own label


# ---------------------------------------------------------------------------
# A. the same, through main() — fakes for Chainlit I/O and the OpenAI client
# ---------------------------------------------------------------------------
class FakeMessage:
    """cl.Message: collects streamed tokens instead of rendering them."""
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
    def __init__(self, name=None):
        self.name, self.output = name, None

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
    """Records the doc filter each sub-query is searched with."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search_with_confidence(self, query, top_k=10, doc_filter=None):
        self.calls.append({"q": query, "doc": doc_filter})
        return list(self.hits), 0.9


class FakeOpenAI:
    """Splits the two calls main() makes: the conductor (JSON, non-streaming)
    and the answer (streaming)."""

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

    # the two calls, by shape
    @property
    def conductor_call(self):
        return next(c for c in self.calls if not c.get("stream"))

    @property
    def answer_call(self):
        return next(c for c in self.calls if c.get("stream"))


def _completion(text):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(
        message=types.SimpleNamespace(content=text))])


HITS = [Hit(text="ARCAFIM is implemented by IFAD.", doc_id=FP220, score=0.8, page=5)]


@pytest.fixture
def app_env(monkeypatch, fake_registry):
    """main() with Chainlit I/O and the OpenAI client faked out."""
    FakeMessage.sent = []
    monkeypatch.setattr(app.cl, "Message", FakeMessage)
    monkeypatch.setattr(app.cl, "Step", FakeStep)
    monkeypatch.setattr(app.cl, "make_async",
                        lambda fn: (lambda *a, **kw: _now(fn, *a, **kw)))
    monkeypatch.setattr(app, "ground_chunk", lambda *a, **kw: None)
    yield


def _now(fn, *a, **kw):
    async def run():
        return fn(*a, **kw)
    return run()


def _run_main(monkeypatch, question, history, client, retriever=None):
    retriever = retriever or FakeRetriever(HITS)
    session = FakeSession(retriever=retriever, history=list(history))
    monkeypatch.setattr(app.cl, "user_session", session)
    monkeypatch.setitem(sys.modules, "openai",
                        types.SimpleNamespace(AsyncOpenAI=lambda **kw: client))
    asyncio.run(app.main(FakeMessage(content=question)))
    return session, retriever


UNRELATED_HISTORY = [
    {"role": "user", "content": "What is the total GCF financing requested in FP254?"},
    {"role": "assistant",
     "content": "FP254 requests USD 58,000,000 from the GCF "
                f"[{FP254}, p. 5]. The accredited entity is the International "
                "Finance Corporation (IFC)."},
    {"role": "user", "content": "And FP248?"},
    {"role": "assistant",
     "content": f"FP248 requests 59,484,751 Euros [{FP248}, p. 5]. The "
                "accredited entity is Deutsche Gesellschaft fur Internationale "
                "Zusammenarbeit (GIZ)."},
]

Q220 = "Which accredited entity implements FP220?"
CONDUCTOR_ONE = {"mode": "retrieve",
                 "queries": [{"q": "accredited entity of FP220", "doc": None}]}


def test_factual_call_contains_no_assistant_history(monkeypatch, app_env):
    """The plan's named regression, at the level it is caused: FP254/FP248
    answers (IFC, GIZ, their figures) must not reach the answer call."""
    client = FakeOpenAI(conductor_json=CONDUCTOR_ONE)
    _run_main(monkeypatch, Q220, UNRELATED_HISTORY, client)

    msgs = client.answer_call["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert not any(m["role"] == "assistant" for m in msgs)
    blob = "\n".join(m["content"] for m in msgs)
    for leak in ("IFC", "GIZ", "58,000,000", "59,484,751", FP254, FP248):
        assert leak not in blob, f"{leak!r} leaked into the factual answer call"
    assert Q220 in msgs[1]["content"]


def test_preceding_conversation_cannot_change_a_specified_factual_prompt(
        monkeypatch, app_env):
    """The plan's gate, expressed on the prompt: same question, three
    different conversations, byte-identical answer call."""
    prompts = []
    for history in ([], UNRELATED_HISTORY, UNRELATED_HISTORY[:2]):
        client = FakeOpenAI(conductor_json=CONDUCTOR_ONE)
        _run_main(monkeypatch, Q220, history, client)
        prompts.append(client.answer_call["messages"])
    assert prompts[0] == prompts[1] == prompts[2]


def test_conductor_still_receives_the_full_conversation(monkeypatch, app_env):
    """History isolation is for the ANSWER call only — the conductor needs the
    conversation to resolve follow-ups."""
    client = FakeOpenAI(conductor_json=CONDUCTOR_ONE)
    _run_main(monkeypatch, "And how much does it request?", UNRELATED_HISTORY, client)
    convo = client.conductor_call["messages"][-1]["content"]
    assert "FP254" in convo and "FP248" in convo and FP254 in convo


def test_chat_mode_keeps_full_history(monkeypatch, app_env):
    """Conversational turns are the exception: continuity is their point."""
    client = FakeOpenAI(conductor_json={"mode": "chat", "queries": []})
    _run_main(monkeypatch, "What did you just tell me?", UNRELATED_HISTORY, client)
    msgs = client.answer_call["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant",
                                         "user", "assistant", "user"]
    assert any("GIZ" in m["content"] for m in msgs)


def test_history_is_still_recorded_for_the_next_turn(monkeypatch, app_env):
    client = FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer="FP220 -> IFAD.")
    session, _ = _run_main(monkeypatch, Q220, UNRELATED_HISTORY, client)
    hist = session.get("history")
    assert hist[-2:] == [{"role": "user", "content": Q220},
                         {"role": "assistant", "content": "FP220 -> IFAD."}]
    assert len(hist) == 6            # the conductor's memory is untouched


def test_followup_referents_reach_the_answer_call_as_ids(monkeypatch, app_env):
    """'their' resolves through the conductor's doc tags; the answer call sees
    the ids, not the earlier answer that named them."""
    history = [
        {"role": "user", "content": "What are FP151 and FP152?"},
        {"role": "assistant",
         "content": f"FP151 is implemented by IUCN [{FP151}, p. 5]. FP152 is "
                    f"implemented by Pegasus Capital Advisors LP [{FP152}, p. 5]."},
    ]
    client = FakeOpenAI(conductor_json={"mode": "retrieve", "queries": [
        {"q": "GCF financing", "doc": FP151},
        {"q": "GCF financing", "doc": FP152}]})
    _run_main(monkeypatch, "Compare their GCF funding.", history, client)
    body = client.answer_call["messages"][1]["content"]
    assert f"FP151 = {FP151}" in body and f"FP152 = {FP152}" in body
    assert "Pegasus" not in body and "IUCN" not in body


# ---------------------------------------------------------------------------
# B. deterministic single-FP pre-scoping
# ---------------------------------------------------------------------------
def test_single_fp_message_is_scoped_and_registry_resolved(fake_registry):
    """The whole point: 'fp152' would match no B.27 filename, so the tag has
    to survive _rescope_items AND be resolved by _resolve_doc_tags."""
    items = app._resolve_doc_tags(app._rescope_items(
        [{"q": "GCF financing of FP152", "doc": None}],
        "What is the GCF financing for FP152?", []))
    assert items[0]["doc"] == FP152
    assert items[0]["q"] == "GCF financing of FP152"      # query untouched


@pytest.mark.parametrize("question", [
    "What is FP86 about?",          # zero-padded stem, unpadded question
    "What is FP-220 about?",        # hyphenated
    "Que finance le FP 274 ?",      # spaced, French
])
def test_id_spelling_variants_all_scope(fake_registry, question, monkeypatch):
    monkeypatch.setitem(registry._cache,
                        "189_12-status-approved-fps-respect-fp086-gcf-b37-07",
                        {"fp": 86, "board": 37, "year": 2023})
    items = app._prescope_single_fp([{"q": question, "doc": None}], question)
    assert items[0]["doc"] in ("fp86", "fp220", "fp274")


def test_multi_fp_message_is_left_to_the_fanout(fake_registry):
    q = "Compare the GCF funding of FP151 and FP152."
    assert app._prescope_single_fp([{"q": q, "doc": None}], q)[0]["doc"] is None


def test_fanout_legs_are_never_prescoped(fake_registry):
    """One FP in the message, two sub-queries: the second is about something
    else entirely and must not inherit the FP's document."""
    items = app._rescope_items(
        [{"q": "adaptation finance in FP214", "doc": None},
         {"q": "typical adaptation projects", "doc": None}],
        "How does FP214 compare to typical projects?", [])
    assert [i["doc"] for i in items] == [None, None]


def test_a_conductor_tag_is_never_overwritten(fake_registry):
    items = app._prescope_single_fp([{"q": "financing", "doc": FP274}],
                                    "And how does FP151 compare?")
    assert items[0]["doc"] == FP274


def test_board_code_only_message_is_not_scoped(fake_registry):
    q = "What does GCF/B.42/02/Add.16 fund?"
    assert app._prescope_single_fp([{"q": q, "doc": None}], q)[0]["doc"] is None


def test_mixed_fp_and_board_code_is_not_scoped(fake_registry):
    """Two documents named two ways: scoping to the FP half would answer the
    board-code half from the wrong document."""
    q = "Are FP274 and GCF/B.37/02/Add.11 the same proposal?"
    assert app._prescope_single_fp([{"q": q, "doc": None}], q)[0]["doc"] is None


def test_unknown_fp_stays_unscoped(fake_registry):
    """fp999 resolves to nothing: a hard filter matching nothing would hide
    the weak-signal path that makes the refusal honest."""
    q = "What does FP999 fund?"
    assert app._prescope_single_fp([{"q": q, "doc": None}], q)[0]["doc"] is None


def test_question_without_an_fp_is_unscoped(fake_registry):
    for q in ("Which proposals were approved in 2020?",
              "Which proposal funds rice farming in Thailand?",
              "fp2023 is a year, not a proposal"):
        assert app._prescope_single_fp([{"q": q, "doc": None}], q)[0]["doc"] is None


def test_registry_unavailable_leaves_the_query_unscoped(monkeypatch):
    monkeypatch.setattr(registry, "resolve_fps",
                        lambda q: (_ for _ in ()).throw(FileNotFoundError()))
    q = "What is the GCF financing for FP152?"
    assert app._prescope_single_fp([{"q": q, "doc": None}], q)[0]["doc"] is None


def test_prescoping_reaches_the_retriever_without_the_conductor(
        monkeypatch, app_env):
    """CONDUCTOR=0 (or a failed conductor call) leaves the raw message as the
    only query, and _rescope_items never runs — the unconditional call in
    main() is what keeps pre-scoping alive on that path."""
    monkeypatch.setattr(config, "CONDUCTOR", False)
    client = FakeOpenAI()
    _, retriever = _run_main(monkeypatch, Q220, [], client)
    assert retriever.calls == [{"q": Q220, "doc": FP220}]


def test_prescoping_reaches_the_retriever_with_the_conductor(monkeypatch, app_env):
    client = FakeOpenAI(conductor_json=CONDUCTOR_ONE)
    _, retriever = _run_main(monkeypatch, Q220, [], client)
    assert retriever.calls == [{"q": "accredited entity of FP220", "doc": FP220}]


def test_prescope_bails_when_the_rewrite_names_a_second_id():
    """B1 regression: a rule-4 conductor rewrite resolves 'it' into a second
    FP the message never typed. Hard-scoping to the message's lone FP would
    starve the partner document, so the item must stay untagged and fall
    through to identifier routing (which handles both)."""
    from gcf_qna.app.chainlit_app import _prescope_single_fp
    items = [{"q": "FP214 total GCF financing compared with FP274", "doc": None}]
    out = _prescope_single_fp(items, "How does FP214's financing compare to it?")
    assert out[0]["doc"] is None


def test_prescope_still_fires_when_query_matches_message():
    from gcf_qna.app.chainlit_app import _prescope_single_fp
    items = [{"q": "accredited entity of FP152", "doc": None}]
    out = _prescope_single_fp(items, "Which accredited entity implements FP152?")
    assert out[0]["doc"] == "fp152"
