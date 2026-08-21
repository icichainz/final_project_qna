"""The one wire that carries the user's own words into retrieval.

`Retriever` can rank a document's pages with two probes — the query it was
given and the caller's `original` — but it cannot know WHOSE message the
original is. main() decides that, and only one rule is defensible:

  * a turn that ran a single query is about a single document, so the message
    is that document's page probe;
  * a turn that fanned out names every document it compares, so inside any one
    of them the message is a probe for the OTHERS' names and figures, and
    nothing is passed.

Both halves are asserted here against the real main(), because the rule lives
in the wiring and nowhere else. Retrieval behaviour itself is in
tests/test_section_retrieval.py.
"""
import asyncio
import json
import sys
import types

import pytest

from gcf_qna.app import chainlit_app as app
from gcf_qna.rag import registry
from gcf_qna.rag.retrieve import Hit

FP151 = "124_gcf-b27-02-add11"
FP152 = "123_gcf-b27-02-add12"
FP172 = "103_gcf-b30-03-add04"

FAKE_REGISTRY = {
    FP151: {"fp": 151, "board": 27, "year": 2020, "accredited_entity": "IUCN"},
    FP152: {"fp": 152, "board": 27, "year": 2020, "accredited_entity": "Pegasus"},
    FP172: {"fp": 172, "board": 30, "year": 2021, "accredited_entity": "IFAD"},
}


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(registry, "_cache", FAKE_REGISTRY)
    yield


# --------------------------------------------------------------- fakes -----
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


class RecordingRetriever:
    """Records the FOURTH argument as well: the point of these tests."""

    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search_with_confidence(self, query, top_k=10, doc_filter=None,
                               original=None):
        self.calls.append({"q": query, "doc": doc_filter, "original": original})
        return list(self.hits), 0.9


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
        return types.SimpleNamespace(choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=json.dumps(payload)))])

    async def _stream(self):
        yield types.SimpleNamespace(choices=[types.SimpleNamespace(
            delta=types.SimpleNamespace(content=self.answer))])


def _now(fn, *a, **kw):
    async def run():
        return fn(*a, **kw)
    return run()


HITS = [Hit(text="A.8 Total GCF funding requested: 18.5 M USD.",
            doc_id=FP172, score=0.8, page=6)]


@pytest.fixture
def app_env(monkeypatch, fake_registry):
    FakeMessage.sent = []
    monkeypatch.setattr(app.cl, "Message", FakeMessage)
    monkeypatch.setattr(app.cl, "Step", FakeStep)
    monkeypatch.setattr(app.cl, "make_async",
                        lambda fn: (lambda *a, **kw: _now(fn, *a, **kw)))
    monkeypatch.setattr(app, "ground_chunk", lambda *a, **kw: None)
    yield


def _run(monkeypatch, question, conductor_json, history=()):
    retriever = RecordingRetriever(HITS)
    session = FakeSession(retriever=retriever, history=list(history))
    client = FakeOpenAI(conductor_json=conductor_json)
    monkeypatch.setattr(app.cl, "user_session", session)
    monkeypatch.setitem(sys.modules, "openai",
                        types.SimpleNamespace(AsyncOpenAI=lambda **kw: client))
    asyncio.run(app.main(FakeMessage(content=question)))
    return retriever


FRENCH = "Quel est le financement GCF du FP172 au Népal ?"
REWRITE = "GCF funding amount for project FP172 in Nepal"


def test_a_single_query_turn_carries_the_message_as_the_page_probe(
        monkeypatch, app_env):
    """The regressed shape: the conductor translated the question, and the
    French original is what ranks p.6 inside the document it chose."""
    r = _run(monkeypatch, FRENCH,
             {"mode": "retrieve", "queries": [{"q": REWRITE, "doc": "fp172"}]})
    assert len(r.calls) == 1
    assert r.calls[0]["q"] == REWRITE, "the rewrite still does the retrieving"
    assert r.calls[0]["original"] == FRENCH


def test_a_fanned_out_turn_passes_no_original(monkeypatch, app_env):
    """'Compare their GCF funding' is FP151's message and FP152's message at
    once; inside either document it probes for the other one."""
    r = _run(monkeypatch, "Compare the GCF funding of FP151 and FP152",
             {"mode": "retrieve", "queries": [
                 {"q": "FP151 GCF funding amount", "doc": "fp151"},
                 {"q": "FP152 GCF funding amount", "doc": "fp152"}]})
    assert len(r.calls) == 2
    assert [c["original"] for c in r.calls] == [None, None]


def test_the_conductor_off_path_still_passes_the_message(monkeypatch, app_env):
    """No rewrite happened, so query and original are the same text — the
    retriever drops the duplicate probe itself (see _probes), and the wiring
    does not need a second rule for it."""
    monkeypatch.setattr(app.config, "CONDUCTOR", False)
    q = "What is the GCF financing of FP172?"
    r = _run(monkeypatch, q, None)
    assert len(r.calls) == 1
    assert r.calls[0]["original"] == q
    assert r.calls[0]["q"] == q
