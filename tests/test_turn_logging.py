"""One structured line per turn, and no silent `except: pass` left.

The turn used to fail quietly in three places — the conductor call, the FP-miss
guard and the registry note each swallowed every exception — and succeeded
quietly everywhere else, so a production incident left nothing behind but the
answer. LOG_LEVEL (config) turns the pipeline's own logger up; `chainlit_app`
configures logging once, for the `gcf_qna` tree only.

What is pinned here:

A. the line: one INFO record per turn, key=value, carrying the turn's shape.
B. what it must never carry: the API key, or excerpt text.
C. the failure paths: each names its stage at WARNING with a traceback, and
   each leaves the turn's behaviour exactly as it was.
D. the level: LOG_LEVEL reaches `gcf_qna`, and NOT the root logger — an
   importer of the app (the eval harness) must not inherit faiss and httpx at
   INFO.
"""
import asyncio
import json
import logging
import sys
import types

import pytest

from gcf_qna import config, pipeline
from gcf_qna.app import chainlit_app as app
from gcf_qna.rag import registry
from gcf_qna.rag.retrieve import Hit

FP220 = "55_gcf-b37-02-add11-funding-proposal-package-fp220"
Q220 = "Which accredited entity implements FP220?"
EXCERPT = "ARCAFIM is implemented by IFAD, a Rome-based UN agency."

FAKE_REGISTRY = {FP220: {"fp": 220, "board": 37, "year": 2023,
                         "accredited_entity": "IFAD"}}


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(registry, "_cache", FAKE_REGISTRY)
    monkeypatch.setattr(registry, "_cache_v2", {})
    yield


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


class FakeSession:
    def __init__(self, **values):
        self.values = values

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search_with_confidence(self, query, top_k=10, doc_filter=None,
                               original=None):
        return list(self.hits), 0.9


class FakeOpenAI:
    def __init__(self, conductor_json=None, answer="IFAD implements it.",
                 conductor_boom=False):
        self.conductor_json = conductor_json
        self.answer = answer
        self.conductor_boom = conductor_boom
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(
            create=self._create))

    async def _create(self, **kw):
        if kw.get("stream"):
            return self._stream()
        if self.conductor_boom:
            raise RuntimeError("conductor endpoint refused the connection")
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


HITS = [Hit(text=EXCERPT, doc_id=FP220, score=0.8, page=5)]


def _run(monkeypatch, question=Q220, client=None, history=()):
    client = client or FakeOpenAI()
    session = FakeSession(retriever=FakeRetriever(HITS), history=list(history),
                          memory=pipeline.ConversationMemory())
    monkeypatch.setattr(app.cl, "user_session", session)
    monkeypatch.setitem(sys.modules, "openai",
                        types.SimpleNamespace(AsyncOpenAI=lambda **kw: client))
    asyncio.run(app.main(FakeMessage(content=question)))
    return session


def _turn_lines(caplog):
    return [r for r in caplog.records
            if r.name == "gcf_qna.pipeline" and r.getMessage().startswith("turn ")]


def _fields(record):
    """The key=value pairs of one turn line, values JSON-decoded when quoted."""
    out, text = {}, record.getMessage()[len("turn "):]
    for token in _split(text):
        k, _, v = token.partition("=")
        out[k] = json.loads(v) if v.startswith('"') else v
    return out


def _split(text):
    """Split on spaces that are not inside a JSON string."""
    parts, cur, in_str, esc = [], "", False, False
    for ch in text:
        if esc:
            cur += ch
            esc = False
            continue
        if ch == "\\" and in_str:
            cur += ch
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
        if ch == " " and not in_str:
            if cur:
                parts.append(cur)
            cur = ""
            continue
        cur += ch
    if cur:
        parts.append(cur)
    return parts


# ---------------------------------------------------------------------------
# A. the line
# ---------------------------------------------------------------------------
def test_one_line_per_turn_carries_the_turns_shape(monkeypatch, app_env, caplog):
    caplog.set_level(logging.INFO, logger="gcf_qna.pipeline")
    _run(monkeypatch, client=FakeOpenAI(conductor_json={
        "mode": "retrieve",
        "queries": [{"q": "accredited entity of FP220", "doc": None}]}))
    lines = _turn_lines(caplog)
    assert len(lines) == 1, "one line per turn, not one per stage"
    got = _fields(lines[0])
    for key in ("mode", "lang", "conductor", "conductor_raw", "queries",
                "guards", "planner", "decomposed", "weak_signal", "notes",
                "conflict_probe_pages", "section_probe_pages", "hits",
                "truncated", "verify", "bad_citations", "answer_chars",
                "conductor_s", "context_s", "answer_s"):
        assert key in got, key
    assert got["mode"] == "retrieve" and got["lang"] == "English"
    assert got["conductor"] == "called"
    assert json.loads(got["queries"]) == [["accredited entity of FP220", FP220]]
    # the guards said what they rewrote: the plain tag became the real stem
    assert "registry_resolved" in got["guards"]
    assert got["notes"] == "registry" and got["hits"] == "1"


def test_a_chat_turn_and_a_guard_turn_each_log_their_own_line(monkeypatch,
                                                              app_env, caplog):
    caplog.set_level(logging.INFO, logger="gcf_qna.pipeline")
    _run(monkeypatch, "thanks, that is all",
         client=FakeOpenAI(conductor_json={"mode": "chat", "queries": []},
                           answer="You're welcome."))
    assert _fields(_turn_lines(caplog)[0])["mode"] == "chat"
    caplog.clear()
    _run(monkeypatch, "What does FP999 fund?")
    assert _fields(_turn_lines(caplog)[0])["mode"] == "guard"


def test_the_conductor_output_is_truncated_in_the_line(monkeypatch, app_env,
                                                       caplog):
    caplog.set_level(logging.INFO, logger="gcf_qna.pipeline")
    _run(monkeypatch, client=FakeOpenAI(conductor_json={
        "mode": "retrieve",
        "queries": [{"q": "accredited entity " + "x" * 900, "doc": None}]}))
    raw = _fields(_turn_lines(caplog)[0])["conductor_raw"]
    assert len(raw) <= 301 and raw.endswith("…")


# ---------------------------------------------------------------------------
# B. what the line must never carry
# ---------------------------------------------------------------------------
def test_the_line_carries_neither_the_api_key_nor_excerpt_text(monkeypatch,
                                                               app_env, caplog):
    caplog.set_level(logging.INFO, logger="gcf_qna.pipeline")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-DO-NOT-LOG-THIS-KEY")
    _run(monkeypatch)
    blob = "\n".join(r.getMessage() for r in caplog.records)
    assert "sk-DO-NOT-LOG-THIS-KEY" not in blob
    assert EXCERPT not in blob and "ARCAFIM" not in blob


# ---------------------------------------------------------------------------
# C. the failure paths that used to be silent
# ---------------------------------------------------------------------------
def test_a_failed_conductor_call_is_reported_and_still_falls_back(monkeypatch,
                                                                  app_env,
                                                                  caplog):
    """Same fallback as before — the raw message is the only query — but the
    stage now says so, with the traceback."""
    caplog.set_level(logging.WARNING, logger="gcf_qna.pipeline")
    _run(monkeypatch, client=FakeOpenAI(conductor_boom=True))
    warn = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warn and "conductor" in warn[0].getMessage()
    assert warn[0].exc_info is not None
    caplog.set_level(logging.INFO, logger="gcf_qna.pipeline")
    caplog.clear()
    _run(monkeypatch, client=FakeOpenAI(conductor_boom=True))
    got = _fields(_turn_lines(caplog)[0])
    assert got["conductor"] == "failed"
    assert json.loads(got["queries"])[0][0] == Q220     # the raw message


def test_a_broken_registry_is_reported_and_the_guard_still_declines(
        monkeypatch, app_env, caplog):
    """The FP-miss guard's `except` used to swallow this whole. The turn still
    proceeds to retrieval unguarded, which is the pre-existing behaviour."""
    caplog.set_level(logging.WARNING, logger="gcf_qna.pipeline")
    monkeypatch.setattr(registry, "load",
                        lambda: (_ for _ in ()).throw(IOError("registry gone")))
    assert pipeline.fp_miss_guard("What does FP999 fund?") is None
    warn = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warn and "fp-miss guard" in warn[0].getMessage()
    assert warn[0].exc_info is not None


def test_no_silent_pass_is_left_in_the_two_stages_that_had_one():
    """Source-level, because the defect is an ABSENCE: a bare
    `except Exception: pass` is exactly what leaves nothing behind.

    Scoped to the two stages the fix covers. The moved helpers keep their own
    `pass` arms — `_resolved_refs_note` and `_turn_doc_ids` degrade to a
    shorter note by design, and their bodies are measured artifacts that were
    not this change's to edit."""
    import inspect
    for fn in (pipeline.conductor_stage, pipeline.fp_miss_guard):
        src = inspect.getsource(fn)
        assert "pass" not in src, fn.__name__
        assert "log.warning(" in src, fn.__name__
        assert "exc_info=True" in src, fn.__name__


# ---------------------------------------------------------------------------
# D. the level
# ---------------------------------------------------------------------------
def test_log_level_is_configuration_not_a_constant():
    assert config.LOG_LEVEL == "INFO"           # the default
    import importlib
    import os
    os.environ["LOG_LEVEL"] = "WARNING"
    try:
        assert importlib.reload(config).LOG_LEVEL == "WARNING"
    finally:
        del os.environ["LOG_LEVEL"]
        importlib.reload(config)


def test_the_app_raises_the_gcf_qna_tree_and_leaves_the_root_alone():
    """An importer of the app — the eval harness — must not inherit faiss,
    sentence-transformers and httpx at INFO."""
    assert logging.getLogger("gcf_qna").level == logging.INFO
    assert logging.getLogger().level <= logging.WARNING
    assert logging.getLogger("faiss").getEffectiveLevel() >= logging.WARNING
