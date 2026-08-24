"""Plan step 6 — the step-5 claim verifier, wired into the app behind its own
switch (config.VERIFY, default OFF).

Since eac4c94 the verifier is a PURE DETECTOR: verify_answer takes no repair
flag, hands back the answer it was given, and never reports status 'repaired'.
The wiring's whole job is therefore to run it and to REPORT — the text the user
reads is the model's own, with exactly one exception (the abstain banner, which
is prefixed above that text, never a substitute for it).

verify.py is tested in test_verify.py; what is tested here is the WIRING,
driven through the real main() with Chainlit I/O and the OpenAI client faked
out (the harness of test_step1_isolation.py / test_planner_wiring.py):

A. the switch. VERIFY=0 must not call the verifier at all — not even to build
   evidence — and must leave the turn byte-identical to today's. A stray
   VERIFY_REPAIR in the environment is inert: nothing reads it.
B. the pass. Evidence is built from THIS turn's hits and the same computed
   note blocks the context received; the answer text and the history entry are
   the model's own whatever the verdicts say; a clean answer shows nothing extra.
C. degradation. A verifier that raises leaves the original answer on screen.
D. the UI, one status at a time, plus the sources/grounding ordering, which
   follows the answer's citations as verify.cited_sources parses them (page
   aware) once verification ran.
E. the live path, at the production config (VERIFY=1, VERIFY_LLM=1): the exact
   message stream, pinned byte for byte.
"""
import asyncio
import importlib
import json
import sys
import types

import pytest

from gcf_qna import config
from gcf_qna.app import chainlit_app as app
from gcf_qna.rag import registry, verify
from gcf_qna.rag.retrieve import Hit

FP220 = "55_gcf-b37-02-add11-funding-proposal-package-fp220"
FP248 = "28_gcf-b40-02-add10-rev01-funding-proposal-package-fp248"
FP254 = "22_gcf-b40-02-add16-rev01-funding-proposal-package-fp254"

FAKE_REGISTRY = {
    FP220: {"fp": 220, "board": 37, "year": 2023, "accredited_entity": "IFAD",
            "gcf_financing": "50,000,000 USD"},
    FP248: {"fp": 248, "board": 40, "year": 2024, "accredited_entity": "GIZ"},
    FP254: {"fp": 254, "board": 40, "year": 2024, "accredited_entity": "IFC"},
}


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(registry, "_cache", FAKE_REGISTRY)
    monkeypatch.setattr(registry, "_cache_v2", {})
    yield


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------
class FakeMessage:
    sent: list = []
    updated: list = []

    def __init__(self, content="", elements=None):
        self.content = content
        self.elements = elements or []

    async def stream_token(self, token):
        self.content += token

    async def send(self):
        FakeMessage.sent.append(self.content)
        return self

    async def update(self):
        FakeMessage.updated.append(self.content)
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
    def __init__(self, hits):
        self.hits = hits
        self.calls = []

    def search_with_confidence(self, query, top_k=10, doc_filter=None,
                               original=None):
        self.calls.append({"q": query, "doc": doc_filter})
        return list(self.hits), 0.9

    def search(self, query, top_k=5, doc_filter=None):
        return list(self.hits)


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

    @property
    def answer_call(self):
        return next(c for c in self.calls if c.get("stream"))


def _now(fn, *a, **kw):
    async def run():
        return fn(*a, **kw)
    return run()


HITS = [Hit(text="FP220 requests USD 50,000,000 from the GCF. The accredited "
                 "entity is IFAD.", doc_id=FP220, score=0.8, page=5)]

Q220 = "Which accredited entity implements FP220?"
CONDUCTOR_ONE = {"mode": "retrieve",
                 "queries": [{"q": "accredited entity of FP220", "doc": None}]}
GOOD_ANSWER = f"FP220 requests USD 50,000,000 from the GCF [{FP220}, p. 5]."


@pytest.fixture
def app_env(monkeypatch, fake_registry):
    FakeMessage.sent, FakeMessage.updated, FakeStep.steps = [], [], []
    monkeypatch.setattr(app.cl, "Message", FakeMessage)
    monkeypatch.setattr(app.cl, "Step", FakeStep)
    monkeypatch.setattr(app.cl, "make_async",
                        lambda fn: (lambda *a, **kw: _now(fn, *a, **kw)))
    monkeypatch.setattr(app, "ground_chunk", lambda *a, **kw: None)
    # the verifier's own LLM layer is never reachable from a test
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield


def _run_main(monkeypatch, question, history, client, retriever=None):
    retriever = retriever or FakeRetriever(HITS)
    session = FakeSession(retriever=retriever, history=list(history))
    monkeypatch.setattr(app.cl, "user_session", session)
    monkeypatch.setitem(sys.modules, "openai",
                        types.SimpleNamespace(AsyncOpenAI=lambda **kw: client))
    asyncio.run(app.main(FakeMessage(content=question)))
    return session, retriever


def _sources_line():
    return next((m for m in FakeMessage.sent if m.startswith("📎 Sources:")), None)


def _claim(text, kind="money"):
    return verify.Claim(text=text, kind=kind)


def _verdict(text, status=verify.UNSUPPORTED, reason="not in the cited page",
             kind="money", flags=()):
    return verify.Verdict(_claim(text, kind), status, reason, flags=list(flags))


def _result(answer, status, verdicts=()):
    """A verification result under the post-eac4c94 contract: `.answer` is the
    text that went IN, `.repaired` is False, and 'repaired' is not a status any
    more. Built from the three fields the detector still fills, so it keeps
    working if verify.py drops the repair-era ones it no longer sets."""
    return verify.RepairResult(answer, status, list(verdicts))


def _stub_verify(monkeypatch, result, record=None):
    """Replace verify_answer with one that returns `result`, recording kwargs."""
    def _fake(answer, evidence, **kw):
        if record is not None:
            record.append({"answer": answer, "evidence": evidence, **kw})
        return result
    monkeypatch.setattr(app.verify, "verify_answer", _fake)


# ---------------------------------------------------------------------------
# A. the switch
# ---------------------------------------------------------------------------
def test_verify_is_off_by_default():
    assert config.VERIFY is False
    assert config.VERIFY_LLM is True


def test_there_is_no_repair_switch_left(monkeypatch):
    """The inertness proof for a stray VERIFY_REPAIR=1.

    The .env that ships to production still carries a VERIFY_REPAIR line, and
    an operator may well write =1 into one. It cannot do anything: config
    exposes no such attribute, so no code path can consult it, and re-importing
    config with the variable set changes not one value.
    """
    assert not hasattr(config, "VERIFY_REPAIR")
    monkeypatch.setenv("VERIFY_REPAIR", "1")
    try:
        fresh = importlib.reload(config)
        assert not hasattr(fresh, "VERIFY_REPAIR")
        assert fresh.VERIFY is False and fresh.VERIFY_LLM is True
    finally:
        monkeypatch.delenv("VERIFY_REPAIR", raising=False)
        importlib.reload(config)


def test_verify_off_never_touches_the_verifier(monkeypatch, app_env):
    """Not even the evidence is built: VERIFY=0 is a turn verify.py never sees."""
    monkeypatch.setattr(config, "VERIFY", False)
    monkeypatch.setattr(app.verify, "verify_answer",
                        lambda *a, **kw: pytest.fail("verify_answer must not run"))
    monkeypatch.setattr(app.verify, "build_evidence",
                        lambda *a, **kw: pytest.fail("build_evidence must not run"))
    client = FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER)
    session, _ = _run_main(monkeypatch, Q220, [], client)

    assert FakeMessage.sent == [GOOD_ANSWER, f"📎 Sources: {FP220} p.5"]
    assert FakeMessage.updated == []
    assert session.get("history")[-1] == {"role": "assistant", "content": GOOD_ANSWER}
    assert not any(s.name == "verification" for s in FakeStep.steps)


def test_verify_off_keeps_the_invalid_citation_fallback(monkeypatch, app_env):
    """_invalid_citations is the VERIFY=0 path and is not deleted."""
    monkeypatch.setattr(config, "VERIFY", False)
    invented = f"FP220 is implemented by IFAD [{FP220}, p. 41]."
    client = FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=invented)
    _run_main(monkeypatch, Q220, [], client)
    assert "⚠️ cited but not among retrieved pages" in _sources_line()
    assert "p.41" in _sources_line()


# ---------------------------------------------------------------------------
# B. the pass
# ---------------------------------------------------------------------------
def test_a_clean_answer_verifies_and_shows_nothing_extra(monkeypatch, app_env):
    """The REAL verifier, deterministic-only: every figure of the answer is
    printed on the page it cites, so the turn looks exactly as it did."""
    monkeypatch.setattr(config, "VERIFY", True)
    monkeypatch.setattr(config, "VERIFY_LLM", False)
    client = FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER)
    session, _ = _run_main(monkeypatch, Q220, [], client)

    assert FakeMessage.sent == [GOOD_ANSWER, f"📎 Sources: {FP220} p.5"]
    assert FakeMessage.updated == []
    assert "⚠️" not in _sources_line() and "✎" not in _sources_line()
    assert session.get("history")[-1]["content"] == GOOD_ANSWER
    step = next(s for s in FakeStep.steps if s.name == "verification")
    assert step.output.startswith("status: verified")


def test_evidence_is_built_from_the_hits_and_the_context_notes(monkeypatch, app_env):
    """The audit must see exactly what the context carried: the excerpts AND
    the computed registry/year/matrix blocks, or every note-backed claim would
    be reported unsupported."""
    monkeypatch.setattr(config, "VERIFY", True)
    seen = []
    real = verify.build_evidence
    monkeypatch.setattr(app.verify, "build_evidence",
                        lambda hits, notes: seen.append((list(hits), list(notes)))
                        or real(hits, notes))
    _stub_verify(monkeypatch, _result(GOOD_ANSWER, "verified"))
    client = FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER)
    _run_main(monkeypatch, "Which entity implements FP220 in 2023?", [], client)

    hits, notes = seen[0]
    assert [h.doc_id for h in hits] == [FP220]
    assert any(n.startswith("Registry —") for n in notes)          # registry note
    assert any("2023" in n and "computed" in n for n in notes)     # year note
    assert all(isinstance(n, str) and n for n in notes)


def test_the_verifier_receives_the_judge_switch_and_nothing_about_repair(
        monkeypatch, app_env):
    """use_llm is the only switch the app forwards. `allow_repair` is not
    passed at all — the parameter is gone from verify_answer, and passing a
    dead kwarg is how a deleted feature comes back as a TypeError."""
    monkeypatch.setattr(config, "VERIFY", True)
    monkeypatch.setattr(config, "VERIFY_LLM", False)
    monkeypatch.setenv("VERIFY_REPAIR", "1")          # inert, and proven so here
    record = []
    _stub_verify(monkeypatch, _result(GOOD_ANSWER, "verified"), record)
    _run_main(monkeypatch, Q220, [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))
    assert record[0]["use_llm"] is False
    assert "allow_repair" not in record[0]
    assert record[0]["answer"] == GOOD_ANSWER
    assert (FP220, 5) in record[0]["evidence"]        # the retrieved page itself


@pytest.mark.parametrize("status,verdicts", [
    ("verified", ()),
    ("partial", (_verdict("FP220 requests USD 58,000,000."),)),
    ("unverified-llm", (_verdict("FP220 requests USD 58,000,000."),)),
])
def test_a_failed_claim_never_edits_the_answer_or_the_history(
        monkeypatch, app_env, status, verdicts):
    """The post-eac4c94 point, and the inverse of the test that used to stand
    here: a wrong figure is REPORTED, never overwritten. Whatever the verdicts
    are, the user reads the sentence the model wrote and the next turn
    remembers that same sentence — no message update happens at all."""
    monkeypatch.setattr(config, "VERIFY", True)
    wrong = f"FP220 requests USD 58,000,000 from the GCF [{FP220}, p. 5]."
    _stub_verify(monkeypatch, _result(wrong, status, verdicts))
    session, _ = _run_main(monkeypatch, Q220, [],
                           FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=wrong))

    assert FakeMessage.updated == []                      # nothing was rewritten
    assert FakeMessage.sent[0] == wrong
    assert session.get("history")[-1] == {"role": "assistant", "content": wrong}
    assert "✎" not in _sources_line()


# ---------------------------------------------------------------------------
# C. degradation
# ---------------------------------------------------------------------------
def test_a_raising_verifier_leaves_the_answer_untouched(monkeypatch, app_env):
    monkeypatch.setattr(config, "VERIFY", True)

    def _boom(*a, **kw):
        raise RuntimeError("judge model exploded")

    monkeypatch.setattr(app.verify, "verify_answer", _boom)
    client = FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER)
    session, _ = _run_main(monkeypatch, Q220, [], client)

    assert FakeMessage.sent[0] == GOOD_ANSWER
    assert FakeMessage.updated == []
    assert session.get("history")[-1]["content"] == GOOD_ANSWER
    # and the turn degrades to the VERIFY=0 shape, sources included
    assert _sources_line() == f"📎 Sources: {FP220} p.5"
    assert not any("exploded" in m for m in FakeMessage.sent)


def test_a_raising_build_evidence_skips_verification(monkeypatch, app_env):
    monkeypatch.setattr(config, "VERIFY", True)
    monkeypatch.setattr(app.verify, "build_evidence",
                        lambda *a, **kw: (_ for _ in ()).throw(ValueError("nope")))
    monkeypatch.setattr(app.verify, "verify_answer",
                        lambda *a, **kw: pytest.fail("must not run without evidence"))
    _run_main(monkeypatch, Q220, [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))
    assert FakeMessage.sent == [GOOD_ANSWER, f"📎 Sources: {FP220} p.5"]


def test_a_failing_message_update_still_leaves_the_original(monkeypatch, app_env):
    """The abstain banner is the only thing that updates the message at all,
    and the update threw: the user must not be left with a half-applied
    message, and history must match what is on screen."""
    monkeypatch.setattr(config, "VERIFY", True)
    _stub_verify(monkeypatch, _result(GOOD_ANSWER, "abstain",
                                      [_verdict("FP220 covers 12 countries.")]))

    async def _bad_update(self):
        raise IOError("websocket closed")

    monkeypatch.setattr(FakeMessage, "update", _bad_update)
    session, _ = _run_main(monkeypatch, Q220, [],
                           FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))
    assert session.get("history")[-1]["content"] == GOOD_ANSWER
    assert "⚠️ Retrieval did not surface evidence" not in (_sources_line() or "")


# ---------------------------------------------------------------------------
# D. the UI, one status at a time
# ---------------------------------------------------------------------------
# 'repaired' is deliberately not in this table: the status is unreachable now
# (verify_answer never returns it), and the ✎ line it used to render is gone —
# the app must never claim a correction it cannot make.
@pytest.mark.parametrize("status,verdicts,expect", [
    ("verified", (), None),
    ("partial", (_verdict("FP220 covers 12 countries."),),
     "⚠️ not supported by the retrieved pages (treat with caution): "
     "FP220 covers 12 countries."),
    ("unverified-llm", (_verdict("FP220 covers 12 countries."),),
     "⚠️ claims could not be re-checked (no verification model available); "
     "the deterministic checks flag: FP220 covers 12 countries."),
])
def test_each_status_renders_its_own_line(monkeypatch, app_env, status, verdicts,
                                          expect):
    monkeypatch.setattr(config, "VERIFY", True)
    _stub_verify(monkeypatch, _result(GOOD_ANSWER, status, verdicts))
    _run_main(monkeypatch, Q220, [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))
    line = _sources_line()
    if expect is None:
        assert line == f"📎 Sources: {FP220} p.5"
    else:
        assert expect in line
    assert "✎" not in line


def test_a_contradicted_only_partial_still_names_its_claims(monkeypatch, app_env):
    """Review finding 1: res.unsupported is EMPTY when every failure is a
    CONTRADICTION, which printed a warning with nothing after the colon while
    the wrong figure sat on screen. The line reads res.failures."""
    monkeypatch.setattr(config, "VERIFY", True)
    bad = _verdict("FP220 requests USD 58,000,000.", verify.CONTRADICTED,
                   "the cited page prints USD 50,000,000")
    res = _result(GOOD_ANSWER, "partial", [bad])
    assert res.unsupported == []                      # the trap
    _stub_verify(monkeypatch, res)
    _run_main(monkeypatch, Q220, [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))
    assert ("⚠️ not supported by the retrieved pages (treat with caution): "
            "FP220 requests USD 58,000,000." in _sources_line())


def test_abstain_leads_the_answer_and_keeps_the_model_s_body(monkeypatch, app_env):
    """Review finding 2, and the one place the verifier touches the message.
    Every fact-bearing claim failed, so the warning goes ABOVE the answer
    instead of below the sources, where it read as a footnote to a
    confident-looking body. What it leads is the model's own text — the banner
    is a PREFIX, and nothing inside the answer is altered."""
    monkeypatch.setattr(config, "VERIFY", True)
    _stub_verify(monkeypatch, _result(
        GOOD_ANSWER, "abstain", [_verdict("FP220 requests USD 99,000,000.")]))
    session, _ = _run_main(monkeypatch, Q220, [],
                           FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))

    shown = FakeMessage.updated[-1]
    assert shown.startswith("⚠️ Retrieval did not surface evidence for this")
    assert "FP220 requests USD 99,000,000." in shown.splitlines()[0]   # the claim
    assert shown.endswith("\n\n" + GOOD_ANSWER)          # the model's own body
    assert shown == app._abstain_banner(
        _result(GOOD_ANSWER, "abstain",
                [_verdict("FP220 requests USD 99,000,000.")])) + "\n\n" + GOOD_ANSWER
    assert session.get("history")[-1]["content"] == shown
    # and it is not repeated as a footnote under the sources
    assert "⚠️" not in _sources_line()


def test_cautions_render_next_to_the_sources_line(monkeypatch, app_env):
    monkeypatch.setattr(config, "VERIFY", True)
    caution = _verdict("FP220 requests USD 50,000,000.", verify.SUPPORTED,
                       "value found in the cited document, but not on the cited page",
                       flags=["citation-page-mismatch"])
    _stub_verify(monkeypatch, _result(GOOD_ANSWER, "verified", [caution]))
    _run_main(monkeypatch, Q220, [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))
    line = _sources_line()
    assert "⚠️ citation cautions:" in line
    assert "citation-page-mismatch" in line
    assert "FP220 requests USD 50,000,000." in line


def test_many_failures_are_capped_and_counted(monkeypatch, app_env):
    monkeypatch.setattr(config, "VERIFY", True)
    verdicts = [_verdict(f"Claim number {i} states something.") for i in range(6)]
    _stub_verify(monkeypatch, _result(GOOD_ANSWER, "partial", verdicts))
    _run_main(monkeypatch, Q220, [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))
    assert "(+3 more)" in _sources_line()
    assert "Claim number 4" not in _sources_line()


def test_a_verdict_line_appears_even_without_retrieved_hits(monkeypatch, app_env):
    """No excerpts means no sources line to hang the warning on; the warning is
    exactly what a note-only answer needs.

    Review finding 4: it ships in the sources message's own shape. The leading
    📎 is what _history_from_thread skips, so a bare cl.Message would come back
    as an assistant turn when the thread is resumed.
    """
    monkeypatch.setattr(config, "VERIFY", True)
    _stub_verify(monkeypatch, _result(GOOD_ANSWER, "partial",
                                      [_verdict("FP220 covers 12 countries.")]))
    _run_main(monkeypatch, Q220, [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER),
              FakeRetriever([]))
    line = _sources_line()
    assert line.startswith("📎 Sources: none retrieved\n⚠️ not supported")
    assert not any(m.startswith("⚠️") for m in FakeMessage.sent)


def test_ui_messages_are_skipped_when_a_thread_is_resumed(monkeypatch, app_env):
    """The end of finding 4, checked against the replay filter itself."""
    monkeypatch.setattr(config, "VERIFY", True)
    _stub_verify(monkeypatch, _result(GOOD_ANSWER, "partial",
                                      [_verdict("FP220 covers 12 countries.")]))
    _run_main(monkeypatch, Q220, [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER),
              FakeRetriever([]))
    steps = [{"type": "assistant_message", "output": m, "createdAt": f"{i}"}
             for i, m in enumerate(FakeMessage.sent)]
    replayed = app._history_from_thread({"steps": steps})
    assert [m["content"] for m in replayed] == [GOOD_ANSWER]


def test_invalid_citations_still_run_under_verification(monkeypatch, app_env):
    """Review finding 3. The verifier only sees units that survive claim
    extraction; a refusal/hedge sentence is dropped as prose — carrying its
    invented page with it. _invalid_citations reads the raw answer and runs on
    every path, so the page is still flagged."""
    monkeypatch.setattr(config, "VERIFY", True)
    hedged = (f"FP220 requests USD 50,000,000 [{FP220}, p. 5].\n"
              f"The excerpts do not state the co-financing [{FP220}, p. 41].")
    assert not any("p. 41" in c.text for c in verify.extract_claims(hedged)), \
        "the hedge must be invisible to claim extraction for this test to mean anything"
    _stub_verify(monkeypatch, _result(hedged, "verified"))
    _run_main(monkeypatch, Q220, [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=hedged))
    line = _sources_line()
    assert "⚠️ cited but not among retrieved pages" in line
    assert "p.41" in line


def test_a_page_the_verifier_already_flagged_is_not_reported_twice(
        monkeypatch, app_env):
    """The merge half of finding 3: one defect, one line."""
    monkeypatch.setattr(config, "VERIFY", True)
    answer = f"FP220 covers 12 countries [{FP220}, p. 41]."
    flagged = _verdict("FP220 covers 12 countries.", verify.UNSUPPORTED,
                       "cited evidence was never retrieved",
                       flags=[f"invalid-citation:{FP220}, p.41"])
    _stub_verify(monkeypatch, _result(answer, "partial", [flagged]))
    _run_main(monkeypatch, Q220, [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=answer))
    line = _sources_line()
    assert "⚠️ not supported by the retrieved pages" in line
    assert "cited but not among retrieved pages" not in line
    assert line.count("p.41") == 0            # the claim text carries no page


def test_cite_key_matches_the_two_reporters_spellings():
    assert app._cite_key(f"{FP220[:34]}… p.41") == app._cite_key(f"{FP220}, p.41")
    assert app._cite_key(f"{FP220}, p.41")[1] == 41
    assert app._cite_key(f"{FP220}, p.41") != app._cite_key(f"{FP220}, p.5")


# ---------------------------------------------------------------------------
# E. the REAL verification pass, with only the OpenAI client mocked
# ---------------------------------------------------------------------------
class FakeVerifierClient:
    """The stand-in test_verify.py uses: replies in order, records calls."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(kw)
                content = outer.replies.pop(0) if outer.replies else ""
                msg = type("M", (), {"content": content})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        self.chat = type("Chat", (), {"completions": _Completions()})()


WRONG_ANSWER = (f"FP220 requests USD 58,000,000 from the GCF [{FP220}, p. 5].\n"
                f"The accredited entity is IFAD [{FP220}, p. 5].")
JUDGE_SAYS_UNSUPPORTED = json.dumps(
    {"verdicts": [{"id": 0, "status": "unsupported", "reason": "page states 50,000,000"}]})


def test_a_real_wrong_answer_is_flagged_and_left_exactly_as_written(
        monkeypatch, app_env):
    """End to end through verify.py itself — extract -> classify -> adjudicate
    -> report — with only the OpenAI client faked.

    This is where the repair used to run: the old wiring swapped a rewrite onto
    the screen and announced '✎ corrected'. None of that happens now. The wrong
    figure stays on screen, in history and in the warning under the sources —
    which is the only thing about the turn that verification changes.

    How many model calls the pass itself makes is verify.py's contract and is
    pinned in test_verify.py; what is pinned HERE is that no result the pass can
    return reaches the answer text, because the wiring never reads res.answer.
    """
    monkeypatch.setattr(config, "VERIFY", True)
    client = FakeVerifierClient(JUDGE_SAYS_UNSUPPORTED)
    monkeypatch.setattr(verify, "_client", lambda: client)
    session, _ = _run_main(monkeypatch, Q220, [],
                           FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=WRONG_ANSWER))

    assert FakeMessage.updated == []
    assert FakeMessage.sent[0] == WRONG_ANSWER
    assert session.get("history")[-1]["content"] == WRONG_ANSWER
    step = next(s for s in FakeStep.steps if s.name == "verification")
    assert not step.output.startswith("status: repaired")
    line = _sources_line()
    assert "⚠️" in line and "58,000,000" in line   # the claim, named as failing
    assert "✎" not in line                        # nothing was corrected


# ---------------------------------------------------------------------------
# D (cont.). sources + grounding follow the answer's own citations
# ---------------------------------------------------------------------------
MIXED_HITS = [
    Hit(text="FP254 background prose.", doc_id=FP254, score=0.9, page=3),
    Hit(text="FP248 requests 59,484,751 Eur.", doc_id=FP248, score=0.7, page=5),
    Hit(text="FP248 annex table.", doc_id=FP248, score=0.6, page=9),
]


def test_sources_ordering_follows_res_sources(monkeypatch, app_env):
    """res.sources — verify.cited_sources over the answer — is what orders and
    filters the sources line once verification ran: 'display only sources
    actually cited by the final verified answer'. The retrieved FP254 page the
    answer never cites is dropped, and the cited page is grounded first."""
    monkeypatch.setattr(config, "VERIFY", True)
    order = []
    monkeypatch.setattr(app, "ground_chunk",
                        lambda payload: order.append((payload["doc_id"], payload["page"]))
                        or None)
    answer = f"FP248 requests 59,484,751 Eur [{FP248}, p. 5]."
    _stub_verify(monkeypatch, _result(answer, "verified"))
    _run_main(monkeypatch, "Compare FP254 and FP248", [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=answer),
              FakeRetriever(MIXED_HITS))

    line = _sources_line()
    assert line == f"📎 Sources: {FP248} p.5, {FP248} p.9"
    assert FP254 not in line
    assert order[0] == (FP248, 5)


def test_the_exact_cited_page_leads_its_document(monkeypatch, app_env):
    monkeypatch.setattr(config, "VERIFY", True)
    order = []
    monkeypatch.setattr(app, "ground_chunk",
                        lambda payload: order.append((payload["doc_id"], payload["page"]))
                        or None)
    answer = f"FP248 annex [{FP248}, p. 9]."
    _stub_verify(monkeypatch, _result(answer, "verified"))
    _run_main(monkeypatch, "Compare FP254 and FP248", [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=answer),
              FakeRetriever(MIXED_HITS))
    assert order[0] == (FP248, 9)          # p.9, not p.5, though both are FP248
    assert _sources_line() == f"📎 Sources: {FP248} p.5, {FP248} p.9"


def test_verify_off_ordering_is_the_old_regex_over_the_answer(monkeypatch, app_env):
    monkeypatch.setattr(config, "VERIFY", False)
    order = []
    monkeypatch.setattr(app, "ground_chunk",
                        lambda payload: order.append((payload["doc_id"], payload["page"]))
                        or None)
    _run_main(monkeypatch, "Compare FP254 and FP248", [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE,
                         answer=f"FP248 requests 59,484,751 Eur [{FP248}, p. 5]."),
              FakeRetriever(MIXED_HITS))
    assert order[0] == (FP248, 5)
    assert _sources_line() == f"📎 Sources: {FP248} p.5, {FP248} p.9"


# ---------------------------------------------------------------------------
# F. the live path, pinned byte for byte
#
# Production runs VERIFY=1 with VERIFY_LLM at its shipped default of 1
# (fp-gcf:22f558b). Deleting the repair pathway must not move a single byte of
# what such a turn puts on screen, so these two pin the whole message stream
# literally — and they run with a stray VERIFY_REPAIR=1 in the environment,
# because that is what a production .env may still carry.
# ---------------------------------------------------------------------------
def _production_turn(monkeypatch, status, verdicts=()):
    monkeypatch.setattr(config, "VERIFY", True)
    assert config.VERIFY_LLM is True            # production leaves the judge on
    monkeypatch.setenv("VERIFY_REPAIR", "1")    # inert: nothing reads it
    _stub_verify(monkeypatch, _result(GOOD_ANSWER, status, verdicts))
    return _run_main(monkeypatch, Q220, [],
                     FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))


def test_the_live_path_of_a_verified_turn_is_unchanged(monkeypatch, app_env):
    session, _ = _production_turn(monkeypatch, "verified")
    assert FakeMessage.sent == [
        f"FP220 requests USD 50,000,000 from the GCF [{FP220}, p. 5].",
        f"📎 Sources: {FP220} p.5",
    ]
    assert FakeMessage.updated == []
    assert session.get("history")[-1] == {"role": "assistant",
                                          "content": GOOD_ANSWER}


def test_the_live_path_of_a_partial_turn_is_unchanged(monkeypatch, app_env):
    session, _ = _production_turn(
        monkeypatch, "partial", (_verdict("FP220 covers 12 countries."),))
    assert FakeMessage.sent == [
        f"FP220 requests USD 50,000,000 from the GCF [{FP220}, p. 5].",
        f"📎 Sources: {FP220} p.5"
        "\n⚠️ not supported by the retrieved pages (treat with caution): "
        "FP220 covers 12 countries.",
    ]
    assert FakeMessage.updated == []
    assert session.get("history")[-1] == {"role": "assistant",
                                          "content": GOOD_ANSWER}
