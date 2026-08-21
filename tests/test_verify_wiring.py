"""Plan step 6 — the step-5 claim verifier, wired into the app behind its own
switch (config.VERIFY, default OFF).

verify.py is tested in test_verify.py; what is tested here is the WIRING,
driven through the real main() with Chainlit I/O and the OpenAI client faked
out (the harness of test_step1_isolation.py / test_planner_wiring.py):

A. the switch. VERIFY=0 must not call the verifier at all — not even to build
   evidence — and must leave the turn byte-identical to today's.
B. the pass. Evidence is built from THIS turn's hits and the same computed
   note blocks the context received; a repair replaces the message content and
   the history entry; a clean answer shows nothing extra.
C. degradation. A verifier that raises leaves the original answer on screen.
D. the UI, one status at a time, plus the sources/grounding ordering, which
   follows the FINAL answer's citations (res.sources) once verification ran.
"""
import asyncio
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


def _result(answer, status, verdicts=(), original=None, notes=()):
    return verify.RepairResult(answer, status, list(verdicts),
                               original if original is not None else answer,
                               repaired=(status == "repaired"), notes=list(notes))


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
    assert config.VERIFY_LLM is True and config.VERIFY_REPAIR is True


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


def test_the_verifier_receives_the_switches(monkeypatch, app_env):
    monkeypatch.setattr(config, "VERIFY", True)
    monkeypatch.setattr(config, "VERIFY_LLM", False)
    monkeypatch.setattr(config, "VERIFY_REPAIR", False)
    record = []
    _stub_verify(monkeypatch, _result(GOOD_ANSWER, "verified"), record)
    _run_main(monkeypatch, Q220, [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))
    assert record[0]["use_llm"] is False and record[0]["allow_repair"] is False
    assert record[0]["answer"] == GOOD_ANSWER
    assert (FP220, 5) in record[0]["evidence"]        # the retrieved page itself


def test_a_repair_replaces_the_message_and_the_history_entry(monkeypatch, app_env):
    """The plan's point: what the user reads and what the next turn remembers
    are both the CORRECTED text, never the figure the model first wrote."""
    monkeypatch.setattr(config, "VERIFY", True)
    wrong = f"FP220 requests USD 58,000,000 from the GCF [{FP220}, p. 5]."
    fixed = f"FP220 requests USD 50,000,000 from the GCF [{FP220}, p. 5]."
    _stub_verify(monkeypatch, _result(fixed, "repaired", original=wrong))
    client = FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=wrong)
    session, _ = _run_main(monkeypatch, Q220, [], client)

    assert FakeMessage.updated == [fixed]                 # the message was updated
    assert session.get("history")[-1] == {"role": "assistant", "content": fixed}
    assert "58,000,000" not in session.get("history")[-1]["content"]
    assert "✎ answer corrected against the cited pages" in _sources_line()


def test_history_keeps_the_original_when_nothing_was_repaired(monkeypatch, app_env):
    monkeypatch.setattr(config, "VERIFY", True)
    _stub_verify(monkeypatch, _result(GOOD_ANSWER, "partial",
                                      [_verdict("FP220 covers 12 countries.")]))
    session, _ = _run_main(monkeypatch, Q220, [],
                           FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))
    assert FakeMessage.updated == []
    assert session.get("history")[-1]["content"] == GOOD_ANSWER


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
    """The repair landed but the UI update threw: the user must not be left
    with a half-applied answer."""
    monkeypatch.setattr(config, "VERIFY", True)
    fixed = f"FP220 requests USD 50,000,000 [{FP220}, p. 5]."
    _stub_verify(monkeypatch, _result(fixed, "repaired", original=GOOD_ANSWER))

    async def _bad_update(self):
        raise IOError("websocket closed")

    monkeypatch.setattr(FakeMessage, "update", _bad_update)
    session, _ = _run_main(monkeypatch, Q220, [],
                           FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))
    assert session.get("history")[-1]["content"] == GOOD_ANSWER
    assert "✎" not in (_sources_line() or "")


# ---------------------------------------------------------------------------
# D. the UI, one status at a time
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("status,verdicts,expect,absent", [
    ("verified", (), None, "⚠️"),
    ("repaired", (), "✎ answer corrected against the cited pages", "⚠️"),
    ("partial", (_verdict("FP220 covers 12 countries."),),
     "⚠️ not supported by the retrieved pages (treat with caution): "
     "FP220 covers 12 countries.", "✎"),
    ("unverified-llm", (_verdict("FP220 covers 12 countries."),),
     "⚠️ claims could not be re-checked (no verification model available); "
     "the deterministic checks flag: FP220 covers 12 countries.", "✎"),
])
def test_each_status_renders_its_own_line(monkeypatch, app_env, status, verdicts,
                                          expect, absent):
    monkeypatch.setattr(config, "VERIFY", True)
    answer = GOOD_ANSWER if status != "repaired" else GOOD_ANSWER + " (corrected)"
    _stub_verify(monkeypatch, _result(answer, status, verdicts, original=GOOD_ANSWER))
    _run_main(monkeypatch, Q220, [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))
    line = _sources_line()
    if expect is None:
        assert line == f"📎 Sources: {FP220} p.5"
    else:
        assert expect in line
    assert absent not in line


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


def test_abstain_leads_the_answer_and_keeps_the_original_body(monkeypatch, app_env):
    """Review finding 2. Every fact-bearing claim failed, so: the body stays
    the model's own text (a repair of an all-failed answer stands on nothing),
    and the warning goes ABOVE it instead of below the sources, where it read
    as a footnote to a confident-looking answer."""
    monkeypatch.setattr(config, "VERIFY", True)
    rewritten = f"FP220 requests USD 99,000,000 [{FP220}, p. 5]."
    _stub_verify(monkeypatch, _result(
        rewritten, "abstain", [_verdict("FP220 requests USD 99,000,000.")],
        original=GOOD_ANSWER))
    session, _ = _run_main(monkeypatch, Q220, [],
                           FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=GOOD_ANSWER))

    shown = FakeMessage.updated[-1]
    assert shown.startswith("⚠️ Retrieval did not surface evidence for this")
    assert "FP220 requests USD 99,000,000." in shown.splitlines()[0]   # the claim
    assert shown.endswith("\n\n" + GOOD_ANSWER)          # the ORIGINAL body
    assert "99,000,000 [" not in shown                   # never the rewrite
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
# E. the REAL repair path, with only the OpenAI client mocked
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


def test_a_real_repair_that_lands_shows_the_corrected_text(monkeypatch, app_env):
    """End to end through verify.py itself: extract -> classify -> adjudicate
    -> repair -> re-verify, with only the OpenAI client faked."""
    monkeypatch.setattr(config, "VERIFY", True)
    fixed = (f"FP220 requests USD 50,000,000 from the GCF [{FP220}, p. 5].\n"
             f"The accredited entity is IFAD [{FP220}, p. 5].")
    client = FakeVerifierClient(JUDGE_SAYS_UNSUPPORTED, fixed)
    monkeypatch.setattr(verify, "_client", lambda: client)
    session, _ = _run_main(monkeypatch, Q220, [],
                           FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=WRONG_ANSWER))

    assert len(client.calls) == 2                       # adjudicate + repair
    assert FakeMessage.updated == [fixed]
    assert session.get("history")[-1]["content"] == fixed
    assert "58,000,000" not in session.get("history")[-1]["content"]
    assert "✎ answer corrected against the cited pages" in _sources_line()


def test_a_real_repair_that_stays_wrong_ends_partial_and_flagged(monkeypatch, app_env):
    """Review finding 20, through the real repair pass: the rewrite still fails
    verification. verify.py's repair gate decides whether that rewrite is kept;
    whatever it returns as res.answer is what the app shows, and the warning
    names res.failures. Pinned here on the integration as it stands — the
    rewrite is rejected, so the model's own text is what the user reads."""
    monkeypatch.setattr(config, "VERIFY", True)
    still_wrong = (f"FP220 requests USD 61,000,000 from the GCF [{FP220}, p. 5].\n"
                   f"The accredited entity is IFAD [{FP220}, p. 5].")
    client = FakeVerifierClient(JUDGE_SAYS_UNSUPPORTED, still_wrong)
    monkeypatch.setattr(verify, "_client", lambda: client)
    session, _ = _run_main(monkeypatch, Q220, [],
                           FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=WRONG_ANSWER))

    assert len(client.calls) == 2                       # adjudicate + repair
    step = next(s for s in FakeStep.steps if s.name == "verification")
    assert step.output.startswith("status: partial")
    shown = (FakeMessage.updated or [WRONG_ANSWER])[-1]
    assert shown == session.get("history")[-1]["content"]      # screen == memory
    line = _sources_line()
    assert "⚠️ not supported by the retrieved pages (treat with caution): " in line
    assert "USD 58,000,000" in line                     # the surviving claim
    assert "✎" not in line                              # nothing was corrected


def test_a_rewritten_partial_is_shown_and_still_flagged(monkeypatch, app_env):
    """The other half of finding 20, at the wiring level: a RepairResult whose
    status is 'partial' AND whose answer differs from the original. The public
    API permits it (repair() returns its rewrite with the re-verified status),
    so the wiring must both DISPLAY the rewrite and warn about what still
    fails — in the rewrite's own wording, never the original's."""
    monkeypatch.setattr(config, "VERIFY", True)
    rewritten = (f"FP220 requests USD 61,000,000 from the GCF [{FP220}, p. 5].\n"
                 f"The accredited entity is IFAD [{FP220}, p. 5].")
    _stub_verify(monkeypatch, _result(
        rewritten, "partial",
        [_verdict("FP220 requests USD 61,000,000 from the GCF.")],
        original=WRONG_ANSWER))
    session, _ = _run_main(monkeypatch, Q220, [],
                           FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=WRONG_ANSWER))

    assert FakeMessage.updated == [rewritten]           # the rewrite is on screen
    assert session.get("history")[-1]["content"] == rewritten
    line = _sources_line()
    assert "USD 61,000,000" in line                     # the REWRITE's figure
    assert "58,000,000" not in line                     # not the original's
    assert "✎" not in line


def test_a_real_repair_that_invents_a_source_is_rejected(monkeypatch, app_env):
    """verify.py drops a repair that introduces a citation; the app must then
    show the ORIGINAL answer, flagged — not the invented one."""
    monkeypatch.setattr(config, "VERIFY", True)
    invented = f"FP220 requests USD 50,000,000 [{FP220}, p. 404]."
    client = FakeVerifierClient(JUDGE_SAYS_UNSUPPORTED, invented)
    monkeypatch.setattr(verify, "_client", lambda: client)
    session, _ = _run_main(monkeypatch, Q220, [],
                           FakeOpenAI(conductor_json=CONDUCTOR_ONE, answer=WRONG_ANSWER))

    assert FakeMessage.updated == []
    assert session.get("history")[-1]["content"] == WRONG_ANSWER
    assert "p. 404" not in "".join(FakeMessage.sent)
    assert "⚠️" in _sources_line()


# ---------------------------------------------------------------------------
# D. sources + grounding follow the FINAL answer's citations
# ---------------------------------------------------------------------------
MIXED_HITS = [
    Hit(text="FP254 background prose.", doc_id=FP254, score=0.9, page=3),
    Hit(text="FP248 requests 59,484,751 Eur.", doc_id=FP248, score=0.7, page=5),
    Hit(text="FP248 annex table.", doc_id=FP248, score=0.6, page=9),
]


def test_sources_ordering_follows_res_sources(monkeypatch, app_env):
    """The answer TEXT cites FP254; the verified result cites FP248 p.5 (this
    is what a repair does). res.sources wins — that is the plan's 'display only
    sources actually cited by the final verified answer'."""
    monkeypatch.setattr(config, "VERIFY", True)
    order = []
    monkeypatch.setattr(app, "ground_chunk",
                        lambda payload: order.append((payload["doc_id"], payload["page"]))
                        or None)
    repaired = f"FP248 requests 59,484,751 Eur [{FP248}, p. 5]."
    _stub_verify(monkeypatch, _result(repaired, "repaired",
                                      original=f"FP254 is bigger [{FP254}, p. 3]."))
    _run_main(monkeypatch, "Compare FP254 and FP248", [],
              FakeOpenAI(conductor_json=CONDUCTOR_ONE,
                         answer=f"FP254 is bigger [{FP254}, p. 3]."),
              FakeRetriever(MIXED_HITS))

    line = _sources_line()
    # the FP254 hit the ORIGINAL answer cited is gone; the repaired answer's
    # document leads, and its cited page is grounded first
    assert line == (f"📎 Sources: {FP248} p.5, {FP248} p.9"
                    "\n✎ answer corrected against the cited pages")
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
