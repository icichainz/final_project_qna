"""The trigger that decides WHICH documents get a registry note.

`registry.registry_note()` reads identifiers out of the question's own text.
That is the whole trigger, and a follow-up defeats it: release-3's
fu-lang-switch asked "Et quelle entité accréditée le met en œuvre ?" of a
thread about FP173. The conductor resolved the referent, the query it produced
named FP173, retrieval returned FP173's package — and the turn reached the
answer model with `notes_used` EMPTY, so the model, obeying cite-or-hedge,
declined to name the accredited entity the registry holds (Inter-American
Development Bank). The registry knew; nothing had put it in front of the
model.

So the trigger widens, in the wiring, from "the question names the document"
to "this turn resolved to the document" — the doc tags that survived the
rewrite guards, and the identifiers inside the resolved sub-queries. What is
pinned here:

A. the widened trigger, through the real main(): a doc tag fires it, and so
   does an identifier that exists only in the conductor's rewrite (the
   fu-lang-switch shape, where the guards strip the lone query's tag).
B. the narrowness. A turn that resolved to nothing produces the note it
   produces today, byte for byte, and a document the question already named
   gets exactly ONE line.
C. isolation. The line comes from THIS turn's resolved items — never from the
   conversation, and never from an earlier answer's prose.
D. degradation, and the four-document cap.
E. parity: chainlit_app.main() and the eval harness's Pipeline emit the same
   note for the same turn.
F. inheritance: the line this path builds comes from registry._fmt, so schema
   2's cover-page provenance reaches a follow-up turn with no wiring of its
   own — and the page it prints is one the app's citation gate credits.
G. the recorded case itself, replayed offline from release-3's own plan.
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
from gcf_qna.app import chainlit_app as app  # noqa: E402
from gcf_qna.rag import registry  # noqa: E402
from gcf_qna.rag import verify  # noqa: E402
from gcf_qna.rag.retrieve import Hit  # noqa: E402

FP151 = "124_gcf-b27-02-add11"
FP152 = "123_gcf-b27-02-add12"
FP173 = "102_gcf-b30-02-add05"
FP220 = "55_gcf-b37-02-add11-funding-proposal-package-fp220"
FP248 = "28_gcf-b40-02-add10-rev01-funding-proposal-package-fp248"
FP254 = "22_gcf-b40-02-add16-rev01-funding-proposal-package-fp254"

FAKE_REGISTRY = {
    FP151: {"fp": 151, "board": 27, "year": 2020, "accredited_entity": "IUCN",
            "gcf_financing": "18.5 M USD"},
    FP152: {"fp": 152, "board": 27, "year": 2020, "accredited_entity": "Pegasus",
            "gcf_financing": "150 M USD"},
    FP173: {"fp": 173, "board": 30, "year": 2021,
            "title": "The Amazon Bioeconomy Fund",
            "accredited_entity": "Inter-American Development Bank"},
    FP220: {"fp": 220, "board": 37, "year": 2023, "accredited_entity": "IFAD"},
    FP248: {"fp": 248, "board": 40, "year": 2024, "accredited_entity": "GIZ"},
    FP254: {"fp": 254, "board": 40, "year": 2024, "accredited_entity": "IFC"},
}


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(registry, "_cache", FAKE_REGISTRY)
    monkeypatch.setattr(registry, "_cache_v2", {})
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
    def answer_call(self):
        return next(c for c in self.calls if c.get("stream"))


def _now(fn, *a, **kw):
    async def run():
        return fn(*a, **kw)
    return run()


HITS = [Hit(text="The programme is described on the cover page.",
            doc_id=FP173, score=0.8, page=1)]


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
    session = FakeSession(retriever=FakeRetriever(HITS), history=list(history))
    client = FakeOpenAI(conductor_json=conductor_json)
    monkeypatch.setattr(app.cl, "user_session", session)
    monkeypatch.setitem(sys.modules, "openai",
                        types.SimpleNamespace(AsyncOpenAI=lambda **kw: client))
    asyncio.run(app.main(FakeMessage(content=question)))
    return client


def _body(client):
    return client.answer_call["messages"][1]["content"]


def _context(client):
    return _body(client).split("Context excerpts:\n", 1)[1]


def _registry_lines(client):
    return [ln for ln in _context(client).splitlines()
            if ln.startswith("Registry — ")]


# the turn release-3 recorded, verbatim (data/eval/release_release-3.jsonl,
# id fu-lang-switch): a French follow-up naming no identifier, and the plan the
# guards left it with — the conductor's tag did not survive a lone query whose
# message names no document, so the FP id survives only in the query text.
FU_QUESTION = "Et quelle entité accréditée le met en œuvre ?"
FU_PLAN = [{"q": "FP173 Amazon Bioeconomy Fund accredited entity "
                 "implementing entity", "doc": None}]
FU_HISTORY = [
    {"role": "user", "content": "Tell me about the Amazon Bioeconomy Fund."},
    {"role": "assistant",
     "content": f"The Amazon Bioeconomy Fund is FP173 [{FP173}, p. 1]."},
]


# ---------------------------------------------------------------------------
# A. the widened trigger
# ---------------------------------------------------------------------------
def test_a_followup_resolved_only_in_the_query_gets_its_registry_line(
        monkeypatch, app_env):
    """The measured defect, end to end: the question spells no identifier, the
    conductor's rewrite does, and the note now states what the model was
    hedging about."""
    client = _run(monkeypatch, FU_QUESTION,
                  {"mode": "retrieve", "queries": FU_PLAN}, FU_HISTORY)
    lines = _registry_lines(client)
    assert len(lines) == 1 and lines[0].startswith("Registry — FP173:")
    assert "Inter-American Development Bank" in lines[0]
    assert f"[{FP173}, cover pages]" in lines[0]


def test_a_surviving_doc_tag_fires_the_same_note(monkeypatch, app_env):
    """The other half of "the turn resolved to it": a conductor tag the guards
    kept (here a fan-out leg, where history may pin the document)."""
    client = _run(monkeypatch, "And how do their entities compare?",
                  {"mode": "retrieve", "queries": [
                      {"q": "accredited entity", "doc": FP151},
                      {"q": "accredited entity", "doc": FP152}]},
                  [{"role": "user", "content": "What are FP151 and FP152?"},
                   {"role": "assistant",
                    "content": f"[{FP151}, p. 5] and [{FP152}, p. 5]."}])
    lines = _registry_lines(client)
    assert [ln.split(":")[0] for ln in lines] == ["Registry — FP151",
                                                  "Registry — FP152"]


def test_the_registry_rules_ship_with_the_new_note(monkeypatch, app_env):
    """`assemble(registry=...)` follows the note, so a turn that only now has
    one must also get the rules that tell the model how to read it."""
    client = _run(monkeypatch, FU_QUESTION,
                  {"mode": "retrieve", "queries": FU_PLAN}, FU_HISTORY)
    system = client.answer_call["messages"][0]["content"]
    plain = _run(monkeypatch, "What does the corpus say about mangroves?",
                 {"mode": "retrieve", "queries": [
                     {"q": "mangrove restoration", "doc": None}]})
    assert system != plain.answer_call["messages"][0]["content"]


# ---------------------------------------------------------------------------
# B. narrowness: no tag, no change — and never two lines for one document
# ---------------------------------------------------------------------------
def test_an_untagged_turn_is_left_byte_identical(monkeypatch, app_env):
    """A general question resolves to no document, so the note it gets is the
    note it got before this wiring existed — the SAME object, unmodified."""
    items = [{"q": "which projects restore mangroves?", "doc": None}]
    note = registry.registry_note("Which projects restore mangroves?")
    assert note is None
    assert app._turn_doc_ids(items) == []
    assert app._extend_registry_note(note, items) is note
    client = _run(monkeypatch, "Which projects restore mangroves?",
                  {"mode": "retrieve", "queries": items})
    assert _registry_lines(client) == []


def test_an_untagged_turn_keeps_a_year_note_untouched(fake_registry):
    """The question-text trigger still owns everything it always did: a year
    listing has no document to widen to, and must come back unchanged."""
    q = "Which proposals were approved in 2020?"
    note = registry.registry_note(q)
    assert note and "2020" in note
    assert app._extend_registry_note(note, [{"q": q, "doc": None}]) is note


def test_a_named_document_that_is_also_tagged_gets_exactly_one_line(
        monkeypatch, app_env):
    """FP151 named in the question AND tagged by the pre-scope: the question
    trigger emits the line, the turn trigger must recognise it as already
    emitted."""
    q = "What is the GCF financing of FP151?"
    client = _run(monkeypatch, q, {"mode": "retrieve", "queries": [
        {"q": "GCF financing of FP151", "doc": "fp151"}]})
    assert len([ln for ln in _registry_lines(client)
                if ln.startswith("Registry — FP151")]) == 1


def test_dedup_holds_when_the_tag_is_the_stem_and_the_question_the_number(
        fake_registry):
    """The two triggers spell the same document differently ('FP151' in the
    question, the corpus stem in the tag); dedup is on the resolved stem."""
    note = registry.registry_note("What is FP151's GCF financing?")
    out = app._extend_registry_note(note, [{"q": "GCF financing", "doc": FP151}])
    assert out is note


# ---------------------------------------------------------------------------
# C. isolation — this turn's resolved items, not the conversation
# ---------------------------------------------------------------------------
def test_the_note_never_comes_from_the_conversation(monkeypatch, app_env):
    """FP152 is all over the history and nowhere in this turn's plan: no line
    for it, and none of the earlier answer's prose in the body either.

    The plan is the lone-query shape a follow-up really has — the guards strip
    a lone query's doc tag unless the message itself names that document, so
    what the turn resolved to survives in the rewrite, and that is what the
    note is allowed to read."""
    history = [
        {"role": "user", "content": "What are FP151 and FP152?"},
        {"role": "assistant",
         "content": f"FP152 is implemented by Pegasus [{FP152}, p. 5]."},
    ]
    client = _run(monkeypatch, "And which entity implements the first one?",
                  {"mode": "retrieve", "queries": [
                      {"q": "FP151 accredited entity", "doc": FP151}]}, history)
    lines = _registry_lines(client)
    assert len(lines) == 1 and lines[0].startswith("Registry — FP151")
    assert "Pegasus" not in _body(client) and FP152 not in _body(client)


def test_a_lone_querys_history_tag_is_gone_before_the_note_is_built(
        monkeypatch, app_env):
    """The guards' verdict is the note's input. A lone query pinned to a
    document the message never names loses its tag in _rescope_items, and the
    widened trigger must not hand that scope back."""
    client = _run(monkeypatch, "And which entity implements the other one?",
                  {"mode": "retrieve", "queries": [
                      {"q": "accredited entity", "doc": FP152}]},
                  [{"role": "user", "content": "What is FP152?"},
                   {"role": "assistant", "content": f"[{FP152}, p. 5]."}])
    assert _registry_lines(client) == []


def test_turn_doc_ids_reads_items_only(fake_registry):
    """The function's whole input is the plan: no question, no history."""
    assert app._turn_doc_ids([{"q": "accredited entity", "doc": "fp151"}]) == [FP151]
    assert app._turn_doc_ids([{"q": "FP151 vs FP152", "doc": None}]) == [FP151, FP152]
    assert app._turn_doc_ids([]) == [] and app._turn_doc_ids(None) == []


# ---------------------------------------------------------------------------
# D. degradation and the cap
# ---------------------------------------------------------------------------
def test_an_identifier_that_resolves_nowhere_adds_nothing(fake_registry):
    """A rewrite is not the user's claim that a document exists, so it never
    earns a NOT FOUND line — that belongs to the question's own words."""
    items = [{"q": "FP999 accredited entity", "doc": None}]
    assert app._turn_doc_ids(items) == []
    assert app._extend_registry_note(None, items) is None


def test_a_missing_registry_leaves_the_note_alone(monkeypatch):
    monkeypatch.setattr(registry, "_cache", {})
    assert app._turn_doc_ids([{"q": "FP151", "doc": FP151}]) == []
    assert app._extend_registry_note("NOTE", [{"q": "FP151", "doc": FP151}]) == "NOTE"


def test_a_raising_registry_leaves_the_note_alone(monkeypatch):
    def boom():
        raise RuntimeError("registry.json is half-written")
    monkeypatch.setattr(registry, "load", boom)
    note = "Registry — something already computed"
    assert app._extend_registry_note(note, [{"q": "FP151", "doc": FP151}]) is note


def test_at_most_four_documents_are_added(fake_registry):
    """A registry line is a paragraph per document; a wide fan-out would bury
    the excerpts. Same cap, same reason, as _resolved_refs_note."""
    items = [{"q": "x", "doc": d} for d in
             (FP151, FP152, FP173, FP220, FP248, FP254)]
    out = app._extend_registry_note(None, items)
    assert len(out.splitlines()) == app._MAX_TURN_NOTE_DOCS == 4


# ---------------------------------------------------------------------------
# E. parity — the app and the harness build the same note for the same turn
# ---------------------------------------------------------------------------
def _harness_pipe(items, hits=HITS):
    """A duck-typed Pipeline: the real run(), the stages around it stubbed."""
    return types.SimpleNamespace(
        app=app, conductor=True,
        planner_plan=lambda q: (None, None),
        conduct=lambda q, turns=(): ("retrieve", [dict(i) for i in items], None),
        fp_guard=lambda q: None,
        _decomposed=lambda it, q, plan=None: len(it) > 1,
        _retrieve=lambda it, dec, original=None: (list(hits), 0.9, False),
    )


def test_the_app_and_the_harness_emit_the_same_note(monkeypatch, app_env):
    """Production parity is a hard requirement: the harness runs the app's own
    function at the app's own point in the turn, so the recorded
    `notes_used.registry` is the string production would have shipped."""
    client = _run(monkeypatch, FU_QUESTION,
                  {"mode": "retrieve", "queries": FU_PLAN}, FU_HISTORY)
    out = ev.Pipeline.run(_harness_pipe(FU_PLAN), FU_QUESTION)
    note = out["notes"]["registry"]
    assert note and "Inter-American Development Bank" in note
    assert note.splitlines() == _registry_lines(client)
    assert out["context"].startswith(note + "\n\n")


def test_the_harness_leaves_an_untagged_turn_alone(monkeypatch, app_env):
    items = [{"q": "which projects restore mangroves?", "doc": None}]
    out = ev.Pipeline.run(_harness_pipe(items), "Which projects restore mangroves?")
    assert out["notes"]["registry"] is None


# ---------------------------------------------------------------------------
# F. the widened trigger inherits the cover-page provenance
# ---------------------------------------------------------------------------
# `_extend_registry_note` calls `registry._fmt` — the emitter, not a copy of
# its output — so schema 2's optional `meta_provenance` reaches the follow-up
# path with no wiring of its own. What is pinned is that the inherited pointer
# is CREDITABLE: the line this path builds is the only registry line a
# follow-up turn gets, and release-6's flagged '[doc, p. 3]' was written for
# exactly this kind of turn.
FP173_PROV = {FP173: {"fp": 173, "meta_provenance": {
    "title": {"page": 1, "quote": "The Amazon Bioeconomy Fund"},
    "accredited_entity": {"page": 3,
                          "quote": "Inter-American Development Bank"}}}}


def test_the_extended_note_inherits_pages_and_they_are_page_creditable(
        monkeypatch, fake_registry):
    """The whole point of the change, on the path that has no note at all until
    the turn's own plan builds one: the entity now arrives with the page it was
    read on, and the app's own citation gate accepts a bracket citing it."""
    monkeypatch.setattr(registry, "_cache_v2", FP173_PROV)
    before = registry.registry_note(FU_QUESTION)
    after = app._extend_registry_note(before, FU_PLAN)
    assert before is None, "the French question spells no identifier"
    (line,) = after.splitlines()
    assert "accredited entity: Inter-American Development Bank (p.3)" in line
    assert '"The Amazon Bioeconomy Fund" (p.1)' in line
    assert line.endswith(f"[{FP173}, cover pages]")
    # the two readers that decide whether a cited page is legal
    assert app._note_pages([after]) == {(FP173, 1), (FP173, 3)}
    assert (verify.note_scope_doc(FP173), 3) in verify.note_page_scopes(line)
    # release-6 wrote this bracket from a GUESS and was flagged; retrieval
    # still never returned page 3, and now it passes because the note prints it
    hits = [Hit(text="x", doc_id=FP173, score=0.8, page=p) for p in (1, 2, 4, 5)]
    cited = f"The accredited entity is the IDB. [{FP173}, p. 3]"
    assert app._invalid_citations(cited, hits, app._note_pages([after])) == []
    assert app._invalid_citations(cited, hits, frozenset()) == [f"{FP173[:34]}… p.3"]


def test_the_extended_note_without_provenance_is_byte_identical(monkeypatch,
                                                                fake_registry):
    """The absent arm on the same path — their file may land after ours, and a
    document whose provenance the builder never found stays page-less forever.
    Byte for byte the line this suite already pinned."""
    monkeypatch.setattr(registry, "_cache_v2", {})
    plain = app._extend_registry_note(None, FU_PLAN)
    assert plain == (
        'Registry — FP173: "The Amazon Bioeconomy Fund"; accredited entity: '
        f"Inter-American Development Bank; board B.30, 2021 "
        f"[{FP173}, cover pages]")
    assert "(p." not in plain
    monkeypatch.setattr(registry, "_cache_v2", FP173_PROV)
    assert app._extend_registry_note(None, FU_PLAN) == plain.replace(
        'Fund";', 'Fund" (p.1);').replace(
        "Development Bank;", "Development Bank (p.3);")


def test_a_v2_row_whose_provenance_is_junk_leaves_the_extended_note_alone(
        monkeypatch, fake_registry):
    """`_extend_registry_note` swallows any exception and returns the note
    unchanged, so a broken row must not cost the turn its registry line at
    all — the degradation is the pointer, never the fact."""
    monkeypatch.setattr(registry, "_cache_v2",
                        {FP173: {"fp": 173, "meta_provenance": "p.3"}})
    out = app._extend_registry_note(None, FU_PLAN)
    assert out and "accredited entity: Inter-American Development Bank;" in out
    assert "(p." not in out


# ---------------------------------------------------------------------------
# G. the recorded case, against the real registry
# ---------------------------------------------------------------------------
def test_the_recorded_release_3_turn_now_carries_its_registry_line():
    """fu-lang-switch replayed offline from its own recorded plan, on the
    shipped registry: `notes_used` was {} and the answer hedged; the note it
    should have had names the entity."""
    before = registry.registry_note(FU_QUESTION)
    after = app._extend_registry_note(before, FU_PLAN)
    assert before is None, "the question's own words name no document"
    assert after and after.splitlines()[0].startswith("Registry — FP173:")
    assert "accredited entity: Inter-American Development Bank" in after
    assert f"[{FP173}, cover pages]" in after
