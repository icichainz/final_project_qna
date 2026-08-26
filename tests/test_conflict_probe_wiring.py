r"""Wiring `Retriever.probe_pages` into the turn: the conflict fallback probe.

The mechanism has been proven and unwired since `8c0bedc`. `registry.
_conflict_lines` tells the model exactly where a document contradicts itself —
'is printed as 28,654 million USD (p.5, A.8); also as 26,654 million USD
(p.48, B.2(b)) — report both figures with their pages' — and then retrieval
does not bring page 48 back: a second printing deep in a component table loses
the similarity contest to the cover page that says the same thing in the
question's own words. Conflict-class evidence pages sat at 8/14 for five
consecutive releases, three cases at 0/2, and `probe_pages` recovers 18/18 of
them on demand. Nothing asked.

What is pinned here:

A. the trigger, through the real `main()`: a CONFLICT line whose pages are not
   in this turn's hits fetches exactly those pages, and they reach the context.
B. narrowness, in both directions — no conflict line, a conflict line whose
   pages retrieval already returned, a main registry line that prints pages,
   and a document named on somebody else's line: none of them probe.
C. the caps: at most four pages asked, at most four excerpts appended, and a
   second document's pages cannot be starved by the first document's chunks.
D. degradation: a raise, an old retriever with no `probe_pages`, no retriever
   at all, an empty result — each leaves the turn byte-identical to today's.
E. guard and chat turns never probe.
F. placement: the supplements ride in FRONT of the ranked hits and evict none
   of them — the 15-hit cap is extended by the probe's count, not spent on it.
G. parity: `chainlit_app.main()` and the harness `Pipeline.run` build the same
   context, byte for byte, marker included.
H. the note-reading patterns cannot drift from the two that already decide
   which cited pages are legal.
I. the recorded turns: release-12's nine conflict-class cases, replayed from
   their own records, ask for exactly the pages they were missing.
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

FP153 = "122_gcf-b27-02-add13"
FP274 = "02_gcf-b42-02-add16-funding-proposal-package-fp274"

FAKE_REGISTRY = {
    FP153: {"fp": 153, "board": 27, "year": 2020, "accredited_entity": "XacBank",
            "gcf_financing": "28,654 million USD"},
    FP274: {"fp": 274, "board": 42, "year": 2025, "accredited_entity": "CAF"},
}
# schema 2 shaped exactly like data/registry_v2.json's own rows: the note under
# test is the real `_conflict_lines` output, not a string this file invented.
FAKE_V2 = {
    FP153: {"fp": 153, "facts": {"gcf_funding_requested": [
        {"raw": "28,654 million USD", "value": None, "currency": "USD",
         "unit": None, "page": 5, "section": "A.8", "status": "canonical"},
        {"raw": "26,654 million USD", "value": None, "currency": "USD",
         "unit": None, "page": 48, "section": "rule:B.2(b)",
         "status": "conflicting"}]}},
    FP274: {"fp": 274, "facts": {"gcf_funding_requested": [
        {"raw": "49,751,264 USD", "value": None, "currency": "USD",
         "unit": None, "page": 8, "section": "A.10", "status": "canonical"},
        {"raw": "40,751,254 USD", "value": None, "currency": "USD",
         "unit": None, "page": 40, "section": "rule:C.1",
         "status": "conflicting"}]}},
}

QUESTION = "How much GCF funding does FP153 request?"


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(registry, "_cache", FAKE_REGISTRY)
    monkeypatch.setattr(registry, "_cache_v2", FAKE_V2)
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


def _page_hit(doc, page, n=0, score=0.0, text=None):
    return Hit(text=text or f"{doc} page {page} chunk {n}", doc_id=doc,
               score=score, page=page, chunk_index=(page * 10 + n))


class FakeRetriever:
    """search_with_confidence + probe_pages, both recording their calls.

    `pages` is {(doc, page): [chunk, ...]} — what the index holds — and the
    probe half honours `probe_pages`'s two rules the wiring depends on: one
    slot per asked page before any page takes a second, and never a page that
    was not asked for.
    """

    def __init__(self, hits, pages=None, boom=False):
        self.hits = hits
        self.pages = pages or {}
        self.boom = boom
        self.calls = []
        self.probes = []

    def search_with_confidence(self, query, top_k=10, doc_filter=None,
                               original=None):
        self.calls.append({"q": query, "doc": doc_filter})
        return list(self.hits), 0.9

    def probe_pages(self, doc_id, pages=(), k=4, query=None, sections=()):
        self.probes.append({"doc": doc_id, "pages": list(pages), "k": k,
                            "query": query})
        if self.boom:
            raise RuntimeError("index cannot serve that page")
        rounds, out = [], []
        for nth in range(4):
            for p in pages:
                chunks = self.pages.get((doc_id, p), [])
                if nth < len(chunks):
                    rounds.append(chunks[nth])
        for h in rounds[:max(0, int(k))]:
            out.append(h)
        return out


class OldRetriever(FakeRetriever):
    """A retriever from before the mechanism existed."""
    probe_pages = None

    def __getattribute__(self, name):
        if name == "probe_pages":
            raise AttributeError("probe_pages")
        return object.__getattribute__(self, name)


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


# retrieval that misses BOTH printed pages — the measured shape of
# conf-fp153-gcf, whose ten hits are pages 36, 68, 35, 1, 81, 29, 30, 6, 14, 44
HITS = [_page_hit(FP153, p, score=0.7) for p in (36, 68, 35, 1, 81)]
# what the index holds on the two pages the note names
PAGES = {
    (FP153, 5): [_page_hit(FP153, 5, 0, text="A.8 Requested GCF amount 28,654"),
                 _page_hit(FP153, 5, 1, text="A.7 Total financing 49,654")],
    (FP153, 48): [_page_hit(FP153, 48, 0, text="E. LOGICAL FRAMEWORK"),
                  _page_hit(FP153, 48, 1, text="B.2(b) Requested GCF 26,654")],
}


@pytest.fixture
def app_env(monkeypatch, fake_registry):
    FakeMessage.sent = []
    monkeypatch.setattr(app.cl, "Message", FakeMessage)
    monkeypatch.setattr(app.cl, "Step", FakeStep)
    monkeypatch.setattr(app.cl, "make_async",
                        lambda fn: (lambda *a, **kw: _now(fn, *a, **kw)))
    monkeypatch.setattr(app, "ground_chunk", lambda *a, **kw: None)
    yield


def _run(monkeypatch, question=QUESTION, retriever=None, conductor_json=None,
         history=()):
    retriever = retriever if retriever is not None else FakeRetriever(HITS, PAGES)
    session = FakeSession(retriever=retriever, history=list(history))
    client = FakeOpenAI(conductor_json=conductor_json
                        or {"mode": "retrieve", "queries": [
                            {"q": question, "doc": None}]})
    monkeypatch.setattr(app.cl, "user_session", session)
    monkeypatch.setitem(sys.modules, "openai",
                        types.SimpleNamespace(AsyncOpenAI=lambda **kw: client))
    asyncio.run(app.main(FakeMessage(content=question)))
    return client, retriever


def _context(client):
    return client.answer_call["messages"][1]["content"].split(
        "Context excerpts:\n", 1)[1]


MARK = "(registry conflict page — fetched by page, not ranked)"


# ---------------------------------------------------------------------------
# A. the trigger
# ---------------------------------------------------------------------------
def test_a_conflict_note_fetches_the_pages_retrieval_missed(monkeypatch, app_env):
    """The whole item, end to end: the note names p.5 and p.48, ten excerpts
    hold neither, and the answer model now reads both."""
    client, retr = _run(monkeypatch)
    ctx = _context(client)
    assert "CONFLICT in this document" in ctx
    assert retr.probes == [{"doc": FP153, "pages": [5, 48], "k": 4,
                            "query": QUESTION}]
    assert f"[{FP153}, p. 5 — FP153, B.27, 2020] {MARK}" in ctx
    assert f"[{FP153}, p. 48 — FP153, B.27, 2020] {MARK}" in ctx
    # the second printing itself — the figure the note asked to be reported
    assert "B.2(b) Requested GCF 26,654" in ctx


def test_the_probe_pages_are_citable_and_verifiable_evidence(monkeypatch,
                                                             app_env):
    """A fetched page is evidence, not decoration: the citation gate accepts a
    bracket citing it, and `build_evidence` files its text under (doc, page)
    so a claim can be checked against the page it cites."""
    _client, retr = _run(monkeypatch)
    probe = app._conflict_probe(retr, registry.registry_note(QUESTION), HITS,
                                QUESTION)
    hits = probe + HITS
    assert app._invalid_citations(f"It requests both [{FP153}, p. 48].",
                                  hits) == []
    ev_map = verify.build_evidence(hits, [registry.registry_note(QUESTION)])
    assert "26,654" in ev_map[(FP153, 48)]


def test_the_query_orders_the_page_and_never_selects_it(monkeypatch, app_env):
    """`probe_pages` is asked with the user's own words — the ordering vote
    that puts FP153's fifth chunk of page 48 first — and with nothing else:
    the pages come from the note, so the query cannot add one."""
    _client, retr = _run(monkeypatch)
    (probe,) = retr.probes
    assert probe["query"] == QUESTION
    assert probe["pages"] == [5, 48]


# ---------------------------------------------------------------------------
# B. narrowness — four ways a turn does NOT probe
# ---------------------------------------------------------------------------
def test_a_document_that_contradicts_nothing_never_probes(monkeypatch, app_env):
    """FP274 is in the fake registry with no conflicting fact for this
    question's field… nothing to fetch, so nothing is fetched."""
    monkeypatch.setattr(registry, "_cache_v2", {FP153: FAKE_V2[FP153]})
    _client, retr = _run(monkeypatch, "What does FP274 fund?",
                         retriever=FakeRetriever(
                             [_page_hit(FP274, 1)], PAGES))
    assert retr.probes == []


def test_a_conflict_page_retrieval_already_found_is_not_fetched_again(
        monkeypatch, app_env):
    """Five of release-12's nine conflict turns already hold both pages. The
    probe must be silent on those, or every conflict turn double-prints."""
    hits = HITS + [_page_hit(FP153, 5), _page_hit(FP153, 48)]
    _client, retr = _run(monkeypatch, retriever=FakeRetriever(hits, PAGES))
    assert retr.probes == []


def test_only_the_missing_half_of_a_conflict_is_fetched(monkeypatch, app_env):
    """The partial case (release-12's l2x-conf-fp259-total): one printing
    landed, the other did not, and only the other is asked for."""
    hits = HITS + [_page_hit(FP153, 5)]
    _client, retr = _run(monkeypatch, retriever=FakeRetriever(hits, PAGES))
    assert [p["pages"] for p in retr.probes] == [[48]]


def test_a_main_registry_line_publishes_pages_and_fires_nothing(fake_registry):
    """`_note_pages` credits every page a note prints — cover pages included.
    The probe reads a strictly smaller set: only the CONFLICT lines."""
    note = registry.registry_note(QUESTION)
    main = "\n".join(ln for ln in note.splitlines()
                     if "CONFLICT in this document" not in ln)
    assert app._note_pages([main]), "the main line publishes pages of its own"
    assert app._conflict_probe_asks(main, []) == []


def test_a_document_named_on_another_line_lends_no_pages(fake_registry):
    """Two documents, two conflict lines: neither may be asked for the other's
    page. The stem on the line owns the pages on the line."""
    note = ("Registry — CONFLICT in this document (%s): gcf_funding_requested "
            "is printed as A (p.5, A.8); also as B (p.48, B.2(b)) — report "
            "both figures with their pages.\n"
            "Registry — CONFLICT in this document (%s): gcf_funding_requested "
            "is printed as C (p.8, A.10); also as D (p.40) — report both "
            "figures with their pages." % (FP153, FP274))
    assert app._conflict_probe_asks(note, []) == [(FP153, [5, 48]),
                                                  (FP274, [8, 40])]


def test_the_probe_reads_the_registry_note_and_not_the_answer(fake_registry):
    """A note is the only input. Nothing here reads the model's prose, so an
    answer that invents '(p.99)' cannot make the next turn fetch page 99."""
    assert app._conflict_probe_asks(None, []) == []
    assert app._conflict_probe_asks("", HITS) == []
    assert app._conflict_probe_asks(
        "The document also states 40,751,254 (p.99).", []) == []


# ---------------------------------------------------------------------------
# C. the caps
# ---------------------------------------------------------------------------
def _long_note(doc, pages):
    return ("Registry — CONFLICT in this document (%s): gcf_funding_requested "
            "is printed as %s — report all of them with their pages."
            % (doc, "; also as ".join(f"{p}00 USD (p.{p}, A.8)" for p in pages)))


def test_at_most_four_pages_are_ever_asked_for(fake_registry):
    """Two conflict lines naming three prints each is six pages a note can
    print. Four is the backstop, and it keeps the ones printed FIRST — every
    conflict line leads with the canonical figure's page."""
    note = _long_note(FP153, [5, 6, 7]) + "\n" + _long_note(FP153, [40, 41, 42])
    assert app._conflict_probe_asks(note, []) == [(FP153, [5, 6, 7, 40])]
    assert app._MAX_PROBE_PAGES == 4


def test_at_most_four_excerpts_are_ever_appended(monkeypatch, app_env):
    """A page is one to five chunks; the budget is over excerpts, so a
    two-page conflict cannot quietly append ten."""
    pages = {(FP153, p): [_page_hit(FP153, p, n) for n in range(5)]
             for p in (5, 48)}
    client, retr = _run(monkeypatch, retriever=FakeRetriever(HITS, pages))
    assert _context(client).count(MARK) == app._MAX_PROBE_HITS == 4


def test_the_leftover_budget_buys_a_second_chunk_not_a_second_page(
        monkeypatch, app_env):
    """Two pages asked, four slots: each asked page is served before any page
    takes a second, and the spare slots go to the pages the note named — never
    to a page nobody asked about."""
    client, _retr = _run(monkeypatch)
    ctx = _context(client)
    assert ctx.count(MARK) == 4
    assert ctx.count(f"[{FP153}, p. 5 —") == 2
    assert ctx.count(f"[{FP153}, p. 48 —") == 2


def test_a_second_document_keeps_its_slots(fake_registry):
    """The starvation case: one document with a deep page must not spend the
    whole budget before the other document is asked at all."""
    deep = {(FP153, 5): [_page_hit(FP153, 5, n) for n in range(5)],
            (FP274, 8): [_page_hit(FP274, 8, 0)]}
    retr = FakeRetriever([], deep)
    note = ("Registry — CONFLICT in this document (%s): f is printed as A "
            "(p.5, A.8) — report both.\n"
            "Registry — CONFLICT in this document (%s): f is printed as B "
            "(p.8, A.10) — report both." % (FP153, FP274))
    got = app._conflict_probe(retr, note, [], "q")
    assert len(got) == 4
    assert [h.doc_id for h in got].count(FP274) == 1
    assert [p["k"] for p in retr.probes] == [3, 1]


# ---------------------------------------------------------------------------
# D. degradation — four failures, one behaviour
# ---------------------------------------------------------------------------
def _baseline_context(monkeypatch, app_env_unused=None):
    """The context this turn produces with the probe returning nothing."""
    client, _ = _run(monkeypatch, retriever=FakeRetriever(HITS, {}))
    return _context(client)


def test_a_probe_that_raises_leaves_the_turn_exactly_as_it_was(monkeypatch,
                                                               app_env):
    boom, _ = _run(monkeypatch, retriever=FakeRetriever(HITS, PAGES, boom=True))
    assert _context(boom) == _baseline_context(monkeypatch)
    assert MARK not in _context(boom)


def test_a_retriever_without_the_mechanism_leaves_the_turn_as_it_was(
        monkeypatch, app_env):
    old, _ = _run(monkeypatch, retriever=OldRetriever(HITS, PAGES))
    assert _context(old) == _baseline_context(monkeypatch)


def test_a_page_the_index_cannot_serve_is_not_invented(monkeypatch, app_env):
    """`probe_pages` returns the pages that exist and no others; an empty
    result is the turn as it is today, not an empty excerpt."""
    client, retr = _run(monkeypatch, retriever=FakeRetriever(HITS, {}))
    assert retr.probes and _context(client) == _baseline_context(monkeypatch)
    assert MARK not in _context(client)


def test_no_retriever_at_all_is_no_probe(fake_registry):
    assert app._conflict_probe(None, registry.registry_note(QUESTION), [], "q") == []


def test_a_probe_hit_for_a_page_already_held_is_dropped(fake_registry):
    """The forgiving document match inside `probe_pages` ('fp153' reaches
    '122_…') can return a page this turn already has when the note spells the
    stem differently. Dropped on the way in: one excerpt, one page."""
    retr = FakeRetriever([], {(FP153, 5): [_page_hit(FP153, 5, 0)],
                              (FP153, 48): [_page_hit(FP153, 48, 0)]})
    note = ("Registry — CONFLICT in this document (%s): f is printed as A "
            "(p.5, A.8); also as B (p.48) — report both." % FP153)
    got = app._conflict_probe(retr, note, [_page_hit(FP153, 5, 9)], "q")
    assert [(h.doc_id, h.page) for h in got] == [(FP153, 48)]


# ---------------------------------------------------------------------------
# E. the turns that never probe
# ---------------------------------------------------------------------------
def test_a_chat_turn_never_probes(monkeypatch, app_env):
    """Conversational turns retrieve nothing, so there is nothing to
    supplement — and the conflict note is not even built."""
    client, retr = _run(monkeypatch, "What did you just say about FP153?",
                        conductor_json={"mode": "chat", "queries": []})
    assert retr.probes == [] and retr.calls == []


def test_a_guard_turn_never_probes(monkeypatch, app_env):
    """The FP-miss refusal answers from the registry before retrieval runs."""
    _client, retr = _run(monkeypatch, "How much does FP999 request?")
    assert retr.probes == [] and retr.calls == []


def test_the_harness_chat_and_guard_turns_carry_no_probe(fake_registry):
    """Same two turns through Pipeline.run, which reports what it probed.

    Both are asked with the question whose note DOES carry a conflict line, so
    an empty `probe_hits` here means the path returned before the probe — not
    that there was nothing to fetch."""
    items = [{"q": QUESTION, "doc": None}]
    assert ev.Pipeline.run(_harness_pipe(items), QUESTION)["probe_hits"], \
        "the same turn probes when it is neither chat nor guard"
    assert ev.Pipeline.run(_harness_pipe(items, chat=True),
                           QUESTION)["probe_hits"] == []
    assert ev.Pipeline.run(_harness_pipe(items, guard="FP153 … registry says"),
                           QUESTION)["probe_hits"] == []


# ---------------------------------------------------------------------------
# F. placement — a supplement, never a replacement
# ---------------------------------------------------------------------------
def test_the_probe_evicts_no_hit_that_earned_its_slot(monkeypatch, app_env):
    """The cap is extended by the probe's count, not spent on it: every ranked
    excerpt is still in the context, in the order retrieval put it."""
    client, _ = _run(monkeypatch)
    ctx = _context(client)
    ranked = [f"[{FP153}, p. {p} — FP153, B.27, 2020] (score 0.70)"
              for p in (36, 68, 35, 1, 81)]
    assert all(r in ctx for r in ranked)
    assert [ctx.index(r) for r in ranked] == sorted(ctx.index(r) for r in ranked)


def test_the_fetched_pages_lead_the_ranked_ones(monkeypatch, app_env):
    """Directly under the note that named them: the instruction to report both
    figures and the evidence for the second one are read together."""
    client, _ = _run(monkeypatch)
    ctx = _context(client)
    assert ctx.index(MARK) < ctx.index("(score 0.70)")
    assert ctx.index("CONFLICT in this document") < ctx.index(MARK)


def test_a_fetched_page_is_labelled_rather_than_scored(monkeypatch, app_env):
    """It has no rank in the similarity order the other excerpts are printed
    in, so it prints no score to compare against theirs."""
    client, _ = _run(monkeypatch)
    for line in _context(client).splitlines():
        if MARK in line:
            assert "(score" not in line


def test_the_context_block_is_byte_identical_without_a_probe():
    """The shared builder is the old f-string when nothing was fetched."""
    hits = [_page_hit(FP153, 5, score=0.734), _page_hit(FP153, 6, score=0.7)]
    assert app._context_block(hits) == "\n\n".join(
        f"[{app._doc_label(h.doc_id, h.page)}] (score {h.score:.2f})\n{h.text}"
        for h in hits)
    assert app._context_block(hits, probe=()) == app._context_block(hits)


def test_only_the_probe_hits_are_marked_even_on_a_page_twice_over():
    """Marking is by object, not by (doc, page): a ranked hit on a page the
    probe also served keeps its score."""
    ranked = _page_hit(FP153, 48, score=0.61)
    fetched = _page_hit(FP153, 48, 1)
    block = app._context_block([fetched, ranked], probe=[fetched])
    assert block.count(MARK) == 1 and "(score 0.61)" in block


# ---------------------------------------------------------------------------
# G. parity
# ---------------------------------------------------------------------------
def _harness_pipe(items, hits=HITS, retriever=None, chat=False, guard=None):
    """A duck-typed Pipeline: the real run(), the stages around it stubbed."""
    retriever = retriever if retriever is not None else FakeRetriever(hits, PAGES)
    return types.SimpleNamespace(
        app=app, conductor=True, retriever=retriever,
        planner_plan=lambda q: (None, None),
        conduct=lambda q, turns=(): (("chat" if chat else "retrieve"),
                                     [dict(i) for i in items], None),
        fp_guard=lambda q: guard,
        _decomposed=lambda it, q, plan=None: len(it) > 1,
        _retrieve=lambda it, dec, original=None: (list(hits), 0.9, False),
    )


def test_the_app_and_the_harness_build_the_same_context(monkeypatch, app_env):
    """Production parity is the point of the harness: a release record's
    excerpts have to be the excerpts production would have shipped, marker
    included, or the conflict-page measurement measures the harness."""
    client, _ = _run(monkeypatch)
    out = ev.Pipeline.run(_harness_pipe([{"q": QUESTION, "doc": None}]), QUESTION)
    assert out["context"] == _context(client).rsplit("\n\nQuestion: ", 1)[0]
    assert [(h.doc_id, h.page) for h in out["probe_hits"]] == [
        (FP153, 5), (FP153, 48), (FP153, 5), (FP153, 48)]


def test_the_harness_probes_through_the_apps_own_function(monkeypatch):
    """Not a second implementation: the harness calls `_conflict_probe`."""
    seen = []
    monkeypatch.setattr(app, "_conflict_probe",
                        lambda *a, **kw: seen.append(a) or [])
    ev.Pipeline.run(_harness_pipe([{"q": QUESTION, "doc": None}]), QUESTION)
    assert len(seen) == 1 and seen[0][3] == QUESTION


def test_a_harness_pipe_with_no_retriever_degrades_like_the_app(monkeypatch,
                                                                fake_registry):
    pipe = _harness_pipe([{"q": QUESTION, "doc": None}])
    del pipe.retriever
    assert ev.Pipeline.run(pipe, QUESTION)["probe_hits"] == []


# ---------------------------------------------------------------------------
# H. the note-reading patterns cannot drift
# ---------------------------------------------------------------------------
def test_the_probe_reads_a_note_line_with_the_patterns_the_gate_reads_it_with():
    r"""Three copies of two patterns now exist — `verify._NOTE_*_RE`,
    `chainlit_app._note_pages`'s inline pair (pinned as source text by
    tests/test_registry_resolver.py, so it cannot be hoisted), and the probe's
    compiled pair. If they drift, the probe fetches a page the citation gate
    then calls invented, or refuses one the note published."""
    src = (ROOT / "src" / "gcf_qna" / "app" / "chainlit_app.py").read_text(
        encoding="utf-8")
    assert app._CONFLICT_PAGE_RE.pattern == verify._NOTE_PAGE_RE.pattern
    assert app._CONFLICT_DOC_RE.pattern == verify._NOTE_DOC_RE.pattern
    for rx in (app._CONFLICT_DOC_RE, app._CONFLICT_PAGE_RE):
        assert f're.findall(r"{rx.pattern}", line)' in src


def test_the_probe_only_asks_for_pages_the_citation_gate_already_allows(
        fake_registry):
    """The pages the probe fetches are a subset of the pages the note
    published — so an answer citing one was legal before it was retrievable."""
    note = registry.registry_note(QUESTION)
    asked = {(d, p) for d, pages in app._conflict_probe_asks(note, [])
             for p in pages}
    assert asked and asked <= app._note_pages([note])


# ---------------------------------------------------------------------------
# I. the recorded turns
# ---------------------------------------------------------------------------
RELEASE_12 = ROOT / "data" / "eval" / "release_release-12.jsonl"


def _recorded_conflict_turns():
    if not RELEASE_12.exists():                  # pragma: no cover
        pytest.skip("release-12 record not present")
    return [r for r in (json.loads(ln) for ln in
                        RELEASE_12.read_text(encoding="utf-8").splitlines() if ln)
            if r.get("class") == "conflict"]


@pytest.mark.parametrize("case,pages", [
    ("conf-fp274-gcf", {}),
    ("conf-fp267-gcf", {}),
    ("conf-fp274-consistency",
     {"02_gcf-b42-02-add16-funding-proposal-package-fp274": [7]}),
    ("conf-fp153-gcf", {"122_gcf-b27-02-add13": [5, 48]}),
    ("conf-fp265-gcf", {}),
    ("conf-fp251-gcf",
     {"25_gcf-b40-02-add13-funding-proposal-package-fp251": [5, 6, 47]}),
    ("conf-fp201-gcf", {"74_gcf-b35-02-add03-rev01_0": [5, 6, 67]}),
    ("l2x-conf-fp259-total",
     {"17_gcf-b41-02-add07-funding-proposal-package-fp259": [42]}),
    ("l2x-conf-fp249-total-consistency", {}),
])
def test_the_recorded_release_12_turn_asks_for_what_it_was_missing(case, pages):
    """Every conflict-class turn of the last release, replayed from its own
    record — the note it shipped with and the excerpts it shipped with. Four
    of the nine were missing a page the note had named; five already held
    every page, and ask for nothing."""
    rec = next(r for r in _recorded_conflict_turns() if r["id"] == case)
    hits = [Hit(text=h["text"], doc_id=h["doc"], score=h["score"],
                page=h["page"]) for h in rec["hits"]]
    note = (rec.get("notes_used") or {}).get("registry")
    assert dict(app._conflict_probe_asks(note, hits)) == pages
