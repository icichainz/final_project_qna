"""Adversarial acceptance suite for the Wave 4 repair-safety gates, retriaged
for the PURE-DETECTOR decision.

ORIGINALLY (e639915): written from the SPEC (commit 52152af's findings and
blocking prerequisites), BEFORE and INDEPENDENTLY of the implementation, so it
could not inherit the implementer's blind spots. 36 tests over four gates:

    (a) sampling pinned on the judge and repair calls
    (b) language flips rejected, symmetrically, without false positives
    (c) a minimum-substance floor that fires when the pre-repair answer has no
        supported required claims
    (d) carry-on / carry-off: `_carry_cleared` must stop certifying rewrites
        the honest recheck rejects

plus the perimeter that must not regress while those land.

WHAT CHANGED, AND WHY
---------------------
eac4c94 abandoned automatic repair adoption. `verify.repair()` and its
adoption gates are gone: `verify_answer` has no `allow_repair`, `.answer` is
always the answer that was passed in, `.repaired` is always False, and no
status is ever 'repaired'. The verifier is a pure DETECTOR.

A gate that decides whether to adopt a rewrite cannot be attacked when no
rewrite is ever adopted. Twenty of the thirty-six tests below tested exactly
that decision, and they are RETIRED here rather than left to rot as tests that
import a symbol that no longer exists:

  RETIRED — (b) language flip, 8 tests  b1 b2 b3 b4 b5 b6 b7 b8
      `_language_flip` compared the answer BEFORE a rewrite with the answer
      AFTER it. Nothing is rewritten, so there is no pair to compare and no
      flip to reject. b5-b8 were false-positive guards ON that comparison
      (English proper nouns in a French answer, a quoted French title, a
      mostly-numeric table, a re-rendered quotation); a guard against
      over-correction by a gate that does not exist cannot fire either.

  RETIRED — (c) substance floor, 6 tests  c1 c2 c3 c4 c5 c6
      `_substance_floor`, `MIN_REPAIR_WORDS` and `SUBSTANTIAL_ANSWER_CHARS`
      answered one question: is this REWRITE too gutted to ship in place of
      the original? The abs-2014 regression the block was built around — a
      351-character registry-backed abstention becoming the word "None." — is
      unreachable when the answer is never replaced. c4's arithmetic
      discriminator (no length ratio separates c2 from c3) measured two
      fixtures that now only feed a deleted gate.

  RETIRED — (d) carry-on / carry-off, 3 tests  d1 d2 d3
      `_carry_cleared` carried a judge ruling made on the PRE-repair answer
      onto the recheck of the POST-repair answer. With one text there is one
      classification, the judge rules on the text being shown, and the
      self-certification pattern has no second text to certify with.

  RETIRED — adoption perimeter, 3 tests  p1 p3 p5
      p1 (a clean repair with no remaining failures is adopted), p3 (the
      anti-gutting guard still fires when supported claims existed) and p5
      (one wrong figure swapped for another is still rejected) each assert on
      the adopt/reject verdict itself. p5's detector half — a figure the cited
      evidence contradicts stays CONTRADICTED — is covered by p4 below.

The full 36-test suite runs green against the repair implementation at
e639915 (and its re-A/B at 9925a2c); eac4c94 is the last commit at which
`verify.repair` exists. Nothing is lost: git history holds the code, and the
recorded A/B artifacts under data/eval hold the measurement.

WHAT SURVIVED
-------------
Sixteen tests. Every one of them exercises the DETECTOR, which eac4c94 left
fully intact:

  (a) is entirely about `_complete` — the module's only remaining model call
      is the judge, and it is still an unpinned sample if it is not pinned.
      a1/a2/a5/a6/a7 asserted on the repair call's payload and now assert on
      the judge call's; a3/a4 already did. a8 never touched repair at all: it
      is `_resolve_doc` returning different verdicts under different
      PYTHONHASHSEEDs, the finding that pinning temperature and seed does NOT
      fix, and it belongs to the deterministic layer.
  p4 was a repair attack over a DETECTOR hole. The conflict gate (22f558b,
      "no SUPPORTED without conflict-testing the scope it stands on") closed
      the hole, so the property is now stated directly against
      `classify_deterministic` and the xfail is gone.
  p2 is kept as an exposure record; read its docstring, the exact-match rule
      it asserted was repair-only.
  p6 (keyless degradation) and p7 (the model-call budget) are perimeter
      properties of the pass as a whole and tighten under pure detection.

NAMING IS STILL THE CONTRACT
  test_<gate><n>_attack_...  a property whose violation is a shipped harm.
  test_<gate><n>_guard_...   a property that must not be broken by fixing an
                             attack — these exist because the cheapest way to
                             satisfy an attack test is to over-correct.

RUN
    pytest tests/test_repair_gate_attacks.py -q
    GCF_QNA_ROOT=/path/to/checkout pytest tests/test_repair_gate_attacks.py -q

NO NETWORK. `verify._client` is patched to raise for the whole session; every
test hands the pass an explicit fake client. The two subprocess tests re-apply
the same patch inside the child.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

# --------------------------------------------------------------------------
# bootstrap: import the repo's REAL verify module, with no repo conftest
# --------------------------------------------------------------------------
ROOT = os.environ.get("GCF_QNA_ROOT", "/home/ssa/Workspace/final_project_qna")
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
os.environ["PRELOAD"] = "0"                 # never load the 730 MB index
os.environ.setdefault("OPENAI_API_KEY", "attack-suite-not-a-real-key")

from gcf_qna.rag import verify as V        # noqa: E402
from gcf_qna.rag import registry as _registry  # noqa: E402

assert os.path.realpath(V.__file__).startswith(os.path.realpath(SRC)), (
    f"imported {V.__file__}, not the module under test in {SRC}")

_REAL_CLIENT = V._client


def test_the_pure_detector_contract_holds():
    """eac4c94's contract, asserted once so the rest of the suite can assume
    it: the module exposes no repair pathway, and the pass returns the answer
    it was given. Every other test below reads the detector; this one states
    what stopped existing, so a reintroduced adoption path fails loudly here
    rather than quietly changing what a dozen assertions mean."""
    assert not hasattr(V, "repair"), (
        "verify.repair is back: this suite's gates were retired at eac4c94 on "
        "the premise that no rewrite is ever adopted")
    import inspect
    params = inspect.signature(V.verify_answer).parameters
    assert "allow_repair" not in params, sorted(params)


# --------------------------------------------------------------------------
# fixture corpus — the tiny four-key set tests/test_verify.py uses, so every
# verdict below can be checked by reading the fixture
# --------------------------------------------------------------------------
DOC = "124_gcf-b27-02-add11"          # FP151 package
DOC2 = "123_gcf-b27-02-add12"         # FP152 package

REGISTRY_LINE = (
    'Registry — FP151: "Technical Assistance (TA) Facility for the Global '
    'Subnational Climate Fund"; accredited entity: International Union for '
    'Conservation of Nature and Natural Resources (IUCN); countries: Angola, '
    'Benin, Kenya; GCF financing (as printed): 18.5 M USD; total financing '
    f'(as printed): 28 M USD; board B.27, 2020 [{DOC}, cover pages]')

YEAR_NOTE = ("Note (computed from the corpus registry, which is complete — treat "
             "it as authoritative): 2020 — 30 proposals; 2014 — no board meeting, "
             "no registered proposals.")

PAGE45 = ("### (a) Requested GCF funding (Total amount)\n"
          "| (vi) Grants | 18,500,000 | 7 | |")

PAGE2_5 = ("Total GCF funding requested: USD 150 million\n"
           "Accredited entity: Pegasus Capital Advisors LP")


def base_evidence():
    """A fresh dict every call: no test may see another test's object."""
    return {
        (DOC, None): REGISTRY_LINE,
        (DOC, 45): PAGE45,
        (DOC2, 5): PAGE2_5,
        V.NOTES_KEY: YEAR_NOTE,
    }


@pytest.fixture
def evidence():
    return base_evidence()


@pytest.fixture(autouse=True)
def _no_registry_no_network(monkeypatch):
    """No registry file, no client construction, no leaked sampling state.

    The registry is stubbed empty so a verdict can only come from the held
    evidence — the suite must not change meaning when data/registry.json does.
    `_client` is made to raise so an accidental network path is a loud failure
    rather than a slow one. `_SAMPLING_UNSUPPORTED` is module-global and a7
    writes to it, so it is reset around every test: without this, a7 running
    first would silently unpin the calls a1/a2/a5 measure.
    """
    monkeypatch.setattr(_registry, "load", lambda: {})
    monkeypatch.setattr(_registry, "facts", lambda doc: {})
    monkeypatch.setattr(V, "_client", _forbidden_client)
    V._reset_sampling_support()
    yield
    V._reset_sampling_support()


def _forbidden_client():
    raise AssertionError("verify._client() was called: this suite makes no "
                         "network calls and always passes an explicit client")


class FakeClient:
    """Stand-in for the OpenAI client: replies in order, records every call.

    ``reject`` names request keys the endpoint refuses (LM Studio, vLLM and
    Azure all 400 on an unknown sampling parameter). Raising from
    ``create`` is exactly what the SDK does on a 400.
    """

    def __init__(self, *replies, reject=()):
        self.replies = list(replies)
        self.reject = tuple(reject)
        self.calls = []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.calls.append(dict(kw))
                for key in outer.reject:
                    if key in kw:
                        raise TypeError(
                            f"Unrecognized request argument supplied: {key}")
                content = outer.replies.pop(0) if outer.replies else ""
                if isinstance(content, Exception):
                    raise content
                msg = type("M", (), {"content": content})()
                return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()

        self.chat = type("Chat", (), {"completions": _Completions()})()


def _system_of(kw):
    msgs = kw.get("messages") or []
    return next((m.get("content", "") for m in msgs
                 if m.get("role") == "system"), "")


def _judge(*pairs):
    """A judge reply marking the given claim indices supported."""
    return json.dumps({"verdicts": [{"id": i, "status": s, "reason": "judged"}
                                    for i, s in pairs]})


def run(answer, judge=None, evidence=None, **kw):
    """One full verify_answer pass with an optional canned judge reply.

    Before eac4c94 this also took the canned REPAIR text, and every caller
    below asserted on whether it was adopted. There is one text now.
    """
    ev = base_evidence() if evidence is None else evidence
    client = FakeClient(*([judge] if judge is not None else []))
    use_llm = kw.pop("use_llm", judge is not None)
    res = V.verify_answer(answer, ev, client=client, use_llm=use_llm, **kw)
    return res, client


def assert_unchanged(res, answer):
    """The pure-detector guarantee: the pass reports, it does not rewrite."""
    __tracebackhide__ = True
    assert res.answer == answer, (
        f"the pass returned text it was not given:\n{res.answer[:300]!r}")
    assert res.repaired is False and res.status != "repaired", (
        f"status={res.status!r} repaired={res.repaired!r}")


_CHILD_FAKE_CLIENT = """
class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies); self.calls = []
        outer = self
        class _C:
            def create(self, **kw):
                outer.calls.append(dict(kw))
                c = outer.replies.pop(0) if outer.replies else ""
                m = type("M", (), {"content": c})()
                return type("R", (), {"choices": [type("C", (), {"message": m})()]})()
        self.chat = type("Chat", (), {"completions": _C()})()
"""


def _child(body, hashseed):
    """Run `body` in a fresh interpreter with a pinned PYTHONHASHSEED."""
    script = (
        "import os, sys, json\n"
        f"sys.path.insert(0, {SRC!r})\n"
        "os.environ['PRELOAD'] = '0'\n"
        "os.environ['OPENAI_API_KEY'] = 'attack-suite-not-a-real-key'\n"
        "from gcf_qna.rag import verify as V\n"
        "from gcf_qna.rag import registry as R\n"
        "R.load = lambda: {}\n"
        "R.facts = lambda d: {}\n"
        "V._client = lambda: (_ for _ in ()).throw(AssertionError('no network'))\n"
        + _CHILD_FAKE_CLIENT
        + textwrap.dedent(body))
    env = dict(os.environ, PYTHONHASHSEED=str(hashseed), PRELOAD="0")
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=180, env=env)
    assert out.returncode == 0, f"child failed:\n{out.stdout}\n{out.stderr}"
    return json.loads(out.stdout.strip().splitlines()[-1])


# ==========================================================================
# (a) SAMPLING MUST BE PINNED
#
# Wave 4: "verify._complete sends no temperature and no seed, so every rewrite
# of user-visible text is an unpinned sample" — from the same answers and the
# same verdict objects the adoption decision flips on 18-27% of cases.
# Prerequisite: "pin temperature/seed on the judge and repair calls".
#
# eac4c94 removed the repair call. The judge call is the whole of `_complete`
# now, and an unpinned judge is an unpinned VERDICT: the status the app shows,
# the cautions it renders and the abstentions it prints are all downstream of
# a ruling that would otherwise be a fresh sample every turn. The gate is the
# same gate; the call it is measured on moved.
# ==========================================================================

A_BAD = (f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 45].\n\n"
         f"The programme covers **Angola**, **Benin** and **Kenya** "
         f"[{DOC}, cover pages].")


def test_a1_attack_the_pass_s_model_call_sends_a_seed():
    """TARGET: (a). VARIES: nothing — the request payload of the call
    `verify_answer` issues. Pre-eac4c94 this measured the repair call.
    ON 52152af: FAILS. It sent only model/max_completion_tokens/messages."""
    res, client = run(A_BAD, judge=_judge((0, "supported")))
    assert_unchanged(res, A_BAD)
    assert client.calls, "the model call did not happen; fixture is stale"
    (kw,) = client.calls
    assert "seed" in kw, (
        "the pass's model call sends no `seed`: the verdict that decides what "
        f"the user is told is an unpinned sample. sent keys={sorted(kw)}")
    assert isinstance(kw["seed"], int) and not isinstance(kw["seed"], bool)


def test_a2_attack_the_pass_s_model_call_pins_temperature():
    """TARGET: (a). VARIES: same payload, the other pinning knob.
    ON 52152af: FAILS — no temperature was sent."""
    res, client = run(A_BAD, judge=_judge((0, "supported")))
    assert_unchanged(res, A_BAD)
    (kw,) = client.calls
    assert "temperature" in kw, f"no temperature pinned; sent keys={sorted(kw)}"
    assert kw["temperature"] == 0, f"temperature={kw['temperature']!r}, not 0"


def test_a3_attack_judge_call_sends_a_seed(evidence):
    """TARGET: (a). VARIES: the entry point — `classify` called directly,
    below `verify_answer`, so a pin applied by the caller rather than by
    `_complete` is visible as a difference against a1.
    ON 52152af: FAILS."""
    answer = "The total GCF funding requested is USD 150 million."
    client = FakeClient(_judge((0, "supported")))
    V.classify(V.extract_claims(answer), evidence, client=client)
    assert client.calls, "the judge call did not happen; fixture is stale"
    kw = client.calls[0]
    assert "seed" in kw, f"the judge call sends no `seed`; keys={sorted(kw)}"
    assert isinstance(kw["seed"], int) and not isinstance(kw["seed"], bool)


def test_a4_attack_judge_call_pins_temperature(evidence):
    """TARGET: (a). VARIES: the judge call's temperature. ON 52152af: FAILS."""
    answer = "The total GCF funding requested is USD 150 million."
    client = FakeClient(_judge((0, "supported")))
    V.classify(V.extract_claims(answer), evidence, client=client)
    kw = client.calls[0]
    assert "temperature" in kw, f"no temperature pinned; keys={sorted(kw)}"
    assert kw["temperature"] == 0


def test_a5_attack_two_identical_runs_send_an_identical_request():
    """TARGET: (a). VARIES: the run, not the input — two independent passes
    over freshly built, equal inputs.

    Catches the pinning that is not pinning: `seed=random.randint(...)`,
    `seed=int(time.time())`, or a seed derived from object identity. Presence
    alone (a1) cannot see any of those.
    ON 52152af: FAILED on the presence half; the payload-equality half already
    held, which is the point — equality is not evidence of pinning on its own."""
    res1, c1 = run(A_BAD, judge=_judge((0, "supported")))
    res2, c2 = run(A_BAD, judge=_judge((0, "supported")))
    kw1, kw2 = c1.calls[0], c2.calls[0]
    assert kw1 == kw2, (
        "two identical inputs produced different model requests:\n"
        f"  only in run 1: { {k: v for k, v in kw1.items() if kw2.get(k) != v} }\n"
        f"  only in run 2: { {k: v for k, v in kw2.items() if kw1.get(k) != v} }")
    assert {"seed", "temperature"} <= set(kw1), (
        "the two requests are equal but neither is pinned; "
        f"keys={sorted(kw1)}")
    assert (res1.status, res1.answer) == (res2.status, res2.answer)
    assert [(v.status, v.source) for v in res1.verdicts] == \
           [(v.status, v.source) for v in res2.verdicts]


def test_a6_attack_the_seed_is_stable_across_processes():
    """TARGET: (a). VARIES: the interpreter's PYTHONHASHSEED.

    A seed derived from `hash(answer)` looks pinned in one process and is a
    different sample in the next, because str hashing is randomised per
    process. Two children, two hash seeds, same input.
    ON 52152af: FAILED — no seed was sent at all (both children reported null)."""
    body = f"""
        ev = {{({DOC!r}, None): {REGISTRY_LINE!r},
              ({DOC!r}, 45): {PAGE45!r}}}
        c = FakeClient({_judge((0, "supported"))!r})
        V.verify_answer({A_BAD!r}, ev, client=c, use_llm=True)
        kw = c.calls[0]
        print(json.dumps({{"seed": kw.get("seed"),
                           "temperature": kw.get("temperature"),
                           "keys": sorted(kw)}}))
    """
    a = _child(body, 0)
    b = _child(body, 987654321)
    assert a["seed"] is not None, (
        f"no seed reaches the endpoint; the call sent {a['keys']}")
    assert a["seed"] == b["seed"], (
        f"the seed depends on the process: {a['seed']} vs {b['seed']} — a "
        "hash()-derived seed is not a pinned sample")


def test_a7_attack_an_endpoint_that_rejects_seed_must_not_pin_silently():
    """TARGET: (a), third question. VARIES: the endpoint, not the input.

    LM Studio / vLLM / Azure 400 on an unrecognised sampling parameter. The
    failure mode to forbid is a silent one: the seed is refused and the pass
    retries without it, so the ruling looks exactly as pinned as one that was.

    Before eac4c94 the shipped harm was an unpinned REWRITE adopted with the
    ordinary note, and the test allowed three answers — refuse, keep the
    deterministic verdicts, or retry and SAY SO. Nothing is adopted now, so
    what remains is `_complete`'s own contract, which is the half that was
    always mechanical: the pin must actually be ATTEMPTED (an endpoint that is
    never sent a seed cannot refuse one), the retry must be bounded, and
    dropping the refused parameter must not quietly drop the other one.
    ON 52152af: FAILED. No seed was ever sent, so nothing was refused, and
    temperature was absent from every call."""
    ev = base_evidence()
    reply = _judge((0, "supported"))
    client = FakeClient(reply, reply, reply, reply, reject=("seed",))
    res = V.verify_answer(A_BAD, ev, client=client, use_llm=True)

    assert any("seed" in kw for kw in client.calls), (
        "no call ever attempted to send a `seed`, so the endpoint had nothing "
        f"to reject; calls={[sorted(kw) for kw in client.calls]}")
    assert len(client.calls) <= 4, "unbounded retry against a 400"
    assert all(kw.get("temperature") == 0 for kw in client.calls), (
        "a retry after the seed was refused dropped temperature pinning too; "
        f"temperatures={[kw.get('temperature') for kw in client.calls]}")
    assert_unchanged(res, A_BAD)


def test_a8_attack_the_verdict_does_not_depend_on_pythonhashseed():
    """TARGET: (a), second question — 'can you construct an input where it
    still varies?'. VARIES: PYTHONHASHSEED, with sampling irrelevant (no model
    call is made at all).

    `_resolve_doc` picked the first prefix match while iterating a SET of held
    document ids. Two held documents sharing a 24-character prefix — the shape
    the corpus is full of, `..add11` / `..add12` — resolved differently in
    different processes, so a CITED claim was supported in one run and
    contradicted in the next.

    This is the finding pinning temperature and seed does NOT fix, and the one
    piece of the (a) block that never involved the repair pass: it lives
    entirely in the deterministic layer, so removing repair removes nothing
    from it. `_resolve_doc` now sorts (see its docstring), which is a stable
    answer to an ambiguous citation, not a right one.
    ON 52152af: FAILED — 'verified' under hash seeds 0-2, 'abstain' under 3-7."""
    a, b = "124_gcf-b27-02-add11", "124_gcf-b27-02-add12"
    answer = "FP151 requests **USD 18.5 million** in GCF funding " \
             "[124_gcf-b27-02-add1, p. 5]."          # a truncated id
    body = f"""
        ev = {{({a!r}, 5): "Total GCF funding requested: USD 18.5 million",
              ({b!r}, 5): "Total GCF funding requested: USD 150 million"}}
        r = V.verify_answer({answer!r}, ev, client=None, use_llm=False)
        print(json.dumps({{"status": r.status,
                           "verdicts": [[v.status, v.scope and v.scope[0][0]]
                                        for v in r.verdicts]}}))
    """
    seen = {}
    for hs in (0, 1, 2, 3, 4, 5, 6, 7):
        got = _child(body, hs)
        seen.setdefault(json.dumps(got, sort_keys=True), []).append(hs)
    assert len(seen) == 1, (
        "identical answer, identical evidence, no model call at all — and the "
        f"verdict depends on the interpreter's hash seed: {seen}")


# ==========================================================================
# PERIMETER
#
# (b), (c) and (d) were retired at eac4c94; see the module docstring. What
# follows is the part of the perimeter that reads the detector.
# ==========================================================================

EN_BAD = A_BAD


@pytest.mark.parametrize("cited,label,caught", [
    (f"{DOC}-annex-volume-2, p. 45", "long suffix", False),
    (f"{DOC}0, p. 45", "one-character suffix", False),
    (f"{DOC}, p. 45; {DOC}-v2, p. 45", "fake chained after a real one", False),
    (f"{DOC[:-1]}, p. 45", "truncated prefix of a real id", False),
    (f"{DOC.upper()}, p. 45", "case-changed id", True),
])
def test_p2_guard_the_detector_s_reach_over_near_miss_document_ids(
        cited, label, caught):
    """TARGET: near-miss document ids. VARIES: the SHAPE of the near-miss —
    suffix, one character, chained behind a real id, a truncated prefix, and
    case.

    AN EXPOSURE RECORD, NOT AN ENDORSEMENT. Before eac4c94 this test asserted
    that all five are REJECTED, and all five passed — but the rule doing the
    rejecting was `verify._introduced_sources`, an adoption gate: it demanded
    that a citation in a REWRITE name a held document EXACTLY, because a claim
    re-attributed to another retrieved document verifies cleanly and is still
    an invented attribution. That gate is repair-only and died with it.

    The detector never had the exact-match rule and was never meant to.
    `_resolve_doc` matches a 24-character prefix forgivingly on purpose,
    because a human-written answer routinely prints a truncated id — and 182
    corpus ids are <= 24 characters. So four of these five now resolve to the
    real document and come back SUPPORTED; only the case change, which is not
    a prefix relation in either direction, is caught as an invalid citation.

    This test pins that split so the loss is visible and measured rather than
    deleted. What it does NOT say is that the four are safe: a citation the
    answer invented is now indistinguishable, to this pass, from one it
    truncated. Closing that is a scoping question for the `_scoped_field_
    conflict` design pass `docs/wave0c-review-verdict.md` asks for, and it is
    a DETECTOR change, not a repair one."""
    answer = f"FP151 requests **USD 18,500,000** in GCF funding [{cited}]."
    res, _ = run(answer, use_llm=False)
    assert_unchanged(res, answer)
    (v,) = res.verdicts
    if caught:
        assert v.status == V.UNSUPPORTED, f"{label}: {v.status} {v.flags}"
        assert any(f.startswith("invalid-citation") for f in v.flags), v.flags
    else:
        assert v.status == V.SUPPORTED and v.scope == [(DOC, 45)], (
            f"{label} no longer resolves to the held document ({v.status}, "
            f"scope={v.scope}) — if the detector gained an exactness rule, "
            "this expectation is the thing to update, and p2's original "
            "assertion is the one to restore")


def test_p4_attack_a_conflict_must_not_be_escaped_by_re_pointing_the_citation():
    """TARGET: the contradiction path. VARIES: the CITATION, with the figure
    held constant — the axis a figure-level check cannot see.

    The cover-page registry line prints two figures for FP151: 18.5 M USD of
    GCF financing and 28 M USD of TOTAL financing. '[doc, cover pages]' with
    28 million is CONTRADICTED, correctly: the answer put the total-financing
    figure under the GCF-financing label. Move the bracket to p.45 — a page
    that prints 18,500,000 and does not print 28 million anywhere — and the
    wide-scope fallback used to find 28 million elsewhere in the document,
    return SUPPORTED with a 'citation-page-mismatch' caution, and never run
    the conflict test on that branch.

    ON e639915 this was an xfail: a repair could keep the wrong figure, move
    the bracket, and be adopted. The hole was always in the DETECTOR, and
    22f558b closed it there — 'no SUPPORTED without conflict-testing the scope
    it stands on'. Stated directly against `classify_deterministic`, which is
    where it belonged, it needs no repair pass to be true or to be false."""
    orig = f"FP151 requests **USD 28 million** in GCF funding [{DOC}, cover pages]."
    repointed = f"FP151 requests **USD 28 million** in GCF funding [{DOC}, p. 45]."
    ev = base_evidence()

    pre = V.classify_deterministic(V.extract_claims(orig), ev)
    assert pre[0].status == V.CONTRADICTED, "fixture no longer starts in conflict"

    after = V.classify_deterministic(V.extract_claims(repointed), base_evidence())
    assert after[0].status == V.CONTRADICTED, (
        "moving the citation to a page that does not print the figure escaped "
        f"the conflict: status={after[0].status}, flags={after[0].flags} — the "
        "wide-scope fallback returned support without conflict-testing it")
    assert "conflict-elsewhere-in-document" in after[0].flags, after[0].flags

    res, _ = run(repointed, use_llm=False)
    assert_unchanged(res, repointed)
    assert res.status == "abstain", res.status


def test_p6_guard_no_api_key_is_still_deterministic_verdicts(monkeypatch):
    """TARGET: degradation. A keyless deployment must not raise, hang, or
    report anything but what the deterministic layer found. ON 52152af:
    PASSED, and it must keep passing — it is now the whole shape of the pass
    with the judge switched off."""
    monkeypatch.setattr(V, "_client", _REAL_CLIENT)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res = V.verify_answer(EN_BAD, base_evidence())
    assert res.status == "unverified-llm"
    assert_unchanged(res, EN_BAD)
    assert all(v.source == "deterministic" for v in res.verdicts)


def test_p7_guard_the_pass_makes_at_most_one_model_call():
    """TARGET: the call budget. The module's contract was 'at most two LLM
    calls per answer' — one judge, one repair — and the (b)/(c)/(d) gates had
    to be decidable in python, because implementing any of them as another
    model call would have put a third unpinned sample inside the gates that
    existed to remove unpinned samples.

    eac4c94 spends the repair call. The budget is ONE, and asserting the old
    ceiling of two would leave room for exactly the thing that was removed."""
    orig = ("The total GCF funding requested is USD 150 million.\n\n"
            f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 45].")
    res, client = run(orig, judge=_judge((0, "unsupported")))
    assert len(client.calls) == 1, (
        f"{len(client.calls)} model calls for one answer; systems="
        f"{[_system_of(kw)[:40] for kw in client.calls]}")
    assert _system_of(client.calls[0]) == V.ADJUDICATE_PROMPT, (
        "the one remaining call is not the judge: "
        f"{_system_of(client.calls[0])[:80]!r}")
    assert_unchanged(res, orig)
