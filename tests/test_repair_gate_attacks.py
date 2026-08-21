"""Adversarial acceptance suite for the three Wave 4 repair-safety gates.

Written from the SPEC (commit 52152af's findings and blocking prerequisites),
BEFORE and INDEPENDENTLY of the implementation, so it cannot inherit the
implementer's blind spots.

    (a) sampling pinned on the judge and repair calls
    (b) language flips rejected, symmetrically, without false positives
    (c) a minimum-substance floor that fires when the pre-repair answer has no
        supported required claims
    (d) carry-on / carry-off: `_carry_cleared` must stop certifying rewrites
        the honest recheck rejects

plus the perimeter that must not regress while those land.

NAMING IS THE CONTRACT
----------------------
  test_<gate><n>_attack_...  MUST FAIL on 52152af (the gate does not exist) and
                             pass only once the gate is correct.
  test_<gate><n>_guard_...   MUST PASS on 52152af and MUST STILL PASS after.
                             These exist because the cheapest way to satisfy an
                             attack test is to over-correct; a guard fails when
                             the new gate is too strict or too crude.

Every attack is built on a CITED answer whose repair is ADOPTED on 52152af
(`res.repaired is True`), verified by running it — the Wave 4 review found a
test guarding the largest hole that used an uncited answer and therefore
returned before it ever reached the branch it claimed to protect. Each attack
below is proven to reach the adoption branch today.

RUN
    pytest attack_repair_gates.py -q
    GCF_QNA_ROOT=/path/to/checkout pytest attack_repair_gates.py -q

NO NETWORK. `verify._client` is patched to raise for the whole session; every
test hands `verify_answer` an explicit fake client. The two subprocess tests
re-apply the same patch inside the child.

BASELINE, measured on 52152af (git archive of the commit, registry stubbed):
36 tests, 15 fail, 21 pass, 0.25 s. The 15 failures are exactly the attacks:

    a1 a2 a3 a4 a5 a6 a7 a8   no temperature, no seed, and one verdict-level
                              nondeterminism that pinning does not touch (a8)
    b1 b2                     both flip directions adopted
    c1 c2                     the abstention becomes "None." and is adopted
    d1 d2                     the carry certifies both rewrites
    p4                        a contradicted figure escapes by re-pointing

p4 is a PRE-EXISTING hole reachable today with VERIFY_REPAIR=1, not a
regression risk introduced by this wave.
"""
from __future__ import annotations

import json
import os
import re
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


# --------------------------------------------------------------------------
# fixture corpus — the tiny four-key set tests/test_verify.py uses, so every
# verdict below can be checked by reading the fixture
# --------------------------------------------------------------------------
DOC = "124_gcf-b27-02-add11"          # FP151 package
DOC2 = "123_gcf-b27-02-add12"         # FP152 package
DOC220 = "55_gcf-b37-02-add11-funding-proposal-package-fp220"
DOC165 = "165_gcf-b23-02-add04"

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
    """No registry file, no client construction.

    The registry is stubbed empty so a verdict can only come from the held
    evidence — the suite must not change meaning when data/registry.json does.
    `_client` is made to raise so an accidental network path is a loud failure
    rather than a slow one.
    """
    monkeypatch.setattr(_registry, "load", lambda: {})
    monkeypatch.setattr(_registry, "facts", lambda doc: {})
    monkeypatch.setattr(V, "_client", _forbidden_client)


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


def run(answer, repaired, evidence=None, judge=None, **kw):
    """One full verify_answer pass with a canned repair (and optional judge)."""
    ev = base_evidence() if evidence is None else evidence
    replies = ([judge] if judge is not None else []) + [repaired]
    client = FakeClient(*replies)
    use_llm = kw.pop("use_llm", judge is not None)
    res = V.verify_answer(answer, ev, client=client, use_llm=use_llm, **kw)
    return res, client


def assert_rejected(res, original):
    __tracebackhide__ = True
    assert res.repair_rejected and not res.repaired, (
        f"the rewrite was ADOPTED (status={res.status!r}); shipped text:\n"
        f"{res.answer[:300]!r}")
    assert res.answer == original, "a rejected repair must return the original"


def assert_adopted(res, repaired):
    __tracebackhide__ = True
    assert res.repaired and not res.repair_rejected, (
        f"the rewrite was REJECTED (status={res.status!r}) notes={res.notes}")
    assert res.answer == repaired


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
# ==========================================================================

A_BAD = (f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 45].\n\n"
         f"The programme covers **Angola**, **Benin** and **Kenya** "
         f"[{DOC}, cover pages].")
A_FIX = (f"FP151 requests **USD 18,500,000** in GCF funding [{DOC}, p. 45].\n\n"
         f"The programme covers **Angola**, **Benin** and **Kenya** "
         f"[{DOC}, cover pages].")


def test_a1_attack_repair_call_sends_a_seed():
    """TARGET: (a). VARIES: nothing — the request payload of the repair call.
    TODAY: FAILS. 52152af sends only model/max_completion_tokens/messages."""
    res, client = run(A_BAD, A_FIX, use_llm=False)
    assert_adopted(res, A_FIX)                    # the call really happened
    (kw,) = client.calls
    assert "seed" in kw, (
        "the repair call sends no `seed`: the rewrite that replaces "
        f"user-visible text is an unpinned sample. sent keys={sorted(kw)}")
    assert isinstance(kw["seed"], int) and not isinstance(kw["seed"], bool)


def test_a2_attack_repair_call_pins_temperature():
    """TARGET: (a). VARIES: same payload, the other pinning knob.
    TODAY: FAILS — no temperature is sent."""
    res, client = run(A_BAD, A_FIX, use_llm=False)
    assert_adopted(res, A_FIX)
    (kw,) = client.calls
    assert "temperature" in kw, f"no temperature pinned; sent keys={sorted(kw)}"
    assert kw["temperature"] == 0, f"temperature={kw['temperature']!r}, not 0"


def test_a3_attack_judge_call_sends_a_seed(evidence):
    """TARGET: (a). VARIES: the OTHER call — a pinned repair over an unpinned
    judge is still an unpinned adoption decision, because the verdict objects
    the repair prompt is built from are themselves a sample.
    TODAY: FAILS."""
    answer = "The total GCF funding requested is USD 150 million."
    client = FakeClient(_judge((0, "supported")))
    V.classify(V.extract_claims(answer), evidence, client=client)
    assert client.calls, "the judge call did not happen; fixture is stale"
    kw = client.calls[0]
    assert "seed" in kw, f"the judge call sends no `seed`; keys={sorted(kw)}"
    assert isinstance(kw["seed"], int) and not isinstance(kw["seed"], bool)


def test_a4_attack_judge_call_pins_temperature(evidence):
    """TARGET: (a). VARIES: the judge call's temperature. TODAY: FAILS."""
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
    TODAY: FAILS on the presence half; the payload-equality half already holds,
    which is the point — equality is not evidence of pinning on its own."""
    res1, c1 = run(A_BAD, A_FIX, use_llm=False)
    res2, c2 = run(A_BAD, A_FIX, use_llm=False)
    assert_adopted(res1, A_FIX)
    assert_adopted(res2, A_FIX)
    kw1, kw2 = c1.calls[0], c2.calls[0]
    assert kw1 == kw2, (
        "two identical inputs produced different repair requests:\n"
        f"  only in run 1: { {k: v for k, v in kw1.items() if kw2.get(k) != v} }\n"
        f"  only in run 2: { {k: v for k, v in kw2.items() if kw1.get(k) != v} }")
    assert {"seed", "temperature"} <= set(kw1), (
        "the two requests are equal but neither is pinned; "
        f"keys={sorted(kw1)}")
    assert (res1.status, res1.answer, res1.repaired) == \
           (res2.status, res2.answer, res2.repaired)


def test_a6_attack_the_seed_is_stable_across_processes():
    """TARGET: (a). VARIES: the interpreter's PYTHONHASHSEED.

    A seed derived from `hash(answer)` looks pinned in one process and is a
    different sample in the next, because str hashing is randomised per
    process. Two children, two hash seeds, same input.
    TODAY: FAILS — no seed is sent at all (both children report null)."""
    body = f"""
        ev = {{({DOC!r}, None): {REGISTRY_LINE!r},
              ({DOC!r}, 45): {PAGE45!r}}}
        c = FakeClient({A_FIX!r})
        V.verify_answer({A_BAD!r}, ev, client=c, use_llm=False)
        kw = c.calls[0]
        print(json.dumps({{"seed": kw.get("seed"),
                           "temperature": kw.get("temperature"),
                           "keys": sorted(kw)}}))
    """
    a = _child(body, 0)
    b = _child(body, 987654321)
    assert a["seed"] is not None, (
        f"no seed reaches the endpoint; the repair call sent {a['keys']}")
    assert a["seed"] == b["seed"], (
        f"the seed depends on the process: {a['seed']} vs {b['seed']} — a "
        "hash()-derived seed is not a pinned sample")


def test_a7_attack_an_endpoint_that_rejects_seed_must_not_pin_silently():
    """TARGET: (a), third question. VARIES: the endpoint, not the input.

    LM Studio / vLLM / Azure 400 on an unrecognised sampling parameter. The
    failure mode to forbid is a silent one: the seed is refused, the pass
    retries without it, and the rewrite is adopted looking exactly as pinned
    as one that was. Any of three behaviours is acceptable — refuse the
    repair, keep the deterministic verdicts, or retry and SAY SO — and one is
    not: adopt without a word.
    TODAY: FAILS. No seed is ever sent, so nothing is refused, temperature is
    absent, and the rewrite is adopted with the ordinary note."""
    ev = base_evidence()
    client = FakeClient(A_FIX, A_FIX, reject=("seed",))
    res = V.verify_answer(A_BAD, ev, client=client, use_llm=False)

    assert any("seed" in kw for kw in client.calls), (
        "no call ever attempted to send a `seed`, so the endpoint had nothing "
        f"to reject; calls={[sorted(kw) for kw in client.calls]}")
    assert len(client.calls) <= 4, "unbounded retry against a 400"
    assert all(kw.get("temperature") == 0 for kw in client.calls), (
        "a retry after the seed was refused dropped temperature pinning too; "
        f"temperatures={[kw.get('temperature') for kw in client.calls]}")
    if res.repaired:
        assert any(re.search(r"seed|pin|unpinn|sampl|determinis", n, re.I)
                   for n in res.notes), (
            "a rewrite produced by an unpinned call was adopted with no note "
            f"saying so: notes={res.notes}")


def test_a8_attack_the_adoption_decision_does_not_depend_on_pythonhashseed():
    """TARGET: (a), second question — 'can you construct an input where it
    still varies?'. VARIES: PYTHONHASHSEED, with sampling irrelevant (the
    canned client returns the same bytes every time).

    `_resolve_doc` picks the first prefix match while iterating a SET of held
    document ids. Two held documents sharing a 24-character prefix — the shape
    the corpus is full of, `..add11` / `..add12` — resolve differently in
    different processes, so a CITED claim is supported in one run and
    contradicted in the next, and the adoption decision follows.

    This is the finding pinning temperature and seed does NOT fix: the
    18-27% adoption flip is attributed in the Wave 4 message entirely to
    `_complete`, and part of it lives in the deterministic layer.
    TODAY: FAILS — 'verified' under hash seeds 0-2, 'abstain' under 3-7."""
    a, b = "124_gcf-b27-02-add11", "124_gcf-b27-02-add12"
    answer = "FP151 requests **USD 18.5 million** in GCF funding " \
             "[124_gcf-b27-02-add1, p. 5]."          # a truncated id
    body = f"""
        ev = {{({a!r}, 5): "Total GCF funding requested: USD 18.5 million",
              ({b!r}, 5): "Total GCF funding requested: USD 150 million"}}
        c = FakeClient({answer!r})
        r = V.verify_answer({answer!r}, ev, client=c, use_llm=False)
        print(json.dumps({{"status": r.status, "repaired": r.repaired,
                           "answer": r.answer}}))
    """
    seen = {}
    for hs in (0, 1, 2, 3, 4, 5, 6, 7):
        got = _child(body, hs)
        seen.setdefault((got["status"], got["repaired"]), []).append(hs)
    assert len(seen) == 1, (
        "identical answer, identical evidence, identical canned completion — "
        f"and the outcome depends on the interpreter's hash seed: {seen}")


# ==========================================================================
# (b) LANGUAGE FLIPS MUST BE REJECTED
#
# Wave 4: "An English answer was rewritten into French in 2 of 3 samples and
# adopted both times; REPAIR_PROMPT rule 5 has no gate behind it."
#
# Every fixture below is a two-paragraph answer, so a detector has real signal
# to work with, and every repair is otherwise PERFECT: correct figure, correct
# citation, substance preserved, nothing introduced. The only thing that can
# decide b1/b2 is the language; the only thing that can decide b3/b4 is that
# it did NOT change.
# ==========================================================================

EN_BAD = (f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 45].\n\n"
          f"The programme covers **Angola**, **Benin** and **Kenya** "
          f"[{DOC}, cover pages].")
EN_FIX = (f"FP151 requests **USD 18,500,000** in GCF funding [{DOC}, p. 45].\n\n"
          f"The programme covers **Angola**, **Benin** and **Kenya** "
          f"[{DOC}, cover pages].")
FR_BAD = (f"La FP151 demande **25 millions USD** de financement du GCF "
          f"[{DOC}, p. 45].\n\n"
          f"Le programme couvre l'**Angola**, le **Bénin** et le **Kenya** "
          f"[{DOC}, cover pages].")
FR_FIX = (f"La FP151 demande **18,5 millions USD** de financement du GCF "
          f"[{DOC}, p. 45].\n\n"
          f"Le programme couvre l'**Angola**, le **Bénin** et le **Kenya** "
          f"[{DOC}, cover pages].")


def test_b1_attack_english_answer_rewritten_into_french_is_rejected():
    """TARGET: (b). VARIES: the language of the rewrite, and only that —
    b3 is the same pair with the rewrite left in English.
    TODAY: FAILS. status='repaired', the user is handed a French answer."""
    res, _ = run(EN_BAD, FR_FIX, use_llm=False)
    assert_rejected(res, EN_BAD)


def test_b2_attack_french_answer_rewritten_into_english_is_rejected():
    """TARGET: (b), symmetry. VARIES: the direction of the flip.

    A gate written as 'reject when the rewrite looks French' passes b1 and
    fails here. French is the corpus's second answer language, not a
    second-class one.
    TODAY: FAILS. status='repaired'."""
    res, _ = run(FR_BAD, EN_FIX, use_llm=False)
    assert_rejected(res, FR_BAD)


def test_b3_guard_english_answer_repaired_in_english_is_still_adopted():
    """TARGET: (b) over-correction. VARIES: nothing but the rewrite's
    language, against b1. A gate that rejects b1 by rejecting everything
    fails here. TODAY: PASSES, and must keep passing."""
    res, _ = run(EN_BAD, EN_FIX, use_llm=False)
    assert_adopted(res, EN_FIX)


def test_b4_guard_french_answer_repaired_in_french_is_still_adopted():
    """TARGET: (b) over-correction, the French side. TODAY: PASSES."""
    res, _ = run(FR_BAD, FR_FIX, use_llm=False)
    assert_adopted(res, FR_FIX)


def _fp220_evidence():
    return {
        (DOC220, None): ('Registry — FP220: "Climate Resilient Livelihoods"; '
                         'accredited entity: Save the Children Australia; '
                         'countries: Vanuatu; GCF financing (as printed): '
                         f'26.8 M USD; board B.37, 2023 [{DOC220}, cover pages]'),
        (DOC220, 12): ("### (a) Requested GCF funding (Total amount)\n"
                       "| (vi) Grants | 26,800,000 | 7 | |\n"
                       "Accredited entity: Save the Children Australia\n"
                       "Delivery partner: Green Climate Fund Secretariat"),
    }


def test_b5_guard_french_answer_full_of_english_proper_nouns_is_not_a_flip():
    """TARGET: (b) false positives. VARIES: the English CONTENT of a French
    answer, not its language.

    'Save the Children Australia', 'Green Climate Fund Secretariat' and
    `55_gcf-b37-02-add11-funding-proposal-package-fp220` are English tokens a
    French answer is REQUIRED to print verbatim — translating an accredited
    entity's registered name or a document id would be the bug. A whole-token
    or bag-of-words detector calls this answer English; both texts here are
    French, and neither the original nor the repair flips.
    TODAY: PASSES. Must keep passing."""
    ev = _fp220_evidence()
    bad = (f"L'entité accréditée de la FP220 est **Save the Children Australia**, "
           f"avec le **Green Climate Fund Secretariat** comme partenaire de mise "
           f"en œuvre [{DOC220}, p. 12].\n\n"
           f"La FP220 demande **31,4 millions USD** de financement du GCF "
           f"[{DOC220}, p. 12].")
    fix = (f"L'entité accréditée de la FP220 est **Save the Children Australia**, "
           f"avec le **Green Climate Fund Secretariat** comme partenaire de mise "
           f"en œuvre [{DOC220}, p. 12].\n\n"
           f"La FP220 demande **26,8 millions USD** de financement du GCF "
           f"[{DOC220}, p. 12].")
    res, _ = run(bad, fix, evidence=ev, use_llm=False)
    assert_adopted(res, fix)


FR_TITLE = "Programme de resilience climatique des communautes rurales du Sahel"


def _french_title_evidence():
    return {
        (DOC, None): REGISTRY_LINE,
        (DOC, 45): PAGE45,
        (DOC, 3): (f"Titre du programme : « {FR_TITLE} »\n"
                   "Entite accreditee : International Union for Conservation "
                   "of Nature"),
    }


def test_b6_guard_english_answer_quoting_a_french_title_is_not_a_flip():
    """TARGET: (b) false positives. VARIES: the language of a QUOTED title
    inside an English answer, held constant across the repair.

    A French document title quoted in an English answer must survive
    untranslated — it is the document's name. The repair changes only the
    figure. TODAY: PASSES. Must keep passing."""
    ev = _french_title_evidence()
    bad = (f"The programme is titled « {FR_TITLE} » [{DOC}, p. 3].\n\n"
           f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 45].")
    fix = (f"The programme is titled « {FR_TITLE} » [{DOC}, p. 3].\n\n"
           f"FP151 requests **USD 18,500,000** in GCF funding [{DOC}, p. 45].")
    res, _ = run(bad, fix, evidence=ev, use_llm=False)
    assert_adopted(res, fix)


def test_b7_guard_a_mostly_numeric_answer_is_not_a_flip():
    """TARGET: (b) false positives. VARIES: the amount of language available
    to detect — a two-row table where the only words are a field label.

    Language detection over this text is a coin toss, and a gate that treats
    'undetermined' as 'flipped' silently disables repair for every table
    answer in the corpus. TODAY: PASSES. Must keep passing."""
    bad = (f"| Requested GCF funding | USD 25,000,000 | [{DOC}, p. 45] |\n"
           f"| Grants | 18,500,000 | [{DOC}, p. 45] |")
    fix = (f"| Requested GCF funding | USD 18,500,000 | [{DOC}, p. 45] |\n"
           f"| Grants | 18,500,000 | [{DOC}, p. 45] |")
    res, _ = run(bad, fix, use_llm=False)
    assert_adopted(res, fix)


FR_Q = ("Le programme vise a renforcer la resilience climatique des communautes "
        "rurales du Sahel par des investissements dans l'agriculture, l'eau et "
        "les services meteorologiques, en partenariat avec les autorites locales")
EN_Q = ("The programme aims to strengthen the climate resilience of rural "
        "communities in the Sahel through investments in agriculture, water and "
        "weather services, in partnership with local authorities")


def test_b8_guard_language_change_inside_a_quotation_only_is_not_a_flip():
    """TARGET: (b). VARIES: the language of a QUOTATION that carries ~60% of
    the answer's characters, while the prose around it stays English.

    The document prints the passage bilingually; the repair quotes the English
    rendering instead of the French one. The ANSWER's language never changes,
    so the gate — whose harm model is 'the user asked in English and was
    answered in French' — must not fire. A detector run over the whole string
    sees a French->English flip and rejects.

    SPEC AMBIGUITY, reported not resolved: an equally defensible reading is
    that re-rendering a quotation is a source-fidelity violation and should be
    rejected — for quote fidelity, not for language. This test asserts the
    reading that keeps the language gate precise; see the report.
    TODAY: PASSES."""
    ev = {(DOC, None): REGISTRY_LINE, (DOC, 45): PAGE45,
          (DOC, 3): (f"Objectif du programme : « {FR_Q} ».\n"
                     f"Programme objective: “{EN_Q}”.")}
    bad = (f"The stated objective is « {FR_Q} » [{DOC}, p. 3].\n\n"
           f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 45].")
    fix = (f"The stated objective is “{EN_Q}” [{DOC}, p. 3].\n\n"
           f"FP151 requests **USD 18,500,000** in GCF funding [{DOC}, p. 45].")
    res, _ = run(bad, fix, evidence=ev, use_llm=False)
    assert_adopted(res, fix)


# ==========================================================================
# (c) MINIMUM-SUBSTANCE FLOOR
#
# Wave 4: "abs-2014's 351-character registry-backed abstention became the
# single word 'None.' — the anti-gutting guard cannot fire when
# _supported_required == 0 before repair, which is exactly the population
# repair is invoked on."
#
# Every fixture in this block has _supported_required == 0 BEFORE repair, so
# the existing guard is provably out of the picture and only a new floor can
# decide the outcome.
# ==========================================================================

def _abs2014_evidence():
    return {
        V.NOTES_KEY: YEAR_NOTE,
        (DOC165, 4): ("The REDD+ working group met in Ecuador in 2014 and again "
                      "in 2016 to prepare the results-based payment request."),
    }


ABS2014 = (
    'None.\n\nThe authoritative registry note states that **2014 has no '
    'registered proposals** (and that there was **no board meeting that year '
    'in this corpus**), so there were **no GCF funding proposals approved in '
    '2014** [Note (computed from the corpus registry)].\n\n'
    'Some retrieved documents mention activities occurring in **2014** (e.g., '
    f'REDD+ working groups in **Ecuador**), but those are not approvals '
    f'[{DOC165}, p. 9].')


def test_c1_attack_the_abs_2014_abstention_must_not_become_the_word_none():
    """TARGET: (c), the literal Wave 4 case. VARIES: nothing — this is the
    recorded regression, rebuilt from data/eval/release_repair-fresh.jsonl.

    A 429-character registry-backed abstention, cited to the computed note and
    to a retrieved document, one claim of which fails. Before repair the
    answer has ZERO supported REQUIRED claims (its supported claim is a year
    claim, and `Claim.required` is money|number|entity), so the anti-gutting
    guard is structurally unable to fire.
    TODAY: FAILS. status='repaired', res.answer == 'None.'"""
    ev = _abs2014_evidence()
    pre = V.classify_deterministic(V.extract_claims(ABS2014), ev)
    assert V._supported_required(pre) == 0, "fixture no longer exercises the hole"
    assert any(v.failed for v in pre), "fixture no longer invokes repair"
    res, _ = run(ABS2014, "None.", evidence=ev, use_llm=False)
    assert_rejected(res, ABS2014)


C2_ORIG = ('The authoritative registry note states that **2014 has no registered '
           'proposals** and that there was **no board meeting that year in this '
           'corpus** [Note (computed from the corpus registry)].\n\n'
           'A REDD+ working group met in **Ecuador** in 2014 to prepare the '
           f'results-based payment request [{DOC165}, p. 9].')
C2_GUT = ('**None.** Beyond that, the retrieved excerpts do not state anything '
          'further about the matter, and no additional value is available from '
          'the context provided here [Note (computed from the corpus registry)].')

C3_ORIG = (f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 45].\n\n"
           f"Its accredited entity is **Pegasus Capital Advisors LP** "
           f"[{DOC}, p. 45].")
C3_FIX = f"FP151 requests **USD 18,500,000** in GCF funding [{DOC}, p. 45]."


def test_c2_attack_a_gutting_padded_past_a_length_ratio_is_still_gutting():
    """TARGET: (c). VARIES: the SHAPE of the deletion, not its size.

    The rewrite keeps 65% of the original's characters and 31 words, and
    carries a citation bracket — every naive floor (>=25% of length, >=10
    words, 'still cites something') admits it. It states nothing: zero claims
    survive extraction. The original's substance — the registry finding and
    the Ecuador working group — is gone.
    TODAY: FAILS. status='repaired'."""
    ev = _abs2014_evidence()
    pre = V.classify_deterministic(V.extract_claims(C2_ORIG), ev)
    assert V._supported_required(pre) == 0, "fixture no longer exercises the hole"
    res, _ = run(C2_ORIG, C2_GUT, evidence=ev, use_llm=False)
    assert_rejected(res, C2_ORIG)


def test_c3_guard_a_legitimately_one_sentence_repair_is_still_adopted():
    """TARGET: (c) over-correction. VARIES: what the short text CONTAINS.

    Both of this answer's claims fail, so _supported_required == 0 before
    repair and the new floor is live. The correct answer really is one
    sentence: the figure the page prints, with the page cited. A floor that
    blocks this blocks the repair pass's best case.
    TODAY: PASSES. Must keep passing."""
    pre = V.classify_deterministic(V.extract_claims(C3_ORIG), base_evidence())
    assert V._supported_required(pre) == 0, "fixture no longer exercises the hole"
    res, _ = run(C3_ORIG, C3_FIX, use_llm=False)
    assert_adopted(res, C3_FIX)


def test_c4_guard_no_length_ratio_can_separate_c2_from_c3():
    """TARGET: (c), the discriminator. VARIES: nothing — it measures the two
    fixtures above and asserts a length floor is the wrong instrument.

    c2 must be REJECTED while retaining 65% of its original's characters;
    c3 must be ADOPTED while retaining 47%. Any floor of the form
    'len(new) >= k * len(old)' either admits c2 or blocks c3. The floor has
    to read what survived, not how much of it there is.
    TODAY: PASSES (it is an arithmetic property of the fixtures) — it exists
    so that a length-ratio implementation cannot claim c2 and c3 as a pair."""
    r2 = len(C2_GUT) / len(C2_ORIG)
    r3 = len(C3_FIX) / len(C3_ORIG)
    assert r2 > r3, (
        f"fixtures drifted: the rewrite that must be rejected retains {r2:.0%} "
        f"and the one that must be adopted retains {r3:.0%}; regenerate them "
        "so the rejected one is the LONGER of the two")


def test_c5_guard_an_answer_that_shrinks_by_dropping_a_fabrication_is_adopted():
    """TARGET: (c) over-correction. VARIES: why the answer shrank.

    Paragraph one is supported and cited; paragraph two is a fabricated
    co-financing figure. Deleting the fabrication is the repair pass working
    exactly as designed, and it costs 46% of the answer's length.
    TODAY: PASSES. Must keep passing."""
    orig = (f"FP151 requests **USD 18,500,000** in GCF funding [{DOC}, p. 45].\n\n"
            f"It is matched by **USD 40 million** of co-financing from the "
            f"private sector, disbursed over seven years [{DOC}, p. 45].")
    fix = f"FP151 requests **USD 18,500,000** in GCF funding [{DOC}, p. 45]."
    assert len(fix) / len(orig) < 0.6
    res, _ = run(orig, fix, use_llm=False)
    assert_adopted(res, fix)


def test_c6_guard_repair_may_still_delete_an_unsupportable_claim_outright():
    """TARGET: (c) over-correction, against the SHIPPED contract.

    `repair`'s docstring and tests/test_verify.py both state that removing an
    unsupportable claim is a valid repair: what is left states no fact, so
    nothing is left to contradict the evidence. The original here has no
    supported claim of any kind, so the floor must not fire.

    A floor written as 'the repaired answer must contain a supported claim'
    breaks this. TODAY: PASSES. Must keep passing."""
    orig = f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 99]."
    fix = "The retrieved excerpts do not state FP151's GCF funding."
    res, _ = run(orig, fix, use_llm=False)
    assert_adopted(res, fix)
    assert V.extract_claims(res.answer) == []


# ==========================================================================
# (d) CARRY-ON vs CARRY-OFF
#
# Wave 4: "_carry_cleared systematically adopts rewrites the honest recheck
# rejects (carry-on vs carry-off differ 1-2 answers per replay, always in that
# direction) — the self-certification pattern the plan forbids."
# Prerequisite: "state and enforce a carry-on/carry-off tolerance or ship
# carry-off."
# ==========================================================================

def test_d1_attack_an_adoption_resting_entirely_on_carried_verdicts():
    """TARGET: (d). VARIES: the FRACTION of the adoption the carry is holding
    up — here 100%.

    The judge clears two uncited sentences on the unrepaired answer. The
    repair deletes the one cited claim and returns those two sentences
    untouched. The honest deterministic recheck fails BOTH of them ('no
    citation on a factual claim'); `_carry_cleared` restores both to supported
    and the pass adopts. Nothing in the shipped answer was verified by
    anything except the judge's opinion of a different text.

    Either shipped choice satisfies this: carry-off rejects it, and a declared
    carry-on tolerance cannot be wide enough to cover 100% without being
    named — so an adoption here must at minimum SAY that it rests on the
    carry.
    TODAY: FAILS. status='repaired', notes=['repaired 1 failing claim(s)'],
    every verdict source='llm'."""
    orig = ("The requested GCF funding is USD 150 million.\n\n"
            "The accredited entity is Pegasus Capital Advisors LP.\n\n"
            f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 45].")
    fix = ("The requested GCF funding is USD 150 million.\n\n"
           "The accredited entity is Pegasus Capital Advisors LP.")
    res, _ = run(orig, fix, judge=_judge((0, "supported"), (1, "supported")))
    if res.repaired:
        carried = [v for v in res.verdicts if v.source == "llm"]
        assert len(carried) < len(res.verdicts), (
            "every claim in the adopted answer is supported only by a carried "
            "judge ruling: the pass certified itself")
        assert any(re.search(r"carr|judge|tolerance|self-cert", n, re.I)
                   for n in res.notes), (
            f"adopted on carried verdicts with nothing saying so: {res.notes}")
    else:
        assert res.answer == orig


def test_d2_attack_the_carry_must_not_survive_a_change_of_scope():
    """TARGET: (d). VARIES: the claim's EVIDENCE POINTER while its text is
    byte-identical — the axis `_carry_cleared`'s `norm_text` key cannot see.

    'The requested GCF funding is USD 150 million.' is uncited in the
    original, so the judge rules on it against p.5 of the OTHER document,
    where 150 million is printed, and clears it. The repair merges it into the
    same paragraph as the cited sentence — citations are inherited within a
    paragraph — so after the rewrite the identical sentence points at
    [{DOC}, p. 45], which prints 18,500,000. The recheck fails it. The carry
    keys on normalised claim text, matches, and restores 'supported'.

    The pass ships 'the requested GCF funding is USD 150 million' cited to a
    page printing 18,500,000, marked verified. A tolerance on how MANY
    verdicts may be carried does not make this one sound: the judge never
    ruled on this claim against this evidence.
    TODAY: FAILS. status='repaired'."""
    orig = ("The requested GCF funding is USD 150 million.\n\n"
            f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 45].")
    fix = ("The requested GCF funding is USD 150 million.\n"
           f"FP151 requests **USD 18,500,000** in GCF funding [{DOC}, p. 45].")
    # the fixture only means anything if the honest recheck really fails it
    honest = V.classify_deterministic(V.extract_claims(fix), base_evidence())
    assert any(v.failed for v in honest), "fixture no longer needs the carry"
    res, _ = run(orig, fix, judge=_judge((0, "supported")))
    assert_rejected(res, orig)


def test_d3_guard_a_rejected_repair_still_reports_the_judge_s_clearance():
    """TARGET: (d) over-correction. VARIES: the REJECTION path's verdicts.

    Shipping carry-off is allowed, and it will reject repairs carry-on
    adopted. What it must not do is tell the user a claim failed that the
    judge cleared on the answer they are actually being shown: the rejected
    path returns the ORIGINAL answer, so it must return the ORIGINAL verdicts,
    judge rulings included.
    TODAY: PASSES. Must keep passing under either choice."""
    orig = ("The requested GCF funding is USD 150 million.\n\n"
            f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 45].")
    # a rewrite that fails for an unrelated, unambiguous reason
    fix = (f"FP151 requests **USD 18,500,000** in GCF funding "
           f"[999_gcf-b99-99-add99, p. 3].")
    res, _ = run(orig, fix, judge=_judge((0, "supported")))
    assert_rejected(res, orig)
    cleared = [v for v in res.verdicts
               if "150 million" in v.claim.text and v.status == V.SUPPORTED]
    assert cleared, (
        "the judge cleared this claim on the answer being shown, and the "
        f"rejected path reports {[ (v.claim.text[:40], v.status) for v in res.verdicts ]}")


# ==========================================================================
# PERIMETER — must not regress while (a)-(d) land
# ==========================================================================

def test_p1_guard_a_clean_repair_with_no_remaining_failures_is_adopted():
    """TARGET: the adoption path itself. Same language, substance kept,
    nothing introduced, zero failures left. If this stops passing, the gates
    have turned the repair pass off rather than made it safe.
    TODAY: PASSES."""
    res, _ = run(EN_BAD, EN_FIX, use_llm=False)
    assert_adopted(res, EN_FIX)
    assert res.status == "repaired" and not res.failures


@pytest.mark.parametrize("cited,label", [
    (f"{DOC}-annex-volume-2, p. 45", "long suffix"),
    (f"{DOC}0, p. 45", "one-character suffix"),
    (f"{DOC}, p. 45; {DOC}-v2, p. 45", "fake chained after a real one"),
    (f"{DOC[:-1]}, p. 45", "truncated prefix of a real id"),
    (f"{DOC.upper()}, p. 45", "case-changed id"),
])
def test_p2_guard_a_doc_id_that_is_not_held_exactly_is_rejected(cited, label):
    """TARGET: exact-match source introduction. VARIES: the SHAPE of the
    near-miss — suffix, one character, chained behind a real id, a truncated
    prefix, and case. `_resolve_doc` matches a 24-character prefix forgivingly
    and 182 corpus ids are <= 24 characters, so a prefix rule here would wave
    all five through.
    TODAY: PASSES all five."""
    orig = f"FP151 requests **USD 58 million** in GCF funding [{DOC}, p. 45]."
    fix = f"FP151 requests **USD 18,500,000** in GCF funding [{cited}]."
    res, _ = run(orig, fix, use_llm=False)
    assert_rejected(res, orig)
    assert any("introduced sources" in n for n in res.notes), res.notes


def test_p3_guard_anti_gutting_still_fires_when_supported_claims_existed():
    """TARGET: the existing anti-gutting guard. The population where
    _supported_required > 0 must keep behaving exactly as it does.
    TODAY: PASSES."""
    orig = (f"FP151 requests **USD 18,500,000** in GCF funding [{DOC}, p. 45].\n\n"
            f"Its accredited entity is **Pegasus Capital Advisors LP** "
            f"[{DOC}, cover pages].")
    res, _ = run(orig, "The retrieved excerpts do not state FP151's funding.",
                 use_llm=False)
    assert_rejected(res, orig)
    assert res.status == "partial"


@pytest.mark.xfail(strict=False, reason="open hole: the wide-scope fallback runs no conflict test, so a wrong figure re-pointed to another page is adopted. Scheduled as its own design pass (wave0b N2 / wave0c 2 and 7) — turns green when fixed.")
def test_p4_attack_a_conflict_must_not_be_escaped_by_re_pointing_the_citation():
    """TARGET: the contradiction path. VARIES: the CITATION, with the figure
    held constant — the axis a figure-level check cannot see.

    The cover-page registry line prints two figures for FP151: 18.5 M USD of
    GCF financing and 28 M USD of TOTAL financing. '[doc, cover pages]' with
    28 million is CONTRADICTED, correctly: the answer put the total-financing
    figure under the GCF-financing label. The repair keeps the wrong figure
    and moves the bracket to p.45 — a page that prints 18,500,000 and does not
    print 28 million anywhere. The wide-scope fallback finds 28 million
    elsewhere in the document, returns SUPPORTED with a
    'citation-page-mismatch' caution, and the conflict test never runs on that
    branch. The pass adopts, and the user is shown the contradicted figure
    marked repaired.

    TODAY: FAILS — and this is a PRE-EXISTING hole, not a regression risk.
    Reported as a finding: it is reachable today with VERIFY_REPAIR=1."""
    orig = f"FP151 requests **USD 28 million** in GCF funding [{DOC}, cover pages]."
    fix = f"FP151 requests **USD 28 million** in GCF funding [{DOC}, p. 45]."
    pre = V.classify_deterministic(V.extract_claims(orig), base_evidence())
    assert pre[0].status == V.CONTRADICTED, "fixture no longer starts in conflict"
    res, _ = run(orig, fix, use_llm=False)
    assert_rejected(res, orig)


def test_p5_guard_one_wrong_figure_swapped_for_another_is_still_rejected():
    """TARGET: the 'fewer failing claims, and none left failing' rule.
    TODAY: PASSES."""
    orig = f"FP151 requests **USD 58 million** in GCF funding [{DOC}, cover pages]."
    fix = f"FP151 requests **USD 61 million** in GCF funding [{DOC}, cover pages]."
    res, _ = run(orig, fix, use_llm=False)
    assert_rejected(res, orig)
    assert "still fail verification" in " ".join(res.notes)


def test_p6_guard_no_api_key_is_still_deterministic_verdicts_and_no_repair(
        monkeypatch):
    """TARGET: degradation. The new gates must not make a keyless deployment
    raise, hang or silently repair. TODAY: PASSES."""
    monkeypatch.setattr(V, "_client", _REAL_CLIENT)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    res = V.verify_answer(EN_BAD, base_evidence())
    assert res.status == "unverified-llm"
    assert res.answer == EN_BAD and not res.repaired
    assert all(v.source == "deterministic" for v in res.verdicts)


def test_p7_guard_the_gates_add_no_third_llm_call():
    """TARGET: the call budget. A language gate, a substance floor and a
    carry rule are all decidable in python; implementing any of them as
    another model call would put a third unpinned sample inside the gate that
    exists to remove unpinned samples — and would break the module's 'at most
    two LLM calls per answer' contract.
    TODAY: PASSES (2 calls). Must keep passing."""
    orig = ("The total GCF funding requested is USD 150 million.\n\n"
            f"FP151 requests **USD 25 million** in GCF funding [{DOC}, p. 45].")
    res, client = run(orig, EN_FIX, judge=_judge((0, "unsupported")))
    assert len(client.calls) <= 2, (
        f"{len(client.calls)} model calls for one answer; systems="
        f"{[_system_of(kw)[:40] for kw in client.calls]}")
    assert res is not None
