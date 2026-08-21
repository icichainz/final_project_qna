#!/usr/bin/env python3
"""What the constrained repair pass does to a recorded run — A/B over IDENTICAL
answers, so the only thing that differs between the arms is repair itself.

    # 1. replay: one judge pass, one repair call, three scorings
    venv/bin/python scripts/audit_repair.py replay \
        --release data/eval/release_release-2.jsonl \
        --out-prefix data/eval/replay

    # 2. audit: pure file comparison, zero API calls
    venv/bin/python scripts/audit_repair.py \
        --off data/eval/replay_repair-off.jsonl \
        --on  data/eval/replay_repair-on-carryoff.jsonl \
        --carry-on data/eval/replay_repair-on.jsonl

WHY THE ARMS ARE REPLAYED AND NOT REGENERATED
---------------------------------------------
Two fresh ``--release`` runs are two samples of the answer distribution before
they are anything about repair. Wave 3 measured that distribution directly:
claim support moves 8.9 pp deterministic / 4.6 pp production between runs of
ONE pinned configuration on ONE tree. A repair effect smaller than that is
invisible in a generated A/B, and a repair effect larger than it is still
unattributable.

So the answers are fixed. A recorded release run carries ``raw_answer`` (the
text the model wrote, before the verifier touched it), the hit TEXT and the
note blocks — everything ``verify.build_evidence`` needs — so the turn's
evidence is rebuilt verbatim and the same answer is pushed through the
verifier twice.

ONE JUDGE PASS, SHARED. The judge is an LLM too, and ``verify_answer`` calls
it before repair is even considered. Calling ``verify_answer`` once per arm
would give the two arms two independent judge samples and put a second
sampling difference back into the comparison. Instead this replays the pass
in its parts, exactly as ``verify_answer`` composes them:

    claims   = verify.extract_claims(raw_answer)
    verdicts = verify.adjudicate(verify.classify_deterministic(claims, ev), ev)
    OFF      = the verdicts as they stand              (verify_answer allow_repair=0)
    ON       = verify.repair(raw_answer, verdicts, ev) (verify_answer allow_repair=1)

Both arms are handed the SAME ``verdicts`` object. The judge's sample is
therefore common to both and cancels; the repair call is the only difference.

ONE REPAIR CALL, TWO SCORINGS (carry-on / carry-off). ``verify._carry_cleared``
copies pre-repair judge SUPPORTED rulings onto the post-repair verdicts by
normalised claim text, so an adoption decision can be carried by the judge's
opinion of the UNREPAIRED answer. The plan gates on the carry-OFF number and
reports carry-on as the production-behaviour figure. Running ``repair()`` twice
would sample the repair model twice, so the repair completion is CAPTURED on
the first (carry-on) pass and REPLAYED from a canned client on the second, with
``_carry_cleared`` patched to the identity for the duration. The repaired text
is byte-identical across the two scorings; only the adoption rule differs.

Nothing in ``src/`` is modified. The patch is a context manager over this
process's ``verify._carry_cleared`` binding, restored on exit, and it is
asserted restored.

WHAT THE AUDIT CHECKS, AND WHY IT DOES NOT TRUST THE REPAIR GATES
----------------------------------------------------------------
``verify.repair`` already refuses a rewrite that introduces a source
(``_introduced_sources``), that leaves any claim failing, or that deletes every
supported claim. Those are the gates whose CORRECTNESS is the question, so the
audit re-derives the same facts from the recorded text with its own code:

  invented_docs     a document id cited in the repaired answer that the turn's
                    evidence does not hold at all
  invented_pages    a (doc, page) cited in the repaired answer that the turn's
                    evidence does not hold
  invented_sources  a doc the repair pass was never SHOWN (the ``allowed`` set
                    ``repair()`` computes) — a claim moved onto another
                    retrieved document verifies cleanly and is still invented
  invented_figures  a numeric token in the repaired answer that is neither in
                    the raw answer nor printed anywhere in the held evidence,
                    checked TWICE: once through verify's own amount matcher
                    and once as a literal digit string over the evidence blob,
                    because a forgiving matcher is exactly what would hide one
  invented_entities a name new in the repaired answer that verify's entity
                    matcher cannot find anywhere in the held evidence
  novel_sentences   every sentence present after and absent before, with its
                    groundedness — the human-readable residue behind the above

and the claim-level movement:

  lost_claims       every claim present before and absent after, WITH its
                    pre-repair status and groundedness (the plan requires the
                    unsupported ones listed too, not only the supported ones)
  corrected / deleted / qualified / retained — a failing claim whose successor
                    sentence is SUPPORTED was corrected; one with no successor
                    was deleted; deletion of a GROUNDED claim is called out
                    separately, because the evidence permitted a correction

REPRODUCIBILITY IS A MEASURED, GATED PROPERTY
---------------------------------------------
Wave 4's finding was not a number in the table: it was that the table moves.
From the same answers and the SAME verdict objects, the adoption decision
flipped on 3 of 11 sampled cases and only 1 of 11 rewrites came back
byte-identical. A treatment that is not a fixed quantity cannot be gated by
one measurement of it, and an instrument that cannot see that is an
instrument that reports luck.

    # measure it: N repair samples per failing case, each fully scored
    venv/bin/python scripts/audit_repair.py replay \
        --release data/eval/release_release-2.jsonl \
        --out-prefix data/eval/replay --repair-samples 3

    # gate on it, zero API calls, from the recorded arms
    venv/bin/python scripts/audit_repair.py \
        --off data/eval/replay_repair-off.jsonl \
        --on  data/eval/replay_repair-on-carryoff.jsonl \
        --carry-on data/eval/replay_repair-on.jsonl \
        --max-adoption-disagreement 0 \
        --require-metrics groundedness_rate,citation_completeness_rate

``reproducibility`` reports (i) the identical-completion rate, (ii) the
adoption-decision agreement rate, (iii) every disagreeing case with both
decisions and both rejection notes, and — through ``sample_worlds`` — (iv)
what the disagreements do to every gated metric: world k is the run this
would have been had each case used its k-th sample, scored by the same
comparison code, so 'groundedness in world 2' and 'groundedness' are the same
measurement.

``--max-adoption-disagreement R`` gates on (ii). It defaults to 0 (see
``DEFAULT_MAX_ADOPTION_DISAGREEMENT``), and asking for it when nothing was
sampled twice FAILS: a gate that passes vacuously is not a gate.

SPREAD-AWARE VERDICTS
---------------------
The plan's rule — *any gate whose margin is narrower than the measured spread
is reported indeterminate, not passed* — is enforced here rather than left to
the reviewer's prose. Every gate carries its margin, the measured spread of
that same quantity, and PASS / FAIL / INDETERMINATE. The spread comes from
the sample worlds above and/or from ``--spread-audit``, another recorded
audit of the same source record; the wider of the two is used. An
INDETERMINATE gate is not a pass and the process exits non-zero.

Count gates and rate gates are treated differently on purpose. An invented
citation observed in this run is a FAIL whatever the spread says — it
happened. A clean count is NOT a pass when a sibling sample was dirty:
'zero this time' is then a draw, not a property.

MISSING BY CONSTRUCTION IS NOT MISSING BY FAILURE
-------------------------------------------------
``--require-metrics`` gates on the claim rates being present and not
regressed, and it distinguishes a value that no run can ever produce from one
that went missing. See ``MISSING_BY_CONSTRUCTION``: guard turns and chat
turns are never verified in production, and an abstention with 0 claims has
no denominator. Counting those as failures is what makes
``eval_answers.py --compare --require-metrics`` exit 1 on every
production-mode record — 4 of 66 on this suite — and a gate that cannot pass
teaches an operator to ignore the exit code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Pinned BEFORE the package is imported. The replay must not depend on the
# operator's .env: repair is a FLAG here and is never read from the
# environment, and the index is never loaded (the record carries hit text).
os.environ["PRELOAD"] = "0"
os.environ.setdefault("INDEX_NAME", "default")


def _load_api_key() -> Optional[str]:
    """Take the API key from an env file — and NOTHING else from it.

    F11's rule, applied: an eval must not inherit production's switches from
    the file that rsyncs to the server, and the natural way to get a repair-ON
    run must not be 'edit .env'. Repair is a FLAG here (the arms are named on
    the command line); only the credentials are shared. ``GCF_QNA_ENV``
    overrides the path, ``data/eval/eval.env`` is preferred when present, and
    ``.env`` is the last resort — with every key but the two below ignored.
    """
    shared = ("OPENAI_API_KEY", "OPENAI_BASE_URL")
    if os.getenv("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    candidates = [Path(os.environ["GCF_QNA_ENV"])] if os.getenv("GCF_QNA_ENV") \
        else [ROOT / "data" / "eval" / "eval.env", ROOT / ".env"]
    for path in candidates:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() in shared and not os.getenv(key.strip()):
                os.environ[key.strip()] = value.strip().strip("'\"")
        if os.getenv("OPENAI_API_KEY"):
            return os.environ["OPENAI_API_KEY"]
    return None


_load_api_key()

from gcf_qna import config                                      # noqa: E402
from gcf_qna.rag import verify                                  # noqa: E402
from gcf_qna.rag.retrieve import Hit                            # noqa: E402

VERIFY_PY = ROOT / "src" / "gcf_qna" / "rag" / "verify.py"

#: The note blocks production hands the verifier, in production's order
#: (`eval_answers.run_case` passes [registry, year, matrix]; a `board` note is
#: recorded in notes_used but never reaches build_evidence).
NOTE_ORDER = ("registry", "year", "matrix")

#: eval_answers.TOKEN_COST_USD, restated rather than imported so the audit's
#: cost line does not change meaning when that file does. Both are estimates.
TOKEN_COST_USD = {"prompt": 1.25 / 1_000_000, "completion": 10.00 / 1_000_000}

ARMS = ("off", "on", "on-carryoff")

#: The reproducibility gate's default tolerance: ZERO adoption disagreements
#: over the sampled cases.
#:
#: WHY ZERO AND NOT A BUDGET. The gate's job is to prove that pinning holds,
#: and adoption is what decides whether the user reads the model's words or
#: the repair pass's. One flip is not a noisy estimate of a small rate — it is
#: an existence proof that the same input produced two different user-visible
#: answers, which is the property the pinning is supposed to remove. A budget
#: of one flip in twenty would also be unmeasurable at this sample size: 11
#: sampled cases cannot distinguish 5% from 0%, so a non-zero threshold would
#: be a number nobody could hold the run to. The pre-fix baseline (adoption
#: agreement 8/11 = 72.7% on the recorded run) is far from 100%, so the
#: threshold separates fixed from unfixed without needing to be delicate.
#:
#: IDENTICAL COMPLETION IS REPORTED, NOT GATED. Byte-identical text is a
#: stronger property than the adoption decision and it is the one a seed is
#: meant to deliver, but provider-side determinism is best-effort even at
#: temperature 0 with a fixed seed, so a gate on it would fail for reasons
#: outside this repository. It is reported next to the gated number, and a
#: run where adoption agrees while the text does not is a run where the
#: instrument is still telling the operator something.
DEFAULT_MAX_ADOPTION_DISAGREEMENT = 0.0

#: Reported always; gated only when named in --require-metrics. These are the
#: two the plan's Gate 4 line requires.
DEFAULT_COVERAGE_METRICS = ("groundedness_rate", "citation_completeness_rate")

#: 'this is not supported by the retrieved evidence', EN and FR — REPAIR_PROMPT
#: rule 3's second option. A successor sentence that says this is a
#: QUALIFICATION, which is a different outcome from a correction and from a
#: deletion, and the plan asks for the three to be distinguished.
_HEDGE_RE = re.compile(
    r"(not\s+(?:stated|found|available|supported|specified|reported|given)"
    r"|no[t]?\s+(?:in|among)\s+the\s+(?:retrieved|provided|available)"
    r"|does\s+not\s+(?:state|specify|report|appear)"
    r"|cannot\s+be\s+(?:determined|confirmed|verified)"
    r"|n[' ]?est\s+pas\s+(?:pr[ée]cis[ée]|indiqu[ée]|mentionn[ée]|fourni)"
    r"|non\s+(?:pr[ée]cis[ée]|indiqu[ée]|mentionn[ée]|disponible)"
    r"|ne\s+(?:figure|pr[ée]cise)\s+pas"
    r"|aucune?\s+(?:information|donn[ée]e)\s+)", re.I)

_DIGITS_RE = re.compile(r"\d[\d,.   ]*\d|\d")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> List[dict]:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Sequence[dict], force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        raise SystemExit(
            f"{path} exists. A recorded arm is the only copy of that "
            f"measurement — pass --force to overwrite deliberately.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")
    return path


def percentile(values, q: float):
    """Nearest-rank percentile — p95 is a latency that actually happened."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = max(0, min(len(xs) - 1, int(-(-len(xs) * q // 1)) - 1))
    return xs[k]


def claim_key(case_id: str, text: str) -> str:
    """Stable id for one claim of one case (same scheme as judge_audit)."""
    h = hashlib.sha1(f"{case_id}\x00{text}".encode("utf-8")).hexdigest()[:12]
    return f"{case_id}:{h}"


def evidence_for(record: dict):
    """The Evidence dict the recorded turn actually held."""
    hits = [Hit(text=h.get("text") or "", doc_id=h["doc"],
                score=float(h.get("score") or 0.0), page=h.get("page") or None)
            for h in (record.get("hits") or [])]
    notes = record.get("notes_used") or {}
    return verify.build_evidence(hits, [notes.get(k) for k in NOTE_ORDER])


def evidence_key_strings(evidence) -> List[str]:
    return [f"{d}|{p if p is not None else '-'}" for d, p in evidence]


def verifiable(record: dict) -> bool:
    """Did production verify this turn at all?

    Guard answers and chat turns return BEFORE verification
    (`chainlit_app` 1116-1133), so they have no verdicts, no repair and no
    claims. They are carried through every arm unchanged and cancel.
    """
    if record.get("error"):
        return False
    if record.get("guard") or record.get("chat"):
        return False
    return bool(record.get("raw_answer") or record.get("answer"))


# ---------------------------------------------------------------------------
# clients: metering, capture, canned replay
# ---------------------------------------------------------------------------
class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)


class _CannedResponse:
    """The minimum an OpenAI response needs to satisfy verify._complete."""

    def __init__(self, content, model="canned-replay"):
        self.choices = [_Choice(content)]
        self.usage = None
        self.model = model


class CannedClient:
    """Returns a fixed completion. Zero network. Counts its calls.

    Used for the carry-off scoring: the repaired TEXT must be the one the
    model actually produced on the carry-on pass, or the two scorings are
    scoring two different rewrites and the comparison means nothing.
    """

    def __init__(self, content: Optional[str]):
        self._content = content
        self.calls = 0
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        self.calls += 1
        if self._content is None:
            raise AssertionError(
                "canned client called with nothing recorded to replay — the "
                "carry-off pass took a branch the carry-on pass did not")
        return _CannedResponse(self._content)


class MeteredClient:
    """Wraps a real client; prices every call and captures the repair text."""

    def __init__(self, inner, model: str):
        self._inner = inner
        self._model = model
        self.calls: List[dict] = []
        self.repair_texts: List[str] = []
        self.chat = self

    @property
    def completions(self):
        return self

    def __getattr__(self, name):                       # pragma: no cover
        return getattr(self._inner, name)

    def create(self, **kwargs):
        role = _call_role(kwargs)
        t0 = time.perf_counter()
        resp = self._inner.chat.completions.create(**kwargs)
        dt = time.perf_counter() - t0
        u = getattr(resp, "usage", None)
        pt = int(getattr(u, "prompt_tokens", 0) or 0)
        ct = int(getattr(u, "completion_tokens", 0) or 0)
        self.calls.append({
            "role": role, "latency_s": round(dt, 3), "model": self._model,
            "snapshot": getattr(resp, "model", None) or self._model,
            "prompt_tokens": pt, "completion_tokens": ct,
            "total_tokens": int(getattr(u, "total_tokens", 0) or 0) or (pt + ct)})
        if role == "repair":
            try:
                self.repair_texts.append(resp.choices[0].message.content or "")
            except Exception:                          # pragma: no cover
                self.repair_texts.append("")
        return resp


def _call_role(kwargs: dict) -> str:
    """judge | repair | other, from the system prompt the call carries."""
    system = ""
    for m in kwargs.get("messages") or []:
        if m.get("role") == "system":
            system = m.get("content") or ""
            break
    if system == verify.ADJUDICATE_PROMPT:
        return "judge"
    if system == verify.REPAIR_PROMPT:
        return "repair"
    return "verify-other"


class no_carry_cleared:
    """Run ``verify.repair`` with the judge-carry disabled, in-process only.

    ``_carry_cleared`` is replaced by the identity for the duration and
    restored — and asserted restored — on exit. Nothing under ``src/`` is
    written; the plan gates on the number this produces precisely because a
    component may not certify itself with its own earlier opinion.
    """

    def __enter__(self):
        self._saved = verify._carry_cleared
        verify._carry_cleared = lambda new_verdicts, old_verdicts: list(new_verdicts)
        return self

    def __exit__(self, *exc):
        verify._carry_cleared = self._saved
        assert verify._carry_cleared is self._saved
        return False


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
def repair_allowed_docs(raw_answer: str, failed, evidence) -> List[str]:
    """The ``allowed`` set ``verify.repair`` builds, recomputed identically.

    It is local to ``repair()``, and the audit needs it: a citation moved onto
    a document the repair pass was never SHOWN passes ``_introduced_sources``'s
    first two tests and is still an invented attribution.
    """
    shown: List[str] = []
    for v in failed:
        strict, wide, _, _r5 = verify._scopes(v.claim, evidence)
        keys = verify._judge_keys(v.claim, evidence, strict, wide)
        shown += [k[0] for k in keys if k[0] != verify.NOTES_DOC]
    return list(dict.fromkeys(
        shown + [d for d, _ in verify.cited_sources(raw_answer)]))


def claim_rows(verdicts, evidence, case_id: str) -> List[dict]:
    """Every claim of one arm, with everything the audit compares on.

    Recorded per claim rather than summarised, because 'which claim was lost'
    is not derivable from a count and the plan requires each one listed.
    """
    claims = [v.claim for v in verdicts]
    grounded = _grounded_flags(claims, evidence)
    out = []
    for v, g in zip(verdicts, grounded):
        out.append({
            "key": claim_key(case_id, v.claim.text),
            "norm": verify.norm_text(v.claim.text),
            "index": v.claim.index,
            "text": v.claim.text,
            "kind": v.claim.kind,
            "required": bool(v.claim.required),
            "cited": bool(v.claim.citations),
            "citations": [[c.doc, c.page] for c in v.claim.citations],
            "status": v.status,
            "source": getattr(v, "source", "deterministic"),
            "reason": (v.reason or "")[:200],
            "flags": list(v.flags or []),
            "grounded": bool(g),
        })
    return out


def _grounded_flags(claims, evidence) -> List[bool]:
    """Does ANY evidence the turn held entail the claim?

    The Wave-0b groundedness definition — verify's own matcher, unmodified,
    over the union of the turn's evidence rather than the cited scope. Kept
    here rather than imported so the audit stays importable without the eval
    harness; ``tests/test_audit_repair.py`` pins it against
    ``eval_answers.grounded_flags``.
    """
    blob = verify._text_of(evidence, list(evidence))
    out = []
    for c in claims:
        try:
            ok, _missing = verify._verify_against(c, blob)
        except Exception:                              # noqa: BLE001
            ok = False
        out.append(bool(ok))
    return out


def _claims_block(verdicts, evidence) -> dict:
    """The release record's ``claims`` shape, so ``--compare`` reads an arm."""
    claims = [v.claim for v in verdicts]
    grounded = _grounded_flags(claims, evidence)
    n = Counter(v.status for v in verdicts)
    cited = citation_supported = n_grounded = 0
    failures = []
    for v, g in zip(verdicts, grounded):
        has_cite = bool(v.claim.citations)
        cited += bool(has_cite)
        n_grounded += bool(g)
        if v.status == verify.SUPPORTED and has_cite:
            citation_supported += 1
        if v.status != verify.SUPPORTED:
            failures.append({"status": v.status, "kind": v.claim.kind,
                             "text": v.claim.text[:160],
                             "reason": (v.reason or "")[:160],
                             "cited": has_cite, "grounded": bool(g),
                             "source": getattr(v, "source", "deterministic")})
    total = len(verdicts)
    supported = n[verify.SUPPORTED]
    return {
        "claims": total,
        "supported": supported,
        "contradicted": n[verify.CONTRADICTED],
        "unsupported": n[verify.UNSUPPORTED],
        "support_rate": (supported / total) if total else None,
        "grounded": n_grounded,
        "groundedness_rate": (n_grounded / total) if total else None,
        "citation_supported": citation_supported,
        "citation_completeness_rate": (citation_supported / total) if total else None,
        "cited": cited,
        "citation_presence_rate": (cited / total) if total else None,
        "judge_promotions": sum(1 for v in verdicts
                                if getattr(v, "source", "") == "llm"
                                and v.status == verify.SUPPORTED),
        "evidence_keys": evidence_key_strings(evidence),
        "verifier_mode": "production",
        "n_failures": len(failures),
        "failures": failures,
    }


def final_answer(original: str, res) -> Tuple[str, str]:
    """(body the app would display, where it came from) — mirrors
    ``eval_answers.final_answer``, which mirrors ``_verify_reply``."""
    if res is None:
        return original, "model"
    if getattr(res, "status", None) == "abstain":
        return original, "abstain-original"
    text = getattr(res, "answer", None) or original
    return text, ("verifier" if text != original else "model")


def _rescore_answer(record: dict, answer: str) -> Tuple[Optional[dict], Optional[dict]]:
    """(checks, fields) for a rewritten answer, through the release harness's
    OWN scorers — a reimplementation here would be a second instrument."""
    try:
        import eval_answers as ev
    except Exception:                                  # pragma: no cover
        return None, None
    case = {"id": record.get("id"), "class": record.get("class"),
            "lang": record.get("lang"), "question": record.get("question"),
            "expect": record.get("expect") or {}}
    hits = [Hit(text=h.get("text") or "", doc_id=h["doc"],
                score=float(h.get("score") or 0.0), page=h.get("page") or None)
            for h in (record.get("hits") or [])]
    try:
        checks = ev.score_answer(case, answer, hits)
    except Exception as e:                             # noqa: BLE001
        checks = {"error": f"{type(e).__name__}: {e}"}
    try:
        fields = ev.score_fields(case, answer)
    except Exception:                                  # noqa: BLE001
        fields = None
    return checks, fields


def _arm_row(record: dict, arm: str, res, evidence, meta: dict) -> dict:
    """One arm's row for one case, in the release-record shape."""
    raw = record.get("raw_answer") or record.get("answer") or ""
    answer, source = final_answer(raw, res)
    row = dict(record)
    checks, fields = _rescore_answer(record, answer)
    row.update({
        "repair_arm": arm,
        "answer": answer,
        "raw_answer": raw,
        "answer_source": source,
        "checks": checks,
        "fields": fields,
        "score": (checks or {}).get("score", record.get("score")),
        "claims": _claims_block(res.verdicts, evidence),
        "claims_skipped": None,
        "claim_rows": claim_rows(res.verdicts, evidence, record.get("id") or "?"),
        "evidence_keys": evidence_key_strings(evidence),
        "verify_status": res.status,
        "repaired": bool(res.repaired),
        "repair_rejected": bool(res.repair_rejected),
        "repair_notes": list(res.notes or []),
        "usage_recorded": record.get("usage"),
        "usage": meta.get("usage"),
        "repair_replay": meta.get("replay"),
    })
    return row


def _skipped_row(record: dict, arm: str, reason: str) -> dict:
    row = dict(record)
    row.update({"repair_arm": arm, "claim_rows": [],
                "repair_replay": {"replayed": False, "reason": reason,
                                  "api_calls": 0},
                "repaired": False, "repair_rejected": False,
                "repair_notes": []})
    return row


#: The claim-level numbers a sample carries at a glance. The full block is
#: stored too when the sample's text differs, but a reader — and a legacy
#: record's reconstruction — needs these five without parsing claim rows.
SAMPLE_METRIC_KEYS = ("claims", "supported", "grounded", "citation_supported",
                      "cited", "contradicted", "unsupported")


def _sample_entry(index: int, text: Optional[str], first_text: Optional[str],
                  res, record: dict, evidence, calls: Sequence[dict],
                  case_id: str) -> dict:
    """One repair sample, scored exactly the way an arm row is scored.

    WHAT IS STORED AND WHY. ``metrics`` is always present: it is the sample's
    contribution to every gated claim metric, and Wave 4's record carried
    none of it — which is why that wave could report that adoption flipped
    but not what the flip did to groundedness.

    ``text``, ``claims``, ``checks`` and ``claim_rows`` are stored ONLY for a
    sample whose completion differs from sample 1's. Repair is deterministic
    downstream of the completion (``_introduced_sources``, a fresh
    ``classify_deterministic``, the substance floor), so an identical
    completion has an identical outcome and re-storing it would triple a
    recorded arm to say nothing. A DIFFERENT completion is the whole finding,
    and the audit needs its text to re-derive invented sources per sample
    rather than trusting the numbers it was handed.
    """
    raw = record.get("raw_answer") or record.get("answer") or ""
    answer, source = final_answer(raw, res)
    checks, _fields = _rescore_answer(record, answer)
    block = _claims_block(res.verdicts, evidence)
    identical = (text or "") == (first_text or "")
    entry = {
        "sample": index,
        "identical_text": bool(identical),
        "text_sha256": _sha256_text(text or "") if text is not None else None,
        "adopted": bool(res.repaired),
        "rejected": bool(res.repair_rejected),
        "status": res.status,
        "notes": list(res.notes or []),
        "answer_sha256": _sha256_text(answer),
        "answer_source": source,
        "carry_cleared": True,
        "metrics": dict({k: block.get(k) for k in SAMPLE_METRIC_KEYS},
                        checks_pass=(bool(checks.get("pass"))
                                     if isinstance(checks, dict)
                                     and "pass" in checks else None)),
        "calls": list(calls),
    }
    if index > 1 and not identical:
        entry.update({"text": text, "claims": block, "checks": checks,
                      "claim_rows": claim_rows(res.verdicts, evidence, case_id)})
    return entry


def replay(release: Path, out_prefix: Path, force: bool = False,
           client: Any = None, repair_samples: int = 1,
           limit: Optional[int] = None, verbose: bool = False) -> dict:
    """Replay one recorded run through the verifier twice, three scorings.

    ``repair_samples`` is the reproducibility instrument: N repair calls per
    failing case on IDENTICAL input (same answer, same verdict OBJECTS, same
    evidence), each one fully scored and recorded. N=1 is the plain A/B; N>1
    is what lets ``audit`` report the identical-completion rate, the
    adoption-decision agreement rate and the spread those flips put on every
    gated metric. It costs one extra repair call per failing case per extra
    sample and nothing else — the judge pass is not re-run.

    Returns the run summary; writes ``<prefix>_repair-off.jsonl``,
    ``<prefix>_repair-on.jsonl``, ``<prefix>_repair-on-carryoff.jsonl`` and
    ``<prefix>_repair-calls.jsonl``.
    """
    rows = read_jsonl(release)
    if limit:
        rows = rows[:limit]
    blob = _sha256_file(VERIFY_PY)
    src_sha = _sha256_file(Path(release))
    if client is None:
        client = verify._client()
    if client is None:
        raise SystemExit(
            "no OPENAI_API_KEY: the production arm calls the judge and the "
            "repair pass. Pass a client in-process for tests.")
    model = getattr(config, "CHAT_MODEL", "?")

    out = {a: [] for a in ARMS}
    call_rows: List[dict] = []
    summary = defaultdict(int)
    for i, rec in enumerate(rows, 1):
        cid = rec.get("id")
        if not verifiable(rec):
            why = ("errored" if rec.get("error") else
                   "chat-mode turn: production answers from history"
                   if rec.get("chat") else
                   "guard-answer: production returns before verification")
            for a in ARMS:
                out[a].append(_skipped_row(rec, a, why))
            summary["skipped"] += 1
            continue

        raw = rec.get("raw_answer") or rec.get("answer") or ""
        ev = evidence_for(rec)
        metered = MeteredClient(client, model)

        # --- the shared pass: extract, deterministic, ONE judge call --------
        claims = verify.extract_claims(raw)
        det = verify.classify_deterministic(claims, ev)
        verdicts = verify.adjudicate(det, ev, client=metered)
        judge_calls = list(metered.calls)
        failed = [v for v in verdicts if v.failed]

        # --- arm OFF: exactly verify_answer(..., allow_repair=False) --------
        if not failed:
            off_res = verify.RepairResult(raw, "verified", verdicts, raw)
        else:
            off_res = verify.RepairResult(
                raw, verify._status_for(verdicts, True, False), verdicts, raw,
                notes=["repair disabled"])

        # --- arm ON: exactly verify_answer(..., allow_repair=True) ----------
        allowed = repair_allowed_docs(raw, failed, ev) if failed else []
        on_res = verify.repair(raw, verdicts, ev, client=metered)
        repair_calls = [c for c in metered.calls if c["role"] == "repair"]
        repair_text = metered.repair_texts[0] if metered.repair_texts else None

        # --- arm ON, carry-off: same TEXT, no judge carry -------------------
        canned = CannedClient(repair_text)
        with no_carry_cleared():
            co_res = verify.repair(raw, verdicts, ev, client=canned)
        assert canned.calls == len(repair_calls), (
            f"{cid}: carry-off pass made {canned.calls} repair call(s), "
            f"carry-on made {len(repair_calls)} — the two scorings did not "
            f"score the same rewrite")

        # --- reproducibility: N samples of repair on IDENTICAL input -------
        # Sample 1 IS the ON arm's call: the same answer, the same verdict
        # OBJECTS, the same evidence, the same prompt. Samples 2..N ask the
        # same question again, so the only thing that varies across them is
        # the model's own sampling. Each sample is scored the way an arm row
        # is scored, because 'the adoption decision flipped' and 'the gated
        # metric moved' are two different questions and a record that carries
        # only the first cannot answer the second.
        samples = []
        if failed:
            samples.append(_sample_entry(1, repair_text, repair_text, on_res,
                                         rec, ev, metered.calls, cid))
            for k in range(2, max(1, int(repair_samples or 1)) + 1):
                mk = MeteredClient(client, model)
                rk = verify.repair(raw, verdicts, ev, client=mk)
                tk = mk.repair_texts[0] if mk.repair_texts else None
                samples.append(_sample_entry(k, tk, repair_text, rk, rec, ev,
                                             mk.calls, cid))
                call_rows += [dict(c, case=cid, sample=k) for c in mk.calls]
        second = samples[1] if len(samples) > 1 else None

        call_rows += [dict(c, case=cid, sample=1) for c in metered.calls]
        pre_gate = _pre_gate(raw, repair_text, verdicts, failed, ev, allowed, cid)
        base_replay = {
            "replayed": True, "source_record": str(release),
            "source_sha256": src_sha, "verify_blob_sha": blob,
            "record_verify_blob_sha": rec.get("verify_blob_sha"),
            "judge_calls": len(judge_calls), "repair_calls": len(repair_calls),
            "repair_text_sha256": _sha256_text(repair_text or "")
                                  if repair_text is not None else None,
            "repair_allowed_docs": allowed,
            "repair_samples": samples or None,
            # Every sample is drawn with `_carry_cleared` LIVE, because sample
            # 1 is the ON arm's own call. The audit compares like with like
            # only if it takes its sample-1 baseline from the carry-ON arm —
            # recorded here so it cannot be guessed wrong.
            "repair_samples_carry": "on" if samples else None,
            "repair_second_sample": second,
            "repair_pre_gate": pre_gate,
        }
        metas = {
            "off": {"usage": _usage(judge_calls),
                    "replay": dict(base_replay, arm="off", carry_cleared=None,
                                   api_calls=len(judge_calls))},
            "on": {"usage": _usage(judge_calls + repair_calls),
                   "replay": dict(base_replay, arm="on", carry_cleared=True,
                                  api_calls=len(judge_calls) + len(repair_calls))},
            "on-carryoff": {"usage": _usage(judge_calls + repair_calls),
                            "replay": dict(base_replay, arm="on-carryoff",
                                           carry_cleared=False,
                                           api_calls=len(judge_calls) + len(repair_calls))},
        }
        for arm, res in (("off", off_res), ("on", on_res), ("on-carryoff", co_res)):
            out[arm].append(_arm_row(rec, arm, res, ev, metas[arm]))

        summary["cases"] += 1
        summary["with_failures"] += bool(failed)
        summary["repair_attempted"] += bool(repair_calls)
        summary["adopted_on"] += bool(on_res.repaired)
        summary["adopted_carryoff"] += bool(co_res.repaired)
        summary["rejected_on"] += bool(on_res.repair_rejected)
        summary["rejected_carryoff"] += bool(co_res.repair_rejected)
        if bool(on_res.repaired) != bool(co_res.repaired):
            summary["carry_disagreements"] += 1
        if len(samples) > 1:
            summary["cases_sampled"] += 1
            summary["identical_completions"] += all(
                s["identical_text"] for s in samples[1:])
            summary["adoption_agreements"] += len(
                {s["adopted"] for s in samples}) == 1
        if verbose:
            print(f"[{i}/{len(rows)}] {cid:26} failed={len(failed):2d} "
                  f"repair={'adopted' if on_res.repaired else ('rejected' if on_res.repair_rejected else 'n/a')}"
                  f" carryoff={'adopted' if co_res.repaired else ('rejected' if co_res.repair_rejected else 'n/a')}",
                  flush=True)

    paths = {}
    for arm in ARMS:
        paths[arm] = str(write_jsonl(Path(f"{out_prefix}_repair-{arm}.jsonl"),
                                     out[arm], force=force))
    paths["calls"] = str(write_jsonl(Path(f"{out_prefix}_repair-calls.jsonl"),
                                     call_rows, force=force))
    return {"summary": dict(summary), "paths": paths,
            "verify_blob_sha": blob, "source_sha256": src_sha,
            "cost": _cost(call_rows)}


def _pre_gate(raw_answer: str, repair_text: Optional[str], verdicts,
              failed, evidence, allowed: Sequence[str], case_id: str):
    """What the repair model PROPOSED, before any adoption gate ran.

    The plan says to probe the invented-source gate adversarially rather than
    trusting its own post-check. The only way to do that offline is to keep
    the model's proposal even when it was thrown away: a rewrite rejected for
    'one claim still fails' may ALSO have invented a citation, and counting
    only the adopted rewrites would report zero inventions from a pass that
    proposed several.
    """
    if not repair_text:
        return None
    text = verify._strip_preamble(repair_text)
    introduced = verify._introduced_sources(text, evidence, allowed)
    new_verdicts = verify.classify_deterministic(verify.extract_claims(text),
                                                 evidence)
    new_failed = [v for v in new_verdicts if v.failed]
    would_adopt = (not introduced and not new_failed
                   and not (verify._supported_required(verdicts)
                            and not verify._supported_required(new_verdicts)))
    return {
        "text": text,
        "introduced_sources": introduced,
        "failures_before": len(failed),
        "failures_after": len(new_failed),
        "failure_reasons_after": [v.reason[:120] for v in new_failed[:6]],
        "supported_required_before": verify._supported_required(verdicts),
        "supported_required_after": verify._supported_required(new_verdicts),
        "would_adopt_without_carry": bool(would_adopt),
        "claim_rows": claim_rows(new_verdicts, evidence, case_id),
    }


def _usage(calls: Sequence[dict]) -> dict:
    calls = [c for c in calls if c]
    if not calls:
        return {}
    return {"calls": list(calls),
            "turn_latency_s": round(sum(c.get("latency_s") or 0.0 for c in calls), 3),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "total_tokens": sum(c["total_tokens"] for c in calls),
            "roles": sorted({c["role"] for c in calls})}


def _cost(calls: Sequence[dict]) -> dict:
    out = {}
    for role in sorted({c["role"] for c in calls} | {"judge", "repair"}):
        rows = [c for c in calls if c["role"] == role]
        pt = sum(c["prompt_tokens"] for c in rows)
        ct = sum(c["completion_tokens"] for c in rows)
        lat = [c.get("latency_s") for c in rows]
        out[role] = {
            "calls": len(rows), "prompt_tokens": pt, "completion_tokens": ct,
            "latency_total_s": round(sum(x or 0.0 for x in lat), 2),
            "latency_p50_s": percentile(lat, 0.50),
            "latency_p95_s": percentile(lat, 0.95),
            "cost_usd": round(pt * TOKEN_COST_USD["prompt"]
                              + ct * TOKEN_COST_USD["completion"], 4)}
    return out


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------
def _tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", verify.norm_text(text or "")))


def _similar(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _digit_strings(text: str) -> List[str]:
    """Every digit run of a text, separators stripped. Deliberately dumber
    than ``verify.amounts``: a figure the matcher does not model as an amount
    (a page number in prose, a count, a percentage) is still a figure a repair
    may not invent."""
    out = []
    for m in _DIGITS_RE.finditer(text or ""):
        d = re.sub(r"[^\d]", "", m.group(0))
        if d:
            out.append(d)
    return out


def _evidence_blob(row: dict) -> str:
    return "\n".join((h.get("text") or "") for h in (row.get("hits") or [])) \
        + "\n" + "\n".join(str(v) for v in (row.get("notes_used") or {}).values() if v)


def _held_keys(row: dict) -> Tuple[set, set]:
    """(docs, (doc,page) keys) the turn's evidence holds."""
    docs, keys = set(), set()
    for k in row.get("evidence_keys") or []:
        doc, _, page = k.rpartition("|")
        if not doc:
            continue
        docs.add(doc)
        keys.add((doc, None if page == "-" else int(page)))
    if not docs:                                        # pre-replay records
        for h in row.get("hits") or []:
            docs.add(h["doc"])
            keys.add((h["doc"], h.get("page") or None))
    return docs, keys


def invented_sources(off_row: dict, on_row: dict) -> dict:
    """Documents, pages and figures the repaired answer names and the evidence
    does not — derived from the recorded text, not from verify's own gate.

    This is the gate prior rounds failed, so it is probed three ways and each
    way is reported separately rather than folded into one boolean.
    """
    on_answer = on_row.get("answer") or ""
    off_answer = off_row.get("answer") or ""
    docs, keys = _held_keys(on_row)
    allowed = set((on_row.get("repair_replay") or {}).get("repair_allowed_docs")
                  or []) or docs

    inv_docs, inv_pages, not_shown = [], [], []
    for c in verify.parse_citations(on_answer):
        if c.doc is None:
            continue
        if c.doc not in docs:
            inv_docs.append({"doc": c.doc, "page": c.page})
        elif c.doc not in allowed:
            not_shown.append({"doc": c.doc, "page": c.page})
        elif c.page is not None and (c.doc, c.page) not in keys:
            inv_pages.append({"doc": c.doc, "page": c.page})

    # a page the repaired answer points at and the raw one did not. Every one
    # is a HELD key (the two lists above would have caught it otherwise), so
    # this is reported, not gated — but re-pointing a claim at a different
    # page of the same document is a substantive rewrite and must be visible.
    before_keys = {(c.doc, c.page) for c in verify.parse_citations(off_answer)}
    new_pages = [{"doc": c.doc, "page": c.page}
                 for c in verify.parse_citations(on_answer)
                 if c.doc is not None and (c.doc, c.page) not in before_keys]

    # figures: present after, absent before, and absent from the evidence.
    # Citations are STRIPPED first: a page number inside a bracket is a
    # citation, and it is already judged by invented_pages above — counting it
    # here as an invented figure would fire on every legitimate re-citation
    # and drown the shape this check exists for (a figure invented in prose).
    blob = _evidence_blob(on_row)
    blob_digits = set(_digit_strings(blob))
    before_digits = set(_digit_strings(off_answer))
    inv_figures = []
    for d in dict.fromkeys(_digit_strings(verify._strip_citations(on_answer))):
        if d in before_digits or d in blob_digits:
            continue
        # second, independent opinion: verify's own amount matcher over the
        # whole held evidence. A literal-digit miss that the matcher CAN place
        # (scale words, decimal-comma conventions) is reported, not counted.
        matched = _amount_in_evidence(d, on_answer, blob)
        inv_figures.append({"digits": d, "matcher_places_it": matched})

    # entities: a name new after repair that the matcher cannot find anywhere.
    inv_entities = []
    for row in on_row.get("claim_rows") or []:
        if row["norm"] in {r["norm"] for r in off_row.get("claim_rows") or []}:
            continue
        try:
            claim = _claim_from_text(row["text"])
            ok, missing = verify._check_entities(claim, blob)
        except Exception:                              # noqa: BLE001
            continue
        if not ok:
            inv_entities += [{"name": vs[0], "claim": row["text"][:120]}
                             for vs in missing]

    return {"invented_docs": inv_docs, "invented_pages": inv_pages,
            "sources_not_shown_to_repair": not_shown,
            "invented_figures": inv_figures, "invented_entities": inv_entities,
            "newly_cited_pages": new_pages}


def _claim_from_text(text: str):
    got = verify.extract_claims(text)
    if got:
        return got[0]
    raise ValueError("not a claim")


def _amount_in_evidence(digits: str, answer: str, blob: str) -> bool:
    """Can verify's amount matcher place this digit run in the evidence?"""
    try:
        want = [a for a in verify.amounts(answer)
                if re.sub(r"[^\d]", "", a.raw) == digits]
        have = verify.amounts(blob)
    except Exception:                                  # noqa: BLE001
        return False
    return any(verify.amount_matches(w, h) for w in want for h in have)


def novel_sentences(off_row: dict, on_row: dict) -> List[dict]:
    """Sentences after repair that were not there before, with groundedness."""
    before = {verify.norm_text(s) for s in
              verify.split_sentences(off_row.get("answer") or "")}
    out = []
    on_claims = {r["norm"]: r for r in on_row.get("claim_rows") or []}
    for s in verify.split_sentences(on_row.get("answer") or ""):
        n = verify.norm_text(s)
        if n in before or not s.strip():
            continue
        row = on_claims.get(n)
        out.append({"text": s[:300], "is_claim": row is not None,
                    "status": (row or {}).get("status"),
                    "grounded": (row or {}).get("grounded"),
                    "hedge": bool(_HEDGE_RE.search(s))})
    return out


def claim_diff(off_row: dict, on_row: dict) -> dict:
    """Per-case claim movement, with each lost claim's fate named.

    ``corrected`` and ``deleted`` are NOT the same outcome and the plan says
    so: a failing claim whose successor sentence is SUPPORTED was corrected;
    one that simply vanished was deleted, and a deletion of a claim the held
    evidence GROUNDS is a correction the repair declined to make.
    """
    before = {r["norm"]: r for r in off_row.get("claim_rows") or []}
    after = {r["norm"]: r for r in on_row.get("claim_rows") or []}
    on_answer = on_row.get("answer") or ""
    on_sentences = verify.split_sentences(on_answer)

    lost, added, fates = [], [], []
    survivors = [r for norm, r in after.items() if norm not in before]
    for norm, r in before.items():
        if norm in after:
            continue
        # The successor is looked for among the CLAIMS of the repaired answer
        # first, not among its sentences: a rewrite that re-splits a sentence
        # would otherwise read as a deletion plus an unrelated addition, and
        # 'corrected' would be unreachable for exactly the rewrites that
        # correct something.
        cands = [(c["text"], _similar(r["text"], c["text"]), c) for c in survivors]
        cands += [(s_, _similar(r["text"], s_), None) for s_ in on_sentences
                  if verify.norm_text(s_) not in before
                  and verify.norm_text(s_) not in after]
        best = max(cands, key=lambda c: c[1], default=(None, 0.0, None))
        if best[1] < 0.34:
            succ_text, sim, succ = None, best[1], None
        else:
            succ_text, sim, succ = best
        if succ_text is not None and _HEDGE_RE.search(succ_text):
            fate = "qualified"
        elif succ is not None:
            fate = ("corrected" if succ["status"] == verify.SUPPORTED
                    else "rewritten-still-failing")
        elif succ_text is not None:
            fate = "rewritten-into-a-non-claim"
        else:
            fate = "deleted"
        entry = {
            "case": off_row.get("id"), "key": r["key"], "text": r["text"],
            "kind": r["kind"], "required": r["required"],
            "status_before": r["status"], "grounded_before": r["grounded"],
            "source_before": r["source"], "reason_before": r["reason"],
            "fate": fate,
            "successor_text": succ_text,
            "successor_status": (succ or {}).get("status"),
            "successor_grounded": (succ or {}).get("grounded"),
            "successor_similarity": round(sim, 3),
            "deleted_though_grounded": bool(fate == "deleted" and r["grounded"]),
        }
        lost.append(entry)
        fates.append(fate)
    for norm, r in after.items():
        if norm in before:
            continue
        added.append({"case": on_row.get("id"), "key": r["key"],
                      "text": r["text"], "kind": r["kind"],
                      "status_after": r["status"], "grounded_after": r["grounded"],
                      "cited": r["cited"]})
    moved = []
    for norm, r in before.items():
        s = after.get(norm)
        if s is not None and s["status"] != r["status"]:
            moved.append({"case": off_row.get("id"), "text": r["text"][:160],
                          "status_before": r["status"],
                          "status_after": s["status"],
                          "grounded_before": r["grounded"],
                          "grounded_after": s["grounded"]})
    return {"lost": lost, "added": added, "status_moved": moved,
            "fates": Counter(fates)}


def _checks_regressions(off_row: dict, on_row: dict) -> List[dict]:
    a, b = off_row.get("checks") or {}, on_row.get("checks") or {}
    out = []
    for name in ("behavior", "language", "citations"):
        if a.get(name) and not b.get(name):
            out.append({"case": off_row.get("id"), "check": name,
                        "before": True, "after": False})
    for name in ("must_contain", "must_not_contain"):
        for pat, ok in (a.get(name) or {}).items():
            if ok and not (b.get(name) or {}).get(pat, True):
                out.append({"case": off_row.get("id"), "check": f"{name}:{pat}",
                            "before": True, "after": False})
    if a.get("pass") and not b.get("pass"):
        out.append({"case": off_row.get("id"), "check": "pass",
                    "before": True, "after": False})
    return out


def _nd(num: int, den: int) -> dict:
    return {"n": num, "d": den,
            "rate": (num / den) if den else None,
            "threshold_95": -(-95 * den // 100) if den else None}


def audit(off_path: Path, on_path: Path, carry_on_path: Path = None,
          justifications: Path = None,
          spread_audits: Sequence[Path] = (),
          require_metrics: Sequence[str] = (),
          max_adoption_disagreement: Optional[float] = None,
          require_spread: bool = False) -> dict:
    """Compare two replayed arms, and say how much of the answer is sampling.

    The comparison itself is ``_compare_arms``. Everything this wrapper adds
    is about the RELIABILITY of that comparison: how far the same measurement
    moves between samples of repair on identical input, and which gate
    margins are inside that movement.
    """
    off = {r["id"]: r for r in read_jsonl(off_path)}
    on = {r["id"]: r for r in read_jsonl(on_path)}
    carry = {r["id"]: r for r in read_jsonl(carry_on_path)} if carry_on_path else {}
    justified = {}
    if justifications:
        for r in read_jsonl(justifications):
            justified[r.get("key") or r.get("claim_key")] = r.get("justification")

    report = _compare_arms(off, on, justified, carry=carry,
                           off_label=str(off_path), on_label=str(on_path))
    shared = [i for i in off if i in on]

    # Provenance. A recorded audit is a measurement of ONE verify.py over ONE
    # pair of arms; re-running it after either changes is a different number,
    # and a saved report that cannot say which is a number without a subject.
    report["inputs"] = {
        "off_sha256": _sha256_file(Path(off_path)),
        "on_sha256": _sha256_file(Path(on_path)),
        "carry_on_sha256": _sha256_file(Path(carry_on_path)) if carry_on_path else None,
        "audit_verify_blob_sha": _sha256_file(VERIFY_PY) if VERIFY_PY.exists() else None,
    }

    # --- reproducibility, and the spread it implies -----------------------
    base, base_arm, mismatch = _sample_base(on, carry)
    report["reproducibility"] = repro = _reproducibility(off, base, shared,
                                                         base_arm, mismatch)
    report["sample_worlds"] = worlds = _sample_worlds(off, base, shared, justified)
    replicates = []
    for p in spread_audits or ():
        other = json.loads(Path(p).read_text(encoding="utf-8"))
        replicates.append({"source": str(p), "quantities": _gate_quantities(other),
                           "cases_compared": other.get("cases_compared")})
    report["spread_replicates"] = [{"source": r["source"],
                                    "cases_compared": r["cases_compared"]}
                                   for r in replicates]
    report["spreads"] = spreads = _spreads(
        _gate_quantities(report),
        [w["quantities"] for w in worlds.get("worlds") or []],
        [r["quantities"] for r in replicates])

    # --- metric coverage: missing-by-construction vs missing-by-failure ---
    report["metric_coverage"] = _metric_coverage(
        off, on, shared, list(require_metrics) or list(DEFAULT_COVERAGE_METRICS))
    report["require_metrics"] = list(require_metrics or ())

    # --- gates, spread-aware ---------------------------------------------
    report["gates"] = gates = _gate_verdicts(
        report, spreads, require_metrics=require_metrics,
        max_adoption_disagreement=max_adoption_disagreement,
        require_spread=require_spread, repro=repro)
    report["breaches"] = [g["breach"] for g in gates
                          if g["verdict"] == "FAIL" and g["breach"]]
    report["indeterminate"] = [
        f"{g['name']}: {g['why']}" for g in gates if g["verdict"] == "INDETERMINATE"]
    report["gate_verdict"] = ("FAIL" if report["breaches"] else
                              "INDETERMINATE" if report["indeterminate"] else "PASS")
    report["gate_pass"] = report["gate_verdict"] == "PASS"
    return report


def _compare_arms(off: dict, on: dict, justified: dict, carry: dict = None,
                  extras: bool = True, off_label: str = "off-arm",
                  on_label: str = "on-arm") -> dict:
    """The file comparison itself — no spreads, no gates but its own.

    Split out of ``audit`` so a SAMPLE WORLD (the run this would have been had
    repair sampled differently) is scored by exactly this code and not by a
    second, simplified summariser.
    """
    carry = carry or {}
    shared = [i for i in off if i in on]
    only = sorted(set(off) ^ set(on))

    agg = defaultdict(int)
    lost_all, added_all, moved_all, fates = [], [], [], Counter()
    inv = {"invented_docs": [], "invented_pages": [], "invented_figures": [],
           "invented_entities": [], "sources_not_shown_to_repair": [],
           "newly_cited_pages": []}
    novel, regressions, per_case = [], [], []
    lang_rows, behaviour = [], []
    rewritten = 0
    for cid in shared:
        a, b = off[cid], on[cid]
        ca = a.get("claims") if isinstance(a.get("claims"), dict) else None
        cb = b.get("claims") if isinstance(b.get("claims"), dict) else None
        if ca:
            for k in ("claims", "supported", "grounded", "citation_supported",
                      "cited", "contradicted", "unsupported"):
                agg[f"{k}_before"] += int(ca.get(k) or 0)
        if cb:
            for k in ("claims", "supported", "grounded", "citation_supported",
                      "cited", "contradicted", "unsupported"):
                agg[f"{k}_after"] += int(cb.get(k) or 0)
        changed = (a.get("answer") or "") != (b.get("answer") or "")
        rewritten += bool(changed)
        if not changed:
            continue
        d = claim_diff(a, b)
        lost_all += d["lost"]
        added_all += d["added"]
        moved_all += d["status_moved"]
        fates.update(d["fates"])
        found = invented_sources(a, b)
        for k in inv:
            inv[k] += [dict(x, case=cid) for x in found[k]]
        novel += [dict(x, case=cid) for x in novel_sentences(a, b)]
        regressions += _checks_regressions(a, b)
        behaviour.append({"case": cid, "expected": (a.get("expect") or {}).get("behavior"),
                          "before": (a.get("checks") or {}).get("behavior"),
                          "after": (b.get("checks") or {}).get("behavior")})
        lang_rows.append({"case": cid, "lang": a.get("lang"),
                          "language_before": (a.get("checks") or {}).get("language"),
                          "language_after": (b.get("checks") or {}).get("language")})
        per_case.append({
            "case": cid, "class": a.get("class"), "lang": a.get("lang"),
            "status_before": a.get("verify_status"),
            "status_after": b.get("verify_status"),
            "claims": [(ca or {}).get("claims"), (cb or {}).get("claims")],
            "supported": [(ca or {}).get("supported"), (cb or {}).get("supported")],
            "grounded": [(ca or {}).get("grounded"), (cb or {}).get("grounded")],
            "citation_supported": [(ca or {}).get("citation_supported"),
                                   (cb or {}).get("citation_supported")],
            "score": [a.get("score"), b.get("score")],
            "chars": [len(a.get("answer") or ""), len(b.get("answer") or "")],
            "lost": len(d["lost"]), "added": len(d["added"]),
            "fates": dict(d["fates"]),
            "repair_notes": b.get("repair_notes"),
        })

    lost_supported = [r for r in lost_all if r["status_before"] == verify.SUPPORTED]
    undocumented = [r for r in lost_supported if not justified.get(r["key"])]
    figures_hard = [r for r in inv["invented_figures"] if not r["matcher_places_it"]]

    # answer-level
    pass_before = sum(1 for i in shared if (off[i].get("checks") or {}).get("pass"))
    pass_after = sum(1 for i in shared if (on[i].get("checks") or {}).get("pass"))
    p2f = [i for i in shared if (off[i].get("checks") or {}).get("pass")
           and not (on[i].get("checks") or {}).get("pass")]
    f2p = [i for i in shared if not (off[i].get("checks") or {}).get("pass")
           and (on[i].get("checks") or {}).get("pass")]
    lang_bad = [r for r in lang_rows if r["language_before"] and not r["language_after"]]
    fields_before = _field_cells(off, shared)
    fields_after = _field_cells(on, shared)

    report = {
        "off": off_label, "on": on_label,
        "cases_compared": len(shared),
        "cases_only_on_one_side": only,
        "answers_rewritten": rewritten,
        "claims_before": agg["claims_before"], "claims_after": agg["claims_after"],
        "supported_before": agg["supported_before"],
        "supported_after": agg["supported_after"],
        "groundedness_before": _nd(agg["grounded_before"], agg["claims_before"]),
        "groundedness_after": _nd(agg["grounded_after"], agg["claims_after"]),
        "citation_completeness_before": _nd(agg["citation_supported_before"],
                                            agg["claims_before"]),
        "citation_completeness_after": _nd(agg["citation_supported_after"],
                                           agg["claims_after"]),
        "citation_presence_before": _nd(agg["cited_before"], agg["claims_before"]),
        "citation_presence_after": _nd(agg["cited_after"], agg["claims_after"]),
        "contradicted_before": agg["contradicted_before"],
        "contradicted_after": agg["contradicted_after"],
        "unsupported_before": agg["unsupported_before"],
        "unsupported_after": agg["unsupported_after"],
        "lost_claims": lost_all,
        "lost_supported_claims": len(lost_supported),
        "lost_supported_undocumented": len(undocumented),
        "added_claims": added_all,
        "status_moved": moved_all,
        "fates": dict(fates),
        "corrected": fates.get("corrected", 0),
        "deleted": fates.get("deleted", 0),
        "qualified": fates.get("qualified", 0),
        "deleted_though_grounded": [r for r in lost_all if r["deleted_though_grounded"]],
        "invented_docs": inv["invented_docs"],
        "invented_pages": inv["invented_pages"],
        "invented_figures": inv["invented_figures"],
        "invented_figures_matcher_cannot_place": figures_hard,
        "invented_entities": inv["invented_entities"],
        "sources_not_shown_to_repair": inv["sources_not_shown_to_repair"],
        "newly_cited_pages": inv["newly_cited_pages"],
        "novel_sentences": novel,
        "answer_checks_pass_before": pass_before,
        "answer_checks_pass_after": pass_after,
        "answer_checks_pass_to_fail": len(p2f),
        "answer_checks_pass_to_fail_cases": p2f,
        "answer_checks_fail_to_pass_cases": f2p,
        "check_regressions": regressions,
        "language_regressions": lang_bad,
        "language_per_case": lang_rows,
        "behaviour_per_case": behaviour,
        "field_coverage_before": fields_before,
        "field_coverage_after": fields_after,
        "per_case": per_case,
    }
    if carry:
        report["carry_on_vs_carry_off"] = _carry_gap(carry, on, shared)
    if extras:
        report["cost"] = _replay_cost(on)
        report["repair_call_spread"] = _repair_spread(carry or on)
        report["pre_gate_probe"] = _pre_gate_probe(off, carry or on, shared)
    report["breaches"] = _breaches(report)
    report["gate_pass"] = not report["breaches"]
    return report


def _field_cells(rows: dict, shared: Sequence[str]) -> dict:
    cov = cells = 0
    for i in shared:
        f = rows[i].get("fields")
        if not isinstance(f, dict):
            continue
        cov += int(f.get("n_covered") or 0)
        cells += int(f.get("n_scorable") or 0)
    return {"n": cov, "d": cells, "rate": (cov / cells) if cells else None}


def _carry_gap(carry: dict, off_carry: dict, shared: Sequence[str]) -> dict:
    """carry-ON (what production ships) vs carry-OFF (what the plan gates)."""
    diff, agg = [], defaultdict(int)
    for i in shared:
        a, b = carry.get(i), off_carry.get(i)
        if a is None or b is None:
            continue
        for k, tag in (("claims", "claims"), ("supported", "supported"),
                       ("grounded", "grounded"),
                       ("citation_supported", "citation_supported")):
            agg[f"{tag}_carry_on"] += int(((a.get("claims") or {}).get(k)) or 0)
            agg[f"{tag}_carry_off"] += int(((b.get("claims") or {}).get(k)) or 0)
        if (a.get("answer") or "") != (b.get("answer") or ""):
            diff.append({"case": i, "carry_on_repaired": a.get("repaired"),
                         "carry_off_repaired": b.get("repaired"),
                         "carry_on_notes": a.get("repair_notes"),
                         "carry_off_notes": b.get("repair_notes")})
    return {"answers_that_differ": diff, "totals": dict(agg)}


def _replay_cost(rows: dict) -> dict:
    """Only the REPLAYED calls. A skipped row carries the original release
    run's usage verbatim, and booking the answer call as a repair-pass cost
    would put the generation bill inside the repair envelope."""
    calls = []
    for r in rows.values():
        if not (r.get("repair_replay") or {}).get("replayed"):
            continue
        calls += list((r.get("usage") or {}).get("calls") or [])
    seen, uniq = set(), []
    for c in calls:                                     # one row per call
        k = (c.get("role"), c.get("latency_s"), c.get("prompt_tokens"),
             c.get("completion_tokens"))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(c)
    return _cost(uniq)


def _pre_gate_probe(off: dict, on: dict, shared: Sequence[str]) -> dict:
    """Every rewrite the repair model PROPOSED, gate or no gate.

    Answers the question the adopted-only tables cannot: does the pass invent
    sources, and is the invented-source gate the thing that stops it?
    """
    proposals, inv = 0, {"invented_docs": [], "invented_pages": [],
                         "sources_not_shown_to_repair": [],
                         "invented_figures": [], "invented_entities": [],
                         "newly_cited_pages": []}
    gate_hits, rows = [], []
    for cid in shared:
        pg = ((on.get(cid) or {}).get("repair_replay") or {}).get("repair_pre_gate")
        if not pg:
            continue
        proposals += 1
        pseudo = dict(on[cid], answer=pg["text"], claim_rows=pg["claim_rows"])
        found = invented_sources(off[cid], pseudo)
        for k in inv:
            inv[k] += [dict(x, case=cid) for x in found[k]]
        if pg["introduced_sources"]:
            gate_hits.append({"case": cid, "introduced": pg["introduced_sources"]})
        rows.append({
            "case": cid,
            "failures": [pg["failures_before"], pg["failures_after"]],
            "supported_required": [pg["supported_required_before"],
                                   pg["supported_required_after"]],
            "would_adopt": pg["would_adopt_without_carry"],
            "verify_introduced_sources": pg["introduced_sources"],
            "audit_invented_docs": len(found["invented_docs"]),
            "audit_invented_pages": len(found["invented_pages"]),
            "audit_not_shown": len(found["sources_not_shown_to_repair"]),
            "audit_invented_figures": len(
                [f for f in found["invented_figures"] if not f["matcher_places_it"]]),
        })
    return {"proposals": proposals,
            "verify_gate_caught_introduced_sources": gate_hits,
            "audit_found": {k: len(v) for k, v in inv.items()},
            "detail": inv, "per_case": rows}


def _repair_spread(rows: dict) -> dict:
    """Wave 4's two-sample summary, kept for the records that carry it.

    SUPERSEDED by `_reproducibility`, which reports the same two counts as
    rates over an explicit denominator, names the disagreeing cases, and
    carries the metric spread the flips produce. Two things this one gets
    wrong and that one fixes: `rows` must be the CARRY-ON arm (every sample is
    drawn with the carry live, so comparing a sample against the carry-off
    arm's decision folds the carry difference into the sampling number — the
    published 2/11 is 3/11 like-for-like), and a case the pass never reached
    is not in the denominator.
    """
    n = ident = flip = 0
    flips = []
    for cid, r in rows.items():
        second = (r.get("repair_replay") or {}).get("repair_second_sample")
        if not second:
            continue
        n += 1
        ident += bool(second.get("identical_text"))
        if bool(second.get("adopted")) != bool(r.get("repaired")):
            flip += 1
            flips.append({"case": cid, "sample_1_adopted": bool(r.get("repaired")),
                          "sample_2_adopted": bool(second.get("adopted")),
                          "sample_2_status": second.get("status"),
                          "sample_2_notes": second.get("notes")})
    return {"cases_sampled_twice": n, "identical_repair_text": ident,
            "adoption_flipped": flip, "flips": flips}


# ---------------------------------------------------------------------------
# metric coverage: missing BY CONSTRUCTION vs missing BY FAILURE
# ---------------------------------------------------------------------------
#: What a per-case claim rate's absence means, and which of the two kinds it
#: is. This distinction is the whole fix for the gate line the plan asks for:
#:
#:     eval_answers.py --compare A B --require-metrics groundedness_rate,...
#:
#: exits 1 on any production-mode record, because it counts EVERY absent value
#: as a failure. On the 66-case suite four cases can never carry one:
#:
#:   guard answer   production returns before the verifier runs
#:                  (chainlit_app.py:1116-1133), so the turn has no claims —
#:                  Wave 3 gap 4 made the harness match, deliberately;
#:   chat turn      answered from history, never verified;
#:   0 claims       a correct abstention: the denominator is zero, so the rate
#:                  is undefined. `None` here is the arithmetic being honest,
#:                  not a measurement that went missing.
#:
#: Those are BY CONSTRUCTION: no run of any pipeline will ever produce a value
#: for them, so a gate that counts them is unpassable by construction too, and
#: an unpassable gate teaches an operator to ignore the exit code. What is
#: left is BY FAILURE and still gates: a harness error, a verified turn whose
#: claims block never got written, a turn that carries claims but no rate.
MISSING_BY_CONSTRUCTION = "by-construction"
MISSING_BY_FAILURE = "by-failure"

#: --require-metrics names -> the key inside a record's `claims` block.
#: eval_answers._REQUIRABLE restated, so this tool's flag is spelled the same
#: as the one in the plan's gate line.
REQUIRABLE = {"support_rate": "support_rate",
              "groundedness_rate": "groundedness_rate",
              "citation_completeness_rate": "citation_completeness_rate",
              "citation_presence_rate": "citation_presence_rate"}


def _metric_presence(row: dict, key: str) -> Tuple[Optional[float], str, str]:
    """(value, kind, reason) for one metric of one recorded case."""
    claims = row.get("claims") if isinstance(row.get("claims"), dict) else None
    value = (claims or {}).get(REQUIRABLE.get(key, key))
    if value is not None:
        return float(value), "present", ""
    if row.get("error"):
        return None, MISSING_BY_FAILURE, \
            f"harness error, no answer to verify: {str(row.get('error'))[:80]}"
    if row.get("guard"):
        return None, MISSING_BY_CONSTRUCTION, \
            "guard answer: production returns before verification " \
            "(chainlit_app.py:1116-1133), so the turn has no claims"
    if row.get("chat"):
        return None, MISSING_BY_CONSTRUCTION, \
            "chat turn: production answers from history without verification"
    if claims is None:
        return None, MISSING_BY_FAILURE, \
            "no claims block on a turn production would have verified"
    if not int(claims.get("claims") or 0):
        return None, MISSING_BY_CONSTRUCTION, \
            "0 claims: the rate's denominator is zero — an abstention with " \
            "nothing to verify has no rate to compare"
    return None, MISSING_BY_FAILURE, \
        f"{key} absent from a record carrying {claims.get('claims')} claim(s)"


def _metric_coverage(off: dict, on: dict, shared: Sequence[str],
                     metrics: Sequence[str]) -> dict:
    """Per required metric: the paired comparison, and every absence named.

    The mean is the per-case (macro) mean, which is what ``--compare`` prints
    and therefore what the plan's gate line means. The report's own
    ``groundedness_after`` is the pooled n/d and is the number the >= 95%
    gates use; the two are different statistics on purpose and both are
    printed.
    """
    out = {}
    for key in metrics:
        paired, by_c, by_f = [], [], []
        for cid in shared:
            va, ka, wa = _metric_presence(off[cid], key)
            vb, kb, wb = _metric_presence(on[cid], key)
            if ka == "present" and kb == "present":
                paired.append((va, vb))
                continue
            for side, kind, why in (("off", ka, wa), ("on", kb, wb)):
                if kind == "present":
                    continue
                (by_c if kind == MISSING_BY_CONSTRUCTION else by_f).append(
                    {"case": cid, "side": side, "kind": kind, "reason": why})
        a = sum(x for x, _ in paired) / len(paired) if paired else None
        b = sum(y for _, y in paired) / len(paired) if paired else None
        out[key] = {
            "cases_compared": len(shared),
            "paired": len(paired),
            "a": a, "b": b,
            "delta": None if a is None else b - a,
            "no_regression": bool(paired) and b >= a - 1e-9 and not by_f,
            "missing_by_construction": by_c,
            "missing_by_failure": by_f,
            "excluded_by_construction": sorted({r["case"] for r in by_c}),
        }
    return out


# ---------------------------------------------------------------------------
# reproducibility: N samples of repair on IDENTICAL input
# ---------------------------------------------------------------------------
def _samples_of(row: dict) -> List[dict]:
    """This case's repair samples, newest record shape or Wave 4's.

    Wave 4 recorded a single extra sample as ``repair_second_sample`` with no
    metrics and no text. It is read here as a two-entry list so the old
    records still answer the two questions they CAN answer — was the
    completion identical, did the adoption decision agree — and are honestly
    unable to answer the third (what it did to the gated metrics), rather
    than being silently counted as 'no movement'.
    """
    rr = row.get("repair_replay") or {}
    got = rr.get("repair_samples")
    if got:
        return list(got)
    legacy = rr.get("repair_second_sample")
    if not legacy:
        return []
    first = {"sample": 1, "identical_text": True,
             "text_sha256": rr.get("repair_text_sha256"),
             "adopted": bool(row.get("repaired")),
             "rejected": bool(row.get("repair_rejected")),
             "status": row.get("verify_status"),
             "notes": list(row.get("repair_notes") or []),
             "metrics": None, "legacy": True}
    return [first, dict(legacy, sample=2, legacy=True)]


def _sample_base(on: dict, carry: dict) -> Tuple[dict, str, bool]:
    """(rows, arm, mismatch) — which arm the samples must be compared against.

    Every sample is drawn with ``_carry_cleared`` LIVE (sample 1 IS the ON
    arm's call), so the sample-1 baseline has to be the carry-ON arm. Wave 4
    ran the audit with ``--on`` pointing at the carry-OFF arm, so its
    published 'adoption flipped 2/11' compared a carry-ON sample against a
    carry-OFF decision and folded the carry difference into the sampling
    number. Like-for-like on the same record is 3/11.
    """
    for rows, label in ((carry or {}, "carry-on"), (on or {}, "--on")):
        if not rows:
            continue
        arms = {(r.get("repair_replay") or {}).get("arm") for r in rows.values()}
        if "on" in arms:
            return rows, label, False
    arms = {(r.get("repair_replay") or {}).get("arm") for r in (on or {}).values()}
    named = sorted(a for a in arms if a)
    return on, "--on", bool(named) and "on" not in named


def _reproducibility(off: dict, base: dict, shared: Sequence[str],
                     base_arm: str, mismatch: bool) -> dict:
    """Repair's run-to-run spread on IDENTICAL input, as a first-class metric.

    (i) identical-completion rate, (ii) adoption-decision agreement rate,
    (iii) the per-case disagreements, (iv) — via ``_sample_worlds`` — what
    those disagreements do to every gated metric.

    The denominator is cases the pass was actually SAMPLED on, not all 66:
    a case with no failing claim never reaches the repair model and would
    count as perfect agreement for free.
    """
    with_repair = sampled = identical = agree = 0
    unsampled, disagreements, per_case = [], [], []
    for cid in shared:
        row = base.get(cid) or {}
        rr = row.get("repair_replay") or {}
        attempted = bool(rr.get("repair_calls"))
        with_repair += attempted
        samples = _samples_of(row)
        if len(samples) < 2:
            if attempted:
                unsampled.append(cid)
            continue
        sampled += 1
        same_text = all(bool(s.get("identical_text")) for s in samples[1:])
        identical += bool(same_text)
        decisions = [bool(s.get("adopted")) for s in samples]
        agreed = len(set(decisions)) == 1
        agree += bool(agreed)
        entry = {"case": cid, "samples": len(samples),
                 "identical_text": bool(same_text),
                 "adoption": decisions,
                 "statuses": [s.get("status") for s in samples],
                 "text_sha256": [s.get("text_sha256") for s in samples],
                 "notes": [s.get("notes") for s in samples]}
        per_case.append(entry)
        if not agreed:
            disagreements.append(entry)
    return {
        "cases_with_repair": with_repair,
        "cases_sampled": sampled,
        "cases_attempted_but_sampled_once": unsampled,
        "baseline_arm": base_arm,
        "carry_mismatch": bool(mismatch),
        "identical_completion": _nd(identical, sampled),
        "adoption_agreement": _nd(agree, sampled),
        "adoption_disagreement_rate": ((sampled - agree) / sampled) if sampled else None,
        "disagreements": disagreements,
        "per_case": per_case,
    }


def _world_row(off_row: dict, base_row: dict, sample: dict) -> Optional[dict]:
    """The arm row this case would have contributed had THIS sample been the
    one the pass produced — or None when the record cannot say.

    Three resolvable shapes and one unresolvable one:

      sample 1 / identical completion  the base row, unchanged
      rewrite NOT adopted              the turn ships the original answer, and
                                       that row already exists: the off arm's
      rewrite adopted, text recorded   the base row with this sample's text,
                                       claims, checks and claim rows
      rewrite adopted, text NOT        unresolvable — Wave 4's record kept the
      recorded                         sha256 of the rewrite and nothing else
    """
    if sample is None or off_row is None or base_row is None:
        return None
    if sample.get("sample") == 1 or sample.get("identical_text"):
        return base_row
    if not sample.get("adopted"):
        row = dict(off_row)
        row.update({"repaired": False,
                    "repair_rejected": bool(sample.get("rejected")),
                    "repair_notes": list(sample.get("notes") or []),
                    "verify_status": sample.get("status") or row.get("verify_status")})
        return row
    if not sample.get("text") or not sample.get("claim_rows"):
        return None
    row = dict(base_row)
    row.update({"answer": sample["text"], "claim_rows": sample["claim_rows"],
                "claims": sample.get("claims") or row.get("claims"),
                "checks": sample.get("checks") or row.get("checks"),
                "repaired": True, "repair_rejected": False,
                "repair_notes": list(sample.get("notes") or []),
                "verify_status": sample.get("status") or row.get("verify_status")})
    return row


def _sample_worlds(off: dict, base: dict, shared: Sequence[str],
                   justified: dict) -> dict:
    """One full audit per sample index — the spread on every gated metric.

    World k is the run this would have been had every case used its k-th
    repair sample. Each world is scored by ``_compare_arms``, the SAME code
    that scores the real comparison, so a world's groundedness and the
    report's groundedness are the same measurement.

    Every world is restricted to the cases resolvable in ALL of them, so the
    worlds are comparable to each other; the excluded cases are named, and a
    spread computed over 65 of 66 cases says so rather than pretending.
    """
    per_case = {cid: _samples_of(base.get(cid) or {}) for cid in shared}
    depth = max([len(v) for v in per_case.values()] or [0])
    if depth < 2:
        return {"worlds": [], "depth": depth, "cases": 0, "unresolvable": [],
                "reason": "no case carries a second repair sample"}
    rows = {k: {} for k in range(1, depth + 1)}
    unresolvable = []
    for cid in shared:
        samples = per_case[cid]
        if not samples:                    # repair never ran: identical in every world
            for k in rows:
                rows[k][cid] = base[cid]
            continue
        if len(samples) < depth:
            unresolvable.append({"case": cid,
                                 "why": f"sampled {len(samples)} time(s), not {depth}"})
            continue
        built = {}
        for k in range(1, depth + 1):
            r = _world_row(off.get(cid), base.get(cid), samples[k - 1])
            if r is None:
                break
            built[k] = r
        if len(built) != depth:
            unresolvable.append(
                {"case": cid, "why": "an adopted rewrite's text was not recorded — "
                                     "pre-spread record, sha256 only"})
            continue
        for k, r in built.items():
            rows[k][cid] = r
    cases = sorted(rows[1])
    off_sub = {cid: off[cid] for cid in cases}
    worlds = []
    for k in range(1, depth + 1):
        rep = _compare_arms(off_sub, {cid: rows[k][cid] for cid in cases},
                            justified, extras=False)
        worlds.append({"world": k, "cases": len(cases),
                       "quantities": _gate_quantities(rep),
                       "breaches": rep["breaches"]})
    return {"worlds": worlds, "depth": depth, "cases": len(cases),
            "unresolvable": unresolvable,
            "excluded_cases": [u["case"] for u in unresolvable]}


# ---------------------------------------------------------------------------
# gates, and the spread that decides whether a verdict is worth anything
# ---------------------------------------------------------------------------
#: gate id -> how the quantity is read out of a report. One place, so a
#: sample world, a replicate audit and this report are all read identically.
def _gate_quantities(rep: dict) -> Dict[str, Any]:
    def n(key):
        v = rep.get(key)
        return len(v) if isinstance(v, list) else int(v or 0)
    sup_b, sup_a = int(rep.get("supported_before") or 0), int(rep.get("supported_after") or 0)
    return {
        "invented_docs": n("invented_docs"),
        "invented_pages": n("invented_pages"),
        "sources_not_shown_to_repair": n("sources_not_shown_to_repair"),
        "invented_figures": n("invented_figures_matcher_cannot_place"),
        "lost_supported_undocumented": n("lost_supported_undocumented"),
        "answer_checks_pass_to_fail": n("answer_checks_pass_to_fail"),
        "language_regressions": n("language_regressions"),
        "supported_non_decreasing": sup_a - sup_b,
        "groundedness": (rep.get("groundedness_after") or {}).get("rate"),
        "citation_completeness": (rep.get("citation_completeness_after") or {}).get("rate"),
    }


def _range(values: Sequence[Any]) -> Optional[float]:
    xs = [v for v in values if v is not None]
    return (max(xs) - min(xs)) if len(xs) > 1 else None


def _spreads(mine: dict, worlds: Sequence[dict],
             replicates: Sequence[dict]) -> dict:
    """Per gate quantity: how far it moved between measurements of the same
    thing. Two independent sources, reported separately and then combined by
    taking the WIDER — a gate certified against the narrower of two known
    spreads is certified against the more flattering one.

      samples     N repair samples on identical input, within one replay.
                  This is the sampling the A/B cannot cancel: the answers are
                  fixed and the judge pass is shared, so the repair call is
                  the only draw left.
      replicates  whole recorded audits of the same source record, supplied
                  with --spread-audit. Wider by construction: a second replay
                  redraws the judge too.
    """
    out = {}
    for key in mine:
        s_samples = _range([w.get(key) for w in worlds])
        s_repl = _range([mine.get(key)] + [r.get(key) for r in replicates]) \
            if replicates else None
        best = max([s for s in (s_samples, s_repl) if s is not None], default=None)
        source = None
        if best is not None:
            source = "samples" if best == s_samples else "replicates"
            if s_samples is not None and s_repl is not None and s_samples == s_repl:
                source = "samples+replicates"
        out[key] = {"value": best, "from_samples": s_samples,
                    "from_replicates": s_repl, "source": source,
                    "n_samples": len(worlds), "n_replicates": len(replicates) + 1}
    return out


def _gate_specs(rep: dict) -> List[dict]:
    """The gate table: what Wave 4 reported in prose, as data.

    ``kind`` decides how a spread applies to the verdict:

      event   a thing that either happened or did not — an invented citation,
              a supported claim lost, an answer that went pass -> fail. An
              observed breach is DETERMINATE: the rewrite really did name a
              document the evidence does not hold, and another sample not
              doing it does not un-invent it. A clean run is NOT determinate
              when a sibling sample was dirty, because 'zero this time' is
              then a draw, not a property.
      margin  a signed distance from a threshold (a rate against 95%, the
              supported delta against 0). Symmetric: if the distance is
              inside the spread, neither PASS nor FAIL is supportable.
    """
    q = _gate_quantities(rep)
    specs = []

    def event(gid, name, breach):
        v = q[gid]
        specs.append({"id": gid, "name": name, "kind": "event", "value": v,
                      "threshold": 0, "margin": v, "unit": "count",
                      "failed": v > 0, "breach": breach(v) if v else None,
                      "blocking": True})

    event("invented_docs", "no document citation invented",
          lambda n: f"{n} invented document citation(s)")
    event("invented_pages", "no page citation invented",
          lambda n: f"{n} invented page citation(s)")
    event("sources_not_shown_to_repair", "no citation on an unshown document",
          lambda n: f"{n} citation(s) on a document the repair pass was never shown")
    event("invented_figures", "no figure invented",
          lambda n: f"{n} invented figure(s)")
    event("lost_supported_undocumented", "no supported claim lost undocumented",
          lambda n: f"{n} supported claim(s) lost with no documented necessity")
    event("answer_checks_pass_to_fail", "no correct answer becomes incorrect",
          lambda n: f"{n} answer(s) pass -> fail")
    event("language_regressions", "no language regression",
          lambda n: f"{n} language regression(s)")

    delta = q["supported_non_decreasing"]
    specs.append({
        "id": "supported_non_decreasing", "name": "supported non-decreasing",
        "kind": "margin", "value": delta, "threshold": 0, "margin": delta,
        "unit": "claims", "failed": delta < 0, "blocking": True,
        "breach": (f"supported fell {rep.get('supported_before')} -> "
                   f"{rep.get('supported_after')}") if delta < 0 else None})

    for gid, name, key in (
            ("groundedness", "groundedness >= 95%", "groundedness_after"),
            ("citation_completeness", "citation completeness >= 95%",
             "citation_completeness_after")):
        nd = rep.get(key) or {}
        rate = nd.get("rate")
        specs.append({
            "id": gid, "name": name, "kind": "margin",
            "value": rate, "threshold": 0.95,
            "margin": None if rate is None else rate - 0.95, "unit": "rate",
            "failed": bool(rate is not None and rate < 0.95), "blocking": True,
            "detail": f"{nd.get('n')}/{nd.get('d')}"
                      + (f" (>= {nd['threshold_95']}/{nd['d']})"
                         if nd.get("threshold_95") else ""),
            "breach": (f"{name.split(' >=')[0]} {nd.get('n')}/{nd.get('d')} "
                       f"({rate:.1%}) below the >= {nd.get('threshold_95')}/"
                       f"{nd.get('d')} gate") if rate is not None and rate < 0.95
                      else None})
    return specs


def _breaches(rep: dict) -> List[str]:
    """The spread-BLIND breach list, kept because a sample world and a
    replicate report are scored without one."""
    return [g["breach"] for g in _gate_specs(rep) if g["failed"] and g["breach"]]


def _gate_verdicts(rep: dict, spreads: dict, require_metrics: Sequence[str] = (),
                   max_adoption_disagreement: Optional[float] = None,
                   require_spread: bool = False,
                   repro: dict = None) -> List[dict]:
    """Every gate with PASS / FAIL / INDETERMINATE / n-a and the reason.

    The plan's rule, enforced instead of narrated: *any gate whose margin is
    narrower than the measured spread is reported indeterminate, not passed*.
    """
    out = []
    for spec in _gate_specs(rep):
        sp = (spreads or {}).get(spec["id"]) or {}
        spread = sp.get("value")
        g = dict(spec, spread=spread, spread_source=sp.get("source"), why="")
        if spec["value"] is None:
            g["verdict"] = "n/a"
            g["why"] = "the metric has no denominator on this record"
        elif spec["failed"]:
            if spec["kind"] == "margin" and spread is not None \
                    and abs(spec["margin"]) < spread:
                g["verdict"] = "INDETERMINATE"
                g["why"] = (f"margin {_fmt_margin(spec)} is inside the measured "
                            f"spread {_fmt_spread(spec, spread)}")
            else:
                g["verdict"] = "FAIL"
                g["why"] = spec["breach"] or "breach"
        elif spec["kind"] == "event" and spread:
            g["verdict"] = "INDETERMINATE"
            g["why"] = (f"clean here, but the same count moved "
                        f"{_fmt_spread(spec, spread)} between measurements of "
                        f"the same input")
        elif spec["kind"] == "margin" and spread is not None \
                and abs(spec["margin"]) < spread:
            g["verdict"] = "INDETERMINATE"
            g["why"] = (f"margin {_fmt_margin(spec)} is inside the measured "
                        f"spread {_fmt_spread(spec, spread)}")
        elif spread is None and require_spread:
            g["verdict"] = "INDETERMINATE"
            g["why"] = ("no replicate measurement: the spread is unknown and "
                        "--require-spread was asked for")
        else:
            g["verdict"] = "PASS"
            g["why"] = ("clean, and the spread is measured at "
                        f"{_fmt_spread(spec, spread)}" if spread is not None
                        else "clean")
        out.append(g)

    # --- metric coverage (the gate that could not pass) -------------------
    cov = rep.get("metric_coverage") or {}
    wanted = list(require_metrics or ())
    if cov:
        failures = [r for k in (wanted or cov) for r in
                    (cov.get(k) or {}).get("missing_by_failure") or []]
        regressed = [k for k in (wanted or ()) if not (cov.get(k) or {}).get("no_regression")]
        bad = bool(failures or regressed)
        out.append({
            "id": "metric_coverage", "name": "required metrics present and not regressed",
            "kind": "event", "value": len(failures) + len(regressed),
            "threshold": 0, "margin": len(failures) + len(regressed),
            "unit": "count", "failed": bad and bool(wanted),
            "blocking": bool(wanted), "spread": None, "spread_source": None,
            "verdict": ("FAIL" if (bad and wanted) else
                        "PASS" if wanted else "reported"),
            "detail": _coverage_detail(cov, wanted or list(cov)),
            "why": ("; ".join([f["reason"] for f in failures[:3]]
                              + [f"{k}: regression" for k in regressed])
                    if bad else "every absent value is absent by construction"),
            "breach": (f"{len(failures)} metric value(s) missing by failure, "
                       f"{len(regressed)} regressed") if (bad and wanted) else None})

    # --- reproducibility --------------------------------------------------
    repro = repro or rep.get("reproducibility") or {}
    tol = (DEFAULT_MAX_ADOPTION_DISAGREEMENT if max_adoption_disagreement is None
           else float(max_adoption_disagreement))
    asked = max_adoption_disagreement is not None
    rate = repro.get("adoption_disagreement_rate")
    sampled = int(repro.get("cases_sampled") or 0)
    if sampled:
        failed = rate > tol + 1e-9
        n = repro["adoption_agreement"]["n"]
        out.append({
            "id": "adoption_reproducibility",
            "name": f"adoption decision reproducible (<= {tol:.0%} disagreement)",
            "kind": "event", "value": rate, "threshold": tol, "margin": rate,
            "unit": "rate", "failed": failed, "blocking": True,
            "spread": None, "spread_source": None,
            "verdict": "FAIL" if failed else "PASS",
            "detail": f"agreement {n}/{sampled}, identical completions "
                      f"{repro['identical_completion']['n']}/{sampled}",
            "why": (f"{sampled - n} of {sampled} sampled case(s) changed the "
                    f"adoption decision on identical input" if failed
                    else f"{n}/{sampled} sampled case(s) agreed"),
            "breach": (f"repair is not reproducible: adoption disagreed on "
                       f"{sampled - n}/{sampled} sampled case(s) "
                       f"({rate:.1%} > {tol:.1%})") if failed else None})
    elif not asked:
        # Named but not measured, so a reader of the table sees that the
        # property exists and this run says nothing about it.
        out.append({
            "id": "adoption_reproducibility",
            "name": f"adoption decision reproducible (<= {tol:.0%} disagreement)",
            "kind": "event", "value": None, "threshold": tol, "margin": None,
            "unit": "rate", "failed": False, "blocking": False,
            "spread": None, "spread_source": None, "verdict": "not-measured",
            "detail": f"{repro.get('cases_with_repair', 0)} case(s) reached the "
                      f"repair model, none sampled more than once",
            "why": "replay with --repair-samples N to measure it",
            "breach": None})
    else:
        # A gate with no evidence must not pass. This is the mirror of the
        # unpassable-gate bug: a gate that passes vacuously is not a gate
        # either, and --repair-samples 1 would otherwise 'prove' pinning.
        out.append({
            "id": "adoption_reproducibility",
            "name": f"adoption decision reproducible (<= {tol:.0%} disagreement)",
            "kind": "event", "value": None, "threshold": tol, "margin": None,
            "unit": "rate", "failed": True, "blocking": True,
            "spread": None, "spread_source": None, "verdict": "FAIL",
            "detail": "0 cases sampled more than once",
            "why": "the gate was requested and nothing was sampled twice",
            "breach": "reproducibility gate requested but no case carries a "
                      "second repair sample (replay with --repair-samples N)"})
    return out


def _fmt_margin(spec: dict) -> str:
    if spec["unit"] == "rate":
        return f"{spec['margin'] * 100:+.1f} pp"
    return f"{spec['margin']:+d}" if isinstance(spec["margin"], int) \
        else f"{spec['margin']}"


def _fmt_spread(spec: dict, spread) -> str:
    return f"{spread * 100:.1f} pp" if spec["unit"] == "rate" else f"{spread}"


def _coverage_detail(cov: dict, keys: Sequence[str]) -> str:
    bits = []
    for k in keys:
        c = cov.get(k) or {}
        by_c = {r["case"] for r in c.get("missing_by_construction") or []}
        by_f = {r["case"] for r in c.get("missing_by_failure") or []}
        bits.append(f"{k} {c.get('paired')}/{c.get('cases_compared')} paired, "
                    f"{len(by_c)} case(s) missing by construction, "
                    f"{len(by_f)} by failure")
    return "; ".join(bits)


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------
def _fmt_nd(b: dict) -> str:
    if not b or not b.get("d"):
        return "   n/a"
    return f"{b['n']}/{b['d']} ({b['rate']:.1%})"


def print_report(rep: dict, verbose: bool = False) -> None:
    w = 92
    print("=" * w)
    print(f"REPAIR A/B — {rep['cases_compared']} cases, "
          f"{rep['answers_rewritten']} answer(s) rewritten")
    print("=" * w)
    print(f"  off : {rep['off']}")
    print(f"  on  : {rep['on']}")
    print()
    print(f"{'metric':30} {'before':>18} {'after':>18}   gate")
    print("-" * w)
    for name, kb, ka, gate in (
            ("claims (denominator)", "claims_before", "claims_after", ""),
            ("supported", "supported_before", "supported_after", "non-decreasing"),
    ):
        print(f"{name:30} {rep[kb]:>18} {rep[ka]:>18}   {gate}")
    for name, kb, ka, gate in (
            ("groundedness", "groundedness_before", "groundedness_after", ">= 95%"),
            ("citation completeness", "citation_completeness_before",
             "citation_completeness_after", ">= 95%"),
            ("citation presence", "citation_presence_before",
             "citation_presence_after", "reported"),
    ):
        b, a = rep[kb], rep[ka]
        print(f"{name:30} {_fmt_nd(b):>18} {_fmt_nd(a):>18}   {gate}"
              + (f"  (>= {a['threshold_95']}/{a['d']})" if gate == ">= 95%" and a.get("d") else ""))
    print(f"{'field coverage':30} {_fmt_nd(rep['field_coverage_before']):>18} "
          f"{_fmt_nd(rep['field_coverage_after']):>18}")
    print(f"{'answer checks pass':30} {rep['answer_checks_pass_before']:>18} "
          f"{rep['answer_checks_pass_after']:>18}   pass->fail == 0")
    print()
    print(f"CLAIM MOVEMENT   lost {len(rep['lost_claims'])}  "
          f"added {len(rep['added_claims'])}  status-moved {len(rep['status_moved'])}")
    print(f"  fates: {rep['fates'] or '{}'}")
    print(f"  corrected {rep['corrected']}   deleted {rep['deleted']}   "
          f"qualified {rep['qualified']}   "
          f"deleted-though-grounded {len(rep['deleted_though_grounded'])}")
    for r in rep["lost_claims"]:
        print(f"  - [{r['status_before']:<12} grounded={str(r['grounded_before']):<5} "
              f"{r['fate']:<22}] {r['case']}")
        print(f"      before: {r['text'][:150]}")
        if r["successor_text"]:
            print(f"      after : {r['successor_text'][:150]}  "
                  f"(sim {r['successor_similarity']}, {r['successor_status']})")
    print()
    print("INVENTED SOURCES (checked independently of verify's own gate)")
    for k in ("invented_docs", "invented_pages", "sources_not_shown_to_repair",
              "invented_entities"):
        print(f"  {k:34} {len(rep[k])}")
        for r in rep[k][:10]:
            print(f"      {r}")
    print(f"  {'invented_figures (literal digits)':34} {len(rep['invented_figures'])}")
    for r in rep["invented_figures"][:20]:
        print(f"      {r}")
    print(f"  {'  of which matcher cannot place':34} "
          f"{len(rep['invented_figures_matcher_cannot_place'])}   <- the gate")
    print(f"  {'newly cited pages (all held)':34} "
          f"{len(rep['newly_cited_pages'])}   reported, not gated")
    for r in rep["newly_cited_pages"][:10]:
        print(f"      {r}")
    print(f"  {'novel sentences':34} {len(rep['novel_sentences'])}")
    for r in rep["novel_sentences"][:20] if verbose else []:
        print(f"      [{r['case']}] grounded={r['grounded']} hedge={r['hedge']} "
              f"{r['text'][:110]}")
    print()
    if rep.get("carry_on_vs_carry_off"):
        c = rep["carry_on_vs_carry_off"]
        print(f"CARRY-ON vs CARRY-OFF   answers that differ: "
              f"{len(c['answers_that_differ'])}")
        print(f"  totals: {c['totals']}")
        for r in c["answers_that_differ"][:10]:
            print(f"    {r}")
        print()
    pg = rep.get("pre_gate_probe") or {}
    if pg.get("proposals"):
        print(f"ADVERSARIAL PROBE — every rewrite PROPOSED, adopted or not "
              f"({pg['proposals']} proposals)")
        print(f"  verify's own gate caught introduced sources in "
              f"{len(pg['verify_gate_caught_introduced_sources'])} of them")
        for r in pg["verify_gate_caught_introduced_sources"]:
            print(f"    {r}")
        print(f"  this audit, over the SAME pre-gate text: {pg['audit_found']}")
        for r in (pg.get("detail") or {}).get("invented_docs", [])[:10]:
            print(f"    invented doc  {r}")
        for r in (pg.get("detail") or {}).get("invented_pages", [])[:10]:
            print(f"    invented page {r}")
        for r in [f for f in (pg.get("detail") or {}).get("invented_figures", [])
                  if not f["matcher_places_it"]][:15]:
            print(f"    invented fig  {r}")
        print(f"  {'case':28} {'fails':>9} {'req-sup':>9} {'would adopt':>12}")
        for r in pg["per_case"]:
            print(f"  {r['case']:28} {str(r['failures']):>9} "
                  f"{str(r['supported_required']):>9} {str(r['would_adopt']):>12}"
                  f"   inv doc/page/fig/not-shown "
                  f"{r['audit_invented_docs']}/{r['audit_invented_pages']}/"
                  f"{r['audit_invented_figures']}/{r['audit_not_shown']}")
        print()
    _print_reproducibility(rep)
    _print_coverage(rep)
    print("COST / LATENCY of the replayed calls")
    for role, c in sorted((rep.get("cost") or {}).items()):
        print(f"  {role:<14} {c['calls']:>3} calls  {c['prompt_tokens']:>7}p + "
              f"{c['completion_tokens']:>6}c  p50 {c['latency_p50_s']}s  "
              f"p95 {c['latency_p95_s']}s  total {c['latency_total_s']}s  "
              f"${c['cost_usd']:.4f}")
    print()
    _print_gates(rep)


def _print_reproducibility(rep: dict) -> None:
    r = rep.get("reproducibility") or {}
    if not r.get("cases_sampled"):
        if r.get("cases_with_repair"):
            print(f"REPRODUCIBILITY   not measured: {r['cases_with_repair']} case(s) "
                  f"reached the repair model, none sampled twice")
            print("  replay with --repair-samples N to measure it")
            print()
        return
    n = r["cases_sampled"]
    print(f"REPRODUCIBILITY — {n} case(s) sampled, repair re-run on IDENTICAL "
          f"answers and IDENTICAL verdict objects")
    print(f"  baseline arm                {r['baseline_arm']}"
          + ("   *** samples are carry-ON; this arm is not — the numbers below "
             "mix carry with sampling ***" if r.get("carry_mismatch") else ""))
    print(f"  identical completion        {_fmt_nd(r['identical_completion'])}")
    print(f"  adoption-decision agreement {_fmt_nd(r['adoption_agreement'])}"
          f"   disagreement {r['adoption_disagreement_rate']:.1%}")
    for d in r.get("disagreements") or []:
        print(f"    {d['case']:28} adopted={d['adoption']} "
              f"statuses={d['statuses']}")
        for note in d.get("notes") or []:
            print(f"        {note}")
    if r.get("cases_attempted_but_sampled_once"):
        print(f"  sampled once only: {', '.join(r['cases_attempted_but_sampled_once'])}")
    w = rep.get("sample_worlds") or {}
    if w.get("worlds"):
        print(f"  SPREAD ON THE GATED METRICS across {len(w['worlds'])} sample "
              f"world(s), {w['cases']} case(s) each")
        head = f"    {'quantity':32}" + "".join(
            f"{'world ' + str(x['world']):>12}" for x in w["worlds"]) + f"{'spread':>10}"
        print(head)
        for key in _gate_quantities(rep):
            vals = [x["quantities"].get(key) for x in w["worlds"]]
            sp = (rep.get("spreads") or {}).get(key) or {}
            print(f"    {key:32}"
                  + "".join(f"{_fmt_q(v):>12}" for v in vals)
                  + f"{_fmt_q(sp.get('from_samples')):>10}")
        for u in w.get("unresolvable") or []:
            print(f"    excluded: {u['case']:26} {u['why']}")
    elif w.get("reason"):
        print(f"  metric spread: {w['reason']}")
    for r_ in rep.get("spread_replicates") or []:
        print(f"  replicate audit: {r_['source']} ({r_['cases_compared']} cases)")
    print()


def _fmt_q(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return f"{v:+d}" if isinstance(v, int) and v < 0 else str(v)


def _print_coverage(rep: dict) -> None:
    cov = rep.get("metric_coverage") or {}
    if not cov:
        return
    want = rep.get("require_metrics") or []
    print("METRIC COVERAGE   (per-case means, the statistic --compare uses)")
    for key, c in cov.items():
        gated = " [gated]" if key in want else ""
        delta = f"{c['delta']:+.1%}" if c["delta"] is not None else "n/a"
        print(f"  {key:28}{gated:8} {c['paired']}/{c['cases_compared']} paired  "
              f"{(c['a'] or 0):.1%} -> {(c['b'] or 0):.1%}  {delta}")
        for r in c["missing_by_construction"]:
            print(f"      by-construction  {r['case']:22} ({r['side']}) {r['reason'][:70]}")
        for r in c["missing_by_failure"]:
            print(f"      BY-FAILURE       {r['case']:22} ({r['side']}) {r['reason'][:70]}")
    print()


def _print_gates(rep: dict) -> None:
    print(f"{'GATES':52} {'value':>7} {'verdict':>14}  spread")
    print("-" * 92)
    for g in rep.get("gates") or []:
        val = g.get("value")
        shown = (f"{val:.1%}" if g.get("unit") == "rate" and val is not None
                 else "-" if val is None else str(val))
        sp = g.get("spread")
        sp_s = ("-" if sp is None else
                f"{sp * 100:.1f} pp" if g.get("unit") == "rate" else str(sp))
        src = f" ({g['spread_source']})" if g.get("spread_source") else ""
        print(f"  {g['name']:50} {shown:>7} {g['verdict']:>14}  {sp_s}{src}")
        if g.get("detail"):
            print(f"      {g['detail']}")
        if g.get("why"):
            print(f"      {g['why']}")
    print()
    print(f"GATE: {rep.get('gate_verdict')}")
    for b in rep.get("breaches") or []:
        print(f"  - FAIL          {b}")
    for b in rep.get("indeterminate") or []:
        print(f"  - INDETERMINATE {b}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _replay_parser():
    ap = argparse.ArgumentParser(prog="audit_repair.py replay",
                                 description="replay a recorded run through "
                                             "the verifier with and without repair")
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--out-prefix", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--repair-samples", type=int, default=1,
                    help="repair calls per failing case on IDENTICAL input. "
                         ">1 measures reproducibility: identical-completion "
                         "rate, adoption agreement, and the spread the flips "
                         "put on every gated metric. Costs one extra repair "
                         "call per failing case per extra sample.")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--verbose", "-v", action="store_true")
    return ap


def _audit_parser():
    ap = argparse.ArgumentParser(
        prog="audit_repair.py",
        description="compare two replayed arms; one JSON object on stdout")
    ap.add_argument("--off", type=Path, required=True)
    ap.add_argument("--on", type=Path, required=True)
    ap.add_argument("--carry-on", type=Path,
                    help="the carry-ON arm: reported beside the gated "
                         "carry-OFF one, AND the baseline the repair samples "
                         "are compared against (they are drawn with the carry "
                         "live, so without this the reproducibility numbers "
                         "mix the carry difference into the sampling one)")
    ap.add_argument("--justifications", type=Path,
                    help="jsonl of {key, justification} documenting a "
                         "necessary supported-claim loss")
    ap.add_argument("--spread-audit", type=Path, action="append", default=[],
                    metavar="AUDIT.json",
                    help="another recorded audit of the same source record; "
                         "its gated quantities widen the measured spread. "
                         "Repeatable.")
    ap.add_argument("--require-metrics",
                    type=lambda s: [x for x in s.split(",") if x], default=[],
                    metavar="A,B",
                    help="gate on these claim rates being present and not "
                         "regressed. A value missing BY CONSTRUCTION (guard "
                         "turn, chat turn, 0-claim abstention) is not a "
                         "failure; one missing by failure is.")
    ap.add_argument("--max-adoption-disagreement", type=float, default=None,
                    metavar="R",
                    help="exit non-zero when repair's adoption decision "
                         f"disagrees on more than R of the sampled cases "
                         f"(default {DEFAULT_MAX_ADOPTION_DISAGREEMENT:g}). "
                         "Passing it when nothing was sampled twice is a "
                         "failure, not a free pass.")
    ap.add_argument("--require-spread", action="store_true",
                    help="a gate with no replicate measurement behind it "
                         "reads INDETERMINATE rather than PASS")
    ap.add_argument("--json", type=Path, help="also write the report here")
    ap.add_argument("--quiet", action="store_true", help="JSON only")
    ap.add_argument("--verbose", "-v", action="store_true")
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "replay":
        args = _replay_parser().parse_args(argv[1:])
        got = replay(args.release, args.out_prefix, force=args.force,
                     repair_samples=args.repair_samples, limit=args.limit,
                     verbose=args.verbose)
        print(json.dumps(got, indent=1))
        return 0
    args = _audit_parser().parse_args(argv)
    rep = audit(args.off, args.on, carry_on_path=args.carry_on,
                justifications=args.justifications,
                spread_audits=args.spread_audit,
                require_metrics=args.require_metrics,
                max_adoption_disagreement=args.max_adoption_disagreement,
                require_spread=args.require_spread)
    if not args.quiet:
        print_report(rep, verbose=args.verbose)
        print()
    payload = {k: rep[k] for k in (
        "cases_compared", "answers_rewritten", "claims_before", "claims_after",
        "supported_before", "supported_after", "groundedness_after",
        "citation_completeness_after", "invented_docs", "invented_pages",
        "invented_figures", "invented_figures_matcher_cannot_place",
        "invented_entities", "sources_not_shown_to_repair",
        "lost_claims", "lost_supported_claims", "lost_supported_undocumented",
        "answer_checks_pass_to_fail", "corrected", "deleted", "qualified",
        "spreads", "metric_coverage", "gates",
        "breaches", "indeterminate", "gate_verdict", "gate_pass")}
    # the per-case sample table is in --json; the contract line carries the
    # rates, the disagreements and nothing that scrolls a terminal off screen
    payload["reproducibility"] = {k: v for k, v in rep["reproducibility"].items()
                                  if k != "per_case"}
    print(json.dumps(payload, indent=1, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    return 0 if rep["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
