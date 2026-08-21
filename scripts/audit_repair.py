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


def replay(release: Path, out_prefix: Path, force: bool = False,
           client: Any = None, repair_samples: int = 1,
           limit: Optional[int] = None, verbose: bool = False) -> dict:
    """Replay one recorded run through the verifier twice, three scorings.

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

        # --- optional: a SECOND repair sample, to size repair-call spread ---
        second = None
        if repair_samples > 1 and failed:
            m2 = MeteredClient(client, model)
            r2 = verify.repair(raw, verdicts, ev, client=m2)
            t2 = m2.repair_texts[0] if m2.repair_texts else None
            second = {"identical_text": (t2 or "") == (repair_text or ""),
                      "text_sha256": _sha256_text(t2 or ""),
                      "adopted": bool(r2.repaired),
                      "status": r2.status,
                      "notes": list(r2.notes or []),
                      "calls": m2.calls}
            call_rows += [dict(c, case=cid, sample=2) for c in m2.calls]

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
          justifications: Path = None) -> dict:
    off = {r["id"]: r for r in read_jsonl(off_path)}
    on = {r["id"]: r for r in read_jsonl(on_path)}
    carry = {r["id"]: r for r in read_jsonl(carry_on_path)} if carry_on_path else {}
    justified = {}
    if justifications:
        for r in read_jsonl(justifications):
            justified[r.get("key") or r.get("claim_key")] = r.get("justification")

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
        "off": str(off_path), "on": str(on_path),
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
    report["cost"] = _replay_cost(on)
    report["repair_call_spread"] = _repair_spread(on)
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
    """The repair pass's OWN run-to-run spread, when a second sample was taken.

    `verify._complete` sends no temperature and no seed, so the repair call is
    an unpinned sample. This is the spread that actually applies to this A/B —
    the answers are fixed and the judge pass is shared, so the only sampling
    left in the comparison is the repair call itself.
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


def _breaches(rep: dict) -> List[str]:
    out = []
    if rep["invented_docs"]:
        out.append(f"{len(rep['invented_docs'])} invented document citation(s)")
    if rep["invented_pages"]:
        out.append(f"{len(rep['invented_pages'])} invented page citation(s)")
    if rep["sources_not_shown_to_repair"]:
        out.append(f"{len(rep['sources_not_shown_to_repair'])} citation(s) on a "
                   "document the repair pass was never shown")
    if rep["invented_figures_matcher_cannot_place"]:
        out.append(f"{len(rep['invented_figures_matcher_cannot_place'])} "
                   "invented figure(s)")
    if rep["lost_supported_undocumented"]:
        out.append(f"{rep['lost_supported_undocumented']} supported claim(s) lost "
                   "with no documented necessity")
    if rep["answer_checks_pass_to_fail"]:
        out.append(f"{rep['answer_checks_pass_to_fail']} answer(s) pass -> fail")
    if rep["language_regressions"]:
        out.append(f"{len(rep['language_regressions'])} language regression(s)")
    if rep["supported_after"] < rep["supported_before"]:
        out.append(f"supported fell {rep['supported_before']} -> "
                   f"{rep['supported_after']}")
    return out


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
    sp = rep.get("repair_call_spread") or {}
    if sp.get("cases_sampled_twice"):
        print(f"REPAIR-CALL SPREAD (the only sampling left in this A/B)")
        print(f"  cases sampled twice {sp['cases_sampled_twice']}   identical text "
              f"{sp['identical_repair_text']}   adoption flipped "
              f"{sp['adoption_flipped']}")
        for f in sp.get("flips") or []:
            print(f"    {f}")
        print()
    print("COST / LATENCY of the replayed calls")
    for role, c in sorted((rep.get("cost") or {}).items()):
        print(f"  {role:<14} {c['calls']:>3} calls  {c['prompt_tokens']:>7}p + "
              f"{c['completion_tokens']:>6}c  p50 {c['latency_p50_s']}s  "
              f"p95 {c['latency_p95_s']}s  total {c['latency_total_s']}s  "
              f"${c['cost_usd']:.4f}")
    print()
    if rep["breaches"]:
        print("GATE: FAIL")
        for b in rep["breaches"]:
            print(f"  - {b}")
    else:
        print("GATE: no breach of the audit's own checks")


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
                    help="repair calls per failing case; >1 measures the "
                         "repair pass's own run-to-run spread")
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
                    help="the carry-ON arm, reported beside the gated carry-OFF one")
    ap.add_argument("--justifications", type=Path,
                    help="jsonl of {key, justification} documenting a "
                         "necessary supported-claim loss")
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
                justifications=args.justifications)
    if not args.quiet:
        print_report(rep, verbose=args.verbose)
        print()
    print(json.dumps({k: rep[k] for k in (
        "cases_compared", "answers_rewritten", "claims_before", "claims_after",
        "supported_before", "supported_after", "groundedness_after",
        "citation_completeness_after", "invented_docs", "invented_pages",
        "invented_figures", "invented_figures_matcher_cannot_place",
        "invented_entities", "sources_not_shown_to_repair",
        "lost_claims", "lost_supported_claims", "lost_supported_undocumented",
        "answer_checks_pass_to_fail", "corrected", "deleted", "qualified",
        "breaches", "gate_pass")}, indent=1, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(rep, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
    return 0 if rep["gate_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
