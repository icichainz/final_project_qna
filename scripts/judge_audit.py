#!/usr/bin/env python3
"""What the LLM judge (``verify.adjudicate``) changed in a recorded run.

Production runs ``VERIFY=1, VERIFY_REPAIR=0`` with ``VERIFY_LLM`` unset, which
``config.py`` defaults to 1 — so the judge adjudicates every live turn. It is
structurally PROMOTE-ONLY: ``adjudicate`` is handed only the claims the
deterministic pass left UNSUPPORTED-and-plausible, so the supported count can
only rise. This script measures what that rise is made of, from an already
recorded release run, with NO API CALL of any kind.

METHOD — why the reconstruction is exact, not an estimate
---------------------------------------------------------
A release record (post-F7) carries the answer the verifier saw
(``raw_answer``), the hit TEXT, the note blocks, the recorded claims block and
the COMPLETE failure list. ``verify.classify_deterministic`` is pure python
over exactly those inputs, so re-running it here reproduces the pre-judge
verdicts bit for bit. Then:

    promotions = {deterministic non-SUPPORTED} - {production non-SUPPORTED}

Everything the judge accepted is in that set difference, and nothing else is:
a claim the judge left alone, or moved to CONTRADICTED, is still a recorded
failure. Four fidelity gates make the reconstruction refuse to guess rather
than report a wrong table (see ``FIDELITY GATES`` below); a case that trips one
has every promotion reported UNDECIDABLE with the gate named.

CLASSIFICATION
--------------
Each promotion is re-checked against the union of the turn's whole evidence
with the harness's own groundedness definition (``eval_answers.grounded_flags``
— verify's matcher, unmodified, over every held key rather than the cited
scope):

  RESCUE            grounded — the fact IS in the held evidence and the
                    deterministic pass missed it in the cited scope. This is
                    the judge's legitimate purpose.
  FABRICATION-PASS  not grounded — the deterministic matcher cannot confirm
                    the claim ANYWHERE in what the turn held.
  UNDECIDABLE       the reconstruction is not trustworthy for this case (a
                    fidelity gate), or the claim carries nothing the matcher
                    can check.

FABRICATION-PASS is a machine verdict about the MATCHER, not a finding of
fabrication: the matcher is literal (substring over normalised text, per
element), so a cross-lingual restatement, a coined acronym, a possessive, or a
phrase the entity extractor glued together all land there too. Every row
therefore carries ``elements_missing`` — the exact sub-strings the matcher
could not find — so the residue can be read by a human. ``--adjudications``
takes those human labels back in and prints them beside the machine verdict;
neither column overwrites the other.

FIDELITY GATES
--------------
1. ``verify.py`` sha256 == the record's ``verify_blob_sha``;
2. the record keeps its whole failure list (``n_failures == len(failures)``);
3. the replayed claim count and evidence keys == the recorded ones;
4. every recorded production failure matches a replayed deterministic failure,
   and the leftover count == the recorded ``judge_promotions``.

USAGE
    venv/bin/python scripts/judge_audit.py data/eval/release_<label>.jsonl
    venv/bin/python scripts/judge_audit.py <run> --adjudications <file.jsonl>
    venv/bin/python scripts/judge_audit.py <run> --jsonl out.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Pinned BEFORE the package is imported, and pinned explicitly: this audit is
# pure replay, and what it reports must not depend on the operator's .env. The
# index is never needed here (the record carries its own hit text), so PRELOAD
# stays off. No verifier switch is read from the environment anywhere below —
# the judge's effect is reconstructed from the RECORD, not re-executed.
os.environ["PRELOAD"] = "0"
os.environ.setdefault("INDEX_NAME", "default")

import eval_answers as ev                                    # noqa: E402
from gcf_qna.rag import verify                                # noqa: E402
from gcf_qna.rag.retrieve import Hit                          # noqa: E402

VERIFY_PY = ROOT / "src" / "gcf_qna" / "rag" / "verify.py"

#: The note blocks production hands the verifier, in production's order.
#: `run_case` passes [registry, year, matrix] — a `board` note is recorded in
#: `notes_used` but never reaches build_evidence, so reading notes_used.values()
#: would build an evidence set the turn did not hold.
NOTE_ORDER = ("registry", "year", "matrix")

RESCUE = "RESCUE"
FABRICATION = "FABRICATION-PASS"
UNDECIDABLE = "UNDECIDABLE"


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
def claim_key(case_id: str, text: str) -> str:
    """Stable id for one claim of one case, for adjudication files."""
    h = hashlib.sha1(f"{case_id}\x00{text}".encode("utf-8")).hexdigest()[:12]
    return f"{case_id}:{h}"


def evidence_for(record: dict):
    """The Evidence dict the recorded turn actually held."""
    hits = [Hit(text=h.get("text") or "", doc_id=h["doc"],
                score=float(h.get("score") or 0.0), page=h.get("page") or None)
            for h in (record.get("hits") or [])]
    notes = record.get("notes_used") or {}
    return verify.build_evidence(hits, [notes.get(k) for k in NOTE_ORDER])


def _evidence_keys(evidence) -> list:
    return [f"{d}|{p if p is not None else '-'}" for d, p in evidence]


def _missing_elements(claim, blob: str) -> list:
    """The sub-strings verify's matcher could not find in `blob`.

    Reported for every promotion, because 'the matcher says no' and 'the
    evidence does not say it' are different statements and the difference
    lives entirely in this list.
    """
    out = []
    if claim.amounts:
        out += [a.raw for a in verify._check_amounts(claim, blob)[1]]
    if claim.entities:
        out += [vs[0] for vs in verify._check_entities(claim, blob)[1]]
    if claim.kind in ("year", "existence"):
        out += list(verify._check_years(claim, blob)[1])
    return list(dict.fromkeys(out))


def _checkable(claim) -> bool:
    """Does the claim carry anything verify's matcher can test at all?"""
    return bool(claim.amounts or claim.entities
                or claim.kind in ("year", "existence"))


def replay_case(record: dict) -> dict:
    """Deterministic pre-judge verdicts and the promotions, for one record.

    Returns ``{"skipped": reason}`` for a turn production never verified
    (guard answers and chat-mode turns build no evidence), or a dict with the
    per-claim rows and any fidelity gate the case tripped.
    """
    rec_claims = record.get("claims")
    if not rec_claims:
        return {"case": record.get("id"), "skipped":
                (record.get("claims_skipped") or {}).get("reason", "no claims block")}

    gates = []
    if rec_claims.get("n_failures") != len(rec_claims.get("failures") or []):
        gates.append("truncated failure list (run not recorded with --release)")
    if not (record.get("hits") and all("text" in h for h in record["hits"])):
        gates.append("record carries no hit text (pre-F7 run)")

    evidence = evidence_for(record)
    claims = verify.extract_claims(record.get("raw_answer") or "")
    det = verify.classify_deterministic(claims, evidence)
    grounded = ev.grounded_flags(claims, evidence)
    blob = verify._text_of(evidence, list(evidence))

    if len(claims) != rec_claims.get("claims"):
        gates.append(f"claim count {len(claims)} != recorded {rec_claims.get('claims')}")
    if sorted(_evidence_keys(evidence)) != sorted(rec_claims.get("evidence_keys") or []):
        gates.append("replayed evidence keys != recorded evidence keys")

    # production's failing set, matched back onto the replayed verdicts
    prod_failures = list(rec_claims.get("failures") or [])
    taken = [False] * len(prod_failures)
    rows = []
    for i, (v, g) in enumerate(zip(det, grounded)):
        prod_status, source, prod_reason = v.status, "deterministic", v.reason
        if v.status != verify.SUPPORTED:
            want = (v.claim.kind, v.claim.text[:160])
            found = None
            for j, pf in enumerate(prod_failures):
                if not taken[j] and (pf.get("kind"), pf.get("text")) == want:
                    found = j
                    break
            if found is None:                       # the judge accepted it
                prod_status, source, prod_reason = verify.SUPPORTED, "llm-promoted", ""
            else:
                taken[found] = True
                prod_status = prod_failures[found].get("status")
                source = prod_failures[found].get("source") or "deterministic"
                prod_reason = prod_failures[found].get("reason") or ""
        rows.append({
            "case": record.get("id"), "class": record.get("class"),
            "lang": record.get("lang"), "claim_index": i,
            "key": claim_key(record.get("id") or "", v.claim.text),
            "kind": v.claim.kind, "cited": bool(v.claim.citations),
            "citations": [f"{c.doc}|{c.page if c.page is not None else '-'}"
                          for c in v.claim.citations],
            "text": v.claim.text,
            "deterministic_status": v.status, "deterministic_reason": v.reason,
            "production_status": prod_status, "verdict_source": source,
            "production_reason": prod_reason,
            "grounded": bool(g), "flags": list(v.flags),
            "elements_missing": _missing_elements(v.claim, blob),
            "checkable": _checkable(v.claim),
            "grounded_keys": [k for k, t in evidence.items()
                              if verify._verify_against(v.claim, t)[0]],
        })

    promotions = [r for r in rows if r["verdict_source"] == "llm-promoted"]
    if any(not t for t in taken):
        gates.append("a recorded production failure has no replayed counterpart")
    if len(promotions) != rec_claims.get("judge_promotions"):
        gates.append(f"reconstructed promotions {len(promotions)} != recorded "
                     f"{rec_claims.get('judge_promotions')}")

    for r in rows:
        r["gates"] = list(gates)
    return {"case": record.get("id"), "gates": gates, "rows": rows,
            "recorded": rec_claims, "n_evidence_keys": len(evidence)}


def classify_promotion(row: dict) -> tuple:
    """(verdict, why) for one promoted claim."""
    if row["gates"]:
        return UNDECIDABLE, "fidelity gate: " + "; ".join(row["gates"])
    if row["grounded"]:
        return RESCUE, "held evidence entails the claim; the cited scope did not"
    if not row["checkable"]:
        return UNDECIDABLE, "the claim carries no figure, name or year to match"
    return FABRICATION, ("not matched anywhere in the held evidence; missing: "
                         + ", ".join(repr(m) for m in row["elements_missing"][:6]))


# ---------------------------------------------------------------------------
# whole-run audit
# ---------------------------------------------------------------------------
def load_run(path: Path) -> list:
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def load_adjudications(path) -> dict:
    """{claim key: {"verdict": ..., "note": ...}} from a jsonl of human labels."""
    out = {}
    if not path:
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row.get("key")
        if key:
            out[key] = {"verdict": row.get("verdict"), "note": row.get("note", "")}
    return out


def audit(records: list, adjudications: dict = None) -> dict:
    """Replay a whole run: promotion rows, per-claim rows, and the totals."""
    adjudications = adjudications or {}
    verify_sha = hashlib.sha256(VERIFY_PY.read_bytes()).hexdigest()
    recorded_shas = {r.get("verify_blob_sha") for r in records if r.get("verify_blob_sha")}
    stale = [s for s in recorded_shas if s and not verify_sha.startswith(s)]

    rows, skipped, cases = [], [], []
    for rec in records:
        got = replay_case(rec)
        if got.get("skipped"):
            skipped.append((got["case"], got["skipped"]))
            continue
        if stale:                       # verify.py moved: nothing below is exact
            for r in got["rows"]:
                r["gates"] = list(r["gates"]) + [
                    "verify.py no longer matches the recorded verify_blob_sha"]
        cases.append(got)
        rows += got["rows"]

    promotions = []
    for r in rows:
        if r["verdict_source"] != "llm-promoted":
            continue
        verdict, why = classify_promotion(r)
        human = adjudications.get(r["key"]) or {}
        promotions.append(dict(r, verdict=verdict, why=why,
                               adjudicated=human.get("verdict"),
                               adjudication_note=human.get("note", "")))

    S = verify.SUPPORTED
    totals = {
        "cases_in_file": len(records),
        "cases_verified": len(cases),
        "cases_skipped": len(skipped),
        "claims": len(rows),
        "supported_judge_on": sum(1 for r in rows if r["production_status"] == S),
        "supported_judge_off": sum(1 for r in rows if r["deterministic_status"] == S),
        "grounded": sum(1 for r in rows if r["grounded"]),
        "cited": sum(1 for r in rows if r["cited"]),
        "citation_supported_on": sum(1 for r in rows
                                     if r["production_status"] == S and r["cited"]),
        "citation_supported_off": sum(1 for r in rows
                                      if r["deterministic_status"] == S and r["cited"]),
        "promotions": len(promotions),
        "promotion_cases": len({p["case"] for p in promotions}),
        "judge_rejected": sum(1 for r in rows if r["verdict_source"] == "llm"),
        "verify_sha_matches": not stale,
    }
    return {"rows": rows, "promotions": promotions, "cases": cases,
            "skipped": skipped, "totals": totals, "verify_sha": verify_sha}


def judge_usage(records: list) -> dict:
    """Per-role calls, tokens, latency and ESTIMATED cost, from the recording.

    Rates are ``eval_answers.TOKEN_COST_USD`` — imported, never restated, so
    the two reports cannot drift apart.
    """
    by_role = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0,
                                   "latencies": []})
    turn_latency = []
    for rec in records:
        usage = rec.get("usage") or {}
        for call in usage.get("calls") or []:
            d = by_role[call.get("role") or "answer"]
            d["calls"] += 1
            d["prompt"] += int(call.get("prompt_tokens") or 0)
            d["completion"] += int(call.get("completion_tokens") or 0)
            d["latencies"].append(float(call.get("latency_s") or 0.0))
        if usage.get("turn_latency_s") is not None:
            turn_latency.append(float(usage["turn_latency_s"]))
    out = {}
    for role, d in by_role.items():
        lat = d["latencies"]
        out[role] = {
            "calls": d["calls"], "prompt_tokens": d["prompt"],
            "completion_tokens": d["completion"],
            "latency_s": round(sum(lat), 1),
            "latency_mean_s": round(statistics.mean(lat), 3) if lat else None,
            "latency_p50_s": round(statistics.median(lat), 3) if lat else None,
            "latency_max_s": round(max(lat), 3) if lat else None,
            "cost_usd": round(d["prompt"] * ev.TOKEN_COST_USD["prompt"]
                              + d["completion"] * ev.TOKEN_COST_USD["completion"], 4),
        }
    out["_turn_latency_sum_s"] = round(sum(turn_latency), 1)
    return out


def counterfactual(result: dict, records: list) -> dict:
    """The run's headline numbers with the judge off, and the status flips.

    Nothing is re-called: production's per-claim verdicts and the pre-judge
    deterministic ones both come out of the same replay, and `_status_for` is
    pure python over the verdict list. VERIFY_REPAIR=0 in production, so the
    answer TEXT is identical either way — only the caption changes.
    """
    by_case = defaultdict(list)
    for r in result["rows"]:
        by_case[r["case"]].append(r)
    recorded = {r.get("id"): r for r in records}
    S = verify.SUPPORTED
    flips = []
    for case, rows in by_case.items():
        rec = (recorded.get(case) or {}).get("claims") or {}
        on = rec.get("verify_status")
        required = [r for r in rows if r["kind"] in ("money", "number", "entity")]
        failed_off = [r for r in rows if r["deterministic_status"] != S]
        if not failed_off:
            off = "verified"
        elif required and all(r["deterministic_status"] != S for r in required):
            off = "abstain"
        else:
            off = "partial"
        if on != off:
            flips.append({"case": case, "class": rows[0]["class"],
                          "status_on": on, "status_off": off,
                          "supported_on": sum(1 for r in rows
                                              if r["production_status"] == S),
                          "supported_off": sum(1 for r in rows
                                               if r["deterministic_status"] == S),
                          "claims": len(rows)})
    return {"flips": flips,
            "status_on": Counter((recorded[c].get("claims") or {}).get("verify_status")
                                 for c in by_case),
            "flip_kinds": Counter(f"{f['status_on']} -> {f['status_off']}"
                                  for f in flips)}


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def _bar(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def report(result: dict, records: list, width: int = 96) -> None:
    t = result["totals"]
    print(f"judge audit — {t['cases_in_file']} cases, {t['cases_verified']} verified "
          f"by production, {t['cases_skipped']} skipped, {t['claims']} claims")
    print(f"verify.py sha256 matches the recording: {t['verify_sha_matches']}")
    gated = sorted(c["case"] for c in result["cases"] if c["gates"])
    print(f"cases tripping a fidelity gate: {len(gated)}"
          + (f" -> {', '.join(gated[:6])}"
             + (f" (+{len(gated) - 6} more)" if len(gated) > 6 else "")
             if gated else ""))
    if gated:
        reasons = Counter(g for c in result["cases"] for g in c["gates"])
        for why, n in reasons.most_common():
            print(f"  {n:>3} x {why}")
    if gated and len(gated) == len(result["cases"]):
        print("  -> NO promotion can be reconstructed from this file; "
              "the table below is empty by refusal, not by finding.")
    for case, why in result["skipped"]:
        print(f"  skipped {case}: {why}")

    _bar("PROMOTION CLASSIFICATION")
    counts = Counter(p["verdict"] for p in result["promotions"])
    print(f"{t['promotions']} promotions in {t['promotion_cases']} cases  "
          f"({counts[RESCUE]} {RESCUE}, {counts[FABRICATION]} {FABRICATION}, "
          f"{counts[UNDECIDABLE]} {UNDECIDABLE})")
    adj = Counter(p["adjudicated"] for p in result["promotions"] if p["adjudicated"])
    if adj:
        print("human adjudication on file: "
              + ", ".join(f"{v} {k}" for k, v in sorted(adj.items())))
    print()
    head = f"{'case':<26}{'kind':<9}{'cite':<5}{'verdict':<19}claim"
    print(head)
    print("-" * min(width, len(head) + 40))
    for p in sorted(result["promotions"], key=lambda r: (r["class"], r["case"])):
        text = " ".join(p["text"].split())[:width - 60]
        mark = "y" if p["cited"] else "n"
        v = p["verdict"] if not p["adjudicated"] else f"{p['verdict']}*"
        print(f"{p['case']:<26}{p['kind']:<9}{mark:<5}{v:<19}{text}")
        if p["verdict"] != RESCUE:
            print(f"{'':<26}{'':<9}{'':<5}{'':<19}-> {p['why'][:width - 64]}")
            if p["adjudication_note"]:
                print(f"{'':<26}{'':<9}{'':<5}{'':<19}"
                      f"** {p['adjudicated']}: {p['adjudication_note'][:width - 68]}")
    if adj:
        print("\n* a human adjudication is on file for this row (** line)")

    _bar("PROMOTIONS BY CASE CLASS")
    allc = Counter(r["class"] for r in result["rows"])
    pc = Counter(p["class"] for p in result["promotions"])
    rc = Counter(p["class"] for p in result["promotions"] if p["verdict"] == RESCUE)
    fc = Counter(p["class"] for p in result["promotions"] if p["verdict"] == FABRICATION)
    print(f"{'class':<14}{'claims':>7}{'promoted':>10}{'RESCUE':>8}{'FAB-PASS':>10}")
    for cls in sorted(allc, key=lambda k: (-pc[k], k)):
        print(f"{cls:<14}{allc[cls]:>7}{pc[cls]:>10}{rc[cls]:>8}{fc[cls]:>10}")

    _bar("PROMOTIONS BY CLAIM KIND")
    allk = Counter(r["kind"] for r in result["rows"])
    pk = Counter(p["kind"] for p in result["promotions"])
    rk = Counter(p["kind"] for p in result["promotions"] if p["verdict"] == RESCUE)
    fk = Counter(p["kind"] for p in result["promotions"] if p["verdict"] == FABRICATION)
    print(f"{'kind':<14}{'claims':>7}{'promoted':>10}{'RESCUE':>8}{'FAB-PASS':>10}")
    for kind in sorted(allk, key=lambda k: (-pk[k], k)):
        print(f"{kind:<14}{allk[kind]:>7}{pk[kind]:>10}{rk[kind]:>8}{fk[kind]:>10}")
    cited = Counter(p["cited"] for p in result["promotions"])
    print(f"\ncited claims promoted: {cited[True]}   uncited: {cited[False]}")

    _bar("SUPPORTED BUT NOT GROUNDED  (what makes supported > grounded)")
    S = verify.SUPPORTED
    bad = [r for r in result["rows"]
           if r["production_status"] == S and not r["grounded"]]
    good = [r for r in result["rows"]
            if r["production_status"] != S and r["grounded"]]
    print(f"supported-and-not-grounded: {len(bad)}  "
          f"({Counter(r['verdict_source'] for r in bad)})")
    print(f"grounded-and-not-supported: {len(good)}  "
          f"({Counter(r['verdict_source'] for r in good)})")
    print(f"net supported - grounded  : "
          f"{result['totals']['supported_judge_on'] - result['totals']['grounded']:+d}")
    for r in bad:
        print(f"  [{r['case']}] {r['verdict_source']:<13} {r['kind']:<8} "
              f"missing {r['elements_missing'][:3]}")

    _bar("JUDGE COST AND LATENCY  (recorded; $ ESTIMATED)")
    usage = judge_usage(records)
    total_cost = sum(v["cost_usd"] for k, v in usage.items() if not k.startswith("_"))
    total_tok = sum(v["prompt_tokens"] + v["completion_tokens"]
                    for k, v in usage.items() if not k.startswith("_"))
    print(f"{'role':<12}{'calls':>6}{'prompt':>10}{'compl':>8}{'lat_s':>9}{'cost_usd':>10}")
    for role in sorted(k for k in usage if not k.startswith("_")):
        u = usage[role]
        print(f"{role:<12}{u['calls']:>6}{u['prompt_tokens']:>10}"
              f"{u['completion_tokens']:>8}{u['latency_s']:>9.1f}{u['cost_usd']:>10.4f}")
    j = usage.get("judge")
    if j:
        print(f"\njudge: {j['calls']} calls (one batched call per turn with a "
              f"plausible residue)")
        print(f"       latency mean {j['latency_mean_s']}s  p50 {j['latency_p50_s']}s  "
              f"max {j['latency_max_s']}s  total {j['latency_s']}s")
        print(f"       {100 * j['cost_usd'] / total_cost:.1f}% of run cost, "
              f"{100 * (j['prompt_tokens'] + j['completion_tokens']) / total_tok:.1f}% "
              f"of run tokens, "
              f"{100 * j['latency_s'] / usage['_turn_latency_sum_s']:.1f}% of "
              f"summed turn latency")
        print(f"       rates: ${ev.TOKEN_COST_USD['prompt'] * 1e6:.2f}/1M prompt, "
              f"${ev.TOKEN_COST_USD['completion'] * 1e6:.2f}/1M completion (ESTIMATE)")

    _bar("COUNTERFACTUAL: THE SAME RUN WITH VERIFY_LLM=0")
    d = t["claims"]
    def _row(name, on, off):
        print(f"  {name:<24} on {on:>4}/{d} ({on / d:6.1%})   "
              f"off {off:>4}/{d} ({off / d:6.1%})   {off - on:+d}")
    _row("support", t["supported_judge_on"], t["supported_judge_off"])
    _row("groundedness", t["grounded"], t["grounded"])
    _row("citation completeness", t["citation_supported_on"], t["citation_supported_off"])
    _row("citation presence", t["cited"], t["cited"])
    cf = counterfactual(result, records)
    print(f"\nanswers changing user-visible status: {len(cf['flips'])} "
          f"({dict(cf['flip_kinds'])})")
    for f in cf["flips"]:
        print(f"  {f['case']:<26}{f['class']:<12}{f['status_on']:>9} -> "
              f"{f['status_off']:<9} supported {f['supported_on']}->{f['supported_off']}"
              f" of {f['claims']}")
    print("\nVERIFY_REPAIR=0, so the answer TEXT is byte-identical either way: "
          "only the verification caption changes.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("record", type=Path, help="data/eval/release_<label>.jsonl")
    ap.add_argument("--adjudications", type=Path, default=None,
                    help="jsonl of human labels: {key, verdict, note}")
    ap.add_argument("--jsonl", type=Path, default=None,
                    help="write one row per promotion to this file")
    ap.add_argument("--all-claims", action="store_true",
                    help="--jsonl writes every claim, not only the promotions")
    ap.add_argument("--quiet", action="store_true", help="write files, print nothing")
    args = ap.parse_args(argv)

    records = load_run(args.record)
    result = audit(records, load_adjudications(args.adjudications))
    if not args.quiet:
        report(result, records)
    if args.jsonl:
        out = result["rows"] if args.all_claims else result["promotions"]
        args.jsonl.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
            encoding="utf-8")
        if not args.quiet:
            print(f"\nwrote {len(out)} rows -> {args.jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
