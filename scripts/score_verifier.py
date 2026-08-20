#!/usr/bin/env python3
"""Score the claim verifier against the adjudicated gold AND against seeded
claims the gold structurally cannot contain.

    venv/bin/python scripts/score_verifier.py                     # the report
    venv/bin/python scripts/score_verifier.py --json before.json  # record it
    venv/bin/python scripts/score_verifier.py --baseline before.json   # diff it
    venv/bin/python scripts/score_verifier.py --arm fabricated --verbose

WHY THE GOLD ALONE CANNOT SCORE A RELAXATION
--------------------------------------------
``data/eval/release_release-1-adjudicated.jsonl`` holds 71 claims — every one
of them a claim the recorded ``release-1`` run FLAGGED.  Nothing the verifier
passed is in it.  So a change that only ever *stops* flagging things cannot
produce a false negative there, and recall over that file is pinned at 100%
by construction: it measures the instrument, not the code.  The first review
of Wave 2 found eight fabricated figures, entities and predicates promoted to
SUPPORTED while this file still read 12/12/0/47.

The scorer therefore carries three arms, and reports them separately as well
as combined:

``gold``          the 71 adjudicated rows.  ``verifier_correct`` is the
                  authority: 12 say the flag was right, 59 say it was wrong.
``held-correct``  every claim of the 66 recorded answers that the release did
                  NOT record as a failure.  These are NOT adjudicated — the
                  property they pin is narrower and is stated as such: *no
                  calibration change may start flagging a claim the recorded
                  release passed*.  Each is `should_flag = False`.
``fabricated``    claims built from the recorded evidence by a mutation that
                  makes them FALSE — a figure no held key prints, a name no
                  held key contains, a citation moved to a document that does
                  not entail it — plus the adversarial shapes that survived
                  the first review.  Each is `should_flag = True`, and each is
                  self-certifying: the mutation is accepted only after the
                  verifier's own matchers confirm the fabricated value is
                  absent from every held key of that turn.

    TP  flagged, and it should be        FP  flagged, and it should not be
    FN  NOT flagged, and it should be    TN  NOT flagged, and it should not be
    precision = TP / (TP + FP)           recall = TP / (TP + FN)

Recall now has something to lose: every promotion of a fabricated claim to
SUPPORTED is an FN, named individually by ``--baseline``.

HOW A CLAIM IS REPLAYED
-----------------------
Deterministically and offline.  ``release_release-1-evidence.jsonl`` carries
the text of every evidence key each turn held; this rebuilds
``{(doc, page): text}`` from it verbatim, re-runs ``verify.extract_claims``
over the answer, and calls ``verify.classify_deterministic``.  The judge is
never called: it can only ever be asked about claims the deterministic layer
left unsupported-and-plausible, so a deterministic verdict is the thing a
matcher change actually moves — and a claim promoted to SUPPORTED never
reaches the judge at all, which is precisely why promotions must be counted.

No file under data/ is written or modified.  The fabricated arm is generated
in memory, from the read-only recordings, by rules written out below.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcf_qna.rag import verify  # noqa: E402

DEFAULT_GOLD = ROOT / "data" / "eval" / "release_release-1-adjudicated.jsonl"
DEFAULT_EVIDENCE = ROOT / "data" / "eval" / "release_release-1-evidence.jsonl"
DEFAULT_RELEASE = ROOT / "data" / "eval" / "release_release-1.jsonl"

ARMS = ("gold", "held-correct", "fabricated")

#: A name no GCF document contains, used to fabricate entity claims.
FAKE_ENTITY = "Wakanda Development Bank"
#: An expansion whose initials fit a real corpus acronym but which no
#: registry row records — the shape that survived the first review.
FAKE_EXPANSIONS = {"IFAD": "International Federation of Arctic Drillers",
                   "IUCN": "Interstate Union of Coastal Navigators",
                   "EBRD": "Eastern Bureau of Regional Drilling"}


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n}: {e}") from None
    return rows


def evidence_of(case: dict) -> Dict[Tuple[str, Optional[int]], str]:
    """``{(doc, page): text}`` exactly as ``build_evidence`` produced it."""
    out: Dict[Tuple[str, Optional[int]], str] = {}
    for entry in case.get("evidence") or ():
        page = entry.get("page")
        out[(entry.get("doc"), int(page) if page else None)] = entry.get("text") or ""
    return out


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

class Replay:
    """One answer, re-verified deterministically against one evidence set."""

    def __init__(self, answer: str, evidence: Dict[Tuple[str, Optional[int]], str]):
        self.answer = answer
        self.evidence = evidence
        self.claims = verify.extract_claims(answer)
        self.verdicts = verify.classify_deterministic(self.claims, evidence)
        self.by_text: Dict[str, List[int]] = defaultdict(list)
        for c in self.claims:
            self.by_text[c.text].append(c.index)

    def verdict_for(self, claim_text: str):
        idx = self.by_text.get(claim_text) or []
        if len(idx) != 1:
            return None, len(idx)
        return self.verdicts[idx[0]], 1

    def status(self, use_llm: bool = False, judge: str = "keep") -> str:
        """The RepairResult status this answer carries, with repair OFF.

        ``judge`` bounds the optional adjudication without calling it: 'keep'
        is the judge declining to move anything, 'clear' is it supporting
        every claim it is shown, 'reject' is it contradicting them. Production
        runs VERIFY_LLM=1, so the honest statement about a matcher change is
        that the status holds under ALL THREE bounds, which needs no key.
        """
        verdicts = list(self.verdicts)
        if use_llm and judge != "keep":
            sent = {id(v) for v in verdicts
                    if v.status == verify.UNSUPPORTED and v.plausible}
            new = verify.SUPPORTED if judge == "clear" else verify.CONTRADICTED
            verdicts = [verify.Verdict(v.claim, new, "judge bound", v.scope,
                                       source="llm", flags=list(v.flags))
                        if id(v) in sent else v for v in verdicts]
        return verify._status_for(verdicts, use_llm, False)

    def cautions(self) -> List[str]:
        """The user-visible caution flags on SUPPORTED claims, deduped."""
        return sorted({f for v in self.verdicts
                       if v.status == verify.SUPPORTED for f in v.flags})


# ---------------------------------------------------------------------------
# the fabricated arm
# ---------------------------------------------------------------------------

#: A claim that asserts an absence, detected from its TEXT so the seed set
#: never depends on the verifier under test.
_ABSENCE_SHAPE = re.compile(
    r"\b(?:not found|does not exist|is not in|no such|none of|aucun|"
    r"n['\u2019]existe pas|ne mentionnent pas|not in this corpus)\b", re.I)


def _digits(text: str) -> str:
    return re.sub(r"[^0-9]", "", text or "")


def _digit_runs(text: str) -> List[str]:
    return re.findall(r"\d[\d.,   ]*\d|\d", text or "")


def _absent_amount(token: str, ev_digits: str) -> Optional[str]:
    """A same-shaped figure whose digits appear NOWHERE in the held evidence.

    Digits are rotated rather than incremented, so the token keeps its
    separators and its magnitude and stays the kind of figure the answer
    model really prints. Validity is decided on the digit run alone — not by
    the verifier's own matcher — because a seed set scored by the code under
    test is not a test. Digit-run absence is strictly stronger than any
    matcher: a figure whose digits are not in the evidence cannot be matched
    by any reading of the separators.
    """
    fake = "".join(str((int(ch) + 3) % 10) if ch.isdigit() else ch for ch in token)
    fake = re.sub(r"^0", "1", fake)
    if fake == token or not _digits(fake):
        return None
    return None if _digits(fake) in ev_digits else fake


_MARKED_RE = re.compile(r"\*\*(.+?)\*\*|[\u201c\"\u00ab]\s*([^\u201d\"\u00bb\n]{3,120}?)\s*[\u201d\"\u00bb]")


def _marked_names(text: str) -> List[str]:
    """Bold or quoted spans that read as a name, from the text alone.

    Deliberately not ``verify.entities`` — the seed set may not be built by
    the code it scores, or a tree that extracts fewer names would be handed a
    smaller and easier test than the tree it is compared against.
    """
    out = []
    for m in _MARKED_RE.finditer(verify._strip_citations(text)):
        span = (m.group(1) or m.group(2) or "").strip(" .,;:*_")
        words = [w for w in span.split() if w]
        if len(words) >= 2 and sum(1 for w in words if w[:1].isupper()) >= 2 \
                and not re.search(r"\d", span):
            out.append(span)
    return out


def _absent_name(name: str, ev_norm: str) -> bool:
    return bool(verify.norm_text(name)) and verify.norm_text(name) not in ev_norm


def _one_claim(answer: str, evidence, needle: str):
    """The single extracted claim of ``answer`` containing ``needle``."""
    rep = Replay(answer, evidence)
    hits = [c for c in rep.claims if needle in c.text]
    if len(hits) != 1:
        return None, None
    return rep, hits[0]


def fabricate(cases: List[dict], basis: Dict[str, List[str]]) -> List[dict]:
    """Claims that are FALSE about the evidence their turn actually held.

    ``basis`` is ``{case_id: [claim texts the RECORDED RELEASE passed]}`` — a
    fact written down in ``release_release-1.jsonl``, not a verdict from the
    verifier being scored. That matters: an earlier revision seeded from
    whatever the current code happened to support, so a more permissive
    verifier was handed a larger and different test, and the before/after
    recalls were not comparable. The basis is now identical for every tree.

    Every mutant is derived from a recorded answer, so its shape, citation
    style and language are the ones the live system really produces; only the
    asserted value is fabricated, and only after the evidence text itself
    confirms that value is absent. A mutant whose extraction does not yield
    exactly one claim carrying the fabrication is skipped, never guessed at.
    """
    out: List[dict] = []

    def add(case_id, answer, evidence, needle, why):
        _rep, claim = _one_claim(answer, evidence, needle)
        if claim is None:
            return
        # Content-addressed, never positional: a row id that moved when the
        # set gained or lost one member made every later row look like a new
        # regression in a before/after diff.
        tag = hashlib.sha256(
            f"{case_id}|{why}|{claim.text}".encode("utf-8")).hexdigest()[:10]
        out.append({"row_id": f"fab-{why}-{case_id}-{tag}",
                    "arm": "fabricated", "case_id": case_id, "why": why,
                    "answer": answer, "evidence": evidence,
                    "claim_text": claim.text, "should_flag": True})

    for case in sorted(cases, key=lambda c: c["case_id"]):
        cid = case["case_id"]
        answer = case.get("answer") or ""
        ev = evidence_of(case)
        ev_text = "\n".join(ev.values())
        ev_digits = _digits(ev_text)
        ev_norm = verify.norm_text(ev_text)
        docs = sorted({k[0] for k in ev if k[0] != verify.NOTES_DOC})
        wanted = set(basis.get(cid) or ())
        if not wanted:
            continue

        for claim in verify.extract_claims(answer):
            if claim.text[:160] not in wanted:
                continue          # only a claim the RELEASE passed can be broken

            # 1. a figure whose digits no held key prints
            runs = _digit_runs(verify._strip_citations(claim.text))
            for run in runs[:1]:
                fake = _absent_amount(run, ev_digits)
                if fake:
                    add(cid, answer.replace(claim.text,
                                            claim.text.replace(run, fake, 1), 1),
                        ev, fake, "figure")

            # 2. a name no held key contains. The span to overwrite is taken
            #    from the TEXT, not from verify.entities(): the seed set must
            #    be byte-identical in every tree it scores, and entity
            #    extraction is itself under test.
            names = _marked_names(claim.text)
            if names and _absent_name(FAKE_ENTITY, ev_norm):
                add(cid, answer.replace(claim.text,
                                        claim.text.replace(names[0], FAKE_ENTITY, 1), 1),
                    ev, FAKE_ENTITY, "entity")

            # 3. the same assertion moved onto a document that does not carry
            #    it — a citation that verifies cleanly is still an invented
            #    attribution
            cited = [c.doc for c in claim.citations if c.doc]
            others = [d for d in docs if d not in cited]
            if cited and others:
                target = others[0]
                tgt = "\n".join(t for k, t in ev.items() if k[0] == target)
                tgt_digits, tgt_norm = _digits(tgt), verify.norm_text(tgt)
                carries = any(_digits(r) and _digits(r) in tgt_digits for r in runs) or \
                    any(verify.norm_text(n) and verify.norm_text(n) in tgt_norm
                        for n in names)
                if not carries:
                    add(cid, answer.replace(claim.text,
                                            claim.text.replace(cited[0], target), 1),
                        ev, target, "citation")

            # 4. a fabricated expansion behind a real corpus acronym
            for acr, fake_exp in sorted(FAKE_EXPANSIONS.items()):
                if re.search(r"\b%s\b" % acr, claim.text) and _absent_name(fake_exp, ev_norm):
                    add(cid, answer.replace(
                        claim.text,
                        claim.text.replace(acr, f"{acr} ({fake_exp})", 1), 1),
                        ev, fake_exp, "acronym")
                    break

            # 5. a rider bolted onto a true negative: the absence is real, the
            #    thing bolted to it is not
            if _ABSENCE_SHAPE.search(claim.text) and _absent_name(FAKE_ENTITY, ev_norm):
                cite = next((f"[{c.doc}, p. {c.page}]"
                             for k in verify.extract_claims(answer)
                             for c in k.citations if c.doc and c.page), "")
                for sep, why in ((", and ", "rider-and"), ("; ", "rider-semicolon")):
                    add(cid, answer.replace(
                        claim.text,
                        claim.text.rstrip(". ") + sep
                        + f"the accredited entity is **{FAKE_ENTITY}** {cite}.", 1),
                        ev, FAKE_ENTITY, why)
    return out


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def build_rows(gold: List[dict], cases: List[dict], release: List[dict]
               ) -> Tuple[List[dict], List[dict]]:
    """(rows, problems) across the three arms."""
    by_case = {c["case_id"]: c for c in cases}
    rows: List[dict] = []
    problems: List[dict] = []

    for g in sorted(gold, key=lambda g: g["claim_id"]):
        case = by_case.get(g.get("case_id"))
        if case is None:
            problems.append({"row_id": g["claim_id"],
                             "why": f"no evidence row for case {g.get('case_id')!r}"})
            continue
        rows.append({"row_id": g["claim_id"], "arm": "gold",
                     "case_id": g["case_id"], "answer": case.get("answer") or "",
                     "evidence": evidence_of(case),
                     "claim_text": g.get("claim_text_full") or g.get("claim_text") or "",
                     "should_flag": bool(g.get("verifier_correct")),
                     "label": g.get("label"), "owner_ruling": g.get("owner_ruling"),
                     "why": g.get("label")})

    # held-correct: every claim the release did NOT record as a failure.
    failed_texts: Dict[str, set] = {}
    for r in release:
        block = r.get("claims") or {}
        fails = block.get("failures") or []
        recorded = (block.get("contradicted", 0) or 0) + (block.get("unsupported", 0) or 0)
        if recorded != len(fails):
            # the release truncates failures at six; a truncated case cannot
            # tell held-correct from unrecorded-failure, so it is excluded
            problems.append({"row_id": r.get("id"),
                             "why": f"release recorded {recorded} failures but listed "
                                    f"{len(fails)}; held-correct claims not separable"})
            continue
        failed_texts[r["id"]] = {f.get("text") for f in fails if isinstance(f, dict)}

    for case in sorted(cases, key=lambda c: c["case_id"]):
        cid = case["case_id"]
        if cid not in failed_texts:
            continue
        ev = evidence_of(case)
        answer = case.get("answer") or ""
        for claim in verify.extract_claims(answer):
            if claim.text[:160] in failed_texts[cid]:
                continue
            rows.append({"row_id": f"held-{cid}-{claim.index}", "arm": "held-correct",
                         "case_id": cid, "answer": answer, "evidence": ev,
                         "claim_text": claim.text, "should_flag": False,
                         "why": "the recorded release passed this claim"})

    basis = {cid: [r["claim_text"][:160] for r in rows
                   if r["arm"] == "held-correct" and r["case_id"] == cid]
             for cid in failed_texts}
    rows += fabricate(cases, basis)
    return rows, problems


def score(rows: List[dict]) -> dict:
    replays: Dict[Tuple[str, int], Replay] = {}
    scored: List[dict] = []
    unmatched: List[dict] = []
    for row in rows:
        key = (row["case_id"], id(row["answer"]))
        rep = replays.get(key)
        if rep is None:
            rep = replays[key] = Replay(row["answer"], row["evidence"])
        verdict, n = rep.verdict_for(row["claim_text"])
        if verdict is None:
            unmatched.append({"row_id": row["row_id"], "arm": row["arm"],
                              "why": f"claim text matches {n} extracted claims"})
            continue
        scored.append({k: row[k] for k in ("row_id", "arm", "case_id", "should_flag",
                                           "why")} | {
            "label": row.get("label"), "owner_ruling": row.get("owner_ruling"),
            "status": verdict.status, "flagged": bool(verdict.failed),
            "reason": verdict.reason, "flags": list(verdict.flags),
            "plausible": bool(verdict.plausible), "kind": verdict.claim.kind,
            "claim_text": row["claim_text"][:200]})

    def matrix(subset):
        tp = [r for r in subset if r["should_flag"] and r["flagged"]]
        fp = [r for r in subset if not r["should_flag"] and r["flagged"]]
        fn = [r for r in subset if r["should_flag"] and not r["flagged"]]
        tn = [r for r in subset if not r["should_flag"] and not r["flagged"]]
        prec = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
        rec = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"counts": {"tp": len(tp), "fp": len(fp), "fn": len(fn), "tn": len(tn)},
                "precision": prec, "recall": rec, "f1": f1,
                "false_negatives": [r["row_id"] for r in fn],
                "false_positives": [r["row_id"] for r in fp]}

    by_label: Dict[str, Dict[str, int]] = {}
    for r in scored:
        if r["arm"] != "gold":
            continue
        b = by_label.setdefault(r["label"] or "(unlabelled)",
                                {"rows": 0, "flagged": 0, "cleared": 0})
        b["rows"] += 1
        b["flagged" if r["flagged"] else "cleared"] += 1

    return {"rows": scored, "unmatched": unmatched,
            "overall": matrix(scored),
            "arms": {a: matrix([r for r in scored if r["arm"] == a]) for a in ARMS},
            "arm_sizes": dict(Counter(r["arm"] for r in scored)),
            "by_label": by_label,
            "fabrication_kinds": dict(Counter(
                r["why"] for r in scored if r["arm"] == "fabricated"))}


def answer_state(cases: List[dict]) -> Dict[str, dict]:
    """Per recorded answer: status under each judge bound, and its cautions."""
    out = {}
    for case in cases:
        rep = Replay(case.get("answer") or "", evidence_of(case))
        out[case["case_id"]] = {
            "det": rep.status(False), "llm_keep": rep.status(True, "keep"),
            "llm_clear": rep.status(True, "clear"),
            "llm_reject": rep.status(True, "reject"),
            "cautions": rep.cautions()}
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x * 100:5.1f}%"


def _matrix_line(name: str, m: dict, n: int) -> str:
    c = m["counts"]
    return (f"  {name:<14} n={n:<4} TP {c['tp']:3d}  FP {c['fp']:3d}  "
            f"FN {c['fn']:3d}  TN {c['tn']:3d}   "
            f"P {_pct(m['precision'])}  R {_pct(m['recall'])}  F1 {_pct(m['f1'])}")


def report(res: dict, states: dict, baseline: Optional[dict] = None,
           verbose: bool = False, arm: Optional[str] = None) -> None:
    print("=" * 92)
    print("VERIFIER vs ADJUDICATED GOLD + SEEDED CLAIMS  (deterministic replay, no API)")
    print("=" * 92)
    print(f"rows scored {len(res['rows'])}   unmatched {len(res['unmatched'])}")
    for u in res["unmatched"][:12]:
        print(f"  ! UNMATCHED [{u['arm']}] {u['row_id']}: {u['why']}")
    print()
    for a in ARMS:
        print(_matrix_line(a, res["arms"][a], res["arm_sizes"].get(a, 0)))
    print(_matrix_line("OVERALL", res["overall"], len(res["rows"])))
    print()
    print("  fabricated shapes: " + ", ".join(
        f"{k} {v}" for k, v in sorted(res["fabrication_kinds"].items())))
    print()
    print("per adjudication label (gold arm):")
    for label in sorted(res["by_label"]):
        b = res["by_label"][label]
        print(f"  {label:<28} rows {b['rows']:>3}  flagged {b['flagged']:>3}"
              f"  cleared {b['cleared']:>3}")

    print()
    print("live path (VERIFY=1, VERIFY_REPAIR=0, VERIFY_LLM=1), 66 recorded answers")
    for bound in ("det", "llm_keep", "llm_clear", "llm_reject"):
        print(f"  {bound:<10} " + ", ".join(
            f"{k} {v}" for k, v in sorted(Counter(
                s[bound] for s in states.values()).items())))
    caut = Counter(f for s in states.values() for f in s["cautions"])
    print(f"  cautions   {dict(caut)}   "
          f"({sum(1 for s in states.values() if s['cautions'])} answers show one)")

    if verbose:
        for r in sorted(res["rows"], key=lambda r: r["row_id"]):
            if arm and r["arm"] != arm:
                continue
            wrong = r["should_flag"] != r["flagged"]
            if not wrong and not arm:
                continue
            mark = "MISS" if (r["should_flag"] and not r["flagged"]) else (
                "NOISE" if r["flagged"] and not r["should_flag"] else "ok  ")
            print(f"  {mark} [{r['arm']:<12}] {r['row_id']:<34} {r['status']:<13} "
                  f"{(r['why'] or '')[:26]:<26} {r['reason'][:60]}")

    if baseline is None:
        return
    print()
    print("=" * 92)
    print("DELTA vs baseline")
    print("=" * 92)
    for a in list(ARMS) + ["overall"]:
        b = baseline["arms"][a] if a in ARMS else baseline["overall"]
        n = res["arms"][a] if a in ARMS else res["overall"]
        bc, nc = b["counts"], n["counts"]
        print(f"  {a:<14} TP {bc['tp']:3d}->{nc['tp']:<3d} FP {bc['fp']:3d}->{nc['fp']:<3d} "
              f"FN {bc['fn']:3d}->{nc['fn']:<3d} TN {bc['tn']:3d}->{nc['tn']:<3d}  "
              f"P {_pct(b['precision'])}->{_pct(n['precision'])}  "
              f"R {_pct(b['recall'])}->{_pct(n['recall'])}")
    was = {r["row_id"]: r for r in baseline["rows"]}
    now = {r["row_id"]: r for r in res["rows"]}
    fixed = [i for i in now if i in was and was[i]["flagged"] and not now[i]["flagged"]]
    broke = [i for i in now if i in was and not was[i]["flagged"] and now[i]["flagged"]]
    print()
    for title, ids in (("cleared since baseline", fixed),
                       ("newly flagged since baseline", broke)):
        print(f"  {title}: {len(ids)}")
        for i in sorted(ids):
            r = now[i]
            good = (r["should_flag"] == r["flagged"])
            print(f"    {'GOOD' if good else 'REGRESSION':<10} [{r['arm']:<12}] {i:<34}"
                  f" {was[i]['status']} -> {r['status']}")
    old_state = baseline.get("answer_state") or {}
    if old_state:
        print()
        print("  live-path and caution changes on the 66 answers:")
        moved = 0
        for cid in sorted(states):
            before, after = old_state.get(cid) or {}, states[cid]
            diff = {k: (before.get(k), after[k]) for k in after if before.get(k) != after[k]}
            if diff:
                moved += 1
                print(f"    {cid}: " + "; ".join(
                    f"{k} {v0} -> {v1}" for k, (v0, v1) in diff.items()))
        print(f"    ({moved} of {len(states)} answers changed)")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    p.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    p.add_argument("--json", type=Path, help="write the full result here")
    p.add_argument("--baseline", type=Path, help="a --json file to diff against")
    p.add_argument("--arm", choices=ARMS, help="list every row of one arm")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--fail-on-fn", action="store_true",
                   help="exit 1 if any claim that should be flagged is not")
    p.add_argument("--min-fabricated", type=int, default=0,
                   help="exit 1 if fewer than this many fabricated rows were seeded")
    args = p.parse_args(argv)

    rows, problems = build_rows(read_jsonl(args.gold), read_jsonl(args.evidence),
                                read_jsonl(args.release))
    res = score(rows)
    res["unmatched"] += problems
    states = answer_state(read_jsonl(args.evidence))
    res["answer_state"] = states

    baseline = json.loads(args.baseline.read_text()) if args.baseline else None
    report(res, states, baseline, verbose=args.verbose or bool(args.arm), arm=args.arm)

    if args.json:
        dump = dict(res)
        args.json.write_text(json.dumps(dump, indent=1, sort_keys=True, default=str))
        print(f"\nwrote {args.json}")

    if args.min_fabricated and res["arm_sizes"].get("fabricated", 0) < args.min_fabricated:
        print(f"\nFAIL: only {res['arm_sizes'].get('fabricated', 0)} fabricated rows "
              f"(wanted {args.min_fabricated})")
        return 1
    if args.fail_on_fn and res["overall"]["counts"]["fn"]:
        print(f"\nFAIL: {res['overall']['counts']['fn']} claim(s) that should be "
              f"flagged are not: {res['overall']['false_negatives'][:8]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
