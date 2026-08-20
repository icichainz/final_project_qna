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

AND WHY THREE ARMS WERE STILL BLIND
-----------------------------------
Two reviews then showed the three arms blind exactly where relaxations live.
Three plausible ones — dropping a retrieval-scope gate, removing a negation
guard, forcing a registry deference to True — each moved NOTHING on any arm.
The reason is structural: ``gold`` cannot produce a false negative (every row
was flagged, so recall is pinned at 100% by construction) and ``fabricated``
only mutates claims the release PASSED, so the absence/negative region and the
"was this rightly flagged?" region are in neither.  ``repaired`` is the fourth
arm, and it is where over-strictness has to show up.

The scorer therefore carries four arms, and reports them separately as well
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
``repaired``      claims the release FLAGGED, mutated until they are TRUE
                  about the evidence their own turn held — the figure fixed to
                  one the cited scope prints, the citation moved to the
                  document that states the name, an absence pointed at a page
                  the turn really held.  Each is `should_flag = False`, and
                  each is certified structurally: EVERY checkable term of the
                  repaired claim must be printed in the scope it now cites.
                  A verifier that still flags one has a false negative in the
                  region where over-strictness lives.

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

No file under data/ is written or modified.  The fabricated and repaired arms
are generated in memory, from the read-only recordings, by rules written out
below.  Neither reads a verdict: ``seed_digest`` prints the sha256 of the whole
seed set, and it is identical on any tree, which is the only thing that makes
a before/after recall comparable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcf_qna.rag import verify  # noqa: E402

DEFAULT_GOLD = ROOT / "data" / "eval" / "release_release-1-adjudicated.jsonl"
DEFAULT_EVIDENCE = ROOT / "data" / "eval" / "release_release-1-evidence.jsonl"
DEFAULT_RELEASE = ROOT / "data" / "eval" / "release_release-1.jsonl"

ARMS = ("gold", "held-correct", "fabricated", "repaired")

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
#:
#: MEASURED, NOT ASSERTED. The first version of this pattern matched 0 of the
#: 71 claims the release flagged and 0 of the claims it passed, so the
#: `rider-and` / `rider-semicolon` shapes its docstring advertised were never
#: once seeded and the absence/negative region sat outside every arm. The
#: alternation below was written against the recorded corpus and its coverage
#: is counted in the report (`absence-shaped rows`) rather than claimed here.
_ABSENCE_SHAPE = re.compile(
    r"\b(?:not found|does not exist|is not in|no such|none of|aucun|"
    r"n['\u2019]existe pas|ne mentionnent pas|not in this corpus|"
    r"no proposal|not\s+in\s+the\s+corpus|there are no|are no excerpts|"
    r"do(?:es)? not (?:mention|contain|state|appear|include)|"
    r"ne (?:mentionne|contiennent|contient|mentionnent) pas|"
    r"without any indication|no (?:excerpt|record|mention))\b", re.I)


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


_MARKED_RE = re.compile(
    r"\*\*(.+?)\*\*"
    r"|[\u201c\"\u00ab]\s*([^\u201d\"\u00bb\n]{3,120}?)\s*[\u201d\"\u00bb]"
    r"|\*([^*\n]{3,120}?)\*")


def _marked_names(text: str) -> List[str]:
    """Bold or quoted spans that read as a name, from the text alone.

    Deliberately not ``verify.entities`` — the seed set may not be built by
    the code it scores, or a tree that extracts fewer names would be handed a
    smaller and easier test than the tree it is compared against.
    """
    out = []
    for m in _MARKED_RE.finditer(verify._strip_citations(text)):
        span = (m.group(1) or m.group(2) or m.group(3) or "").strip(" .,;:*_")
        words = [w for w in span.split() if w]
        if len(words) >= 2 and sum(1 for w in words if w[:1].isupper()) >= 2 \
                and not re.search(r"\d", span):
            out.append(span)
    return out


def _absent_name(name: str, ev_norm: str) -> bool:
    return bool(verify.norm_text(name)) and verify.norm_text(name) not in ev_norm


def _one_claim(answer: str, evidence, needle: str):
    """The single extracted claim of ``answer`` containing ``needle``.

    Extraction only. It used to build a ``Replay``, which calls
    ``classify_deterministic`` and then ignored the verdicts — harmless but
    unprovable. No seed-set path now constructs a verdict at all, which is
    what ``test_the_seed_set_is_bit_identical_under_a_forced_verdict`` pins.
    """
    hits = [c for c in verify.extract_claims(answer) if needle in c.text]
    if len(hits) != 1:
        return None, None
    return None, hits[0]


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
# the repaired arm
# ---------------------------------------------------------------------------

#: A digit run, with the scale word that gives it its magnitude. Both halves
#: move together: swapping '28.025' for '49,312' inside 'USD 28.025 million'
#: would leave a scale word the evidence never printed beside the new digits.
#: Thousands may be grouped by comma or by (narrow/non-breaking) space, and
#: the decimal mark may be a period or a comma. What must NOT be swallowed
#: is a comma FOLLOWED BY A SPACE: "p.40, 2024" is two numbers, and reading
#: it as one produced the repair "'40, 2024' is printed verbatim".
_FIGURE_RUN = re.compile(
    r"\d+(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?")
_SCALE_AFTER = re.compile(
    r"\A[  ]*(?:millions?|billions?|thousands?|milliards?|"
    r"m\b|bn\b|k\b|M\b|Md\b)", re.I)

#: Tokens that identify a document, a board or a place without being a bold
#: name — what an absence claim is usually about ('FP999', 'GCF/B.42/02/Add.99',
#: 'B.44', 'Antarctica').
_ID_TOKEN = re.compile(r"GCF/B\.\d+[0-9A-Za-z./]*|\bFP\s?\d{1,4}\b|\bB\.\d{1,3}\b")
_PROPER = re.compile(r"\b[A-Z][A-Za-zÀ-ɏ]{3,}\b")

#: A bracket, for pointing a citation somewhere else. Deliberately not
#: ``verify.parse_citations``: the citation grammar is part of what the
#: repaired arm tests, so the seed set may not be built out of it.
_BRACKET = re.compile(r"\[[^\[\]\n]{0,300}\]")

_STOP = {"the", "and", "for", "with", "from", "that", "this", "are", "was",
         "were", "has", "have", "not", "but", "its", "per", "les", "des",
         "une", "dans", "pour", "est", "sont", "aux", "par", "sur", "que"}


def _content_words(text: str) -> set:
    """Lowercase, deaccented words of >=3 characters, minus function words."""
    return {w for w in re.findall(r"[a-z0-9]{3,}", verify.norm_text(text))
            if w not in _STOP}


def _held_items(evidence) -> List[Tuple[Tuple[str, Optional[int]], str]]:
    """Held keys in one fixed order, so every choice below is deterministic."""
    return sorted(evidence.items(),
                  key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1]))


def _figure_spans(text: str) -> List[str]:
    """Digit runs of >=3 digits, each carrying its own scale word."""
    out = []
    for m in _FIGURE_RUN.finditer(text):
        if len(_digits(m.group(0))) < 3:
            continue          # '(a)', 'p. 5', 'A.8' — a marker, not a figure
        tail = _SCALE_AFTER.match(text[m.end():])
        out.append(m.group(0) + (tail.group(0) if tail else ""))
    return out


def _is_money_shaped(span: str) -> bool:
    """A figure a financing sentence could be stating.

    Rejects the two things that made the first version of this arm hand
    financing claims nonsense: a bare four-digit YEAR ('USD 2024' came from
    context-matching a date), and a short ungrouped run, which in this corpus
    is a page, a section or a list index. A grouped figure (49,751,264 /
    49 151 817) or five digits and up is the real population.
    """
    d = _digits(span)
    if len(d) < 3 or re.fullmatch(r"(?:19|20)\d\d", span.strip()):
        return False
    return bool(re.search(r"[,.  ]", span.strip())) or len(d) >= 5


def _certify(claim_text: str, scope_text: str, exempt: Sequence[str] = ()
             ) -> Tuple[bool, List[str]]:
    """Is EVERY checkable term of this repaired claim printed in the scope it
    now cites?  ``exempt`` are the tokens the row certifies as ABSENT.

    This is the arm's validity bar and it is deliberately stricter than the
    task's 'the repaired value must appear in the held evidence': a citation
    moved onto a page that holds ONE of three figures is not a repaired claim,
    it is a differently broken one, and four such rows were counted as
    verifier false negatives before this check existed.
    """
    scope_norm = verify.norm_text(scope_text)
    scope_digits = _digits(scope_text)
    skip = {verify.norm_text(x) for x in exempt}
    missing = []
    for f in _figure_spans(verify._strip_citations(claim_text)):
        if verify.norm_text(f) not in skip and _digits(f) not in scope_digits:
            missing.append(f)
    for n in _marked_names(claim_text):
        nn = verify.norm_text(n)
        if nn and nn not in skip and nn not in scope_norm:
            missing.append(n)
    return (not missing), missing


def _printed_figure(claim_text: str, items) -> Optional[Tuple[str, tuple]]:
    """The figure the evidence prints for what this claim is talking about.

    'A figure the evidence prints' is the validity bar this arm certifies and
    any printed run clears it — but handing a financing sentence a t-CO2
    tonnage would make the row wrong for a reason that has nothing to do with
    the verifier. So among the printed runs the one whose own neighbourhood
    shares the most words with the claim wins, ties broken by (document, page,
    offset) so the choice never depends on dict order.
    """
    words = _content_words(claim_text)
    best = None
    for key, text in items:
        for m in _FIGURE_RUN.finditer(text):
            if len(_digits(m.group(0))) < 3:
                continue
            tail = _SCALE_AFTER.match(text[m.end():])
            span = m.group(0) + (tail.group(0) if tail else "")
            if not _is_money_shaped(m.group(0)):
                continue
            ctx = _content_words(text[max(0, m.start() - 70):m.end() + 70])
            rank = (-len(words & ctx), key[0], -1 if key[1] is None else key[1],
                    m.start())
            if best is None or rank < best[0]:
                best = (rank, span, key)
    return (best[1], best[2]) if best else None


def _cite_for(key) -> str:
    """The bracket that points at one held key, in the corpus' own style."""
    return f"[{key[0]}]" if key[1] is None else f"[{key[0]}, p. {key[1]}]"


def _point_at(claim_text: str, cite: str) -> str:
    """``claim_text`` carrying ``key`` as its ONLY citation.

    Every existing bracket goes: a repair that leaves a second bracket behind
    widens the scope instead of moving it, and the row would then be certified
    against a union nobody chose.
    """
    stripped = _BRACKET.sub("", claim_text)
    stripped = re.sub(r"[  ]{2,}", " ", stripped).strip()
    return stripped.rstrip(".").rstrip() + " " + cite + "."


def _keys_holding(items, needle_norm: str) -> List[tuple]:
    """Held keys whose text contains ``needle_norm``, notes excluded.

    The computed note is not the document saying it — an absence note reading
    'FP999: NOT FOUND' contains 'FP999', and counting that as the corpus
    printing FP999 is what made the first version of this arm seed zero
    absence rows.
    """
    if not needle_norm:
        return []
    return [k for k, text in items
            if k[0] != verify.NOTES_DOC and needle_norm in verify.norm_text(text)]


def _scope_candidates(items) -> List[Tuple[str, str, str, int]]:
    """Every scope a citation is allowed to name, narrowest first.

    Two shapes, both of which the recorded answers really use: one held page
    ('[doc, p. 5]') and a whole document ('[doc]', which ruling 5 resolves to
    every held key of that document). The notes key is never a candidate — it
    has no citable spelling and a computed note is not the corpus saying it.
    """
    out = []
    for key, text in items:
        if key[0] != verify.NOTES_DOC:
            out.append((_cite_for(key), text, f"{key[0]} p.{key[1]}", 0))
    for d in sorted({k[0] for k, _t in items if k[0] != verify.NOTES_DOC}):
        out.append((f"[{d}]", "\n".join(t for k, t in items if k[0] == d),
                    f"{d} (whole document)", 1))
    return out


def _best_scope(items, claim_text: str, exempt: Sequence[str] = ()):
    """The narrowest scope that prints EVERY checkable term, or None.

    Ranked by (terms it fails to print, width, position), so a single page
    always beats the whole document and the order never depends on a dict.
    """
    best = None
    for i, (cite, text, label, width) in enumerate(_scope_candidates(items)):
        _ok, missing = _certify(claim_text, text, exempt)
        rank = (len(missing), width, i)
        if best is None or rank < best[0]:
            best = (rank, cite, text, label)
    if best is None or best[0][0]:
        return None
    return best[1], best[2], best[3]


def _doc_text(items) -> str:
    return "\n".join(t for k, t in items if k[0] != verify.NOTES_DOC)


def repair(cases: List[dict], flagged: Dict[str, List[str]]) -> List[dict]:
    """Claims the RELEASE FLAGGED, mutated until they are TRUE about the
    evidence their own turn held.

    THE REGION THIS ARM EXISTS FOR.  ``gold`` cannot produce a false negative:
    every one of its 71 rows was flagged, so a verifier that flags everything
    scores a perfect recall on it.  ``fabricated`` only ever mutates claims the
    release PASSED, so the "was this rightly flagged?" region is in neither.
    Three plausible relaxations — dropping a retrieval-scope gate, removing a
    negation guard, forcing a registry deference to True — moved NOTHING on any
    arm, because over-strictness had nowhere to show up.  A repaired row is a
    claim that WAS flagged and is now true about its own evidence, so a
    verifier that still flags it has a false negative exactly where
    over-strictness lives.  Every row is ``should_flag = False``.

    ``flagged`` is ``{case_id: [claim texts the RECORDED RELEASE flagged]}`` —
    read out of ``release_release-1.jsonl``, the same provenance discipline
    ``fabricate`` uses for its basis, so the seed set is identical in every
    tree.  Nothing below reads a verdict, a status or a reason: the mutation is
    chosen from the claim's TEXT and the evidence TEXT, and every row records
    the structural fact that certifies it.

    Four shapes, one per way a flagged claim can be made true:

    ``figure``    the claim states a figure the scope it cites does not print;
                  it is replaced by a figure that scope DOES print.
    ``entity``    the claim names something its cited scope does not contain
                  but another held key does; the citation moves to that key.
    ``citation``  the claim is uncited, or cites a key this turn never held,
                  while a held key does print its content; the bracket is
                  pointed there.
    ``absence``   an absence/negative claim whose negated token appears in NO
                  held document key, given a citation to a page the turn did
                  hold.  The previous seed set contained ZERO absence-shaped
                  rows despite a docstring that said otherwise, so this shape
                  is counted in the report, never asserted here.
    """
    out: List[dict] = []

    def add(case_id, answer, evidence, fixed, why, valid):
        """Record the row, once the mutated answer really yields that claim.

        Located by exact text, not by a needle substring: a bracket like
        '[103_gcf-b30-03-add04, p. 140]' occurs in several claims of the same
        answer, and matching on it silently dropped most of this arm.
        """
        claims = verify.extract_claims(answer)
        hits = [c for c in claims if c.text == fixed] or \
            [c for c in claims if fixed and fixed in c.text]
        if len(hits) != 1:
            return
        claim = hits[0]
        tag = hashlib.sha256(
            f"{case_id}|{why}|{claim.text}".encode("utf-8")).hexdigest()[:10]
        row_id = f"rep-{why}-{case_id}-{tag}"
        if any(r["row_id"] == row_id for r in out):
            return
        out.append({"row_id": row_id, "arm": "repaired", "case_id": case_id,
                    "why": why, "answer": answer, "evidence": evidence,
                    "claim_text": claim.text, "should_flag": False,
                    "validity": valid})

    for case in sorted(cases, key=lambda c: c["case_id"]):
        cid = case["case_id"]
        answer = case.get("answer") or ""
        ev = evidence_of(case)
        items = _held_items(ev)
        doc_norm = verify.norm_text(_doc_text(items))
        wanted = set(flagged.get(cid) or ())
        if not wanted:
            continue

        for claim in verify.extract_claims(answer):
            if claim.text[:160] not in wanted:
                continue      # only a claim the RELEASE FLAGGED is repairable
            bare = verify._strip_citations(claim.text)
            cited = [(c.doc, c.page) for c in claim.citations if c.doc]
            # the scope the claim points at, and what it prints
            in_scope = [(k, t) for k, t in items if k in cited] or \
                ([] if cited else items)
            scope_digits = _digits("\n".join(t for _k, t in in_scope))
            scope_norm = verify.norm_text("\n".join(t for _k, t in in_scope))
            names = _marked_names(claim.text)

            unheld = bool(cited) and any(k not in ev for k in cited)
            # An absence-shaped claim is repaired by shape 4 and shape 4 only:
            # shapes 1-3 produce the SAME mutated text for an uncited negative,
            # which double-counted the region under two labels and made
            # `citation` look worse than it is.
            negative = bool(_ABSENCE_SHAPE.search(claim.text))

            # 1. FIGURE — the value the scope it cites really prints.
            #    Not for a claim whose citation was never retrieved: there is
            #    no scope to be wrong about, and shape 3 is that claim's repair.
            wrong = ([f for f in _figure_spans(bare) if _digits(f) not in scope_digits]
                     if not (unheld or negative) else [])
            got = _printed_figure(claim.text, in_scope or items) if wrong else None
            if got and _digits(got[0]) != _digits(wrong[0]):
                span, key = got
                fixed = claim.text.replace(wrong[0], span, 1)
                ok, _miss = _certify(fixed, "\n".join(t for _k, t in in_scope))
                if ok:
                    add(cid, answer.replace(claim.text, fixed, 1), ev, fixed,
                        "figure",
                        f"'{span}' is printed verbatim on {key[0]} p.{key[1]}, "
                        f"and the cited scope prints every other term")

            # 2. ENTITY — the document that actually states the name
            for name in (() if negative else names[:1]):
                norm = verify.norm_text(name)
                if not norm or norm in scope_norm:
                    continue          # its own scope already states it
                got = _best_scope(items, claim.text)
                if got is None or norm not in verify.norm_text(got[1]):
                    continue
                fixed = _point_at(claim.text, got[0])
                if fixed != claim.text:
                    add(cid, answer.replace(claim.text, fixed, 1), ev, fixed,
                        "entity", f"'{name}' appears in {got[2]}, which prints "
                        f"every other checkable term of the claim")

            # 3. CITATION — uncited, or citing a key this turn never held
            if (not cited or unheld) and not negative:
                got = _best_scope(items, claim.text)
                if got is not None:
                    fixed = _point_at(claim.text, got[0])
                    if fixed != claim.text:
                        add(cid, answer.replace(claim.text, fixed, 1), ev, fixed,
                            "citation",
                            f"{got[2]} prints every checkable term of this claim")

            # 4. ABSENCE — a negative whose subject appears in no held document
            if negative:
                # Identifiers and bold names first, bare capitalised words
                # last and never the claim's own first word: 'None of the
                # excerpts mention X' opens with a capital that carries no
                # information, and naming it as the absent subject made the
                # validity line read "'None' appears in NO held document key".
                opener = bare.split()[0].strip("*\u201c\"'(") if bare.split() else ""
                subj = [t for t in (_ID_TOKEN.findall(bare) + names
                                    + [w for w in _PROPER.findall(bare)
                                       if w != opener])
                        if verify.norm_text(t)
                        and verify.norm_text(t) not in doc_norm]
                got = _best_scope(items, claim.text, exempt=subj) if subj else None
                if got is not None:
                    fixed = _point_at(claim.text, got[0])
                    if fixed != claim.text:
                        add(cid, answer.replace(claim.text, fixed, 1), ev, fixed,
                            "absence",
                            f"'{subj[0]}' appears in NO held document key; the "
                            f"cited {got[2]} was held and prints every term the "
                            f"claim does NOT deny")
    return out


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def release_failures(release: List[dict]) -> Tuple[Dict[str, set], List[dict]]:
    """``({case_id: {flagged claim texts}}, [excluded cases])`` from the release.

    A recorded fact, not a verdict from the verifier under test: it is what
    both the held-correct arm (the complement) and the repaired arm (this set)
    are drawn from, so both are identical in every tree.
    """
    out: Dict[str, set] = {}
    problems: List[dict] = []
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
        out[r["id"]] = {f.get("text") for f in fails if isinstance(f, dict)}
    return out, problems


def absence_census(cases: List[dict], flagged: Dict[str, set]) -> Dict[str, int]:
    """How many FLAGGED claims are absence/negative-shaped, before any repair.

    Counted, not asserted. The previous seed set's docstring advertised
    absence shapes it never once produced; the gap between ``candidates`` and
    the ``absence`` rows the arm ends up with is the honest statement of how
    much of the region the instrument reaches.
    """
    seen = {"flagged": 0, "candidates": 0}
    for case in cases:
        wanted = flagged.get(case["case_id"]) or set()
        for claim in verify.extract_claims(case.get("answer") or ""):
            if claim.text[:160] not in wanted:
                continue
            seen["flagged"] += 1
            seen["candidates"] += bool(_ABSENCE_SHAPE.search(claim.text))
    return seen


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
    failed_texts, truncated = release_failures(release)
    problems += truncated

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
    # The mirror image of `basis`, from the same recorded fact: what the
    # release FLAGGED. `failed_texts` already holds it, truncation-checked.
    rows += repair(cases, {cid: sorted(t for t in texts if t)
                           for cid, texts in failed_texts.items()})
    return rows, problems


def seed_digest(rows: List[dict]) -> str:
    """sha256 of the seed set itself — every row, every mutated answer, every
    byte of evidence behind it, and nothing the verifier decided.

    Two trees that build the same seed set print the same digest. That is the
    only thing that makes a before/after recall comparable: an earlier revision
    seeded from whatever the current code happened to support, so a more
    permissive verifier was handed a smaller and easier test and its better
    score meant nothing.
    """
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda r: (r["arm"], r["row_id"])):
        ev = sorted([k[0], -1 if k[1] is None else k[1],
                     hashlib.sha256((v or "").encode("utf-8")).hexdigest()]
                    for k, v in r["evidence"].items())
        h.update(json.dumps(
            [r["row_id"], r["arm"], r["case_id"], r.get("why"), r["claim_text"],
             r["should_flag"], r["answer"], r.get("validity"), ev],
            sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


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
            "validity": row.get("validity"),
            "status": verdict.status, "flagged": bool(verdict.failed),
            "reason": verdict.reason, "flags": list(verdict.flags),
            "plausible": bool(verdict.plausible), "kind": verdict.claim.kind,
            "claim_text": row["claim_text"][:200]})

    def escapes(r):
        """Does this claim reach the user as a fact under the LIVE settings?

        Production runs `VERIFY_LLM=1`. An UNSUPPORTED-and-plausible verdict is
        exactly the residue `adjudicate` is handed, so a clearing judge ships
        it; a CONTRADICTED one is never shown to the judge at all. Collapsing
        both into 'flagged' hides a real relaxation: forcing the registry
        deference to True moved two fabricated rows CONTRADICTED ->
        UNSUPPORTED, changing nothing in TP/FP/FN/TN while making both
        judge-promotable. That is the shape this counter exists to see.
        """
        return (not r["flagged"]) or (r["status"] == verify.UNSUPPORTED
                                      and r["plausible"])

    def matrix(subset):
        tp = [r for r in subset if r["should_flag"] and r["flagged"]]
        fp = [r for r in subset if not r["should_flag"] and r["flagged"]]
        fn = [r for r in subset if r["should_flag"] and not r["flagged"]]
        tn = [r for r in subset if not r["should_flag"] and not r["flagged"]]
        # judge-bound views: what survives a fully-clearing judge, and what no
        # judge can rescue
        fn_j = [r for r in subset if r["should_flag"] and escapes(r)]
        fp_hard = [r for r in subset
                   if not r["should_flag"] and r["status"] == verify.CONTRADICTED]
        prec = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
        rec = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"counts": {"tp": len(tp), "fp": len(fp), "fn": len(fn), "tn": len(tn),
                           "fn_clearing_judge": len(fn_j),
                           "fp_contradicted": len(fp_hard)},
                "precision": prec, "recall": rec, "f1": f1,
                "false_negatives": [r["row_id"] for r in fn],
                "false_positives": [r["row_id"] for r in fp],
                "escapes_clearing_judge": [r["row_id"] for r in fn_j],
                "contradicted_but_should_not_be": [r["row_id"] for r in fp_hard]}

    by_label: Dict[str, Dict[str, int]] = {}
    for r in scored:
        if r["arm"] != "gold":
            continue
        b = by_label.setdefault(r["label"] or "(unlabelled)",
                                {"rows": 0, "flagged": 0, "cleared": 0})
        b["rows"] += 1
        b["flagged" if r["flagged"] else "cleared"] += 1

    def shapes(arm):
        return dict(Counter(r["why"] for r in scored if r["arm"] == arm))

    def by_shape(arm):
        """Per shape: rows, and how many the verifier got wrong. The repaired
        arm's shapes do NOT behave alike — the absence block is a deliberate
        false-negative region (rulings 3 and 7 were deleted) and reporting it
        inside one aggregate would hide both it and everything else."""
        out = {}
        for r in scored:
            if r["arm"] != arm:
                continue
            b = out.setdefault(r["why"] or "(none)", {"rows": 0, "flagged": 0})
            b["rows"] += 1
            b["flagged"] += int(r["flagged"])
        return out

    return {"rows": scored, "unmatched": unmatched,
            "overall": matrix(scored),
            "arms": {a: matrix([r for r in scored if r["arm"] == a]) for a in ARMS},
            "arm_sizes": dict(Counter(r["arm"] for r in scored)),
            "by_label": by_label,
            "fabrication_kinds": shapes("fabricated"),
            "repair_kinds": shapes("repaired"),
            "repair_by_shape": by_shape("repaired"),
            "fabricated_by_shape": by_shape("fabricated"),
            "absence_shaped": {
                a: sum(1 for r in scored if r["arm"] == a and r["why"] in
                       ("absence", "rider-and", "rider-semicolon"))
                for a in ARMS}}


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
            f"P {_pct(m['precision'])}  R {_pct(m['recall'])}  F1 {_pct(m['f1'])}"
            f"   FN|judge-clears {c.get('fn_clearing_judge', 0):3d}"
            f"  FP|contradicted {c.get('fp_contradicted', 0):3d}")


def report(res: dict, states: dict, baseline: Optional[dict] = None,
           verbose: bool = False, arm: Optional[str] = None) -> None:
    print("=" * 92)
    print("VERIFIER vs ADJUDICATED GOLD + SEEDED CLAIMS  (deterministic replay, no API)")
    print("=" * 92)
    print(f"seed set sha256 {res.get('seed_sha256', '(not computed)')}")
    print(f"rows scored {len(res['rows'])}   unmatched {len(res['unmatched'])}")
    for u in res["unmatched"][:12]:
        print(f"  ! UNMATCHED [{u['arm']}] {u['row_id']}: {u['why']}")
    print()
    for a in ARMS:
        print(_matrix_line(a, res["arms"][a], res["arm_sizes"].get(a, 0)))
    print(_matrix_line("OVERALL", res["overall"], len(res["rows"])))
    print()
    for arm, key in (("fabricated", "fabricated_by_shape"),
                     ("repaired", "repair_by_shape")):
        block = res.get(key) or {}
        verb = "still flagged" if arm == "repaired" else "caught"
        print(f"  {arm} shapes ({verb} / rows):")
        for shape in sorted(block):
            b = block[shape]
            n = b["flagged"] if arm == "fabricated" else b["flagged"]
            print(f"    {shape:<18} {n:>3} / {b['rows']:<3}"
                  + ("   <- deliberate FN region (rulings 3 and 7 deleted)"
                     if arm == "repaired" and shape == "absence" else ""))
    abs_counts = res.get("absence_shaped") or {}
    print("  absence/negative-shaped rows per arm: " + ", ".join(
        f"{a} {abs_counts.get(a, 0)}" for a in ARMS))
    cen = res.get("absence_census") or {}
    if cen:
        print(f"  of the {cen.get('flagged', 0)} claims the release flagged, "
              f"{cen.get('candidates', 0)} are absence/negative-shaped; "
              f"{abs_counts.get('repaired', 0)} certify into the repaired arm "
              f"(the rest name a figure or a name NO single held scope prints)")
    sample = [r for r in sorted(res["rows"], key=lambda r: r["row_id"])
              if r["arm"] == "repaired" and r.get("validity")]
    print(f"  repaired validity (structural, {len(sample)}/"
          f"{res['arm_sizes'].get('repaired', 0)} rows carry one); sample:")
    step = max(1, len(sample) // 5)
    for r in sample[::step][:5]:
        print(f"    {r['row_id']:<40} {r['validity'][:70]}")
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
              f"FN|judge {bc.get('fn_clearing_judge', 0):3d}->"
              f"{nc.get('fn_clearing_judge', 0):<3d} "
              f"FP|contra {bc.get('fp_contradicted', 0):3d}->"
              f"{nc.get('fp_contradicted', 0):<3d}  "
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
    res["seed_sha256"] = seed_digest(rows)
    _failed, _trunc = release_failures(read_jsonl(args.release))
    res["absence_census"] = absence_census(read_jsonl(args.evidence), _failed)
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
