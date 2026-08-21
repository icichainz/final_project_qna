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
``contradicted``  claims whose CITED key prints a genuinely different value
                  under the claim's OWN field label — the one shape where a
                  correct verifier may not answer UNSUPPORTED but must answer
                  CONTRADICTED.  Each is `should_flag = True` AND
                  `must_contradict = True`, and each is certified structurally
                  from the evidence text: the rival really is printed, under
                  that field's own label, on the key the claim now cites.

AND WHY A FOURTH ARM WAS STILL BLIND TO THE CONTRADICTION PATH
--------------------------------------------------------------
The merge review of 4a04d32 replaced ``verify._field_conflict`` with
``return None`` — deleting EVERY evidence-text contradiction the verifier can
emit — and this report did not move: gold 30.8%, overall 73.1%, recall 68.0%,
the four matrices identical to the count, and the caution census identical.
Two causes, both now closed:

  (a) ``Replay.cautions()`` counted flags on SUPPORTED claims only.  Two of
      the verifier's flags — ``conflict-elsewhere-in-document`` and
      ``known-document-conflict`` — are emitted ONLY on a CONTRADICTED verdict,
      so the census could not see them appear or disappear.  It is now taken
      over every verdict, tagged with the status it sits on, and over the
      SEEDED rows as well as the 66 recorded answers: the recorded answers
      contain zero contradicted verdicts, so on their own they can never
      witness this path at all.
  (b) No arm contained a claim that MUST be contradicted.  The two fabricated
      rows that happened to come out CONTRADICTED are flagged either way, so
      losing the contradiction moved no cell of a matrix that only counts
      ``flagged``.  The ``contradicted`` arm supplies rows whose ONLY defect is
      the field conflict, so losing it PROMOTES them, and ``TP|contra`` counts
      the status directly for the rows where it does not.

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

ARMS = ("gold", "held-correct", "fabricated", "repaired", "contradicted")

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
        """Every flag these verdicts carry, deduped and TAGGED WITH ITS STATUS.

        It used to read ``if v.status == verify.SUPPORTED``, and that made the
        census blind to the whole contradiction path.  Two of the verifier's
        flags — ``conflict-elsewhere-in-document`` and
        ``known-document-conflict`` — are appended only on the branches that
        return CONTRADICTED, so under the SUPPORTED-only filter they could
        never be counted, never appear and never disappear.  Worse, the census
        deduped by flag NAME: promoting a contradicted claim to SUPPORTED
        hands its remaining flags to the census, and if any other claim of the
        same answer already carried that name the promotion cancelled out to
        nothing.  Tagging by status makes both directions visible —
        ``contradicted:citation-page-mismatch`` and
        ``supported:citation-page-mismatch`` are different census entries.

        ``user_cautions`` keeps the narrower thing the app actually renders.
        """
        return sorted({f"{v.status}:{f}" for v in self.verdicts for f in v.flags})

    def user_cautions(self) -> List[str]:
        """The subset the app shows: flags on claims that did NOT fail."""
        return sorted({f for v in self.verdicts if not v.failed for f in v.flags})


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
# the contradiction arm
# ---------------------------------------------------------------------------

#: Field labels FOR THE SEED SET, and deliberately not ``verify._FIELD_LABELS``
#: / ``verify.claim_field``.  Which field a claim is about is the hinge of the
#: whole contradiction path — "make conflict detection ignore the field" is one
#: of the ablations this arm exists to detect — so a seed set built out of the
#: detector under test would move with it and see nothing.  Same argument as
#: ``_marked_names`` (not ``verify.entities``) and ``_BRACKET`` (not
#: ``verify.parse_citations``).
#: THE LABEL IS THE FIELD'S NAME, and nothing longer.  'Requested GCF
#: funding' is the name 'GCF funding' with prose in front of it, and prose in
#: front of a label is what tells a reader the segment is ABOUT the field
#: rather than STATING it.  Writing the compound in here instead let a seed row
#: read '- Requested GCF funding: EUR 5,600,000' as a field statement whose
#: label heads its segment, which no careful reader of that line would say —
#: and the arm then blamed the verifier for a row the seed set had got wrong.
_SEED_FIELDS: List[Tuple[str, str]] = [
    ("gcf_financing",
     r"gcf\s+(?:financing|funding|contribution|proceeds)"
     r"|financement\s+(?:du\s+)?(?:fvc|gcf)"),
    ("total_financing",
     r"total\s+(?:financing|project\s+(?:cost|financing|value|budget)|investment)"
     r"|financement\s+total|co[uû]t\s+total"),
    ("co_financing", r"co-?financing|cofinancement"),
]
_SEED_FIELD_RES = [(f, re.compile(r, re.I)) for f, r in _SEED_FIELDS]

#: What may precede a label and still leave it heading its own segment: a
#: bullet, a table cell, a numbering.  Prose may not.
_SEED_HEAD_OK = re.compile(r"^[\s|#>*–—•()a-z0-9.,:\-]{0,40}$")
#: Template prose that MENTIONS a field without stating one.
_SEED_INSTRUCTION = re.compile(
    r"\b(?:in case of|please|specify|indicate|enter\s|if applicable|"
    r"choose an item|for each|see annex|e\.g\.|guidance|max\s+\d+\s+words)\b", re.I)
#: A scale word attached to a run, for deciding whether a figure is the kind a
#: financing field states.
_SEED_SCALE = re.compile(
    r"(?:millions?|billions?|thousands?|milliards?|\bM\b|\bbn\b|\bMd\b)", re.I)


def _seed_field_of(text: str) -> Optional[str]:
    """The ONE field this text names, or None when it names none or several.

    'Exactly one' is what keeps the arm honest across two independent
    detectors.  A claim naming two fields has a field only by whoever's label
    list is consulted first, and a row whose expected verdict turns on that
    ordering would be testing the list rather than the path.
    """
    body = verify.norm_text(verify._strip_citations(text))
    hit = [f for f, rx in _SEED_FIELD_RES if rx.search(body)]
    return hit[0] if len(hit) == 1 else None


def _seed_runs(text: str) -> List[str]:
    """Every digit run of the text, carrying its own scale word."""
    out = []
    for m in _FIGURE_RUN.finditer(text or ""):
        if len(_digits(m.group(0))) < 3:
            continue
        tail = _SCALE_AFTER.match(text[m.end():])
        out.append(m.group(0) + (tail.group(0) if tail else ""))
    return out


def _seed_is_rival(span: str) -> bool:
    """Is this a figure a financing field could be stating?

    Either the run carries its own scale word, or it is at least ten thousand.
    A grouped four-digit run ('1,234') is a page, a section or a table index in
    this corpus, and a bare year is a date; reading either as a financing
    figure would seed rows no reading of the evidence calls a contradiction.
    """
    d = _digits(span)
    if len(d) < 3 or re.fullmatch(r"(?:19|20)\d\d", span.strip()):
        return False
    return bool(_SEED_SCALE.search(span)) or int(d) >= 10000


#: A figure the text prints AS MONEY — a currency token in front of it, a
#: currency token behind it, or a scale word attached.  The ``elsewhere`` shape
#: needs this and the label-anchored scan does not: a value that follows a
#: field label is money BECAUSE of the label, but a figure picked out of a page
#: that carries no label has only its own neighbourhood to say what it is, and
#: without this bar the shape seeded '134,000 Households' and '77634 x 432
#: (CO2 eq)' as financing figures and then scored the verifier wrong for not
#: contradicting them.
_SEED_CUR = r"(?:USD|US\$|EUR|GBP|CHF|XOF|ZAR|\$|€|£)"
_SEED_MONEY_RE = re.compile(
    r"(?P<pre>" + _SEED_CUR + r"\s*)?"
    r"(?P<run>\d+(?:[   ]\d{3})+(?:[.,]\d+)?|\d+(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)"
    r"(?P<sc>\s*(?:millions?|billions?|thousands?|milliards?|M|bn|Md)\b)?"
    r"(?P<post>\s*" + _SEED_CUR + r")?", re.I)


def _seed_money_spans(text: str) -> List[str]:
    """Every money-shaped span of the text, currency and scale word included.

    The span carries its currency because the row's mutation has to keep the
    claim readable against the page: dropping 'EUR' out of 'EUR 5,600,000' and
    pasting the bare digits into a sentence that says 'USD' produces a claim
    the page does not print for a reason that has nothing to do with the field.
    """
    out = []
    for m in _SEED_MONEY_RE.finditer(text or ""):
        # A CURRENCY TOKEN, not merely a scale word. '123 million tCO2eq' is a
        # mitigation tonnage and '134,000 Households' a beneficiary count; both
        # were seeded as financing figures by the first version of this scan,
        # and the arm then scored the verifier wrong for not contradicting a
        # claim about a number that was never money.
        if not (m.group("pre") or m.group("post")):
            continue
        d = _digits(m.group("run"))
        if not d or (len(d) < 3 and not m.group("sc")):
            continue
        if re.fullmatch(r"(?:19|20)\d\d", m.group("run").strip()):
            continue
        out.append(m.group(0).strip())
    return out


def _seed_printed_fields(text: str) -> Dict[str, Tuple[str, str]]:
    """``{field: (value, the segment printing it)}`` for ONE evidence key.

    A segment counts only when the label heads it, when it names exactly one
    field, and when a rival-sized figure follows that label.  A field this key
    prints TWICE with two different values is dropped rather than guessed at:
    the key already disagrees with itself and no single rival can be named.
    """
    found: Dict[str, List[Tuple[str, str]]] = {}
    for line in (text or "").splitlines():
        if _SEED_INSTRUCTION.search(line):
            continue
        for seg in re.split(r"(?<=[;|])|(?<=\.\s)", line):
            # EARLIEST match wins, then the head test — not first-pattern
            # wins.  A label further along the segment has more in front of it,
            # and 'more in front of it' is exactly what the head test judges;
            # taking a pattern's turn in a list as the tie-breaker would make
            # the seed set's reading of a line depend on the order somebody
            # wrote the list in.
            hits = []
            for f, rx in _SEED_FIELD_RES:
                m = rx.search(seg)
                if m:
                    hits.append((m.start(), f, m))
            if not hits:
                continue
            hits.sort()
            if len(hits) > 1 and hits[0][0] == hits[1][0]:
                continue                     # two labels, one span: ambiguous
            _at, f, m = hits[0]
            if not _SEED_HEAD_OK.match(seg[:m.start()]):
                continue
            val = next((s for s in _seed_runs(seg[m.end():m.end() + 80])
                        if _seed_is_rival(s)), None)
            if val:
                found.setdefault(f, []).append((val, seg.strip()[:200]))
    return {f: vs[0] for f, vs in found.items()
            if len({_digits(v) for v, _s in vs}) == 1}


#: The fact-registry field name behind each of the seed set's field names.
#: The v2 schema, read straight out of ``data/registry_v2.json`` — a recorded
#: corpus fact like the release files, not a verdict and not code under test.
_SEED_V2_FIELD = {"gcf_financing": "gcf_funding_requested",
                  "total_financing": "total_financing",
                  "co_financing": "co_financing"}


def _canonical_digits(doc_id: str, field: str) -> Optional[str]:
    """The digits of the value the fact registry ELECTED for this doc/field.

    Why the seed set has to know this.  ``verify.registry_ruled_compatible``
    may outrank a page-level disagreement, and its first clause is that the
    answer is stating the registry's own canonical reading.  FP152 is the row
    it exists for: the document prints 'Total financing (SCF + co-finance) 720
    M USD' on p.5 and '(a) Total project financing: $100,000,000' on p.55
    inside a per-project cost calculation, and the registry filed the second
    as ``supporting`` — a component, not a rival total.  A seed row stating
    720 M and citing p.55 is therefore NOT a contradiction; the verifier is
    right to stay silent, and the first version of this arm seeded it anyway
    and scored the verifier wrong for it.

    So no row may state a canonical value.  That breaks clause 1 for every
    row, which means no row's expected verdict depends on the deference at all
    — and the deference stays fully ablatable: forcing it True still has to
    skip rivals it was never consulted about, and the arm still sees that.
    """
    try:
        blob = json.loads((ROOT / "data" / "registry_v2.json")
                          .read_text(encoding="utf-8"))
    except Exception:                # no registry is silence, and silence is fine
        return None
    v2 = _SEED_V2_FIELD.get(field)
    row = (blob.get("documents") or {}).get(doc_id) or {}
    for c in ((row.get("facts") or {}).get(v2) or []):
        if c.get("status") == "canonical":
            return _digits(c.get("raw", "")) or None
    return None


def _sibling_digits(claims, claim, doc: str) -> set:
    """Figures the answer's OTHER claims state while naming ``doc``.

    A registry conflict note says, verbatim, 'report both figures with their
    pages', and an answer that obeys prints one figure per bullet.  The
    verifier is RIGHT not to contradict a bullet whose sibling reports the
    other side — 22 of the adjudicated false positives were exactly that
    mistake — so such a claim is not a contradiction and may not be seeded as
    one.  The bracket text is read with the seed set's own ``_BRACKET``, never
    ``verify.parse_citations``: the citation grammar is under test too.
    """
    out = set()
    for other in claims:
        if other.text == claim.text:
            continue
        if doc[:24] not in " ".join(_BRACKET.findall(other.text)):
            continue
        out |= {_digits(s) for s in _seed_runs(verify._strip_citations(other.text))}
    return out


def contradict(cases: List[dict]) -> List[dict]:
    """Claims whose CITED key prints a different value under their OWN field.

    THE REGION THIS ARM EXISTS FOR.  Replacing ``verify._field_conflict`` with
    ``return None`` deletes every evidence-text contradiction the verifier can
    emit, and the other four arms did not move one digit.  They count
    ``flagged``, and a claim that loses its contradiction usually keeps being
    flagged for some second reason — it comes back UNSUPPORTED instead, which
    is a different verdict shown to a different downstream (the judge is
    handed an UNSUPPORTED-and-plausible claim and may clear it; it is never
    shown a CONTRADICTED one) but the same cell of the matrix.

    So every row here is built so the field conflict is the ONLY thing wrong
    with it: after the mutation EVERY other checkable term of the claim is
    printed in the scope it now cites (``_certify``, the repaired arm's bar).
    Losing the conflict then does not degrade the verdict, it PROMOTES the
    claim, and a promotion is a false negative the matrix counts.

    THE MUTATION.  A recorded claim is usually RIGHT about its field, so the
    run this replaces is, by preference, the run that states the rival — the
    claim is turned into one that is wrong about that field and about nothing
    else — and the citation is re-pointed at the key that disagrees.  Three
    conditions then have to hold, and each of them is a case where a correct
    verifier must NOT contradict, so a row that fails one is dropped rather
    than counted against the verifier:

    * the mutated claim may not still state the rival itself;
    * no sibling claim naming that document may state it either (the 'report
      both figures with their pages' behaviour — ``_sibling_digits``);
    * the value the claim now states may not be the registry's own canonical
      reading (``_canonical_digits``), which is clause 1 of the deference
      ``verify.registry_ruled_compatible`` is allowed to claim.

    Nothing here reads a verdict, a status or a reason.  The field is decided
    by ``_seed_field_of`` and the rival by ``_seed_printed_fields``, both
    written for the seed set; the mutation is chosen from the claim's TEXT and
    the evidence TEXT; and every row records the structural fact certifying it.
    Four shapes, one per branch of the path:

    ``same-key``    the cited key prints W for the claim's field and the claim
                    states a figure whose digits occur in NO held key at all.
                    ``_field_conflict`` over the strict scope.
    ``transposed``  the claim states the value the cited key prints for a
                    DIFFERENT field.  The figure verifies against the page, so
                    this row reaches the POST-SUPPORT conflict check — the
                    branch whose loss promotes the claim to SUPPORTED.
    ``wrong-page``  the claim states the value the SAME DOCUMENT prints for the
                    SAME field on another key and cites the key that
                    disagrees.  This is the corpus' own shape; registry v2
                    records 120 documents with an internal financing conflict.
                    Losing the conflict lets the same-document fallback
                    support it.
    ``elsewhere``   the cited key prints the claim's figure but carries no
                    label for its field, while another key of the document
                    prints the rival.  The only shape that reaches the
                    cross-page scan and its ``conflict-elsewhere-in-document``
                    caution.
    """
    out: List[dict] = []

    for case in sorted(cases, key=lambda c: c["case_id"]):
        cid = case["case_id"]
        answer = case.get("answer") or ""
        ev = evidence_of(case)
        items = _held_items(ev)
        all_digits = _digits("\n".join(ev.values()))
        printed: Dict[tuple, Dict[str, Tuple[str, str]]] = {}
        for k, t in items:
            if k[0] == verify.NOTES_DOC:
                continue          # the notes have no citable spelling
            got = _seed_printed_fields(t)
            if got:
                printed[k] = got
        if not printed:
            continue

        claims = verify.extract_claims(answer)
        for claim in claims:
            field = _seed_field_of(claim.text)
            if not field:
                continue
            bare = verify._strip_citations(claim.text)
            runs = [s for s in _seed_runs(bare) if _seed_is_rival(s)]
            if not runs:
                continue          # it states no value for the field
            here = [k for k in printed if field in printed[k]]
            if not here:
                continue

            def emit(why, key, value, rival, valid, exempt=(), target=None):
                """``key`` is the key the claim will CITE, ``rival`` the value
                the field is really given.  Same key for shapes 1-3; different
                for shape 4, where the rival sits elsewhere in the document."""
                rd, vd = _digits(rival), _digits(value)
                if not vd or vd == rd:
                    return False
                if target is None:
                    target = next((s for s in runs if _digits(s) == rd), runs[0])
                if claim.text.count(target) != 1:
                    return False
                fixed = _point_at(claim.text.replace(target, value, 1),
                                  _cite_for(key))
                if fixed == claim.text:
                    return False
                if rd in {_digits(s) for s in
                          _seed_runs(verify._strip_citations(fixed))}:
                    return False        # the claim still states the rival
                if rd in _sibling_digits(claims, claim, key[0]):
                    return False        # the answer reports both sides
                canon = _canonical_digits(key[0], field)
                if canon and canon == vd:
                    return False        # the registry's own canonical reading
                ok, _missing = _certify(fixed, ev.get(key, ""), exempt=list(exempt))
                if not ok:
                    return False        # something ELSE is wrong with it too
                mutated = answer.replace(claim.text, fixed, 1)
                if len([c for c in verify.extract_claims(mutated)
                        if c.text == fixed]) != 1:
                    return False
                tag = hashlib.sha256(
                    f"{cid}|{why}|{fixed}".encode("utf-8")).hexdigest()[:10]
                row_id = f"con-{why}-{cid}-{tag}"
                if any(r["row_id"] == row_id for r in out):
                    return False
                out.append({"row_id": row_id, "arm": "contradicted",
                            "case_id": cid, "why": why, "answer": mutated,
                            "evidence": ev, "claim_text": fixed,
                            "should_flag": True, "must_contradict": True,
                            "field": field, "validity": valid,
                            # the key the claim CITES and the value that key's
                            # own field label really gives, recorded so the row
                            # can be re-certified from the evidence without
                            # re-deriving which key the mutation chose
                            "cited": [key[0], key[1]], "rival": rival})
                return True

            # 1. SAME KEY — the cited key prints the rival; the claim states a
            #    figure whose digits occur in no held key at all.
            for key in here:
                rival, seg = printed[key][field]
                fake = _absent_amount(
                    next((s for s in runs if _digits(s) == _digits(rival)), runs[0]),
                    all_digits)
                if fake and emit(
                        "same-key", key, fake, rival,
                        f"the cited {key[0]} p.{key[1]} prints '{rival}' for "
                        f"{field} ({seg[:80]}); '{fake}' occurs in NO held key",
                        exempt=[fake]):
                    break

            # 2. TRANSPOSED — the value the cited key prints for another field.
            for key in here:
                rival, seg = printed[key][field]
                other = [(g, v) for g, (v, _s) in sorted(printed[key].items())
                         if g != field and _digits(v) != _digits(rival)]
                if other and emit(
                        "transposed", key, other[0][1], rival,
                        f"the cited {key[0]} p.{key[1]} prints '{other[0][1]}' "
                        f"for {other[0][0]} and '{rival}' for {field}, the field "
                        f"this claim names ({seg[:80]})"):
                    break

            # 3. WRONG PAGE — the same document's value for the same field on
            #    another key, cited to the key that disagrees.
            done = False
            for kp in here:
                rival, seg = printed[kp][field]
                for kq in here:
                    if kq == kp or kq[0] != kp[0]:
                        continue
                    val = printed[kq][field][0]
                    if _digits(val) == _digits(rival):
                        continue
                    if emit("wrong-page", kp, val, rival,
                            f"{kp[0]} prints '{rival}' for {field} on p.{kp[1]} "
                            f"({seg[:70]}) and '{val}' for the same field on "
                            f"p.{kq[1]}; the claim states the p.{kq[1]} figure "
                            f"and cites p.{kp[1]}",
                            exempt=[val]):
                        done = True
                        break
                if done:
                    break

            # 4. ELSEWHERE IN THE DOCUMENT — cite a key that prints the claim's
            #    figure but names no field, while another key of the same
            #    document prints the rival.
            done = False
            for kq in here:
                rival, seg = printed[kq][field]
                for kp, txt in items:
                    if kp[0] != kq[0] or kp == kq or kp[0] == verify.NOTES_DOC:
                        continue
                    if field in printed.get(kp, {}):
                        continue          # then it is shape 1/2/3, not this one
                    # money spans on BOTH sides: the whole amount the claim
                    # states is swapped for the whole amount the page prints,
                    # currency and scale word together, so the only thing left
                    # wrong with the claim is which field it attaches it to
                    mine = [s for s in _seed_money_spans(claim.text)
                            if claim.text.count(s) == 1]
                    val = next((s for s in _seed_money_spans(txt)
                                if _digits(s) != _digits(rival)), None)
                    if val and mine and emit(
                            "elsewhere", kp, val, rival,
                            f"the cited {kp[0]} p.{kp[1]} prints '{val}' but "
                            f"labels no {field}; p.{kq[1]} of the same document "
                            f"prints '{rival}' for {field} ({seg[:70]})",
                            target=mine[0]):
                        done = True
                        break
                if done:
                    break
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
    # The contradiction arm draws on no release split at all: its rows are
    # certified structurally from the evidence text, and a carrier claim's
    # recorded verdict plays no part in what makes the mutated claim false.
    # Restricting it to one side of the split would have cut it by two thirds
    # and bought nothing — both populations are recorded facts, identical in
    # every tree.
    rows += contradict(cases)
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
             r["should_flag"], r["answer"], r.get("validity"), ev,
             r.get("must_contradict", False), r.get("field"),
             r.get("cited"), r.get("rival")],
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
            "must_contradict": bool(row.get("must_contradict")),
            "field": row.get("field"), "cited": row.get("cited"),
            "rival": row.get("rival"),
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
        # THE CONTRADICTION-PATH COUNTERS. `flagged` collapses CONTRADICTED and
        # UNSUPPORTED, which is why deleting `_field_conflict` moved nothing:
        # a claim that loses its contradiction is usually still flagged, just
        # for a weaker reason and to a different downstream. These count the
        # status itself, so a CONTRADICTED -> UNSUPPORTED degradation is a
        # digit that moves even when no cell of TP/FP/FN/TN does.
        tp_hard = [r for r in subset
                   if r["should_flag"] and r["status"] == verify.CONTRADICTED]
        must = [r for r in subset if r.get("must_contradict")]
        lost = [r for r in must if r["status"] != verify.CONTRADICTED]
        promoted = [r for r in lost if not r["flagged"]]
        prec = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
        rec = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return {"counts": {"tp": len(tp), "fp": len(fp), "fn": len(fn), "tn": len(tn),
                           "fn_clearing_judge": len(fn_j),
                           "fp_contradicted": len(fp_hard),
                           "tp_contradicted": len(tp_hard),
                           "must_contradict": len(must),
                           "contradiction_lost": len(lost),
                           "contradiction_promoted": len(promoted)},
                "precision": prec, "recall": rec, "f1": f1,
                "false_negatives": [r["row_id"] for r in fn],
                "false_positives": [r["row_id"] for r in fp],
                "escapes_clearing_judge": [r["row_id"] for r in fn_j],
                "contradicted_but_should_not_be": [r["row_id"] for r in fp_hard],
                "contradiction_lost_rows": [r["row_id"] for r in lost],
                "contradiction_promoted_rows": [r["row_id"] for r in promoted]}

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

    def contra_shapes(scored):
        """Per shape of the contradiction arm: rows, flagged, and CONTRADICTED.

        Three numbers, not one: a shape whose rows are all still `flagged` but
        no longer `contradicted` has lost the path while leaving every matrix
        cell where it was, which is exactly how the blindness this arm closes
        looked from the outside."""
        notes = {
            "same-key": "conflict on the strictly-cited key",
            "transposed": "value verifies on the page; the FIELD disagrees",
            "wrong-page": "the document's own other figure for the field",
            "elsewhere": "cross-page scan (conflict-elsewhere-in-document)"}
        out = {}
        for r in scored:
            if r["arm"] != "contradicted":
                continue
            b = out.setdefault(r["why"] or "(none)",
                               {"rows": 0, "flagged": 0, "contradicted": 0,
                                "note": notes.get(r["why"], "")})
            b["rows"] += 1
            b["flagged"] += int(r["flagged"])
            b["contradicted"] += int(r["status"] == verify.CONTRADICTED)
        return out

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

    # The caution census over the SEEDED rows. The 66 recorded answers carry
    # zero contradicted verdicts, so their census — however it is filtered —
    # can never witness the contradiction path; these rows can.
    flag_census = Counter(f"{r['status']}:{f}" for r in scored for f in r["flags"]
                          if not f.startswith("invalid-citation:"))

    # Where the rival TEXT actually sits.  Re-derived from the evidence rather
    # than taken from the row, and printed every run: which key the rival is
    # read from does not change which branch of the verifier runs, but it does
    # change which text the arm can claim to cover.
    rival_kinds: Counter = Counter()
    for r in scored:
        if r["arm"] != "contradicted" or not r.get("cited"):
            continue
        rival_kinds["document-level key" if r["cited"][1] is None
                    or r["why"] == "elsewhere" else "numbered page"] += 1

    return {"rows": scored, "unmatched": unmatched,
            "flag_census": dict(flag_census),
            "rival_key_kinds": dict(rival_kinds),
            "status_census": {a: dict(Counter(
                r["status"] for r in scored if r["arm"] == a)) for a in ARMS},
            "overall": matrix(scored),
            "arms": {a: matrix([r for r in scored if r["arm"] == a]) for a in ARMS},
            "arm_sizes": dict(Counter(r["arm"] for r in scored)),
            "by_label": by_label,
            "fabrication_kinds": shapes("fabricated"),
            "repair_kinds": shapes("repaired"),
            "repair_by_shape": by_shape("repaired"),
            "fabricated_by_shape": by_shape("fabricated"),
            "contradicted_by_shape": contra_shapes(scored),
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
            "cautions": rep.cautions(),
            "user_cautions": rep.user_cautions(),
            "statuses": dict(Counter(v.status for v in rep.verdicts))}
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
            f"  FP|contra {c.get('fp_contradicted', 0):3d}"
            f"  TP|contra {c.get('tp_contradicted', 0):3d}"
            f"  must-contra {c.get('contradiction_lost', 0):3d}/"
            f"{c.get('must_contradict', 0):<3d}lost")


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
    # NOT `for arm, ...`: this loop used to rebind the `arm` parameter that
    # the verbose listing at the bottom filters on, so `--arm gold` silently
    # listed the repaired arm instead. Found while inspecting the new arm.
    for which, key in (("fabricated", "fabricated_by_shape"),
                       ("repaired", "repair_by_shape")):
        block = res.get(key) or {}
        verb = "still flagged" if which == "repaired" else "caught"
        print(f"  {which} shapes ({verb} / rows):")
        for shape in sorted(block):
            b = block[shape]
            print(f"    {shape:<18} {b['flagged']:>3} / {b['rows']:<3}"
                  + ("   <- deliberate FN region (rulings 3 and 7 deleted)"
                     if which == "repaired" and shape == "absence" else ""))
    con = res["overall"]["counts"]
    print()
    print("CONTRADICTION PATH")
    print(f"  rows that must come back CONTRADICTED: {con.get('must_contradict', 0)}"
          f"   got it right: {con.get('must_contradict', 0) - con.get('contradiction_lost', 0)}"
          f"   LOST: {con.get('contradiction_lost', 0)}"
          f"  (of which PROMOTED to a passing verdict: "
          f"{con.get('contradiction_promoted', 0)})")
    print("  TP|contra per arm: " + ", ".join(
        f"{a} {res['arms'][a]['counts'].get('tp_contradicted', 0)}" for a in ARMS))
    block = res.get("contradicted_by_shape") or {}
    print("  contradicted shapes (contradicted / flagged / rows):")
    for shape in sorted(block):
        b = block[shape]
        print(f"    {shape:<18} {b['contradicted']:>3} / {b['flagged']:>3} / {b['rows']:<3}"
              f"   {b['note']}")
    lost = res["overall"].get("contradiction_lost_rows") or []
    for i in sorted(lost)[:10]:
        r = next(x for x in res["rows"] if x["row_id"] == i)
        print(f"    ! LOST {i:<44} {r['status']:<12} "
              f"{'PROMOTED' if not r['flagged'] else 'degraded'}  {r['reason'][:52]}")
    csample = [r for r in sorted(res["rows"], key=lambda r: r["row_id"])
               if r["arm"] == "contradicted" and r.get("validity")]
    print(f"  contradiction validity (structural, {len(csample)}/"
          f"{res['arm_sizes'].get('contradicted', 0)} rows carry one); sample:")
    cstep = max(1, len(csample) // 6)
    for r in csample[::cstep][:6]:
        print(f"    {r['row_id']:<44} {r['validity'][:88]}")
    # A MEASURED LIMIT OF THE ARM, printed every run rather than left in a
    # docstring.  Which evidence key the rival is read from does not change
    # which branch of the verifier runs — the conflict scan reads the evidence
    # dict the same way for both — but a contradiction whose rival is printed
    # on a NUMBERED PAGE is a different text, and if none can be seeded the arm
    # cannot say it covers that text.
    kinds = Counter("document-level key" if r.get("cited") and r["cited"][1] is None
                    else "numbered page"
                    for r in res["rows"] if r["arm"] == "contradicted")
    print("  the key each row CITES: "
          + ", ".join(f"{k} {v}" for k, v in sorted(kinds.items())))
    print("  the key the RIVAL is printed on: " + ", ".join(
        f"{k} {v}" for k, v in sorted(
            (res.get("rival_key_kinds") or {}).items()))
        + "   <- MEASURED LIMIT: no contradiction whose rival is printed on a "
          "numbered page could be seeded from this recorded evidence")
    print("  flag census over the seeded rows (status:flag -> rows):")
    for k, v in sorted((res.get("flag_census") or {}).items()):
        print(f"    {k:<52} {v}")
    print("  verdict census per arm: " + "; ".join(
        f"{a} {res['status_census'][a]}" for a in ARMS))

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
              f"{nc.get('fp_contradicted', 0):<3d} "
              f"TP|contra {bc.get('tp_contradicted', 0):3d}->"
              f"{nc.get('tp_contradicted', 0):<3d} "
              f"must-contra-lost {bc.get('contradiction_lost', 0):3d}->"
              f"{nc.get('contradiction_lost', 0):<3d}  "
              f"R {_pct(b['recall'])}->{_pct(n['recall'])}")
    oldc, newc = baseline.get("flag_census") or {}, res.get("flag_census") or {}
    moved_flags = {k: (oldc.get(k, 0), newc.get(k, 0))
                   for k in set(oldc) | set(newc) if oldc.get(k, 0) != newc.get(k, 0)}
    print()
    print(f"  flag census over the seeded rows: {len(moved_flags)} entries moved")
    for k, (v0, v1) in sorted(moved_flags.items()):
        print(f"    {k:<52} {v0} -> {v1}")
    was = {r["row_id"]: r for r in baseline["rows"]}
    now = {r["row_id"]: r for r in res["rows"]}
    downgraded = [i for i in now if i in was and was[i]["status"] != now[i]["status"]]
    print(f"  rows whose STATUS changed (flagged or not): {len(downgraded)}")
    for i in sorted(downgraded)[:20]:
        r = now[i]
        bad = r.get("must_contradict") and r["status"] != verify.CONTRADICTED
        print(f"    {'CONTRADICTION LOST' if bad else 'status':<19} "
              f"[{r['arm']:<13}] {i:<44} {was[i]['status']} -> {r['status']}")
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
    p.add_argument("--max-lost-contradictions", type=int, default=None,
                   help="exit 1 if MORE than this many rows that must come back "
                        "CONTRADICTED do not (this tree's baseline is 1, named "
                        "in tests/test_verify.py:CONTRADICTION_ARM_MISSES)")
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
    lost = res["overall"]["counts"]["contradiction_lost"]
    if args.max_lost_contradictions is not None and lost > args.max_lost_contradictions:
        print(f"\nFAIL: {lost} claim(s) that must be CONTRADICTED are not "
              f"(allowed {args.max_lost_contradictions}): "
              f"{res['overall']['contradiction_lost_rows'][:8]}")
        return 1
    if args.fail_on_fn and res["overall"]["counts"]["fn"]:
        print(f"\nFAIL: {res['overall']['counts']['fn']} claim(s) that should be "
              f"flagged are not: {res['overall']['false_negatives'][:8]}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
