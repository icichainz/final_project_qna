"""Claim-level answer verification (plan step 5). A PURE DETECTOR.

The answer model is the only component in the pipeline that is allowed to
*write* facts, and it is the one component we cannot audit by construction.
This module audits its output after the fact — and only audits it:

    claims   = extract_claims(answer)                    # pure python
    evidence = build_evidence(hits, notes)               # what the turn held
    verdicts = classify(claims, evidence)                # deterministic first
    result   = verify_answer(answer, evidence)           # verdicts, never a rewrite

NOTHING HERE REWRITES ANSWER TEXT, and that is a decision, not an omission.
A constrained repair pass lived in this module through four waves of
calibration and was deleted at eac4c94. When it was finally allowed to act,
3 of the 5 rewrites it adopted had deleted verified evidence; the harm was
the most reproducible thing about it; and the cause was structural rather
than a matter of prompt wording or of one more gate — an adopt-if-clean rule
makes DELETING a claim the cheapest way to be clean, so every gate added to
stop deletion was a gate the next rewrite was selected to slip past. So the
pass is gone, ``verify_answer(...).answer`` is always the text it was given,
and the app has one fewer way to put words in front of a user.

What that leaves is what three waves of calibration made good, all of it
kept: extraction, deterministic classification, the conflict gate, the LLM
judge over the residue, cautions and sources.

Design rules, in the order they matter:

1. **Deterministic before LLM.** Every money figure, count, entity name and
   year is checked in pure python against the exact text the turn retrieved.
   A judge model is asked only about the residue: claims the string/number
   checks could not confirm *but whose cited evidence we actually hold*. A
   claim citing a page that was never retrieved needs no judge — that verdict
   is already certain.
2. **At most ONE LLM call per answer**, and it is skippable: one batched
   adjudication for the plausible residue. With no ``OPENAI_API_KEY`` the
   module still returns deterministic verdicts (``status='unverified-llm'``).
3. **The answer text is never touched.** Every caller gets back the string it
   passed in. A detector that can be talked into editing the answer is worse
   than no detector, and the only rule that cannot be talked out of is the one
   that has no code behind it to talk to.
4. **Never crash on model output.** A malformed citation ('[registry note in
   your context]') is a claim with an unresolvable pointer, not an exception.

Number normalization is ported from ``scripts/build_registry_v2.py`` — same
separator rules, same unit-word plausibility ceiling, same 'the printed unit
word and the mantissa contradict each other' case (28,654 million USD), so a
figure the registry publishes verifies identically here. The port is
deliberate: ``scripts/`` is not an importable package, and importing it runs
``load_dotenv()`` at module scope, which would inject API credentials into a
verifier whose whole point is that it also works without them.
"""
from __future__ import annotations

import dataclasses
import functools
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field as dc_field
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple)

from gcf_qna import config

EvidenceKey = Tuple[str, Optional[int]]
Evidence = Dict[EvidenceKey, str]

# Computed context blocks (registry lines, year notes, evidence matrix) that
# name no document live under this pseudo-document, so a note-level citation
# ('[Registry — 30 funding-proposal documents from 2020 ...]') has somewhere
# legitimate to point.
NOTES_DOC = "__notes__"
NOTES_KEY: EvidenceKey = (NOTES_DOC, None)


def note_scope_doc(doc: str) -> str:
    """The pseudo-document a note line's OWN printed pages are held under.

    A computed note prints its provenance — 'GCF funding requested:
    21,128,224 USD (p.6, A.8)', 'also as 49,151,817 USD (p.76, B.2(b))' — and
    the answer model is told to cite the page a row prints. The app
    (``chainlit_app._note_pages``) and the harness (``eval_answers``) both
    treat those printed pages as legal citation targets; this namespace is how
    the verifier holds the same fact without disturbing anything else.

    Namespaced, and DERIVED rather than stored (``note_scopes``). Two things
    it must not do, and the namespace is what stops both:

    * filing the line at ``(doc, page)`` would merge note text into a
      RETRIEVED page's evidence and hand every document-wide scan (``wide``,
      ruling 5's ``widened``, the per-key conflict scans,
      ``_reported_elsewhere``) a key it never had — a content widening riding
      along with a scope fix;
    * adding ANY key would change ``claims.evidence_keys``, which recorded
      runs carry and the release backfill asserts it can reproduce exactly.

    Under this pseudo-document the line is reachable only from a citation that
    names exactly that document and exactly that page, which is the whole of
    the rule.
    """
    return f"{NOTES_DOC}:{doc}"


def is_notes_doc(name: str) -> bool:
    """True for the notes pseudo-document and every per-document note scope.

    Everywhere the verifier asks 'is this evidence key a real corpus
    document?' — registry lookups, the ``_reported_elsewhere`` attribution
    pool, the snippet label — it must answer NO for both, or a note scope
    would be read as a document id.
    """
    return name == NOTES_DOC or name.startswith(NOTES_DOC + ":")


# ---------------------------------------------------------------------------
# numbers  (ported from scripts/build_registry_v2.py — see module docstring)
# ---------------------------------------------------------------------------

_NUM = r"\d{1,3}(?:[,.   ]\d{3})+(?:[.,]\d+)?(?!\d)|\d+(?:[.,]\d+)?"

_UNIT_MULT = {"million": 1e6, "millions": 1e6, "m": 1e6, "mn": 1e6,
              "billion": 1e9, "billions": 1e9, "bn": 1e9,
              "milliard": 1e9, "milliards": 1e9,
              "thousand": 1e3, "thousands": 1e3, "k": 1e3}
# a unit word is only applied when the bare number is small enough for it to be
# plausible: the GCF template prints "million USD ($)" as a *currency column
# label*, so "40,751,254 | million USD ($)" is 40.7m, not 40.7 trillion
_UNIT_CEILING = {1e6: 1e4, 1e9: 1e3, 1e3: 1e7}
_MAX_PLAUSIBLE = 5e9
_CUR_MAP = {"usd": "USD", "us$": "USD", "$": "USD", "u$s": "USD",
            "eur": "EUR", "euro": "EUR", "euros": "EUR", "€": "EUR"}

_AMOUNT_RE = re.compile(
    r"(?P<pre>USD|US\$|EUR|€|\$)?[ \t]{0,2}"
    r"(?P<num>" + _NUM + r")"
    r"(?P<sep>[ \t|,;:]{0,4})"
    r"(?P<unit>millions?|billions?|thousands?|milliards?|bn|m\b|k\b)?"
    r"(?P<sep2>[ \t|(]{0,4})(?P<post>USD|US\$|EUR|euros?|€|\$)?", re.I)

_CUR_NEARBY = re.compile(r"(USD|US\$|EUR|euros?|€|\$)", re.I)
_NOISE_BEFORE = re.compile(
    r"(?:enter\s+(?:number|amount)|e\.?\s?g\.?|example|page|version|v\.)[^\d]{0,6}$", re.I)
_GLUED = re.compile(r"[A-Za-z0-9]$")
# an identifier's digits are not a quantity: 'GCF/B.27/02/Add.12' must not
# read as three amounts, and 'p. 45' is a pointer, not a figure
_CODE_BEFORE = re.compile(r"(?:[A-Za-z]\.|/)[ \t]?$")
_NOT_MONEY_AFTER = re.compile(
    r"^\s{0,3}(?:%|percent|pour ?cent|years?|ans\b|months?|mois\b|days?|weeks?|"
    r"tco2|tco₂|t\s?co2|ha\b|km|people|persons|beneficiaries|households|"
    r"of\s+the\s+total|pages?)", re.I)


def to_number(tok: str) -> Optional[float]:
    """'49,751,264' -> 49751264.0 ; '46,10' -> 46.1 ; '18,5' -> 18.5.

    US-first separator rules, which is what the corpus prints, with the
    French decimal comma falling out of the last branch: a single separator
    followed by anything other than exactly three digits is a decimal point.
    """
    s = tok.strip().replace(" ", " ").replace(" ", " ")
    s = re.sub(r"(?<=\d)[ ](?=\d)", "", s)              # space-grouped thousands
    if not re.fullmatch(r"\d[\d.,]*", s):
        return None
    if "," in s and "." in s:
        cut = max(s.rfind(","), s.rfind("."))
        s2 = re.sub(r"[.,]", "", s[:cut]) + "." + re.sub(r"[.,]", "", s[cut + 1:])
    else:
        sep = "," if "," in s else ("." if "." in s else None)
        if sep is None:
            s2 = s
        else:
            groups = s.split(sep)
            grouped = all(len(g) == 3 for g in groups[1:])
            if len(groups) > 2 and grouped:
                s2 = "".join(groups)                     # 1,234,567 / 1.234.567
            elif len(groups) == 2 and grouped and sep == ",":
                s2 = "".join(groups)                     # 1,234 (US thousands)
            else:
                s2 = groups[0] + "." + "".join(groups[1:])
    try:
        return float(s2)
    except ValueError:
        return None


def granularity(tok: str, mult: float) -> float:
    """Smallest amount the printed token can distinguish, in absolute units.

    '18.5' with a million unit distinguishes 0.1m = 100 000; '18,500,000'
    distinguishes 1. Two figures agree when they differ by less than the
    coarser of the two printed precisions, so a rounded restatement is not
    reported as a contradiction while two fully printed figures that differ
    by 2,000,000 are.
    """
    s = re.sub(r"(?<=\d)[   ](?=\d)", "", tok.strip())
    dec = 0
    if "," in s and "." in s:
        dec = len(s) - max(s.rfind(","), s.rfind(".")) - 1
    else:
        sep = "," if "," in s else ("." if "." in s else None)
        if sep:
            groups = s.split(sep)
            grouped = all(len(g) == 3 for g in groups[1:])
            if not (len(groups) > 2 and grouped) and not (
                    len(groups) == 2 and grouped and sep == ","):
                dec = len("".join(groups[1:]))
    return (10.0 ** -dec) * mult


def _decimal_comma_reading(tok: str, mult: float) -> Optional[Tuple[float, float]]:
    """(value, grain) for a clashing token read with a DECIMAL comma.

    The corpus prints European decimal marks — 'USD 28,025 million',
    'a grant funding of USD 26,958 million' — where an answer writes
    'USD 28.025 million'. Read US-first, '28,025' is twenty-eight thousand,
    which with a million unit word is 2.8e10 and therefore implausible; that
    implausibility is exactly the ``clash`` flag, and it is also the signal
    that the separator was a decimal point. This reading is offered ONLY for
    a token the unit ceiling already rejected, so a plainly grouped figure
    ('40,751,254') is never re-read (claim-c83fbe25).
    """
    s = re.sub(r"(?<=\d)[   ](?=\d)", "", (tok or "").strip())
    m = re.fullmatch(r"(\d{1,3}),(\d{3})", s)
    if not m:
        return None
    return (float(f"{m.group(1)}.{m.group(2)}") * mult, (10.0 ** -3) * mult)


@dataclass(frozen=True)
class Amount:
    """One printed figure, normalized. ``value`` is None when the mantissa and
    the printed unit word cannot both be true ('28,654 million USD'): the
    scale is then unknown and only ``bare`` (the mantissa) is comparable.

    ``alt`` is that same token read with a decimal comma instead of a
    thousands comma — the only other reading the printed characters allow —
    and is set only when ``value`` is None."""
    raw: str
    value: Optional[float]
    bare: float
    currency: Optional[str]
    unit: Optional[str]
    grain: float
    at: int = 0
    num: str = ""
    alt: Optional[float] = None
    alt_grain: float = 0.0

    @property
    def clash(self) -> bool:
        return self.value is None


def iter_amounts(text: str, money_only: bool = False) -> Iterable[Amount]:
    """Every figure in ``text``. ``money_only`` applies the registry's money
    floor (a bare '30' next to a financing label is table furniture)."""
    for m in _AMOUNT_RE.finditer(text or ""):
        num = m.group("num")
        head = m.start() if m.group("pre") else m.start("num")
        before = text[max(0, head - 24):head]
        after = text[m.end("num"):m.end("num") + 24]
        if _NOISE_BEFORE.search(before) or _GLUED.search(before):
            continue
        if _CODE_BEFORE.search(before) and not m.group("pre"):
            continue
        if _NOT_MONEY_AFTER.match(after):
            continue
        val = to_number(num)
        if val is None or val == 0:
            continue
        unit_tok = (m.group("unit") or "").lower().strip()
        mult = _UNIT_MULT.get(unit_tok, 1.0)
        clash = mult > 1 and (val >= _UNIT_CEILING.get(mult, 0)
                              or val * mult > _MAX_PLAUSIBLE)
        cur_tok = (m.group("pre") or m.group("post") or "").lower()
        if not cur_tok:
            m2 = _CUR_NEARBY.search(text[m.end():].split("\n", 1)[0][:40])
            cur_tok = m2.group(1).lower() if m2 else ""
        if not clash:
            if not cur_tok and mult == 1.0 and re.fullmatch(r"(19|20)\d{2}", num):
                continue                       # a bare 4-digit integer is a year
            if money_only and mult == 1.0 and val < 1e4 and not cur_tok:
                continue
        raw = text[m.start():m.end()].strip(" \t|,;:(")
        alt = _decimal_comma_reading(num, mult) if clash else None
        yield Amount(
            raw=raw or num,
            value=None if clash else round(val * mult, 2),
            bare=round(val, 6),
            currency=_CUR_MAP.get(cur_tok if cur_tok == "us$" else cur_tok.rstrip("s")),
            unit={1e6: "million", 1e9: "billion", 1e3: "thousand"}.get(mult),
            grain=granularity(num, 1.0 if clash else mult),
            at=m.start("num"),
            num=num,
            alt=alt[0] if alt else None,
            alt_grain=alt[1] if alt else 0.0)


_COMMA3_RE = re.compile(r"\d{1,3},\d{3}")


def _decimal_comma_convention(got: List[Amount]) -> List[Amount]:
    """Spread a decimal-comma reading across one text that demonstrates it.

    'Co-financing: USD 28,025 million ... USD 26,958 million from APFC; and
    ... USD 1,066 million from local government' prints one convention three
    times. Two of the three are self-evidently decimal commas — 28,025 million
    would be 2.8e10 — and the third, 1,066 million, is merely implausible
    rather than impossible, so the unit ceiling lets it through as 1.066
    BILLION and the answer's 'USD 1.066 million' reads as a thousandfold
    error (claim-c83fbe25).

    The convention is therefore taken from the text itself: only when a blob
    already contains a 'd,ddd <unit>' figure whose US reading is impossible do
    its other 'd,ddd <unit>' figures gain the decimal-comma reading as an
    ALTERNATIVE. A text that never demonstrates the convention never gets it,
    and no reading is ever replaced — only added.
    """
    if not any(a.clash and a.alt is not None for a in got):
        return got
    out: List[Amount] = []
    for a in got:
        if a.alt is None and a.unit and _COMMA3_RE.fullmatch(a.num or ""):
            alt = _decimal_comma_reading(a.num, _UNIT_MULT.get(a.unit, 1.0))
            if alt:
                a = dataclasses.replace(a, alt=alt[0], alt_grain=alt[1])
        out.append(a)
    return out


def amounts(text: str, money_only: bool = False) -> List[Amount]:
    return _decimal_comma_convention(list(iter_amounts(text, money_only)))


def _money_like_amount(a: Amount) -> bool:
    """A figure that carries money semantics: a currency, a scale word, a
    self-contradictory scale, or a magnitude no table index reaches."""
    return bool(a.currency or a.unit or a.clash or (a.value or 0) >= 1e4)


def amount_matches(a: Amount, b: Amount) -> bool:
    """Do two printed figures state the same thing?

    Currencies that are both printed and differ are never a match: EUR 87
    million and USD 87 million are two facts, and conflating them is exactly
    the cross-currency error the answer prompt spends a paragraph forbidding.
    """
    if a.currency and b.currency and a.currency != b.currency:
        return False
    if a.value is not None and b.value is not None:
        tol = max(a.grain, b.grain) * 0.5 + 1e-6
        if abs(a.value - b.value) <= tol:
            return True
    # the corpus's decimal comma: 'USD 28,025 million' and 'USD 28.025
    # million' are one figure printed two ways. Strictly ADDITIVE — the
    # alternative reading is tried after the printed one, never instead of it
    for x, y in ((a, b), (b, a)):
        if x.value is not None and y.alt is not None:
            tol = max(x.grain, y.alt_grain) * 0.5 + 1e-6
            if abs(x.value - y.alt) <= tol:
                return True
    if a.value is not None and b.value is not None:
        return False
    # a scale that is self-contradictory on both sides: the mantissa is still
    # comparable, and '28,654 million' vs '26,654 million' still disagree
    tol = max(granularity(a.raw, 1.0), granularity(b.raw, 1.0)) * 0.5 + 1e-6
    return abs(a.bare - b.bare) <= tol


# ---------------------------------------------------------------------------
# citations
# ---------------------------------------------------------------------------

_DOC_RE = r"[0-9]{1,3}_[\w.\-]+"
_BRACKET_RE = re.compile(r"\[([^\[\]]{1,400})\]")
_INSIDE_RE = re.compile(r"(" + _DOC_RE + r")|\bpp?\.?\s*(\d{1,3})\b"
                        r"|\b(cover\s+pages?|pages?\s+de\s+garde|registry|registre)\b",
                        re.I)
_PAGE_RANGE_RE = re.compile(r"\s*[-–—]\s*(\d{1,3})")


@dataclass(frozen=True)
class Citation:
    """One (document, page) pointer parsed out of a bracket.

    ``kind``: 'page' (an explicit page), 'cover' (registry / cover-page scope,
    page None), 'doc' (a bare document id), 'note' (a bracket naming no
    document at all — a pointer at the computed notes), 'malformed' (a bracket
    we could not read as any of those).
    """
    doc: Optional[str]
    page: Optional[int]
    kind: str
    raw: str

    @property
    def key(self) -> EvidenceKey:
        return (self.doc or NOTES_DOC, self.page)


def parse_citations(text: str) -> List[Citation]:
    """Every citation in ``text``, in order.

    A bracket may chain several ('[docA, p. 5; docB, p. 6]') and a page
    belongs to the NEAREST PRECEDING document id, never to the bracket's
    first: attributing every page to the first id is how a valid citation
    gets reported as invented (observed live on gpt-5.2 output, and the same
    rule chainlit_app._invalid_citations already applies).
    """
    out: List[Citation] = []
    for br in _BRACKET_RE.finditer(text or ""):
        inner = br.group(1)
        cur: Optional[str] = None
        got_any = False
        pending_doc = False          # a doc id not yet paired with a page
        for m in _INSIDE_RE.finditer(inner):
            doc, page = m.group(1), m.group(2)      # group(3): 'cover pages'
            if doc:
                if pending_doc and cur:
                    out.append(Citation(cur, None, "doc", inner))
                cur, pending_doc, got_any = doc, True, True
            elif page:
                try:
                    n = int(page)
                except ValueError:      # unreachable via the regex, cheap to keep
                    continue
                pages = [n]
                rng = _PAGE_RANGE_RE.match(inner, m.end())
                if rng:
                    hi = int(rng.group(1))
                    if 0 < hi - n <= 20:
                        pages = list(range(n, hi + 1))
                for p in pages:
                    out.append(Citation(cur, p, "page" if cur else "malformed", inner))
                pending_doc, got_any = False, True
            else:            # 'cover pages' / 'registry' — doc-level scope
                out.append(Citation(cur, None, "cover" if cur else "note", inner))
                pending_doc, got_any = False, True
        if pending_doc and cur:
            out.append(Citation(cur, None, "doc", inner))
        if not got_any and _looks_like_pointer(inner):
            out.append(Citation(None, None, "note", inner))
    return out


_POINTER_WORDS = re.compile(
    r"registry|registre|note|context|prompt|matrix|matrice|excerpt|extrait|corpus",
    re.I)


def _looks_like_pointer(inner: str) -> bool:
    """A bracket with no document id is a citation only when it reads like
    one ('[registry note in your context]'), not when it is markdown."""
    return bool(_POINTER_WORDS.search(inner)) and len(inner) < 400


def cited_sources(final_answer: str) -> List[Tuple[str, Optional[int]]]:
    """The (doc, page) pairs a finished answer actually cites, deduped, in
    order — the plan's 'display only sources actually cited by the final
    verified answer'. Note-level brackets carry no document and are omitted."""
    out: List[Tuple[str, Optional[int]]] = []
    for c in parse_citations(final_answer):
        if not c.doc:
            continue
        k = (c.doc, c.page)
        if k not in out:
            out.append(k)
    return out


# ---------------------------------------------------------------------------
# claims
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^\s*(?:[-*•‣]|\d+[.)])\s+")
# a line carrying citations and nothing else ('[doc, p. 4]' under a bullet list)
_CITE_ONLY_RE = re.compile(r"^\s*(?:\[[^\]]*\][\s.,;]*)+$")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s")
_TABLE_RULE_RE = re.compile(r"^[\s|:\-—+]+$")

# tokens whose '.' is not a sentence end; masked before sentence splitting
_MASK_RE = re.compile(
    r"\[[^\]]*\]"                       # whole citation brackets
    r"|\bpp?\.\s*\d+"                   # p. 5 / pp. 12
    r"|\b[A-Z]\.\d+(?:\.\d+)*"          # A.8, B.27
    r"|\bAdd\.\s*\d+|\bNo\.\s*\d+"
    r"|\be\.g\.|\bi\.e\.|\bcf\.|\bapprox\.|\bvs\.|\betc\.|\bU\.S\."
    r"|\bMr\.|\bMs\.|\bDr\.|\bSt\.|\bp\.ex\.|\bc\.-à-d\."
    r"|\d[\d.,  ]*\d|\d"
    # an ellipsis inside an elided list ('FP124 … FP153') is not a
    # sentence end: splitting there left the trailing citation on the
    # fragment after it, and the fact before it looked uncited
    r"|\.\.\.|…")

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])[\s]+(?=[\"“«\*_A-ZÀ-ÖØ-Þ0-9])")

# Prose that carries no checkable assertion: connectives, meta-commentary and
# refusals. Checked BEFORE the claim triggers, because a refusal sentence is
# usually stuffed with the very identifiers and figures the triggers look for
# ("Retrieval did not surface the FP151 and FP152 ... documents").
_GLUE_RE = re.compile(
    r"\b(?:in summary|in short|to summari[sz]e|overall|in conclusion|"
    r"en r[ée]sum[ée]|pour r[ée]sumer|en bref|"
    # 'i can answer' is the alternation this list was short of: the
    # adjudicated row `claim-2bca31faa865e0d00c91c737` is the conversational
    # offer 'tell me which one and I can answer precisely to the extent the
    # excerpts include the relevant Board-meeting field', labelled
    # `not_a_claim` (glue) — and it is glue, not a lead-in, so the form test
    # is the wrong place to fix it and the verb list is the right one.
    r"if you want|let me know|i can (?:answer|compare|provide|list|run)|"
    r"happy to|feel free)\b"
    # meta-commentary about what the evidence does or does not carry: the
    # sentence is about the retrieval, not about the corpus
    r"|\b(?:what|the only thing|all that)\b[^.]{0,40}\b(?:is |are )?supported by\b",
    re.I)

_REFUSAL_RE = re.compile(
    r"(?:retrieval|the search|the excerpts?|the context|the retrieved excerpts?)"
    r"[^.]{0,80}?\b(?:did not|does not|do not|don't|doesn't|cannot|can't|"
    r"never)\b"
    r"|\b(?:i|we)\s+(?:can(?:not|'t|’t)|could not|couldn't|am unable|are unable)"
    r"|\bso\s+i\s+can(?:not|'t|’t)"
    r"|\b(?:no|not)\s+(?:figure|amount|value|page|excerpt)s?\s+(?:is\s+|are\s+)?"
    r"(?:stated|given|provided|available|surfaced)"
    r"|\bnot stated in the (?:excerpts?|context)"
    r"|\b(?:do(?:es)? not|don't|doesn't)\s+(?:explain|indicate|identify|specify|"
    r"contain|state|say|allow|support)"
    r"|\bla r[ée]cup[ée]ration n['’]a pas|\bles extraits ne\b|\baucun extrait\b"
    r"|\bje ne (?:peux|puis) pas\b|\bne permet(?:tent)? pas\b", re.I)

# number followed by something that makes it a measured quantity
_COUNT_RE = re.compile(
    # up to two words may sit between the figure and its noun ('30 funding
    # proposals', '18.5 million USD grant')
    r"\b\d[\d.,  ]*\s*(?:[a-zà-öø-þ][\w\-]*\s+){0,2}"
    r"(?:%|percent|pour ?cent|"
    r"years?|ans\b|months?|mois\b|"
    r"million|billion|thousand|milliards?|millions?|"
    r"proposals?|propositions?|documents?|projects?|projets?|"
    r"countr(?:y|ies)|pays\b|beneficiaries|b[ée]n[ée]ficiaires|"
    r"tco2|tco₂|tonnes?|ha\b|km\b|people|households)", re.I)
_BIGNUM_RE = re.compile(r"\b\d{1,3}(?:[,.   ]\d{3})+\b|\b\d{5,}\b")
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_BOARD_RE = re.compile(r"\bB\.?\s?\d{2}\b")
_EXISTENCE_RE = re.compile(
    r"\b(?:there (?:is|are|were|was)|the corpus (?:contains|holds|has)|"
    r"exists?|does not exist|n'existe pas|le corpus (?:contient|compte)|"
    r"no (?:funding )?proposals?|aucune? proposition)\b", re.I)

_ENT_STOP = {
    "gcf", "green climate fund", "fund", "registry", "registre", "annex",
    "annexe", "board", "note", "notes", "excerpt", "excerpts", "extrait",
    "extraits", "document", "documents", "context", "prompt", "matrix",
    "usd", "eur", "us", "us$", "the", "a", "an", "on", "among", "and",
    "recommendation", "recommandation", "decision", "d[ée]cision", "page",
    "pages", "corpus", "total", "grant", "loan", "equity", "guarantee",
    "registry metadata", "registry metadata line", "some", "other", "another",
    "i", "if", "it", "this", "these", "those", "fp", "add", "p", "pp", "b",
    "none", "no", "yes", "note that", "among",
    "million", "millions", "billion", "billions", "thousand", "milliard",
    "milliards", "financing", "funding", "financement",
}
_ENT_DROP_RE = re.compile(
    r"^(?:FP\s?-?\d{1,3}|B\.?\d{1,2}(?:[./]\d{1,2})*|GCF/[\w./]+|"
    r"[IVXLC]{1,6}|[0-9][\w.\-]*)$", re.I)
# An identifier anywhere inside a candidate makes the whole candidate a
# pointer: 'Funding Proposal FP173' is a reference to a document, and
# demanding that exact string in the page reports a correct answer as
# unsupported. The identifier is verified by the citation machinery instead.
_ENT_HAS_ID_RE = re.compile(
    r"\b(?:FP\s?-?\d{1,3}|B\.\s?\d{1,2}(?:[./]\d{1,2})*|GCF/[\w./]+"
    r"|\d{1,3}_[\w.\-]{4,})\b", re.I)
# single capitalized words that are grammar or template furniture, never names
_ENT_GENERIC = {"unlocking", "valuing", "including", "funding", "financing",
                "proposal", "proposals", "facility", "programme", "program",
                "project", "projects", "activity", "equity", "annex",
                "recommendation", "summary", "however", "therefore", "based",
                # form-field words. '**Funding proposal ID / name:** FP86'
                # bolds the LABEL, and demanding the page print the label as
                # written reported the value beside it as unsupported
                # (claim-87ad9dbb). Checked over the WHOLE candidate, so a
                # name containing one of these ('Green Cities Facility')
                # still stands.
                "id", "ids", "name", "names", "title", "titles", "label",
                "section", "field", "amount", "value", "number", "type",
                "entity", "entities", "date", "dates", "status", "nom",
                "titre", "montant"}


def _all_generic(cand: str) -> bool:
    """Is every word of this candidate template furniture?

    'Funding proposal ID / name' is a form field's LABEL: each of its words is
    generic, and no page prints the label as the answer spells it. 'Green
    Cities Facility' contains one generic word and is still a name, so the
    test is over the whole candidate, never over its parts.
    """
    words = [w for w in re.split(r"[\s/]+", norm_text(cand)) if w]
    return bool(words) and all(w in _ENT_GENERIC or w in _ENT_STOP
                               or w in _CONNECTORS for w in words)


def _deaccent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


#: Every apostrophe- and quote-shaped codepoint the corpus and the answer
#: model print, folded onto the ASCII pair. Row `id-fp203-objective` failed on
#: `Colombia’s` (U+2019) against a page printing `Colombia`; the possessive
#: is stripped elsewhere (``_depossess``), and this table exists so that WHICH
#: apostrophe was typed can never decide a verdict on its own.
_QUOTE_FOLD = {ord(c): "'" for c in "‘’‚‛′ʼʹ"}
_QUOTE_FOLD.update({ord(c): '"' for c in "“”„‟"})


def norm_text(s: str) -> str:
    """Fold case, accents, markdown emphasis and punctuation runs, so that a
    substring test compares words rather than typography."""
    s = _deaccent((s or "").lower()).translate(_QUOTE_FOLD)
    s = re.sub(r"[*_`]", "", s)
    s = re.sub(r"[^\w'&/]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _strip_citations(text: str) -> str:
    return _BRACKET_RE.sub(" ", text or "")


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_QUOTE_RE = re.compile(r"[“\"«]\s*([^”\"»\n]{3,120}?)\s*[”\"»]")
_ACRONYM_RE = re.compile(r"\b([A-Z]{2,8})\b")
_CAPRUN_RE = re.compile(
    r"\b([A-ZÀ-ÖØ-Þ][\w’'\-]+"
    r"(?:\s+(?:of|and|for|the|de|du|des|la|le|les|et|pour)\s+[A-ZÀ-ÖØ-Þa-zà-öø-þ][\w’'\-]+"
    r"|\s+[A-ZÀ-ÖØ-Þ][\w’'\-]+)*)")


_CONNECTORS = {"of", "and", "for", "the", "in", "on", "to", "a", "an", "by",
               "de", "du", "des", "la", "le", "les", "et", "pour", "dans", "au",
               "aux", "d'", "l'"}


# ---------------------------------------------------------------------------
# CLASS 2 — possessives.  Row `id-fp203-objective`: the extractor produced the
# candidate `Colombia’s` from "to support Colombia’s climate goals", and the
# cited cover-page evidence prints `Colombia`. An English possessive is an
# INFLECTION of the name, not part of it, so it is offered as an extra spelling
# of the same candidate — never as a replacement, and never anywhere but the
# end of the candidate.
#
# The gate is the APOSTROPHE. A bare trailing 's' is a plural or simply the
# last letter of the name ('Andes', 'Barbados', 'Comoros'), and stripping it
# would let a name match a shorter unrelated one under the substring test that
# follows. Pinned by ``test_a_bare_trailing_s_is_not_a_possessive``.
# ---------------------------------------------------------------------------
_POSSESSIVE_RE = re.compile(r"(?:['’‘‚‛′ʼʹ]s|s['’‘‚‛′ʼʹ])$", re.I)


def _depossess(cand: str) -> Optional[str]:
    """``Colombia’s`` -> ``Colombia``; ``Andes`` -> None.

    Only the LAST word of the candidate is de-inflected, and only when it
    carries an apostrophe: 'Colombia’s climate goals' is not a name, and a
    candidate whose interior words were also stripped would stop being the
    string the answer actually asserted.
    """
    words = (cand or "").split()
    if not words:
        return None
    tail = _POSSESSIVE_RE.sub("", words[-1])
    if tail == words[-1] or not tail:
        return None
    out = " ".join(words[:-1] + [tail]).strip()
    return out or None


# ---------------------------------------------------------------------------
# CLASS 5 — cross-lingual proper nouns.  Rows `fr-disc-thai-rice` (both
# claims): a French answer over English pages, where the only unmatched
# elements were the exonym `Thaïlande` and the cognate `Autorité` while the
# cited page prints `Thailand` and `national designated authority`.
#
# This is a CLOSED TABLE, not a translator: every key is one French printed
# form and every value is the one English printed form of the SAME referent.
# The country half is the corpus's own country list (``registry.load()``),
# restricted to the names whose French form still differs after accents are
# folded — 'Sénégal', 'Bénin' and 'Côte d’Ivoire' need no entry because
# ``norm_text`` already deaccents them onto the English spelling. A key is
# admitted only when its English side is a country this corpus actually
# records, so the table cannot introduce a referent the corpus never had.
#
# Nothing here is fuzzy and nothing here is compositional: the lookup is over
# the WHOLE candidate and the key must match it exactly. 'Taïwan' does not
# become 'Thailand' because it is not a key.
#
# WORD-FOR-WORD SUBSTITUTION WAS BUILT HERE AND THEN REMOVED. Rewriting the
# French words INSIDE a longer candidate turned
# 'Ministry of Agriculture and Cooperatives (MOAC) – Thaïlande' into a string
# the cited page prints, and that cleared adjudicated defect
# `claim-6c4788ddf1da438d7049706e` (label `missing_retrieval_evidence`: the
# answer names the wrong proposal). Translating a whole name the corpus has an
# English spelling for is a spelling change; translating a fragment of a longer
# assertion rebuilds the assertion. Pinned by
# ``test_a_cross_lingual_variant_is_an_exact_table_hit`` and
# ``test_a_french_word_inside_a_longer_name_is_not_translated``.
# ---------------------------------------------------------------------------
_FR_EN_NAMES = {
    # countries recorded in this corpus whose French exonym survives deaccenting
    "thailande": "thailand",            # fr-disc-thai-rice (both rows)
    "ethiopie": "ethiopia", "egypte": "egypt", "maroc": "morocco",
    "cambodge": "cambodia", "ouganda": "uganda", "colombie": "colombia",
    "bresil": "brazil", "perou": "peru", "mexique": "mexico",
    "inde": "india", "indonesie": "indonesia", "tanzanie": "tanzania",
    "zambie": "zambia", "mongolie": "mongolia", "barbade": "barbados",
    "mauritanie": "mauritania", "tunisie": "tunisia", "cameroun": "cameroon",
    "argentine": "argentina", "tadjikistan": "tajikistan",
    "jamaique": "jamaica", "namibie": "namibia", "chili": "chile",
    "ouzbekistan": "uzbekistan", "fidji": "fiji", "comores": "comoros",
    "albanie": "albania", "armenie": "armenia", "georgie": "georgia",
    "serbie": "serbia", "grenade": "grenada", "gambie": "gambia",
    "malaisie": "malaysia", "liban": "lebanon", "palaos": "palau",
    "soudan": "sudan", "kirghizistan": "kyrgyzstan", "moldavie": "moldova",
    "dominique": "dominica", "bhoutan": "bhutan", "chine": "china",
    "turquie": "turkey", "bulgarie": "bulgaria", "bolivie": "bolivia",
    "erythree": "eritrea", "jordanie": "jordan", "somalie": "somalia",
    "tchad": "chad", "guinee": "guinea", "equateur": "ecuador",
    "haiti": "haiti", "birmanie": "myanmar",
    # multi-word exonyms, matched on the whole candidate
    "afrique du sud": "south africa",
    "republique dominicaine": "dominican republic",
    "macedoine du nord": "north macedonia",
    "bosnie herzegovine": "bosnia and herzegovina",
    "papouasie nouvelle guinee": "papua new guinea",
    "iles salomon": "solomon islands",
    "iles marshall": "marshall islands",
    "iles cook": "cook islands",
    "antigua et barbuda": "antigua and barbuda",
    "trinite et tobago": "trinidad and tobago",
    "guinee bissau": "guinea bissau",
    "sainte lucie": "saint lucia",
    "republique democratique du congo": "democratic republic of congo",
    # the one institutional cognate an audited row needs
    "autorite": "authority",            # fr-disc-thai-rice:109b8222296e
}


def _crosslingual(cand: str) -> Optional[str]:
    """The English printed form of a French candidate, or None.

    The WHOLE candidate must be a key ('thailande', 'afrique du sud'). A
    candidate the table does not name is checked exactly as it was written.
    """
    return _FR_EN_NAMES.get(norm_text(cand))


def _entity_variants(cand: str) -> List[str]:
    """One entity, as its printed forms: 'International Union ... (IUCN)' is
    the same fact as 'IUCN', so either spelling in the evidence supports it.

    A long verbatim title also yields its first eight words: extraction
    reflows lines, and demanding a 30-word title character-for-character
    would report a real title as unsupported.
    """
    cand = re.sub(r"\s+", " ", cand.strip(" .,;:*_\"“”«»")).strip()
    words = cand.split()
    while words and words[0].lower() in _CONNECTORS:
        words = words[1:]
    while words and words[-1].lower() in _CONNECTORS:
        words = words[:-1]
    cand = " ".join(words)
    if not cand:
        return []
    out = [cand]
    m = re.match(r"^(.*?)\s*\(([^)]{2,60})\)\s*$", cand)
    if m:
        out += [m.group(1).strip(), m.group(2).strip()]
    if len(words) > 8:
        out.append(" ".join(words[:8]))
    # CLASS 2 / CLASS 5: two more SPELLINGS of the same candidate. Both are
    # appended, never substituted — a claim keeps every form it already had,
    # so neither can remove a check, only add a way of satisfying one.
    for extra in (_depossess(cand), _crosslingual(cand)):
        if extra:
            out.append(extra)
    return [v for v in dict.fromkeys(out) if v]


def _entity_core(cand: str) -> str:
    """The candidate minus digits, furniture words and identifiers.

    'USD 150 million' and 'GCF FP173' are pointers dressed as names; what is
    left of them after this is nothing, and nothing is not an entity claim.
    """
    toks = [t for t in re.split(r"\s+", cand.strip()) if t]
    keep = [t for t in toks
            if norm_text(t) and norm_text(t) not in _ENT_STOP
            and not re.search(r"\d", t)          # 'US$116', 'FP152:' are pointers
            and not _ENT_DROP_RE.match(t.strip("()[],.;:"))
            and re.search(r"[A-Za-zÀ-ÖØ-Þà-öø-þ]{2}", t)]
    return " ".join(keep)


def _looks_like_name(cand: str, quoted: bool = False) -> bool:
    """Is this candidate a name, or a sentence the model happened to bold?

    Names are short and mostly capitalized. 'accredited entity is IFAD' and
    'no GCF funding proposals approved in 2014' are assertions whose checkable
    part (IFAD, 2014) is extracted separately anyway.
    """
    words = [w for w in re.split(r"\s+", cand.strip()) if w]
    if not words:
        return False
    if not quoted and len(words) > 12:
        return False
    body = [w for w in words if w.lower() not in _CONNECTORS
            and re.search(r"[A-Za-zÀ-ÖØ-Þà-öø-þ]", w)]
    if not body:
        return False
    caps = sum(1 for w in body if re.match(r"[\"“«(]?[A-ZÀ-ÖØ-Þ]", w))
    return caps / len(body) >= 0.5



def _trim_run(cand: str) -> str:
    """A capitalized run, cut back to its last capitalized word.

    ``_CAPRUN_RE`` lets a connective pull the next word in, so that
    'International Union for Conservation of Nature' survives whole. The same
    rule swallows the sentence's grammar after a name: 'one with Pegasus and
    one with IUCN' yields the candidate 'Pegasus and one', which no page
    prints and which reported a correct sentence as unsupported
    (claim-e79ef060). The name ends where the capitals end.
    """
    words = cand.split()
    last = max((i for i, w in enumerate(words)
                if re.match(r"[\"\u201c\u00ab(]?[A-Z\u00c0-\u00d6\u00d8-\u00de]", w)),
               default=-1)
    return " ".join(words[:last + 1]) if last >= 0 else cand


# ---------------------------------------------------------------------------
# CLASS 4 — a DENIED term is not an asserted one.  Row `abs-antarctica`: the
# answer's rider reads "and other non-Antarctica infrastructure/sector
# activities", and the extractor lifted `Antarctica` out of it as a name the
# answer asserts. It asserts the opposite, and the case exists precisely
# because Antarctica is absent from the corpus.
#
# TWO EARLIER ROUNDS DIED HERE AND THIS IS NOT WHAT THEY DID. A character
# window and a clause split both operated at MATCH time — they excused a term
# the evidence did not contain because a negator stood near it, which is how
# 'None of the excerpts mention Antarctica; it is a Wakanda Development Bank
# project' shipped verified. This operates at EXTRACTION time and turns on
# MORPHOLOGY, not distance: only a term carrying a BOUND negative prefix
# ('non-Antarctica') is dropped, only that term, and only when EVERY printing
# of it in the unit is bound the same way. A second name in the same sentence,
# clause or window is untouched, and a term the answer also asserts positively
# somewhere in the unit is untouched. Pinned by the two adversarial tests
# ``test_a_bound_negative_prefix_does_not_excuse_its_neighbours`` and
# ``test_a_positive_printing_defeats_the_negative_prefix``.
# ---------------------------------------------------------------------------
_NEG_BOUND_RE = re.compile(r"\bnon[-\u2010\u2011\u2012\u2013\u2014]"
                           r"([A-Za-zÀ-ÖØ-Þà-öø-þ][\w’'\-]*)")


def _denied_terms(body: str) -> set:
    """Terms this text prints ONLY under a bound negative prefix."""
    hay = norm_text(body)
    out = set()
    for m in _NEG_BOUND_RE.finditer(body or ""):
        term = norm_text(m.group(1))
        if not term:
            continue
        total = len(re.findall(r"(?<![\w'])" + re.escape(term) + r"(?![\w'])", hay))
        bound = sum(1 for x in _NEG_BOUND_RE.finditer(body or "")
                    if norm_text(x.group(1)) == term)
        if total and total == bound:
            out.add(term)
    return out


# ---------------------------------------------------------------------------
# CLASS 3 — composite gluing.  Row `disc-subnational-pair`: "A funding proposal
# submitted by **Pegasus Capital Advisors** for the **Global Subnational
# Climate Fund (SoFC Global)**" made ``_CAPRUN_RE`` run one name into the other
# across 'for the', inventing the entity `Pegasus Capital Advisors for the
# Global Subnational Climate Fund`, which no page prints and which the matcher
# then reported as unsupported.
#
# WHY THIS LOSES NO CHECK — the whole argument. The composite is dropped ONLY
# when BOTH halves survive as independent candidate groups of their own, so
# every word of it is still checked, just as the two names it really was. The
# halves must be whole extracted names, not substrings of one, and both are
# required: a composite whose second half nothing else attests ('… for the
# Wakanda Development Bank') keeps the glued form and is still checked.
# Pinned by ``test_a_glued_pair_needs_both_halves_attested``,
# ``test_a_glued_half_is_not_covered_by_containment`` and
# ``test_dropping_a_glued_pair_deletes_no_check``.
# ---------------------------------------------------------------------------
_ENT_GLUE_RE = re.compile(
    r"\s+(?:and|et|for|pour|with|avec|by|par|in|dans|on|at|to)\s+"
    r"(?:(?:the|a|an|le|la|les|un|une|des|du)\s+)*")


def _glue_splits(name: str):
    """Every binary split of a normalised candidate at ONE connective."""
    for m in _ENT_GLUE_RE.finditer(name):
        left, right = name[:m.start()].strip(), name[m.end():].strip()
        if left and right:
            yield left, right


def _drop_glued(groups: List[List[str]]) -> List[List[str]]:
    """Remove candidates that are two attested names run together.

    Iterated to a fixed point, and re-checked after each removal against the
    names that ACTUALLY SURVIVE: a half covered only by a group that is itself
    dropped is not covered, and the composite stays.
    """
    keep = list(groups)
    while True:
        for vs in keep:
            others = {norm_text(v) for g in keep if g is not vs for v in g}
            others.discard("")
            if any(left in others and right in others
                   for left, right in _glue_splits(norm_text(vs[0]))):
                keep = [g for g in keep if g is not vs]
                break
        else:
            return keep


def entities(text: str) -> List[List[str]]:
    """Proper-noun assignments a sentence makes, each as a variant list.

    Bold and quoted spans first (the answer model marks its own assertions),
    then acronyms and capitalized runs. Citation brackets are stripped before
    anything else: a document id is a pointer, never an entity claim.
    """
    body = _strip_citations(text)
    denied = _denied_terms(body)                 # CLASS 4
    cands: List[Tuple[str, bool]] = []           # (candidate, came from quotes)
    for m in _BOLD_RE.finditer(body):
        cands.append((m.group(1) or m.group(2) or "", False))
    for m in _QUOTE_RE.finditer(body):
        cands.append((m.group(1), True))
    plain = re.sub(r"[*_`]", "", body)
    # drop the leading word of the sentence: capitalization there is grammar
    # ...including a sentence that opens inside a bracket or a quote:
    # '(Separately, the excerpts ...' capitalizes an adverb because a sentence
    # starts there, not because anything is named (claim-e79ef060). Same rule
    # the line above already applies to an unbracketed opening.
    plain = re.sub(r"(^|(?<=[.!?…]\s))(\s*[(\[\"“«]?\s*)([A-ZÀ-ÖØ-Þ])",
                   lambda m: m.group(1) + m.group(2) + m.group(3).lower(),
                   plain)
    for m in _CAPRUN_RE.finditer(plain):
        cands.append((_trim_run(m.group(1)), False))
    for m in _ACRONYM_RE.finditer(plain):
        cands.append((m.group(1), False))

    out: List[List[str]] = []
    seen = set()
    for c, quoted in cands:
        parts = (re.split(r"\s*(?:,| and | et |;)\s*", c) if _is_list(c) else [c])
        for part in parts:
            variants = _entity_variants(part)
            if not variants:
                continue
            v = variants[0]
            key = norm_text(v)
            if not key or key in seen or key in _ENT_STOP:
                continue
            if key in denied:
                continue                 # CLASS 4: the unit denies this term
            if _ENT_HAS_ID_RE.search(v):
                continue                 # a pointer, not a name
            if key in _ENT_GENERIC or _all_generic(v):
                continue
            if not _entity_core(v) or not _looks_like_name(v, quoted):
                continue
            seen.add(key)
            out.append(variants)

    # 'Angola, Benin and Kenya' yields the three countries AND the run 'Benin
    # and Kenya' the capitalized-run scan saw. The run is an artifact: the page
    # prints 'Benin, Kenya', so requiring it verbatim fails a correct answer.
    names = {norm_text(vs[0]) for vs in out}

    def _joined_artifact(name: str) -> bool:
        parts = [p for p in re.split(r"\s+(?:and|et)\s+", name) if p]
        return len(parts) > 1 and all(p in names for p in parts)

    out = [vs for vs in out if not _joined_artifact(norm_text(vs[0]))]

    # CLASS 3: the same argument one connective wider — 'X for the Y' where X
    # and Y are both extracted names of their own. Applied after the 'and'
    # rule above and never before it, so the pool it reads is already free of
    # the list artifacts that rule removes.
    out = _drop_glued(out)

    # A single-word candidate already contained in a MULTI-WORD candidate adds
    # no check: 'Unlocking' was cut out of the title it belongs to, and
    # verifying the longer form covers the fragment. Note the scope: the
    # `len(vs[0].split()) > 1` guard means only single-word candidates are
    # ever suppressed — a multi-word alias like 'SoCF Global' stays an
    # independent group even when a longer name carries it.
    longer = [norm_text(vs[0]) for vs in out if len(vs[0].split()) > 1]
    return [vs for vs in out
            if len(vs[0].split()) > 1
            or not any(norm_text(vs[0]) in ln for ln in longer)]


def _is_list(c: str) -> bool:
    """'Bosnia and Herzegovina, Kazakhstan, Moldova' is three entities; a
    single 'European Bank for Reconstruction and Development' is one.

    The comma is the deciding mark: an 'and' with no comma anywhere belongs to
    an institution's name far more often than to a list, and splitting
    'International Union for Conservation of Nature and Natural Resources' in
    half invents two entities neither of which the page prints.
    """
    return "," in c


@dataclass
class Claim:
    """One atomic factual assertion and the evidence it points at."""
    text: str
    kind: str                                   # money|number|entity|year|existence
    citations: List[Citation] = dc_field(default_factory=list)
    amounts: List[Amount] = dc_field(default_factory=list)
    entities: List[List[str]] = dc_field(default_factory=list)
    index: int = 0
    unit_kind: str = "sentence"                 # sentence|bullet|table-row
    inherited: bool = False                     # citations borrowed from the block
    #: Heading-form units this claim COMPLETES (adjudication ruling 6). The
    #: lead-in is not a claim of its own; whatever it stated that this claim
    #: does not restate is checked here, against this claim's own scope.
    lead_ins: List[str] = dc_field(default_factory=list)
    lead_in_amounts: List[Amount] = dc_field(default_factory=list)
    lead_in_entities: List[List[str]] = dc_field(default_factory=list)
    lead_in_years: List[str] = dc_field(default_factory=list)

    @property
    def carries_lead_in(self) -> bool:
        return bool(self.lead_in_amounts or self.lead_in_entities
                    or self.lead_in_years)

    @property
    def required(self) -> bool:
        """Fact-bearing claims: an answer that loses one of these has lost its
        substance, which is what separates 'partial' from 'abstain'."""
        return self.kind in ("money", "number", "entity")

    @property
    def cited(self) -> bool:
        return bool(self.citations)


def _mask(text: str) -> Tuple[str, List[str]]:
    store: List[str] = []

    def sub(m):
        store.append(m.group(0))
        return "\x00%d\x00" % (len(store) - 1)

    return _MASK_RE.sub(sub, text), store


def _unmask(text: str, store: Sequence[str]) -> str:
    return re.sub(r"\x00(\d+)\x00", lambda m: store[int(m.group(1))], text)


def split_sentences(text: str) -> List[str]:
    """Sentence split that survives 'p. 5', 'B.27', 'Add.12' and '18.5'."""
    masked, store = _mask(text)
    return [s for s in (_unmask(p, store).strip()
                        for p in _SENT_SPLIT_RE.split(masked)) if s]


def _blocks(answer: str) -> List[List[str]]:
    """Blank-line-separated blocks of non-empty, non-heading lines."""
    out: List[List[str]] = []
    cur: List[str] = []
    for raw in (answer or "").splitlines():
        line = raw.strip()
        if not line:
            if cur:
                out.append(cur)
                cur = []
            continue
        if _HEADING_RE.match(line):
            continue
        cur.append(line)
    if cur:
        out.append(cur)
    return out


def _units(answer: str) -> List[Tuple[str, str, List[Citation], bool]]:
    """(text, unit_kind, citations, inherited) for every candidate claim unit.

    Citations are inherited within a paragraph. The dominant citation style in
    these answers puts one bracket at the END of a two- or three-sentence
    paragraph — under strict per-sentence attribution the sentences before it
    all read as 'uncited factual claim', which was 4 of the 22 gold-passing
    recorded answers. A sentence therefore borrows the nearest FOLLOWING
    bracket of its own paragraph (falling back to the nearest preceding one),
    and the borrowing is recorded so a verdict can say so.

    Bullets and table rows never borrow: a list is a set of independent
    statements, and one bullet's citation is not another's.
    """
    out: List[Tuple[str, str, List[Citation], bool]] = []
    for block in _blocks(answer):
        # A line that is nothing but citations is the whole block's source —
        # the shape a bulleted list of countries takes, with one bracket under
        # the last bullet. Every item of the block may borrow it, which is not
        # the same as one bullet borrowing another bullet's inline citation.
        block_cits: List[Citation] = []
        for line in block:
            if _CITE_ONLY_RE.match(line):
                block_cits += parse_citations(line)
        seq: List[Tuple[str, str, List[Citation], bool]] = []
        for line in block:
            if line.startswith("|"):
                if not _TABLE_RULE_RE.match(line):
                    seq.append((line, "table-row", parse_citations(line), True))
                continue
            if _BULLET_RE.match(line):
                body = _BULLET_RE.sub("", line)
                seq.append((body, "bullet", parse_citations(body), True))
                continue
            for s in split_sentences(line):
                seq.append((s, "sentence", parse_citations(s), False))
        for i, (text, kind, cits, isolated) in enumerate(seq):
            if cits:
                out.append((text, kind, cits, False))
                continue
            if isolated:
                out.append((text, kind, block_cits, bool(block_cits)))
                continue
            after = next((seq[j][2] for j in range(i + 1, len(seq))
                          if seq[j][2] and not seq[j][3]), None)
            before = next((seq[j][2] for j in range(i - 1, -1, -1)
                           if seq[j][2] and not seq[j][3]), None)
            borrowed = after or before or block_cits or []
            out.append((text, kind, borrowed, bool(borrowed)))
    return out


def claim_kind(text: str) -> Optional[str]:
    """The strongest factual trigger in a unit, or None when it is prose glue.

    Order matters: a sentence stating a figure is verified as a figure even
    when it also names an entity, because the figure is the falsifiable part.
    """
    body = _strip_citations(text)
    if not re.search(r"[A-Za-zÀ-ÖØ-Þà-öø-þ]", body):
        return None
    amts = amounts(body)
    if _GLUE_RE.search(body) or _REFUSAL_RE.search(body):
        # ... unless the hedge still states a cited figure. 'In summary, FP151
        # requests USD 18,500,000 [doc, p.5]' is a checkable claim wearing a
        # connective; dropping the whole unit dropped the figure with it.
        cited_figure = any(_money_like_amount(a) for a in amts) and \
            bool(parse_citations(text))
        if not cited_figure:
            return None
    if any(_money_like_amount(a) for a in amts):
        return "money"
    if _COUNT_RE.search(body) or _BIGNUM_RE.search(body):
        return "number"
    if entities(body):
        return "entity"
    if _YEAR_RE.search(body) or _BOARD_RE.search(body):
        return "year"
    if _EXISTENCE_RE.search(body):
        return "existence"
    return None


# ---------------------------------------------------------------------------
# HEADING FORM — adjudication rulings 1, 2 and 6.
#
# Ruling 6 is binding and is stated as a SHAPE rule: "when a unit has the form
# of a heading, bold label, or colon lead-in, it is `not_a_claim` EVEN IF it
# carries a checkable proposition — the predicate is completed by the unit
# below it", and "the fix is claim extraction". Twelve of the 71 adjudicated
# rows are that shape; the re-A/B at 9925a2c measured ~7 of 22 repair
# rejections blocking on exactly these, with the judge answering "no claim
# text provided" — a lead-in is INHERITED failure that no rewrite keeping the
# document's structure can clear. (The repair pass is gone as of eac4c94. The
# extraction defect it made visible is not, and this ruling stands on the 12
# adjudicated rows, not on the rejection count.)
#
# THE DEFINITION IS TIGHT ON PURPOSE, in three ways:
#
#   * SENTENCE UNITS ONLY. `_units` already treats bullets and table rows as
#     independent statements that never borrow another item's citation; a list
#     item is a statement, not a heading. Without this, the bullet
#     `- **USD 50,000,000** [doc, p.5]` reads as a bold-only label the moment
#     its citation is stripped, and the figure would stop being checked.
#   * COLON AT THE END, never a colon inside. 'The cover page states: "Total
#     financing 720 M USD" [doc, p.5].' is an ordinary sentence that happens
#     to introduce a quotation, and it keeps every check it had.
#   * A BOLD LABEL IS THE WHOLE UNIT. '**FP220 (ARCAFIM) — Kenya/Uganda**' is
#     a label; 'the accredited entity is **Save the Children Australia**' is
#     bold emphasis inside prose and is untouched, because the emphasis does
#     not span the unit.
#
# Markdown headings are named here for completeness of the shape, but the
# line never reaches `_units`: `_blocks` drops `#`-prefixed lines whole, and
# this wave does not change that. What a `#` heading states is therefore still
# invisible to verification — 18 heading lines across the two recorded release
# corpora, none carrying a money-like amount. Pre-existing, measured, and
# deliberately not bundled into this change.
# ---------------------------------------------------------------------------

#: A unit whose entire text is one bold (or underscored) span, with an
#: optional trailing colon. The inner span may not itself contain a marker,
#: so '**A** and **B**' is prose with two emphases, not one label.
_BOLD_LABEL_RE = re.compile(r"^(?:\*\*[^*]+\*\*|__[^_]+__)\s*[:：]?\s*$")
#: Colon-terminated, tolerating the French thin/no-break space before the mark.
_COLON_END_RE = re.compile(r"[:：]$")

HEADING_SHAPES = ("markdown-heading", "bold-label", "colon-lead-in")


@dataclass
class LeadIn:
    """A heading-form unit ruling 6 keeps out of the claim list.

    It is recorded rather than merely skipped: a release recorded BEFORE this
    rule existed lists these units among its failures, and every tool that
    joins a recording to a fresh extraction has to be able to say 'this row is
    resolved by extraction' instead of 'this row vanished'.
    """
    text: str
    shape: str                                   # one of HEADING_SHAPES
    unit_index: int
    kind: Optional[str] = None                   # the kind it would have had
    carried_to: Optional[int] = None             # claim index that took its content
    #: the Claim this unit WOULD have been. Never returned by
    #: ``extract_claims`` and never classified; it exists so a reconciling
    #: tool can still show a reviewer the unit's text, citations and terms.
    claim: Optional["Claim"] = None


def heading_form(text: str, unit_kind: str = "sentence") -> Optional[str]:
    """Which heading shape this unit has, or None — rulings 1, 2 and 6.

    Citations are stripped first: a lead-in may carry the block's bracket
    ('What the excerpts show [doc, p.5]:'), and a bracket is typography, not
    a predicate.
    """
    if unit_kind != "sentence":
        return None
    t = _strip_citations(text or "").strip()
    if not t:
        return None
    if _HEADING_RE.match(t):
        return "markdown-heading"
    if _BOLD_LABEL_RE.match(t):
        return "bold-label"
    if _COLON_END_RE.search(t):
        return "colon-lead-in"
    return None


def _year_tokens(body: str) -> List[str]:
    """Year and board tokens, the shape ``_check_years`` verifies."""
    return sorted(set(_YEAR_RE.findall(body))
                  | {m.group(0) for m in _BOARD_RE.finditer(body)})


def _lead_in_content(text: str) -> Tuple[List[Amount], List[List[str]], List[str]]:
    """(money-like amounts, entities, year tokens) a heading-form unit states.

    MONEY-LIKE AMOUNTS ONLY, and the reason is measured: the bold label
    '**FP172 (103_gcf-b30-03-add04)**' yields the bare 'amounts' 103 and 03
    out of a DOCUMENT ID. Carrying those onto the figure bullet below would
    demand the evidence print '103', which is a fabricated check, not a
    preserved one. ``_money_like_amount`` is the same filter ``claim_kind``
    uses to call a unit a money claim, so what is carried is exactly what
    would have been checked had the unit stayed a claim.
    """
    body = _strip_citations(text or "")
    return ([a for a in amounts(body) if _money_like_amount(a)],
            entities(body), _year_tokens(body))


def _carry_onto(text: str, amts: Sequence[Amount], ents: Sequence[Sequence[str]],
                years: Sequence[str]
                ) -> Tuple[List[Amount], List[List[str]], List[str]]:
    """The lead-in content this unit does NOT already state itself.

    Ruling 6's own rationale — 'the predicate is completed by the unit below
    it' — is the whole rule here: content the unit below restates is already
    checked as that unit's own, and carrying it twice would only widen what a
    single scope has to print at once.
    """
    body = _strip_citations(text or "")
    own_amounts = amounts(body)
    hay = norm_text(body)
    keep_a = [a for a in amts
              if not any(amount_matches(a, o) for o in own_amounts)]
    keep_e = [list(vs) for vs in ents
              if not any(norm_text(v) and norm_text(v) in hay for v in vs)]
    keep_y = [y for y in years
              if y.replace(" ", "") not in body.replace(" ", "")]
    return keep_a, keep_e, keep_y


def extract_claims(answer: str) -> List[Claim]:
    """Split an answer into atomic factual claims carrying their citations.

    A claim is a sentence, bullet or table row asserting a money amount, a
    number with a unit, a proper-noun assignment, a year/board fact or an
    existence fact. Prose glue, hedges and refusals are dropped; an uncited
    factual sentence is still a claim — with no evidence pointer, which is
    exactly what makes it interesting to the classifier.

    HEADING-FORM UNITS ARE NOT CLAIMS (ruling 6, ``heading_form``), and the
    rule has exactly one escape clause, stated as an invariant:

        a heading-form unit stops being a claim only when its checkable
        content is either empty or CARRIED onto the claim it introduces.
        Checkable content is never silently dropped.

    So '**Accredited entity:**' disappears, while
    '**FP151 (total financing: USD 999 million)**:' disappears as a claim and
    hands 'USD 999 million' to the first claim below it, which then has to
    find it in the evidence THAT claim cites. A heading-form unit that states
    a figure or a name and has no claim under it to hand them to stays a claim
    itself, because nothing else would ever check it.

    The one thing the invariant does not cover is a bare year/board token: it
    is carried when there is somewhere to carry it, but it does not by itself
    keep a lead-in alive. Two adjudicated rows are year-only lead-ins
    ('...the Board meeting dates shown for **2021** are:') and blocking on the
    token would leave ruling 6 unimplemented for them. Measured residue on the
    record: one answer (`agg-2020-range`) whose lead-in names 2020 and whose
    following unit yields no claim, so that token stops being checked.
    """
    return _walk_units(answer)[0]


def lead_ins(answer: str) -> List[LeadIn]:
    """Every heading-form unit extraction DROPS, in document order.

    Produced by the same walk as ``extract_claims`` — never by a second
    reading of the answer — so the two can never disagree about which units
    ruling 6 removed. A recorded release lists these units among its failures;
    a tool reconciling a recording against the current extractor needs them by
    name, and 'the claim vanished' is not a name.
    """
    return _walk_units(answer)[1]


def unminted_units(answer: str) -> List[Claim]:
    """Every unit of this answer that ``extract_claims`` does NOT return.

    Ruling-6 lead-ins are most of them; a unit the hedge list drops is the
    rest. A release recorded before either rule lists such units among its
    failures, so a tool joining a recording to a fresh extraction needs to
    show them — with their text, citations and terms — rather than report the
    row as unrecoverable. Built from the same walk as the claims, so what is
    'not minted' can never drift from what was.

    These Claim objects are a VIEW for reconciliation. Nothing classifies
    them; ``extract_claims`` remains the only source of claims.
    """
    claims, _dropped = _walk_units(answer)
    minted = {c.text for c in claims}
    out: List[Claim] = []
    for text, unit_kind, citations, inherited in _units(answer):
        if text.strip() in minted:
            continue
        out.append(_as_claim(text, unit_kind, citations, inherited,
                             claim_kind(text), len(out)))
    return out


def _as_claim(text: str, unit_kind: str, citations: List[Citation],
              inherited: bool, kind: Optional[str], index: int) -> Claim:
    body = _strip_citations(text)
    return Claim(text=text.strip(), kind=kind or "", citations=citations,
                 amounts=amounts(body), entities=entities(body),
                 index=index, unit_kind=unit_kind, inherited=inherited)


def _walk_units(answer: str) -> Tuple[List[Claim], List[LeadIn]]:
    """(claims, dropped lead-ins) — the single pass both entry points share."""
    units = _units(answer)
    kinds = [claim_kind(text) for text, _k, _c, _i in units]
    later_claim = [False] * len(units)
    seen = False
    for i in range(len(units) - 1, -1, -1):
        later_claim[i] = seen
        seen = seen or kinds[i] is not None

    claims: List[Claim] = []
    dropped: List[LeadIn] = []
    pending: List[LeadIn] = []
    pend_texts: List[str] = []
    pend_a: List[Amount] = []
    pend_e: List[List[str]] = []
    pend_y: List[str] = []
    for i, (text, unit_kind, citations, inherited) in enumerate(units):
        shape = heading_form(text, unit_kind)
        if shape:
            amts, ents, years = _lead_in_content(text)
            # THE INVARIANT: demote only when nothing checkable is lost — the
            # unit says nothing checkable, or there is a claim below it to
            # hand what it does say to. A TRAILING lead-in with content is
            # therefore verified where it stands (nothing below completes it),
            # and a trailing contentless one simply disappears.
            if not (amts or ents) or later_claim[i]:
                li = LeadIn(text=text.strip(), shape=shape, unit_index=i,
                            kind=kinds[i], carried_to=None,
                            claim=_as_claim(text, unit_kind, citations,
                                            inherited, kinds[i], i))
                dropped.append(li)
                pending.append(li)
                pend_texts.append(text.strip())
                pend_a += amts
                pend_e += ents
                pend_y += years
                continue
        kind = kinds[i]
        if kind is None:
            continue
        body = _strip_citations(text)
        keep_a, keep_e, keep_y = _carry_onto(text, pend_a, pend_e, pend_y)
        for li in pending:
            li.carried_to = len(claims)
        claims.append(Claim(
            text=text.strip(), kind=kind, citations=citations,
            amounts=amounts(body), entities=entities(body),
            index=len(claims), unit_kind=unit_kind, inherited=inherited,
            lead_ins=list(pend_texts), lead_in_amounts=keep_a,
            lead_in_entities=keep_e, lead_in_years=keep_y))
        pending, pend_texts, pend_a, pend_e, pend_y = [], [], [], [], []
    return claims, dropped


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

_MATRIX_DOC_RE = re.compile(r"^(?P<label>[^|\n]{1,40}?)\s*->\s*(?P<doc>" + _DOC_RE + r")")
_MATRIX_PAGE_RE = re.compile(r"\(p\.\s*(\d{1,3})")
_REG_DOC_RE = re.compile(r"\[(" + _DOC_RE + r"),\s*cover pages?\]", re.I)

# NOTE-PAGE SCOPE — the two regexes are ``chainlit_app._note_pages``'s, byte
# for byte, and that is the point: the app decides which cited pages are legal
# with these, so the verifier deciding it with anything else is how three
# instruments end up disagreeing about one citation. A document is named on a
# line when it appears in a bracket or a parenthesis (main registry lines end
# '[stem, cover pages]', conflict lines name the stem in parentheses); a page
# is printed when the line prints '(p.<n>,' or '(p.<n>)'; and a page belongs
# to every document named on ITS OWN line — never to a document named on the
# line above or below, which is what keeps two documents' notes in one block
# from lending each other pages.
_NOTE_DOC_RE = re.compile(r"[\[(](" + _DOC_RE + r")")
_NOTE_PAGE_RE = re.compile(r"\(p\.(\d{1,3})[,)]")


def note_page_scopes(line: str) -> List[EvidenceKey]:
    """The note-scope keys ONE note line publishes: [(notes:doc, page), ...].

    Deliberately NOT ``_MATRIX_PAGE_RE.search`` — that reads the first pointer
    only, which is exactly why 'is printed as 21,128,224 USD (p.6, A.8); also
    as 49,151,817 USD (p.76, B.2(b))' published p.6 and swallowed p.76, and
    why release-3's fr-fp172-nepal lost the second half of a conflict report
    it had been instructed to write.
    """
    docs = list(dict.fromkeys(_NOTE_DOC_RE.findall(line)))
    if not docs:
        return []
    return [(note_scope_doc(d), int(p))
            for p in dict.fromkeys(_NOTE_PAGE_RE.findall(line)) for d in docs]


@functools.lru_cache(maxsize=64)
def _note_scopes_of(text: str) -> Dict[EvidenceKey, str]:
    """``{(notes:doc, page): the line(s) that printed it}`` for one note blob.

    Cached on the blob's own text and never mutated by its callers: the
    verifier is asked for the same turn's scopes once per claim and once per
    conflict scan, and re-splitting the registry note each time is the kind of
    cost that turns a deterministic pass into a slow one.
    """
    out: Dict[EvidenceKey, str] = {}
    for line in (text or "").splitlines():
        line = line.strip()
        for key in note_page_scopes(line):
            prev = out.get(key)
            out[key] = f"{prev}\n{line}" if prev and line not in prev else \
                (prev or line)
    return out


def note_scopes(evidence: Evidence) -> Dict[EvidenceKey, str]:
    """The note-page scopes THIS turn publishes, derived not stored.

    Read off the whole note blob at ``NOTES_KEY`` — which ``build_evidence``
    always holds when a note reached the prompt — so nothing has to be added
    to the evidence dict for a citation to reach a note line, and an evidence
    set assembled by hand (the scorer's frozen snapshots, this suite's
    fixtures) gets the same scopes from the same bytes.
    """
    return _note_scopes_of(evidence.get(NOTES_KEY, ""))


def build_evidence(hits: Sequence[Any] = (), notes: Any = None) -> Evidence:
    """{(doc_id, page): text} for one turn — the only thing claims may cite.

    Hits contribute their source text at (doc_id, page). Computed note blocks
    (registry lines, year notes, the evidence matrix) contribute twice: under
    the document they name — page-level when the line prints '(p.5, A.8)',
    document-level otherwise, which is the 'cover pages' scope an answer cites
    — and, whole, under the notes pseudo-document, so a note-level citation
    still resolves to something we hold.

    THE KEY SET IS FROZEN. Note-page scopes (``note_scopes``) are derived
    from the block held at ``NOTES_KEY`` when a citation asks for one; they
    are deliberately NOT added here. Recorded runs carry this dict's keys as
    ``claims.evidence_keys`` and the release backfill re-runs this function
    and asserts the reconstruction reproduces them exactly, so a new key is a
    change to a frozen artifact's contents — and a citation that resolves to a
    line of the notes needs no key of its own to be read.
    """
    ev: Evidence = {}

    def add(key: EvidenceKey, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        prev = ev.get(key)
        ev[key] = f"{prev}\n{text}" if prev and text not in prev else (prev or text)

    for h in hits or ():
        doc = getattr(h, "doc_id", None) or (h.get("doc_id") if isinstance(h, dict) else None)
        if not doc:
            continue
        page = getattr(h, "page", None) if not isinstance(h, dict) else h.get("page")
        text = getattr(h, "text", None) if not isinstance(h, dict) else h.get("text")
        add((doc, int(page) if page else None), text or "")

    blocks = [notes] if isinstance(notes, str) else list(notes or [])
    for block in blocks:
        if not block:
            continue
        add(NOTES_KEY, block)
        current: Optional[str] = None
        for line in str(block).splitlines():
            m = _MATRIX_DOC_RE.match(line.strip())
            if m:
                current = m.group("doc")
                add((current, None), line.strip())
                continue
            reg = _REG_DOC_RE.search(line)
            if reg:
                add((reg.group(1), None), line.strip())
                continue
            docs_in_line = re.findall(_DOC_RE, line)
            owner = docs_in_line[0] if docs_in_line else current
            if not owner:
                continue
            pg = _MATRIX_PAGE_RE.search(line)
            add((owner, int(pg.group(1)) if pg else None), line.strip())
    return ev


def _resolve_doc(cited: str, docs: Iterable[str]) -> Optional[str]:
    """Forgiving document match — the answer may print a truncated id.

    SORTED, and that is the whole point. Both callers build their candidate
    pool as a set comprehension over the evidence keys
    (``{k[0] for k in evidence ...}``), and a truncated id is routinely a
    prefix of more than one held document — ``...add11`` and ``...add12`` sit
    side by side all over this corpus. Picking "the first prefix match" out of
    a set made the resolution depend on the process's string hash seed, so the
    DETERMINISTIC layer returned different verdicts in different processes
    from byte-identical inputs:

        PYTHONHASHSEED 0,1,2  ->  supported     (resolved to ...add11)
        PYTHONHASHSEED 3..7   ->  contradicted  (resolved to ...add12)

    — same answer, same evidence, opposite adoption decision, with the LLM
    layer removed entirely. Pinning temperature and seed on the model calls
    does not touch this; a canary sampling this would be sampling the hash
    seed. Sorting is done HERE rather than at the two call sites so that a
    future caller cannot reintroduce it by passing another set.

    The tie-break is now the contents of ``docs`` and nothing else. It is
    still a tie-break, not a resolution: an id that truly names two documents
    is ambiguous, and choosing the lexicographically first is a stable answer
    to an ambiguous question, not a right one. Whether such a citation should
    resolve at all is a scoping question for the `_scoped_field_conflict`
    design pass `docs/wave0c-review-verdict.md` asks for; making it
    reproducible is a prerequisite for measuring it either way.
    """
    docs = sorted(set(docs))
    if cited in docs:
        return cited
    return next((d for d in docs
                 if d.startswith(cited[:24]) or cited.startswith(d[:24])), None)


# ---------------------------------------------------------------------------
# deterministic classification
# ---------------------------------------------------------------------------

SUPPORTED, CONTRADICTED, UNSUPPORTED = "supported", "contradicted", "unsupported"

# label -> the field it names, for the 'different value for the same labelled
# field' contradiction test. Ordered: the first match wins.
_FIELD_LABELS: List[Tuple[str, str]] = [
    ("gcf_financing",
     r"gcf\s+(?:financing|funding|amount|contribution|request(?:ed)?|proceeds)"
     r"|requested\s+gcf\s+amount|(?:total\s+)?gcf\s+funding\s+requested"
     r"|financement\s+(?:du\s+)?(?:fvc|gcf)|montant\s+demand[ée]"),
    ("total_financing",
     r"total\s+(?:financing|project\s+(?:cost|value|financing)|investment)"
     r"|financement\s+total|co[uû]t\s+total"),
    ("co_financing", r"co-?financing|cofinancement"),
    ("accredited_entity",
     r"accredited\s+entity|entit[ée]\s+accr[ée]dit[ée]e|implementing\s+entity"),
    ("countries", r"countr(?:y|ies)|pays\b|host\s+countr"),
    ("title", r"\btitle\b|\btitre\b|project\s+name"),
    ("duration", r"duration|dur[ée]e|implementation\s+period|lifespan"),
    ("beneficiaries", r"beneficiaries|b[ée]n[ée]ficiaires"),
    # --- Phase 1 additions, and they are additions in the strict sense -------
    # Three fields the registry holds for 193 / 101 / 70 documents and that
    # nothing could reach until `planner.FIELD_ORDER` and `registry.
    # _served_bits` began serving them. A served field is only as checkable as
    # the vocabulary that can find its label, so the label a value is printed
    # under there is the label read for it here, and the two lists are edited
    # together or not at all.
    #
    # APPENDED, never inserted, and none of the three patterns matches a
    # phrase an existing arm already claims. `claim_field` is first-match-wins,
    # so a sentence naming two fields keeps the field it resolves to today:
    # 'the accredited entity is X and the executing entity is Y' stays
    # `accredited_entity`, exactly as before. ('executing entity' has never
    # matched the accredited arm — that arm reads 'accredited entity',
    # 'entité accréditée' and 'implementing entity' — so the new arm claims
    # only sentences that named this field and no other.)
    ("executing_entity",
     r"executing\s+entit|executing\s+agenc|entit[ée]s?\s+d.ex[ée]cution"
     r"|agence\s+d.ex[ée]cution"),
    ("national_designated_authority",
     r"national\s+designated\s+authorit|designated\s+authorit|\bNDA\b"
     r"|autorit[ée]\s+nationale\s+d[ée]sign[ée]e"),
    ("financial_instruments",
     r"financial\s+instrument|instruments?\s+financiers?"),
]
_FIELD_RES = [(f, re.compile(r, re.I)) for f, r in _FIELD_LABELS]


def claim_field(text: str) -> Optional[str]:
    body = norm_text(_strip_citations(text))
    for name, rx in _FIELD_RES:
        if rx.search(body):
            return name
    return None


# ---------------------------------------------------------------------------
# DERIVED ARITHMETIC — a computed difference is not a printed figure
#
# THE LIMITATION THIS CLOSES (docs/l1-l2-coverage-review.md §7.1,
# docs/build-report.html Act XIV, row `fu-compare-those`). The answer writes
#
#     FP152 requests 131.5 M USD more GCF funding than FP151
#     (150.0 - 18.5 = 131.5).  [124_..., p. 5] [123_..., p. 5]
#
# cites both operands to the pages that print them, and every matcher in this
# module answers the only question it knows how to ask — 'is 131.5 M USD
# printed in the cited evidence?' — with NO. The cited note prints
# '150 M USD' under the claim's own field label, so `_key_conflict` fires and
# the verdict is CONTRADICTED: the strongest thing this module can say, said
# about a computation that is exactly right. It was the only contradicted
# verdict in release-6 and the shape recurs in release-7.
#
# WHAT QUALIFIES, and every clause of it is a gate:
#
#   1. THE TEXT CARRIES THE WHOLE DERIVATION — both operands AND the result,
#      joined by an operator or by a comparative that names one. A claim that
#      states only the result ('131.5 M USD more than FP151') derives nothing
#      this module can check and keeps today's verdict.
#   2. ONE CURRENCY AND ONE SCALE. Cross-currency arithmetic is the act
#      `prompts.CORE` spends a paragraph forbidding, and a verifier that
#      blessed 'EUR 87 million - USD 18.5 million' would be certifying the
#      forbidden act. Mixed scale words are refused for the same reason the
#      `clash` machinery exists: this corpus prints '28,654 million USD' and
#      means 28.654 million, so a derivation spanning two printed scales is
#      not one this module may resolve. Both refusals are silent: the claim
#      simply does not qualify and today's outcome stands.
#   3. EVERY OPERAND VERIFIES AGAINST THE CLAIM'S OWN CITED SCOPES, under the
#      matchers and at the strictness every other claim gets — the held keys
#      the citations resolve to, and nothing else. The registry-backed
#      promotion is deliberately NOT part of qualification: an operand no key
#      of this turn prints is not an operand this turn may compute with.
#   4. THE CLAIM AS WRITTEN MUST NOT ALREADY VERIFY. This is the narrowest
#      and most important gate, and it is enforced by the CALLER (the branch
#      is reached only when the ordinary matchers have already failed).
#      Release-7's `txt-fu-fp171-total-fr` writes '(a)+(b) = 21 929 123,33
#      USD, composee de 20 986 732,33 USD ... et 748 460,00 USD' — an
#      equation QUOTED off the document's own C.1 table, whose three printed
#      figures do not add up. It verifies today, because every figure in it
#      is printed. Re-routing it through this path would call the document's
#      own arithmetic a contradiction of the answer, which it is not: the
#      answer reported what the page prints. Quoting is not deriving, and the
#      existing matcher owns quoting.
#
# THEN, and only then, the computation decides: right -> SUPPORTED carrying
# `derived-arithmetic`; wrong -> CONTRADICTED, because a wrong computation
# over operands the evidence really prints is the one thing this module can
# say with certainty about a figure no page contains.
#
# TOLERANCE. `amount_matches`'s, reused rather than invented: two figures
# agree when they differ by less than half the coarser of the two PRINTED
# precisions (`granularity`). So 150.0 - 18.5 = 131.5 is exact and needs no
# tolerance at all; '(150 - 18.5 = 132)' is accepted, because a result
# printed to whole millions cannot distinguish 131.5 from 132; and 131.6,
# printed to a tenth, is refused, because a result printed to tenths can.
# 'Correct arithmetic' and 'the same printed figure' therefore mean the same
# thing here as everywhere else in this module.
#
# WHAT IS OUT OF SCOPE, deliberately, and each keeps today's behaviour:
# ratios and percentages (the result of a division carries no currency, so
# clause 2 has nothing to check and the rounding of '8.1x' is not the
# rounding `granularity` models — release-7's '150 / 18.5' claim stays on the
# judge's path exactly as it is); dimensionless counts (a bare operand
# matches any digit run a page prints, which is the loosest the matchers get
# and no place to add a new way to become SUPPORTED); and French comparative
# phrasing (an under-firing rule keeps today's verdict, which is the safe
# direction to be incomplete in).
# ---------------------------------------------------------------------------

#: The caution-class flag a SUPPORTED verdict carries when the figure the
#: answer states was COMPUTED rather than read off a page. It rides the
#: verdict exactly as `citation-page-mismatch` and `unit-scale-clash` do, so
#: `RepairResult.cautions` — and the app's caution line, which formats
#: whatever flags it is given — shows it with no change on either side.
DERIVED_ARITHMETIC = "derived-arithmetic"

#: `iter_amounts(money_only=True)`'s floor, reused: below it a bare figure
#: with no scale word is table furniture, not an amount of money.
_ARITH_MONEY_FLOOR = 1e4

_ARITH_SCALE = r"(?:millions?|billions?|thousands?|milliards?|bn|m|k)"
_ARITH_CUR = r"(?:USD|US\$|EUR|euros?|€|\$)"
#: The gap BETWEEN two printed mantissas, which is the only place an operator
#: is allowed to be. Anchored at both ends (`^...$`), so anything else in the
#: gap — a word, a second bracket, a comma — means these two figures are not
#: the two sides of one operator.
_ARITH_OP_GAP = re.compile(
    r"^\s*(?:" + _ARITH_SCALE + r"\b)?\s*(?:" + _ARITH_CUR + r")?\s*"
    r"(?P<op>[-+−–—])\s*(?:" + _ARITH_CUR + r")?\s*$", re.I)
_ARITH_EQ_GAP = re.compile(
    r"^\s*(?:" + _ARITH_SCALE + r"\b)?\s*(?:" + _ARITH_CUR + r")?\s*"
    r"=\s*(?:" + _ARITH_CUR + r")?\s*$", re.I)
#: 'X is <result> HIGHER THAN Y'. The cue must be followed by `than` with no
#: figure in between, which is what makes the comparative a two-place one and
#: fixes which side of it each operand is on.
_ARITH_MORE = {"more", "higher", "greater", "larger"}
_ARITH_CMP = re.compile(
    r"^[^0-9=]{0,40}?\b(?P<cue>more|higher|greater|larger|less|lower|smaller)"
    r"\b[^0-9=]{0,40}?\bthan\b", re.I)
#: 'DIFFERENCE OF / BETWEEN' WAS BUILT HERE AND THEN REMOVED, and the reason
#: is that the phrase does not say which figure is the result. 'A difference
#: of 131.5 M USD between 150 M USD and 18.5 M USD' puts the result right
#: after the preposition; 'the difference between 150 M USD and 18.5 M USD'
#: puts an OPERAND in the same place, and 'the difference of X and Y' — a
#: real if formal construction — puts one there too. Reading the first figure
#: after the phrase as the result therefore turns a correct sentence into
#: |18.5 - 131.5| = 150 and reports a contradiction the answer never made.
#: The two forms below are two-place and positional, so each of them says
#: which figure is which; this one does not, and no such claim exists in
#: releases 6-8. Pinned by `test_a_difference_phrase_does_not_qualify`.


@dataclass(frozen=True)
class Derivation:
    """One arithmetic derivation a claim states about its own figures."""
    form: str                       # equation | comparative | difference
    op: str                         # + | -
    left: Amount
    right: Amount
    result: Amount
    currency: Optional[str]
    unit: Optional[str]
    mult: float
    computed: float
    stated: float
    tolerance: float
    directed: bool = True

    @property
    def correct(self) -> bool:
        a, b = (self.computed, self.stated) if self.directed \
            else (abs(self.computed), abs(self.stated))
        return abs(a - b) <= self.tolerance

    def _fmt(self, value: float) -> str:
        v = value / (self.mult or 1.0)
        s = f"{v:.4f}".rstrip("0").rstrip(".")
        return f"{s} {self.unit} {self.currency}".replace(" None", "").strip()

    @property
    def shown(self) -> str:
        return f"{self.left.raw} {self.op} {self.right.raw} = {self.result.raw}"

    @property
    def support_reason(self) -> str:
        return ("the answer's own operands are printed in the cited evidence "
                f"and its arithmetic holds ({self.shown})")

    @property
    def failure_reason(self) -> str:
        return (f"the answer's own arithmetic does not hold: {self.left.raw} "
                f"{self.op} {self.right.raw} is {self._fmt(self.computed)}, "
                f"the answer states '{self.result.raw}'")


def _same_span(a: Amount, b: Amount) -> bool:
    """One printed figure, identified by WHERE it stands in the claim.

    Not object identity: `derived_probe` re-reads the claim's text, so the
    Amount it holds for a figure is a different object from the one the
    derivation was read off. Not equality either: two identical printings of
    the same number in one sentence are two figures, and the operand is the
    one at the operator.
    """
    return a.at == b.at and a.num == b.num


def _term_mult(a: Amount, d: "Derivation") -> float:
    """The scale one figure of a derived claim is stated at.

    Its own scale word when it prints one; the derivation's when it prints no
    mark at all and therefore inherits; and 1 when it prints a currency and no
    scale word, which is a figure written out in full.
    """
    if a.unit:
        return _UNIT_MULT.get(a.unit, 1.0)
    return 1.0 if a.currency else d.mult


def _same_mantissa(a: Amount, b: Amount) -> bool:
    """The same printed number, scale words ignored — '131.5 M USD' and the
    '131.5' of the equation beside it are one figure written twice."""
    tol = max(granularity(a.num, 1.0), granularity(b.num, 1.0)) * 0.5 + 1e-6
    return abs(a.bare - b.bare) <= tol


def _arith_context(triple: Sequence[Amount], local: Sequence[Amount]
                   ) -> Optional[Tuple[Optional[str], Optional[str]]]:
    """The ONE (currency, scale) the whole derivation is stated in, or None.

    Marks are read off the three terms AND off any other printing of one of
    their mantissas in the same claim: '(150.0 - 18.5 = 131.5)' writes three
    bare numbers, and the sentence around it writes '131.5 M USD'. That is
    where the unit comes from, and the link is the mantissa — never proximity,
    never the claim's other figures.

    Two disagreeing currencies, or two disagreeing scale words, return None:
    the claim does not qualify and keeps the verdict it has today. A term that
    prints a mark must print the resolved one exactly; a term that prints none
    inherits. A derivation with no mark anywhere is dimensionless and is
    refused here — that is the money scope, stated once.
    """
    curs = {a.currency for a in triple if a.currency}
    units = {a.unit for a in triple if a.unit}
    for a in local:
        if any(a is t for t in triple) or a.value is None:
            continue
        if not (a.currency or a.unit):
            continue
        if not any(_same_mantissa(a, t) for t in triple):
            continue
        if a.currency:
            curs.add(a.currency)
        if a.unit:
            units.add(a.unit)
    if len(curs) > 1 or len(units) > 1:
        return None                     # cross-currency, or two printed scales
    if not curs and not units:
        return None                     # dimensionless: not this rule's scope
    cur = next(iter(curs), None)
    unit = next(iter(units), None)
    if unit is None and any(a.bare < _ARITH_MONEY_FLOOR for a in triple):
        # A CURRENCY IS NOT ALWAYS A MARK. `iter_amounts` attaches one from up
        # to 40 characters further along the same line, so 'FP152 covers 7
        # more provinces than FP151 (12 - 5 = 7), out of a 28 M USD request'
        # hands 'USD' to a province count and the money scope would let it in.
        # With no scale word to pin them, the terms have to clear the
        # registry's own money floor — the same 1e4 `iter_amounts(money_only)`
        # applies to 'a bare 30 next to a financing label'. Pinned by
        # `test_a_scale_word_may_not_be_borrowed_from_an_unrelated_figure`.
        return None
    for a in triple:
        if not (a.currency or a.unit):
            continue                    # bare: inherits both
        if (a.currency and a.currency != cur) or a.unit != unit:
            return None                 # a term printed in another scale/currency
    return cur, unit


def _arith_shapes(body: str, local: Sequence[Amount]):
    """(form, op, left, right, result) for every derivation shape in a claim.

    Position-anchored throughout: an operand is a figure standing in a
    specific place relative to the operator or the comparative, never 'some
    figure in the sentence that happens to make the sum work'. Searching for
    a triple that adds up is how a rule like this starts certifying
    coincidences.
    """
    usable = sorted((a for a in local if a.value is not None),
                    key=lambda a: a.at)

    def gap(a: Amount, b: Amount) -> str:
        return body[a.at + len(a.num):b.at]

    # FORM A — the written equation 'x <op> y = z'. Three CONSECUTIVE printed
    # figures with nothing but the operator and the equals sign between them.
    for i in range(len(usable) - 2):
        x, y, z = usable[i], usable[i + 1], usable[i + 2]
        m = _ARITH_OP_GAP.match(gap(x, y))
        if m and _ARITH_EQ_GAP.match(gap(y, z)):
            yield "equation", ("+" if m.group("op") == "+" else "-"), x, y, z

    # FORMS B — prose. Every term must carry its own currency or scale word:
    # prose states no operator, so the marks are the only thing that pins the
    # three figures to one quantity, and the equation form's inheritance would
    # be resolving an ambiguity the text never wrote down.
    marked = [a for a in usable if a.currency or a.unit]
    for k, z in enumerate(marked):
        before, after = marked[:k], marked[k + 1:]
        m = _ARITH_CMP.match(body[z.at + len(z.num):])
        # 'A ... is z <cue> than B' — the operands STRADDLE the result, which
        # is the grammar of the comparative and is what says which of them is
        # the minuend. Both on one side is a sentence whose direction cannot
        # be read off the text, so it does not qualify.
        if m and len(before) == 1 and len(after) == 1:
            hi, lo = (before[0], after[0]) \
                if m.group("cue").lower() in _ARITH_MORE else (after[0], before[0])
            yield "comparative", "-", hi, lo, z


def derived_arithmetic(claim: Claim) -> Optional[Derivation]:
    """The derivation this claim states, or None when it states none.

    PURE READING. It decides nothing about evidence and nothing about a
    verdict — it says what the answer's own words compute, so the caller can
    check the operands against the evidence and the arithmetic against
    itself.
    """
    body = _strip_citations(claim.text)
    local = amounts(body)
    for form, op, x, y, z in _arith_shapes(body, local):
        ctx = _arith_context((x, y, z), local)
        if ctx is None:
            continue
        cur, unit = ctx
        mult = _UNIT_MULT.get(unit or "", 1.0)
        val = [a.bare * mult for a in (x, y, z)]
        grains = [granularity(a.num, mult) for a in (x, y, z)]
        return Derivation(
            form=form, op=op, left=x, right=y, result=z,
            currency=cur, unit=unit, mult=mult,
            computed=(val[0] + val[1]) if op == "+" else (val[0] - val[1]),
            stated=val[2], tolerance=max(grains) * 0.5 + 1e-6,
            directed=(form != "difference"))
    return None


def derived_probe(claim: Claim, d: Derivation) -> Claim:
    """The claim this module actually checks once a derivation is read off it.

    The same claim with its arithmetic accounted for: the two operands
    resolved into the derivation's own currency and scale so the ordinary
    matchers can look for them, and the DERIVED figure — every printing of
    it — dropped, because no page is expected to print it. Nothing else about
    the claim moves: same text, same field, same entities, same lead-in. Every
    later check in `classify_deterministic` then runs unchanged, on this.
    """
    def resolved(a: Amount) -> Amount:
        return dataclasses.replace(
            a, value=round(a.bare * d.mult, 2), currency=a.currency or d.currency,
            unit=a.unit or d.unit, grain=granularity(a.num, d.mult),
            alt=None, alt_grain=0.0)

    keep: List[Amount] = []
    for a in amounts(_strip_citations(claim.text)):
        if _same_span(a, d.left) or _same_span(a, d.right):
            keep.append(resolved(a))
            continue
        mult = _term_mult(a, d)
        tol = max(granularity(a.num, mult),
                  granularity(d.result.num, d.mult)) * 0.5 + 1e-6
        if a.value is not None and abs(a.bare * mult - d.stated) <= tol:
            continue                    # another printing of the derived figure
        keep.append(a)
    return dataclasses.replace(claim, amounts=keep)


@dataclass
class Verdict:
    claim: Claim
    status: str
    reason: str
    scope: List[EvidenceKey] = dc_field(default_factory=list)
    source: str = "deterministic"        # deterministic | llm
    flags: List[str] = dc_field(default_factory=list)
    plausible: bool = False              # residue worth one LLM adjudication

    @property
    def failed(self) -> bool:
        return self.status in (CONTRADICTED, UNSUPPORTED)


def _scopes(claim: Claim, evidence: Evidence
            ) -> Tuple[List[EvidenceKey], List[EvidenceKey], List[str], List[EvidenceKey]]:
    """(strict scope, same-document fallback, unresolvable citations, ruling-5 scope).

    Strict scope is what the claim actually cites. The fallback widens to the
    rest of the cited document: a figure that is real but attached to the
    wrong page is a citation defect, not an invented fact, and the two deserve
    different verdicts. The ruling-5 scope is every held key of a document
    cited WITHOUT a page — a coarse bracket, satisfied by any key of the
    document it names, and reported as a coarse citation rather than a
    silent one.

    NOTE-PAGE SCOPE. A cited (doc, page) that no held key carries resolves,
    additionally, to the NOTE LINE that printed that page for that document —
    and to nothing else. The computed notes publish their provenance, the
    prompt tells the model to cite the page a row prints, and both other
    instruments (``chainlit_app._invalid_citations``,
    ``eval_answers.score_answer``) already count such a page as a legal
    target; this verifier called it 'cited evidence was never retrieved' and
    scored UNSUPPORTED, which cost release-3 the second half of two conflict
    reports the registry note had ORDERED the answer to write
    (conf-fp153-gcf, fr-fp172-nepal).

    It is a SCOPE fix and nothing more. The note line is then read by the same
    matchers at the same strictness as any other scope: a claim whose value
    the line does not print still fails, and a page no note line printed for
    that document still lands in ``bad`` exactly as before. The lookup is
    reached only when ``(d, page)`` is absent, so a page that WAS retrieved
    keeps being judged on the page's own text with no note text mixed in.
    """
    docs = {k[0] for k in evidence if not is_notes_doc(k[0])}
    strict: List[EvidenceKey] = []
    wide: List[EvidenceKey] = []
    widened: List[EvidenceKey] = []
    bad: List[str] = []
    for c in claim.citations:
        if c.doc is None:
            # only a bracket that READS as a pointer at the computed notes
            # resolves to them. A bare '[p. 5]' names no document at all: it
            # is a broken citation, and letting it land on the notes made
            # every page-less page reference verify against a year note.
            if c.kind == "note" and NOTES_KEY in evidence:
                strict.append(NOTES_KEY)
            else:
                bad.append(c.raw.strip()[:60])
            continue
        d = _resolve_doc(c.doc, docs)
        if d is None:
            bad.append(f"{c.doc}" + (f", p.{c.page}" if c.page else ""))
            continue
        wide += [k for k in evidence if k[0] == d]
        if c.page is None:
            # RULING 5 (docs/adjudication-taxonomy.md): a bracket naming only
            # the document ('[doc]', '[doc, cover pages]') is satisfied by ANY
            # held evidence key of that document. That widening is returned
            # SEPARATELY, as ``widened``, and not folded into ``strict``:
            # collapsing the two made strict == wide, which silently killed the
            # intra-document conflict detector for every page-less bracket and
            # dropped the citation-page-mismatch caution the head emitted.
            here = [k for k in evidence if k[0] == d and k[1] is None]
            strict += here or [k for k in evidence if k[0] == d]
            widened += [k for k in evidence if k[0] == d]
        elif (d, c.page) in evidence:
            strict.append((d, c.page))
        elif (note_scope_doc(d), c.page) in note_scopes(evidence):
            # the page a computed note printed for THIS document, on a line
            # that named THIS document. Not added to `wide`: the note's text
            # is already inside the cited document's own doc-level key, so
            # widening the doc-wide pool here would only duplicate it under a
            # second name and change which key a conflict is reported on.
            strict.append((note_scope_doc(d), c.page))
        else:
            bad.append(f"{d}, p.{c.page}")
    ded = lambda xs: list(dict.fromkeys(xs))     # noqa: E731
    keep = ded(strict)
    return keep, ded(wide), ded(bad), [k for k in ded(widened) if k not in keep]


def _text_at(evidence: Evidence, key: EvidenceKey) -> str:
    """The text ONE scope key carries: a held key, or a derived note scope.

    Every read of a scope goes through here. ``evidence.get(k, "")`` scattered
    across the conflict scans is how a scope that resolves would come back
    EMPTY on one path and populated on another — verified against a note line
    and conflict-tested against nothing.
    """
    if key in evidence:
        return evidence[key]
    if is_notes_doc(key[0]) and key[1] is not None:
        return note_scopes(evidence).get(key, "")
    return ""


def _text_of(evidence: Evidence, keys: Sequence[EvidenceKey]) -> str:
    return "\n".join(t for t in (_text_at(evidence, k) for k in keys) if t)


# A field label only states that field when it heads its line: template
# instructions ('In case of a multi-country-region program, specify indicative
# requested GCF funding amount for each country') and narrative prose mention
# the same words and were the source of every false contradiction measured on
# the recorded answers.
_FIELD_PREFIX_OK = re.compile(r"^[\s|#>*\-–—•()a-z0-9.,:]{0,40}$")
_INSTRUCTION_RE = re.compile(
    r"\b(?:in case of|please|specify|indicate|enter\s+(?:number|amount)|"
    r"if applicable|choose an item|for each|see annex|e\.g\.|guidance)\b", re.I)


def _value_after(line: str, at: int, window: int = 80) -> List[Amount]:
    """Amounts printed just after a field label — its value, not the line's."""
    return list(iter_amounts(line[at:at + window]))


def _field_lines(text: str, rx) -> Iterable[Tuple[str, int]]:
    """Segments of ``text`` whose own head is the field label.

    Segments, not lines: the registry publishes a whole document on one line
    ('Registry — FP151: "..."; accredited entity: ...; GCF financing (as
    printed): 18.5 M USD'), where every field heads its semicolon-separated
    part. A label buried mid-sentence is prose about the field, not the field.
    """
    for line in (text or "").splitlines():
        if _INSTRUCTION_RE.search(line):
            continue
        for seg in re.split(r"(?<=[;|])|(?<=\.\s)", line):
            m = rx.search(seg)
            if not m or not _FIELD_PREFIX_OK.match(seg[:m.start()]):
                continue                 # the label is buried in prose
            yield seg, m.end()


def _money_like(a: Amount) -> bool:
    return bool(a.currency or a.unit or a.clash or (a.value or 0) >= 1e4)


def _field_conflict(claim: Claim, text: str,
                    also_reported: Sequence[Amount] = (),
                    settled: Optional[Callable[[Amount], bool]] = None
                    ) -> Optional[Tuple[Amount, str]]:
    """A figure printed under the claim's own field label that disagrees.

    Line-scoped and label-anchored: the registry prints 'GCF financing (as
    printed): 28,654 million USD; total financing (as printed): 49,654 million
    USD' on ONE line, so the value of a field is the first amount AFTER its
    label, not the first amount on the line.

    ``also_reported`` are the figures OTHER claims of the same answer state for
    the same document. A registry conflict note says, verbatim, 'report both
    figures with their pages'; an answer that obeys it prints one figure per
    bullet, and reading each bullet alone turns compliance into a
    contradiction — 14 adjudicated claims (claim-11d3a178, claim-1cdbc791,
    claim-ca1c1388, claim-d43027b0, claim-4650af24, ...) failed exactly that
    way. A disagreeing figure the answer ITSELF reports for this document is
    the instructed behaviour, not a contradiction to flag.

    ``settled`` decides ONE rival print at a time (see ``_key_conflict``); a
    rival it skips does not end the scan, so a second, unsettled rival printed
    further down the same page is still reported.
    """
    field = claim_field(claim.text)
    if not field or not claim.amounts:
        return None
    rx = dict(_FIELD_RES)[field]
    lines = list(_field_lines(text, rx))
    for line, at in lines:
        for cand in _value_after(line, at)[:2]:
            if any(amount_matches(cand, a) for a in claim.amounts):
                return None                       # the field agrees somewhere
    for line, at in lines:
        for cand in _value_after(line, at)[:1]:
            if not _money_like(cand) or any(amount_matches(cand, a)
                                            for a in claim.amounts):
                continue
            if any(amount_matches(cand, a) for a in also_reported):
                return None                       # the answer reports both
            if settled is not None and settled(cand):
                continue          # the registry recorded this print, and not
                                  # as a conflict — keep looking on this page
            return cand, line.strip()[:200]
    return None


# claim field -> registry v2 field, for the conflicts the corpus already knows
_V2_FIELD = {"gcf_financing": "gcf_funding_requested",
             "total_financing": "total_financing",
             "co_financing": "co_financing",
             "duration": "implementation_period",
             "beneficiaries": "beneficiaries_direct"}


def _as_amount(raw: str) -> Optional[Amount]:
    got = amounts(raw or "")
    return got[0] if got else None


def registry_conflict(doc_id: str, claim: Claim,
                      also_reported: Sequence[Amount] = ()) -> Optional[str]:
    """A conflict the fact registry already recorded for this document/field.

    Retrieval is a sample: the page that disagrees may simply not be in this
    turn's ten excerpts (measured: FP153's p.48 was not). The registry is not
    a sample — ``scripts/build_registry_v2.py`` scanned every page and marked
    the disagreeing candidate 'conflicting' — so an answer that states one
    side of a known conflict as the figure can be caught deterministically,
    with the other side's page, no matter what retrieval returned.
    """
    v2 = _V2_FIELD.get(claim_field(claim.text) or "")
    if not v2 or not claim.amounts:
        return None
    try:
        from gcf_qna.rag import registry
        cands = registry.facts(doc_id).get(v2) or []
    except Exception:                # the registry is an enhancement, never a blocker
        return None
    others = [c for c in cands if c.get("status") == "conflicting"]
    if not others:
        return None
    stated = [c for c in cands if c.get("status") != "conflicting"
              and any((a := _as_amount(c.get("raw", ""))) and amount_matches(a, x)
                      for x in claim.amounts)]
    if not stated:
        return None                  # the answer states neither side; other checks own it
    # An answer that already reports BOTH figures is the behaviour the prompt
    # asks for, not a contradiction to flag. The reporting is ANSWER-level,
    # not sentence-level: the registry note says 'report both figures with
    # their pages', and an answer that obeys prints one figure per bullet, so
    # ``also_reported`` carries what the answer's other claims state for this
    # same document (claim-03d4cab1, claim-0c2cdfab, claim-0ceca63e,
    # claim-4b104f74, claim-5b351da8, claim-79a9c71a, claim-bdadca96,
    # claim-e33e0ea0, claim-ea234d3b).
    reported = list(claim.amounts) + list(also_reported)
    if any((a := _as_amount(c.get("raw", ""))) and
           any(amount_matches(a, x) for x in reported) for c in others):
        return None
    alt = others[0]
    return (f"the corpus registry records a conflicting figure in this document: "
            f"'{stated[0].get('raw')}' (p.{stated[0].get('page')}) vs "
            f"'{alt.get('raw')}' (p.{alt.get('page')}) — both must be reported")


def _check_amounts(claim: Claim, text: str) -> Tuple[bool, List[Amount]]:
    """Are all the claim's figures printed in this evidence text?"""
    ev = amounts(text)
    missing = [a for a in claim.amounts
               if not any(amount_matches(a, e) for e in ev)]
    return (not missing), missing


def _registry_blob(doc_id: str) -> str:
    """Everything the registry prints for a document, as one searchable text.

    Both schemas: the v1 cover-page row (title, entity, countries) and every
    v2 candidate's raw source string.
    """
    parts: List[str] = []
    try:
        from gcf_qna.rag import registry
        row = registry.load().get(doc_id) or {}
        for v in row.values():
            if isinstance(v, str):
                parts.append(v)
            elif isinstance(v, list):
                parts += [str(x) for x in v]
        for cands in (registry.facts(doc_id) or {}).values():
            parts += [str(c.get("raw") or "") for c in cands]
    except Exception:               # the registry is an enhancement, never a blocker
        return ""
    return "\n".join(parts)


def registry_named(doc_id: str, names: Sequence[Sequence[str]]) -> Optional[str]:
    """Names the registry records for this document, in any of its fields.

    Same argument as ``registry_backed``, for the text facts: '[doc] mentions
    REDD+ meetings in Ecuador' cites the Ecuador document, whose registry row
    says countries: Ecuador — the answer is right even when the ten retrieved
    passages happen not to print the country name.
    """
    if not names:
        return None
    hay = norm_text(_registry_blob(doc_id))
    if not hay:
        return None
    for variants in names:
        if not any(norm_text(v) and norm_text(v) in hay for v in variants):
            return None
    return "registry row for this document names " + ", ".join(
        f"'{vs[0]}'" for vs in names)


def registry_backed(doc_id: str, claim: Claim,
                    want: Sequence[Amount]) -> Optional[str]:
    """Figures the fact registry records for this document, at any status.

    Retrieval returns ten passages; the registry scanned every page. An answer
    that correctly reports BOTH sides of a known conflict cites a second page
    that this turn may not hold — so without this check no conflict-aware
    answer could ever verify, and the detector would be punishing exactly the
    honesty it exists to encourage.
    """
    if not want:
        return None
    try:
        from gcf_qna.rag import registry
        facts = registry.facts(doc_id)
    except Exception:               # the registry is an enhancement, never a blocker
        return None
    if not facts:
        return None
    field = _V2_FIELD.get(claim_field(claim.text) or "")
    fields = [field] if field and field in facts else list(facts)
    where: List[str] = []
    for a in want:
        hit = None
        for f in fields:
            for cand in facts.get(f) or []:
                got = _as_amount(cand.get("raw", ""))
                if got and amount_matches(got, a):
                    hit = f"'{cand.get('raw')}' (p.{cand.get('page')})"
                    break
            if hit:
                break
        if not hit:
            return None             # every missing figure must be backed
        where.append(hit)
    return "; ".join(dict.fromkeys(where))


# ---------------------------------------------------------------------------
# CLASS 1 — acronym vs expansion.  Row `cid-fp0086-padded`: the answer writes
# "ESIA/ESMP (if applicable)" and cited page 39 prints "Environmental and
# Social Impact Assessment (ESIA) or Environmental and Social Management Plan
# (if applicable)". `ESIA` matches; `ESMP` is the answer's own compression of a
# phrase the page prints IN FULL.
#
# ONE DIRECTION ONLY, AND THE ASYMMETRY IS THE POINT.
#
#   acronym claimed, expansion PRINTED  -> accepted here. The initialism is a
#       lossy FUNCTION of a string the evidence actually contains: to satisfy
#       it, the evidence has to print all four words, in order, adjacent.
#       Nothing is invented — the claim says strictly less than the page.
#
#   expansion claimed, acronym printed  -> REFUSED, and stays refused. A page
#       printing 'IFAD' does not say what IFAD stands for, so accepting
#       'International Fund for Agricultural Development' would be reading five
#       words out of four letters. That is the direction 'ADB (Asian
#       Development Bank of Wakanda)' rode in on, it is the direction the
#       fabricated arm mutates (``score_verifier.FAKE_EXPANSIONS``), and it is
#       pinned shut by ``test_an_acronym_never_vouches_for_a_spelled_out_name``.
#       Row `id-fp220-entity` is that direction and is deliberately NOT fixed.
#       Where the corpus itself records the pairing, ``registry_named`` already
#       supplies it, from the registry rather than from initials.
#
# The gates, each with an adversarial test that varies it and nothing else:
#   * >= 3 letters — two-letter initialisms ('AE', 'EE') collide with ordinary
#     capitalised pairs, and the corpus is full of both.
#   * exact letter sequence, one word per letter, in order.
#   * ADJACENT words: only a connective ('of', 'and', 'for', 'the', ...) may
#     stand between two of them. A phrase that merely CONTAINS the letters in
#     order is not an expansion of the acronym.
#   * each expansion word is capitalised where the evidence prints it.
# ---------------------------------------------------------------------------
_BARE_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9]{2,7}$")
#: what may sit between two words of an expansion: connectives, nothing else
_ACR_GAP = (r"(?:\s+(?i:of|and|for|the|in|on|to|a|an|by|de|du|des|d’|d'|"
            r"la|le|les|et|pour|sur|aux|au))*[\s\u00a0]+")


@functools.lru_cache(maxsize=512)
def _expansion_re(acr: str):
    """A pattern matching the spelled-out form of ``acr``, and only that."""
    return re.compile(r"(?<![A-Za-z])"
                      + _ACR_GAP.join(letter + r"[\w’'\-]*" for letter in acr))


def _spelled_out_in(variants: Sequence[str], text: str) -> Optional[str]:
    """The phrase this evidence prints in full that ``variants`` abbreviates."""
    for v in variants:
        acr = (v or "").strip()
        if not _BARE_ACRONYM_RE.match(acr):
            continue
        m = _expansion_re(acr).search(text or "")
        if m:
            return m.group(0)
    return None


def _check_entities(claim: Claim, text: str) -> Tuple[bool, List[List[str]]]:
    """(ok, the variant lists that appear nowhere in this text)."""
    hay = norm_text(text)
    missing = []
    for variants in claim.entities:
        if any(norm_text(v) and norm_text(v) in hay for v in variants):
            continue
        if _spelled_out_in(variants, text):
            continue                    # CLASS 1: the page prints it in full
        missing.append(variants)
    return (not missing), missing


def _check_years(claim: Claim, text: str) -> Tuple[bool, List[str]]:
    """(ok, missing) for the year and board tokens a claim states.

    Wave 2 briefly let a board token be satisfied by the CITED DOCUMENT'S ID
    ('B.27' inside ``124_gcf-b27-02-add11``). It was reverted: a year/board
    claim is checked by its token and nothing else, so satisfying the token
    from the citation left the predicate unverified — 'GCF/B.27/02/Add.11 was
    withdrawn by the Board' verified against a page about formatting. It also
    earned zero adjudicated rows once the other changes were in place.
    """
    hay = text or ""
    want = set(_YEAR_RE.findall(_strip_citations(claim.text))) | \
        {m.group(0) for m in _BOARD_RE.finditer(_strip_citations(claim.text))}
    missing = [w for w in sorted(want) if w.replace(" ", "") not in hay.replace(" ", "")]
    return (not missing), missing


def _check_lead_in(claim: Claim, text: str) -> Tuple[bool, str]:
    """(ok, what is missing) for the content a dropped lead-in handed over.

    THE OTHER HALF OF RULING 6. Extraction stops minting heading-form units as
    claims; if that were all, '**FP151 (total financing: USD 999 million)**:'
    would delete a fabricated figure from verification altogether. The figure
    is carried onto the claim the lead-in introduces and checked here.

    THE SCOPE IS THE CITED DOCUMENT, not the cited page, and that is measured
    rather than chosen for comfort. A lead-in cites nothing — it names what the
    block below is about — so the honest question is whether the DOCUMENT the
    block cites prints it. Checking it against the page instead cost a
    contradiction: 'For FP274 ("Building the Climate Resilience ... (BRACE)"),
    the excerpts show conflicting figures:' hands the title to a bullet citing
    p.40, p.40 does not print the title, and the bullet came back UNSUPPORTED
    on the title instead of CONTRADICTED on its figure — the arm went 34/34 to
    33/34. The caller therefore hands this the strict, ruling-5 and widened
    keys together, and the result gates SUPPORT only.

    It can only ever make a claim fail, and it never preempts the conflict
    path. Carried content is deliberately not given to the conflict detector
    either: which field a claim is about is read from the claim's own text, and
    attributing a heading's figure to the field of the line below it would
    invent an attribution nobody wrote.
    """
    if not claim.carries_lead_in:
        return True, ""
    ev = amounts(text)
    hay = norm_text(text)
    missing: List[str] = []
    missing += [a.raw for a in claim.lead_in_amounts
                if not any(amount_matches(a, e) for e in ev)]
    for variants in claim.lead_in_entities:
        if any(norm_text(v) and norm_text(v) in hay for v in variants):
            continue
        if _spelled_out_in(variants, text):
            continue
        missing.append(variants[0])
    missing += [y for y in claim.lead_in_years
                if y.replace(" ", "") not in (text or "").replace(" ", "")]
    return (not missing), ", ".join(missing)


def _verify_against(claim: Claim, text: str) -> Tuple[bool, str]:
    """(ok, what is missing) for one claim against one blob of evidence.

    The claim's OWN asserted content only. Content a heading-form lead-in
    handed it is checked separately and on a different scope
    (``_check_lead_in``), so this stays exactly the per-scope question every
    branch of ``classify_deterministic`` already asks it.
    """
    if claim.kind in ("money", "number") and claim.amounts:
        ok, missing = _check_amounts(claim, text)
        return ok, ", ".join(a.raw for a in missing)
    if claim.kind == "entity" and claim.entities:
        ok, missing = _check_entities(claim, text)
        return ok, ", ".join(vs[0] for vs in missing)
    if claim.kind in ("year", "existence"):
        ok, missing = _check_years(claim, text)
        return ok, ", ".join(missing)
    # a claim whose trigger left nothing normalizable (a bare number with no
    # unit, say): fall back to the entity check, then give it to the judge
    ok, missing = _check_entities(claim, text)
    return ok, ", ".join(vs[0] for vs in missing)


def registry_records(doc_id: str, field: Optional[str], want: Amount) -> bool:
    """Does the corpus registry print ``want`` for this document and field?

    The 'report both figures' relaxation exists because a registry CONFLICT
    note tells the answer to report both. That instruction is the licence, so
    the licence is checked: a sibling figure the registry does not record for
    this field is not the conflict's other side, it is just another number in
    the answer, and letting it suppress the verdict is how 'GCF funding
    requested: 40,751,254' sat next to a mislabelled 'roughly USD 38,000,000
    was leveraged from partners' and both verified clean.
    """
    v2 = _V2_FIELD.get(field or "")
    if not v2:
        return False
    try:
        from gcf_qna.rag import registry
        cands = (registry.facts(doc_id) or {}).get(v2) or []
    except Exception:           # the registry is an enhancement, never a blocker
        return False
    return any((a := _as_amount(c.get("raw", ""))) and amount_matches(a, want)
               for c in cands)


def registry_ruled_compatible(doc_id: str, field: Optional[str],
                              stated: Sequence[Amount], rival: Amount) -> bool:
    """Has the registry READ BOTH prints and filed the rival as not-a-conflict?

    ``scripts/build_registry_v2.py`` scans every page of a document, elects one
    ``canonical`` value per field, and then rules on every other print of that
    field: ``conflicting`` when it "IS comparable with the canonical one ...
    and disagrees", ``supporting`` when it "agrees with the canonical one, or
    is not comparable with it (no parsed value / incompatible currency / text
    field / a figure far below the canonical total, i.e. a component or a
    tranche). Never an assertion of conflict." Deferring to that ruling is the
    only thing that may outrank a page-level disagreement, and it is claimed
    only when the registry actually read the pair:

    1. **the answer states the CANONICAL value** for this document and field.
       The registry elected it; the answer is repeating the registry's own
       reading. Drop this clause and the pair inverts — 'Total financing:
       **$100,000,000** [p.55]' would suppress FP152's canonical 720 M USD on
       p.5 and a per-project tranche would verify clean as the programme
       total.
    2. **the registry RECORDED the rival print** for the same field. SILENCE
       IS NOT A RULING: this is the clause whose absence sank the deleted
       ``registry_settled``, where a document whose registry knows 26,736,295
       verified clean against a p.99 printing 999,111,222, with no flag at
       all. An unbuilt registry, an unregistered document and an unmapped
       field are all silence and all reach the same answer here.
    3. **every record of the rival is filed ``supporting``.** ``conflicting``
       is the registry saying these ARE two readings of one field, which is
       the contradiction, not an excuse for it; ``canonical`` cannot occur
       under clause 1 but is excluded anyway rather than argued about.

    Only the money/duration fields ``_V2_FIELD`` maps are askable; for every
    other field the registry has no opinion to defer to.
    """
    v2 = _V2_FIELD.get(field or "")
    if not v2 or not stated:
        return False
    try:
        from gcf_qna.rag import registry
        cands = (registry.facts(doc_id) or {}).get(v2) or []
    except Exception:           # the registry is an enhancement, never a blocker
        return False
    canon = next((c for c in cands if c.get("status") == "canonical"), None)
    ca = _as_amount(canon.get("raw", "")) if canon else None
    if ca is None or not any(amount_matches(ca, s) for s in stated):
        return False                                        # clause 1
    recorded = [c for c in cands
                if (a := _as_amount(c.get("raw", ""))) and amount_matches(a, rival)]
    if not recorded:
        return False                                        # clause 2 — SILENCE
    return all(c.get("status") == "supporting" for c in recorded)   # clause 3


def _key_conflict(claim: Claim, evidence: Evidence, keys: Sequence[EvidenceKey],
                  also: Sequence[Amount] = (), registry_rulings: bool = True
                  ) -> Tuple[Optional[Tuple[Amount, str]], Optional[EvidenceKey]]:
    """The first held key that prints a DIFFERENT value under the claim's field.

    One key at a time, deliberately. ``_field_conflict`` stops at the first
    page that agrees, so handing it several concatenated keys let a document
    that prints 40,751,254 on p.5 and 38,000,000 on p.48 verify clean against
    a claim citing the document as a whole.

    A ``registry_settled`` escape once sat here, deferring to the registry
    whenever it recorded the CLAIM's figure and had not marked the rival
    'conflicting'. It was deleted: the registry's SILENCE about a figure is
    not a ruling that the figure is compatible, so a document whose registry
    knows 26,736,295 while p.99 prints 999,111,222 verified clean with no
    flag at all. It earned no adjudicated row.

    What is here instead is the narrower condition the review that rejected it
    specified: defer only when the registry has actually RECORDED THE RIVAL
    print for this document and field — the print about to be called a
    contradiction, not just the claim's own figure — and filed it
    ``supporting`` rather than ``conflicting`` (``registry_ruled_compatible``,
    which also keeps ``registry_settled``'s requirement that the answer be
    stating the registry's own canonical reading). Silence still suppresses
    nothing, so the 26,736,295/999,111,222 probe stays CONTRADICTED.

    The row it exists for is ``id-fp152-financing``. Its document prints
    'A7. Total financing (SCF + co-finance) 720 M USD' on p.5 and, on p.55,
    '(a) Total project financing: $100,000,000' inside an E.2.2 cost-per-tonne
    calculation whose sibling line reads '(b) Expected GCF contribution:
    $75,000,000' — half the programme's own 150 M USD, which is what makes it
    a per-project template row and not the programme total. Same words, a
    different scope. The registry filed it 'supporting' (a figure far below
    the canonical total is a component or a tranche), so it is settled, while
    the 210 candidates it filed 'conflicting' corpus-wide are not.

    The predicate is applied PER RIVAL, not per key: a settled print does not
    excuse the next print on the same page.
    """
    field = claim_field(claim.text)
    for k in keys:
        settled = None
        if registry_rulings and not is_notes_doc(k[0]):
            settled = (lambda rival, d=k[0]:
                       registry_ruled_compatible(d, field, claim.amounts, rival))
        got = _field_conflict(claim, _text_at(evidence, k), also, settled)
        if got:
            return got, k
    return None, None


def _reported_elsewhere(claims: Sequence[Claim],
                        scopes: Sequence[Tuple[List[EvidenceKey], ...]]):
    """``claim -> the figures the answer's OTHER claims state for ITS FIELD``.

    The registry's conflict note is an instruction to the ANSWER ('report both
    figures with their pages'), and an answer that obeys it prints one figure
    per bullet. Judging a bullet on its own therefore turns compliance into a
    contradiction, which is what 22 of the adjudicated false positives are.

    Three things scope it, and each closes a hole the earlier document-only
    version had:

    * the same DOCUMENT — a figure another claim attributes elsewhere says
      nothing about this one;
    * a COMPATIBLE field — a sibling that explicitly claims a DIFFERENT field
      is not reporting this one's other side, and letting it suppress the
      conflict made two transposed values license each other ('GCF funding
      requested: USD 38,000,000' beside 'total co-financing: USD 40,751,254'
      when the page prints them the other way round). A sibling with no field
      label is compatible: the bullets that obey the note print a bare figure
      and a section — '**USD 49,751,264** (p.8, A.10 "Grant")' — and demanding
      that each repeat the label costs seven adjudicated rows;
    * an UNAMBIGUOUS attribution — a claim whose chained brackets name several
      documents attributes its figure to none of them in particular, so it
      contributes to no pool rather than to all of them.

    The claim's own figures may sit in its pool: they cannot change anything,
    because ``_field_conflict`` never emits a candidate that matches one of
    them and ``registry_conflict`` already unions them into what it checks.
    An explicit self-exclusion was removed rather than tested — a gate whose
    removal cannot change a verdict is not a gate.
    """
    by_doc: Dict[str, List[Tuple[Optional[str], Amount]]] = {}
    slot: List[Optional[Tuple[str, Optional[str]]]] = []
    for c, (strict, wide, _bad, r5) in zip(claims, scopes):
        docs = {k[0] for k in list(strict) + list(wide) + list(r5)
                if not is_notes_doc(k[0])}
        field = claim_field(c.text)
        if len(docs) != 1:
            slot.append(None)
            continue
        doc = next(iter(docs))
        slot.append((doc, field))
        by_doc.setdefault(doc, []).extend((field, a) for a in c.amounts)

    index = {id(c): i for i, c in enumerate(claims)}

    def for_claim(claim: Claim) -> Tuple[Optional[str], Optional[str], List[Amount]]:
        i = index[id(claim)]
        if slot[i] is None:
            return None, None, []
        doc, field = slot[i]
        return doc, field, [a for f, a in by_doc.get(doc, ())
                            if f is None or f == field]

    return for_claim



def _conflict_before_support(claim: Claim, evidence: Evidence,
                             strict: Sequence[EvidenceKey],
                             wide: Sequence[EvidenceKey],
                             r5: Sequence[EvidenceKey],
                             also: Sequence[Amount] = (),
                             also_all: Sequence[Amount] = (),
                             registry_conflicts: bool = True,
                             cross_page_conflicts: bool = True
                             ) -> Optional[Tuple[str, Optional[str]]]:
    """The ONE gate between a verified claim and SUPPORTED: ``(reason, flag)``.

    THE INVARIANT: a verdict may not answer SUPPORTED on a scope it has not
    conflict-tested. Support was decided in three places — the strictly cited
    keys, the rest of a page-less bracket's document (ruling 5), and, when
    neither held the figure, the rest of a PAGED bracket's document — while
    the conflict test ran in the first two only. So a false figure escaped by
    moving its own citation: 'FP151 requests USD 28 million [doc, cover
    pages]' is CONTRADICTED (the cover line prints 18.5 M USD for GCF
    financing and 28 M USD for TOTAL financing), and the identical sentence
    re-pointed to p.45 — a page that prints 18,500,000 and never prints 28
    million — came back SUPPORTED with a citation-page-mismatch caution and
    nothing else. Same figure, same document, a strictly WORSE citation, a
    passing verdict; and, while the repair pass still existed, a rewrite whose
    only diff was the page number was adopted as a correction — zero factual
    change, a strictly worse citation, a green status. THIS IS A VERIFICATION
    DEFECT and it was fixed here, which is why deleting repair costs nothing
    against it. The registry-backed branch
    answered SUPPORTED from the registry with only the cited page tested, the
    same shape one branch over. One gate, called from all three, is the only
    arrangement in which a fourth support branch cannot reopen this.

    PER KEY, NEVER CONCATENATED — this is where the previous attempt died.
    ``_field_conflict`` stops at the first page that AGREES, so handing it a
    merged blob (``strict + wide_only``, wave-0c finding 2) lets the agreeing
    page hide the disagreeing one: it closed the reported row and left the
    vector open. ``_key_conflict`` takes one key at a time and keeps every
    escape the strict branch has, so widening the scope cannot widen the
    verdict: the answer's own 'report both figures with their pages'
    compliance (``also``), and a rival the registry itself read and filed
    'supporting' rather than 'conflicting' (``registry_ruled_compatible``,
    decided per rival, inside ``_key_conflict``).

    Scope, in the order a caution would name it: the strictly cited keys, then
    the rest of a page-less bracket's document, then the rest of every cited
    document — that last scan gated by ``cross_page_conflicts``, which is the
    switch that exists to name it. Nothing here is scoped to the branch that
    called it: the three callers differ in what made them SUPPORTED, not in
    which evidence may contradict them.
    """
    others = [k for k in wide if k not in strict and k not in r5]
    scans: List[Tuple[Sequence[EvidenceKey], Optional[str]]] = [
        (strict, None), (r5, "conflict-elsewhere-in-document")]
    if cross_page_conflicts:
        scans.append((others, "conflict-elsewhere-in-document"))
    for keys, tag in scans:
        if not keys:
            continue
        conflict, _key = _key_conflict(claim, evidence, keys, also,
                                       registry_conflicts)
        if conflict:
            cand, line = conflict
            return (f"the cited document also prints '{cand.raw}' for this field "
                    f"({line})"), tag
    if registry_conflicts:
        for d in dict.fromkeys(k[0] for k in list(strict) + list(wide)
                               if not is_notes_doc(k[0])):
            known = registry_conflict(d, claim, also_all)
            if known:
                return known, "known-document-conflict"
    return None


def classify_deterministic(claims: Sequence[Claim], evidence: Evidence,
                           cross_page_conflicts: bool = True,
                           registry_conflicts: bool = True) -> List[Verdict]:
    """Pure-python verdicts: no network, no model, no hidden state.

    * SUPPORTED    — every figure/name/year of the claim is printed in the
                     evidence the claim cites (or, flagged, elsewhere in the
                     cited document).
    * CONTRADICTED — the cited evidence prints a DIFFERENT value under the
                     same field label. With ``cross_page_conflicts`` the same
                     test runs over the rest of the cited document, which is
                     how 'p.5 says 28,654 / p.48 says 26,654' surfaces even
                     when the answer only cites the cover page.
    * UNSUPPORTED  — no citation, a citation pointing at evidence we do not
                     hold, or a value that appears nowhere in what it cites.
    """
    out: List[Verdict] = []
    all_text = "\n".join(evidence.values())
    scopes = [_scopes(c, evidence) for c in claims]
    elsewhere = _reported_elsewhere(claims, scopes)
    for c, (strict, wide, bad, r5) in zip(claims, scopes):
        flags = ["invalid-citation:" + b for b in bad]
        also_doc, also_field, also_all = elsewhere(c)
        # `also` excuses a field conflict only for a figure the registry
        # records for this document and field; `registry_conflict` matches
        # against its own recorded candidates and needs no second filter.
        also = [a for a in also_all
                if also_doc and registry_records(also_doc, also_field, a)] \
            if registry_conflicts else []
        # Rulings 3 and 7 (closed-world and retrieval-scoped negatives) were
        # implemented here and BOTH deleted. Ruling 7 excused a name precisely
        # BECAUSE it was absent from every held key — and absence from the
        # evidence is the definition of a fabrication, so the condition
        # selected for the thing it was meant to exclude. Ruling 3 supported
        # an uncited claim whenever a computed note confirmed the absence and
        # the rest of the unit appeared ANYWHERE in the held set — any
        # document, any field — so 'FP999 does not exist, and the total
        # co-financing is USD 18.5 million' passed against a page printing
        # 18.5M as GCF funding requested. Attributing that rider needs a
        # document, and an uncited claim names none, so the gate cannot be
        # written. Both cost adjudicated rows to remove; both were removed.
        if not c.citations:
            found, _ = _verify_against(c, all_text)
            out.append(Verdict(c, UNSUPPORTED, "no citation on a factual claim",
                               [], flags=flags + (["value-present-elsewhere"]
                                                  if found else []),
                               plausible=found))
            continue
        if not strict:
            out.append(Verdict(c, UNSUPPORTED,
                               "cited evidence was never retrieved: "
                               + "; ".join(bad or ["unresolvable citation"]),
                               [], flags=flags, plausible=False))
            continue

        # RULING 6, the carried half. A heading-form lead-in is no longer a
        # claim; whatever it stated that this claim does not restate is checked
        # over the whole cited DOCUMENT (see `_check_lead_in`) and gates
        # SUPPORT — never the conflict path, which reads the claim's own text.
        lead_gap = ""
        if c.carries_lead_in:
            _lok, lead_gap = _check_lead_in(
                c, _text_of(evidence, list(dict.fromkeys(
                    list(strict) + list(r5) + list(wide)))))

        def _lead_in_blocked(scope: List[EvidenceKey]) -> Optional[Verdict]:
            if not lead_gap:
                return None
            return Verdict(c, UNSUPPORTED,
                           f"the lead-in this claim completes states "
                           f"{lead_gap}, which the cited document does not "
                           f"print", scope, flags=flags, plausible=True)

        strict_text = _text_of(evidence, strict)
        ok, missing = _verify_against(c, strict_text)
        if not ok and r5:
            # RULING 5: the bracket named the document and no page, so any held
            # key of that document may carry the claim. It is a coarse
            # citation, and it is reported as one — the caution the head
            # emitted for a wrong page is exactly the right caution here.
            ok5, _ = _verify_against(c, _text_of(evidence, r5))
            if ok5:
                ok = True
                flags.append("citation-page-mismatch")

        # DERIVED ARITHMETIC (see `derived_arithmetic`). Reached only when the
        # matchers have already failed on the claim AS WRITTEN, which is the
        # gate that keeps a QUOTED equation — release-7's
        # '(a)+(b) = 21 929 123,33 USD', whose three figures are all printed
        # and do not add up — on the path that already verifies it.
        #
        # `probe` is the claim with its arithmetic accounted for, and it is
        # what every check below runs on. When no derivation is read off the
        # claim, `probe is c` and this module behaves exactly as it did: the
        # substitution below is the whole of the change, and it is a no-op on
        # every claim that states no arithmetic.
        probe, deriv = c, None
        if not ok:
            deriv = derived_arithmetic(c)
        if deriv is not None:
            cand = derived_probe(c, deriv)
            # QUALIFICATION, clause 3: the operands, against the evidence this
            # claim CITES — the strictly cited keys, a page-less bracket's
            # document, the rest of a cited document — under the same matcher
            # every other claim gets. Not the registry: a figure no key of
            # this turn prints is not an operand this turn may compute with.
            cited = _text_of(evidence, list(dict.fromkeys(
                list(strict) + list(r5) + list(wide))))
            if not _verify_against(cand, cited)[0]:
                deriv = None            # operands unverified: today's verdict
            elif not deriv.correct:
                # A wrong computation over operands the evidence really prints
                # is not a missing figure — it is the answer contradicting
                # itself about evidence we hold, and CONTRADICTED is the only
                # verdict that says so.
                out.append(Verdict(c, CONTRADICTED, deriv.failure_reason,
                                   strict, flags=flags))
                continue
            else:
                probe = cand
                flags.append(DERIVED_ARITHMETIC)
                ok, missing = _verify_against(probe, strict_text)
                if not ok and r5:
                    ok5, _ = _verify_against(probe, _text_of(evidence, r5))
                    if ok5:
                        ok = True
                        flags.append("citation-page-mismatch")
        # NOTE: a registry-confirmed absence supports the ABSENCE, never the
        # rest of the unit. An earlier revision short-circuited the whole
        # verdict here, so 'FP999 does not exist in this corpus, and FP151
        # requests USD 61 million [doc, p.5]' shipped as verified. The four
        # adjudicated ruling-3 rows are all UNCITED and clear through the
        # branch above; a CITED claim keeps every other check it had.
        # Every SUPPORTED exit below goes through `_conflict_before_support`,
        # and none of them may be given a scope it did not test. The three of
        # them differ in what made the claim verify — the cited page, a
        # page-less bracket's document, the rest of a paged bracket's document
        # or the registry — never in which evidence is allowed to contradict
        # it. Two of the three used to skip the test entirely.
        if ok:
            hit = _conflict_before_support(probe, evidence, strict, wide, r5,
                                           also, also_all, registry_conflicts,
                                           cross_page_conflicts)
            if hit:
                reason, tag = hit
                out.append(Verdict(c, CONTRADICTED, reason, strict,
                                   flags=flags + ([tag] if tag else [])))
                continue
            blocked = _lead_in_blocked(strict)
            if blocked is not None:
                out.append(blocked)
                continue
            if any(a.clash for a in probe.amounts):
                flags.append("unit-scale-clash")
            out.append(Verdict(c, SUPPORTED,
                               deriv.support_reason if deriv is not None
                               else "value found in the cited evidence",
                               strict, flags=flags))
            continue

        # The strictly cited keys disagree AND do not hold the value: this is
        # the one conflict whose reason names what the answer stated, so it is
        # reported before any widening is even attempted — a claim the cited
        # page itself refutes must never be described as a page-number defect.
        conflict, _where = _key_conflict(probe, evidence, strict, also,
                                         registry_conflicts)
        if conflict:
            cand, line = conflict
            out.append(Verdict(c, CONTRADICTED,
                               f"cited evidence states '{cand.raw}' for this field, "
                               f"the answer states '{missing}' ({line})",
                               strict, flags=flags))
            continue

        # Two widenings, tried in order, each a candidate SUPPORTED verdict —
        # collected rather than emitted, because the gate is what decides.
        wide_only = [k for k in wide if k not in strict and k not in r5]
        promoted: Optional[Tuple[str, List[EvidenceKey], str]] = None
        if wide_only:
            ok2, _ = _verify_against(probe, _text_of(evidence, wide_only))
            if ok2:
                promoted = (
                    "value found in the cited document, but not on the cited page",
                    strict + wide_only, "citation-page-mismatch")

        if promoted is None and registry_conflicts:
            held = _text_of(evidence, strict + wide_only)
            gaps = _check_amounts(probe, held)[1] if probe.amounts else []
            gap_names = (_check_entities(probe, held)[1]
                         if probe.kind == "entity" and probe.entities else [])
            backed = None
            for d in dict.fromkeys(k[0] for k in strict + wide
                                   if not is_notes_doc(k[0])):
                backed = (registry_backed(d, probe, gaps) if gaps else None) or \
                    (registry_named(d, gap_names) if gap_names else None)
                if backed:
                    break
            if backed:
                promoted = (
                    f"figure recorded by the corpus registry for this document, "
                    f"on a page this turn did not retrieve: {backed}",
                    list(strict), "registry-backed-page-not-retrieved")

        if promoted is not None:
            reason, scope, promo = promoted
            if deriv is not None:
                reason = f"{deriv.support_reason}; {reason}"
            hit = _conflict_before_support(probe, evidence, strict, wide, r5,
                                           also, also_all, registry_conflicts,
                                           cross_page_conflicts)
            if hit:
                # The promotion flag is kept on the CONTRADICTED verdict: it is
                # still true (the cited page does not carry the figure) and it
                # is what names the escape that was attempted. Flags do not
                # caution here — `RepairResult.cautions` reads SUPPORTED
                # verdicts only — so this costs no user-facing text.
                creason, tag = hit
                out.append(Verdict(c, CONTRADICTED, creason, strict,
                                   flags=flags + [promo] + ([tag] if tag else [])))
                continue
            blocked = _lead_in_blocked(scope)
            if blocked is not None:
                out.append(blocked)
                continue
            out.append(Verdict(c, SUPPORTED, reason, scope,
                               flags=flags + [promo]))
            continue

        out.append(Verdict(c, UNSUPPORTED,
                           f"not found in the cited evidence: {missing or 'the claim'}",
                           strict, flags=flags, plausible=True))
    return out


# ---------------------------------------------------------------------------
# LLM layer (both calls optional, both skippable)
# ---------------------------------------------------------------------------

ADJUDICATE_PROMPT = (
    "You are a strict evidence auditor for a Green Climate Fund document Q&A\n"
    "system. For each claim you are given the exact evidence text it cites.\n"
    "Decide ONLY whether the evidence states the claim. Never use outside\n"
    "knowledge, never infer, never do arithmetic across figures.\n"
    "supported   = the evidence states it (paraphrase or different wording is\n"
    "              fine, a different number or a different name is not).\n"
    "contradicted= the evidence states something incompatible with it.\n"
    "unsupported = the evidence does not state it either way.\n"
    'Reply with JSON only: {"verdicts": [{"id": <int>, '
    '"status": "supported|contradicted|unsupported", "reason": "<12 words>"}]}'
)


def _client() -> Optional[Any]:
    """An OpenAI-compatible client, or None when no key is configured.

    None is a supported operating mode, not an error: the deterministic layer
    is the part of this module that must never be unavailable.
    """
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    try:
        # same construction as chainlit_app / eval_answers: the key comes from
        # the environment, the base url only when one is configured
        return OpenAI(base_url=getattr(config, "OPENAI_BASE_URL", "") or None)
    except Exception:
        return None


#: The sampling this module pins on the ONE call it makes — the batched judge.
#: (It applied to the constrained repair rewrite too, until eac4c94 deleted
#: it; the measurements below were taken on that call and are kept because
#: what they establish about the ENDPOINT is what still governs the judge.)
#:
#: Wave 4 measured what an unpinned rewrite costs. From identical answers and
#: an identical verdict object, repair's adoption decision flipped on 18-27%
#: of cases and two full replays disagreed on whether the pass corrected 5
#: claims or deleted 7. Both numbers are properties of the sampler, not of the
#: repair prompt: ``_complete`` sent neither temperature nor seed, so every
#: rewrite of user-visible text was a fresh draw. An operator cannot audit a
#: rewrite that the same inputs do not reproduce, and a canary cannot measure
#: a treatment that is not a fixed quantity.
#:
#: These are CONSTANTS, deliberately not environment-overridable: a seed the
#: deployment can change is not a pin, and `config.py` already carries the
#: switches an operator is meant to turn.
#:
#: The ANSWER-GENERATION call is not made here and is not touched: this module
#: only ever calls the judge.
#:
#: WHAT THE PIN DOES AND DOES NOT BUY — MEASURED, NOT ASSUMED. Against the
#: configured endpoint (api.openai.com, `CHAT_MODEL=gpt-5.2`) both parameters
#: are ACCEPTED, and pinning them still does not make the completion
#: reproducible. Eight repair calls on one recorded turn (`id-fp269-gcf`,
#: identical answer, identical verdicts, deterministic verdicts only so the
#: judge sample is out of the experiment):
#:
#:     unpinned (HEAD)                 2 distinct completions / 8
#:     temperature=0 + seed pinned     2 distinct completions / 8
#:
#: and the responses carry NO `system_fingerprint` at all, which is OpenAI's
#: own signal for "this backend can be reasoned about deterministically". So
#: `seed` here is best-effort and is not honoured to the byte. The pin is kept
#: because it is free, because it removes the two degrees of freedom that ARE
#: under our control, and because an endpoint that honours seed then reproduces
#: exactly — but nothing in this module may be read as a claim that a
#: completion can be re-derived on this deployment.
#:
#: This is also the shortest statement of why repair is gone. A rewrite that
#: the same inputs do not reproduce cannot be audited by re-running it, only
#: by recording it; a detector's output is re-derivable from the answer and
#: the evidence by pure python, judge or no judge, and that is the difference
#: between a component an operator can check and one they can only trust.
SAMPLING_TEMPERATURE = 0
SAMPLING_SEED = 20260821

#: Dropped one at a time, most-optional FIRST. ``seed`` is the newer of the
#: two and the one OpenAI-compatible servers most often refuse; ``temperature``
#: is the pin that matters more, so it is the last thing given up.
_PINNED_SAMPLING: Tuple[str, ...] = ("seed", "temperature")

#: Parameters THIS endpoint has refused, remembered for the process so a
#: server that does not support one is not paid for a rejected first attempt
#: on every subsequent turn. Only a failure that NAMES the parameter lands
#: here, so a 502 or a rate limit can never silently unpin the module.
_SAMPLING_UNSUPPORTED: set = set()


def _reset_sampling_support() -> None:
    """Forget which parameters the endpoint refused (tests; process restart)."""
    _SAMPLING_UNSUPPORTED.clear()


def _sampling_kwargs() -> Dict[str, Any]:
    """The pinned sampling parameters still believed to be accepted."""
    out: Dict[str, Any] = {}
    if "temperature" not in _SAMPLING_UNSUPPORTED:
        out["temperature"] = SAMPLING_TEMPERATURE
    if "seed" not in _SAMPLING_UNSUPPORTED:
        out["seed"] = SAMPLING_SEED
    return out


def _blames_sampling_param(exc: BaseException, sent: Dict[str, Any]
                           ) -> Optional[str]:
    """Which pinned parameter this failure blames — or None for every other
    failure.

    The degradation this exists for is narrow and must stay narrow: an
    endpoint that rejects ``seed`` outright must not cost the turn, and an
    endpoint that is merely down must not cost the PIN. So a retry is only
    taken when the error names one of the parameters we actually sent:

      * the OpenAI SDK reports it structurally (``BadRequestError.param``),
        measured on this deployment as ``param='seed'`` / ``code='invalid_type'``;
      * other OpenAI-compatible servers only put it in the message text, so a
        4xx whose text names the parameter counts too;
      * a client whose ``create()`` does not accept the keyword at all raises
        ``TypeError`` before any request is made.

    A 500, a timeout, a rate limit or a 400 about ``messages`` returns None and
    takes the existing keep-the-deterministic-verdicts path.
    """
    if not sent:
        return None
    named = getattr(exc, "param", None)
    if isinstance(named, str) and named.split(".")[-1] in sent:
        return named.split(".")[-1]
    text = str(exc)
    status = getattr(exc, "status_code", None)
    client_side = isinstance(exc, TypeError)
    if not (client_side or status in (400, 422)
            or re.search(r"\b(?:400|422)\b", text)):
        return None
    for p in _PINNED_SAMPLING:
        if p in sent and re.search(rf"\b{p}\b", text, re.I):
            return p
    if client_side:                     # named nothing; give up the most optional
        return next((p for p in _PINNED_SAMPLING if p in sent), None)
    return None


def _complete(client: Any, system: str, user: str, max_tokens: int) -> Optional[str]:
    """One pinned chat completion, or None.

    Pinned means ``temperature`` and ``seed`` travel with the request (see
    ``SAMPLING_TEMPERATURE``). An endpoint that refuses one of them gets ONE
    retry per refused parameter and keeps the other, so the turn degrades to a
    less reproducible sample instead of failing — and says so, once, in the
    log, because a rewrite that is no longer pinned is a different object from
    one that is.
    """
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    for _ in range(len(_PINNED_SAMPLING) + 1):
        pinned = _sampling_kwargs()
        try:
            resp = client.chat.completions.create(
                model=config.CHAT_MODEL,
                max_completion_tokens=max_tokens,
                messages=messages,
                **pinned,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:          # a failed audit must not fail the answer
            dropped = _blames_sampling_param(e, pinned)
            if dropped is None:
                print(f"verify: LLM call failed, keeping deterministic "
                      f"verdicts: {e}", flush=True)
                return None
            _SAMPLING_UNSUPPORTED.add(dropped)
            print(f"verify: endpoint rejected {dropped!r}; retrying without it "
                  f"— this call is no longer fully pinned: {e}", flush=True)
    return None


def _json_object(raw: str) -> Optional[dict]:
    """First JSON object in a model reply, fences and prose tolerated."""
    if not raw:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    for candidate in (text, text[text.find("{"):text.rfind("}") + 1] if "{" in text else ""):
        try:
            got = json.loads(candidate)
            if isinstance(got, dict):
                return got
        except Exception:
            continue
    return None


def _where_found(claim: Claim, evidence: Evidence, limit: int = 3
                 ) -> List[EvidenceKey]:
    """Evidence keys whose text does state the claim, wherever they are.

    An uncited claim has no scope, and handing the judge '(no evidence held
    for this citation)' can only get the deterministic verdict rubber-stamped.
    What the judge needs is the passage that carries the value, so it can rule
    on whether that passage really says what the sentence says.
    """
    out = [k for k, t in evidence.items() if _verify_against(claim, t)[0]]
    return out[:limit]


def _judge_keys(claim: Claim, evidence: Evidence,
                strict: Sequence[EvidenceKey],
                wide: Sequence[EvidenceKey]) -> List[EvidenceKey]:
    """The evidence a claim is adjudicated against."""
    return (list(strict) or list(wide[:2]) or _where_found(claim, evidence)
            or list(evidence)[:3])


def _evidence_snippet(evidence: Evidence, keys: Sequence[EvidenceKey],
                      limit: int = 1200) -> str:
    parts = []
    for k in keys:
        text = _text_at(evidence, k)
        if not text:
            continue
        label = ("computed notes" if k[0] == NOTES_DOC else
                 f"computed notes for {k[0].split(':', 1)[1]}, p.{k[1]}"
                 if is_notes_doc(k[0]) else
                 f"{k[0]}, p.{k[1]}" if k[1] else
                 f"{k[0]}, cover pages / registry")
        parts.append(f"[{label}]\n{text[:limit]}")
    return "\n\n".join(parts) or "(no evidence held for this citation)"


def adjudicate(verdicts: Sequence[Verdict], evidence: Evidence,
               client: Any = None, max_claims: int = 12) -> List[Verdict]:
    """ONE batched judge call over the deterministic residue.

    Only claims marked unsupported-but-plausible are sent — a claim citing a
    page we never retrieved is already decided, and paying a model to agree is
    latency without information. Returns a NEW verdict list; anything the
    model does not answer keeps its deterministic verdict.
    """
    todo = [v for v in verdicts if v.status == UNSUPPORTED and v.plausible][:max_claims]
    if not todo:
        return list(verdicts)
    if client is None:
        client = _client()
    if client is None:
        return list(verdicts)

    payload = []
    for v in todo:
        strict, wide, _, _r5 = _scopes(v.claim, evidence)
        payload.append({
            "id": v.claim.index,
            "claim": _strip_citations(v.claim.text).strip()[:400],
            "cited_evidence": _evidence_snippet(
                evidence, _judge_keys(v.claim, evidence, strict, wide)),
        })
    user = "Claims to audit:\n" + "\n\n".join(
        f"--- claim {p['id']} ---\n{p['claim']}\n\nEVIDENCE:\n{p['cited_evidence']}"
        for p in payload)
    raw = _complete(client, ADJUDICATE_PROMPT, user,
                    max_tokens=min(8192, 400 * len(todo) + 800))
    data = _json_object(raw or "")
    if not data or not isinstance(data.get("verdicts"), list):
        return list(verdicts)

    by_index = {}
    for item in data["verdicts"]:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        status = str(item.get("status", "")).lower().strip()
        if status in (SUPPORTED, CONTRADICTED, UNSUPPORTED):
            by_index[idx] = (status, str(item.get("reason", ""))[:200])

    sent = {id(v) for v in todo}
    out = []
    for v in verdicts:
        got = by_index.get(v.claim.index) if id(v) in sent else None
        if got is None:
            out.append(v)
            continue
        status, reason = got
        out.append(Verdict(v.claim, status, reason or "judge verdict", v.scope,
                           source="llm", flags=list(v.flags), plausible=False))
    return out


def classify(claims: Sequence[Claim], evidence: Evidence, client: Any = None,
             use_llm: bool = True, cross_page_conflicts: bool = True,
             registry_conflicts: bool = True) -> List[Verdict]:
    """Deterministic verdicts, then at most one batched LLM adjudication.

    ``use_llm=False`` (or no API key, or no plausible residue) makes this a
    pure-python, network-free call.
    """
    verdicts = classify_deterministic(claims, evidence, cross_page_conflicts,
                                      registry_conflicts)
    if not use_llm:
        return verdicts
    return adjudicate(verdicts, evidence, client)


# ---------------------------------------------------------------------------
# the result object
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """The verified answer, its verdicts, and what the app must show.

    THE NAME IS HISTORICAL and is kept deliberately. It was the return type of
    the repair pass, every consumer already imports and constructs it, and
    renaming a type to announce that a feature was deleted buys a churned
    diff across four files and no behaviour. What changed is the CONTENT of
    the contract:

      * ``answer`` is ALWAYS the text that was passed in. This class no longer
        carries a rewrite, because nothing produces one.
      * ``original_answer`` is therefore the same string. Kept because the
        recorded eval rows carry both and a reader comparing them is entitled
        to see them agree.
      * ``repaired`` is ALWAYS False, and ``repair_rejected`` is ALWAYS False.
        Kept as fields, not deleted, for record compatibility:
        ``scripts/eval_answers.py`` writes both into every claims block of
        every release run, and the 66-answer baselines this tree is scored
        against have them. A field that disappears makes an old record and a
        new record incomparable at exactly the moment someone is trying to
        prove nothing changed.

    ``status``, and all four values are reachable:
      verified        every claim is supported by the evidence it cites
      partial         some fact-bearing claims remain unsupported
      abstain         every fact-bearing claim failed — nothing left to show
      unverified-llm  claims failed but no judge was available (no key)

    'repaired' is NOT among them any more, and no code path can produce it.
    """
    answer: str
    status: str
    verdicts: List[Verdict] = dc_field(default_factory=list)
    original_answer: str = ""
    #: vestigial, always False — see the class docstring
    repaired: bool = False
    #: vestigial, always False — see the class docstring
    repair_rejected: bool = False
    notes: List[str] = dc_field(default_factory=list)

    @property
    def failures(self) -> List[Verdict]:
        return [v for v in self.verdicts if v.failed]

    @property
    def unsupported(self) -> List[Verdict]:
        return [v for v in self.verdicts if v.status == UNSUPPORTED]

    @property
    def contradicted(self) -> List[Verdict]:
        return [v for v in self.verdicts if v.status == CONTRADICTED]

    @property
    def cautions(self) -> List[Verdict]:
        """Supported claims that still deserve a warning: a figure whose
        printed scale is self-contradictory, or one cited to the wrong page."""
        return [v for v in self.verdicts if v.status == SUPPORTED and v.flags]

    @property
    def sources(self) -> List[Tuple[str, Optional[int]]]:
        return cited_sources(self.answer)

    @property
    def ok(self) -> bool:
        return self.status == "verified"

    def counts(self) -> Dict[str, int]:
        out = {SUPPORTED: 0, CONTRADICTED: 0, UNSUPPORTED: 0}
        for v in self.verdicts:
            out[v.status] = out.get(v.status, 0) + 1
        return out


def _status_for(verdicts: Sequence[Verdict], llm_available: bool,
                _repaired: bool = False) -> str:
    """Which of the four statuses these verdicts carry.

    ``_repaired`` is vestigial and IGNORED — there is no repaired status for
    it to select. It is still in the signature, with a default, for exactly
    one reason: ``scripts/score_verifier.py`` calls this helper positionally
    with a third argument (always ``False``) to compute the status census the
    five arms are scored against, and that instrument is not this module's to
    edit. Taking the parameter and ignoring it keeps the scorer's numbers
    identical across this change, which is the property that had to be
    proved. Delete it when the scorer stops passing it.
    """
    failed = [v for v in verdicts if v.failed]
    if not failed:
        return "verified"
    if not llm_available:
        return "unverified-llm"
    required = [v for v in verdicts if v.claim.required]
    if required and all(v.failed for v in required):
        return "abstain"
    return "partial"


def verify_answer(answer: str, evidence: Evidence, client: Any = None,
                  use_llm: bool = True,
                  cross_page_conflicts: bool = True,
                  registry_conflicts: bool = True) -> RepairResult:
    """extract -> classify -> report. The whole step-5 pass in one call.

    ``result.answer is answer``: this function detects, it does not write. The
    ``allow_repair`` switch it used to take is gone with the pass it gated —
    production ran ``VERIFY_REPAIR=0``, so its only reachable value was False
    and removing it changes nothing that ever ran.

    ``use_llm`` is the one switch left. Turning it off (``VERIFY_LLM=0``, or
    no API key) makes this pure python and network-free; the verdicts are then
    the deterministic ones and the status says so ('unverified-llm') instead
    of pretending the residue was adjudicated.
    """
    claims = extract_claims(answer)
    verdicts = classify(claims, evidence, client=client, use_llm=use_llm,
                        cross_page_conflicts=cross_page_conflicts,
                        registry_conflicts=registry_conflicts)
    if not any(v.failed for v in verdicts):
        return RepairResult(answer, "verified", verdicts, answer)
    llm = client is not None or _client() is not None
    # THE NOTE STRING IS FROZEN, and the reason is the gate this change had to
    # pass: deleting a pass that production never ran must change NOTHING an
    # operator can observe, and `notes` travels into the app's verification
    # step and into what an A/B over the recorded runs compares. 62 of the 132
    # recorded answers carry exactly this string today. Rewording it to
    # something truer of a detector ("the answer is never rewritten") would be
    # the one observable difference in an otherwise byte-identical change — so
    # it is left alone here, to be changed on purpose or not at all.
    return RepairResult(answer, _status_for(verdicts, llm), verdicts,
                        answer, notes=["repair disabled"])
