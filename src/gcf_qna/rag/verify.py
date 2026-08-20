"""Claim-level answer verification and one constrained repair pass (plan step 5).

The answer model is the only component in the pipeline that is allowed to
*write* facts, and it is the one component we cannot audit by construction.
This module audits its output after the fact:

    claims   = extract_claims(answer)                    # pure python
    evidence = build_evidence(hits, notes)               # what the turn held
    verdicts = classify(claims, evidence)                # deterministic first
    result   = repair(answer, verdicts, evidence)        # at most one LLM pass

Design rules, in the order they matter:

1. **Deterministic before LLM.** Every money figure, count, entity name and
   year is checked in pure python against the exact text the turn retrieved.
   A judge model is asked only about the residue: claims the string/number
   checks could not confirm *but whose cited evidence we actually hold*. A
   claim citing a page that was never retrieved needs no judge — that verdict
   is already certain.
2. **At most two LLM calls per answer**, both skippable: one batched
   adjudication for the plausible residue, one repair. With no
   ``OPENAI_API_KEY`` the module still returns deterministic verdicts and
   simply refuses to rewrite anything (``status='unverified-llm'``).
3. **Repair may not introduce sources.** The repaired answer's citations are
   re-parsed and must be a subset of the evidence we hold; a repair that
   invents a document or a page is dropped, and the original answer is flagged
   instead. A verifier that can be talked into new citations is worse than no
   verifier.
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
    r"if you want|let me know|i can (?:compare|provide|list|run)|"
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


def norm_text(s: str) -> str:
    """Fold case, accents, markdown emphasis and punctuation runs, so that a
    substring test compares words rather than typography."""
    s = _deaccent((s or "").lower())
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
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


def entities(text: str) -> List[List[str]]:
    """Proper-noun assignments a sentence makes, each as a variant list.

    Bold and quoted spans first (the answer model marks its own assertions),
    then acronyms and capitalized runs. Citation brackets are stripped before
    anything else: a document id is a pointer, never an entity claim.
    """
    body = _strip_citations(text)
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

    # A candidate already contained in ANOTHER candidate's spelling adds no
    # check: 'Unlocking' was cut out of the title it belongs to, and
    # 'SoCF Global' / 'Pegasus Capital Advisors LLP' are the alias and the
    # entity that 'Funding proposal submitted by Pegasus Capital Advisors LLP
    # (Peganas)' and 'Global Subnational Climate Fund (SoCF Global)' already
    # carry as variants (claim-a59cc16e). Verifying the longer form covers the
    # fragment, and the fragment is what turns one reflowed name into two
    # independent 'unsupported' verdicts — the standalone alias failed while
    # the group that owns it matched.
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


def extract_claims(answer: str) -> List[Claim]:
    """Split an answer into atomic factual claims carrying their citations.

    A claim is a sentence, bullet or table row asserting a money amount, a
    number with a unit, a proper-noun assignment, a year/board fact or an
    existence fact. Prose glue, hedges and refusals are dropped; an uncited
    factual sentence is still a claim — with no evidence pointer, which is
    exactly what makes it interesting to the classifier.
    """
    claims: List[Claim] = []
    for text, unit_kind, citations, inherited in _units(answer):
        kind = claim_kind(text)
        if kind is None:
            continue
        body = _strip_citations(text)
        claims.append(Claim(
            text=text.strip(), kind=kind, citations=citations,
            amounts=amounts(body), entities=entities(body),
            index=len(claims), unit_kind=unit_kind, inherited=inherited))
    return claims


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

_MATRIX_DOC_RE = re.compile(r"^(?P<label>[^|\n]{1,40}?)\s*->\s*(?P<doc>" + _DOC_RE + r")")
_MATRIX_PAGE_RE = re.compile(r"\(p\.\s*(\d{1,3})")
_REG_DOC_RE = re.compile(r"\[(" + _DOC_RE + r"),\s*cover pages?\]", re.I)


def build_evidence(hits: Sequence[Any] = (), notes: Any = None) -> Evidence:
    """{(doc_id, page): text} for one turn — the only thing claims may cite.

    Hits contribute their source text at (doc_id, page). Computed note blocks
    (registry lines, year notes, the evidence matrix) contribute twice: under
    the document they name — page-level when the line prints '(p.5, A.8)',
    document-level otherwise, which is the 'cover pages' scope an answer cites
    — and, whole, under the notes pseudo-document, so a note-level citation
    still resolves to something we hold.
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
    """Forgiving document match — the answer may print a truncated id."""
    docs = list(docs)
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
]
_FIELD_RES = [(f, re.compile(r, re.I)) for f, r in _FIELD_LABELS]


def claim_field(text: str) -> Optional[str]:
    body = norm_text(_strip_citations(text))
    for name, rx in _FIELD_RES:
        if rx.search(body):
            return name
    return None


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
    """
    docs = {k[0] for k in evidence if k[0] != NOTES_DOC}
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
        else:
            bad.append(f"{d}, p.{c.page}")
    ded = lambda xs: list(dict.fromkeys(xs))     # noqa: E731
    keep = ded(strict)
    return keep, ded(wide), ded(bad), [k for k in ded(widened) if k not in keep]


def _text_of(evidence: Evidence, keys: Sequence[EvidenceKey]) -> str:
    return "\n".join(evidence[k] for k in keys if k in evidence)


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
    the instructed behaviour, not a contradiction to repair.

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
    # asks for, not a contradiction to repair. The reporting is ANSWER-level,
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
    that this turn may not hold — and REPAIR_PROMPT rule 2 asks for exactly
    that sentence — so without this check no conflict-aware answer could ever
    verify, and the repair pass could only produce answers it would then
    reject.
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


def _check_entities(claim: Claim, text: str) -> Tuple[bool, List[List[str]]]:
    """(ok, the variant lists that appear nowhere in this text)."""
    hay = norm_text(text)
    missing = []
    for variants in claim.entities:
        if not any(norm_text(v) and norm_text(v) in hay for v in variants):
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


def _verify_against(claim: Claim, text: str) -> Tuple[bool, str]:
    """(ok, what is missing) for one claim against one blob of evidence."""
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
        if registry_rulings and k[0] != NOTES_DOC:
            settled = (lambda rival, d=k[0]:
                       registry_ruled_compatible(d, field, claim.amounts, rival))
        got = _field_conflict(claim, evidence.get(k, ""), also, settled)
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
                if k[0] != NOTES_DOC}
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
        # NOTE: a registry-confirmed absence supports the ABSENCE, never the
        # rest of the unit. An earlier revision short-circuited the whole
        # verdict here, so 'FP999 does not exist in this corpus, and FP151
        # requests USD 61 million [doc, p.5]' shipped as verified. The four
        # adjudicated ruling-3 rows are all UNCITED and clear through the
        # branch above; a CITED claim keeps every other check it had.
        if ok:
            # Conflicts are tested PER KEY, over the ruling-5 scope as well as
            # the strict one. Concatenating the keys first let one agreeing
            # page hide a disagreeing one, and folding ruling 5 into `strict`
            # emptied the cross-page scope altogether: a document printing two
            # figures for the same field, cited '[doc]', verified clean.
            conflict, where = _key_conflict(c, evidence, strict, also,
                                            registry_conflicts)
            if conflict is None and r5:
                conflict, where = _key_conflict(c, evidence, r5, also,
                                                registry_conflicts)
                if conflict:
                    flags.append("conflict-elsewhere-in-document")
            if conflict is None and cross_page_conflicts:
                others = [k for k in wide if k not in strict and k not in r5]
                conflict, where = _key_conflict(c, evidence, others, also,
                                                registry_conflicts)
                if conflict:
                    flags.append("conflict-elsewhere-in-document")
            if conflict:
                cand, line = conflict
                out.append(Verdict(
                    c, CONTRADICTED,
                    f"the cited document also prints '{cand.raw}' for this field "
                    f"({line})", strict, flags=flags))
                continue
            known = None
            if registry_conflicts:
                for d in dict.fromkeys(k[0] for k in strict + wide if k[0] != NOTES_DOC):
                    known = registry_conflict(d, c, also_all)
                    if known:
                        break
            if known:
                out.append(Verdict(c, CONTRADICTED, known, strict,
                                   flags=flags + ["known-document-conflict"]))
                continue
            if any(a.clash for a in c.amounts):
                flags.append("unit-scale-clash")
            out.append(Verdict(c, SUPPORTED, "value found in the cited evidence",
                               strict, flags=flags))
            continue

        conflict, _where = _key_conflict(c, evidence, strict, also,
                                         registry_conflicts)
        if conflict:
            cand, line = conflict
            out.append(Verdict(c, CONTRADICTED,
                               f"cited evidence states '{cand.raw}' for this field, "
                               f"the answer states '{missing}' ({line})",
                               strict, flags=flags))
            continue

        wide_only = [k for k in wide if k not in strict and k not in r5]
        if wide_only:
            ok2, _ = _verify_against(c, _text_of(evidence, wide_only))
            if ok2:
                out.append(Verdict(
                    c, SUPPORTED,
                    "value found in the cited document, but not on the cited page",
                    strict + wide_only,
                    flags=flags + ["citation-page-mismatch"]))
                continue

        if registry_conflicts:
            held = _text_of(evidence, strict + wide_only)
            gaps = _check_amounts(c, held)[1] if c.amounts else []
            gap_names = (_check_entities(c, held)[1]
                         if c.kind == "entity" and c.entities else [])
            backed = None
            for d in dict.fromkeys(k[0] for k in strict + wide if k[0] != NOTES_DOC):
                backed = (registry_backed(d, c, gaps) if gaps else None) or \
                    (registry_named(d, gap_names) if gap_names else None)
                if backed:
                    break
            if backed:
                out.append(Verdict(
                    c, SUPPORTED,
                    f"figure recorded by the corpus registry for this document, "
                    f"on a page this turn did not retrieve: {backed}",
                    strict, flags=flags + ["registry-backed-page-not-retrieved"]))
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

REPAIR_PROMPT = (
    "You repair an answer that failed evidence verification, for a Green\n"
    "Climate Fund document Q&A system.\n"
    "Rules, in order:\n"
    "1. Keep every supported sentence exactly as it is.\n"
    "2. For a CONTRADICTED claim, replace the value with the value the\n"
    "   evidence prints, keeping its citation; when the evidence prints two\n"
    "   disagreeing figures, give BOTH with their pages.\n"
    "3. For an UNSUPPORTED claim, either delete the sentence or qualify it as\n"
    "   not supported by the retrieved evidence. Never keep it as a fact.\n"
    "4. NEVER introduce a document id, a page number or a figure that is not\n"
    "   in the evidence below. Do not add new citations.\n"
    "5. Keep the answer's language (French answer stays French) and format.\n"
    "Reply with the repaired answer text only — no preamble, no commentary."
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


def _complete(client: Any, system: str, user: str, max_tokens: int) -> Optional[str]:
    try:
        resp = client.chat.completions.create(
            model=config.CHAT_MODEL,
            max_completion_tokens=max_tokens,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:              # a failed audit must not fail the answer
        print(f"verify: LLM call failed, keeping deterministic verdicts: {e}",
              flush=True)
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
    """The evidence a claim is adjudicated or repaired against."""
    return (list(strict) or list(wide[:2]) or _where_found(claim, evidence)
            or list(evidence)[:3])


def _evidence_snippet(evidence: Evidence, keys: Sequence[EvidenceKey],
                      limit: int = 1200) -> str:
    parts = []
    for k in keys:
        if k not in evidence:
            continue
        label = "computed notes" if k[0] == NOTES_DOC else (
            f"{k[0]}, p.{k[1]}" if k[1] else f"{k[0]}, cover pages / registry")
        parts.append(f"[{label}]\n{evidence[k][:limit]}")
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
# repair
# ---------------------------------------------------------------------------

@dataclass
class RepairResult:
    """The verified answer, its verdicts, and what the app must show.

    ``status``:
      verified        every claim is supported by the evidence it cites
      repaired        the repair pass fixed every failing claim
      partial         some fact-bearing claims remain unsupported
      abstain         every fact-bearing claim failed — nothing left to show
      unverified-llm  claims failed but no judge/repair was available (no key)
    """
    answer: str
    status: str
    verdicts: List[Verdict] = dc_field(default_factory=list)
    original_answer: str = ""
    repaired: bool = False
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
        return self.status in ("verified", "repaired")

    def counts(self) -> Dict[str, int]:
        out = {SUPPORTED: 0, CONTRADICTED: 0, UNSUPPORTED: 0}
        for v in self.verdicts:
            out[v.status] = out.get(v.status, 0) + 1
        return out


def _status_for(verdicts: Sequence[Verdict], llm_available: bool,
                repaired: bool) -> str:
    failed = [v for v in verdicts if v.failed]
    if not failed:
        return "repaired" if repaired else "verified"
    if not llm_available and not repaired:
        return "unverified-llm"
    required = [v for v in verdicts if v.claim.required]
    if required and all(v.failed for v in required):
        return "abstain"
    return "partial"


def _introduced_sources(answer: str, evidence: Evidence,
                        allowed_docs: Optional[Sequence[str]] = None) -> List[str]:
    """Citations in a repaired answer that the evidence cannot back.

    The deterministic guard on rule 4 of REPAIR_PROMPT: a repair that invents
    a document or a page is not a repair, and no amount of prompt wording is
    a substitute for checking.

    Matching here is EXACT, unlike ``_resolve_doc``. The forgiving prefix rule
    exists because the answer model truncates ids it was shown; a repair that
    emits an id we were not holding is the case this function exists to catch,
    and a 24-character prefix match would wave through any suffix appended to
    the 182 corpus ids that are 24 characters or shorter.

    ``allowed_docs`` narrows it further to the documents the repair prompt
    actually showed: moving a claim onto another retrieved document's figure
    verifies cleanly and is still an invented attribution.
    """
    docs = {k[0] for k in evidence if k[0] != NOTES_DOC}
    permitted = set(allowed_docs) if allowed_docs is not None else docs
    bad = []
    for c in parse_citations(answer):
        if c.doc is None:
            continue
        if c.doc not in docs:
            bad.append(f"{c.doc}" + (f", p.{c.page}" if c.page else ""))
        elif c.doc not in permitted:
            bad.append(f"{c.doc} (not shown to the repair pass)")
        elif c.page is not None and (c.doc, c.page) not in evidence:
            bad.append(f"{c.doc}, p.{c.page}")
    return list(dict.fromkeys(bad))


_PREAMBLE_RE = re.compile(
    r"^\s*(?:sure|certainly|of course|here(?:'s| is| are)|voici|voil[àa]|bien s[ûu]r|"
    r"repaired answer|corrected answer|r[ée]ponse corrig[ée]e)\b[^\n]{0,120}:?\s*$",
    re.I)


def _strip_preamble(text: str) -> str:
    """Drop a chat preamble line and any code fence around the answer.

    'Sure! Here is the repaired answer:' is not part of the answer, and
    shipping it into the chat window is how a verification pass announces
    itself to the user as a machine.
    """
    body = re.sub(r"^\s*```[a-zA-Z]*\n|\n?```\s*$", "", (text or "").strip())
    lines = body.splitlines()
    while len(lines) > 1 and (not lines[0].strip() or _PREAMBLE_RE.match(lines[0])):
        lines = lines[1:]
    return "\n".join(lines).strip()


def _supported_required(verdicts: Sequence[Verdict]) -> int:
    return sum(1 for v in verdicts
               if v.status == SUPPORTED and v.claim.required)


def _carry_cleared(new_verdicts: List[Verdict],
                   old_verdicts: Sequence[Verdict]) -> List[Verdict]:
    """Keep judge rulings across the post-repair recheck.

    The recheck is deterministic-only, so a claim the judge cleared as a
    paraphrase would come back unsupported and make a good repair look like a
    failed one. Sentences the repair left untouched are matched by normalized
    text and keep their verdict.
    """
    cleared = {norm_text(v.claim.text): v for v in old_verdicts
               if v.source == "llm" and v.status == SUPPORTED}
    if not cleared:
        return new_verdicts
    out = []
    for v in new_verdicts:
        got = cleared.get(norm_text(v.claim.text))
        if got is not None and v.failed:
            out.append(Verdict(v.claim, SUPPORTED, got.reason, v.scope,
                               source="llm", flags=list(v.flags)))
        else:
            out.append(v)
    return out


def repair(answer: str, verdicts: Sequence[Verdict], evidence: Evidence,
           client: Any = None, cross_page_conflicts: bool = True,
           registry_conflicts: bool = True) -> RepairResult:
    """ONE constrained repair pass over the failing claims.

    The model may remove, qualify or correct claims; it may not introduce
    sources. Its text is then earned, not assumed: it is re-verified
    deterministically against the same evidence and adopted only when it is
    strictly better than what it replaced —

      * no citation we cannot back, and none the repair prompt was not shown
        (a claim re-attributed to another retrieved document verifies cleanly
        and is still an invented attribution);
      * fewer failing claims than before, and none left failing at all —
        swapping one wrong figure for a different wrong figure is not a
        repair, and it is what an unguarded pass actually produced;
      * at least as much substance: if the answer had supported fact-bearing
        claims, the repaired one must still have one. A rewrite that deletes
        every supported claim replaces a mostly-correct answer with a
        refusal, which is a regression the user cannot see.

    Anything else keeps the ORIGINAL answer and flags it — an honest 'partial'
    beats a confident rewrite.
    """
    verdicts = list(verdicts)
    failed = [v for v in verdicts if v.failed]
    if not failed:
        return RepairResult(answer, "verified", verdicts, answer)

    if client is None:
        client = _client()
    if client is None:
        return RepairResult(answer, _status_for(verdicts, False, False), verdicts,
                            answer, notes=["no LLM available: deterministic "
                                           "verdicts only, answer left as written"])

    blocks, shown_docs = [], []
    for v in failed:
        strict, wide, _, _r5 = _scopes(v.claim, evidence)
        keys = _judge_keys(v.claim, evidence, strict, wide)
        shown_docs += [k[0] for k in keys if k[0] != NOTES_DOC]
        blocks.append(
            f"--- {v.status.upper()} claim ---\n{v.claim.text}\n"
            f"why: {v.reason}\n"
            f"EVIDENCE IT MAY USE:\n{_evidence_snippet(evidence, keys)}")
    # what the repair may cite: what it was shown, plus what the answer
    # already cited and kept
    allowed = list(dict.fromkeys(
        shown_docs + [d for d, _ in cited_sources(answer)]))
    user = (f"ANSWER TO REPAIR:\n{answer}\n\n"
            f"FAILED CLAIMS ({len(failed)}):\n" + "\n\n".join(blocks))
    raw = _complete(client, REPAIR_PROMPT, user,
                    max_tokens=min(6000, 900 + len(answer) // 2))
    if not raw:
        return RepairResult(answer, _status_for(verdicts, True, False), verdicts,
                            answer, notes=["repair call failed; answer left as written"])
    raw = _strip_preamble(raw)

    def _keep(note: str) -> RepairResult:
        return RepairResult(answer, _status_for(verdicts, True, False), verdicts,
                            answer, repair_rejected=True, notes=[note])

    introduced = _introduced_sources(raw, evidence, allowed)
    if introduced:
        return _keep("repair rejected: it introduced sources not in the evidence — "
                     + "; ".join(introduced[:4]))

    new_verdicts = _carry_cleared(
        classify_deterministic(extract_claims(raw), evidence,
                               cross_page_conflicts, registry_conflicts),
        verdicts)
    new_failed = [v for v in new_verdicts if v.failed]
    if new_failed:
        return _keep(
            f"repair rejected: {len(new_failed)} claim(s) still fail verification "
            f"(was {len(failed)}) — " + new_failed[0].reason[:120])
    if _supported_required(verdicts) and not _supported_required(new_verdicts):
        return _keep("repair rejected: it removed every supported factual claim "
                     "instead of correcting the failing one")

    return RepairResult(raw, _status_for(new_verdicts, True, True), new_verdicts,
                        answer, repaired=True,
                        notes=[f"repaired {len(failed)} failing claim(s)"])


def verify_answer(answer: str, evidence: Evidence, client: Any = None,
                  use_llm: bool = True, allow_repair: bool = True,
                  cross_page_conflicts: bool = True,
                  registry_conflicts: bool = True) -> RepairResult:
    """extract -> classify -> repair, the whole step-5 pass in one call.

    ``use_llm`` and ``allow_repair`` are INDEPENDENT switches (plan step 6:
    'roll out behind independent switches'). Turning the judge off leaves the
    repair pass working on deterministic verdicts — which are the verdicts the
    repair prompt is built from anyway — so a deployment can run
    deterministic-verify + repair, or judge-only, or both.

    At most two LLM calls, both skippable. With no key this is pure python and
    returns 'verified' or 'unverified-llm' with the failing claims attached,
    so the app can display honestly instead of silently.
    """
    claims = extract_claims(answer)
    verdicts = classify(claims, evidence, client=client, use_llm=use_llm,
                        cross_page_conflicts=cross_page_conflicts,
                        registry_conflicts=registry_conflicts)
    if not any(v.failed for v in verdicts):
        return RepairResult(answer, "verified", verdicts, answer)
    if not allow_repair:
        llm = client is not None or _client() is not None
        return RepairResult(answer, _status_for(verdicts, llm, False), verdicts,
                            answer, notes=["repair disabled"])
    return repair(answer, verdicts, evidence, client=client,
                  cross_page_conflicts=cross_page_conflicts,
                  registry_conflicts=registry_conflicts)
