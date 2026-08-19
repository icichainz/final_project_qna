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

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class Amount:
    """One printed figure, normalized. ``value`` is None when the mantissa and
    the printed unit word cannot both be true ('28,654 million USD'): the
    scale is then unknown and only ``bare`` (the mantissa) is comparable."""
    raw: str
    value: Optional[float]
    bare: float
    currency: Optional[str]
    unit: Optional[str]
    grain: float
    at: int = 0

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
        yield Amount(
            raw=raw or num,
            value=None if clash else round(val * mult, 2),
            bare=round(val, 6),
            currency=_CUR_MAP.get(cur_tok if cur_tok == "us$" else cur_tok.rstrip("s")),
            unit={1e6: "million", 1e9: "billion", 1e3: "thousand"}.get(mult),
            grain=granularity(num, 1.0 if clash else mult),
            at=m.start("num"))


def amounts(text: str, money_only: bool = False) -> List[Amount]:
    return list(iter_amounts(text, money_only))


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
        return abs(a.value - b.value) <= tol
    # at least one figure's scale is self-contradictory: the mantissa is still
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
    r"|\d[\d.,  ]*\d|\d")

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
    plain = re.sub(r"(^|(?<=[.!?…]\s))\s*([A-ZÀ-ÖØ-Þ])", lambda m: m.group(1) + m.group(2).lower(),
                   plain)
    for m in _CAPRUN_RE.finditer(plain):
        cands.append((m.group(1), False))
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
            if not _entity_core(v) or not _looks_like_name(v, quoted):
                continue
            seen.add(key)
            out.append(variants)
    return out


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


def _units(answer: str) -> List[Tuple[str, str]]:
    """(text, unit_kind) for every candidate claim unit of an answer."""
    out: List[Tuple[str, str]] = []
    for raw in (answer or "").splitlines():
        line = raw.strip()
        if not line or _HEADING_RE.match(line):
            continue
        if line.startswith("|"):
            if not _TABLE_RULE_RE.match(line):
                out.append((line, "table-row"))
            continue
        if _BULLET_RE.match(line):
            out.append((_BULLET_RE.sub("", line), "bullet"))
            continue
        for s in split_sentences(line):
            out.append((s, "sentence"))
    return out


def claim_kind(text: str) -> Optional[str]:
    """The strongest factual trigger in a unit, or None when it is prose glue.

    Order matters: a sentence stating a figure is verified as a figure even
    when it also names an entity, because the figure is the falsifiable part.
    """
    body = _strip_citations(text)
    if not re.search(r"[A-Za-zÀ-ÖØ-Þà-öø-þ]", body):
        return None
    if _GLUE_RE.search(body) or _REFUSAL_RE.search(body):
        return None
    amts = amounts(body)
    if any(a.currency or a.unit or a.clash or (a.value or 0) >= 1e4 for a in amts):
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
    for text, unit_kind in _units(answer):
        kind = claim_kind(text)
        if kind is None:
            continue
        body = _strip_citations(text)
        claims.append(Claim(
            text=text.strip(), kind=kind, citations=parse_citations(text),
            amounts=amounts(body), entities=entities(body),
            index=len(claims), unit_kind=unit_kind))
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
            ) -> Tuple[List[EvidenceKey], List[EvidenceKey], List[str]]:
    """(strict scope, same-document fallback scope, unresolvable citations).

    Strict scope is what the claim actually cites. The fallback widens to the
    rest of the cited document: a figure that is real but attached to the
    wrong page is a citation defect, not an invented fact, and the two deserve
    different verdicts.
    """
    docs = {k[0] for k in evidence if k[0] != NOTES_DOC}
    strict: List[EvidenceKey] = []
    wide: List[EvidenceKey] = []
    bad: List[str] = []
    for c in claim.citations:
        if c.doc is None:
            if NOTES_KEY in evidence:
                strict.append(NOTES_KEY)
            else:
                bad.append(c.raw.strip())
            continue
        d = _resolve_doc(c.doc, docs)
        if d is None:
            bad.append(f"{c.doc}" + (f", p.{c.page}" if c.page else ""))
            continue
        wide += [k for k in evidence if k[0] == d]
        if c.page is None:
            here = [k for k in evidence if k[0] == d and k[1] is None]
            strict += here or [k for k in evidence if k[0] == d]
        elif (d, c.page) in evidence:
            strict.append((d, c.page))
        else:
            bad.append(f"{d}, p.{c.page}")
    ded = lambda xs: list(dict.fromkeys(xs))     # noqa: E731
    return ded(strict), ded(wide), ded(bad)


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


def _field_conflict(claim: Claim, text: str) -> Optional[Tuple[Amount, str]]:
    """A figure printed under the claim's own field label that disagrees.

    Line-scoped and label-anchored: the registry prints 'GCF financing (as
    printed): 28,654 million USD; total financing (as printed): 49,654 million
    USD' on ONE line, so the value of a field is the first amount AFTER its
    label, not the first amount on the line.
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
            if _money_like(cand) and not any(amount_matches(cand, a)
                                             for a in claim.amounts):
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


def registry_conflict(doc_id: str, claim: Claim) -> Optional[str]:
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


def _check_entities(claim: Claim, text: str) -> Tuple[bool, List[str]]:
    hay = norm_text(text)
    missing = []
    for variants in claim.entities:
        if not any(norm_text(v) and norm_text(v) in hay for v in variants):
            missing.append(variants[0])
    return (not missing), missing


def _check_years(claim: Claim, text: str) -> Tuple[bool, List[str]]:
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
        return ok, ", ".join(missing)
    if claim.kind == "year":
        ok, missing = _check_years(claim, text)
        return ok, ", ".join(missing)
    if claim.kind == "existence":
        ok, missing = _check_years(claim, text)
        return ok, ", ".join(missing)
    # a claim whose trigger left nothing normalizable (a bare number with no
    # unit, say): fall back to the entity check, then give it to the judge
    ok, missing = _check_entities(claim, text)
    return ok, ", ".join(missing)


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
    for c in claims:
        strict, wide, bad = _scopes(c, evidence)
        flags = ["invalid-citation:" + b for b in bad]

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
        if ok:
            conflict = _field_conflict(c, strict_text)
            if conflict is None and cross_page_conflicts:
                others = [k for k in wide if k not in strict]
                conflict = _field_conflict(c, _text_of(evidence, others))
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
                    known = registry_conflict(d, c)
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

        conflict = _field_conflict(c, strict_text)
        if conflict:
            cand, line = conflict
            out.append(Verdict(c, CONTRADICTED,
                               f"cited evidence states '{cand.raw}' for this field, "
                               f"the answer states '{missing}' ({line})",
                               strict, flags=flags))
            continue

        wide_only = [k for k in wide if k not in strict]
        if wide_only:
            ok2, _ = _verify_against(c, _text_of(evidence, wide_only))
            if ok2:
                out.append(Verdict(
                    c, SUPPORTED,
                    "value found in the cited document, but not on the cited page",
                    strict + wide_only,
                    flags=flags + ["citation-page-mismatch"]))
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
        strict, wide, _ = _scopes(v.claim, evidence)
        payload.append({
            "id": v.claim.index,
            "claim": _strip_citations(v.claim.text).strip()[:400],
            "cited_evidence": _evidence_snippet(evidence, strict or wide[:2]),
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


def _introduced_sources(answer: str, evidence: Evidence) -> List[str]:
    """Citations in a repaired answer that the evidence cannot back.

    The deterministic guard on rule 4 of REPAIR_PROMPT: a repair that invents
    a document or a page is not a repair, and no amount of prompt wording is
    a substitute for checking.
    """
    docs = {k[0] for k in evidence if k[0] != NOTES_DOC}
    bad = []
    for c in parse_citations(answer):
        if c.doc is None:
            continue
        d = _resolve_doc(c.doc, docs)
        if d is None:
            bad.append(f"{c.doc}" + (f", p.{c.page}" if c.page else ""))
        elif c.page is not None and (d, c.page) not in evidence:
            bad.append(f"{d}, p.{c.page}")
    return list(dict.fromkeys(bad))


def repair(answer: str, verdicts: Sequence[Verdict], evidence: Evidence,
           client: Any = None, use_llm: bool = True,
           cross_page_conflicts: bool = True,
           registry_conflicts: bool = True) -> RepairResult:
    """ONE constrained repair pass over the failing claims.

    The model may remove, qualify or correct claims; it may not introduce
    sources. The repaired text is re-verified deterministically against the
    same evidence, and a repair that introduced a citation we cannot back is
    dropped in favour of flagging the original answer.
    """
    verdicts = list(verdicts)
    failed = [v for v in verdicts if v.failed]
    if not failed:
        return RepairResult(answer, "verified", verdicts, answer)

    if client is None and use_llm:
        client = _client()
    if client is None or not use_llm:
        return RepairResult(answer, _status_for(verdicts, False, False), verdicts,
                            answer, notes=["no LLM available: deterministic "
                                           "verdicts only, answer left as written"])

    blocks = []
    for v in failed:
        strict, wide, _ = _scopes(v.claim, evidence)
        blocks.append(
            f"--- {v.status.upper()} claim ---\n{v.claim.text}\n"
            f"why: {v.reason}\n"
            f"EVIDENCE IT MAY USE:\n{_evidence_snippet(evidence, strict or wide[:2])}")
    user = (f"ANSWER TO REPAIR:\n{answer}\n\n"
            f"FAILED CLAIMS ({len(failed)}):\n" + "\n\n".join(blocks))
    raw = _complete(client, REPAIR_PROMPT, user,
                    max_tokens=min(6000, 900 + len(answer) // 2))
    if not raw:
        return RepairResult(answer, _status_for(verdicts, True, False), verdicts,
                            answer, notes=["repair call failed; answer left as written"])

    introduced = _introduced_sources(raw, evidence)
    if introduced:
        return RepairResult(
            answer, _status_for(verdicts, True, False), verdicts, answer,
            repair_rejected=True,
            notes=["repair rejected: it introduced sources not in the evidence — "
                   + "; ".join(introduced[:4])])

    new_verdicts = classify_deterministic(extract_claims(raw), evidence,
                                          cross_page_conflicts, registry_conflicts)
    return RepairResult(raw, _status_for(new_verdicts, True, True), new_verdicts,
                        answer, repaired=True,
                        notes=[f"repaired {len(failed)} failing claim(s)"])


def verify_answer(answer: str, evidence: Evidence, client: Any = None,
                  use_llm: bool = True, allow_repair: bool = True,
                  cross_page_conflicts: bool = True,
                  registry_conflicts: bool = True) -> RepairResult:
    """extract -> classify -> repair, the whole step-5 pass in one call.

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
        llm = client is not None or (use_llm and _client() is not None)
        return RepairResult(answer, _status_for(verdicts, llm, False), verdicts,
                            answer, notes=["repair disabled"])
    return repair(answer, verdicts, evidence, client=client, use_llm=use_llm,
                  cross_page_conflicts=cross_page_conflicts,
                  registry_conflicts=registry_conflicts)
