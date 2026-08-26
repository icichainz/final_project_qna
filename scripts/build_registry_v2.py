#!/usr/bin/env python3
"""Build the provenance-aware fact registry (data/registry_v2.json, schema 2).

Step 2 of docs/rag-correctness-next-steps.html: move canonical proposal fields
out of probabilistic chunk selection while KEEPING every source candidate.

How it works
------------
1. Deterministic pass (pure regex, no model): every page of every extracted
   document is scanned for the GCF funding-proposal template labels. A label
   match opens a small "value window" (rest of the line plus a few following
   lines, stopped by the next heading) and the value is read out of that window.
   Page numbers come from the extraction page markers (rag.parse.split_pages).
2. LLM fallback (optional, capped): ONLY for documents where the deterministic
   pass found nothing at all. Every LLM-proposed raw string is verified to be a
   literal substring of the page it claims, or it is dropped. LLM candidates are
   never marked canonical.
3. Ratified data decisions (2026-08-26). Two files under data/, read here so
   that data/registry_v2.json is a BUILD PRODUCT and is never hand-edited:
   `registry_corrections.json` (58 adjudicated-wrong values plus four ratified
   riders, each with the corrected figure, the page that prints it and the
   quoted print) and
   `registry_absences.json` (51 confirmed (document, field) absences plus the
   corpus-level finding that no document prints its own GCF Board approval
   date). What they change is recorded per document under `meta`; a document
   named by neither file is byte-identical to a build without them. Pass
   --no-decisions to build the pre-ratification baseline.
4. Cross-extractor verification (2026-08-26, serving-wave session). A standing
   arm, not a one-off audit: every CANONICAL money fact the build publishes has
   its figure re-read out of an INDEPENDENT text extraction of the same PDF
   (data/extracted/pymupdf/, produced by a different tool from the qwen VLM
   markdown this builder parses). Offline, deterministic, no model call. A
   figure the independent extraction does not print is FLAGGED, never silently
   trusted: the candidate gains a `cross_check` verdict, the document gains
   `meta.cross_check`, and the corpus census is published under
   `meta.cross_check` at the top level. A flag is a QUESTION for the next
   adjudication, never an automatic correction -- the build never overwrites a
   ratified figure with something an extractor thinks it saw. Pass
   --no-cross-check to build without the arm.

   Why it exists: phase 3 proved every registry raw is literally printed on its
   cited MARKDOWN page. The serving wave then found eight pages where the
   markdown itself prints what the PDF does not -- digit misreads, invented
   table cells, an invented entity name -- which a literal pass over that same
   markdown can never see. Two extractors disagreeing about a figure is the
   cheapest signal there is that one of them made it up.

Serving precedence (OWNER RATIFICATION 2026-08-26, serving-wave session)
------------------------------------------------------------------------
A note voices exactly ONE of these per field, in this order:

    fact-canonical  >  top-level-as-printed  >  confirmed-absence

the template-section value first ("GCF funding requested"), then the flat
registry field the corrections settled ("GCF financing (as printed)"), and a
confirmed absence LAST -- an absence is voiceable only when the build holds no
print of the field anywhere. `absence_meta()` already enforces exactly this and
needed no change for the ratification: it withholds publication of an absence
over any print the fact layer holds, canonical or supporting, and records why.

The FP100/FP142 shape is what the rule is for. Both are REDD+ RBP pilots whose
cover genuinely prints no financing field -- so the absence row is TRUE about
the template -- while the payment figure IS printed under labels deeper in
(FP100 p.103, Table 19 "GCF RBP : 96,452,228"; FP142 p.90 "Total Budget |
$82,000,000"), which is what the ratified top-level correction serves. Recording
those prints as fact-layer SUPPORTING candidates (the add-candidate action)
makes the two claims stop contradicting each other mechanically: the print
outranks the absence, the absence row stays on the record, and no code changed
to make it so.

Nothing is ever invented: a candidate always carries the exact source text
(`raw`). If that text does not parse into a number — or if the page contradicts
itself, printing '28,654 million USD' where the figure and the unit word cannot
both be true — `value` stays null and the raw is published as-is. Document-level
arithmetic that cannot hold (a GCF request above total financing) is reported in
coverage.suspect instead of being shipped as a clean fact.

  python scripts/build_registry_v2.py                 # all docs -> data/registry_v2.json
  python scripts/build_registry_v2.py --limit 5       # smoke test
  python scripts/build_registry_v2.py --no-llm        # deterministic pass only
  python scripts/build_registry_v2.py --only fp274    # substring filter on doc id
  python scripts/build_registry_v2.py --dry-run --out /tmp/reg.json   # nothing written
                                                     # under data/, no model call
  python scripts/build_registry_v2.py --dry-run --no-fallback --out /tmp/base.json
                                                     # the strict-rules baseline
  python scripts/build_registry_v2.py --dry-run --no-decisions --out /tmp/pre.json
                                                     # before the ratified corrections
  python scripts/build_registry_v2.py --dry-run --no-decisions --no-extra-rules \
      --out /tmp/A.json                              # ... and before the new fields

Candidate schema (one entry per source location):
  {"raw": "<exact source text>", "value": <number|null>, "currency": "USD"|"EUR"|null,
   "unit": "million"|"thousand"|"billion"|"years"|"months"|null,
   "page": <int>, "section": "A.8", "status": "canonical"|"supporting"|"conflicting"}

A candidate corrected by a ratified decision carries two more keys:
  "corrected": true, "corrected_from": {<the candidate as it was shipped>}
and on such a candidate `raw` is the RATIFIED figure rather than a literal
page string — the page's own print is quoted in meta.corrections[].quote.
Everywhere else `raw` keeps its usual contract of being copied from the page.

status semantics
  canonical    the template-section value (A.7/A.8/... , or the best available
               templated source for older template eras). At most one per field.
  conflicting  a candidate that IS comparable with the canonical one (both
               numeric, compatible currency and unit) and disagrees beyond the
               precision implied by the two raw strings.
  supporting   agrees with the canonical one, or is not comparable with it
               (no parsed value / incompatible currency / text field / a figure
               far below the canonical total, i.e. a component or a tranche).
               Never an assertion of conflict.

`section` is the section id AS PRINTED on the page; when the page printed none
it reads 'rule:A.8' — the rule that fired, not a claim about the page.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from gcf_qna import config                                    # noqa: E402
from gcf_qna.boards import board_of, year_of                  # noqa: E402
from gcf_qna.rag import split_pages                           # noqa: E402

SOURCE_DIR = config.EXTRACTED_DIR / "vlm" / "qwen_qwen2.5-vl-7b"
REGISTRY_V1 = config.DATA_DIR / "registry.json"
REGISTRY_V2 = config.DATA_DIR / "registry_v2.json"
MAX_LLM_CALLS = 40
# text fields (title / countries / entity) only exist in the front matter; past
# that the same words appear in narrative and assessment sections
TEXT_FIELD_MAX_PAGE = 25

# ---------------------------------------------------------------------------
# number / amount normalization
# ---------------------------------------------------------------------------

# 1,234,567 | 44,709782 | 46,10 | 358.26 | 544.0 | 21128224
# the grouped form must not be followed by more digits, otherwise the mangled
# '44,709782' (a decimal the VLM printed with a comma) reads as '44,709'
_NUM = r"\d{1,3}(?:[,.  ]\d{3})+(?:[.,]\d+)?(?!\d)|\d+(?:[.,]\d+)?"

_UNIT_MULT = {"million": 1e6, "millions": 1e6, "m": 1e6, "mm": 1e6,
              "billion": 1e9, "billions": 1e9, "bn": 1e9,
              "thousand": 1e3, "thousands": 1e3, "k": 1e3}
# a unit word is only applied when the bare number is small enough for it to be
# plausible: the GCF template prints "million USD ($)" as a *currency column
# label*, so "40,751,254 | million USD ($)" is 40.7m, not 40.7 trillion.
_UNIT_CEILING = {1e6: 1e4, 1e9: 1e3, 1e3: 1e7}
# an amount with neither currency nor unit has to be big enough to be money:
# '1', '6' and '30' next to a financing label are table furniture
_MONEY_FLOOR = 1e4
# ... and an INFERRED scale that lands above any funding proposal ever approved
# (the largest is ~USD 1.5bn) is the printed unit word being wrong, not a real
# figure: '68.780 | billion USD ($)' on a page whose rows sum to 68.78m
_MAX_PLAUSIBLE = 5e9

_CUR_MAP = {"usd": "USD", "us$": "USD", "$": "USD", "u$s": "USD",
            "eur": "EUR", "euro": "EUR", "euros": "EUR", "€": "EUR"}

_PLACEHOLDER = re.compile(
    r"enter\s+(?:number|amount|years?|date|%|the)|^\W*options\W*$|^\W*n/?a\W*$"
    r"|click here|choose an item|^\W*tbd\W*$|^\W*\.{2,}\W*$|^\W*select\b", re.I)

# a number that is really a placeholder example, a template hint or page furniture
_NOISE_BEFORE = re.compile(
    r"(?:enter\s+(?:number|amount)|e\.?\s?g\.?|example|page|version|v\.)[^\d]{0,6}$", re.I)
# digits glued to a word ('JAH30', 'V2.1') are never the value of a field
_GLUED = re.compile(r"[A-Za-z0-9]$")
# units that mean "this is not money"
_NOT_MONEY_AFTER = re.compile(
    r"^\s{0,3}(?:%|percent|years?|months?|days?|weeks?|tco2|tco₂|t\s?co2|ha\b|km|people|"
    r"persons|beneficiaries|households|of\s+the\s+total|pages?)", re.I)


def to_number(tok: str) -> Optional[float]:
    """'49,751,264' -> 49751264.0 ; '46,10' -> 46.1 ; '358.26' -> 358.26.

    Separator rules (US-first, which is what the GCF corpus prints):
      * both ',' and '.' present -> the LAST one is the decimal separator;
      * a single separator followed by exactly 3 digits -> thousands when ','
        (US grouping), decimal when '.' (US decimal);
      * a single separator followed by any other digit count -> decimal
        ('44,709782' is a mangled '44.709782').
    Returns None when the token is not a number at all.
    """
    s = tok.strip().replace(" ", " ")
    s = re.sub(r"(?<=\d) (?=\d)", "", s)              # space-grouped thousands
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
                s2 = "".join(groups)                   # 1,234,567 / 1.234.567
            elif len(groups) == 2 and grouped and sep == ",":
                s2 = "".join(groups)                   # 1,234 (US thousands)
            else:
                s2 = groups[0] + "." + "".join(groups[1:])
    try:
        return float(s2)
    except ValueError:
        return None


def granularity(tok: str, mult: float) -> float:
    """Smallest amount the raw token can distinguish (its printed precision).

    '46,10' with a million unit distinguishes 0.01m = 10 000; '40,751,254'
    distinguishes 1. Used so a rounded restatement is not called a conflict
    while two fully-printed figures that differ by 240k are.
    """
    s = tok.strip().replace(" ", " ")
    s = re.sub(r"(?<=\d) (?=\d)", "", s)
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


_CUR_THEN_SCALE = re.compile(
    r"\s*(?:US\$|USD|EUR|euros?|\$)\s*(?P<u>millions?|billions?|thousands?)\b",
    re.I)
_AMOUNT_RE = re.compile(
    r"(?P<pre>USD|US\$|EUR|€|\$)?[ \t]{0,2}"
    r"(?P<num>" + _NUM + r")"
    # 'M'/'MM'/'K' only count as scale words when a currency follows them
    # ('28 M USD'), never in prose ('5 m of pipe', '60 m onths')
    r"(?P<sep>[ \t|,;:]{0,4})(?P<unit>million[s]?|billion[s]?|thousand[s]?|bn|MM?|K)?"
    r"(?P<sep2>[ \t|(]{0,4})(?P<post>USD|US\$|EUR|euros?|€|\$)?", re.I)

_CUR_NEARBY = re.compile(r"(USD|US\$|EUR|euros?|€|\$)", re.I)
_CUR_TAIL = re.compile(r"(?:USD|US\$|EUR|euros?|€|\$)[ \t]*$", re.I)
# a scale word later in the same row, with whatever sits between it and the figure
_LOOSE_UNIT = re.compile(r"(?P<gap>[^\n]{0,24}?)\b(?P<u>million|billion|thousand)s?\b", re.I)
_ONLY_PLACEHOLDER = re.compile(r"[\s|(),;:*-]*(?:enter\s+(?:amount|number|years?)|options|n/?a)?"
                               r"[\s|(),;:*-]*", re.I)


# The 'e.g.' arm of _NOISE_BEFORE is dot-optional, so 'e' + optional space +
# 'g' fires on ordinary words: 'by the GCF\n\n32.8 million Euros' reads as an
# example and A.8's value is dropped (FP230). This arm demands a real dot. It
# is used ONLY by the fallback pass, so no shipped reading moves.
_NOISE_BEFORE_STRICT = re.compile(
    r"(?:enter\s+(?:number|amount)|e\.\s?g\.?|eg\.|example|page|version|v\.)[^\d]{0,6}$", re.I)
# markdown emphasis between a figure and the unit word that scales it
# ('| **37.6** million USD ($) |'). Emphasis is markup, not print, so the
# fallback pass reads the window with it removed; `raw` then carries the
# printed text without the markdown, exactly as read_text already does.
_EMPH = re.compile(r"\*\*|__")
# 'US$ 1,358.0/tonne' under 'Total Programme financing' is a unit cost, not the
# programme's financing. Fallback-only, alongside the tightened noise guard.
_PER_UNIT = re.compile(r"^\s{0,2}(?:/|per\s)\s?(?:t\b|ton|tonne|tco2|ha\b|capita|"
                       r"year|yr\b|person|household|beneficiar)", re.I)


def read_amounts(window: str, limit: int = 1,
                 skip_lines: Optional[re.Pattern] = None,
                 loose: bool = False) -> List[dict]:
    """Up to `limit` money amounts from distinct lines of a value window.

    A template block often prints several figures under one label ('Total GCF
    funding requested: 358.26 million USD' / 'Multi-country ...: 190.00 million
    USD'); each is kept as its own candidate so the disagreement stays visible.
    `skip_lines` drops whole lines whose own label belongs to another field —
    an 'A.8.1' block that lists '- Co-financing: 60,477' before '- Total:
    82,849' must not hand the co-financing figure to the GCF request.
    """
    out, lines_used = [], set()
    starts = [0]
    for ln in window.split("\n"):
        starts.append(starts[-1] + len(ln) + 1)
    for p in _iter_amounts(window, loose):
        at = p.pop("_at")
        line = window.count("\n", 0, at)
        if line in lines_used:
            continue
        text = window[starts[line]:starts[line + 1] - 1]
        if skip_lines and skip_lines.search(text) and not _NOT_GCF_NEGATED.search(text):
            lines_used.add(line)
            continue
        lines_used.add(line)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def read_amount(window: str) -> Optional[dict]:
    """First real money amount in a value window -> {raw,value,currency,unit}."""
    got = read_amounts(window, 1)
    return got[0] if got else None


def _iter_amounts(window: str, loose: bool = False):
    """Every real money amount in a window, in order.

    Skips template placeholders ('Enter amount'), percentages, durations and
    bare years. `raw` is the exact source substring that was read.
    `loose` swaps in the dot-requiring 'e.g.' guard (fallback pass only).
    """
    noise = _NOISE_BEFORE_STRICT if loose else _NOISE_BEFORE
    for m in _AMOUNT_RE.finditer(window):
        num = m.group("num")
        # 'USD35 million' glues the currency to the figure: that is a currency
        # marker, not the 'JAH30' word-glue the guard is meant to reject
        head = m.start() if m.group("pre") else m.start("num")
        # 'USD$ 500 M': the currency may be spelled twice, and the word-glue
        # guard must not read the 'D' of USD as the glue
        before = _CUR_TAIL.sub("", window[max(0, head - 24):head])
        after = window[m.end("num"):m.end("num") + 24]
        if noise.search(before) or _GLUED.search(before):
            continue
        if _NOT_MONEY_AFTER.match(after):
            continue
        if loose and _PER_UNIT.match(after):
            continue
        val = to_number(num)
        if val is None or val == 0:
            continue
        unit_tok = (m.group("unit") or "").lower()
        mult = _UNIT_MULT.get(unit_tok, 1.0)
        # 'M'/'MM'/'K' are scale words only beside a currency ('28 M USD',
        # 'USD$ 500 M'); anywhere else 'm' is metres and 'K' is a label
        abbrev = unit_tok in ("m", "mm", "k")
        # The number and the printed unit word contradict each other: '28,654
        # million USD' / '68.780 | billion USD ($)' — the GCF template prints
        # 'million USD ($)' as a currency-COLUMN label, so the scale is unknown.
        # Publish the raw with NO value rather than a number that contradicts
        # what the page says (either reading would be an invention).
        clash = mult > 1 and (val >= _UNIT_CEILING.get(mult, 0)
                              or val * mult > _MAX_PLAUSIBLE)
        if abbrev and not (m.group("pre") or m.group("post")
                           or _CUR_NEARBY.match(window[m.end():m.end() + 6] or "")):
            unit_tok, mult = "", 1.0
        if mult == 1.0 and not unit_tok:
            # '25 USD million' (FP155 p.8, cross-check stop #3): the template
            # sometimes prints the currency BETWEEN the figure and its scale
            # word, so <unit> never binds adjacently. Bind it only when
            # nothing but a currency token separates them, and only when the
            # bound value stays plausible - otherwise fall through to the
            # clash path, which publishes the raw without a value.
            m4 = _CUR_THEN_SCALE.match(window[m.end("num"):m.end("num") + 30])
            if m4:
                cand = _UNIT_MULT.get(m4.group("u").lower(), 1.0)
                if cand > 1 and not (val >= _UNIT_CEILING.get(cand, 0)
                                     or val * cand > _MAX_PLAUSIBLE):
                    unit_tok, mult = m4.group("u").lower(), cand
        if not clash and mult == 1.0:
            # a scale word sits further along the same table row but could not
            # bind to the figure ('| 32,500 plus | million USD ($) |'). Binding
            # it would contradict the figure, so the row's scale is unknowable:
            # publish the raw without a value rather than an unscaled number.
            # Text that is only a template PLACEHOLDER between the two means the
            # cell was left unfilled ('40,751,254Enter amount | million USD ($)')
            # — boilerplate, not a scale the author chose.
            m3 = _LOOSE_UNIT.search(window[m.end("num"):m.end("num") + 34])
            if m3 and not _ONLY_PLACEHOLDER.fullmatch(m3.group("gap")):
                loose = _UNIT_MULT.get(m3.group("u").lower(), 1.0)
                if val >= _UNIT_CEILING.get(loose, 0) or val * loose > _MAX_PLAUSIBLE:
                    clash, mult = True, loose
        cur_tok = (m.group("pre") or m.group("post") or "").lower()
        if not cur_tok:
            line_tail = window[m.end():].split("\n", 1)[0][:40]
            m2 = _CUR_NEARBY.search(line_tail)
            cur_tok = m2.group(1).lower() if m2 else ""
        if not clash:
            # a bare 4-digit integer with no currency and no unit is a year
            if not cur_tok and mult == 1.0 and re.fullmatch(r"(19|20)\d{2}", num):
                continue
            if mult == 1.0 and val < _MONEY_FLOOR:
                if not (m.group("pre") or m.group("post")):
                    # bare '1' / '6' / '30' / the '9' in 'proposals, 9 out of 17':
                    # table furniture and prose, even when the ROW mentions USD
                    continue
                # '999.9 USD' under 'A.7 Total financing' is a real print whose
                # scale the page does not give (no GCF proposal is under $1M):
                # publish it with no value instead of a $999.90 total
                clash, mult = True, 1.0
        raw = window[m.start():m.end()].strip(" \t|,;:(")
        if raw.count("(") > raw.count(")"):      # 'USD 258 million (USD' -> drop the
            raw = raw[:raw.rfind("(")].strip()   # half-eaten trailing cell
        yield {"raw": raw or num,
               "value": None if clash else round(val * mult, 2),
               # scale unknown, mantissa known: '28,654 million' vs '26,654
               # million' still disagree, and that stays detectable
               "_bare": round(val, 6) if clash else None,
               "_bare_unit": unit_tok if clash else None,
               "currency": _CUR_MAP.get(cur_tok.rstrip("s") if cur_tok not in ("us$",) else cur_tok),
               "unit": None if clash else {1e6: "million", 1e9: "billion",
                                           1e3: "thousand"}.get(mult),
               "_grain": granularity(num, 1.0 if clash else mult), "_at": m.start("num")}


def read_count(window: str) -> Optional[dict]:
    """A plain count (beneficiaries, tCO2eq): number without currency semantics."""
    for m in _AMOUNT_RE.finditer(window):
        num = m.group("num")
        before = window[max(0, m.start("num") - 24):m.start("num")]
        if _NOISE_BEFORE.search(before) or _GLUED.search(before):
            continue
        if window[m.end("num"):m.end("num") + 2].startswith("%"):
            continue
        val = to_number(num)
        if val is None or val == 0:
            continue
        unit_tok = (m.group("unit") or "").lower()
        mult = _UNIT_MULT.get(unit_tok, 1.0)
        # 'M'/'MM'/'K' are scale words only beside a currency ('28 M USD',
        # 'USD$ 500 M'); anywhere else 'm' is metres and 'K' is a label
        abbrev = unit_tok in ("m", "mm", "k")
        if mult > 1 and val >= _UNIT_CEILING.get(mult, 0):
            unit_tok, mult = "", 1.0
        raw = window[m.start():m.end("num") + 24].split("\n")[0].strip(" \t|,;:")
        return {"raw": raw, "value": round(val * mult, 2), "currency": None,
                "unit": {1e6: "million", 1e9: "billion", 1e3: "thousand"}.get(mult),
                "_grain": granularity(num, mult)}
    return None


_PERIOD_RE = re.compile(r"(?P<n>\d{1,4}(?:[.,]\d{1,2})?)\s*(?P<u>years?|months?)", re.I)


def read_period(window: str) -> Optional[dict]:
    """'5 years / 60 months' -> 60 months; '60 Months' -> 60 months."""
    hits = [(m.group(0), to_number(m.group("n")), m.group("u").lower().rstrip("s"))
            for m in _PERIOD_RE.finditer(window.split("\n\n")[0])]
    hits = [h for h in hits if h[1]]
    if not hits:
        return None
    best = next((h for h in hits if h[2] == "month"), hits[0])
    raw_line = window.strip().split("\n")[0].strip(" \t|-*:")
    raw = raw_line if best[0] in raw_line else best[0]
    return {"raw": raw[:120], "value": best[1], "currency": None,
            "unit": best[2] + "s", "_grain": 0.5}


_ESS_RE = re.compile(r"(?:categor(?:y|ies)\W{0,6}|^\W{0,4})(?P<c>I-?[123]|FI-?[123]?|[ABC])\b",
                     re.I | re.M)


def read_ess(window: str) -> Optional[dict]:
    for m in _ESS_RE.finditer(window):
        c = m.group("c").upper().replace(" ", "")
        # a standalone 'A'/'B'/'C' must really be alone on its line fragment
        tail = window[m.end():m.end() + 3]
        if tail[:1].isalpha():
            continue
        return {"raw": m.group(0).strip(" \t|-*:"), "value": None, "currency": None,
                "unit": None, "_grain": 0.0, "_text": c}
    return None


# A printed date, in the forms the corpus prints them. Taken verbatim from the
# H9 cover-page probe that measured the two date fields below (69 / 42
# documents), so the rules read exactly what the diagnosis counted.
_DATE = (r"(?:\d{1,2}[ /.-]\d{1,2}[ /.-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}"
         r"|\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|"
         r"October|November|December)\s+\d{4}"
         r"|(?:January|February|March|April|May|June|July|August|September|October|"
         r"November|December)\s+\d{1,2},?\s+\d{4})")
_DATE_RE = re.compile(_DATE, re.I)
# 'N/A' under a date label is the template saying the field does not apply — it
# is not a date, and recording it as one would be inventing a fact
_DATE_NOT_A_DATE = re.compile(r"^\W{0,4}(?:n\.?/?a\.?|tbd|tbc|not applicable|none|-+)\W{0,4}$",
                              re.I)


# the cover table's OTHER labelled rows. A date label whose own cell is empty
# ('Expected approval from accredited entity\'s Board (if applicable) | TBD')
# must not read the next row's date ('Estimated implementation start and end
# date | Start: 01/12/2020') — that would publish a start date as an approval
# date. Used as the extra window terminator of the EXTRA pass only.
_STOP_DATE = re.compile(
    r"[\s|\-–*>#]{0,8}(?:Estimated\s+implementation|Expected\s+financial\s+close"
    r"|Expected\s+lifetime|Project\s*/?\s*programme?\s+lifespan|Total\s+lifespan"
    r"|Project\s+contact|Contact\s+person|ES[SG]\s+category|Project\s+size"
    r"|Financial\s+instrument|Implementation\s+period|Total\s+financing|Total\s+GCF"
    r"|Results?\s+area|Sector\b|Modality|Accredited\s+[Ee]ntit|Countr(?:y|ies)\b"
    r"|Date\s+of\s+\w{0,9}\s?submission|Expected\s+approval|(?:Expected|Estimated)\s+date"
    r"|Environmental\s+and\s+social)", re.I)


def read_date(window: str, skip_lines: Optional[re.Pattern] = None) -> Optional[dict]:
    """The first printed date in the window, kept exactly as printed.

    `value` stays null: a date is not a number, and the corpus prints forms that
    cannot be disambiguated without inventing something ('[2020/13/05]',
    '25 June 2018; V2 28 May 2019'). The raw print is the fact; parsing it into
    an ISO date is a decision for whoever serves the field, not for the store.
    """
    for line in window.split("\n")[:4]:
        v = line.strip().strip("|").strip()
        v = re.sub(r"^[\s#>*\-–:.]+", "", v)
        v = re.sub(r"\s*\|\s*", " | ", v).strip(" |")
        v = re.sub(r"\*\*|__", "", v).strip()
        if not v or _DATE_NOT_A_DATE.match(v) or _PLACEHOLDER.search(v):
            continue
        if skip_lines and skip_lines.search(v):
            continue                      # a date about somebody ELSE's approval
        m = _DATE_RE.search(v)
        if not m:
            continue
        # the template also prints an explanatory sentence under the label and
        # ends it with the date ('This is the date that the Accredited Entity
        # ... (if applicable): 6/15/2020'). Clipping that line at 120 characters
        # would publish a fact whose own raw does not contain the date, so a
        # date printed far into a long line is quoted on its own.
        raw = v[:120] if m.start() < 60 and len(v) <= 120 else m.group(0)
        return {"raw": raw, "value": None, "currency": None, "unit": None,
                "_grain": 0.0}
    return None


def read_text(window: str) -> Optional[dict]:
    """First non-empty, non-placeholder text value in the window."""
    for line in window.split("\n"):
        v = line.strip().strip("|").strip()
        v = re.sub(r"^[\s#>*\-–:.]+", "", v)
        v = re.sub(r"\s*\|\s*", " | ", v).strip(" |")
        v = re.sub(r"\*\*|__", "", v).strip()
        v = re.sub(r'^[“"\'\[]+|[”"\'\];,]+$', "", v).strip()
        if not v or _PLACEHOLDER.search(v) or len(v) < 2:
            continue
        if re.fullmatch(r"[-–—_=|\s]+", v):
            continue
        return {"raw": v[:300], "value": None, "currency": None, "unit": None,
                "_grain": 0.0, "_text": v[:300]}
    return None


# ---------------------------------------------------------------------------
# label rules
# ---------------------------------------------------------------------------

# markdown noise before a label: heading hashes, bullets, table pipes, bold
# markers, and template enumerators like '(a)' / '(i + ii + iii)'. Kept as flat
# character-class repetitions — a nested '(?:[#|-]+ *)*' backtracks
# catastrophically on markdown table rules ('|---|---|---|').
_PREFIX = r"(?:^|\n)[ \t#>*|\-–]{0,24}(?:\([^)\n]{1,14}\)[ \t]{0,4}){0,2}[ \t*]{0,4}"
# an optional printed section number, kept as provenance ('A.8', 'A.1.5', 'C.1')
_SEC = r"(?:(?P<sec>[A-H][.\s]?\s?\d{1,2}(?:\s?[.]\s?\d{1,2}){0,2})\s*[.):\-–]?\s*\**[ \t]*)?"


# A fallback label often sits behind an enumerator the strict prefix does not
# model — 'a) A funding proposal titled', '(b) Requested GCF amount' — or
# behind one the extraction mangled: 'A.B. Total GCF funding requested'
# (FP268), '* B. GCF financing to recipient' (b20-10-add04). Only the fallback
# pass uses this prefix, and only for a field the document has NO candidate
# for, so a looser prefix can never re-read a document the strict rules read.
_ENUM = r"(?:\([^)\n]{1,14}\)[ \t]{0,4}|\(?[A-Za-z0-9]{1,3}[.)][ \t]{0,4}){0,3}"
_PREFIX_LOOSE = r"(?:^|\n)[ \t#>*|\-–]{0,24}" + _ENUM + r"[ \t*]{0,4}"


def _mk(label: str, terminated: bool, loose: bool = False) -> re.Pattern:
    """Label regex. `terminated` (text fields) demands that the label is followed
    by ':' , '|' or the end of the line — 'Programme Title: X' and '| Title | X |'
    are field labels, 'the Accredited Entity shall ...' is prose."""
    tail = r"[ \t]*\**" + (r"(?=[ \t]*(?:[:|]|$|\n))" if terminated else "") + r"[ \t]*[:|.\-–]?"
    prefix = _PREFIX_LOOSE if loose else _PREFIX
    return re.compile(prefix + _SEC + r"\**[ \t]*(?:" + label + r")" + tail, re.I)


class Rule:
    """One labelled source of one field.

    `fallback` rules are consulted in a second pass, and only for a field the
    strict pass left with no candidate at all. That is what keeps the parser
    additive: a document the shipped rules already read cannot change.
    """

    def __init__(self, field, label, section, kind, rank=0, max_page=None, window=(6, 400),
                 amounts=1, skip_lines=None, fallback=False, era=None,
                 only_if_missing=False, label_note=None):
        self.field, self.section, self.kind, self.rank = field, section, kind, rank
        self.max_page, self.window, self.amounts = max_page, window, amounts
        self.fallback = fallback
        # EXTRA_RULES only: the template family the rule is written for, whether
        # it may fire at all when the field already has a candidate, and the
        # label AS PRINTED when it differs from the field's name (kept in the
        # candidate's section, the way an instrument row keeps 'A.10 Grant')
        self.era, self.only_if_missing, self.label_note = era, only_if_missing, label_note
        self.skip_lines = re.compile(skip_lines, re.I) if skip_lines else None
        self.re = _mk(label, terminated=(kind in ("text", "date")), loose=fallback)


# lines inside a value window that belong to a DIFFERENT field: their figure may
# never be read as the GCF request (co-financing, counterpart, in-kind, etc.)
_NOT_GCF_LINE = (r"co[\s\-]?financ|co[\s\-]?funding|counterpart|in[\s\-]?kind|"
                 r"parallel financing|other sources|government contribution")
# ... except when the line says it EXCLUDES them: 'Total requested (excl.
# co-financing): 35 million USD' is the GCF request, not a co-financing line
_NOT_GCF_NEGATED = re.compile(
    r"(?:excl(?:uding|\.)?|without|net of|other than|apart from|before)\W{0,8}"
    r"(?:the\s+|any\s+)?(?:co[\s\-]?financ|co[\s\-]?funding)", re.I)


# A.7 / A.8 (template v2/v3) come first; the section-C and old section-B labels
# are separate rules so that a candidate's default section id stays truthful
# when the document does not print the section number itself.
_TOTAL_FIN_A7 = r"Total financing(?:\s*\((?:GCF|SCF)[^)\n]{0,40}\))?"
_TOTAL_FIN_B2 = r"Total project financing|Total project cost"
_GCF_REQ_A8 = (r"Total GCF funding(?:\s+(?:requested|required))?|GCF funding requested"
               r"|Amount of GCF funding requested")
_GCF_REQ_C1 = r"(?:Requested|Received) GCF funding|Total funding requested"
_GCF_REQ_B2 = r"Requested GCF amount|GCF fin(?:ancing|ance) to recipient"
# lines inside a TOTAL-financing window that state the GCF part instead: FP233
# prints 'A7 Total funding required (GCF + co-financing)' over two rows, the
# request and the total, and the request is not a second reading of the total.
_NOT_TOTAL_LINE = (r"GCF funding requ|Total GCF funding|requested GCF|co[\s\-]?financ"
                   r"|co[\s\-]?funding|counterpart|in[\s\-]?kind")

RULES: List[Rule] = [
    # --- cover / section A.1 identity -------------------------------------
    Rule("title", r"Project\s*/?\s*(?:or\s+)?[Pp]rogramme?\s*title|Project title|Programme title",
         "A.1.1", "text", rank=0, max_page=TEXT_FIELD_MAX_PAGE),
    Rule("title", r"A funding proposal (?:titled|entitled)", "cover", "text", rank=1, max_page=4),
    Rule("countries", r"Countr(?:y|ies)\s*(?:\(\s?ies\s?\))?(?:\s*\(?s?\)?)?(?:\s*[/&]\s*[Rr]egion)?"
                      r"|Country of operation|Country \(ies\)", "A.1.3", "text",
         rank=0, max_page=TEXT_FIELD_MAX_PAGE),
    Rule("accredited_entity", r"Accredited\s+[Ee]ntity(?:\s*\(ies\))?", "A.1.5", "text",
         rank=0, max_page=TEXT_FIELD_MAX_PAGE),
    Rule("national_designated_authority", r"National designated authority\s*(?:\(\s?i?e?s?\s?\))?",
         "A.1.4", "text", rank=0, max_page=TEXT_FIELD_MAX_PAGE),
    Rule("executing_entity", r"Executing [Ee]ntity(?:\s*(?:/|\s)\s*[Bb]eneficiary)?"
                             r"(?:\s+information)?", "A.20", "text",
         rank=0, max_page=TEXT_FIELD_MAX_PAGE),

    # --- A.5 - A.14 template block ----------------------------------------
    # the parenthetical must be swallowed by the label: '(Core indicator 1 - GHG
    # emissions avoided or removed/sequenced)' would otherwise donate its '1'
    Rule("mitigation_outcome", r"Expected mitigation (?:outcomes?|impacts?)(?:\s*\([^)\n]{0,90}\))?",
         "A.5", "count", rank=0),
    Rule("adaptation_outcome", r"Expected adaptation (?:outcomes?|impacts?|benefits?)"
                               r"(?:\s*\([^)\n]{0,90}\))?", "A.6", "beneficiaries", rank=0),
    # amounts=3: one template label often heads several printed figures
    # ('Total GCF funding requested: 358.26 million USD' + 'Multi-country: 190.00
    # million USD'); each line becomes its own candidate instead of vanishing
    Rule("total_financing", _TOTAL_FIN_A7, "A.7", "amount", rank=0, amounts=3),
    Rule("gcf_funding_requested", _GCF_REQ_A8, "A.8", "amount", rank=0, amounts=3,
         skip_lines=_NOT_GCF_LINE),
    Rule("project_size", r"Project size(?:\s+category)?(?:\s*\([^)\n]{0,50}\))?", "A.9", "text",
         rank=0, max_page=TEXT_FIELD_MAX_PAGE),
    Rule("financial_instruments",
         r"Financial instruments?\s*\(?[as]?\)?\s*requested(?:\s+for\s+(?:the\s+)?(?:GCF\s+funding|this project))?"
         r"|Financial instrument\(s\) requested[^\n]{0,40}", "A.10", "instruments",
         rank=0, window=(10, 700)),
    Rule("implementation_period", r"Implementation period|Expected implementation period",
         "A.11", "period", rank=0),
    Rule("lifespan", r"Total lifespan|Project lifespan", "A.12", "period", rank=0),
    Rule("ess_category", r"ES[SG] category|Environmental and social(?: risk)? category",
         "A.14", "ess", rank=0),

    # --- section C (modern) / section B (old) financing -------------------
    Rule("gcf_funding_requested", _GCF_REQ_C1, "C.1(a)", "amount", rank=1, window=(8, 500),
         amounts=2, skip_lines=_NOT_GCF_LINE),
    Rule("gcf_funding_requested", _GCF_REQ_B2, "B.2(b)", "amount", rank=2, window=(8, 500),
         skip_lines=_NOT_GCF_LINE),
    Rule("total_financing", _TOTAL_FIN_B2, "B.2(a)", "amount", rank=1, window=(8, 500)),
    Rule("co_financing", r"Total co-financing|Co-financing to recipient", "C.1(b)", "amount",
         rank=0),
]

# ---------------------------------------------------------------------------
# fallback rules: template VARIANTS, consulted only for a field the strict pass
# left empty
# ---------------------------------------------------------------------------
# Every entry below was written against a document the strict rules leave with
# no candidate for that field, and each names the corpus evidence it was
# written for. Because the fallback pass runs per field and only when that
# field is empty, adding one can never move a value the strict rules found.
FALLBACK_RULES: List[Rule] = [
    # 'Program title' (one m) is the REDD+ RBP cover's spelling and the v1
    # template's 'A.1.1 Project / program title'; 'Projects/Programme title'
    # and 'Project (programme) title' are extraction spellings of the same
    # label.  b27-02-add04, b23-02-add04, b22-10-add02, b30-02-add07,
    # b15-13-add01, b14-07-add10, b11-04-add03, b35-02-add07.
    Rule("title", r"(?:Projects?\s*[/&(]{0,2}\s*)?(?:or\s+)?[Pp]rogramm?e?\)?\s*title"
                  r"|Project\s*title",
         "A.1.1", "text", rank=0, max_page=TEXT_FIELD_MAX_PAGE, fallback=True),
    # the addendum cover line, in the forms the strict rank-1 rule misses:
    # 'a) A funding proposal SUMMARY titled', 'THE funding proposal titled'.
    Rule("title", r"(?:[a-z]\s*[.)]\s*)?(?:An?|The|This)\s+funding proposal"
                  r"(?:\s+(?:summary|package))?\s+(?:titled|entitled)",
         "cover", "text", rank=1, max_page=4, fallback=True),
    # 'Country/cities' and 'Country/countries' are extraction spellings of
    # 'Country(ies)'; 'A.1.2 Country location' is the v1 label. b39-02-add08,
    # b36-02-add02, b16-07-add05.
    Rule("countries", r"Countr(?:y|ies)\s*[/&]\s*(?:cities|countries|regions?|areas?)"
                      r"|Country location|Country of implementation",
         "A.1.3", "text", rank=0, max_page=TEXT_FIELD_MAX_PAGE, fallback=True),
    # 'Accredited Entities' (plural cover, b42-02-add10) and 'Accrediting
    # Entity' (b26-02-add03).
    Rule("accredited_entity", r"Accredit(?:ed|ing)\s+[Ee]ntit(?:y|ies)(?:\s*\(\s?i?e?s?\s?\))?",
         "A.1.5", "text", rank=0, max_page=TEXT_FIELD_MAX_PAGE, fallback=True),

    # A.7 / A.8 behind a mangled enumerator ('A.B. Total GCF funding
    # requested', b42-02-add10) or behind the 'e.g.' guard ('by the GCF' ahead
    # of the figure, b38-02-add10).
    Rule("total_financing", _TOTAL_FIN_A7 + r"|Total funding requi?red(?:\s*\([^)\n]{0,40}\))?"
                                            r"|Total funding\s*\((?:GCF|SCF)[^)\n]{0,40}\)",
         "A.7", "amount", rank=0, amounts=3, skip_lines=_NOT_TOTAL_LINE, fallback=True),
    Rule("gcf_funding_requested",
         _GCF_REQ_A8 + r"|GCF\s+total\s+(?:\w+\s+){0,2}funding\s+requested",
         "A.8", "amount", rank=0, amounts=3, skip_lines=_NOT_GCF_LINE, fallback=True),
    # 'Total project finance' without the -ing (b14-07-add08) and the
    # emphasis-split '| **37.6** million USD ($) |' (b21-10-add06).
    Rule("total_financing", r"Total (?:project|programme|program)\s+financ(?:ing|e)"
                            r"|Total (?:project|programme|program)\s+cost",
         "B.2(a)", "amount", rank=1, window=(8, 500), fallback=True),
    # the v1 B.2(b) block's own total row. The strict B.2(b) rule opens on
    # 'Requested GCF amount' but its 8-line window closes before this row,
    # which is what the block actually sums to.
    Rule("gcf_funding_requested",
         r"Total requested(?:\s+financing)?(?:\s*\([^)\n]{0,60}\))?",
         "B.2(b)", "amount", rank=2, window=(12, 700), skip_lines=_NOT_GCF_LINE,
         fallback=True),
    # B.2(b) label spellings the strict rule does not carry.
    Rule("gcf_funding_requested",
         r"(?:Requested|Required|Disaggregated)\s+GCF\s+amount"
         r"|(?:Total\s+)?GCF\s+funds\s+requested"
         r"|GCF fin(?:ancing|ance) to recipient",
         "B.2(b)", "amount", rank=2, window=(12, 700), skip_lines=_NOT_GCF_LINE,
         fallback=True),
]

# ---------------------------------------------------------------------------
# EXTRA rules: the owner-ratified data decisions of 2026-08-26, run in a THIRD
# pass with its own pre-filter
# ---------------------------------------------------------------------------
# Why a third pass rather than more entries in the two above: adding a label to
# `_ANY_LABEL` / `_ANY_LABEL_FB` widens which PAGES those passes scan, and a
# page they did not scan before can hand an existing field a new candidate. A
# separate pass with its own pre-filter cannot: the strict and fallback passes
# see exactly the pages they saw before, and this one may only write a field it
# is named for.
#
# Two of the three rules add NEW fields. They are honestly named after the
# labels they read and they are NOT approval dates: the H9 sweep found that no
# document in the corpus prints its own GCF Board approval date (0 true
# positives in 273), because the approval is created by the Board decision
# document, which is not in this corpus. Nothing here may be reached by an
# approval-shaped question — that mapping does not exist and must not be added
# without the field being renamed first.
_A1X_ERAS = ("A.1.x block (FP template v1)",
             "A.1.x block (FP template v1, variant numbering)")

EXTRA_RULES: List[Rule] = [
    # 'Date of first submission' / 'Date of current submission' / 'A.1.9 Date of
    # submission' — a date printed beside the label in 69 of 273 documents. The
    # '/ version number' tail is part of the label in the v2/v3 cover, and
    # binding it is what keeps 'Date of first submission/version number:
    # 2020-03-11[v.1' out of the accredited-entity slot it bled into (FP144).
    Rule("date_of_submission",
         r"Date of (?:first|current|second|third|initial)?\s*submission"
         r"(?:\s*/\s*version\s*number)?",
         "A.1.9", "date", rank=0, max_page=15, window=(4, 200)),
    # 'A.13 Expected date of AE internal approval' / 'Expected approval from
    # accredited entity's Board' — the ACCREDITED ENTITY's own internal board,
    # dated in 42 of 273 documents. Not the GCF Board, and not an approval that
    # has happened: the label itself says 'expected'.
    Rule("ae_board_approval_date",
         r"(?:Expected\s+)?approval\s+from\s+accredited\s+entity'?s?\s+Board"
         r"(?:\s+of\s+Directors)?(?:\s*\([^)\n]{0,30}\))?"
         r"|(?:Expected|Estimated)\s+date\s+of\s+AE\s*(?:internal)?\s*approval",
         "A.13", "date", rank=0, max_page=15, window=(4, 200),
         # the template's own explanatory sentence sometimes carries the date,
         # and sometimes carries a sentence about the GCF Board's decision
         # instead ('IDB approval ... will follow GCF board approval on
         # 15/03/2022'). That date is not this field, and H9 says the corpus
         # never prints this proposal's GCF approval date at all.
         skip_lines=r"GCF\s*board|Board of the Green Climate Fund|GCF\s+Board's"),
    # OWNER RATIFICATION 2026-08-26 (3): in the earliest template the slot that
    # later reads 'Accredited entity' is printed 'A.1.5 Implementing entity'
    # (273_gcf-b11-04-add01, the only document in the corpus that turns on it).
    # Mapping the two is a decision about what the field MEANS, which is why the
    # diagnosis left it to the owner. Era-gated to the v1 families and consulted
    # only when the document has no accredited_entity at all, so it can neither
    # rename a modern AE nor displace one the other passes read. The printed
    # label rides along in the section.
    Rule("accredited_entity", r"Implementing\s+[Ee]ntit(?:y|ies)", "A.1.5", "text",
         rank=3, max_page=TEXT_FIELD_MAX_PAGE, era=_A1X_ERAS, only_if_missing=True,
         label_note="Implementing entity"),
]

_ANY_LABEL_EXTRA = re.compile(
    r"Date of \w{0,8}\s?submission|date of AE|approval from accredited entity"
    r"|Implementing\s+[Ee]ntit", re.I)


# every label word at once: one cheap pre-filter search per page, so pages that
# cannot carry a template field never run the full rule set
_ANY_LABEL = re.compile(
    r"Total GCF funding|GCF funding requ|Requested GCF|Received GCF|GCF fin\w+ to recipient"
    r"|Total (?:project )?financing|Total co-financing|Total funding requested|Total project cost"
    r"|Financial instrument|Implementation period|lifespan|ES[SG] category"
    r"|Expected (?:mitigation|adaptation)|Accredited\s+[Ee]ntity|Countr(?:y|ies)"
    r"|programme?\s*title|Project title|Executing [Ee]ntity|National designated authority"
    r"|funding proposal (?:titled|entitled)|Project size", re.I)

# the same pre-filter for the fallback pass: its labels are variants the strict
# one does not carry, so a page that only prints 'Total requested' or
# 'Program title' must still be scanned.
_ANY_LABEL_FB = re.compile(
    r"Total requested|Requested GCF amount|Required GCF amount|Disaggregated GCF amount"
    r"|GCF funds requested|GCF fin\w+ to recipient|Total GCF funding"
    r"|Total (?:project|programme|program) (?:financ\w+|cost)|Total financing"
    r"|Program\w*\s*title|Project\s*[/(]|Countr(?:y|ies)\s*[/&]|Country location"
    r"|Accredit\w+\s+[Ee]ntit|funding proposal", re.I)

# a value window ends at the next labelled field. The bullet/pipe prefix matters:
# '- **A.8.1 Total GCF funding requested:**' is the next field even though it is
# printed as a list item, and without it an A.7 window swallows A.8's components.
# Only (a)-(d) head template sub-sections: '(i) Grants: $20,546,756' is an
# instrument ROW of the current field and must stay inside the window.
_STOP = re.compile(r"[ \t]*[-–*>|]{0,3}[ \t]*(?:#{1,6}[ \t]"
                   r"|\**[ \t]*[A-H][.\s]\s?\d{1,2}[.\s:)]"
                   r"|\*\*\([a-d]\)|\(\s?[a-z]\s?\)\s*(?:Total|GCF|Co-financ))")

_INSTRUMENTS = (r"senior loans?|subordinated loans?|reimbursable grants?|results?[- ]based payments?"
                r"|guarantees?|equity|grants?|loans?")
_INSTR_RE = re.compile(r"(?:^|\n|\|)[ \t]*[#>*\-–\s]*\**[ \t]*(?P<name>" + _INSTRUMENTS + r")\**"
                       r"[ \t]*[:|\-–]?(?P<tail>[^\n]{0,80})", re.I)

_DIRECT_RE = re.compile(r"(?P<num>" + _NUM + r")[ \t]{0,3}(?P<kind>direct|indirect)\b"
                        r"|(?<!in)(?P<kind2>\bdirect|\bindirect)\b[^\n\d%]{0,24}(?P<num2>" + _NUM + r")",
                        re.I)


# fallback-only window terminator. In the v1 template only the CO-FINANCING and
# the GCF-to-AE tables carry a 'Name of Institution' column and a 'Lead
# financing institution' line: a (b)-block window that reaches one of those has
# left the GCF request behind, and the first amount past it belongs to a
# co-financier ('Total requested (a)+(b)+(c)... | Senior Loans | 99,596,000 |
# ADB' is the ADB loan, not the request). Strict rules never see this.
_STOP_FB = re.compile(r"[^\n]{0,200}?\b(?:Name of Institutions?|Lead financing institution)\b",
                      re.I)


def _window(body: str, start: int, max_lines: int, max_chars: int,
            stop_extra: Optional[re.Pattern] = None) -> str:
    seg = body[start:start + max_chars * 3]
    lines = seg.split("\n")
    out = [lines[0]]
    for ln in lines[1:max_lines]:
        if _STOP.match(ln) or (stop_extra and stop_extra.match(ln)):
            break
        out.append(ln)
    return "\n".join(out)[:max_chars]


def _norm_sec(sec: Optional[str], default: str) -> str:
    """The section id AS PRINTED on the page, or 'rule:<default>' when the page
    printed none. A bare default would claim a section number the page does not
    have — page 55 of a 100-page package is not 'B.2(b)' just because the label
    'GCF financing to recipient' appeared there."""
    if not sec:
        return "rule:" + default
    return re.sub(r"\s+", "", sec).rstrip(".").upper()


def _sec_key(c: dict) -> str:
    """Section identity for comparisons: 'A.8' and 'rule:A.8' are one source."""
    return (c.get("section") or "").split(" ")[0].replace("rule:", "")


def _cand(rule: Rule, page: int, sec: Optional[str], parsed: dict, suffix: str = "") -> dict:
    # a rule that reads a field out of a DIFFERENTLY LABELLED template slot keeps
    # the printed label in the section, so the provenance says which words the
    # page actually used ('A.1.5 Implementing entity')
    suffix = suffix or (" " + rule.label_note if rule.label_note else "")
    return {"raw": parsed["raw"], "value": parsed.get("value"),
            "currency": parsed.get("currency"), "unit": parsed.get("unit"),
            "page": page, "section": _norm_sec(sec, rule.section) + suffix,
            "status": "supporting", "_rank": rule.rank, "_grain": parsed.get("_grain", 0.0),
            "_bare": parsed.get("_bare"), "_bare_unit": parsed.get("_bare_unit"),
            # a template TOTAL line may be elected canonical and may contradict
            # the canonical one; a component/instrument line may do neither
            "_canon": True, "_conflict": rule.kind in ("amount", "count", "period",
                                                       "beneficiaries", "ess", "text")}


def extract_candidates(pages: List[Tuple[int, str]],
                       fallback: bool = True, era: Optional[str] = None,
                       extra_rules: bool = True) -> Dict[str, List[dict]]:
    """Deterministic pass over one document's pages -> {field: [candidate, ...]}.

    Three passes. The strict pass is RULES, unchanged. The fallback pass is
    FALLBACK_RULES and is restricted to the fields the strict pass left with no
    candidate at all — the template VARIANTS, read with the relaxed number
    guards. A document whose fields the strict pass read is therefore
    bit-for-bit unaffected by anything in the second pass.

    The third pass is EXTRA_RULES (the ratified data decisions of 2026-08-26):
    its own pre-filter, its own labels, and it may only write a field it is
    named for — two new date fields, plus the era-gated 'Implementing entity'
    reading of accredited_entity. `era` gates that last one; passing None means
    'no era known', and an era-gated rule then does not fire.
    """
    out = _scan(pages, RULES, _ANY_LABEL)
    if fallback:
        missing = {r.field for r in FALLBACK_RULES} - set(out)
        if missing:
            got = _scan(pages, [r for r in FALLBACK_RULES if r.field in missing],
                        _ANY_LABEL_FB, loose=True)
            for f, cs in got.items():
                if f in missing and cs:         # never touch a field already read
                    out[f] = cs
    if not extra_rules:
        return out
    rules = [r for r in EXTRA_RULES
             if (not r.era or (era in r.era))
             and not (r.only_if_missing and r.field in out)]
    if not rules:
        return out
    got = _scan(pages, rules, _ANY_LABEL_EXTRA)
    for f, cs in got.items():
        if f not in out and cs:                 # never touch a field already read
            out[f] = cs
    return out


def _scan(pages: List[Tuple[int, str]], rules: List["Rule"], prefilter: re.Pattern,
          loose: bool = False) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for page, body in pages:
        if not prefilter.search(body):
            continue
        for rule in rules:
            if rule.max_page and page > rule.max_page:
                continue
            for m in rule.re.finditer(body):
                win = _window(body, m.end(), *rule.window,
                              stop_extra=(_STOP_FB if loose else
                                          _STOP_DATE if rule.kind == "date" else None))
                if loose:
                    win = _EMPH.sub("", win)
                sec = m.groupdict().get("sec")
                if rule.kind == "amount":
                    for p in read_amounts(win, rule.amounts, rule.skip_lines, loose):
                        out.setdefault(rule.field, []).append(_cand(rule, page, sec, p))
                elif rule.kind == "count":
                    p = read_count(win)
                    if p:
                        out.setdefault(rule.field, []).append(_cand(rule, page, sec, p))
                elif rule.kind == "period":
                    p = read_period(win)
                    if p:
                        out.setdefault(rule.field, []).append(_cand(rule, page, sec, p))
                elif rule.kind == "ess":
                    p = read_ess(win)
                    if p:
                        c = _cand(rule, page, sec, p)
                        c["raw"] = p["_text"]
                        out.setdefault(rule.field, []).append(c)
                elif rule.kind == "date":
                    p = read_date(win, rule.skip_lines)
                    if p:
                        out.setdefault(rule.field, []).append(
                            _cand(rule, page, sec, p, suffix=_label_text(m, sec)))
                elif rule.kind == "text":
                    p = read_text(win)
                    if p:
                        out.setdefault(rule.field, []).append(_cand(rule, page, sec, p))
                elif rule.kind == "beneficiaries":
                    _read_beneficiaries(out, rule, page, sec, win)
                elif rule.kind == "instruments":
                    _read_instruments(out, rule, page, sec, win)
    return out


def _label_text(m: re.Match, sec: Optional[str]) -> str:
    """The label AS PRINTED, for the section of a candidate whose field name is
    not the label ('rule:A.1.9 Date of current submission'). Two submission
    dates are printed under two different labels; without this the store would
    hold two dates and no way to say which is which."""
    lab = m.group(0)
    if sec:
        lab = lab.replace(sec, " ", 1)
    lab = re.sub(r"\([^)]*\)", " ", lab)             # '(if applicable)' is not the label
    lab = re.sub(r"[*#>|\-–:.\s]+", " ", lab).strip()
    return " " + lab if lab and len(lab) <= 48 else ""


def _read_beneficiaries(out, rule, page, sec, win):
    seen = set()
    for m in _DIRECT_RE.finditer(win):
        kind = (m.group("kind") or m.group("kind2") or "").lower().lstrip()
        num = m.group("num") or m.group("num2")
        if not num or kind in seen:
            continue
        before = win[max(0, m.start() - 24):m.start()]
        end = m.end("num") if m.group("num") else m.end("num2")
        # '13.1% (direct benefits)' is a share of a population, not a headcount
        if _NOISE_BEFORE.search(before) or "%" in m.group(0) or win[end:end + 1] == "%":
            continue
        val = to_number(num)
        if val is None or val == 0:
            continue
        seen.add(kind)
        field = "beneficiaries_direct" if kind == "direct" else "beneficiaries_indirect"
        p = {"raw": m.group(0).strip(" \t|-*:"), "value": val, "currency": None,
             "unit": None, "_grain": granularity(num, 1.0)}
        c = _cand(rule, page, sec, p)
        c["section"] = _norm_sec(sec, rule.section)
        out.setdefault(field, []).append(c)
    # a bare adaptation-outcome number (no direct/indirect wording)
    if not seen:
        p = read_count(win)
        if p:
            out.setdefault(rule.field, []).append(_cand(rule, page, sec, p))


def _read_instruments(out, rule, page, sec, win):
    amounts = []
    for m in _INSTR_RE.finditer(win):
        name = re.sub(r"\s+", " ", m.group("name")).title()
        p = read_amount(m.group("tail"))
        if not p:
            continue
        c = _cand(rule, page, sec, p, suffix=" " + name)
        out.setdefault("financial_instruments", []).append(c)
        out.setdefault("instruments", []).append(
            {"raw": name, "value": None, "currency": None, "unit": None, "page": page,
             "section": _norm_sec(sec, rule.section) + " " + name, "status": "supporting",
             "_rank": rule.rank, "_grain": 0.0, "_canon": True, "_conflict": False})
        amounts.append(c)
    # a single instrument carries the whole GCF request: expose it as a
    # gcf_funding_requested candidate so it can be compared with A.8 (never
    # canonical). Several instruments would need a sum -> not invented.
    if len(amounts) == 1:
        c = dict(amounts[0])
        c["_rank"], c["_canon"], c["_conflict"] = 9, False, True
        out.setdefault("gcf_funding_requested", []).append(c)


# ---------------------------------------------------------------------------
# status assignment
# ---------------------------------------------------------------------------

NUMERIC_FIELDS = {"total_financing", "gcf_funding_requested", "co_financing",
                  "financial_instruments", "mitigation_outcome", "adaptation_outcome",
                  "beneficiaries_direct", "beneficiaries_indirect",
                  "implementation_period", "lifespan"}
# one candidate per instrument: a grant of 20m next to a loan of 30m is the
# breakdown, not a contradiction
NO_CONFLICT_FIELDS = {"financial_instruments", "instruments"}


def _compatible(a: dict, b: dict) -> bool:
    """Same currency (or unknown on one side) and same normalization unit class."""
    ca, cb = a.get("currency"), b.get("currency")
    if ca and cb and ca != cb:
        return False
    ua = a.get("unit") in ("years", "months")
    ub = b.get("unit") in ("years", "months")
    if ua != ub:
        return False
    if ua and ub and a.get("unit") != b.get("unit"):
        return False              # 5 years vs 60 months: not compared, not a conflict
    return True


def _comparable_pair(a: dict, b: dict) -> Optional[Tuple[dict, dict]]:
    """The two readings to compare, or None when the pair cannot be compared.

    Normally the normalized values. When BOTH sides had their printed unit word
    overridden (value null), their mantissas are still comparable as long as the
    same unit word was printed on both — '28,654 million' vs '26,654 million'
    disagree whatever the true scale is.
    """
    if a["value"] is not None and b["value"] is not None:
        return a, b
    if (a["value"] is None and b["value"] is None and a.get("_bare") is not None
            and b.get("_bare") is not None and a.get("_bare_unit") == b.get("_bare_unit")):
        return ({"value": a["_bare"], "_grain": a.get("_grain", 0.0)},
                {"value": b["_bare"], "_grain": b.get("_grain", 0.0)})
    return None


def _same_value(a: dict, b: dict) -> bool:
    """Equal within the precision the two raw strings actually print.

    '46.10 million' can only be as exact as 10 000, so 46,104,231 supports it;
    '40,751,254' vs '40,511,264' print full dollars and differ by 240 000 —
    that is a real conflict. Documents also truncate instead of rounding
    ('68,746,295' printed as '68.74 million'), so a full grain is allowed on
    each side rather than half.
    """
    tol = a.get("_grain", 0.0) + b.get("_grain", 0.0) + 1e-6
    return abs(a["value"] - b["value"]) <= max(tol, 0.5)


def _digits(c: dict) -> str:
    return re.sub(r"\D", "", c["raw"].split("\n")[0])


def _truncated_twin(c: dict, others: List[dict]) -> bool:
    """'60' next to '601,550' in the same section of the same page is the page
    break cutting the number in half, not a second reading of the field."""
    d = _digits(c)
    if not d:
        return False
    return any(o is not c and o["page"] == c["page"] and o["section"] == c["section"]
               and len(_digits(o)) > len(d) and _digits(o).startswith(d) for o in others)


# the document's OWN template heading for a field: when the page prints this,
# the field's canonical value has to come from a printed section, never from a
# rule that happened to match prose 200 pages later
_TEMPLATE_HEADING = {
    "total_financing": re.compile(
        r"(?m)^\W{0,8}A\.?\s?7[.\s):]\s*\**\s*Total\s+financing", re.I),
    "gcf_funding_requested": re.compile(
        r"(?m)^\W{0,8}A\.?\s?8[.\s):]\s*\**\s*(?:Total\s+)?GCF\s+funding", re.I),
}


def _printed(c: dict) -> bool:
    """The page itself printed this section id (not the rule's fallback name)."""
    return not str(c.get("section", "")).startswith("rule:")


def finalize(field: str, cands: List[dict], template_heading: bool = False) -> List[dict]:
    """Dedupe, elect the canonical candidate, mark conflicts.

    `template_heading` says the document prints this field's A.x heading. Then a
    prose match elsewhere may NOT be canonical: either a printed section supplies
    the value or the field has no canonical at all — 'the template section is
    empty' is a truthful answer, 'USD 32,500 from a C.1(c) row relabelled A.7'
    is not.
    """
    seen, uniq = set(), []
    for c in sorted(cands, key=lambda c: (c["page"], c["_rank"], c["section"])):
        # the same printed value on the same page is ONE fact even when two
        # labels lead to it; the best-ranked rule keeps its section id
        key = (c["page"], re.sub(r"\s+", " ", c["raw"].strip().lower()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    uniq = [c for c in uniq if not _truncated_twin(c, uniq)]
    numeric = field in NUMERIC_FIELDS
    # best rule first; within a rule the page's own section id beats a rule name
    eligible = [c for c in uniq if c["_canon"]]
    if template_heading and not any(_printed(c) for c in eligible):
        eligible = []
    canon = (min(eligible, key=lambda c: (c["_rank"], not _printed(c), c["page"]))
             if eligible else None)
    for c in uniq:
        c["status"] = "supporting"
    if canon:
        canon["status"] = "canonical"
        if numeric and field not in NO_CONFLICT_FIELDS:
            for c in uniq:
                if c is canon or not c["_conflict"] or not _compatible(c, canon):
                    continue
                pair = _comparable_pair(c, canon)
                if pair is None:
                    continue
                cv, kv = pair
                # a figure far below the canonical total is a component, a
                # tranche or a portfolio target, not a competing reading of the
                # same field — unless it is printed in the same section
                same_source = _sec_key(c) == _sec_key(canon) and c["page"] != canon["page"]
                if cv["value"] < 0.5 * kv["value"] and not same_source:
                    continue
                if not _same_value(cv, kv):
                    c["status"] = "conflicting"
    for c in uniq:
        for k in ("_rank", "_grain", "_canon", "_conflict", "_bare", "_bare_unit"):
            c.pop(k, None)
    return uniq


# ---------------------------------------------------------------------------
# eras / coverage
# ---------------------------------------------------------------------------

_ERA_MODERN = re.compile(r"A[.\s]?\s?7[.\s)]{0,3}\s*\**\s*Total (?:project )?financing"
                         r"|A[.\s]?\s?8[.\s)]{0,3}\s*\**\s*Total GCF funding", re.I)
_ERA_OLD = re.compile(r"A[.\s]?1[.\s]\s?[15][.\s)]{0,3}\s*\**\s*(?:Project\s*/|Accredited)", re.I)

# --- template families the two rules above do not name -----------------------
# The REDD+ results-based-payment pilot (Decision B.18/07) is its OWN funding
# proposal template: a cover of Programme Title / Country / Results period /
# NDA / REDD-plus entity / Accredited Entity, then sections A-E about carbon
# results. It has no A.7 and no A.8, so it can never look 'modern', and calling
# it 'unrecognized' hid the fact that its financing fields do not exist.
_ERA_RBP = re.compile(
    r"(?m)^\W{0,10}REDD[\s\-+]*(?:plus)?[\s\-]+results[\s\-]?based[\s\-]?payments?\b"
    # ... or the RBP cover's own field, which no other template has, for the
    # extractions that lost the heading (b24-02-add06)
    r"|REDD[\s\-+]*(?:plus)?\s+entity\s*[/\s]\s*(?:focal|local|fiscal)\s*point", re.I)
# The v2/v3 cover, identified by its A.9-A.21 labels rather than by A.7/A.8:
# an extraction that drops or garbles the two financing headings (FP226's page
# 5, FP268's 'A.B.') is still that template. Three distinct labels, so a stray
# 'A.14' in prose cannot carry the vote.
# Counted by LABEL, not by number: FP214's extraction renumbered the whole
# block ('A.0. Financial instruments', 'A.1. Implementation period', 'A.2.
# Total lifespan') and FP226's dropped A.7/A.8 altogether, yet both are plainly
# this template. Three distinct labels, so one stray heading cannot decide it.
_MODERN_TAG = re.compile(
    r"(?m)^\W{0,10}A[.\s]?\s?\d{1,2}[.\s):]?\s*\**\s*"
    r"(?P<t>Project size|Financial instruments?|Financial benefits"
    r"|Implementation period|Total lifespan|ESS? category"
    r"|Environmental and social(?: risk)? category|Executing [Ee]ntity information"
    r"|Expected date of AE|Estimated date of AE|Has this FP|Is this FP"
    r"|Complementarity and coherence|Executive summary|Result areas?)", re.I)
# ... and the v1 cover by its A.1.x / A3.x identity block, whatever wording
# follows the number ('A.1.2 Project or programme title', 'A3.5a Accredited
# entity', 'A.5. Accredited entity').
_OLD_TAG = re.compile(
    r"(?m)^\W{0,10}A\.?\s?\d{1,2}(?:\.\s?\d{1,2}[a-z]?)?[.\s):]\s*\**\s*"
    r"(?P<t>Project\W{0,4}(?:or\s+)?(?:programm?e?)?\W{0,2}title|Project or programme"
    r"|Accredited entity|Approved entity|Implementing entity|Executing entity"
    r"|Countr(?:y|ies)|National designated authority|Project size category"
    r"|Access modality|Mitigation\s*/\s*adaptation focus)", re.I)
# the B.11-era 'funding proposal SUMMARY': a free-form summary the Board
# considered before the A/B template existed. Not a defect and not out of
# scope — a document class with no template block to parse.
_ERA_SUMMARY = re.compile(
    r"(?m)^#{1,4}\s*Funding Proposal Summary\b"
    r"|A funding proposal summary (?:titled|entitled)", re.I)
# Board notices that are not proposals at all: withdrawals, corrigenda, status
# papers. Consulted last, so a proposal that merely mentions a withdrawal in
# prose is already classified by then.
_ERA_NOTICE = re.compile(
    r"has been withdrawn (?:from|by|at)|been withdrawn from consideration"
    r"|(?m:^#{1,3}\s*Status of approved funding proposals)"
    r"|(?m:^#{1,3}[^\n]{0,80}Corrigendum\s*$)", re.I)

CORE_FIELDS = ["title", "countries", "accredited_entity",
               "total_financing", "gcf_funding_requested"]


def era_of(text: str) -> str:
    """The template family the document belongs to.

    The first two branches are the shipped ones and are consulted first, so no
    document they already name can be renamed here. Everything after them only
    ever splits what used to be called 'unrecognized template'.
    """
    if _ERA_MODERN.search(text):
        return "A5-A14 block (FP template v2/v3)"
    if _ERA_OLD.search(text):
        return "A.1.x block (FP template v1)"
    if _ERA_RBP.search(text[:120000]):
        return "REDD+ RBP block (RBP pilot template v1.0)"
    if len({m.group("t").lower() for m in _MODERN_TAG.finditer(text)}) >= 3:
        return "A5-A14 block (FP template v2/v3, variant numbering)"
    if len({m.group("t").lower() for m in _OLD_TAG.finditer(text)}) >= 3:
        return "A.1.x block (FP template v1, variant numbering)"
    if _ERA_NOTICE.search(text):
        return "board notice (not a proposal template)"
    if _ERA_SUMMARY.search(text[:20000]):
        return "funding proposal summary (pre-template, B.11 era)"
    return "unrecognized template"


# ---------------------------------------------------------------------------
# ratified data decisions: corrections and confirmed absences
# ---------------------------------------------------------------------------
# Two data files, both ratified by the owner on 2026-08-26, both consumed here
# so that data/registry_v2.json is never hand-edited:
#
#   data/registry_corrections.json  62 ratified rows: the 58 the adjudication
#       proved WRONG, plus the four RIDERS carried inside CONFIRMED rows
#       ("keep, but also ..."), ratified in their own session. Each
#       names the document, the field, the layer, the value AS SHIPPED, the
#       corrected value with the page that prints it, and the quoted print.
#       An `add-candidate` row adds a print the store was not carrying rather
#       than overwriting one, and carries no wrong value. A `drop-candidate`
#       row removes a print the PDF does not contain, and carries no corrected
#       value: its `dropped` block records the independent extractions that
#       were searched and found nothing. A `reextracted` block on any row says
#       'this row's page went for VLM re-extraction': if the fresh page reads
#       the ratified figure the row is recorded as resolved by the
#       re-extraction instead of shouted about as unapplied — and if it does
#       NOT, the row is still reported unapplied, loudly.
#       Ratified later, in the serving-wave session of the same day: ten more
#       money-fact rows (four correct-to, two value-fix, four drop-candidate)
#       and the two RBP add-candidate rows that stop an absence and a top-level
#       print from both being live at once.
#   data/registry_absences.json     51 (document, field) pairs read and found
#       ABSENT, plus the corpus-level finding that no document prints its own
#       GCF Board approval date.
#
# The schema addition is additive and lives in `meta`: no existing key moves, no
# candidate list changes shape, and a document with neither a correction nor an
# absence serializes byte-identically to before.
#
#   documents[doc].meta.corrections       what was changed here and why
#   documents[doc].meta.confirmed_absence {field: evidence} — absence-as-fact
#   documents[doc].meta.mapped_labels     a field read from a differently
#                                         labelled template slot
#
# A corrected candidate carries `corrected: true` and `corrected_from`. On such
# a candidate `raw` is the RATIFIED figure rather than a literal page string —
# the page's own print is quoted in meta.corrections[].quote. Everywhere else
# `raw` keeps its usual contract.
CORRECTIONS_FILE = config.DATA_DIR / "registry_corrections.json"
ABSENCES_FILE = config.DATA_DIR / "registry_absences.json"
RATIFIED = "owner, 2026-08-26"
# the same owner, later the same day: the serving-wave rows (ten money-fact
# corrections, the two RBP add-candidates) and the cross-extractor arm
RATIFIED_SERVING = "owner, 2026-08-26 (serving-wave session)"

# the two grounds on which a ratified row may DELETE a print outright. Both have
# to be PROVED in the row, because a drop is the one action that leaves the
# store holding less than the extraction found and offers nothing in its place.
DROP_GROUNDS = {
    # an independent text extraction of the whole PDF prints nothing like it:
    # the candidate is not a reading of the document, it is a VLM invention
    "printed-nowhere": "the figure is printed on no page of the PDF",
    # the figure IS printed — under a DIFFERENT field's label, and the field it
    # belongs to already holds it, so dropping it loses nothing
    "label-bleed": "the figure belongs to another field, which already holds it",
}

# the candidate status a fact-layer row asks for, used when a ratified figure has
# to be installed rather than written onto a candidate the fresh parse still has
_LAYER_STATUS = {"fact-canonical": "canonical", "fact-supporting": "supporting",
                 "fact-conflicting": "conflicting"}

# a top-level (registry.json) field and the schema-2 fact it is the flat view of
_TOP_TO_FACT = {"gcf_financing": "gcf_funding_requested",
                "total_financing": "total_financing",
                "title": "title", "accredited_entity": "accredited_entity",
                "countries": "countries"}


class Decisions:
    """The ratified corrections and absences, plus the report of what happened.

    Nothing here fails a build: a correction whose target has moved (the corpus
    is being re-extracted underneath us) is NOT applied and IS shouted about, so
    a rebuild says out loud that a ratified decision did not land.
    """

    def __init__(self, corrections=None, absences=None, meta=None):
        self.by_doc: Dict[str, List[dict]] = {}
        for e in corrections or []:
            self.by_doc.setdefault(e["doc_id"], []).append(e)
        self.absent: Dict[str, List[dict]] = {}
        for a in absences or []:
            self.absent.setdefault(a["doc_id"], []).append(a)
        self.meta = meta or {}
        self.applied: List[str] = []
        self.deferred: Dict[str, List[dict]] = {}
        self.unapplied: List[dict] = []
        self.carried: List[dict] = []
        self.alarms: List[str] = []
        self.absences_published: set = set()
        self.absences_skipped: List[dict] = []

    @classmethod
    def load(cls, corrections_path: Path = None, absences_path: Path = None) -> "Decisions":
        cp = Path(corrections_path or CORRECTIONS_FILE)
        ap = Path(absences_path or ABSENCES_FILE)
        corr = json.loads(cp.read_text(encoding="utf-8")) if cp.exists() else {}
        absc = json.loads(ap.read_text(encoding="utf-8")) if ap.exists() else {}
        return cls(corr.get("corrections"), absc.get("absences"),
                   meta={"corrections": {"file": str(cp.name),
                                         "present": cp.exists(),
                                         "count": len(corr.get("corrections") or []),
                                         "ratified": corr.get("ratified")},
                         "absences": {"file": str(ap.name),
                                      "present": ap.exists(),
                                      "count": len(absc.get("absences") or []),
                                      "ratified": absc.get("ratified"),
                                      "corpus_level": absc.get("corpus_level") or []}})

    # -- reporting ---------------------------------------------------------
    def alarm(self, msg: str) -> None:
        self.alarms.append(msg)

    def miss(self, entry: dict, why: str) -> None:
        self.unapplied.append({"id": entry["id"], "doc_id": entry["doc_id"],
                               "field": entry["field"], "layer": entry["layer"],
                               "why": why})
        self.alarm(f"NOT APPLIED {entry['id']} {entry['doc_id']} {entry['field']} "
                   f"[{entry['layer']}]: {why}")

    def carry(self, entry: dict, why: str) -> None:
        """A third outcome, and it needs its own name.

        The row did not land on the candidate it names — so it is not APPLIED —
        but its ratified figure is in the store all the same, so it is not
        UNAPPLIED either. Calling it one of the two would either hide a
        disagreement between the owner and the parser or claim a figure was
        lost that was not. It is alarmed like a miss, because a build where the
        fresh parse and a ratified row disagree is a build somebody has to look
        at.
        """
        self.carried.append({"id": entry["id"], "doc_id": entry["doc_id"],
                             "field": entry["field"], "layer": entry["layer"],
                             "why": why})
        self.alarm(f"CARRIED FORWARD {entry['id']} {entry['doc_id']} "
                   f"{entry['field']} [{entry['layer']}]: {why}")

    def for_doc(self, doc_id: str, layers) -> List[dict]:
        return [e for e in self.by_doc.get(doc_id, []) if e["layer"] in layers]


def _record(entry: dict, before, after) -> dict:
    """The meta row: what moved, what it came from, and the print that decides it."""
    return {"id": entry["id"], "field": entry["field"], "layer": entry["layer"],
            "action": entry["action"], "from": before, "to": after,
            "page_of_quote": (entry.get("corrected") or entry.get("add")
                              or entry.get("dropped") or {}).get("page"),
            "quote": (entry.get("corrected") or entry.get("add")
                      or entry.get("dropped") or {}).get("quote"),
            "adjudication_note": entry.get("adjudication_note"),
            "row_ref": entry.get("row_ref"), "ratified": entry.get("ratified", RATIFIED),
            **({"pending_reextraction": True} if entry.get("pending_reextraction") else {}),
            **({"riders_not_applied": entry["riders_not_applied"]}
               if entry.get("riders_not_applied") else {})}


def _shape(c: dict) -> dict:
    return {k: c.get(k) for k in ("raw", "value", "currency", "unit", "page",
                                  "section", "status")}


def _pick(cands: List[dict], raw, page=None, status=None) -> Optional[dict]:
    hits = [c for c in cands if c["raw"] == raw]
    if status:
        hits = [c for c in hits if c["status"] == status] or hits
    if page is not None and len(hits) > 1:
        hits = [c for c in hits if c["page"] == page] or hits
    return hits[0] if len(hits) == 1 else None


def _corrected_candidate(target: dict, entry: dict) -> dict:
    """Overwrite one candidate in place, keeping where it came from. Returns the
    candidate as it was, which is also what `corrected_from` records."""
    to = entry["corrected"]
    before = _shape(target)
    target["raw"] = to["raw"]
    target["value"] = to["value"]
    target["currency"] = to["currency"]
    target["unit"] = to["unit"]
    target["page"] = to["page"]
    # a corrected candidate does not claim a section the page may not print: it
    # says 'corrected' and points at the quote, unless the correction kept the
    # original print (a scale fix), which keeps its own section id
    target["section"] = to.get("section") or "corrected"
    target["corrected"] = True
    target["corrected_from"] = before
    return before


def _grain_of(c: dict) -> float:
    """The precision the candidate's own print implies, recomputed from `raw`.

    finalize() carries it in a private key and drops it before publishing; a
    correction arrives after that, so the comparison it re-runs recomputes the
    number from the same reader that produced it.
    """
    p = read_amount(c.get("raw") or "")
    return float(p.get("_grain", 0.0)) if p else 0.0


def remark_conflicts(field: str, cands: List[dict]) -> None:
    """Re-run the status half of finalize() for a field a correction moved.

    A candidate was marked 'conflicting' because it disagreed with the value the
    adjudication has since refuted. Left alone it warns the reader of a conflict
    with a figure the store no longer holds — FP169's page-46 print, which the
    correction ADOPTED, would have been published as disagreeing with itself.

    Conservative by construction: when the corrected field has no canonical, or
    its canonical carries no parsed value, nothing is re-marked — the build does
    not invent a comparison it cannot make.
    """
    if field not in NUMERIC_FIELDS or field in NO_CONFLICT_FIELDS:
        return
    canon = next((c for c in cands if c["status"] == "canonical"), None)
    if canon is None or canon.get("value") is None:
        return
    kg = _grain_of(canon)
    for c in cands:
        if c is canon:
            continue
        if c.get("value") is None or not _compatible(c, canon):
            c["status"] = "supporting"
            continue
        same_source = _sec_key(c) == _sec_key(canon) and c["page"] != canon["page"]
        if c["value"] < 0.5 * canon["value"] and not same_source:
            c["status"] = "supporting"          # a component, not a rival reading
            continue
        tol = max(kg + _grain_of(c) + 1e-6, 0.5)
        c["status"] = ("supporting" if abs(c["value"] - canon["value"]) <= tol
                       else "conflicting")


#: How close a fresh parse has to be before it may SUPERSEDE a ratified figure.
#: Not `_agree`'s 0.5%: that tolerance exists to let two prints of the same
#: reading ('40.15 million' / '40,150,000') recognise each other, and it is far
#: too wide to certify that a re-extracted page came back with the ratified
#: DIGITS. FP274 is the row that proved it — the cure left p.40's '40,751,254'
#: standing against a ratified 40,751,264, a single-digit misread the
#: cross-extractor arm independently flags `not-in-document`, and 10 is well
#: inside 0.5% of 40 million. A supersession claim is a claim about digits, so
#: it is tested on digits.
def _same_figure(a: Optional[float], b: Optional[float]) -> bool:
    return a is not None and b is not None and abs(a - b) < 0.5


def reextraction_settled(entry: dict, facts: Dict[str, List[dict]]) -> Optional[str]:
    """Did the page re-extraction make this ratified row unnecessary?

    A correction and a page re-extraction are two ways of fixing the same
    defect, and the adjudication pairs them on purpose ("correct-to 79,690,370;
    re-extract p5"). When the fresh page reads the figure correctly, the row's
    target no longer exists — and the build must be able to tell THAT apart
    from a target that went missing for some other reason.

    WHAT SETTLES A ROW IS THE OUTCOME, NOT A DECLARATION. This used to demand
    that the row had said in advance (`reextracted`) that its page was going
    for re-extraction. The corpus cure of 2026-08-26 re-extracted 95 pages
    across a hundred-odd ratified rows and annotated none of them, so every one
    of those rows arrived here undeclared, was reported NOT APPLIED, and had
    its ratified figure dropped on the floor — the cure leaving 25 fields worse
    off than the corrections alone had left them. A declaration was never the
    evidence; the fresh page reading the ratified figure is.

    So: settled iff the field's CANONICAL candidate — the one the store
    publishes — carries the ratified figure exactly (`_same_figure`). A field
    whose canonical reads something else, or which has no canonical at all, has
    not settled anything, and `carry_forward_correction` takes it from here.
    """
    field, action = entry["field"], entry["action"]
    if action == "drop-candidate":
        # the row exists to delete a print the PDF does not contain. The target
        # being gone IS the outcome it asked for — but 'gone' is only evidence
        # of a re-extraction when the row said one was coming, because a
        # drop row has no figure to check an outcome against.
        if not entry.get("reextracted"):
            return None
        return (f"the re-extraction of p.{entry['wrong'].get('page')} removed the "
                f"print this row was ratified to drop")
    want = (entry.get("corrected") or {}).get("value")
    if want is None:
        return None
    canon = _canon_of(facts, field)
    if canon is None or canon.get("value") is None:
        return None
    if not _same_figure(canon["value"], want):
        return None
    return (f"the re-extracted page reads the ratified figure: {field} is now "
            f"{canon['raw']!r} (p.{canon['page']}), value {canon['value']}")


def _superseded_link(entry: dict, dec: "Decisions") -> bool:
    """Does a LATER ratified row correct this row's own output?

    The corrections are a ledger, not a set: a doc/field the owner touched in
    two sessions carries two rows, and the second one's `wrong` block is the
    first one's `corrected` block. Only the last link is a live statement about
    the document.
    """
    mine = (entry.get("corrected") or {}).get("value")
    if mine is None:
        return False
    for other in dec.by_doc.get(entry["doc_id"], ()):
        if other is entry or other["field"] != entry["field"]:
            continue
        if other["layer"] != entry["layer"]:
            continue
        if _same_figure((other.get("wrong") or {}).get("value"), mine):
            return True
    return False


def carry_forward_correction(doc_id: str, entry: dict, facts: Dict[str, List[dict]],
                             dec: "Decisions") -> Optional[dict]:
    """The ratified figure survives a re-extraction that did not deliver it.

    Reached only when the row's named target is gone AND the fresh parse did
    not settle it. Two shapes, one rule — a ratified correction is superseded
    only by a parse that actually YIELDS the ratified figure:

    * THE FIGURE IS THERE BUT UNELECTED. Some candidate carries it and the
      store publishes another. Nothing needs inventing: that candidate is
      promoted, and the demotion is said out loud.
    * THE FIGURE IS NOT THERE AT ALL. Either the field came back empty (the
      parser's field-mapping gap — 106_gcf-b30-02-add01's cured p.5 plainly
      prints '- [x] Grant: 16,591,556' and `financial_instruments` still came
      back with nothing) or the fresh parse elected a different reading. The
      ratified candidate is installed from the row itself, carrying
      `carried_forward` so no reader mistakes it for a page print the parser
      found, and the reading it displaced is named in the alarm.

    The row's `wrong` block is NEVER applied to a candidate it does not name —
    that safety property is untouched. What changes is that a ratified figure
    is no longer silently lost when its target moves.
    """
    if entry["action"] not in ("correct-to", "value-fix"):
        return None
    to = entry.get("corrected") or {}
    want = to.get("value")
    if want is None:
        return None
    field = entry["field"]
    cands = facts.setdefault(field, [])
    incumbent = _canon_of(facts, field)

    held = next((c for c in cands if _same_figure(c.get("value"), want)), None)
    if held is not None:
        if held is incumbent:                    # pragma: no cover - settled above
            return None
        was = _shape(held)
        if incumbent is not None:
            incumbent["status"] = "supporting"
        held["status"] = "canonical"
        dec.carry(entry, f"the fresh parse holds the ratified figure "
                         f"({held['raw']!r} p.{held['page']}) but published "
                         f"{incumbent['raw']!r} (p.{incumbent['page']}) instead"
                         if incumbent is not None else
                         f"the fresh parse holds the ratified figure "
                         f"({held['raw']!r} p.{held['page']}) and elected no "
                         f"canonical for the field")
        return {**_record(entry, was, _shape(held)), "carried_forward": "promoted"}

    c = {"raw": to["raw"], "value": want, "currency": to.get("currency"),
         "unit": to.get("unit"), "page": to.get("page"),
         "section": to.get("section") or "corrected",
         "status": _LAYER_STATUS.get(entry["layer"], "supporting"),
         "corrected": True, "corrected_from": None, "carried_forward": True}
    displaced = None
    if c["status"] == "canonical" and incumbent is not None:
        displaced = _shape(incumbent)
        incumbent["status"] = "supporting"
    cands.append(c)
    dec.carry(entry, (f"the fresh parse reads {displaced['raw']!r} "
                      f"(p.{displaced['page']}) where the ratified figure is "
                      f"{to['raw']!r}" if displaced is not None else
                      f"the fresh parse yields nothing for {field}; the ratified "
                      f"figure {to['raw']!r} would have been lost"))
    return {**_record(entry, displaced, _shape(c)), "carried_forward":
            "displaced" if displaced is not None else "restored"}


def apply_fact_corrections(doc_id: str, facts: Dict[str, List[dict]],
                           dec: "Decisions", entries: Optional[List[dict]] = None,
                           defer: bool = False) -> List[dict]:
    """The fact-layer half of the corrections. Runs BEFORE coverage is computed,
    so core_found / core_missing / suspect describe the corrected document.

    `defer`: a correction whose target is not among the deterministic candidates
    is not an error yet — the verified llm-fallback candidates are merged one
    step later, and four of the 58 rows correct one of those. Deferred rows are
    retried there, and only then can they be reported as not applied.
    """
    records: List[dict] = []
    for entry in (entries if entries is not None
                  else dec.for_doc(doc_id, ("fact-canonical", "fact-supporting",
                                            "fact-conflicting"))):
        field, action = entry["field"], entry["action"]
        cands = facts.get(field) or []
        if action == "add-candidate":
            # a print the document holds and the store was not carrying. There is
            # no wrong value here: nothing is overwritten, one candidate appears,
            # and the conflict machinery treats it like any other print.
            add = entry["add"]
            if _pick(cands, add["raw"], add.get("page")) is not None:
                dec.miss(entry, f"the candidate it adds is already in this build "
                                f"({add['raw'][:40]!r} p.{add.get('page')})")
                continue
            c = {"raw": add["raw"], "value": add.get("value"),
                 "currency": add.get("currency"), "unit": add.get("unit"),
                 "page": add["page"], "section": add.get("section") or "added",
                 "status": add.get("status", "supporting"), "added": True}
            if add.get("derived"):
                # the value the printed operands IMPLY, never a figure the page
                # prints: the raw quotes the operands and names the sum as a sum
                c["derived"] = True
                c["derived_from"] = add["derived_from"]
            facts.setdefault(field, []).append(c)
            records.append({**_record(entry, None, _shape(c)),
                            "declared_status": c["status"]})
            continue
        target = _pick(cands, entry["wrong"]["raw"], entry["wrong"].get("page"),
                       entry["wrong"].get("status"))
        if target is None:
            settled = reextraction_settled(entry, facts)
            if settled:
                records.append({**_record(entry, None, None),
                                "resolved_by_reextraction": settled})
                continue
            if defer:
                dec.deferred.setdefault(doc_id, []).append(entry)
                continue
            if _superseded_link(entry, dec):
                # A LINK IN A CHAIN, NOT A LIVE ROW. Four doc/field pairs were
                # corrected twice: phase 3 moved 106_gcf-b30-02-add01's request
                # to 18,591,556, the cross-check round then read the PDF itself
                # and moved it again to 16,591,556 — its `wrong` block quotes
                # the earlier row's output verbatim, section 'corrected' and
                # all. While both targets existed the chain resolved itself in
                # order. Once a cured page reads the FINAL figure directly BOTH
                # targets are gone, and carrying the intermediate row forward
                # would reinstate a reading the owner has since superseded over
                # the one they ratified last. The later row speaks for this
                # doc/field; this one is history, and history is not an alarm.
                records.append({**_record(entry, None, None),
                                "superseded_by_a_later_row":
                                    f"a later ratified row corrects this row's own "
                                    f"figure ({(entry.get('corrected') or {}).get('raw')!r}); "
                                    f"it speaks for {field} instead"})
                continue
            carried = carry_forward_correction(doc_id, entry, facts, dec)
            if carried is not None:
                records.append(carried)
                continue
            dec.miss(entry, f"the candidate it corrects is not in this build "
                            f"({entry['wrong']['raw']!r} "
                            f"p.{entry['wrong'].get('page')}) — re-extraction may "
                            f"have moved it")
            continue
        before = _shape(target)
        if action in ("correct-to", "value-fix"):
            _corrected_candidate(target, entry)
            for d in entry.get("drop") or []:
                gone = _pick(cands, d["raw"], d.get("page"))
                if gone is None:
                    dec.miss(entry, f"the candidate it drops is not in this build "
                                    f"({d['raw']!r})")
                else:
                    cands.remove(gone)
                    records.append({"id": entry["id"], "field": field,
                                    "action": "drop", "from": _shape(gone), "to": None,
                                    "quote": entry.get("adjudication_note"),
                                    "ratified": entry.get("ratified", RATIFIED)})
        elif action == "promote":
            rise = _pick(cands, entry["promote"]["raw"], entry["promote"].get("page"))
            if rise is None:
                dec.miss(entry, f"the candidate it promotes is not in this build "
                                f"({entry['promote']['raw']!r})")
                continue
            rose = _shape(rise)
            rise["status"] = "canonical"
            rise["corrected"] = True
            rise["corrected_from"] = rose
            cands.remove(target)                     # the wrong print is dropped
            records.append(_record(entry, before, _shape(rise)))
            continue
        elif action == "reclassify":
            to_field = entry["to_field"]
            cands.remove(target)
            moved = dict(target)
            _corrected_candidate(moved, entry)
            moved["status"] = entry.get("to_status", "supporting")
            sitting = next((c for c in facts.get(to_field, [])
                            if c["status"] == "canonical"), None)
            if moved["status"] == "canonical" and sitting is not None:
                # the ratified print outranks an extracted one that reads the SAME
                # figure (here the same page-32 digits behind an unbindable scale
                # word); it never silently displaces a different reading
                if _digits(sitting) and _digits(sitting) == _digits(moved):
                    sitting["status"] = "supporting"
                    dec.alarm(f"{entry['id']} {doc_id}: {to_field} already carried "
                              f"{sitting['raw']!r} (p.{sitting['page']}) — same figure, "
                              f"so the ratified print takes canonical and that one "
                              f"becomes supporting")
                else:
                    moved["status"] = "supporting"
                    dec.alarm(f"{entry['id']} {doc_id}: {to_field} already has a "
                              f"canonical candidate reading {sitting['raw']!r} — the "
                              f"reclassified print was filed as supporting instead")
            moved["reclassified_from"] = field
            facts.setdefault(to_field, []).append(moved)
            records.append({**_record(entry, before, _shape(moved)),
                            "to_field": to_field})
            if not cands:
                facts.pop(field, None)
            continue
        elif action == "confirm-absence":
            cands.remove(target)
            if not cands:
                facts.pop(field, None)
            records.append(_record(entry, before, None))
            continue
        elif action == "drop-candidate":
            # a candidate proven printed NOWHERE in the PDF: an independent text
            # extraction of the same pages prints nothing like it, so it is not a
            # reading of the document at all and there is nothing to correct it
            # TO. It is removed and the search that proved it is recorded.
            #
            # Not the same thing as the `drop` rider on a correct-to row: that
            # one clears a candidate a NEW ratified figure supersedes. This is a
            # decision in its own right, and it is the only action that removes a
            # print without either replacing it or confirming the field absent —
            # so it may never be used on a candidate the document does print.
            dropped = entry.get("dropped") or {}
            ground = dropped.get("ground")
            if ground not in DROP_GROUNDS:
                dec.miss(entry, f"a drop needs a ratified ground and this row "
                                f"carries {ground!r}; known grounds are "
                                f"{sorted(DROP_GROUNDS)}")
                continue
            if ground == "label-bleed":
                # 'it belongs to another field' is only a reason to delete when
                # that other field DOES hold the print. Otherwise the drop would
                # lose the only copy the store has.
                holder = dropped.get("belongs_to")
                if not _pick(facts.get(holder) or [], dropped.get("belongs_to_raw"),
                             dropped.get("page")):
                    dec.miss(entry, f"label-bleed drop: {holder!r} does not hold "
                                    f"{dropped.get('belongs_to_raw')!r}, so the print "
                                    f"would be lost, not moved")
                    continue
            cands.remove(target)
            if not cands:
                facts.pop(field, None)
            records.append({**_record(entry, before, None),
                            "ground": ground, "why": DROP_GROUNDS[ground],
                            "searched": dropped.get("searched"),
                            "evidence": dropped.get("evidence")})
            continue
        elif action == "re-extract":
            # adjudicated WRONG with no defensible replacement: the print stays,
            # the claim does not. Nothing canonical is asserted for the field.
            if target["status"] == "canonical":
                target["status"] = "supporting"
            target["disputed"] = True
            target["dispute"] = entry.get("adjudication_note") or entry["action"]
            records.append(_record(entry, before, _shape(target)))
            continue
        else:                                        # pragma: no cover - guarded above
            dec.miss(entry, f"unknown action {action!r}")
            continue
        records.append(_record(entry, before, _shape(target)))
    for r in records:
        if not r.get("carried_forward"):
            dec.applied.append(r["id"])
        for f in (r["field"], r.get("to_field")):
            if f and f in facts:
                remark_conflicts(f, facts[f])
    # a ratified status the conflict rules then disagree with is a disagreement
    # between the owner and the parser, and it gets said out loud rather than
    # silently resolved either way
    for r in records:
        want = r.get("declared_status")
        if not want:
            continue
        now = _pick(facts.get(r["field"]) or [], r["to"]["raw"], r["to"]["page"])
        if now is not None and now["status"] != want:
            r["status_after_remark"] = now["status"]
            dec.alarm(f"{r['id']} {doc_id} {r['field']}: the row asks for status "
                      f"{want!r} and the conflict rules make it {now['status']!r} "
                      f"against the canonical figure")
    return records


def _agree(a: Optional[float], b: Optional[float]) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= max(1.0, 0.005 * abs(b))


def apply_top_level_corrections(doc_id: str, row: dict, dec: "Decisions") -> List[dict]:
    """The flat registry.json fields carried on the v2 row (title, entity,
    countries, total_financing, gcf_financing)."""
    records: List[dict] = []
    for entry in dec.for_doc(doc_id, ("top-level",)):
        field = entry["field"]
        before = row.get(field)
        if before != entry["wrong"]["raw"]:
            dec.miss(entry, f"the shipped top-level value has moved "
                            f"({before!r} != {entry['wrong']['raw']!r})")
            continue
        if entry["action"] == "confirm-absence":
            row[field] = None
            records.append(_record(entry, before, None))
            continue
        to = entry["corrected"]
        rec_extra = {}
        if entry.get("pending_reextraction"):
            fresh = _canon_of(row.get("facts") or {}, _TOP_TO_FACT.get(field, field))
            if fresh is not None and fresh.get("value") is not None:
                if _agree(fresh["value"], to["value"]):
                    # the re-extracted page agrees: publish the PAGE, not the table
                    row[field] = fresh["raw"]
                    records.append({**_record(entry, before, fresh["raw"]),
                                    "resolved_by": "re-extraction agrees with the "
                                                   "correction; the fresh page parse is "
                                                   "published",
                                    "fresh": _shape(fresh)})
                    dec.applied.append(entry["id"])
                    continue
                dec.alarm(
                    "*** PENDING RE-EXTRACTION DISAGREES *** "
                    f"{entry['id']} {doc_id} {field}: the ratified correction says "
                    f"{to['raw']!r} ({to['value']}) but the re-extracted page parses "
                    f"{fresh['raw']!r} ({fresh['value']}) at p.{fresh['page']}. The "
                    "correction was applied; an owner must settle which is right.")
                rec_extra = {"reextraction_disagreement": _shape(fresh)}
            else:
                rec_extra = {"reextraction": "no parsed canonical yet — correction applied"}
        row[field] = to["raw"]
        records.append({**_record(entry, before, to["raw"]), **rec_extra})
        dec.applied.append(entry["id"])
    return records


def absence_meta(doc_id: str, facts: Dict[str, List[dict]],
                 dec: "Decisions") -> Dict[str, dict]:
    """confirmed_absence per (document, field): the runtime may say 'this
    document does not print it', with the pages that were read to find out."""
    out: Dict[str, dict] = {}
    # recomputed from scratch for this document: the llm-fallback merge can give
    # a field its first print AFTER the first pass ran, and a published absence
    # has to be withdrawn when that happens
    dec.absences_published = {k for k in dec.absences_published if k[0] != doc_id}
    dec.absences_skipped = [x for x in dec.absences_skipped if x["doc_id"] != doc_id]
    for a in dec.absent.get(doc_id, []):
        field = a["field"]
        if a.get("status") == "superseded":
            if not any(x["doc_id"] == doc_id and x["field"] == field
                       for x in dec.absences_skipped):
                dec.absences_skipped.append({**a, "why": "superseded by a ratified "
                                                         "decision (see superseded_by)"})
            continue
        held = facts.get(field) or []
        if held:
            # absence-as-fact is a claim about the DOCUMENT: it may never be
            # published over a print the build is holding. A canonical one means
            # the ratified row and the parser flatly disagree and somebody has to
            # look; a supporting one (239's title, named in prose on p.1 but under
            # no template label) means the absence was about the template field
            # and the store has the print anyway.
            if _canon_of(facts, field) is not None:
                dec.alarm(f"ABSENCE CONTRADICTED {doc_id} {field}: a canonical "
                          f"candidate exists, so the ratified absence was NOT published")
                why = "a canonical candidate exists"
            else:
                why = (f"the build holds a non-canonical print of this field "
                       f"({held[0]['raw'][:60]!r}, p.{held[0]['page']}) — absence not "
                       f"published over a print")
            if not any(x["doc_id"] == doc_id and x["field"] == field
                       for x in dec.absences_skipped):
                dec.absences_skipped.append({**a, "why": why})
            continue
        out[field] = {"pages_checked": a.get("pages_checked"),
                      "evidence": a.get("evidence"),
                      "group": a.get("group"),
                      "row_ref": a.get("row_ref"),
                      "ratified": a.get("ratified", RATIFIED)}
        dec.absences_published.add((doc_id, field))
    return out


def mapped_label_meta(pages: List[Tuple[int, str]],
                      facts: Dict[str, List[dict]]) -> List[dict]:
    """A field read out of a differently labelled template slot, with the label
    as the page prints it (OWNER RATIFICATION 2026-08-26: A.1.5 'Implementing
    entity' is the accredited-entity slot in the earliest template)."""
    notes = {r.label_note for r in EXTRA_RULES if r.label_note}
    by_page = dict(pages)
    out = []
    for field, cands in facts.items():
        for c in cands:
            label = next((n for n in notes if str(c.get("section", "")).endswith(n)), None)
            if not label:
                continue
            quote = ""
            for line in by_page.get(c["page"], "").split("\n"):
                if label.lower() in line.lower():
                    quote = line.strip()
                    if c["raw"][:24] not in line:
                        quote += " ⏎ " + next(
                            (l.strip() for l in by_page.get(c["page"], "").split("\n")
                             if c["raw"][:24] in l), "")
                    break
            out.append({"field": field, "printed_label": label, "page": c["page"],
                        "section": c["section"], "value": c["raw"], "quote": quote,
                        "decision": "OWNER RATIFICATION 2026-08-26: the A.1.x-era "
                                    "'Implementing entity' slot IS the accredited-entity "
                                    "slot; the printed label is kept in the section",
                        "ratified": RATIFIED})
    return out


# ---------------------------------------------------------------------------
# the cross-extractor verification arm (OWNER RATIFICATION 2026-08-26,
# serving-wave session: adopted as a STANDING arm, not a one-off audit)
# ---------------------------------------------------------------------------
# Everything above this line reads ONE rendering of the corpus: the qwen2.5-vl-7b
# markdown. Phase 3 proved every registry raw is literally printed on the
# markdown page it cites — and that proof is silent about the markdown being
# wrong, because it checks the markdown against itself. The serving wave found
# eight pages where the markdown prints a figure the PDF does not.
#
# So: read the figure again out of a DIFFERENT extraction of the same PDF.
# data/extracted/pymupdf/ is produced by a text-layer extractor with no model in
# it, page-marked the same way, and already on disk — the check costs a file
# read and no network. Two extractors that disagree about a figure is the
# cheapest evidence there is that one of them invented it.
#
# What the arm does NOT do: correct anything. A flag says "these two readings of
# the same page disagree, an owner must look", and that is all it says. The
# independent extractor is not a better witness than the VLM by fiat — it drops
# table structure, it merges cells, it misses scanned pages — so a disagreement
# is a question, never a verdict on which side is right.
INDEPENDENT_DIR = config.EXTRACTED_DIR / "pymupdf"
INDEPENDENT_NAME = "pymupdf"
# the fields the ratification scopes the arm to: money, where a wrong digit is
# an answer that looks authoritative and is false
MONEY_FIELDS = ("total_financing", "gcf_funding_requested", "co_financing",
                "financial_instruments")
CROSS_CHECK_STATUSES = ("canonical",)
CROSS_CHECK_FLAGS = ("not-in-document", "not-on-cited-page")

_INDEP_PAGE = re.compile(r"(?m)^=== PAGE (\d+) ===$")
# a figure as one token. The spaced variant exists only for the page side: PDF
# text layers print '1 234 567' and break numbers over lines, and joining those
# must never be allowed to invent a figure on the CANDIDATE side.
_FIGURE = re.compile(r"\d[\d.,]*\d|\d")
_FIGURE_SPACED = re.compile(r"\d[\d.,\s]*\d")
# below this a page carries no readable text layer (scan, or a page the
# extractor dropped): unknown, not absent
_MIN_INDEP_CHARS = 80
# a superscript footnote marker the text layer glued to a figure is 1-2 digits
# and is never 0 or 0-led: '7', '10', '39' — never '0' or '00', because
# stripping a trailing zero is a change of MAGNITUDE ('49,944,050' against
# '4,994,405'), which is exactly the class of error the arm exists to catch
_GLUE_MARKER = re.compile(r"[1-9]\d?$")
# and never chew a key down to something too short to mean anything
_MIN_GLUE_KEY = 3
# a number whose thousands are grouped with spaces, and unmistakably that: a
# 1-3 digit head, 3-digit groups after it, and no digit or separator touching
# either end. '27 054' / '55 000 000' match; '1,234 5,678' does not
_SPACED_THOUSANDS = re.compile(r"(?<![\d.,])\d{1,3}(?:[  ]\d{3})+(?:[.,]\d+)?(?![\d.,])")


def independent_pages(doc_id: str, root: Optional[Path] = None) -> Optional[Dict[int, str]]:
    """The independent extraction of one document, {page: text}, or None."""
    path = Path(root or INDEPENDENT_DIR) / f"{doc_id}.txt"
    if not path.exists():
        return None
    parts = _INDEP_PAGE.split(path.read_text(encoding="utf-8", errors="replace"))
    return {int(parts[i]): parts[i + 1] for i in range(1, len(parts), 2)}


def _figure_key(tok: str) -> Optional[str]:
    """A figure's digits, separators and leading zeros removed.

    '49,944,050' / '49944050' / '49.944.050' all key to the same thing, so the
    two extractors' different ideas of a thousands separator never read as a
    disagreement. Single digits are dropped: '$5 million' cannot be told apart
    from a bullet number and would confirm itself against anything.
    """
    d = re.sub(r"\D", "", tok)
    return (d.lstrip("0") or "0") if len(d) >= 2 else None


def figure_keys(text: str, spaced: bool = False) -> set:
    out = {_figure_key(m.group(0)) for m in _FIGURE.finditer(text)}
    if spaced:
        out |= {_figure_key(m.group(0)) for m in _FIGURE_SPACED.finditer(text)}
    out.discard(None)
    return out


def unsplit_thousands(text: str) -> str:
    """Close the space inside a space-GROUPED number, and nothing else.

    A store raw prints a figure the way the markdown did, and the markdown
    prints what the form's cell held: '27 054 | million USD' is one number with
    a space inside it. Keyed strictly it becomes the fragments '27' and '54',
    and comparing a fragment is worse than not comparing at all — FP68 was
    flagged against its own correct figure, because '54' is not on its page and
    27054 is.

    The page side may join any run of digits and spaces (`_FIGURE_SPACED`)
    because a text layer breaks numbers anywhere. The candidate side may not:
    joining '1,234 5,678' into 12345678 would invent a figure and could confirm
    the store against a page that prints no such thing. So only a WELL-FORMED
    thousands grouping closes here — a 1-3 digit head, then 3-digit groups, not
    touching a digit or separator at either end. '27 054' and '55 000 000'
    qualify; '1,234 5,678' does not, and stays two figures.
    """
    return _SPACED_THOUSANDS.sub(lambda m: re.sub(r"[  ]", "", m.group(0)), text)


def glued_footnote_key(keys: set, page_keys: set) -> Optional[str]:
    """The page key that IS one of `keys` with a footnote marker glued to it.

    A PDF text layer renders a superscript beside a figure as more digits:
    '880,000,000' + marker 7 comes out '880,000,0007'; marker 4 + '152,500,000'
    comes out '4152,500,000'; '580.0' + marker 10 comes out '580.010'. Three of
    the eight false positives the cross-check census turned up are this and
    nothing else.

    Deliberately narrow: strip 1-2 digits off ONE end, require what remains to
    equal a candidate key EXACTLY, and require the stripped digits to look like
    a marker (1-99, never 0-led). No substring test and no edit distance, so
    two genuinely different figures can never meet here.
    """
    for pk in sorted(page_keys):                 # sorted: a total order, so the
        for n in (1, 2):                         # same build reports the same key
            if len(pk) - n < _MIN_GLUE_KEY:
                continue
            for marker, rest in ((pk[:n], pk[n:]), (pk[-n:], pk[:-n])):
                if not _GLUE_MARKER.fullmatch(marker):
                    continue
                if (rest.lstrip("0") or "0") in keys:
                    return pk
    return None


def _prints_the_value(cand: dict, body: str) -> bool:
    """The page prints the same AMOUNT in another notation ('$ 40 million' for
    a raw of 'USD 40,000,000').

    Read with the builder's own amount reader and compared with the builder's
    own precision rule, so 'agrees' means here exactly what it means everywhere
    else in this file.
    """
    if cand.get("value") is None:
        return False
    grain = _grain_of(cand)
    for got in _iter_amounts(body):
        if got["value"] is None:
            continue
        if abs(got["value"] - cand["value"]) <= max(grain + got.get("_grain", 0.0)
                                                    + 1e-6, 0.5):
            return True
    return False


def _nearest_figures(key: str, page_keys: set, limit: int = 3) -> List[str]:
    """The page's own figures closest to the one the store claims — the first
    thing a human wants to see beside a flag ('79690370' beside '69830370')."""
    def dist(other: str) -> int:
        if abs(len(other) - len(key)) > 1:
            return 99
        prev = list(range(len(other) + 1))
        for i, a in enumerate(key, 1):
            cur = [i]
            for j, b in enumerate(other, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
            prev = cur
        return prev[-1]
    # a TOTAL order: page_keys is a set, and ties broken by iteration order
    # would make the same build produce different bytes on different runs
    scored = sorted(((dist(k), len(k), k) for k in page_keys))
    return [k for d, _, k in scored[:limit] if d < 99]


def cross_check_candidate(cand: dict, pages: Dict[int, str],
                          doc_keys: Optional[set] = None) -> Tuple[str, dict]:
    """One candidate against the independent extraction. Returns (verdict, detail)."""
    # the raw's own space-grouped thousands are closed first, so the candidate
    # is checked on the figure it prints rather than on a fragment of it
    line = unsplit_thousands(cand.get("raw", "").split("\n")[0])
    keys = figure_keys(line)
    if not keys:
        return "no-figure", {}
    body = pages.get(cand.get("page"))
    if body is None or len(body.strip()) < _MIN_INDEP_CHARS:
        return "no-independent-page", {}
    page_keys = figure_keys(body, spaced=True)
    if keys & page_keys:
        return "confirmed-print", {}
    if glued_footnote_key(keys, page_keys):
        return "confirmed-footnote-glue", {}
    if _prints_the_value(cand, body):
        return "confirmed-value", {}
    figure = max(keys, key=len)
    detail = {"figure": figure,
              "independent_page_prints": _nearest_figures(figure, page_keys)}
    if doc_keys is not None and (keys & doc_keys):
        return "not-on-cited-page", detail
    return "not-in-document", detail


def cross_check_meta(doc_id: str, facts: Dict[str, List[dict]],
                     root: Optional[Path] = None,
                     statuses: Tuple[str, ...] = CROSS_CHECK_STATUSES
                     ) -> Tuple[Optional[dict], Dict[str, int]]:
    """Re-read every canonical money figure out of the independent extraction.

    Returns (meta block or None, verdict counts). Mutates the flagged candidates
    — they gain `cross_check` — and nothing else: a document whose figures all
    check out is left byte-identical to a build without the arm, so the census
    comes back in the counts rather than in the document. The arm writes into
    the store only where it disagrees.
    """
    pages = independent_pages(doc_id, root)
    if pages is None:
        return None, {"no-independent-extraction": 1}
    doc_keys = None
    checked, verdicts, flagged = 0, {}, []
    for field in MONEY_FIELDS:
        for cand in facts.get(field) or []:
            if cand.get("status") not in statuses:
                continue
            if doc_keys is None:
                doc_keys = figure_keys("\n".join(pages.values()), spaced=True)
            checked += 1
            verdict, detail = cross_check_candidate(cand, pages, doc_keys)
            verdicts[verdict] = verdicts.get(verdict, 0) + 1
            if verdict not in CROSS_CHECK_FLAGS:
                cand.pop("cross_check", None)
                continue
            cand["cross_check"] = verdict
            flagged.append({"field": field, "raw": cand["raw"], "page": cand["page"],
                            "status": cand["status"], "value": cand.get("value"),
                            "verdict": verdict, **detail,
                            **({"corrected": True} if cand.get("corrected") else {})})
    if not flagged:
        return None, verdicts
    return {"independent": INDEPENDENT_NAME, "checked": checked,
            "verdicts": dict(sorted(verdicts.items())), "flagged": flagged,
            "meaning": "the figure the store publishes is not printed by an "
                       "independent text extraction of the same PDF page. A "
                       "question for the next adjudication, not a correction.",
            "ratified": RATIFIED_SERVING}, verdicts


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build_document(doc_id: str, text: str, fallback: bool = True,
                   decisions: Optional["Decisions"] = None,
                   extra_rules: bool = True) -> dict:
    pages = split_pages(text)
    era = era_of(text)
    raw = extract_candidates(pages, fallback=fallback, era=era, extra_rules=extra_rules)
    heads = {f: bool(rx.search(text)) for f, rx in _TEMPLATE_HEADING.items()}
    facts = {f: finalize(f, cs, heads.get(f, False)) for f, cs in raw.items() if cs}
    facts = {f: cs for f, cs in facts.items() if cs}
    board = board_of(doc_id)
    code = None
    m = re.search(r"(?:^|_)gcf[-_]b(\d{1,2})[-_](\d{2})[-_]add(\d{2})", doc_id, re.I)
    if m:
        code = f"GCF/B.{int(m.group(1))}/{m.group(2)}/Add.{m.group(3)}"
    if code:
        facts.setdefault("board_code", []).append(
            {"raw": code, "value": None, "currency": None, "unit": None, "page": 0,
             "section": "doc_id", "status": "canonical"})
    for page, body in pages[:6]:
        for m in re.finditer(r"GCF/B\.\s?\d{1,2}/\d{2}/Add\.\s?\d{1,2}", body):
            facts.setdefault("board_code", []).append(
                {"raw": m.group(0), "value": None, "currency": None, "unit": None,
                 "page": page, "section": "cover", "status": "supporting"})
    meta: Dict[str, object] = {}
    if decisions is not None:
        recs = apply_fact_corrections(doc_id, facts, decisions, defer=True)
        if recs:
            meta["corrections"] = recs
        absent = absence_meta(doc_id, facts, decisions)
        if absent:
            meta["confirmed_absence"] = absent
    labels = mapped_label_meta(pages, facts)
    if labels:
        meta["mapped_labels"] = labels
    found = [f for f in CORE_FIELDS if f in facts]
    coverage = {
        "era": era,
        "pages": len(pages),
        "fields": len(facts),
        "core_found": found,
        "core_missing": [f for f in CORE_FIELDS if f not in facts],
        "llm_fallback": False,
    }
    flags = suspect_flags(facts)
    if flags:
        coverage["suspect"] = ";".join(flags)
    out = {"facts": facts, "coverage": coverage, "_board": board}
    if meta:
        meta["ratified"] = RATIFIED
        out["meta"] = meta
    return out


def _canon_of(facts: Dict[str, List[dict]], field: str) -> Optional[dict]:
    return next((c for c in facts.get(field, []) if c["status"] == "canonical"), None)


def suspect_flags(facts: Dict[str, List[dict]]) -> List[str]:
    """Document-level arithmetic that cannot be true, surfaced instead of shipped.

    The GCF request is a PART of total financing (GCF + co-finance), so a
    canonical request above the canonical total means one of the two was read
    from the wrong line — the row says so rather than looking authoritative.
    """
    flags = []
    gcf, tot = _canon_of(facts, "gcf_funding_requested"), _canon_of(facts, "total_financing")
    if (gcf and tot and gcf["value"] is not None and tot["value"] is not None
            and _compatible(gcf, tot) and gcf["value"] > tot["value"] * 1.001):
        flags.append("gcf>total")
    return flags


LLM_PROMPT = (
    "You read one Green Climate Fund funding-proposal document, supplied as "
    "'[page N]' blocks. Extract ONLY facts that are literally printed. Respond "
    "with JSON:\n"
    '{"title": {"raw": "<exact text>", "page": <int>}, '
    '"countries": {"raw": "...", "page": <int>}, '
    '"accredited_entity": {"raw": "...", "page": <int>}, '
    '"total_financing": {"raw": "<amount exactly as printed>", "page": <int>}, '
    '"gcf_funding_requested": {"raw": "<amount exactly as printed>", "page": <int>}}\n'
    "Every 'raw' MUST be copied character for character from the page it cites. "
    "Use null for anything not printed. Never compute or round a number."
)


def uncorrected(cand: dict) -> dict:
    """A carried-forward llm candidate as the fallback pass first produced it.

    The reuse path exists to avoid re-spending the model budget, NOT to carry a
    ratified correction forward: corrections are applied fresh on every build,
    out of the data file. A candidate carried forward with its correction
    already baked in is a candidate no correction row can find any more — so
    reusing the SHIPPED registry would make its own ratified rows stop landing,
    quietly, one rebuild at a time. Restoring `corrected_from` makes the reuse
    idempotent: the row finds the same target it found the first time and
    produces the same result.
    """
    if not cand.get("corrected"):
        return dict(cand)
    return dict(cand.get("corrected_from") or cand)


def carry_forward_llm(paths: List[Path], docs: Dict[str, dict],
                      previous: Dict[str, dict],
                      empty: Sequence[Path]) -> Tuple[List[Path], int]:
    """Merge a previous build's fallback pass into this one: ``(todo, reused)``.

    Carry forward EVERY verified fallback candidate the previous build
    published, not only those of documents that are still empty: when a parser
    improvement gives a document its first deterministic financing fact, the
    title/countries/entity the model had already verified for it must not be
    deleted as a side effect (b21-10-add06, b19-22-add09).

    THE FLAG IS A CALL RECORD, NOT A CANDIDATE COUNT — and that is the defect
    this function was extracted to close. On a fresh build ``llm_fallback()``
    sets ``coverage.llm_fallback = True`` for every document it SENDS to the
    model, before it knows whether anything came back: a call that returns
    nothing verifiable (every candidate dropped by the ``raw not in page``
    check) still flags the document, because what the flag tells a reader is
    "the deterministic parser found nothing here and the model was asked",
    which is exactly the caveat ``registry._extraction_flags`` publishes on
    the line.

    The reuse path used to set the flag only inside the branch that had
    candidates to carry, so a document whose call produced NOTHING lost its
    flag on the next rebuild — silently, and permanently, since a stem present
    in ``previous`` is also never re-queued in ``todo``. Two documents lost it
    that way: ``193_gcf-b22-10-add01-rev01`` and
    ``196_gcf-b19-22-add21-rev01``, both two-page board notices with no
    template block, both flagged at HEAD and both unflagged after a reuse
    rebuild. The registry then published their lines with no caveat at all.

    So the flag is now carried from the reuse seed FIRST, independent of
    whether any candidate survived, and the candidate merge only ever adds to
    it. The seed row's own ``coverage.llm_fallback`` is the ledger of the call
    — it is what the fresh build wrote down at call time — so reuse is
    faithful to the fresh build by construction rather than by coincidence.
    """
    todo: List[Path] = []
    reused = 0
    empty_paths = set(empty)
    for path in paths:
        seed = previous.get(path.stem) or {}
        if ((seed.get("coverage") or {}).get("llm_fallback")):
            docs[path.stem]["coverage"]["llm_fallback"] = True
        # a correction MOVES the section ('llm' -> 'corrected'), so a
        # carried-forward candidate is recognised by where it came from as
        # well as by where it is now
        old = [(f, c) for f, cs in (seed.get("facts") or {}).items()
               for c in cs
               if c.get("section") == "llm"
               or (c.get("corrected_from") or {}).get("section") == "llm"]
        if not old:
            if path in empty_paths and path.stem not in previous:
                todo.append(path)
            continue
        for f, c in old:
            docs[path.stem]["facts"].setdefault(f, []).append(uncorrected(c))
        docs[path.stem]["coverage"]["llm_fallback"] = True
        docs[path.stem]["coverage"]["fields"] = len(docs[path.stem]["facts"])
        reused += 1
    return todo, reused


def llm_fallback(paths: List[Path], docs: Dict[str, dict], max_calls: int) -> Tuple[int, int]:
    """Second pass for documents where the deterministic pass found nothing."""
    if not paths:
        return 0, 0
    import openai
    client = openai.OpenAI(base_url=config.OPENAI_BASE_URL or None, timeout=90)
    calls = ok = 0
    for p in paths[:max_calls]:
        pages = split_pages(p.read_text(encoding="utf-8", errors="replace"))
        blob = "\n\n".join(f"[page {n}]\n{b[:2500]}" for n, b in pages[:12])[:24000]
        calls += 1
        try:
            resp = client.chat.completions.create(
                model=config.CHAT_MODEL, max_completion_tokens=700,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": LLM_PROMPT},
                          {"role": "user", "content": blob}])
            data = json.loads(resp.choices[0].message.content or "{}")
        except Exception as e:                                   # noqa: BLE001
            print(f"  llm fail {p.stem}: {type(e).__name__}: {str(e)[:90]}")
            continue
        by_page = dict(pages)
        row = docs[p.stem]
        added = 0
        for field, got in (data or {}).items():
            if not isinstance(got, dict) or not got.get("raw"):
                continue
            raw, page = str(got["raw"]).strip(), got.get("page")
            if not isinstance(page, int) or raw not in by_page.get(page, ""):
                continue                                  # unverifiable -> dropped
            parsed = (read_amount(raw) if field in NUMERIC_FIELDS else None)
            row["facts"].setdefault(field, []).append({
                "raw": raw[:300],
                "value": parsed["value"] if parsed else None,
                "currency": parsed["currency"] if parsed else None,
                "unit": parsed["unit"] if parsed else None,
                "page": page, "section": "llm", "status": "supporting"})
            added += 1
        row["coverage"]["llm_fallback"] = True
        row["coverage"]["fields"] = len(row["facts"])
        row["coverage"]["core_found"] = [f for f in CORE_FIELDS if f in row["facts"]]
        row["coverage"]["core_missing"] = [f for f in CORE_FIELDS if f not in row["facts"]]
        ok += bool(added)
    return calls, ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--only", default=None, help="substring filter on the doc id")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--force-llm", action="store_true",
                    help="re-call the model instead of reusing the previous build's candidates")
    ap.add_argument("--out", default=str(REGISTRY_V2))
    ap.add_argument("--reuse-llm-from", default=str(REGISTRY_V2),
                    help="build whose verified llm candidates are reused (default: the "
                         "shipped registry). Decoupled from --out so a rebuild into a "
                         "scratch path costs no model calls.")
    ap.add_argument("--dry-run", action="store_true",
                    help="never write data/registry_v2.json and never call the model: "
                         "the mode Phase 3 diagnosis runs in")
    ap.add_argument("--no-fallback", action="store_true",
                    help="strict rules only — the pre-fallback baseline, for the "
                         "additive-discipline diff")
    ap.add_argument("--no-extra-rules", action="store_true",
                    help="skip the third (EXTRA_RULES) pass — no date fields and no "
                         "'Implementing entity' mapping; the pre-decision baseline")
    ap.add_argument("--no-decisions", action="store_true",
                    help="ignore data/registry_corrections.json and "
                         "data/registry_absences.json — the pre-ratification baseline, "
                         "for the before/after diff")
    ap.add_argument("--no-cross-check", action="store_true",
                    help="skip the cross-extractor verification arm — the "
                         "pre-arm baseline, for the additive-discipline diff")
    ap.add_argument("--independent", default=str(INDEPENDENT_DIR),
                    help="the independent extraction the arm reads (default: "
                         "data/extracted/pymupdf)")
    ap.add_argument("--corrections", default=str(CORRECTIONS_FILE))
    ap.add_argument("--absences", default=str(ABSENCES_FILE))
    a = ap.parse_args()

    out = Path(a.out)
    if a.dry_run and out.resolve() == REGISTRY_V2.resolve():
        ap.error("--dry-run must not write the shipped registry; pass --out <scratch path>")

    v1 = {}
    if REGISTRY_V1.exists():
        v1 = json.loads(REGISTRY_V1.read_text(encoding="utf-8")).get("documents", {})

    paths = sorted(SOURCE_DIR.glob("*.md"))
    if a.only:
        paths = [p for p in paths if a.only.lower() in p.stem.lower()]
    paths = paths[: a.limit]

    dec = None if a.no_decisions else Decisions.load(a.corrections, a.absences)
    if dec is not None and not (dec.meta["corrections"]["present"]
                                and dec.meta["absences"]["present"]):
        print("WARNING: a ratified decisions file is missing — "
              f"corrections={dec.meta['corrections']['present']} "
              f"absences={dec.meta['absences']['present']}")

    t0 = time.time()
    docs: Dict[str, dict] = {}
    for p in paths:
        built = build_document(p.stem, p.read_text(encoding="utf-8", errors="replace"),
                               fallback=not a.no_fallback, decisions=dec,
                               extra_rules=not a.no_extra_rules)
        base = dict(v1.get(p.stem) or {})
        base.pop("error", None)
        base.setdefault("fp", int(m.group(1)) if (m := re.search(r"fp(\d{2,3})", p.stem)) else None)
        base.setdefault("board", built.pop("_board"))
        built.pop("_board", None)
        base.setdefault("year", year_of(p.stem))
        docs[p.stem] = {**base, "facts": built["facts"], "coverage": built["coverage"]}
        if built.get("meta"):
            docs[p.stem]["meta"] = built["meta"]
        if dec is not None:
            top = apply_top_level_corrections(p.stem, docs[p.stem], dec)
            if top:
                m = docs[p.stem].setdefault("meta", {"ratified": RATIFIED})
                m["corrections"] = list(m.get("corrections") or []) + top
    det_secs = time.time() - t0

    # "deterministic parsing found nothing": no financing fact at all. These are
    # the REDD+ results-based-payment proposals and withdrawal notices, which
    # carry no A.x / B.2 / C.1 template block for a regex to lock onto.
    empty = [p for p in paths
             if not (set(docs[p.stem]["facts"]) & {"total_financing", "gcf_funding_requested"})]
    calls = fixed = reused = 0
    if empty and not a.no_llm:
        # a rebuild after a parser change must not re-spend the call budget:
        # verified fallback candidates from the previous build are reused unless
        # --force-llm is given
        previous = {}
        prev_path = Path(a.reuse_llm_from)
        if prev_path.exists() and not a.force_llm:
            previous = json.loads(prev_path.read_text(encoding="utf-8")).get("documents", {})
        todo, reused = carry_forward_llm(paths, docs, previous, empty)
        print(f"llm fallback: {len(empty)} documents with no deterministic financing fact "
              f"| {reused} reused from the previous build | {len(todo)} to call "
              f"(cap {MAX_LLM_CALLS})")
        if a.dry_run and todo:
            print(f"  dry run: {len(todo)} documents would be sent to the model — not called")
            todo = []
        calls, fixed = llm_fallback(todo, docs, MAX_LLM_CALLS)

    # four of the 58 ratified rows correct a verified llm-fallback candidate,
    # which only exists once the merge above has run
    if dec is not None and dec.deferred:
        for stem, entries in list(dec.deferred.items()):
            dec.deferred[stem] = []
            row = docs[stem]
            recs = apply_fact_corrections(stem, row["facts"], dec, entries=entries)
            if recs:
                m = row.setdefault("meta", {"ratified": RATIFIED})
                m["corrections"] = list(m.get("corrections") or []) + recs
                row["coverage"]["fields"] = len(row["facts"])
                row["coverage"]["core_found"] = [f for f in CORE_FIELDS if f in row["facts"]]
                row["coverage"]["core_missing"] = [f for f in CORE_FIELDS
                                                   if f not in row["facts"]]
                flags = suspect_flags(row["facts"])
                row["coverage"].pop("suspect", None)
                if flags:
                    row["coverage"]["suspect"] = ";".join(flags)
                absent = absence_meta(stem, row["facts"], dec)
                if absent:
                    m["confirmed_absence"] = absent
                else:
                    m.pop("confirmed_absence", None)

    # the cross-extractor verification arm: runs LAST, over the finished facts,
    # so it checks what the build actually publishes — corrections applied, llm
    # candidates merged, deferred rows landed
    census: Dict[str, object] = {}
    if not a.no_cross_check:
        counts: Dict[str, int] = {}
        flagged_docs, no_independent = [], []
        for stem, row in docs.items():
            block, got = cross_check_meta(stem, row["facts"], root=Path(a.independent))
            if got.pop("no-independent-extraction", None):
                no_independent.append(stem)
            for k, n in got.items():
                counts[k] = counts.get(k, 0) + n
            if block is not None:
                flagged_docs.append(stem)
                row.setdefault("meta", {"ratified": RATIFIED})["cross_check"] = block
        census = {"independent": INDEPENDENT_NAME,
                  "source": Path(a.independent).name,
                  "scope": f"canonical candidates of {', '.join(MONEY_FIELDS)}",
                  "checked": sum(counts.values()),
                  "confirmed": sum(n for k, n in counts.items()
                                   if k.startswith("confirmed")),
                  "flagged": sum(counts.get(k, 0) for k in CROSS_CHECK_FLAGS),
                  "verdicts": dict(sorted(counts.items())),
                  "documents_flagged": len(flagged_docs),
                  "documents_without_an_independent_extraction": sorted(no_independent),
                  "meaning": "a flag is a question for the next adjudication, "
                             "never an automatic correction",
                  "ratified": RATIFIED_SERVING}

    payload = {"schema_version": 2, "source": SOURCE_DIR.name}
    if dec is not None:
        payload["meta"] = {
            "data_decisions": {
                **dec.meta,
                "applied": len(dec.applied),
                "carried_forward": dec.carried,
                "unapplied": dec.unapplied,
                "absences_published": len(dec.absences_published),
                "absences_not_published": dec.absences_skipped,
                "alarms": dec.alarms,
            }}
    if census:
        payload.setdefault("meta", {})["cross_check"] = census
    payload["documents"] = dict(sorted(docs.items()))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    fields = sorted({f for r in docs.values() for f in r["facts"]})
    print(f"\n{len(docs)} documents | deterministic pass {det_secs:.1f}s "
          f"({det_secs / max(len(docs), 1) * 1000:.0f} ms/doc) | llm calls {calls} "
          f"(usable {fixed}, reused {reused}) | "
          f"eligible-but-uncalled {max(0, len(empty) - calls - reused)}")
    print(f"{'field':<28}{'docs':>6}{'cands':>7}{'canon':>7}{'conflict':>10}")
    for f in fields:
        cs = [c for r in docs.values() for c in r["facts"].get(f, [])]
        nd = sum(1 for r in docs.values() if f in r["facts"])
        print(f"{f:<28}{nd:>6}{len(cs):>7}"
              f"{sum(1 for c in cs if c['status'] == 'canonical'):>7}"
              f"{sum(1 for c in cs if c['status'] == 'conflicting'):>10}")
    if dec is not None:
        print(f"\ndata decisions | corrections applied {len(dec.applied)}/"
              f"{dec.meta['corrections']['count']} | carried forward "
              f"{len(dec.carried)} | absences published "
              f"{len(dec.absences_published)}/{dec.meta['absences']['count']} "
              f"(not published {len(dec.absences_skipped)})")
        for msg in dec.alarms:
            print(f"  !! {msg}")
        stale = [u for u in dec.unapplied
                 if (next((e for e in dec.by_doc.get(u["doc_id"], [])
                           if e["id"] == u["id"]), {}).get("wrong") or {}
                     ).get("section") == "llm"]
        if stale:
            print(f"  !! {len(stale)} of the unapplied rows correct a VERIFIED "
                  f"LLM-FALLBACK candidate ({', '.join(u['id'] for u in stale)}). "
                  f"The reuse source ({a.reuse_llm_from}) is a build those rows had "
                  f"already been applied to, and a row whose target was DELETED "
                  f"(confirm-absence) cannot be restored from it. Rebuild with "
                  f"--reuse-llm-from <a --no-decisions build> or --force-llm.")
        if not dec.alarms:
            print("  every ratified decision landed on the candidate it names")
    if census:
        print(f"\ncross-extractor arm ({census['source']}) | checked "
              f"{census['checked']} canonical money facts | confirmed "
              f"{census['confirmed']} | FLAGGED {census['flagged']} across "
              f"{census['documents_flagged']} documents")
        for k, n in census["verdicts"].items():
            print(f"  {k:<22}{n:>6}")
        if census["documents_without_an_independent_extraction"]:
            print("  no independent extraction: "
                  f"{len(census['documents_without_an_independent_extraction'])} documents")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
