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

Candidate schema (one entry per source location):
  {"raw": "<exact source text>", "value": <number|null>, "currency": "USD"|"EUR"|null,
   "unit": "million"|"thousand"|"billion"|"years"|"months"|null,
   "page": <int>, "section": "A.8", "status": "canonical"|"supporting"|"conflicting"}

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
from typing import Dict, List, Optional, Tuple

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


def read_amounts(window: str, limit: int = 1,
                 skip_lines: Optional[re.Pattern] = None) -> List[dict]:
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
    for p in _iter_amounts(window):
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


def _iter_amounts(window: str):
    """Every real money amount in a window, in order.

    Skips template placeholders ('Enter amount'), percentages, durations and
    bare years. `raw` is the exact source substring that was read.
    """
    for m in _AMOUNT_RE.finditer(window):
        num = m.group("num")
        # 'USD35 million' glues the currency to the figure: that is a currency
        # marker, not the 'JAH30' word-glue the guard is meant to reject
        head = m.start() if m.group("pre") else m.start("num")
        # 'USD$ 500 M': the currency may be spelled twice, and the word-glue
        # guard must not read the 'D' of USD as the glue
        before = _CUR_TAIL.sub("", window[max(0, head - 24):head])
        after = window[m.end("num"):m.end("num") + 24]
        if _NOISE_BEFORE.search(before) or _GLUED.search(before):
            continue
        if _NOT_MONEY_AFTER.match(after):
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


def _mk(label: str, terminated: bool) -> re.Pattern:
    """Label regex. `terminated` (text fields) demands that the label is followed
    by ':' , '|' or the end of the line — 'Programme Title: X' and '| Title | X |'
    are field labels, 'the Accredited Entity shall ...' is prose."""
    tail = r"[ \t]*\**" + (r"(?=[ \t]*(?:[:|]|$|\n))" if terminated else "") + r"[ \t]*[:|.\-–]?"
    return re.compile(_PREFIX + _SEC + r"\**[ \t]*(?:" + label + r")" + tail, re.I)


class Rule:
    """One labelled source of one field."""

    def __init__(self, field, label, section, kind, rank=0, max_page=None, window=(6, 400),
                 amounts=1, skip_lines=None):
        self.field, self.section, self.kind, self.rank = field, section, kind, rank
        self.max_page, self.window, self.amounts = max_page, window, amounts
        self.skip_lines = re.compile(skip_lines, re.I) if skip_lines else None
        self.re = _mk(label, terminated=(kind == "text"))


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

# every label word at once: one cheap pre-filter search per page, so pages that
# cannot carry a template field never run the full rule set
_ANY_LABEL = re.compile(
    r"Total GCF funding|GCF funding requ|Requested GCF|Received GCF|GCF fin\w+ to recipient"
    r"|Total (?:project )?financing|Total co-financing|Total funding requested|Total project cost"
    r"|Financial instrument|Implementation period|lifespan|ES[SG] category"
    r"|Expected (?:mitigation|adaptation)|Accredited\s+[Ee]ntity|Countr(?:y|ies)"
    r"|programme?\s*title|Project title|Executing [Ee]ntity|National designated authority"
    r"|funding proposal (?:titled|entitled)|Project size", re.I)

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


def _window(body: str, start: int, max_lines: int, max_chars: int) -> str:
    seg = body[start:start + max_chars * 3]
    lines = seg.split("\n")
    out = [lines[0]]
    for ln in lines[1:max_lines]:
        if _STOP.match(ln):
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
    return {"raw": parsed["raw"], "value": parsed.get("value"),
            "currency": parsed.get("currency"), "unit": parsed.get("unit"),
            "page": page, "section": _norm_sec(sec, rule.section) + suffix,
            "status": "supporting", "_rank": rule.rank, "_grain": parsed.get("_grain", 0.0),
            "_bare": parsed.get("_bare"), "_bare_unit": parsed.get("_bare_unit"),
            # a template TOTAL line may be elected canonical and may contradict
            # the canonical one; a component/instrument line may do neither
            "_canon": True, "_conflict": rule.kind in ("amount", "count", "period",
                                                       "beneficiaries", "ess", "text")}


def extract_candidates(pages: List[Tuple[int, str]]) -> Dict[str, List[dict]]:
    """Deterministic pass over one document's pages -> {field: [candidate, ...]}."""
    out: Dict[str, List[dict]] = {}
    for page, body in pages:
        if not _ANY_LABEL.search(body):
            continue
        for rule in RULES:
            if rule.max_page and page > rule.max_page:
                continue
            for m in rule.re.finditer(body):
                win = _window(body, m.end(), *rule.window)
                sec = m.groupdict().get("sec")
                if rule.kind == "amount":
                    for p in read_amounts(win, rule.amounts, rule.skip_lines):
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
                elif rule.kind == "text":
                    p = read_text(win)
                    if p:
                        out.setdefault(rule.field, []).append(_cand(rule, page, sec, p))
                elif rule.kind == "beneficiaries":
                    _read_beneficiaries(out, rule, page, sec, win)
                elif rule.kind == "instruments":
                    _read_instruments(out, rule, page, sec, win)
    return out


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

CORE_FIELDS = ["title", "countries", "accredited_entity",
               "total_financing", "gcf_funding_requested"]


def era_of(text: str) -> str:
    if _ERA_MODERN.search(text):
        return "A5-A14 block (FP template v2/v3)"
    if _ERA_OLD.search(text):
        return "A.1.x block (FP template v1)"
    return "unrecognized template"


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build_document(doc_id: str, text: str) -> dict:
    pages = split_pages(text)
    raw = extract_candidates(pages)
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
    found = [f for f in CORE_FIELDS if f in facts]
    coverage = {
        "era": era_of(text),
        "pages": len(pages),
        "fields": len(facts),
        "core_found": found,
        "core_missing": [f for f in CORE_FIELDS if f not in facts],
        "llm_fallback": False,
    }
    flags = suspect_flags(facts)
    if flags:
        coverage["suspect"] = ";".join(flags)
    return {"facts": facts, "coverage": coverage, "_board": board}


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
    a = ap.parse_args()

    v1 = {}
    if REGISTRY_V1.exists():
        v1 = json.loads(REGISTRY_V1.read_text(encoding="utf-8")).get("documents", {})

    paths = sorted(SOURCE_DIR.glob("*.md"))
    if a.only:
        paths = [p for p in paths if a.only.lower() in p.stem.lower()]
    paths = paths[: a.limit]

    t0 = time.time()
    docs: Dict[str, dict] = {}
    for p in paths:
        built = build_document(p.stem, p.read_text(encoding="utf-8", errors="replace"))
        base = dict(v1.get(p.stem) or {})
        base.pop("error", None)
        base.setdefault("fp", int(m.group(1)) if (m := re.search(r"fp(\d{2,3})", p.stem)) else None)
        base.setdefault("board", built.pop("_board"))
        built.pop("_board", None)
        base.setdefault("year", year_of(p.stem))
        docs[p.stem] = {**base, "facts": built["facts"], "coverage": built["coverage"]}
    det_secs = time.time() - t0

    # "deterministic parsing found nothing": no financing fact at all. These are
    # the REDD+ results-based-payment proposals and withdrawal notices, which
    # carry no A.x / B.2 / C.1 template block for a regex to lock onto.
    empty = [p for p in paths
             if not (set(docs[p.stem]["facts"]) & {"total_financing", "gcf_funding_requested"})]
    out = Path(a.out)
    calls = fixed = reused = 0
    if empty and not a.no_llm:
        # a rebuild after a parser change must not re-spend the call budget:
        # verified fallback candidates from the previous build are reused unless
        # --force-llm is given
        previous = {}
        if out.exists() and not a.force_llm:
            previous = json.loads(out.read_text(encoding="utf-8")).get("documents", {})
        todo = []
        for p in empty:
            old = [(f, c) for f, cs in (previous.get(p.stem, {}).get("facts") or {}).items()
                   for c in cs if c.get("section") == "llm"]
            if not old and p.stem not in previous:
                todo.append(p)
                continue
            for f, c in old:
                docs[p.stem]["facts"].setdefault(f, []).append(c)
            docs[p.stem]["coverage"]["llm_fallback"] = bool(old)
            docs[p.stem]["coverage"]["fields"] = len(docs[p.stem]["facts"])
            reused += bool(old)
        print(f"llm fallback: {len(empty)} documents with no deterministic financing fact "
              f"| {reused} reused from the previous build | {len(todo)} to call "
              f"(cap {MAX_LLM_CALLS})")
        calls, fixed = llm_fallback(todo, docs, MAX_LLM_CALLS)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"schema_version": 2, "source": SOURCE_DIR.name,
         "documents": dict(sorted(docs.items()))}, ensure_ascii=False, indent=1),
        encoding="utf-8")

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
    print(f"-> {out}")


if __name__ == "__main__":
    main()
