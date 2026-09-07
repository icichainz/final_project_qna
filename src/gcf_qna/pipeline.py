"""The per-turn orchestration, with no UI in it.

`chainlit_app.py` was the only place a turn existed, and
`scripts/eval_answers.py` re-implemented it "turn for turn" so a release run
could measure the path production runs. Two spellings of one turn is how a
recorded context and a live one come to differ in a byte nobody looked at —
the failure mode `_context_block`'s docstring already names for one f-string,
applied to the whole turn.

So the turn lives here, and both callers import it:

* the helpers that compute a turn's notes, probes, guards and prompts —
  moved from `chainlit_app` unchanged, byte for byte, because the context
  they build is a measured artifact;
* `TurnSettings` / `TurnPlan` / `TurnContext` / `TurnResult`, the data a turn
  carries between its stages;
* the stages themselves — `plan_turn` (conductor, guards, planner gate, the
  evidence matrix), `build_context` (retrieval fan-out, notes, probes, prompt
  assembly, evidence), `answer_turn` (the streamed answer call) — and
  `run_turn`, the driver that is `chainlit_app.main()` minus its rendering;
* `ConversationMemory`, the thread's first-class record of which documents
  the conversation has been about.

Nothing here imports chainlit, reads `sys.argv`, or touches the environment
beyond `gcf_qna.config`: the app builds a `TurnSettings` from `config`, and
the eval harness passes one explicitly (by design it reads no ambient env, so
a flag flipped in a deployed `.env` can never change what it measures).

`chainlit_app` re-exports every name it used to define, so the app's private
helpers stay importable where tests and the harness already look them up.
"""
from __future__ import annotations

import json
import logging
import re
import time
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from gcf_qna import config
from gcf_qna.boards import BOARD_YEARS, board_of, year_of
from gcf_qna.rag import planner, registry, verify
# ONE definition of the identifier patterns, shared with the registry: a second
# copy of the FP one here drifted (it lacked the trailing boundary and the
# zero-strip) and resolved 'fp2023' to FP202's document. _FP_RE stays as the
# module-local name every call site (and test) already uses.
from gcf_qna.rag.registry import FP_RE as _FP_RE
from gcf_qna.rag.registry import _BOARD_CODE_RE
# SYSTEM_PROMPT is imported for the compatibility re-export chain only
# (chainlit_app re-exports what this module holds).
from gcf_qna.app.prompts import (  # noqa: F401
    CONDUCTOR_PROMPT, SYSTEM_PROMPT, assemble, assemble_chat)

log = logging.getLogger("gcf_qna.pipeline")


def _registry_fp(doc_id: str):
    """The corpus registry's FP number for a stem, or None.

    An exact dict lookup on the id, never a parse of it: 'FP151' is
    '124_gcf-b27-02-add11' and no amount of reading that filename yields 151
    (the B.27-era stems carry no FP number at all — the same fact
    `_registry_doc` exists for, from the other direction).
    """
    try:
        return (registry.load().get(doc_id) or {}).get("fp")
    except Exception:
        return None      # the registry is an enhancement, never a blocker


def _doc_label(doc_id: str, page) -> str:
    """Citation header with precomputed FP number and board/year: the model
    reads identifiers and dates instead of deriving them. Uses the shared
    boards parser, which handles every corpus id format incl.
    '72_GCF_B.35_02_Add.05...' (review #4).

    The FP number is here because of `disc-subnational-pair`, 0.60 in six
    consecutive releases: retrieval put BOTH of the Global Subnational Climate
    Fund's proposals at rank 1 and 2, the model named both, cited both stems —
    and never wrote FP151 or FP152, because no note fired for that turn
    (`notes_used == {}`) and the retrieved page is a no-objection letter that
    names the two accredited entities and no identifier. Nothing in the whole
    context held the FP numbers, so the only way to that answer was a guess.
    The board/year precedent settles where the fix belongs: a per-document
    fact the model would otherwise have to invent goes in the header, computed
    once, exactly, from the registry.

    The CITATION is still the stem and the page — CORE says so explicitly,
    because an identifier in the header is an invitation to cite it.
    """
    label = doc_id + (f", p. {page}" if page else "")
    bits = []
    fp = _registry_fp(doc_id)
    if fp:
        bits.append(f"FP{fp}")
    b, y = board_of(doc_id), year_of(doc_id)
    if b and y:
        bits.append(f"B.{b}, {y}")
    if bits:
        label += " — " + ", ".join(bits)
    return label


_FR_WORDS = {"le", "la", "les", "des", "une", "un", "est", "et", "que", "qui",
             "quel", "quelle", "quels", "quelles", "pour", "dans", "avec", "sur",
             "comment", "pourquoi", "combien", "cette", "ce", "aux", "du", "de",
             "peux", "fais", "fait", "donne", "moi", "tableau", "merci", "bonjour"}
_EN_WORDS = {"the", "is", "are", "what", "which", "how", "of", "for", "in",
             "does", "do", "and", "that", "this", "with", "on", "to", "from",
             "give", "me", "show", "compare", "summary", "thanks", "hello"}


def _detect_lang(text: str):
    """FR/EN heuristic — code-detected so the answer language follows the
    LATEST message, not conversational momentum. An explicit in-message
    request ("présente ta réponse en français" inside an English sentence)
    beats the statistics, which would otherwise fight the user's ask."""
    low = text.lower()
    if re.search(r"\ben fran[cç]ais\b|\bin french\b", low):
        return "French"
    if re.search(r"\ben anglais\b|\bin english\b", low):
        return "English"
    toks = re.findall(r"[a-zàâçéèêëîïôùûüÿœ']+", text.lower())
    fr = sum(t in _FR_WORDS for t in toks)
    en = sum(t in _EN_WORDS for t in toks)
    if re.search(r"[àâçéèêëîïôùûüÿœ]", text.lower()):
        fr += 2
    if fr > en:
        return "French"
    if en > fr:
        return "English"
    return None


def _note_pages(notes) -> set:
    """(doc, page) pairs the computed notes publish. Registry notes now carry
    page-cited figures ('18.5 M USD (p.5, A.8)'), so an answer citing that
    page is grounded even when retrieval never returned it — the checker must
    not flag it as invented. A page belongs to every document named on its
    own line (main lines end '[stem, cover pages]', conflict lines name the
    stem in parentheses)."""
    out = set()
    for n in notes or []:
        for line in (n or "").splitlines():
            docs = re.findall(r"[\[(]([0-9]{1,3}_[\w.\-]+)", line)
            if not docs:
                continue
            for pg in re.findall(r"\(p\.(\d{1,3})[,)]", line):
                for d in docs:
                    out.add((d, int(pg)))
    return out


# ---------------------------------------------------------------------------
# the conflict probe (campaign Phase 2)
# ---------------------------------------------------------------------------
# `registry._conflict_lines` prints WHERE a document contradicts itself —
# 'gcf_funding_requested is printed as 28,654 million USD (p.5, A.8); also as
# 26,654 million USD (p.48, B.2(b)) — report both figures with their pages' —
# and the answer is instructed to report both. Retrieval does not always bring
# those pages back: the second printing sits deep in a component table and
# loses the similarity contest to the cover page that says the same thing in
# the question's own words. Measured at 8/14 conflict-class evidence pages for
# five consecutive releases, three cases at 0/2, unchanged by reranking.
#
# `Retriever.probe_pages` asks the other question — "show me THAT page of THAT
# document" — and recovers 18/18 of those pages on demand. It never wires
# itself in (a supplementary query that fired on its own would spend top-k
# slots on every turn that merely mentions a document), so this is the caller
# that decides when to ask: only when a CONFLICT line named pages this turn's
# retrieval did not return.
#: Distinct pages one turn may fetch by name. `_conflict_lines` prints at most
#: `_MAX_CONFLICT_LINES` lines naming at most `_MAX_CONFLICT_ALTS` + 1 prints
#: each, so six is the ceiling a note can name; no recorded turn asks for more
#: than three. Four is a backstop, and a turn that hits it keeps the pages the
#: note printed FIRST — the canonical figure's page leads every conflict line.
_MAX_PROBE_PAGES = 4
#: Excerpts one turn may append. Spent on chunks as well as pages: a page is
#: 1–5 chunks and the print the note names is not always in the first — FP153's
#: second printing is the fifth chunk of page 48 — and `probe_pages` gives
#: every asked page a slot before any page takes a second, so a leftover slot
#: buys a second chunk of a page rather than a page nobody asked for.
_MAX_PROBE_HITS = 4
#: The two patterns `_note_pages` reads a note line with, compiled for the
#: probe: a document is named on a line when it appears in a bracket or a
#: parenthesis, a page when the line prints '(p.<n>,' or '(p.<n>)'.
#: `_note_pages` keeps its inline copy — `tests/test_registry_resolver.py`
#: pins the app's SOURCE TEXT against `verify._NOTE_PAGE_RE`, so that copy is
#: not ours to move — and `tests/test_conflict_probe_wiring.py` pins these two
#: against both of the others. Three spellings of one rule, none free to
#: drift: the probe must be unable to ask for a page the citation gate would
#: then call invented.
_CONFLICT_DOC_RE = re.compile(r"[\[(]([0-9]{1,3}_[\w.\-]+)")
_CONFLICT_PAGE_RE = re.compile(r"\(p\.(\d{1,3})[,)]")


def _conflict_probe_asks(note, hits) -> list:
    """[(doc_id, [page, ...]), ...] a CONFLICT line names and this turn lacks.

    Read off the note's own text with the same two regexes that decide which
    cited pages are legal, so the probe can only ever ask for a page the model
    was already told about. Everything else is a refusal:

    * only lines that ARE conflict warnings — a main registry line prints the
      cover pages, which retrieval has no obligation to hold;
    * only pages missing from THIS turn's hits — the common case is that
      retrieval already found them (5 of release-12's 9 conflict turns), and
      re-fetching one would print the same excerpt twice;
    * at most `_MAX_PROBE_PAGES`, in the order the note printed them.
    """
    have = {(h.doc_id, h.page) for h in (hits or [])}
    asks: dict = {}
    n = 0
    for line in (note or "").splitlines():
        if "CONFLICT in this document" not in line:
            continue
        docs = list(dict.fromkeys(_CONFLICT_DOC_RE.findall(line)))
        pages = [int(p) for p in dict.fromkeys(_CONFLICT_PAGE_RE.findall(line))]
        for d in docs:
            for p in pages:
                if (d, p) in have or p in asks.get(d, []):
                    continue
                if n >= _MAX_PROBE_PAGES:
                    return list(asks.items())
                asks.setdefault(d, []).append(p)
                n += 1
    return list(asks.items())


def _conflict_probe(retriever, note, hits, query=None) -> list:
    """Excerpts for the conflict pages the note named and retrieval missed.

    `query` orders, it never selects (see `probe_pages`): the user's own words
    decide WHICH chunk of an asked-for page comes first, and nothing about
    them can add a page or a document. That is what puts FP153's second
    printing — page 48's fifth chunk — in front of the model; in document
    reading order the same budget returns the section heading instead.

    The budget is spent document by document with every still-queued page
    holding a slot, so the second document of a two-document conflict turn
    cannot be starved by the first one's extra chunks.

    Returns [] on any failure — a retriever too old to have `probe_pages`, an
    index that cannot serve the page, a raise inside it. The supplement is
    never allowed to cost a turn the answer it would have given today.
    """
    try:
        asks = _conflict_probe_asks(note, hits)
        if not asks or retriever is None:
            return []
        have = {(h.doc_id, h.page) for h in (hits or [])}
        queued = sum(len(p) for _, p in asks)
        out, budget = [], _MAX_PROBE_HITS
        for doc, pages in asks:
            queued -= len(pages)
            k = budget - queued          # every queued page keeps its slot
            if k <= 0:
                break
            got = [h for h in retriever.probe_pages(doc, pages, k=k, query=query)
                   if (h.doc_id, h.page) not in have][:budget]
            out += got
            budget -= len(got)
            if budget <= 0:
                break
        return out
    except Exception as e:                     # noqa: BLE001 — never a blocker
        print(f"conflict probe unavailable: {e}", flush=True)
        return []


#: The ask side of `probe_pages(sections=...)`, which sat dormant until the
#: rebuilt default index gained `section_path` on its chunks. The measured
#: shape (release-19 arm 1, l1x-sec-c2-fp126, page_rate 0.0): 'What does
#: section C.2 of FP126 say?' names a section whose heading text shares
#: almost no vocabulary with the question, so similarity search returns
#: cover-page chunks and the C.2 table on p. 40 never surfaces. Asking for
#: the section by its printed id is the cure, with the same honesty rules
#: the conflict probe follows: the fence decides WHEN, `probe_pages`
#: decides WHAT, and a failure returns [] rather than costing the turn.
_SECTION_ASK_RE = re.compile(
    r"\bsections?\s+([A-Ha-h])\s*\.?\s*(\d{1,2}(?:\.\d{1,2})?)(?!\d)")


def _section_probe_asks(question) -> list:
    """[(doc_id, [code, ...])] for a question naming a section AND one document.

    Three refusals keep it a supplement:

    * the word 'section' (the same word in French) must introduce the code —
      a bare 'C.2' or a board code ('approved at B.42') never fires;
    * the question must resolve to exactly ONE registry document: zero means
      there is no document to scope to, two or more is the comparison path's
      territory;
    * at most two codes, in question order — an ask that names a section of
      a document the corpus lacks yields an empty probe, never a search.
    """
    codes = []
    for letter, num in _SECTION_ASK_RE.findall(question or ""):
        code = f"{letter.upper()}.{num}"
        if code not in codes:
            codes.append(code)
    if not codes:
        return []
    try:
        rows = registry.resolve_fps(question or "")[0]
        docs = list(dict.fromkeys(r["doc_id"] for r in rows))
    except Exception:
        return []
    if len(docs) != 1:
        return []
    return [(docs[0], codes[:2])]


def _section_probe(retriever, question, hits, query=None) -> list:
    """Excerpts for the section the question names, in the document it names.

    `probe_pages(sections=...)` reduces each stored path component to its
    printed id, so 'C.2' matches the 'C.2. Financing by Component' heading
    and anything beneath it, and never 'C.20'. Pages this turn already holds
    are not fetched twice, `query` orders chunks within the section and
    selects nothing, and any failure — no retriever, an index without
    section paths, a raise — returns [], because the supplement is never
    allowed to cost a turn the answer it would have given without it.

    Two stages, because a section id only reaches the chunks whose path
    carries it. Measured (release-20, both arms, l1x-sec-c2-fp126): FP126's
    C.2 path sits on a 163-character heading chunk, while the financing
    table's body is a sibling chunk whose path is the table's own header row
    — the VLM markdown promoted it to a heading — so an id-only probe served
    the header and none of the rows. The id resolves the section to its
    PAGES (stage 1, reading order, query-blind); the pages are then fetched
    whole (stage 2), which carries every chunk printed on them regardless of
    what the tracker filed it under.
    """
    try:
        asks = _section_probe_asks(question)
        if not asks or retriever is None:
            return []
        have = {(h.doc_id, h.page) for h in (hits or [])}
        doc, codes = asks[0]
        found = retriever.probe_pages(doc, sections=codes, k=_MAX_PROBE_HITS)
        pages = []
        for h in found:
            if h.page is not None and h.page not in pages:
                pages.append(h.page)
        pages = sorted(pages)[:_MAX_PROBE_PAGES]
        if pages:
            found = retriever.probe_pages(doc, pages=pages, sections=codes,
                                          k=_MAX_PROBE_HITS, query=query)
        got = [h for h in found if (h.doc_id, h.page) not in have]
        return got[:_MAX_PROBE_HITS]
    except Exception as e:                     # noqa: BLE001 — never a blocker
        print(f"section probe unavailable: {e}", flush=True)
        return []


def _context_block(hits, probe=(), section_probe=()) -> str:
    """The excerpt half of the context — one definition, two callers.

    `scripts/eval_answers.py` builds the same block for the release run, and a
    second copy of this f-string there is how a recorded context and a live one
    would come to differ in a byte nobody looked at.

    A probe excerpt is LABELLED rather than scored. It was fetched by page
    because a registry CONFLICT line named that page (or by section id because
    the question asked for that section), so it has no rank in the similarity
    order the other excerpts are printed in, and a '(score 0.44)' at the head
    of a ranked list invites exactly the comparison that number cannot
    support — the cosine `probe_pages` returns ranks chunks WITHIN the
    asked-for pages and nothing else.
    """
    marked = {id(h) for h in (probe or ())}
    sec = {id(h) for h in (section_probe or ())}
    def _tag(h):
        if id(h) in marked:
            return "(registry conflict page — fetched by page, not ranked)"
        if id(h) in sec:
            return "(the asked-for section — fetched by section id, not ranked)"
        return f"(score {h.score:.2f})"
    return "\n\n".join(
        f"[{_doc_label(h.doc_id, h.page)}] " + _tag(h) + f"\n{h.text}"
        for h in hits)


def _invalid_citations(answer: str, hits: list, note_pages: set = frozenset()):
    """Pages cited in the answer that were never retrieved (observed: a
    correct doc cited with an invented 'p. 35'). Returns labels to flag."""
    by_doc = {}
    for d, p in note_pages:
        by_doc.setdefault(d, set()).add(p)
    for h in hits:
        by_doc.setdefault(h.doc_id, set()).add(h.page)
    def _resolve(ref):
        return next((d for d in by_doc if d.startswith(ref[:24])
                     or ref.startswith(d[:24])), None)
    bad = []
    for m in re.finditer(r"\[([0-9]{1,3}_[^\]]+)\]", answer):
        # a bracket may chain citations ("[docA, p. 5; docB, p. 6]"): each
        # page belongs to the NEAREST PRECEDING doc id, not the bracket's
        # first — attributing all pages to the first doc flagged valid
        # citations as invented (observed live on gpt-5.2 output)
        cur = None
        for part in re.finditer(r"([0-9]{1,3}_[\w.\-]+)|\bpp?\.?\s*(\d{1,3})\b",
                                m.group(1)):
            if part.group(1):
                cur = _resolve(part.group(1))
            elif cur is not None:   # unresolved = registry/cover-page cite
                pg = int(part.group(2))
                if pg not in by_doc[cur]:
                    bad.append(f"{cur[:34]}… p.{pg}")
    return bad


# Corpus span, derived from the board table: the ceiling/floor a half-open
# range phrasing ("after 2023") expands to.
_CORPUS_LO, _CORPUS_HI = min(BOARD_YEARS.values()), max(BOARD_YEARS.values())

_YEAR_RE = re.compile(r"\b(20[12]\d)\b")
# a range word within a short gap of the year ("approved after 2023",
# "depuis 2023", "after the year 2023"); the gap is bounded so an unrelated
# "in 2019, after the board met" cannot turn into a range
_OPEN_RANGE_RE = re.compile(
    r"\b(after|since|from|before|until|apr[eè]s|depuis|avant|[àa] partir de)\b"
    r"[^\d]{0,12}?(20[12]\d)\b", re.I)
# closed range: "from 2019 to 2021", "2019-2021", "de 2019 à 2021". 'and'/'et'
# are deliberately absent — "2019 and 2021" names two years, not a span.
_CLOSED_RANGE_RE = re.compile(
    r"\b(20[12]\d)\s*(?:-|–|—|to|through|jusqu'?[àa]|[àa])\s*(20[12]\d)\b", re.I)
# 'from 2020' alone reads as "of 2020" in both languages (and is the exact
# phrasing of the year-aggregate questions this note exists for), so 'from'
# only opens a range when the message says so explicitly.
_ONWARD_RE = re.compile(r"\bonwards?\b|\bor later\b|\bto date\b|\bto now\b", re.I)


def _scan_years(question: str):
    """(years asked about, phrases that fall entirely outside the corpus).

    Literal tokens are exact; a range word attached to a year opens the set
    (corpus span 2015-2025, from BOARD_YEARS):

        after / après Y      -> Y+1 .. 2025     (bound excluded)
        since / depuis Y     -> Y   .. 2025     (bound included)
        à partir de Y        -> Y   .. 2025
        from Y (+ 'onwards') -> Y   .. 2025
        before / avant Y     -> 2015 .. Y-1     (bound excluded)
        until Y              -> 2015 .. Y       (bound included)
        Y to / à / - Y2      -> Y .. Y2         (both bounds included)

    A year consumed as a range bound is not re-added as a literal, which is
    what keeps 'after 2023' off 2023 itself. Without this, "approved after
    2023" fired for 2023 only and the note never mentioned FP274 (2025).

    A range that lands outside the corpus ('before 2015') yields NO years —
    listing 2015's proposals under a pre-2015 question is a wrong answer
    stated with confidence. It returns a phrase instead, which _year_assist
    turns into a definitive out-of-range note.
    """
    years: set = set()
    consumed: list = []          # spans of year tokens a range already covers
    outside: list = []           # phrasings with nothing in the corpus

    for m in _CLOSED_RANGE_RE.finditer(question):
        a, b = int(m.group(1)), int(m.group(2))
        if a <= b:
            got = {y for y in range(a, b + 1) if _CORPUS_LO <= y <= _CORPUS_HI}
            years |= got
            consumed += [m.span(1), m.span(2)]
            if not got:
                outside.append(f"between {a} and {b}")

    onward = bool(_ONWARD_RE.search(question))
    for m in _OPEN_RANGE_RE.finditer(question):
        if any(s <= m.start(2) < e for s, e in consumed):
            continue                                  # already a closed bound
        word = re.sub(r"\s+", " ", m.group(1).lower())
        y = int(m.group(2))
        if word == "from" and not onward:
            continue                                  # "proposals from 2020"
        if word in ("after", "après", "apres"):
            rng, phrase = range(y + 1, _CORPUS_HI + 1), f"after {y}"
        elif word in ("before", "avant"):
            rng, phrase = range(_CORPUS_LO, y), f"before {y}"
        elif word == "until":
            rng, phrase = range(_CORPUS_LO, y + 1), f"up to {y}"
        else:                                         # since / depuis / from / à partir de
            rng, phrase = range(y, _CORPUS_HI + 1), f"from {y} onwards"
        got = {yy for yy in rng if _CORPUS_LO <= yy <= _CORPUS_HI}
        years |= got
        consumed.append(m.span(2))
        if not got and phrase not in outside:
            outside.append(phrase)

    for m in _YEAR_RE.finditer(question):
        if not any(s <= m.start() < e for s, e in consumed):
            years.add(int(m.group(1)))
    return years, outside


def _year_scope(question: str) -> set:
    """The years a question asks about (see _scan_years)."""
    return _scan_years(question)[0]


def _outside_corpus_note(outside: list) -> str:
    """Definitive 'the corpus has nothing there' for a range with no overlap."""
    lo, hi = min(BOARD_YEARS), max(BOARD_YEARS)
    return (f"Note (computed): this corpus covers board meetings B.{lo} "
            f"({BOARD_YEARS[lo]}) through B.{hi} ({BOARD_YEARS[hi]}) completely "
            f"and contains no proposals {', '.join(outside)}. State this "
            f"definitively.")


# One instruction, carried by the NOTE rather than the prompt: a note fires
# per trigger, the system prompt is paid for on every turn. Measured (F11/P4):
# asked to total the 2020 year note the model returned $29.0B against a truth
# near $1.36B (21x), and an unprompted 2020-vs-2021 comparison came out
# backwards — the note prints one figure per proposal, in whatever currency
# and unit that proposal printed, and nothing behind it can add them up.
#
# The refusal is still the right answer for the PRINTED strings. It was the
# wrong answer for the QUESTION: "did 2020 or 2021 get more?" has a direction,
# the corpus knows it, and a flat refusal leaves the user with the 21x guess
# they came in with. So the note now computes the one total that is defensible
# — same currency, unambiguous prints only — and this rule licenses THAT
# figure and nothing else. The licence and the prohibition share a sentence on
# purpose: the F11 protection is that the model never does the arithmetic, and
# a rule that grants an exception in one breath and forbids self-summation two
# sentences later is a rule with a gap in the middle of it.
_NO_SUM_RULE = (
    "Amounts are quoted exactly as each proposal prints them, in mixed "
    "currencies and sometimes ambiguous units, so they MUST NOT be summed, "
    "totalled or averaged: answer a request for a year-wide or corpus-wide "
    "total by refusing the sum and giving the per-proposal figures instead. "
    "The single exception is a 'Computed total' sentence in this note: quote "
    "that total with the coverage and the exclusions it states, and even then "
    "do NOT add, subtract, extend or convert any figure yourself — a total "
    "this note does not print is a total the answer does not have.")


# The comparative the totals were built for (P4 asked "2020 or 2021?" and got
# the direction backwards). _NO_SUM_RULE licenses QUOTING a computed total and
# forbids arithmetic on it; it never says whether two totals may be RANKED.
# release-10's l2x-xyear got the direction right on a single sample by reading
# "do NOT add, subtract, extend or convert" as not covering "which is larger"
# — a correct inference, not an instruction, and one run is not a measurement.
# This sentence states the licence and leaves the prohibition exactly where it
# was: two printed totals may be compared, while their difference, ratio and
# sum are figures the note does not print. It ships ONLY when the note really
# prints two totals, so one-year notes and the coverage note keep their text
# byte for byte.
_COMPARE_RULE = (
    "This note prints a 'Computed total' for two years: those two totals MAY "
    "be compared as printed — say which year is larger and quote both totals "
    "with the coverage each states — but their difference, their ratio and "
    "their sum are figures this note does not print, so the answer does not "
    "have them either.")


def _v2_money(row: dict):
    """The v2 canonical `gcf_funding_requested` candidate for a registry row.

    v1's `gcf_financing` is the string the cover page was OCR'd into and
    nothing re-read it. v2 parses the same field out of the template section,
    and publishes `value=None` for a print whose mantissa and scale word
    cannot both be true ("28,654 million USD" — FP153, the single print that
    made a summed 2020 note 21x too high). Preferring v2 is what puts that
    mark in front of the model; the v1 text stays as the fallback for the
    rows v2 has no canonical figure for (FP150, FP142).

    Looked up by document id, which is exact — the FP number is a fallback
    for a row without one. Never raises: an unreadable v2 file means the note
    falls back to v1, not that the year note disappears.
    """
    try:
        c = registry.canonical(row.get("doc_id") or row.get("fp"),
                               "gcf_funding_requested")
    except Exception:
        return None
    return c if c and c.get("raw") else None


def _money(row: dict) -> str:
    """' (18.5 M USD GCF)' — the GCF request as the document prints it.

    The raw string is printed as printed, never a reformatted float: the
    print is what the answer has to be able to cite back. A print v2 could
    not parse is quoted and flagged, in the same words `registry._money_bit`
    uses for it, so an unusable figure reads as unusable here too.
    """
    c = _v2_money(row)
    if c:
        return (f" ({c['raw']} GCF)" if c.get("value") is not None else
                f' ("{c["raw"]}" GCF, unit as printed is ambiguous)')
    # Before the v1 fallback, consult the ratified corrections: five documents
    # have no v2 canonical AND an adjudicated-wrong v1 string (FP100, FP067,
    # FP054, FP245, FP240) — a refuted figure must not ride into a note.
    rec = registry._ratified_top(row.get("doc_id"), "gcf_financing")
    if rec is not None:
        to = rec.get("to")
        return f" ({to} GCF)" if to else ""
    return f" ({row['gcf_financing']} GCF)" if row.get("gcf_financing") else ""


def _span_text(labels: list, prefix: str = "") -> str:
    """'2015, 2016' but '2015–2025' once a contiguous run gets long."""
    if len(labels) > 3 and labels[-1] - labels[0] == len(labels) - 1:
        return f"{prefix}{labels[0]}–{prefix}{labels[-1]}"
    return ", ".join(f"{prefix}{v}" for v in labels)


def _boards_in(year: int) -> str:
    """'B.28, B.29, B.30' — every board meeting BOARD_YEARS puts in `year`.

    RULING 10 (2026-08-26): board→year facts are evidence, not prompt
    knowledge, so the note prints them and the verifier can read them.

    EVERY board, spelled out, and NOT `_span_text`: the ruling's own example
    writes 'boards B.28–B.30', and a dash prints B.28 and B.30 while hiding
    B.29 — which is one of the two claims this ruling exists to make
    verifiable (`verify._check_years` matches board TOKENS against the
    evidence text and nothing else). Four boards is the corpus maximum for a
    year, so the full list costs ~24 characters.
    """
    return ", ".join(f"B.{b}" for b, yy in sorted(BOARD_YEARS.items())
                     if yy == year)


# --- the one total the corpus can defend (P4/F11) --------------------------
#
# P4 asked which of 2020 and 2021 requested more and got the direction WRONG;
# F11 asked for the sums outright and got 21x/35x. Both failed on the same
# missing piece: the year note prints v1/v2 raw strings and no layer behind it
# holds a number. registry v2 does — `canonical()['value']` is a float parsed
# from the template section, and it is None exactly when the print cannot be
# trusted ("28,654 million USD"). So a total is computable here, from the
# floats, never from the strings — but only over prints that are in ONE
# currency and unambiguous. Everything left out is NAMED: a total whose
# coverage is invisible is the same defect as the truncated country list of
# §7.1, which said "five" and meant "five of 44".
#
# Two years is the cap. The shape this answers is "year A vs year B"; a third
# year's listing plus a third exclusion list stops being a note and starts
# being a report, and wide spans already drop to per-year counts (no money at
# all), so there is nothing there to total.
_TOTAL_YEARS_MAX = 2


def _usd_amount(value: float) -> str:
    """'USD 1,157,208,843.80': grouped, no '.00' tail on a whole figure."""
    text = f"{value:,.2f}"
    return "USD " + (text[:-3] if text.endswith(".00") else text)


def _fp_list(fps) -> str:
    return ", ".join(f"FP{n}" for n in sorted(fps))


def _usd_total(rows: list):
    """(total, [fp included], {reason: [fp excluded]}) over the v2 floats.

    A row enters the sum only with a v2 CANONICAL fact that parsed to a float
    in USD. The four ways out are kept apart because they mean different
    things to a reader: another currency is a real figure this total cannot
    hold, an ambiguous print is a figure nothing can trust, an unnormalised
    one is printed in the listing above but never re-read, and 'no figure' is
    a silence. Never the printed string, in any branch.
    """
    total, included = 0.0, []
    excluded = {"currency": [], "ambiguous": [], "unnormalised": [],
                "silent": []}
    for row in rows:
        fp = row.get("fp")
        cand = _v2_money(row)
        if not cand:
            # `_money` still prints the v1 string for these, so say which
            excluded["unnormalised" if row.get("gcf_financing")
                     else "silent"].append(fp)
        elif cand.get("value") is None:
            excluded["ambiguous"].append(fp)
        elif (cand.get("currency") or "").upper() != "USD":
            currency = (cand.get("currency") or "").upper()
            excluded["currency"].append((fp, currency or "currency not stated"))
        else:
            total += float(cand["value"])
            included.append(fp)
    return total, included, excluded


def _year_total_line(year: int, rows: list):
    """'Computed total for 2020 …', or None when nothing is summable.

    Carries no document id and no '(p.N' pointer BY CONSTRUCTION: it names
    proposals as FP numbers, exactly as the listing above it does, so the
    sentence publishes no note-page scope for the verifier to credit a
    citation against (mirrors the _NO_SUM_RULE safety test).
    """
    total, included, excluded = _usd_total(rows)
    if not included:
        return None
    bits = []
    by_currency: dict = {}
    for fp, cur in excluded["currency"]:
        by_currency.setdefault(cur, []).append(fp)
    for cur, fps in sorted(by_currency.items()):
        bits.append(f"{_fp_list(fps)} ({cur})")
    if excluded["ambiguous"]:
        bits.append(f"{_fp_list(excluded['ambiguous'])} "
                    f"(unit as printed is ambiguous)")
    if excluded["unnormalised"]:
        bits.append(f"{_fp_list(excluded['unnormalised'])} "
                    f"(figure printed above but not normalised)")
    if excluded["silent"]:
        n = len(excluded["silent"])
        bits.append(f"{n} proposal{'s' if n != 1 else ''} stating no figure")
    tail = ("; excluded from this total: " + ", ".join(bits)
            + " — the figures listed above for them stand as printed."
            if bits else ".")
    return (f"Computed total for {year} (computed by the system from the "
            f"registry's normalised values, NOT by adding the strings above): "
            f"{len(included)} of the {len(rows)} proposals state their GCF "
            f"request as an unambiguous USD figure, and those {len(included)} "
            f"total {_usd_amount(total)}{tail}")


def _year_assist(question: str, hits: list):
    """Code-side year matching: if the question names a year, sort matching
    excerpts first and emit a note computed from the REGISTRY, which is
    complete for the corpus — retrieval never surfaces all of a year's
    proposals, so excerpt-scoped notes made the model refuse year
    aggregates ('which proposals were approved in 2020?').

    Range phrasings ('after 2023') expand across the corpus span; past three
    years the per-FP listing would swamp the context, so wide spans get
    per-year counts and FP ranges instead. A range with no overlap at all
    ('before 2015') gets a definitive out-of-range note, never a listing of
    the nearest year.

    One or two years also get a COMPUTED same-currency total per year (see
    _year_total_line): the printed figures still may not be summed by the
    model, but the question "2020 or 2021?" has an answer and the registry
    holds the floats to compute it.
    """
    years, outside = _scan_years(question)
    if not years:
        return hits, (_outside_corpus_note(outside) if outside else None)
    matched = [h for h in hits if year_of(h.doc_id) in years]
    rest = [h for h in hits if h not in matched]
    ys = _span_text(sorted(years))
    try:
        registry.load()
    except Exception:
        # registry unavailable -> the old excerpt-scoped note; never claim
        # "no proposals that year" on the strength of a missing file
        boards = _span_text(sorted(b for b, y in BOARD_YEARS.items() if y in years),
                            prefix="B.")
        note = (f"Note (computed from document ids): "
                + (("excerpts dated " + ys + ": "
                    + "; ".join(_doc_label(h.doc_id, h.page) for h in matched)
                    + f". The corpus may contain more documents from {ys}.")
                   if matched else
                   f"none of the retrieved excerpts are from {ys} (boards "
                   f"{boards}). Answer what the excerpts support and state "
                   f"this limit."))
        if outside:      # e.g. "before 2015 or in 2020": half is out of range
            note += " " + _outside_corpus_note(outside)
        return matched + rest, note
    detailed = len(years) <= 3
    lines, totals = [], 0
    for y in sorted(years):
        rows = [r for r in registry.by_year(y) if r.get("fp")]
        # RULING 10: the boards of the year this line lists, printed on the
        # line that lists it. The empty arm below has always done this; the
        # populated arms did not, so 'B.28 (2021)' was derivable from the
        # prompt's board table and printed by no evidence.
        #
        # AT THE END OF THE LINE, not in the ruling's illustrative
        # '28 proposals (boards B.28–B.30):' slot, and both halves of that are
        # measured. The dash form prints B.28 and B.30 and hides B.29, which is
        # one of the two claims this ruling exists to verify. And inserting ~26
        # characters between the year and the line's first money figure flips
        # a recorded verdict that has nothing to do with boards:
        # `verify.iter_amounts` un-skips a bare 4-digit year when a currency
        # token sits within 40 characters after it, so 'l2x-xyear's cited
        # '**2020**' stopped matching the note's own '2020' the moment the
        # first 'US$' moved out of that window. The fact the ruling asks for is
        # on the line either way; the placement that leaves every other
        # recorded verdict untouched is the one to ship.
        boards = _boards_in(y)
        at = f" Boards in {y}: {boards}." if boards else ""
        if rows and detailed:
            fps = "; ".join(f"FP{r['fp']}{_money(r)}" for r in rows)
            lines.append(f"{y} — {len(rows)} proposals: {fps}.{at}")
            if len(years) <= _TOTAL_YEARS_MAX:
                total_line = _year_total_line(y, rows)
                if total_line:
                    lines.append(total_line)
                    totals += 1
        elif rows:
            # 'FPa–FPb' claims every number between a and b belongs to this
            # year, which is false for most years (2023 spans FP86..FP224 with
            # 110 outsiders) — and this note is labelled authoritative. Emit
            # the range ONLY when the numbers really are consecutive.
            fps = sorted(r["fp"] for r in rows)
            if len(fps) == 1:
                span = f": FP{fps[0]}"
            elif fps == list(range(fps[0], fps[-1] + 1)):
                span = f": FP{fps[0]}–FP{fps[-1]}"
            else:
                span = f" (FP{fps[0]} … FP{fps[-1]}, not contiguous)"
            lines.append(f"{y} — {len(rows)} proposals{span}.{at}")
        else:
            lines.append(f"{y} — no registered proposals"
                         + (f" (boards {boards})." if boards else
                            " (no board meeting that year in this corpus)."))
    # ONE LINE PER RETRIEVED EXCERPT, and that is the half of ruling 10 the
    # verifier actually reads. `verify.build_evidence` walks a note block LINE
    # BY LINE and files each line under the FIRST document id it names; as one
    # long line, this tail filed the whole 2021 note — 28 proposals, a computed
    # total and every board token — under whichever document happened to be
    # matched first, and under no other. That is why release-7's 'B.30 (2021)'
    # verified (its document was first) while 'B.28 (2021)' and 'B.29 (2021)',
    # citing pages of documents named later in the same sentence, came back
    # unsupported. Split, each label lands under its own document, so the board
    # and year of the document a claim cites are held as evidence for THAT
    # document — and the accidental attribution of the whole note to one
    # document is gone with it.
    if matched:
        tail = ([f"Retrieved excerpts dated {ys}:"]
                + [_doc_label(h.doc_id, h.page) for h in matched])
    else:
        tail = [f"None of the retrieved excerpts are dated {ys}; answer "
                f"year-level questions from the registry list above and say so."]
    rules = [_NO_SUM_RULE] + ([_COMPARE_RULE] if totals >= 2 else [])
    note = ("Note (computed from the corpus registry, which is complete — "
            "this list is authoritative, unlike the excerpts):\n"
            + "\n".join(lines + tail + rules))
    if outside:          # e.g. "before 2015 or in 2020": half is out of range
        note += " " + _outside_corpus_note(outside)
    return matched + rest, note


# 'B.<n>' is written for two different things in this corpus. Board meetings
# are B.11-B.43 (BOARD_YEARS); the funding-proposal TEMPLATE numbers its own
# headings in the same shape, and those run low — every B-section registry v2
# ever reads a figure out of is B.2(a) or B.2(b), alongside A.x and C.1(x).
# So a number at or below _TEMPLATE_SECTION_MAX is ambiguous, and 'What does
# section B.3 of FP172 say?' was answered with 'B.3 is not in this corpus …
# State this definitively.' — a false authoritative note about a real section
# of a real document (H5/P6). Above 10 nothing is ambiguous: the template has
# no such heading, so the token is a board code and an out-of-range one is
# still told definitively (B.44, B.45).
_TEMPLATE_SECTION_MAX = 10
_SECTION_WORD_RE = re.compile(r"§|\bsections?\b|\brubriques?\b", re.I)
# An explicit board frame: the words that make a low 'B.<n>' a claim about a
# meeting rather than a heading ('approved at B.3', 'GCF board B.3',
# 'réunion B.3'). 'approv'/'approuv' covers the 'B.x approval' phrasing.
_BOARD_WORD_RE = re.compile(
    r"\bboards?\b|\bmeetings?\b|\bsessions?\b|\bconseil\b|"
    r"\br[ée]unions?\b|\bapprov|\bapprouv", re.I)
# group(2) catches the paragraph letter a template heading carries and a
# board code never does: 'B.2(a)'.
_BOARD_TOKEN_RE = re.compile(r"\bb\.?\s?(\d{1,2})\b(\s*\([a-z]\))?", re.I)


def _board_range_note(question: str):
    """A board meeting outside the corpus range deserves a definitive 'no',
    not an excerpt-scoped shrug ('B.44?' has no year token, so _year_assist
    never fires for it).

    The rule, in order, for each 'B.<n>' the question prints:

      * n in BOARD_YEARS            -> in range, no note (unchanged).
      * n > 10                      -> a board code; out of range, so the
                                       definitive note fires (unchanged).
      * n <= 10 and the question says 'section'/'§'/'rubrique', or the token
        itself carries a paragraph letter ('B.2(a)')
                                    -> a template heading. NEVER a board.
      * n <= 10 with an explicit board frame ('board', 'meeting', 'session',
        'conseil', 'réunion', 'approv*')
                                    -> a claimed board meeting, and B.1-B.10
                                       genuinely are outside this corpus, so
                                       the definitive note is correct: 'What
                                       was approved at B.3?' still gets it.
      * n <= 10 unframed            -> ambiguous, and the cost is asymmetric.
                                       A missing note loses a definitive
                                       phrasing; a wrong note tells the model
                                       to deny a section it can see. No note.
    """
    lo, hi = min(BOARD_YEARS), max(BOARD_YEARS)
    q = question or ""
    section_ctx = bool(_SECTION_WORD_RE.search(q))
    board_ctx = bool(_BOARD_WORD_RE.search(q))
    codes = set()
    for m in _BOARD_TOKEN_RE.finditer(q):
        n = int(m.group(1))
        if n in BOARD_YEARS:
            continue                       # in range: the corpus has it
        if n <= _TEMPLATE_SECTION_MAX and (section_ctx or m.group(2)
                                           or not board_ctx):
            continue                       # a heading, or too ambiguous to deny
        codes.add(n)
    if not codes:
        return None
    out = ", ".join(f"B.{n}" for n in sorted(codes))
    return (f"Note (computed): {out} "
            f"{'is' if len(codes) == 1 else 'are'} not in this corpus, which "
            f"covers board meetings B.{lo} ({BOARD_YEARS[lo]}) through "
            f"B.{hi} ({BOARD_YEARS[hi]}) completely. State this definitively.")


_COVERAGE_ASK_RE = re.compile(
    r"which board meetings|what board meetings|which years|what years|"
    r"how many (?:[\w-]+ ){0,2}(?:proposals|documents)|"
    r"combien de (?:[\w-]+ ){0,2}(?:propositions|documents)|"
    r"quelles (?:r[ée]unions|ann[ée]es)", re.I)
_CORPUS_TOKEN_RE = re.compile(r"corpus|collection|dataset|base documentaire",
                              re.I)

# THE THEMATIC FENCE (H12, probe P7). `_COVERAGE_ASK_RE` matches "how many
# <up to two words> proposals", which is exactly the shape of "how many
# AGRICULTURE proposals are in the corpus?" — and of "how many proposals in
# the corpus CONCERN AGRICULTURE?", where the restriction sits after the noun
# instead of before it. Both got a note that holds per-year counts and ends
# "Answer corpus-coverage questions from this note", handed to a question the
# registry has no field for. P7 measured the model coping (it said it has no
# theme field and refused the count), so this is hygiene — but a note labelled
# authoritative in front of a question it cannot answer is the one failure
# shape this system is built not to have.
#
# The fence is an ALLOWLIST, in the house style, and it is deliberately
# asymmetric: the vocabulary below is the meta-language of the corpus itself —
# counting words, the corpus's own nouns, the copulas and articles that join
# them, in both languages. Any other content word means the question restricts
# the count to a subset (a theme, an entity, a country, a status), and the note
# does not know that subset. Over-fencing costs a definitive phrasing and
# leaves the excerpt-scoped answer that predates the note; under-fencing puts
# a false authority in front of the model. That is why an unrecognised word
# silences the note rather than being ignored.
_COVERAGE_VOCAB = frozenset("""
how many much what which whats does do did is are was were be been there
this that these those the an and or of in on at to for with by from within
inside across per each its it their they all total totals in-total overall
altogether exactly currently now today please tell me us give show list
corpus collection dataset database base documentaire jeu donnees data
document documents doc docs proposal proposals proposition propositions
funding financement finance financing fund funds gcf green climate gsf
board boards meeting meetings session sessions conseil reunion reunions
year years annee annees calendar
cover covers covered coverage contain contains contained contents include
includes included hold holds held have has spans span comprise comprises
comprised consist consists made up size number count counted counts
represented present available indexed stored listed sampled range scope
whole entire full still only about combien quel quelle quels quelles
most least fewest fewer more less largest smallest biggest highest lowest
top earliest latest first second third last plus moins plupart grand grande
hi hello hey good morning afternoon evening thanks thank you your yours
could would can may might want wanted wondering need needed like know tell
say question questions bonjour salut merci svp sil plait je jai voudrais
aimerais savoir pouvez peux dire dis dites vous nous rapide quick just
simply also again actually roughly approximately around exact au juste
petit petite premier premiere dernier derniere haut bas eleve eleves
ce cet cette ces le la les de du des au aux en dans sur par pour et
contient contiennent couvre couvrent compte comptent comprend comprennent
contenus contenues comporte comportent figure figurent sont est ete etre
il elle ils elles on ny na dont total totale totaux totales nombre
""".split())


def _deaccent(text: str) -> str:
    """'réunions' -> 'reunions' — one folding, shared by the fence's vocabulary
    and the question it reads, so an accent never decides whether a note
    fires."""
    return "".join(c for c in unicodedata.normalize("NFKD", text or "")
                   if not unicodedata.combining(c))


def _off_vocabulary(question: str) -> list:
    """Words a coverage question carries that are not the corpus's own.

    Single characters are dropped before the comparison: they are French
    elisions and hyphen glue ("l'agriculture", "couvre-t-il"), never the
    content word that restricts a count.
    """
    words = re.findall(r"[^\W\d_]{2,}", _deaccent(question or "").lower())
    return [w for w in words if w not in _COVERAGE_VOCAB]


def _corpus_coverage_note(question: str):
    """Corpus-coverage questions name no year and no board code, so neither
    _year_assist nor _board_range_note fires — and the excerpts rightly
    cannot answer a corpus-wide question (the prompt forbids stating
    corpus-wide facts from a retrieved sample). The registry can: it is
    complete. Fires only when the question asks about the corpus's OWN
    coverage or size; a year in the question hands off to the year note and
    a board code to the board paths, so no turn gets two authorities.
    (Measured: release-4 agg-corpus-boards answered with portfolio-company
    'board meetings' found in excerpts, 0.43, because no note fired.)"""
    q = question or ""
    if not (_CORPUS_TOKEN_RE.search(q) and _COVERAGE_ASK_RE.search(q)):
        return None
    if re.search(r"\b(?:19|20)\d\d\b", q):
        return None
    if re.search(r"\bb\.?\s?\d{1,2}\b", q, re.I):
        return None
    off = _off_vocabulary(q)
    if off:
        return None      # a restricted count: see _COVERAGE_VOCAB (H12/P7)
    lo, hi = min(BOARD_YEARS), max(BOARD_YEARS)
    counts = {y: len([r for r in registry.by_year(y) if r.get("fp")])
              for y in sorted(set(BOARD_YEARS.values()))}
    total = sum(counts.values())
    # RULING 10 again: the boards of every year this note counts, on the line
    # that counts it — the same fact the year note now prints, for the note
    # that fires when no year is named at all.
    per_year = "; ".join(f"{y}: {n} ({_boards_in(y)})"
                         for y, n in sorted(counts.items()))
    return ("Note (computed from the corpus registry, which is complete — "
            "authoritative, unlike the excerpts): this corpus covers Green "
            f"Climate Fund board meetings B.{lo} ({BOARD_YEARS[lo]}) through "
            f"B.{hi} ({BOARD_YEARS[hi]}) completely — {total} funding-proposal "
            f"documents, one per proposal. Proposals per year: {per_year}. "
            "Answer corpus-coverage questions from this note. " + _NO_SUM_RULE)


def _fp_of(text: str):
    """First FP number in a string ('...package-fp214' -> '214').

    Zero-padding is stripped by the shared pattern ('fp086' -> '86'), and the
    trailing boundary keeps 'fp2023' from reading as FP202 — a truncated match
    used to resolve a doc filter to a real but WRONG document, which is worse
    than no filter at all.
    """
    m = _FP_RE.search(text or "")
    return m.group(1) if m else None


def _cited_docs(history: list) -> list:
    """Document ids cited so far, oldest first.

    The conductor sees each history message truncated to 1200 chars, which
    cuts most citations loose; this list is appended whole so its doc tags
    can point at real documents. Uses the same pattern as the other two
    citation readers — the narrower '\\d+_gcf-' shape silently missed corpus
    ids like '72_GCF_B.35_02_Add.05_Funding_proposal_package_for_FP203'.
    """
    cited: list = []
    for m in history:
        for d in re.findall(r"\[([0-9]{1,3}_[\w.\-]+)", m["content"]):
            if d not in cited:
                cited.append(d)
    return cited


def _resolve_doc(tag: str, history_docs: list):
    """The real corpus id a decomposer doc tag stands for, or None.

    Tags are fabricated as often as they are copied — '02_fp214' is a
    history prefix mashed onto a message id and matches no document at all.
    Retriever._doc_match compares by equality/prefix/substring, so an
    invented tag quietly degrades to unscoped search; the ids actually cited
    in the conversation are what pin a mangled tag back to a document.
    """
    t = (tag or "").lower()
    if len(t) < 6:              # too short to identify anything on its own
        return None
    for hd in history_docs:
        h = hd.lower()
        if h == t or h.startswith(t) or t in h:
            return hd
    fp = _fp_of(t)
    if fp:
        same = [hd for hd in history_docs if _fp_of(hd) == fp]
        if len(same) == 1:
            return same[0]
    return None


def _with_ids(query: str, fps: list) -> str:
    """Append the 'FPnnn' tokens a sub-query is missing."""
    have = set(_FP_RE.findall(query.lower()))
    add = [f"FP{n}" for n in fps if n not in have][:3]
    return (query + " " + " ".join(add)).strip() if add else query


def _prescope_single_fp(items: list, msg_text: str) -> list:
    """Doc-scope a lone sub-query whose message names exactly ONE FP number.

    The conductor emits no doc tag for a cold single-topic question
    (CONDUCTOR_PROMPT rules 1 and 4), so 'What is FP152's GCF financing?' ran
    as an unscoped semantic query and drew its evidence pages from
    neighbouring B.27 documents. The FP number the user typed is a
    deterministic filter, and every piece of machinery to honour it already
    exists: _rescope_items keeps a tag the message itself names, and
    _resolve_doc_tags maps the plain 'fpNNN' token onto the authoritative
    stem (B.27 filenames carry no FP number). Measured over the 66-case
    answer set this recovers EVERY retrieval miss the unscoped baseline had —
    r@5 88% -> 96%, evidence-page hit 81% -> 94%, no regressions
    (data/eval/answers_baseline_retrieval-scoped-ab.jsonl).

    The conditions are narrow on purpose; each one is a way to scope wrongly:

    * exactly one FP in the message — several means a comparison, which the
      per-document fan-out already scopes one document at a time;
    * exactly one search query — a fan-out's other legs are about something
      else ('typical adaptation projects'), and pinning them to the single FP
      named would answer them from the wrong document;
    * no board code in the message — 'Are FP218 and GCF/B.42/02/Add.16 the
      same?' names two documents, only one of them as an FP token;
    * the item carries no tag of its own — the conductor's tag, and the
      guards' verdict on it, always win;
    * the FP resolves in the registry — an id that exists nowhere (fp999)
      stays untagged, so the weak-signal path still fires on it instead of a
      hard filter that matches nothing.
    """
    if len(items) != 1 or items[0].get("doc"):
        return items
    ids = set(_FP_RE.findall((msg_text or "").lower()))
    if len(ids) != 1 or _BOARD_CODE_RE.search(msg_text or ""):
        return items
    # A rule-4 conductor rewrite may resolve a pronoun into a SECOND id the
    # message never typed ('compare to it' -> 'FP214 ... compared with FP274').
    # Hard-scoping that query to the message's lone FP starves the partner
    # document, so bail whenever the sub-query names ids beyond the message's.
    q_ids = set(_FP_RE.findall((items[0].get("q") or "").lower()))
    if not q_ids <= ids:
        return items
    try:
        resolved, missing = registry.resolve_fps(msg_text or "")
    except Exception:
        return items                 # registry unavailable: leave it unscoped
    if missing or len(resolved) != 1:
        return items
    items[0]["doc"] = "fp" + next(iter(ids))
    return items


def _rescope_items(items: list, msg_text: str, history_docs: list) -> list:
    """Deterministic guards against decomposer rewrite contamination.

    The decomposer tags each sub-query with the document it is about, and
    gets that wrong two ways: fabricated ids ('02_fp214'), and one
    document's tag copied onto every sub-query ('FP214 ... FP265?' answered
    entirely from the previous turn's FP274).

    Nulling every tag as soon as the message names its own FP numbers stops
    the contamination but destroys correct scoping too: the FP274 half of
    "how does FP214's financing compare to it?" degrades to the bare phrase
    "total financing", which carries no identifier for the retriever's
    two-stage routing to latch onto, so round-robin merges global noise and
    that half of the comparison is answered from unrelated documents — the
    starvation the per-document fan-out exists to prevent. So instead:

    * a tag whose FP number the MESSAGE names is kept, rewritten to the
      plain 'fpNNN' token (or the full cited id when history pins one);
      both resolve through _doc_match's substring compare;
    * a tag pinned to a document cited earlier is kept only for the fan-out
      slots the message's own ids do not account for — that is the "it" of a
      comparison, never a licence for history to outvote explicit ids;
    * any other tag is stripped, and the message's identifiers are appended
      to its sub-query so identifier routing engages on the document the
      user actually named instead of running an unscoped generic phrase.

    With no ids in the message nothing is stripped from a fan-out (the
    "compare those two" case arrives here with history-derived tags by
    design), though fabricated tags are still repaired; a lone query still
    keeps a tag only if the message names that document.

    An untagged lone query whose message names exactly one FP is scoped to it
    first (_prescope_single_fp); everything below then treats that tag like
    any other tag the message names.
    """
    items = _prescope_single_fp(items, msg_text)
    msg_l = (msg_text or "").lower()
    msg_ids = set(_FP_RE.findall(msg_l))
    hist = [d for d in (history_docs or []) if d]
    # sub-queries the message's own ids cannot account for: those, and only
    # those, may be filled from the documents cited earlier
    slack = max(0, len(items) - len(msg_ids))
    for item in items:
        tag = str(item.get("doc") or "")
        if not tag:
            continue
        if msg_ids:
            fp = _fp_of(tag)
            if fp and fp in msg_ids:
                item["doc"] = _resolve_doc(tag, hist) or f"fp{fp}"
                continue
            pinned = _resolve_doc(tag, hist)
            if pinned and slack > 0:
                item["doc"] = pinned
                slack -= 1
                continue
            item["doc"] = None
            item["q"] = _with_ids(item["q"], sorted(msg_ids))
        elif len(items) == 1:
            # a lone query stays doc-scoped only if the message itself names
            # that document (board code prefix or FP token)
            fp = _fp_of(tag)
            if tag.lower()[:20] not in msg_l and not (fp and "fp" + fp in msg_l):
                item["doc"] = None
        else:
            item["doc"] = _resolve_doc(tag, hist) or tag
    return items


def _registry_doc(tag: str):
    """The authoritative corpus stem a doc tag stands for, or the tag itself.

    B.27-era filenames carry no FP number: FP152's document is
    '123_gcf-b27-02-add12'. A tag of 'fp152' — whether the conductor wrote it
    or _rescope_items rewrote it to that plain token — therefore matches no
    document in Retriever._doc_match, and the scoped search silently degrades
    to an unscoped generic phrase (observed live: a question about FP152's
    board conditions answered from FP242 pages). The registry maps FP numbers
    to stems, so resolve through it.

    A tag that already IS a corpus id is left alone: the guard pinned it to a
    document cited in the conversation, and swapping that for the FP-numbered
    package doc would change which document the turn is scoped to. The
    registry is an enhancement — unavailable, the tag survives untouched.
    """
    if not tag:
        return tag
    try:
        docs = registry.load()
        if tag in docs:
            return tag
        low = tag.lower()
        if any(k.lower() == low for k in docs):
            return tag
        fp = _fp_of(tag)
        row = registry.by_fp(int(fp)) if fp else None
        return (row.get("doc_id") or tag) if row else tag
    except Exception:
        return tag


def _resolve_doc_tags(items: list) -> list:
    """Registry-resolve every surviving doc tag (see _registry_doc).

    Runs AFTER _rescope_items: which tags survive is the guard's decision,
    this only upgrades what a surviving tag points at. Stripped tags (None)
    stay stripped — an FP number inside a tag the guard rejected must not
    bring the scope back.
    """
    for item in items:
        if item.get("doc"):
            item["doc"] = _registry_doc(str(item["doc"]))
    return items


def _resolved_refs_note(items: list, msg_text: str):
    """The short 'this question refers to: FP274 = 02_gcf-b42-…' line, or None.

    The factual answer call no longer sees the conversation (see
    _answer_messages), so a follow-up's referents have to reach it some other
    way. They reach it as IDENTIFIERS, never as prose: every entry comes from
    a doc tag that survived the guards (the conductor's resolution of 'it' /
    'those', already checked against the ids actually cited and the registry)
    or from an FP number in the user's own message, mapped through the
    registry. Nothing here is read out of an earlier answer's text — that
    text, with its figures, is precisely the pseudo-evidence this step
    removes.

    Deterministic and short by construction: tag order, then message order,
    capped at four documents.
    """
    docs = []
    for item in items or []:
        tag = str(item.get("doc") or "")
        if tag and tag not in docs:
            docs.append(tag)
    try:
        for row in registry.resolve_fps(msg_text or "")[0]:
            if row["doc_id"] not in docs:
                docs.append(row["doc_id"])
    except Exception:
        pass                    # registry is an enhancement, never a blocker
    refs = []
    for doc in docs[:4]:
        try:
            fp = (registry.load().get(doc) or {}).get("fp")
        except Exception:
            fp = None
        fp = fp or _fp_of(doc)
        label = f"FP{fp}" if fp else ""
        # a bare 'fp152' tag (registry unavailable) is its own label
        refs.append(f"{label} = {doc}" if label and label.lower() != doc.lower()
                    else (label or doc))
    if not refs:
        return None
    return ("This question refers to: " + "; ".join(refs)
            + ". (Identifiers resolved by the system; the excerpts below are "
              "the only evidence.)")


# A registry line is a paragraph of metadata per document; a fan-out that
# resolved six of them would bury the excerpts under cover-page facts. Four is
# the cap _resolved_refs_note already applies to the same list, for the same
# reason.
_MAX_TURN_NOTE_DOCS = 4

# The stem a main registry line ends with ('… [102_gcf-b30-02-add05, cover
# pages]') — i.e. which documents a note already speaks for. registry._fmt
# writes that trailer, so this reads what was actually emitted rather than
# recomputing it from the question.
_NOTE_DOC_RE = re.compile(r"\[([0-9]{1,3}_[\w.\-]+), cover pages\]")


def _turn_doc_ids(items: list) -> list:
    """Corpus stems THIS TURN's resolved search items are about.

    Two sources, both of them the turn's own resolved plan — never the
    conversation, and never an earlier answer's prose:

    * a doc tag that survived the rewrite guards (a conductor tag the message
      itself names, a pre-scope tag, a planner scope), mapped to the
      authoritative stem by the same _registry_doc call retrieval filters on;
    * an FP id or board code inside a resolved sub-query. That is where a
      follow-up's referent ends up: the conductor rewrites "Et quelle entité
      accréditée le met en œuvre ?" into "FP173 Amazon Bioeconomy Fund
      accredited entity", and that rewrite is already what routes retrieval —
      a document good enough to retrieve from is good enough to state the
      registry's line for.

    An identifier that resolves nowhere is dropped rather than reported: this
    list is not the user's claim that a document exists (registry_note answers
    that, for the question's own words, with a NOT FOUND line). A machine
    rewrite is not a claim, and "FP999 does not exist" is not something the
    turn asked.
    """
    docs = []
    try:
        rows = registry.load()
        if not rows:
            return docs

        def _add(doc_id):
            if doc_id and doc_id in rows and doc_id not in docs:
                docs.append(doc_id)

        for item in items or []:
            tag = str(item.get("doc") or "")
            if tag:
                _add(_registry_doc(tag))
            q = str(item.get("q") or "")
            for n in dict.fromkeys(_FP_RE.findall(q.lower())):
                row = registry.by_fp(int(n))
                if row:
                    _add(row["doc_id"])
            for b_, item_, add_ in dict.fromkeys(_BOARD_CODE_RE.findall(q)):
                row = registry.resolve_board_code(
                    int(b_), int(add_), int(item_) if item_ else None)
                if row:
                    _add(row["doc_id"])
    except Exception:
        pass            # the registry is an enhancement, never a blocker
    return docs


def _extend_registry_note(note, items):
    """`note`, plus a registry line for each document the turn RESOLVED to.

    registry_note() keys off identifiers in the question TEXT, and a follow-up
    has none: release-3's fu-lang-switch asked "Et quelle entité accréditée le
    met en œuvre ?" of a thread about FP173. The conductor resolved it, the
    query it produced named FP173, retrieval returned FP173's package — and
    the answer model, correctly obeying cite-or-hedge, said the excerpts do
    not state the accredited entity. The registry states it (Inter-American
    Development Bank); nothing had put it in front of the model, because the
    French sentence spells no identifier.

    So the trigger widens from "the question names the document" to "this turn
    resolved to the document". Same emitter and same format — registry._fmt
    and registry._conflict_lines are the functions registry_note itself calls,
    so the two can never drift — and a document the question already named
    keeps exactly ONE line (`have`, seeded from the note's own trailers).
    """
    lines, added = [], 0
    try:
        # Seed `have` from FULL registry lines only ('Registry — FP…'), not
        # from every bracketed stem: since the serving wave, inverse/board
        # LISTING items each end '[stem, cover pages]', and counting those as
        # "already covered" would rob a resolved document of its full _fmt
        # line on any turn that both fires a listing and resolves into it.
        have = set()
        for ln in (note or "").splitlines():
            if ln.startswith("Registry — FP") or ln.startswith("Registry — CONFLICT"):
                have |= set(_NOTE_DOC_RE.findall(ln))
        rows = registry.load()
        for doc in _turn_doc_ids(items):
            if doc in have:
                continue
            if added >= _MAX_TURN_NOTE_DOCS:
                break
            row = {"doc_id": doc, **(rows.get(doc) or {})}
            lines.append("Registry — " + registry._fmt(row))
            lines += registry._conflict_lines(row)
            have.add(doc)
            added += 1
    except Exception:
        return note     # an enhancement of an enhancement: never a blocker
    if not lines:
        return note
    return "\n".join(([note] if note else []) + lines)


# ---------------------------------------------------------------------------
# Comparison-planner intent gate (config.PLANNER).
#
# planner.detect() fires on ANY message naming >= 2 documents, comparative or
# not: "FP254 is interesting. Separately, FP248 was approved last year." names
# two and asks nothing, and a deterministic 2x4 evidence matrix is the wrong
# answer to it. The planner is deliberately not the place to judge intent — it
# resolves identifiers and fields, two closed problems — so the gate lives
# here, in the wiring, where the alternative (the LLM conductor, which reads
# the conversation) is in scope.
# ---------------------------------------------------------------------------
_COMPARE_INTENT_RE = re.compile(
    r"\bcompare[sd]?\b|\bcomparing\b|\bcomparison\b|\bversus\b|\bvs\b"
    r"|\bdiffer(?:s|ed|ent|ently|ence|ences)?\b"
    r"|\bcompar(?:er|ez|ons|aison|atif|ative)\b|\bdiff[ée]rence?s?\b"
    r"|\blequel\b|\blaquelle\b|\blesquel(?:le)?s\b|\bpar rapport\b", re.I)


def _ids_in(text: str) -> set:
    """The distinct document identifiers in a string: FP tokens and board codes."""
    return ({("fp", n) for n in _FP_RE.findall(text or "")}
            | {("board",) + tuple(t) for t in _BOARD_CODE_RE.findall(text or "")})


def _asks_about_both(text: str) -> bool:
    """A question mark and >= 2 identifiers inside ONE sentence.

    "Which of FP254 and FP248 is bigger?" carries no comparison keyword and no
    field word, yet the two ids share a question. The prose case the gate
    exists to exclude never puts them in one interrogative sentence.
    """
    for sent in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        if "?" in sent and len(_ids_in(sent)) >= 2:
            return True
    return False


def _plan_query(plan, doc) -> str:
    """The English retrieval query for one planned document.

    The user's own wording does not survive as a query here, and that is the
    point. The index is built over the English extracted corpus, and the
    conductor path translates every sub-query into English before retrieving;
    the planner path skips the conductor, so a French question used to reach
    the retriever verbatim — measured on 'Comparez le financement de FP151 et
    FP152', that returns excerpts with no financing figure at all.

    The plan already knows which fields the question asked for, and the planner
    publishes the English phrasing it uses for each of them (the same map its
    own cell retrieval uses), so the translation is a lookup, not a model call.
    The document's own identifier leads the query: retrieve.py routes on
    identifiers, and the scope filter degrades to an unscoped search when it
    matches nothing — in which case the id is the only thing keeping the query
    on the right document.
    """
    parts = [planner._FIELD_QUERIES.get(f, f.replace("_", " "))
             for f in list(plan.fields)[:4]]      # bounded: a query is a phrase
    return " ".join([doc.label] + parts)


def _planner_intent(text: str, plan) -> bool:
    """Does this >=2-id message actually ask for a document-by-field answer?

    Three independent signals, any of which is enough:
      * a comparison word (compare / versus / vs / difference / differ,
        comparer / différence / lequel);
      * a field keyword — `plan.default_fields` is False exactly when the
        planner's own field map matched something the question asked for;
      * both identifiers inside one question sentence.
    """
    if _COMPARE_INTENT_RE.search(text or ""):
        return True
    if plan is not None and not plan.default_fields:
        return True
    return _asks_about_both(text)


# ---------------------------------------------------------------------------
# Answer verification (config.VERIFY, plan step 5 — gcf_qna.rag.verify).
#
# The answer model is the one component allowed to WRITE facts, and the one we
# cannot audit by construction. verify.py audits its output claim by claim
# against the exact pages this turn retrieved; the app's job is to run it after
# the stream and say plainly what could not be verified. Every failure path here
# keeps the original answer — and so does every SUCCESS path: the verifier is a
# DETECTOR, not an editor (eac4c94 deleted the adopt-if-clean repair, whose
# rewrite destroyed the evidence of what had been wrong). The only thing it may
# add to the message body is the abstain banner above the model's own text.
# ---------------------------------------------------------------------------

def _claim_texts(verdicts: list, limit: int = 3, width: int = 110) -> str:
    """Claim sentences of some verdicts, trimmed to fit a one-line warning."""
    out = []
    for v in verdicts[:limit]:
        text = re.sub(r"\s+", " ", (v.claim.text or "").strip())
        out.append(text if len(text) <= width else text[:width].rstrip() + "…")
    more = len(verdicts) - len(out)
    return "; ".join(out) + (f" (+{more} more)" if more > 0 else "")


def _abstain_banner(res) -> str:
    """The line an abstained answer is prefixed with.

    'abstain' means every fact-bearing claim failed: there is nothing left in
    the answer that the cited pages support. A warning appended below the
    sources reads as a footnote to a confident-looking body, so it LEADS the
    message instead. The body it leads is the answer the model wrote: the
    verifier never rewrites it (see the block comment above), and an answer
    whose every claim failed is in any case the last one a rewrite could stand
    on.
    """
    return ("⚠️ Retrieval did not surface evidence for this — none of these "
            "claims could be checked against the cited pages, so treat the "
            "answer below as unverified: " + _claim_texts(res.failures))


def _verification_lines(res) -> list:
    """What the user is told about a verification result — nothing when the
    answer verified clean.

    Same shape as the existing invalid-citation warning: ⚠️ lines appended to
    the sources block, never a separate ceremony. Each line names the claims
    the verifier could not stand behind, because an unflagged partial answer
    reads exactly like a verified one. 'abstain' is absent on purpose: it leads
    the answer message itself (see _abstain_banner). There is no '✎ corrected'
    line any more — 'repaired' became unreachable when verification turned into
    a pure detector (eac4c94), and a line announcing a correction that never
    happened would misdescribe the text on screen.
    """
    if res is None:
        return []
    lines = []
    if res.status == "partial":
        # failures, not just `unsupported`: a CONTRADICTED claim is the worse
        # of the two failure kinds, and printing only the unsupported ones left
        # an empty list under the warning while a wrong figure sat on screen
        lines.append("⚠️ not supported by the retrieved pages (treat with "
                     "caution): " + _claim_texts(res.failures))
    elif res.status == "unverified-llm":
        lines.append("⚠️ claims could not be re-checked (no verification model "
                     "available); the deterministic checks flag: "
                     + _claim_texts(res.failures))
    if res.cautions:
        lines.append("⚠️ citation cautions: " + "; ".join(
            f"{_claim_texts([v], width=70)} [{', '.join(v.flags[:2])}]"
            for v in res.cautions[:3]))
    return lines


def _cite_key(label: str):
    """('doc prefix', page) for a citation label, however it was written.

    The two reporters format the same defect differently — _invalid_citations
    prints '55_gcf-b37-02-add11-funding-propos… p.41', the verifier's flag
    carries '55_gcf-b37-02-add11-…-fp220, p.41' — and both truncate the id, so
    the comparable part is the leading 24 characters: the same prefix width
    _cited and _resolve_doc already compare on.
    """
    s = (label or "").strip()
    m = re.search(r"[0-9]{1,3}_[\w.\-]+", s)
    page = re.search(r"\bpp?\.\s*(\d{1,3})\b", s)
    return ((m.group(0) if m else s).lower()[:24],
            int(page.group(1)) if page else None)


def _verifier_flagged_cites(res) -> set:
    """Citations the verifier already reported as pointing outside the evidence."""
    out = set()
    for v in (res.verdicts if res is not None else []):
        for f in v.flags:
            if f.startswith("invalid-citation:"):
                out.add(_cite_key(f.split(":", 1)[1]))
    return out


def _answer_messages(system_prompt: str, context: str, question: str,
                     refs_note: str = None) -> list:
    """The factual answer call's messages array: system + ONE user turn.

    Prior turns are deliberately absent. They used to be prepended as
    conversation, and the model read old ANSWERS as evidence — a figure
    stated three turns ago outranked the excerpts retrieved for the question
    actually asked, so the same fully specified question was answered
    differently depending on what preceded it (the plan's FP220-after-
    FP254/FP248 regression). The conductor still receives the whole
    conversation, because resolving 'it' / 'those' is its job; what it
    resolves arrives here as ids in `refs_note`, not as text to quote.

    Chat-mode turns are the exception and keep their history: continuity IS
    the answer there, and no excerpt is in play.
    """
    user = f"Context excerpts:\n{context}\n\nQuestion: {question}"
    if refs_note:
        user = f"{refs_note}\n\n{user}"
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user}]


def _answer_cap() -> dict:
    """kwargs for the answer calls' output cap — empty when MAX_ANSWER_TOKENS
    is unset (the default), so the request carries no cap at all rather than
    an explicit null that OpenAI-compatible servers may reject."""
    return ({"max_completion_tokens": config.MAX_ANSWER_TOKENS}
            if config.MAX_ANSWER_TOKENS else {})


def _finish_reason(part) -> Optional[str]:
    """The finish_reason of a streamed chunk, or None.

    Every access is defensive on purpose: only the LAST chunk of a stream
    carries the field at all, some OpenAI-compatible servers omit it entirely,
    and a stream reader that raises turns a cap into a lost answer — which is
    strictly worse than an unmarked truncation.
    """
    choices = getattr(part, "choices", None) or []
    return getattr(choices[0], "finish_reason", None) if choices else None


#: What the reader is told when the model stopped because it ran out of budget
#: rather than because it had finished. MAX_ANSWER_TOKENS is unset by default
#: (d30da86) precisely because a cap silently truncated answers — agg-inv-undp
#: named 34 of 41 proposals and read as a complete list. This fires only when
#: an operator re-imposes a budget, and then it says so in the answer's own
#: language: a list that stops early must never look like a list that ended.
def _truncation_marker(lang: str = None) -> str:
    return ("\n\n… [réponse tronquée : le budget de jetons configuré a été "
            "atteint ; la suite n'a pas été rédigée]"
            if lang == "French" else
            "\n\n… [answer truncated at the configured token budget; the rest "
            "was never written]")


#: The same fact, for the verification step: the verdicts below it cover the
#: text that was written, and the claims the answer never got to are neither
#: supported nor unsupported — they are absent, and a status of 'verified' on
#: a truncated answer must not be read as 'complete'.
_TRUNCATION_STEP_NOTE = ("answer truncated: the model stopped at the "
                         "configured token budget (finish_reason='length'), "
                         "so these verdicts cover only the text that was "
                         "written")


# ---------------------------------------------------------------------------
# Conversation memory: which documents this thread has been ABOUT.
# ---------------------------------------------------------------------------
# The conductor used to be told the conversation's documents by `_cited_docs`,
# a regex over the ANSWER TEXT of the last messages — and those messages reach
# it truncated to 1200 characters, so the citation that names the document is
# routinely the part that got cut. A turn whose answer was truncated, or which
# cited nothing at all, left the conductor with no referent to resolve 'it'
# against and `_resolve_doc` with no list to pin a mangled tag to.
#
# The memory is built from the turn's own RESOLVED PLAN instead — identifiers
# the registry resolved, doc tags that survived the rewrite guards, and the
# citations the answer actually made — never by reading prose for facts. That
# is the same discipline `_resolved_refs_note` already applies to the answer
# call, applied to the conductor.
#: How a document got into the memory, strongest provenance first. An entry
#: keeps the strongest reason it was ever added: a document the user named by
#: identifier is 'asked' for the rest of the conversation even if later turns
#: only cite it.
_MEMORY_PROVENANCE = ("asked", "resolved", "cited")

#: Documents the conductor line may name. `_cited_docs` shipped its last 12,
#: and this list is the same budget spent on better-chosen entries.
_MEMORY_MAX_DOCS = 12

#: Characters the conductor line may spend. The conductor prompt is a length
#: budget like the answer prompt (`tests/test_prompts.py`), and this line is
#: the only unbounded thing in it: 12 entries of 'FP254 = 22_gcf-b40-02-add16-
#: rev01-funding-proposal-package-fp254' is ~800 characters. Entries are
#: dropped from the OLDEST end when the budget binds, because the referent of
#: 'it' is at the recent end.
_MEMORY_LINE_CHARS = 900

_MEMORY_LINE_PREFIX = "Documents discussed so far, most recent last: "


@dataclass
class MemoryEntry:
    """One document this conversation has been about."""
    doc_id: str
    fp: Optional[int] = None
    board: Optional[int] = None
    year: Optional[int] = None
    first_turn: int = 0
    last_turn: int = 0
    how: str = "resolved"

    @property
    def label(self) -> str:
        """'B.40, 2024' — the board/year label the excerpt headers print."""
        return f"B.{self.board}, {self.year}" if self.board and self.year else ""

    def as_dict(self) -> dict:
        return {"doc_id": self.doc_id, "fp": self.fp, "board": self.board,
                "year": self.year, "first_turn": self.first_turn,
                "last_turn": self.last_turn, "how": self.how}


@dataclass
class ConversationMemory:
    """The ordered, de-duplicated documents of one conversation.

    Ordered by when each document was last seen — oldest first, so the line
    the conductor reads ends on the referent 'it' most likely means. Capped at
    `_MEMORY_MAX_DOCS`; the oldest entries fall off, never the newest.
    """
    entries: list = field(default_factory=list)
    turn: int = 0
    max_docs: int = _MEMORY_MAX_DOCS

    # -- reading -----------------------------------------------------------
    def __bool__(self) -> bool:
        return bool(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def doc_ids(self) -> list:
        """Oldest first — the shape `_resolve_doc`/`_rescope_items` expect of
        `history_docs` (they scan it in order and `_cited_docs` was oldest
        first too)."""
        return [e.doc_id for e in self.entries]

    def get(self, doc_id: str):
        return next((e for e in self.entries if e.doc_id == doc_id), None)

    def conductor_line(self) -> Optional[str]:
        """The one line the conductor is told, or None when nothing is known.

        'FP220 = 55_gcf-…; FP254 = 22_gcf-…': an identifier the conductor can
        put in a doc tag, next to the stem retrieval actually filters on. A
        document with no FP number (the B.27-era stems carry none) prints its
        stem alone rather than an empty label.
        """
        if not self.entries:
            return None
        bits = [(f"FP{e.fp} = {e.doc_id}" if e.fp else e.doc_id)
                for e in self.entries]
        line = _MEMORY_LINE_PREFIX + "; ".join(bits)
        while len(bits) > 1 and len(line) > _MEMORY_LINE_CHARS:
            bits.pop(0)                       # the oldest entry pays first
            line = _MEMORY_LINE_PREFIX + "; ".join(bits)
        return line

    # -- writing -----------------------------------------------------------
    def update(self, *, asked=(), resolved=(), cited=(), turn: int = None):
        """Fold one turn's own resolved plan and answer into the memory.

        The three sources are the three honest ways a document can be what the
        conversation is about, and they are applied strongest-first so an
        identifier the user typed outranks a citation the model made. Ids that
        are not in the registry are dropped: an invented tag is not a document
        the conversation was about.
        """
        if turn is None:
            turn = self.turn + 1
        self.turn = max(self.turn, turn)
        for how, docs in (("asked", asked), ("resolved", resolved),
                          ("cited", cited)):
            for doc_id in docs or ():
                self._touch(doc_id, how, turn)
        if len(self.entries) > self.max_docs:
            del self.entries[:len(self.entries) - self.max_docs]
        return self

    def _touch(self, doc_id: str, how: str, turn: int):
        if not doc_id:
            return
        row = _registry_row(doc_id)
        if row is None:
            return                     # not a document: never a memory entry
        entry = self.get(doc_id)
        if entry is None:
            entry = MemoryEntry(doc_id=doc_id, fp=row.get("fp"),
                                board=row.get("board"), year=row.get("year"),
                                first_turn=turn, last_turn=turn, how=how)
            self.entries.append(entry)
            return
        # seen again: it moves to the recent end, keeps its first sighting and
        # the strongest reason it was ever here
        if _MEMORY_PROVENANCE.index(how) < _MEMORY_PROVENANCE.index(entry.how):
            entry.how = how
        if turn >= entry.last_turn:
            entry.last_turn = turn
            self.entries.remove(entry)
            self.entries.append(entry)

    # -- persistence -------------------------------------------------------
    def to_metadata(self) -> dict:
        return {"turn": self.turn, "docs": [e.as_dict() for e in self.entries]}

    @classmethod
    def from_metadata(cls, data):
        """Rebuild from what `to_metadata` wrote, or None when it is absent or
        unreadable — the caller then falls back to replaying the steps."""
        try:
            rows = (data or {}).get("docs")
            if not rows:
                return None
            mem = cls(turn=int((data or {}).get("turn") or 0))
            for r in rows:
                if not r.get("doc_id"):
                    continue
                mem.entries.append(MemoryEntry(
                    doc_id=r["doc_id"], fp=r.get("fp"), board=r.get("board"),
                    year=r.get("year"), first_turn=int(r.get("first_turn") or 0),
                    last_turn=int(r.get("last_turn") or 0),
                    how=r.get("how") if r.get("how") in _MEMORY_PROVENANCE
                    else "cited"))
            return mem or None
        except Exception:                              # noqa: BLE001
            log.warning("conversation memory metadata unreadable; rebuilding "
                        "from the thread's steps", exc_info=True)
            return None

    @classmethod
    def from_history(cls, history, *, max_docs: int = _MEMORY_MAX_DOCS):
        """Rebuild from a replayed conversation — the resume fallback.

        A resumed thread has no plan to read: the turns are gone and only the
        text survives. The citations an assistant turn actually made are the
        one honest record left of which documents it was about, so they are
        replayed turn by turn (registry-resolved, so a truncated or invented
        bracket contributes nothing) and the ids the user's own messages name
        are replayed as 'asked' alongside them.
        """
        mem = cls(max_docs=max_docs)
        turn = 0
        for m in history or []:
            content = (m or {}).get("content") or ""
            if (m or {}).get("role") == "user":
                turn += 1
                mem.update(asked=_asked_doc_ids(content), turn=turn)
            else:
                mem.update(cited=_cited_docs([{"content": content}]),
                           turn=max(turn, 1))
        return mem


def _registry_row(doc_id: str):
    """The registry's row for a stem, or None when it names no document."""
    try:
        rows = registry.load() or {}
    except Exception:                                  # noqa: BLE001
        return None                # the registry is an enhancement, not a gate
    row = rows.get(doc_id)
    return dict(row or {}) if doc_id in rows else None


def _asked_doc_ids(text: str) -> list:
    """Corpus stems the identifiers in a MESSAGE resolve to, through the
    registry — the 'asked' half of the memory."""
    try:
        return list(dict.fromkeys(r["doc_id"]
                                  for r in registry.resolve_fps(text or "")[0]))
    except Exception:                                  # noqa: BLE001
        return []


def _answer_cited_doc_ids(answer: str, res=None) -> list:
    """Documents the ANSWER cited: the verifier's page-aware parse when
    verification ran, else the same citation regex the sources block falls
    back to. Registry-resolved by `_touch`, so a truncated bracket that names
    no real document contributes nothing."""
    if res is not None:
        try:
            return list(dict.fromkeys(d for d, _ in res.sources))
        except Exception:                              # noqa: BLE001
            pass
    return _cited_docs([{"content": answer or ""}])


def _memory_docs_for_turn(items, plan=None) -> list:
    """The 'resolved' half: this turn's own plan, never its prose.

    `_turn_doc_ids` reads the doc tags that survived the guards and the
    identifiers inside the resolved sub-queries; a planner turn adds the
    matrix's own documents, whose stems came straight from the registry.
    """
    docs = list(_turn_doc_ids(items))
    for doc in (getattr(plan, "docs", None) or ()):
        stem = _registry_doc(getattr(doc, "scope", None) or "")
        if getattr(doc, "missing", False) or not stem:
            continue
        if stem not in docs:
            docs.append(stem)
    return docs


def conversation_docs(history, *, memory=None, cited_docs=None):
    """(the line the conductor is told, the ids the rewrite guards resolve on).

    The memory wins when it holds anything; `_cited_docs` is the fallback for
    a conversation that has none — an old session, or a thread resumed from
    steps that cite nothing.
    """
    if memory is not None and len(memory):
        return memory.conductor_line(), memory.doc_ids()
    cited = list(cited_docs) if cited_docs is not None else _cited_docs(history or [])
    if cited:
        return ("Documents cited in conversation: " + ", ".join(cited[-12:]),
                cited)
    return None, cited


# ---------------------------------------------------------------------------
# The turn: settings in, four stages, one driver.
# ---------------------------------------------------------------------------
#: Excerpts one turn's ranked fan-out may carry, before the probes extend it.
_MERGED_HITS_CAP = 15
#: The conductor's window over the conversation, and what each message of it
#: is truncated to. Not settings: both callers ship the same numbers, and a
#: knob is how they would come to differ.
_CONDUCTOR_TURNS = 6
_CONDUCTOR_MSG_CHARS = 1200
_CONDUCTOR_MAX_TOKENS = 300
_CONDUCTOR_MAX_QUERIES = 6
#: Exchanges the app keeps in `cl.user_session["history"]`.
_HISTORY_TURNS = 12

_WEAK_SIGNAL_NOTE = ("Note: retrieval confidence for this question is LOW — the "
                     "excerpts below may not actually be relevant. Do not force an "
                     "answer from marginal matches; say plainly that the corpus "
                     "does not appear to cover this.")


@dataclass
class TurnSettings:
    """Every switch a turn reads, resolved before it starts.

    The app builds one from `config` per turn (so an operator's `.env` still
    decides production's behaviour, and a test that monkeypatches
    `config.PLANNER` still reaches it); the eval harness passes one built from
    its own CLI flags and reads no ambient environment at all.
    """
    conductor: bool = True
    planner: bool = True
    verify: bool = False
    verify_llm: bool = True
    top_k: int = 10
    chat_model: str = "gpt-5.2"
    max_answer_tokens: Optional[int] = None
    min_dense_score: float = 0.5
    #: sampling parameters sent with every model call (the harness pins
    #: temperature and seed; production sends neither)
    call_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg=config, **overrides):
        base = dict(conductor=cfg.CONDUCTOR, planner=cfg.PLANNER,
                    verify=cfg.VERIFY, verify_llm=cfg.VERIFY_LLM,
                    top_k=cfg.TOP_K, chat_model=cfg.CHAT_MODEL,
                    max_answer_tokens=cfg.MAX_ANSWER_TOKENS,
                    min_dense_score=cfg.MIN_DENSE_SCORE)
        base.update(overrides)
        return cls(**base)


@dataclass
class TurnPlan:
    """What the turn decided to search for, and how it got there."""
    mode: str = "retrieve"                 # retrieve | chat
    plan: Any = None                       # planner.Plan, or None
    search_queries: list = field(default_factory=list)
    decomposed: bool = False
    matrix_block: Optional[str] = None
    guard_answer: Optional[str] = None
    lang: Optional[str] = None
    # -- what the stages observed (the harness's parity block reads these) --
    conductor_called: bool = False
    conductor_chat: bool = False
    conductor_fanned_out: bool = False
    conductor_failed: bool = False
    conductor_raw: Optional[str] = None
    conductor_response: Any = None
    conductor_latency_s: Optional[float] = None
    planner_detected: bool = False
    planner_intent_ok: bool = False
    planner_matrix_built: bool = False
    planner_matrix_failed: bool = False
    guards: dict = field(default_factory=dict)
    # -- UI step outputs, rendered by whoever has a UI ----------------------
    matrix_step: Optional[str] = None
    retrieval_step: Optional[str] = None


@dataclass
class TurnContext:
    """Everything assembled before the answer call."""
    hits: list = field(default_factory=list)
    ranked_hits: list = field(default_factory=list)
    probe_hits: list = field(default_factory=list)
    section_hits: list = field(default_factory=list)
    confidence: float = 0.0
    weak_signal: bool = False
    reg_note: Optional[str] = None
    year_note: Optional[str] = None
    board_note: Optional[str] = None
    coverage_note: Optional[str] = None
    matrix_block: Optional[str] = None
    context: str = ""
    system_prompt: str = ""
    refs_note: Optional[str] = None
    user: str = ""
    messages: list = field(default_factory=list)
    evidence: Any = None
    lang: Optional[str] = None

    @property
    def note_blocks(self) -> list:
        """The computed blocks, in the order `verify.build_evidence` is given
        them — registry, year, matrix."""
        return [self.reg_note, self.year_note, self.matrix_block]


@dataclass
class SourcesReport:
    """The sources line and its warnings — computed from the answer, not from
    the UI that displays it."""
    hits: list = field(default_factory=list)
    shown: list = field(default_factory=list)
    sources: str = ""
    bad_citations: list = field(default_factory=list)
    lines: list = field(default_factory=list)

    def rendered(self) -> str:
        out = self.sources
        if self.bad_citations:
            out += ("\n⚠️ cited but not among retrieved pages (treat with "
                    "caution): " + "; ".join(self.bad_citations[:4]))
        for line in self.lines:
            out += "\n" + line
        return out


@dataclass
class TurnResult:
    """One finished turn."""
    question: str = ""
    plan: TurnPlan = None
    context: TurnContext = None
    answer: str = ""
    truncated: bool = False
    verification: Any = None
    verification_step: Optional[str] = None
    abstain_banner: Optional[str] = None
    sources: SourcesReport = None
    calls: list = field(default_factory=list)     # per-call metadata
    memory_update: dict = field(default_factory=dict)
    # wall clock of the stages that can be slow, for the per-turn log line
    context_s: Optional[float] = None
    answer_s: Optional[float] = None
    verify_s: Optional[float] = None

    @property
    def mode(self) -> str:
        if self.plan is not None and self.plan.guard_answer is not None:
            return "guard"
        return (self.plan.mode if self.plan is not None else "retrieve")


# ---------------------------------------------------------------------------
# stage helpers: the rewrite guards, with a record of what they rewrote
# ---------------------------------------------------------------------------
def _apply_prescope(items: list, msg_text: str, guards: dict) -> list:
    """`_prescope_single_fp`, counted. The count is what the release record's
    parity block publishes for "production's single-FP pre-scoping also ran
    here" — an audit that answers that with a hard-coded True is worth
    nothing."""
    before = [bool(i.get("doc")) for i in (items or [])]
    out = _prescope_single_fp(items, msg_text)
    guards["prescope_calls"] = guards.get("prescope_calls", 0) + 1
    guards["prescope_tagged"] = guards.get("prescope_tagged", 0) + sum(
        1 for was, item in zip(before, out or []) if not was and item.get("doc"))
    return out


def _apply_rescope(items: list, msg_text: str, history_docs: list,
                   guards: dict) -> list:
    """`_rescope_items`, with the pre-scope it runs first hoisted so it can be
    counted.

    Hoisted, not moved: `_rescope_items` still calls it, and calling it twice
    is a no-op by construction (an item that already carries a tag is returned
    untouched). The alternative — a counter inside the moved function — would
    edit a function whose body is a measured artifact.
    """
    items = _apply_prescope(items, msg_text, guards)
    before = [i.get("doc") for i in (items or [])]
    out = _rescope_items(items, msg_text, history_docs)
    guards.setdefault("rescoped", []).extend(
        [was, item.get("doc")] for was, item in zip(before, out or [])
        if was != item.get("doc"))
    return out


def _apply_resolve_tags(items: list, guards: dict) -> list:
    """`_resolve_doc_tags`, recording every tag the registry moved."""
    before = [i.get("doc") for i in (items or [])]
    out = _resolve_doc_tags(items)
    guards.setdefault("registry_resolved", []).extend(
        [was, item.get("doc")] for was, item in zip(before, out or [])
        if was != item.get("doc"))
    return out


# ---------------------------------------------------------------------------
# stage 1a: the deterministic comparison planner and its evidence matrix
# ---------------------------------------------------------------------------
def planner_detect(question: str, settings: TurnSettings, turn: TurnPlan = None):
    """The plan a comparison question resolves to, behind the app's intent gate.

    `planner.detect` fires on ANY message naming >= 2 documents, comparative
    or not; the gate is here, in the wiring, because judging intent needs the
    conversation and the planner resolves identifiers and fields — two closed
    problems it is deliberately not asked to leave.
    """
    if not settings.planner:
        return None
    plan = planner.detect(question)
    if plan is None:
        return None
    if turn is not None:
        turn.planner_detected = True
    if not _planner_intent(question, plan):
        return None
    if turn is not None:
        turn.planner_intent_ok = True
    return plan


def planner_matrix(plan, retriever, turn: TurnPlan = None):
    """(rendered matrix, the step output that says what happened).

    A matrix whose every cell is empty is discarded: the turn then proceeds
    exactly as PLANNER=0 would, and the step says so rather than failing the
    turn. Never raises.
    """
    try:
        matrix = planner.build_matrix(plan, retriever)
        if not any(c.status not in ("missing", "missing-document")
                   for c in matrix.cells):
            raise ValueError("no cell carries evidence")
        block = planner.render(matrix)
        step = (f"{plan.trigger}\n"
                + ", ".join(f"{k} {v}" for k, v in matrix.counts.items()))
        if turn is not None:
            turn.planner_matrix_built = True
        return block, step
    except Exception as e:                    # noqa: BLE001 — never user-facing
        if turn is not None:
            turn.planner_matrix_failed = True
        log.warning("planner: evidence matrix skipped (%s: %s); falling back "
                    "to the LLM conductor", type(e).__name__, e, exc_info=True)
        return None, (f"planner skipped for this turn "
                      f"({type(e).__name__}: {e}); falling back to the "
                      f"LLM conductor")


def planner_stage(question: str, retriever, settings: TurnSettings,
                  turn: TurnPlan = None):
    """(plan, matrix block) — detect, gate, build, in one call."""
    plan = planner_detect(question, settings, turn)
    if plan is None:
        return None, None
    block, step = planner_matrix(plan, retriever, turn)
    if turn is not None:
        turn.matrix_step = step
    return (plan, block) if block is not None else (None, None)


# ---------------------------------------------------------------------------
# stage 1b: the per-turn LLM conductor
# ---------------------------------------------------------------------------
def conductor_request(question: str, history: list, settings: TurnSettings, *,
                      memory=None, cited_docs=None):
    """(the call kwargs, the doc ids the rewrite guards will resolve against).

    THE one place the conductor's prompt is built. The call itself is not here
    because the two callers make it differently — the app awaits an
    `AsyncOpenAI` stream-capable client, the harness calls a synchronous
    metered one and books the usage — and a shared prompt with two callers is
    the whole point; a shared call would need one of them to lie about its
    client.
    """
    convo = ("\n".join(f"{m['role']}: {m['content'][:_CONDUCTOR_MSG_CHARS]}"
                       for m in history[-_CONDUCTOR_TURNS:])
             if history else "((no prior conversation))")
    docs_line, history_docs = conversation_docs(history, memory=memory,
                                                cited_docs=cited_docs)
    if docs_line:
        convo += "\n" + docs_line
    kwargs = dict(
        model=settings.chat_model,
        max_completion_tokens=_CONDUCTOR_MAX_TOKENS,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": CONDUCTOR_PROMPT},
            {"role": "user", "content":
                f"Conversation:\n{convo}\n\nLatest message: {question}"},
        ],
        **(settings.call_kwargs or {}),
    )
    return kwargs, history_docs


def conductor_adopt(raw: str, question: str, history_docs: list,
                    turn: TurnPlan) -> TurnPlan:
    """Parse the conductor's JSON and run the rewrite guards over it.

    Adopting the raw model output is the contamination `_rescope_items` exists
    for, so nothing reaches `turn.search_queries` that has not been through it.
    """
    turn.conductor_raw = raw
    data = json.loads(raw or "{}")
    if data.get("mode") == "chat":
        turn.mode = "chat"
        turn.conductor_chat = True
    parsed = []
    for item in (data.get("queries") or [])[:_CONDUCTOR_MAX_QUERIES]:
        if isinstance(item, str) and item.strip():
            parsed.append({"q": item.strip(), "doc": None})
        elif isinstance(item, dict) and (item.get("q") or "").strip():
            parsed.append({"q": item["q"].strip(), "doc": item.get("doc") or None})
    parsed = _apply_rescope(parsed, question, history_docs, turn.guards)
    if parsed:
        turn.search_queries = parsed
        if len(parsed) > 1:
            turn.conductor_fanned_out = True
    return turn


async def conductor_stage(question: str, history: list, llm,
                          settings: TurnSettings, turn: TurnPlan, *,
                          memory=None, cited_docs=None) -> TurnPlan:
    """The app's `run_conductor`: best-effort, and every failure path leaves
    the raw message as the only query.

    Follow-ups carry references the embedder cannot resolve ("how does THAT
    compare..."), so retrieval on the raw message fetches noise. One LLM call
    rewrites the message against recent history — into a single standalone
    query, or, for comparisons over documents the conversation named, one
    doc-scoped sub-query per entity. See docs/query-decomposition.html.
    """
    if llm is None:
        return turn
    try:
        kwargs, history_docs = conductor_request(
            question, history, settings, memory=memory, cited_docs=cited_docs)
        t0 = time.perf_counter()
        resp = await llm.chat.completions.create(**kwargs)
        turn.conductor_latency_s = round(time.perf_counter() - t0, 3)
        turn.conductor_called = True
        turn.conductor_response = resp
        conductor_adopt(resp.choices[0].message.content or "{}", question,
                        history_docs, turn)
    except Exception:                                  # noqa: BLE001
        turn.conductor_failed = True
        log.warning("conductor: call or parse failed; falling back to the raw "
                    "message as the only search query", exc_info=True)
    return turn


# ---------------------------------------------------------------------------
# stage 1c: the registry FP-miss guard
# ---------------------------------------------------------------------------
def fp_miss_guard(question: str):
    """'FP999 does not exist in this corpus', or None.

    Named FPs that resolve nowhere must not fall back to unscoped semantic
    retrieval: an FP999 refusal answered from unrelated documents looks
    grounded and is not.
    """
    try:
        if registry.load():
            resolved, missing = registry.resolve_fps(question)
            if missing and not resolved:
                lang = _detect_lang(question)
                miss = ", ".join(f"FP{n}" for n in missing)
                return (f"{miss} n'existe pas dans le corpus (registre de 273 documents)."
                        if lang == "French" else
                        f"{miss} does not exist in this corpus (273-document registry).")
    except Exception:                                  # noqa: BLE001
        log.warning("fp-miss guard: registry lookup failed; the turn proceeds "
                    "to retrieval unguarded", exc_info=True)
    return None


# ---------------------------------------------------------------------------
# stage 1: the whole plan — `chainlit_app.main()` up to the first search
# ---------------------------------------------------------------------------
async def plan_turn(question: str, history: list, retriever, llm,
                    settings: TurnSettings, *, memory=None,
                    cited_docs=None) -> TurnPlan:
    """What this turn will search for, in the app's own order.

    planner gate -> conductor -> chat -> FP-miss guard -> evidence matrix ->
    tag guards -> `decomposed`. The order is load-bearing: the guard returns
    before the matrix, so a message whose every identifier is unknown is
    refused without the planner's fallback spending a conductor call on it.
    """
    turn = TurnPlan(search_queries=[{"q": question, "doc": None}],
                    lang=_detect_lang(question))
    plan = planner_detect(question, settings, turn)
    if plan is None and settings.conductor:
        await conductor_stage(question, history, llm, settings, turn,
                              memory=memory, cited_docs=cited_docs)
    if turn.mode == "chat":
        return turn
    turn.guard_answer = fp_miss_guard(question)
    if turn.guard_answer is not None:
        return turn
    if plan is not None:
        block, step = planner_matrix(plan, retriever, turn)
        turn.matrix_step = step
        if block is None:
            # the conductor call the planner path skipped happens now, and the
            # turn proceeds exactly as PLANNER=0 would
            plan = None
            if settings.conductor:
                await conductor_stage(question, history, llm, settings, turn,
                                      memory=memory, cited_docs=cited_docs)
                turn.mode = "retrieve"   # a late chat verdict is ignored: the
                                         # chat branch is already behind us
        else:
            turn.matrix_block = block
            # Authoritative stems, and an English query per document built from
            # the fields the plan resolved: the raw message is the wrong query
            # here, and the conductor that would have translated it was
            # skipped. No model wrote these tags, so the rewrite guards have
            # nothing to repair and are skipped below.
            scoped = [{"q": _plan_query(plan, d), "doc": d.scope}
                      for d in plan.docs if not d.missing]
            turn.search_queries = scoped or turn.search_queries
    turn.plan = plan
    if plan is None:
        # On the unconditional path so it also covers the conductor-off and
        # conductor-failed fallbacks, where the raw message is the only query
        # and `_rescope_items` never ran. Already-tagged items are untouched.
        turn.search_queries = _apply_prescope(turn.search_queries, question,
                                              turn.guards)
        turn.search_queries = _apply_resolve_tags(turn.search_queries,
                                                  turn.guards)
    # A one-document plan (its partner identifier resolves nowhere) still ships
    # the comparison rules: the answer has to say so, document by document.
    turn.decomposed = len(turn.search_queries) > 1 or plan is not None
    if turn.decomposed or turn.search_queries[0]["q"] != question:
        turn.retrieval_step = "\n".join(
            f"{sq['q']}" + (f"   [{sq['doc']}]" if sq.get("doc") else "")
            for sq in turn.search_queries)
    return turn


# ---------------------------------------------------------------------------
# stage 2: retrieval and the context
# ---------------------------------------------------------------------------
def retrieve_stage(retriever, items: list, decomposed: bool,
                   settings: TurnSettings, original: str = None):
    """(hits, best dense confidence, weak-signal flag) — the fan-out.

    Per-query quota and a round-robin merge, so the global cap cannot starve
    the later documents of a multi-document turn.

    `original` — the user's own wording — rides along ONLY on a turn that
    stayed on one query: it lets the retriever rank pages INSIDE the document
    the rewrite chose, and never chooses documents. A message that fanned out
    names every document it compares, so inside any one of them it is a probe
    for the OTHERS' names and figures.
    """
    from itertools import zip_longest
    per_query = (settings.top_k if not decomposed
                 else max(3, settings.top_k // max(1, len(items))))
    best, weak, per_lists = None, True, []
    for sq in items:
        got, conf = retriever.search_with_confidence(
            sq["q"], per_query, sq.get("doc"),
            original if len(items) == 1 else None)
        best = conf if best is None else max(best, conf)
        if conf >= settings.min_dense_score:
            weak = False
        per_lists.append(got)
    seen, hits = set(), []
    for tier in zip_longest(*per_lists):
        for h in tier:
            if h is None:
                continue
            key = (h.doc_id, h.page, h.text[:120])
            if key not in seen:
                seen.add(key)
                hits.append(h)
    return hits[:_MERGED_HITS_CAP], (best if best is not None else 0.0), weak


def build_context(question: str, turn: TurnPlan, retriever,
                  settings: TurnSettings, *, retrieve=None) -> TurnContext:
    """Everything between the plan and the answer call.

    Retrieval fan-out, the computed notes, the two probes, the context block
    in the order the model reads it — matrix, registry line, weak-signal, year
    note, excerpts — the per-turn system prompt, the resolved-references note,
    the messages array and, when verification is on, the evidence the audit is
    allowed to hold.

    Synchronous on purpose: it is CPU and index work with no I/O of its own,
    so a caller with an event loop offloads the whole of it in one go instead
    of interleaving four awaits through a computation that has to stay in one
    order.

    `retrieve` overrides the fan-out for a caller that instruments it (the
    eval harness counts and records what each sub-query returned).
    """
    items = turn.search_queries
    if retrieve is None:
        def retrieve(it, dec, original=None):
            return retrieve_stage(retriever, it, dec, settings, original)
    hits, confidence, weak_signal = retrieve(items, turn.decomposed,
                                             original=question)
    ranked = list(hits)

    hits, year_note = _year_assist(question, hits)
    board_note = _board_range_note(question)
    if board_note:
        year_note = f"{year_note} {board_note}" if year_note else board_note
    coverage_note = _corpus_coverage_note(question)
    if coverage_note:
        year_note = (f"{year_note} {coverage_note}" if year_note
                     else coverage_note)

    # The registry note is computed BEFORE the context is assembled, because
    # its CONFLICT lines decide whether this turn still has to fetch a page by
    # name. Where it is PRINTED has not moved: it is prepended below, in the
    # same order it always was — matrix, registry, weak-signal, year, excerpts.
    reg_note = None
    try:
        reg_note = registry.registry_note(question)
        # …and a line for whatever else THIS turn resolved to. The question's
        # own words are not the only evidence of which document a turn is
        # about: a follow-up names none, and its resolved query names one.
        reg_note = _extend_registry_note(reg_note, items)
    except Exception:                                  # noqa: BLE001
        log.warning("registry note unavailable for this turn", exc_info=True)

    # The conflict probe SUPPLEMENTS this turn's excerpts and never edits
    # them: the pages it fetches go in front of the ranked list and the cap
    # above is extended by their count, so no hit that earned a slot loses one.
    probe_hits = _conflict_probe(retriever, reg_note, hits, question)
    if probe_hits:
        hits = probe_hits + hits
    # The section probe goes in front even of the conflict pages: its pages
    # are the question's own direct object, and the dedup inside it already
    # saw the conflict supplement in `hits`.
    section_hits = _section_probe(retriever, question, hits, question)
    if section_hits:
        hits = section_hits + hits

    context = _context_block(hits, probe_hits, section_hits)
    if year_note:
        context = year_note + "\n\n" + context
    if weak_signal:
        context = _WEAK_SIGNAL_NOTE + "\n\n" + context
    if reg_note:
        context = reg_note + "\n\n" + context
    if turn.matrix_block:
        # ABOVE the registry note and the excerpts: the matrix is the complete
        # half of the evidence (every named document, every asked field, empty
        # cells included) and has to be read before the retrieved sample.
        context = turn.matrix_block + "\n\n" + context

    # What the answer is allowed to have used: this turn's excerpts plus the
    # computed blocks prepended above — built here, from the same strings the
    # context got, so the audit cannot drift from what the model actually read.
    evidence = None
    if settings.verify:
        try:
            evidence = verify.build_evidence(
                hits, [n for n in (reg_note, year_note, turn.matrix_block) if n])
        except Exception:                              # noqa: BLE001
            evidence = None                     # never blocks the answer
            log.warning("verification evidence unavailable", exc_info=True)

    lang = turn.lang if turn.lang is not None else _detect_lang(question)
    system_prompt = assemble(year=bool(year_note), registry=bool(reg_note),
                             comparison=turn.decomposed,
                             matrix=bool(turn.matrix_block), lang=lang)
    refs_note = _resolved_refs_note(items, question)
    # Evidence isolation (plan step 1): this call gets the question, the
    # resolved-reference ids and THIS turn's excerpts — never prior answers.
    messages = _answer_messages(system_prompt, context, question, refs_note)
    user = messages[-1]["content"]
    return TurnContext(
        hits=hits, ranked_hits=ranked, probe_hits=probe_hits,
        section_hits=section_hits, confidence=confidence,
        weak_signal=weak_signal, reg_note=reg_note, year_note=year_note,
        board_note=board_note, coverage_note=coverage_note,
        matrix_block=turn.matrix_block, context=context,
        system_prompt=system_prompt, refs_note=refs_note, user=user,
        messages=messages, evidence=evidence, lang=lang)


def chat_context(question: str, history: list,
                 settings: TurnSettings = None) -> TurnContext:
    """The conversational turn: no retrieval, no sources, no evidence.

    FULL history on purpose — the evidence isolation of a factual turn exists
    because old answers become pseudo-evidence there; here the conversation IS
    the subject ('what did you just say?').
    """
    lang = _detect_lang(question)
    system = assemble_chat(lang)
    messages = ([{"role": "system", "content": system}]
                + [{"role": m["role"], "content": m["content"]}
                   for m in (history or [])]
                + [{"role": "user", "content": question}])
    return TurnContext(system_prompt=system, messages=messages, user=question,
                       lang=lang)


# ---------------------------------------------------------------------------
# stage 3: the answer call
# ---------------------------------------------------------------------------
def _cap_kwargs(max_answer_tokens) -> dict:
    """kwargs for an answer call's output cap — empty when there is none, so
    the request carries no cap at all rather than an explicit null that
    OpenAI-compatible servers may reject."""
    return ({"max_completion_tokens": max_answer_tokens}
            if max_answer_tokens else {})


async def answer_turn(llm, messages: list, settings: TurnSettings, *,
                      on_token=None):
    """(answer text, truncated) from the streamed answer call.

    `on_token` is where the UI lives: an async callback per delta, so the
    caller streams into whatever it renders into and this function keeps no
    opinion about it. `truncated` is read off `finish_reason`, which only the
    last chunk carries and some servers omit entirely.
    """
    stream = await llm.chat.completions.create(
        model=settings.chat_model,
        **_cap_kwargs(settings.max_answer_tokens),
        messages=messages,
        stream=True,
        **(settings.call_kwargs or {}),
    )
    text, truncated = "", False
    async for part in stream:
        if part.choices and part.choices[0].delta.content:
            token = part.choices[0].delta.content
            text += token
            if on_token is not None:
                await on_token(token)
        if _finish_reason(part) == "length":
            truncated = True
    return text, truncated


# ---------------------------------------------------------------------------
# stage 4: verification, and what the user is told about it
# ---------------------------------------------------------------------------
def verification_step(res, truncated: bool = False) -> str:
    """The verification step's own report."""
    return "\n".join(
        [f"status: {res.status}",
         "claims: " + ", ".join(f"{k} {n}" for k, n in res.counts().items())]
        + [f"- {v.status}: {_claim_texts([v], width=90)} — {v.reason}"
           for v in res.failures[:6]] + list(res.notes)
        + ([_TRUNCATION_STEP_NOTE] if truncated else []))


def run_verification(answer: str, evidence, settings: TurnSettings,
                     truncated: bool = False):
    """(result, step output, abstain banner) — the audit, never an edit.

    `verify_answer` hands back the text it was given (eac4c94 deleted the
    adopt-if-clean repair), so the only thing this can add to a message is the
    banner an abstained answer is LED by. It never raises either: a verifier
    that breaks answering is strictly worse than no verifier.
    """
    if evidence is None:
        return None, None, None
    try:
        res = verify.verify_answer(answer, evidence, use_llm=settings.verify_llm)
    except Exception as e:                             # noqa: BLE001
        log.warning("verification skipped: %s: %s", type(e).__name__, e,
                    exc_info=True)
        return None, None, None
    banner = _abstain_banner(res) if res.status == "abstain" else None
    return res, verification_step(res, truncated), banner


def sources_report(hits: list, answer: str, res, note_blocks) -> SourcesReport:
    """The 📎 sources line, its ordering and its warnings.

    Order by what the answer actually CITED: cited (doc, page) pairs first,
    uncited retrieved hits after. With verification on the citations are the
    verifier's own page-aware parse; with it off, the citation regex.
    """
    if not hits:
        return SourcesReport(lines=_verification_lines(res))
    if res is not None:
        cited_docs = {d for d, _ in res.sources}
        cited_keys = {(d, p) for d, p in res.sources if p}
    else:
        cited_docs = set(re.findall(r"\[([0-9]{1,3}_[\w.\-]+)", answer or ""))
        cited_keys = set()

    def _cited(h):
        return any(h.doc_id.startswith(c[:24]) or c.startswith(h.doc_id[:24])
                   for c in cited_docs)

    def _cited_page(h):
        return any(h.page == p and (h.doc_id.startswith(d[:24])
                                    or d.startswith(h.doc_id[:24]))
                   for d, p in cited_keys)
    # the exact cited page leads its document's other pages; with no
    # page-level citations (VERIFY=0) this is the old doc-level sort
    hits = sorted(hits, key=lambda h: (not _cited(h), not _cited_page(h)))
    shown = [h for h in hits if _cited(h)] or hits
    sources = ", ".join(sorted({f"{h.doc_id} p.{h.page}" if h.page else h.doc_id
                                for h in shown}))
    # Runs on EVERY path. The verifier only sees text that survives claim
    # extraction, so a hedge sentence ('the excerpts do not state X [doc,
    # p. 41]') carries an invented page past it untouched; this check reads
    # the raw answer. Whatever the verifier already flagged is dropped, so the
    # same page is never reported twice in two wordings.
    bad_cites = _invalid_citations(answer or "", hits, _note_pages(note_blocks))
    if res is not None:
        flagged = _verifier_flagged_cites(res)
        bad_cites = [b for b in bad_cites if _cite_key(b) not in flagged]
    return SourcesReport(hits=hits, shown=shown, sources=sources,
                         bad_citations=bad_cites,
                         lines=_verification_lines(res))


# ---------------------------------------------------------------------------
# the driver
# ---------------------------------------------------------------------------
@dataclass
class TurnHooks:
    """The optional callbacks a caller with a UI supplies.

    Every one of them is optional, and a caller that supplies none (the eval
    harness) gets the same turn with nothing rendered.

    step             async (name, output) -> None: one named step's output
    on_token         async (token) -> None: a delta of the streamed answer
    answer_complete  async (text) -> None: the answer is finished, before the
                     audit that may prefix a banner to it
    offload          (fn) -> async callable: run blocking work off the event
                     loop (chainlit's `cl.make_async`); absent, it runs inline
    """
    step: Optional[Callable] = None
    on_token: Optional[Callable] = None
    answer_complete: Optional[Callable] = None
    offload: Optional[Callable] = None

    async def emit(self, name: str, output):
        if self.step is not None and output is not None:
            await self.step(name, output)

    async def run(self, fn, *a, **kw):
        if self.offload is not None:
            return await self.offload(fn)(*a, **kw)
        return fn(*a, **kw)


_NO_HOOKS = TurnHooks()


async def run_turn(question: str, *, history=None, retriever=None, llm=None,
                   settings: TurnSettings = None, memory=None,
                   cited_docs=None, hooks: TurnHooks = None) -> TurnResult:
    """One turn, end to end: `chainlit_app.main()` with the rendering lifted
    out into `hooks` and the sources block returned rather than sent.

    Returns a `TurnResult` whose `mode` is 'chat' (answered from the
    conversation), 'guard' (an identifier the corpus does not have — no model
    call at all) or 'retrieve'.
    """
    settings = settings or TurnSettings.from_config()
    hooks = hooks or _NO_HOOKS
    history = list(history or [])
    result = TurnResult(question=question)

    plan = await plan_turn(question, history, retriever, llm, settings,
                           memory=memory, cited_docs=cited_docs)
    result.plan = plan
    await hooks.emit("evidence matrix", plan.matrix_step)

    if plan.guard_answer is not None:
        result.answer = plan.guard_answer
        result.context = TurnContext(lang=plan.lang)
        _log_turn(result, settings)
        return result

    if plan.mode == "chat":
        result.context = chat_context(question, history, settings)
        t0 = time.perf_counter()
        result.answer, result.truncated = await answer_turn(
            llm, result.context.messages, settings, on_token=hooks.on_token)
        result.answer_s = round(time.perf_counter() - t0, 3)
        # No verification step on this path to carry the note, so the marker is
        # the whole report — streamed in, so what the user sees and what enters
        # history are the same string.
        if result.truncated:
            marker = _truncation_marker(result.context.lang)
            result.answer += marker
            if hooks.on_token is not None:
                await hooks.on_token(marker)
        if hooks.answer_complete is not None:
            await hooks.answer_complete(result.answer)
        _log_turn(result, settings)
        return result

    await hooks.emit("retrieval query", plan.retrieval_step)
    t0 = time.perf_counter()
    context = await hooks.run(build_context, question, plan, retriever, settings)
    result.context_s = round(time.perf_counter() - t0, 3)
    result.context = context

    t0 = time.perf_counter()
    result.answer, result.truncated = await answer_turn(
        llm, context.messages, settings, on_token=hooks.on_token)
    result.answer_s = round(time.perf_counter() - t0, 3)
    if hooks.answer_complete is not None:
        await hooks.answer_complete(result.answer)

    # Claim-level audit of what was just written, BEFORE it becomes history: an
    # abstained answer enters history banner and all, so the next turn's
    # conductor remembers the caveat rather than a bare unverified claim.
    if context.evidence is not None:
        t0 = time.perf_counter()
        res, step, banner = await hooks.run(
            run_verification, result.answer, context.evidence, settings,
            result.truncated)
        result.verify_s = round(time.perf_counter() - t0, 3)
        result.verification, result.verification_step = res, step
        await hooks.emit("verification", step)
        if banner:
            result.abstain_banner = banner
            result.answer = banner + "\n\n" + result.answer
    # The marker goes on AFTER the audit, and that is deliberate: it is the
    # system's own line, not the model's, and claim extraction must never see
    # it. It still lands in the answer before history is written, so the next
    # turn's conductor knows the answer was cut short.
    if result.truncated:
        result.answer = (result.answer or "").rstrip() + _truncation_marker(
            context.lang)

    result.sources = sources_report(context.hits, result.answer,
                                    result.verification, context.note_blocks)
    result.memory_update = {
        "asked": _asked_doc_ids(question),
        "resolved": _memory_docs_for_turn(plan.search_queries, plan.plan),
        "cited": _answer_cited_doc_ids(result.answer, result.verification),
    }
    _log_turn(result, settings)
    return result


# ---------------------------------------------------------------------------
# one structured line per turn
# ---------------------------------------------------------------------------
def _clip(text, width: int = 300) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= width else text[:width] + "…"


def _log_turn(result: TurnResult, settings: TurnSettings) -> None:
    """ONE INFO line per turn, key=value, JSON-quoted where a value can hold a
    space. Never the API key and never excerpt text: the queries, the notes
    that fired and the counts are the turn's shape, and the shape is what a
    production incident needs.
    """
    if not log.isEnabledFor(logging.INFO):
        return
    plan = result.plan or TurnPlan()
    ctx = result.context or TurnContext()
    notes = [n for n, fired in (("registry", ctx.reg_note), ("year", ctx.year_note),
                                ("board", ctx.board_note),
                                ("coverage", ctx.coverage_note),
                                ("matrix", ctx.matrix_block)) if fired]
    probe_pages = [f"{h.doc_id}:{h.page}" for h in (ctx.probe_hits or [])]
    section_pages = [f"{h.doc_id}:{h.page}" for h in (ctx.section_hits or [])]
    guards = {k: v for k, v in (plan.guards or {}).items() if v}
    fields = [
        ("mode", result.mode),
        ("lang", ctx.lang),
        ("conductor", "off" if not settings.conductor else
         ("failed" if plan.conductor_failed else
          ("called" if plan.conductor_called else "skipped"))),
        ("conductor_raw", _clip(plan.conductor_raw) if plan.conductor_raw else None),
        ("queries", json.dumps([[sq.get("q"), sq.get("doc")]
                                for sq in (plan.search_queries or [])],
                               ensure_ascii=False)),
        ("guards", json.dumps(guards, ensure_ascii=False) if guards else None),
        ("planner", "matrix" if plan.matrix_block else
         ("matrix-failed" if plan.planner_matrix_failed else
          ("intent-vetoed" if plan.planner_detected and not plan.planner_intent_ok
           else "off" if not settings.planner else "none"))),
        ("decomposed", plan.decomposed),
        ("weak_signal", ctx.weak_signal),
        ("notes", ",".join(notes) or "-"),
        ("conflict_probe_pages", ",".join(probe_pages) or "-"),
        ("section_probe_pages", ",".join(section_pages) or "-"),
        ("hits", len(ctx.hits or [])),
        ("truncated", result.truncated),
        ("verify", getattr(result.verification, "status", None)
         or ("off" if not settings.verify else "none")),
        ("bad_citations", len((result.sources.bad_citations
                               if result.sources else []) or [])),
        ("answer_chars", len(result.answer or "")),
        # per-call / per-stage wall clock, in the order the turn spends it
        ("conductor_s", plan.conductor_latency_s),
        ("context_s", result.context_s),
        ("answer_s", result.answer_s),
        ("verify_s", result.verify_s),
    ]
    log.info("turn " + " ".join(
        f"{k}={v if not isinstance(v, str) or (v and ' ' not in v) else json.dumps(v, ensure_ascii=False)}"
        for k, v in fields if v is not None))


# ---------------------------------------------------------------------------
# resumed threads
# ---------------------------------------------------------------------------
def history_from_steps(steps) -> list:
    """Rebuild the conductor's memory from persisted steps.

    Without this, the first follow-up in a resumed thread regresses to the
    starved-decomposer bug: pronouns unresolvable, cited doc ids invisible.
    Sources lines (📎) are UI furniture, not conversation — skipped.
    """
    steps = sorted(steps or [], key=lambda s: s.get("createdAt") or "")
    history = []
    for st in steps:
        out = (st.get("output") or "").strip()
        if not out or out.startswith("📎"):
            continue
        if st.get("type") == "user_message":
            history.append({"role": "user", "content": out})
        elif st.get("type") == "assistant_message":
            history.append({"role": "assistant", "content": out})
    return history[-_HISTORY_TURNS:]
