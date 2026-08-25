"""Chainlit chat over the indexed GCF corpus.

Run:   chainlit run src/gcf_qna/app/chainlit_app.py
Needs: pip install -e ".[app]"  and OPENAI_API_KEY in the environment
       (never hardcoded — the old repo leaked five keys that way).
       Set OPENAI_BASE_URL to target any OpenAI-compatible server instead
       (e.g. LM Studio); then the key may be empty.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import threading
import time
from typing import Optional

import chainlit as cl
from chainlit.types import ThreadDict

from gcf_qna import config
from gcf_qna.boards import BOARD_YEARS, board_of, year_of
from gcf_qna.rag import registry
# ONE definition of the identifier patterns, shared with the registry: a second
# copy of the FP one here drifted (it lacked the trailing boundary and the
# zero-strip) and resolved 'fp2023' to FP202's document. _FP_RE stays as the
# module-local name every call site (and test) already uses.
from gcf_qna.rag.registry import FP_RE as _FP_RE
from gcf_qna.rag.registry import _BOARD_CODE_RE
from gcf_qna.app.highlight import annotated_page
from gcf_qna.rag import Embedder, Retriever, load_index
from gcf_qna.rag import planner, verify
from gcf_qna.rag.ground import ground_chunk

from gcf_qna.app.prompts import (CONDUCTOR_PROMPT, SYSTEM_PROMPT, assemble,
                                 assemble_chat)


def _doc_label(doc_id: str, page) -> str:
    """Citation header with precomputed board/year: the model reads dates
    instead of deriving them. Uses the shared boards parser, which handles
    every corpus id format incl. '72_GCF_B.35_02_Add.05...' (review #4)."""
    label = doc_id + (f", p. {page}" if page else "")
    b, y = board_of(doc_id), year_of(doc_id)
    if b and y:
        label += f" — B.{b}, {y}"
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


def _span_text(labels: list, prefix: str = "") -> str:
    """'2015, 2016' but '2015–2025' once a contiguous run gets long."""
    if len(labels) > 3 and labels[-1] - labels[0] == len(labels) - 1:
        return f"{prefix}{labels[0]}–{prefix}{labels[-1]}"
    return ", ".join(f"{prefix}{v}" for v in labels)


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
    lines = []
    for y in sorted(years):
        rows = [r for r in registry.by_year(y) if r.get("fp")]
        if rows and detailed:
            def _money(r):
                return f" ({r['gcf_financing']} GCF)" if r.get("gcf_financing") else ""
            fps = "; ".join(f"FP{r['fp']}{_money(r)}" for r in rows)
            lines.append(f"{y} — {len(rows)} proposals: {fps}.")
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
            lines.append(f"{y} — {len(rows)} proposals{span}.")
        else:
            boards = ", ".join(f"B.{b}" for b, yy in sorted(BOARD_YEARS.items())
                               if yy == y)
            lines.append(f"{y} — no registered proposals"
                         + (f" (boards {boards})." if boards else
                            " (no board meeting that year in this corpus)."))
    tail = (" Retrieved excerpts dated " + ys + ": "
            + "; ".join(_doc_label(h.doc_id, h.page) for h in matched) + "."
            if matched else
            f" None of the retrieved excerpts are dated {ys}; answer year-level "
            f"questions from the registry list above and say so.")
    note = ("Note (computed from the corpus registry, which is complete — "
            "this list is authoritative, unlike the excerpts): "
            + " ".join(lines) + tail)
    if outside:          # e.g. "before 2015 or in 2020": half is out of range
        note += " " + _outside_corpus_note(outside)
    return matched + rest, note


def _board_range_note(question: str):
    """A board meeting outside the corpus range deserves a definitive 'no',
    not an excerpt-scoped shrug ('B.44?' has no year token, so _year_assist
    never fires for it)."""
    lo, hi = min(BOARD_YEARS), max(BOARD_YEARS)
    out = sorted({f"B.{int(m.group(1))}" for m in
                  re.finditer(r"\bb\.?\s?(\d{1,2})\b", question.lower())
                  if int(m.group(1)) not in BOARD_YEARS})
    if not out:
        return None
    return (f"Note (computed): {', '.join(out)} is not in this corpus, which "
            f"covers board meetings B.{lo} ({BOARD_YEARS[lo]}) through "
            f"B.{hi} ({BOARD_YEARS[hi]}) completely. State this definitively.")


_COVERAGE_ASK_RE = re.compile(
    r"which board meetings|what board meetings|which years|what years|"
    r"how many (?:[\w-]+ ){0,2}(?:proposals|documents)|"
    r"combien de (?:[\w-]+ ){0,2}(?:propositions|documents)|"
    r"quelles (?:r[ée]unions|ann[ée]es)", re.I)
_CORPUS_TOKEN_RE = re.compile(r"corpus|collection|dataset|base documentaire",
                              re.I)


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
    lo, hi = min(BOARD_YEARS), max(BOARD_YEARS)
    counts = {y: len([r for r in registry.by_year(y) if r.get("fp")])
              for y in sorted(set(BOARD_YEARS.values()))}
    total = sum(counts.values())
    per_year = "; ".join(f"{y}: {n}" for y, n in sorted(counts.items()))
    return ("Note (computed from the corpus registry, which is complete — "
            "authoritative, unlike the excerpts): this corpus covers Green "
            f"Climate Fund board meetings B.{lo} ({BOARD_YEARS[lo]}) through "
            f"B.{hi} ({BOARD_YEARS[hi]}) completely — {total} funding-proposal "
            f"documents, one per proposal. Proposals per year: {per_year}. "
            "Answer corpus-coverage questions from this note.")


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
        have = set(_NOTE_DOC_RE.findall(note or ""))
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


async def _verify_reply(reply, evidence: dict):
    """Audit a finished answer; returns the verification result, or None when
    the verification could not run.

    The audit does not edit the answer: verify_answer hands back the text it was
    given (eac4c94), so the only thing this can add to the message is the
    abstain banner. It never raises either — a verifier that breaks answering is
    strictly worse than no verifier, so every failure path ends with the answer
    exactly as the model wrote it.
    """
    original = reply.content
    try:
        async with cl.Step(name="verification") as step:
            res = await cl.make_async(verify.verify_answer)(
                original, evidence, use_llm=config.VERIFY_LLM)
            step.output = "\n".join(
                [f"status: {res.status}",
                 "claims: " + ", ".join(f"{k} {n}" for k, n in res.counts().items())]
                + [f"- {v.status}: {_claim_texts([v], width=90)} — {v.reason}"
                   for v in res.failures[:6]] + list(res.notes))
        # 'abstain' — every fact-bearing claim failed — is the one status that
        # touches the message at all, and it PREFIXES the model's own body
        # rather than replacing any part of it.
        if res.status == "abstain":
            reply.content = _abstain_banner(res) + "\n\n" + original
            await reply.update()
        return res
    except Exception as e:                        # noqa: BLE001
        reply.content = original
        print(f"verification skipped: {type(e).__name__}: {e}", flush=True)
        return None


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


def _index_dir():
    return config.INDEX_DIR / os.getenv("INDEX_NAME", "default")


# One retriever per process, shared by every chat session: the FAISS index is
# ~730 MB on disk and the embedder holds GPU state — loading them per session
# made the first question of every chat pay ~1 min of cold start.
_retriever: Optional[Retriever] = None
_retriever_meta: dict = {}
_retriever_lock = threading.Lock()


def get_retriever() -> Optional[Retriever]:
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            idx_dir = _index_dir()
            if not (idx_dir / "index.faiss").exists():
                return None
            t0 = time.perf_counter()
            index, chunks, cfg = load_index(idx_dir)
            embedder = Embedder(cfg.get("embedding_model"))
            embedder.encode(["warmup"])   # load weights + CUDA context now
            _retriever = Retriever(index, chunks, embedder, index_dir=idx_dir)
            _retriever_meta.update(cfg)
            print(f"retriever ready: {cfg.get('n_chunks')} chunks, "
                  f"{cfg.get('embedding_model')} in {time.perf_counter() - t0:.1f}s",
                  flush=True)
    return _retriever


# Warm up in the background at server start, so even the first session's first
# question hits a hot retriever. PRELOAD=0 disables (e.g. for quick UI work).
if os.getenv("PRELOAD", "1") == "1":
    threading.Thread(target=get_retriever, daemon=True).start()


# ---------------------------------------------------------------------------
# Conversation history: SQLite-backed thread persistence + auth.
# Chainlit's sidebar (threads, resume, feedback) activates when a data layer
# AND authentication are configured. Threads live in data/app.db; element
# files (evidence images) are copied under public/app_files/ so resumed
# threads render across restarts. Schema: scripts/init_appdb.py.
# ---------------------------------------------------------------------------
_data_layer_instance = None


def _make_sqlite_layer():
    """SQLAlchemyDataLayer with SQLite-shape normalization.

    The layer targets Postgres, whose driver auto-parses JSONB -> dict and
    BOOLEAN -> bool. aiosqlite returns raw TEXT/int, and chainlit's frontend
    is written against the Postgres shape — replayed assistant messages
    render blank when step.metadata arrives as a string. Normalize at the
    one read path both the sidebar and thread replay flow through.
    """
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

    class SqliteNormalizedLayer(SQLAlchemyDataLayer):
        async def get_all_user_threads(self, user_id=None, thread_id=None):
            threads = await super().get_all_user_threads(user_id, thread_id)
            for t in threads or []:
                if isinstance(t.get("metadata"), str):
                    try:
                        t["metadata"] = json.loads(t["metadata"] or "{}")
                    except ValueError:
                        t["metadata"] = {}
                for st in t.get("steps") or []:
                    if isinstance(st.get("metadata"), str):
                        try:
                            st["metadata"] = json.loads(st["metadata"] or "{}")
                        except ValueError:
                            st["metadata"] = {}
                    for k in ("streaming", "waitForAnswer", "isError", "defaultOpen"):
                        if isinstance(st.get(k), int):
                            st[k] = bool(st[k])
            return threads

    return SqliteNormalizedLayer


@cl.data_layer
def _data_layer():
    global _data_layer_instance
    if _data_layer_instance is None:
        from gcf_qna.app.storage_local import LocalStorageClient
        SQLAlchemyDataLayer = _make_sqlite_layer()
        if not config.APP_DB.exists():
            # first boot: create the schema (idempotent DDL)
            import runpy
            runpy.run_path(str(config.PROJECT_ROOT / "scripts" / "init_appdb.py"),
                           run_name="__main__")
        _data_layer_instance = SQLAlchemyDataLayer(
            conninfo=f"sqlite+aiosqlite:///{config.APP_DB}",
            storage_provider=LocalStorageClient(),
        )
    return _data_layer_instance


@cl.password_auth_callback
async def auth(username: str, password: str) -> Optional[cl.User]:
    from gcf_qna.app import accounts
    users = accounts.parse_env_users()
    expected = users.get(username.strip())
    if expected and hmac.compare_digest(password, expected):
        return cl.User(identifier=username.strip())
    # Self-registered accounts (scrypt-hashed, data/app.db). Both halves of
    # the check block: scrypt is ~100 ms of deliberate CPU and sqlite3 is
    # synchronous. Run on the event loop they freeze token streaming and
    # websocket heartbeats for EVERY connected session, so offload to a
    # worker thread — and throttle, since /login (unlike /register) has no
    # rate limit of its own to stop a login flood from renting that CPU.
    if not accounts.login_allowed(username):
        return None
    if await cl.make_async(accounts.check_login)(username, password):
        return cl.User(identifier=username.strip())
    return None


# Self-registration page + API (/register); ALLOW_SIGNUP=0 disables.
try:
    from gcf_qna.app.register import mount as _mount_register
    _mount_register()
except Exception as _e:   # never let signup wiring break the chat app
    print(f"signup routes not mounted: {_e}", flush=True)


def _history_from_thread(thread: ThreadDict) -> list:
    """Rebuild the condenser's memory from persisted steps.

    Without this, the first follow-up in a resumed thread regresses to the
    starved-decomposer bug: pronouns unresolvable, cited doc ids invisible.
    Sources lines (📎) are UI furniture, not conversation — skipped.
    """
    steps = sorted(thread.get("steps") or [], key=lambda s: s.get("createdAt") or "")
    history = []
    for st in steps:
        out = (st.get("output") or "").strip()
        if not out or out.startswith("📎"):
            continue
        if st.get("type") == "user_message":
            history.append({"role": "user", "content": out})
        elif st.get("type") == "assistant_message":
            history.append({"role": "assistant", "content": out})
    return history[-12:]


@cl.on_chat_resume
async def on_resume(thread: ThreadDict):
    retriever = await cl.make_async(get_retriever)()
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("history", _history_from_thread(thread))


@cl.on_chat_start
async def start():
    if not os.getenv("OPENAI_API_KEY") and not config.OPENAI_BASE_URL:
        await cl.Message(
            content="⚠️ `OPENAI_API_KEY` is not set (and no `OPENAI_BASE_URL` for a "
                    "local server). Copy `.env.example` to `.env`, fill it in, restart."
        ).send()
        return
    idx_dir = _index_dir()
    if not (idx_dir / "index.faiss").exists():
        await cl.Message(
            content=f"⚠️ No index found at `{idx_dir}`.\n"
                    "Build one first:\n```\npython scripts/build_index.py "
                    "--source data/extracted/vlm/qwen_qwen2.5-vl-7b --name default\n```"
        ).send()
        return

    retriever = await cl.make_async(get_retriever)()
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("history", [])
    await cl.Message(
        content="👋 **Welcome to SSA CHATBOT.** Ask questions, compare projects, and explore "
                "insights across Green Climate Fund proposals. Every answer is grounded in the "
                "source documents, with citations you can review and verify."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    retriever = cl.user_session.get("retriever")
    if retriever is None:
        await cl.Message(content="Session not initialised — fix the startup warning first.").send()
        return

    import openai

    client = openai.AsyncOpenAI(base_url=config.OPENAI_BASE_URL or None)
    history = cl.user_session.get("history") or []

    # Deterministic comparison planning (plan step 4), behind config.PLANNER.
    # When the question NAMES its documents and asks about their fields,
    # nothing about the plan needs a model: the identifiers resolve through the
    # registry and the fields through a keyword map, so the plan is built
    # before any retrieval runs and every (document, field) cell is rendered —
    # including the empty ones a fan-out silently drops. detect() returns None
    # for follow-ups, single-id and purely semantic questions; _planner_intent
    # returns non-comparative two-id prose to the conductor, which reads the
    # conversation and is the better judge of it.
    plan = planner.detect(message.content) if config.PLANNER else None
    if plan is not None and not _planner_intent(message.content, plan):
        plan = None

    search_queries = [{"q": message.content, "doc": None}]
    mode = "retrieve"

    async def run_conductor():
        """Follow-ups carry references the embedder cannot resolve ("how does
        THAT compare..."), so retrieval on the raw message fetches noise. One
        LLM call rewrites the message against recent history — into a single
        standalone query, or, for comparisons/aggregations over documents named
        in the conversation, one doc-scoped sub-query per entity (per-document
        quota). See docs/query-decomposition.html. Best-effort: any failure
        falls back to the raw message; the original wording still goes to the
        answer model.

        A function, not a straight-line block, because the planner path skips
        it and may still need it: a matrix that fails to build falls back here,
        after the point the call would normally have happened.
        """
        nonlocal mode, search_queries
        try:
            # citation lists are extracted from FULL history — truncation once
            # left the conductor blind to most cited docs.
            convo = ("\n".join(f"{m['role']}: {m['content'][:1200]}" for m in history[-6:])
                     if history else "((no prior conversation))")
            cited = _cited_docs(history)
            if cited:
                convo += "\nDocuments cited in conversation: " + ", ".join(cited[-12:])
            resp = await client.chat.completions.create(
                model=config.CHAT_MODEL,
                max_completion_tokens=300,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": CONDUCTOR_PROMPT},
                    {"role": "user", "content":
                        f"Conversation:\n{convo}\n\nLatest message: {message.content}"},
                ],
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            if data.get("mode") == "chat":
                mode = "chat"
            parsed = []
            for item in (data.get("queries") or [])[:6]:
                if isinstance(item, str) and item.strip():
                    parsed.append({"q": item.strip(), "doc": None})
                elif isinstance(item, dict) and (item.get("q") or "").strip():
                    parsed.append({"q": item["q"].strip(), "doc": item.get("doc") or None})
            # Deterministic guards against rewrite contamination: keep the
            # scoping that is defensible, re-scope the sub-query when the tag
            # goes (see _rescope_items).
            parsed = _rescope_items(parsed, message.content, cited)
            if parsed:
                search_queries = parsed
        except Exception:
            pass

    if plan is None and config.CONDUCTOR:
        await run_conductor()

    if mode == "chat":
        # conversational/meta turn: answer from history, no retrieval,
        # no sources, no evidence images. FULL history on purpose — the
        # evidence isolation below applies to factual turns, where old
        # answers become pseudo-evidence; here the conversation IS the
        # subject ('what did you just say?').
        reply = cl.Message(content="")
        stream = await client.chat.completions.create(
            model=config.CHAT_MODEL,
            max_completion_tokens=config.MAX_ANSWER_TOKENS,
            messages=[{"role": "system",
                       "content": assemble_chat(_detect_lang(message.content))}] + history +
                     [{"role": "user", "content": message.content}],
            stream=True,
        )
        async for part in stream:
            if part.choices and part.choices[0].delta.content:
                await reply.stream_token(part.choices[0].delta.content)
        await reply.send()
        history += [{"role": "user", "content": message.content},
                    {"role": "assistant", "content": reply.content}]
        cl.user_session.set("history", history[-12:])
        return

    # Named FPs that resolve nowhere must not fall back to unscoped semantic
    # retrieval (review: FP999 refusals looked "grounded" in unrelated docs).
    try:
        from gcf_qna.rag.registry import load as _reg_load, resolve_fps
        if _reg_load():
            _resolved, _missing = resolve_fps(message.content)
            if _missing and not _resolved:
                lang = _detect_lang(message.content)
                miss = ", ".join(f"FP{n}" for n in _missing)
                text = (f"{miss} n'existe pas dans le corpus (registre de 273 documents)."
                        if lang == "French" else
                        f"{miss} does not exist in this corpus (273-document registry).")
                await cl.Message(content=text).send()
                history += [{"role": "user", "content": message.content},
                            {"role": "assistant", "content": text}]
                cl.user_session.set("history", history[-12:])
                return
    except Exception:
        pass

    # The evidence matrix, built AFTER the abstention above: a message whose
    # every identifier is unknown is refused per document there, and one that
    # only partly resolves falls through to here, where the matrix carries a
    # missing-document row for each id the registry does not have.
    matrix_block = None
    if plan is not None:
        async with cl.Step(name="evidence matrix") as step:
            try:
                matrix = await cl.make_async(planner.build_matrix)(plan, retriever)
                if not any(c.status not in ("missing", "missing-document")
                           for c in matrix.cells):
                    raise ValueError("no cell carries evidence")
                matrix_block = planner.render(matrix)
                step.output = (f"{plan.trigger}\n"
                               + ", ".join(f"{k} {v}" for k, v in matrix.counts.items()))
            except Exception as e:            # noqa: BLE001 — never user-facing
                step.output = (f"planner skipped for this turn "
                               f"({type(e).__name__}: {e}); falling back to the "
                               f"LLM conductor")
        if matrix_block is None:
            # Fallback: the conductor call that the planner path skipped
            # happens now, and this turn proceeds exactly as PLANNER=0 would.
            plan = None
            if config.CONDUCTOR:
                await run_conductor()       # a late mode='chat' is ignored: the
                                            # chat branch is already behind us,
                                            # and retrieving is the safe default
        else:
            # Authoritative stems, straight from the registry, and an English
            # query per document built from the fields the plan resolved (see
            # _plan_query — the raw message is the wrong query here, and the
            # conductor that would have translated it was skipped). The rewrite
            # guards have nothing to repair either — no model wrote these tags —
            # so _rescope_items/_resolve_doc_tags are skipped below.
            scoped = [{"q": _plan_query(plan, d), "doc": d.scope}
                      for d in plan.docs if not d.missing]
            search_queries = scoped or search_queries

    if plan is None:
        # Single-FP pre-scoping, on the unconditional path so it also covers the
        # conductor-off and conductor-failed fallbacks, where the raw message is
        # the only query and _rescope_items never ran. Already-tagged items are
        # untouched, so running it twice on the conductor path is a no-op.
        search_queries = _prescope_single_fp(search_queries, message.content)
        # Doc tags become retrieval filters here: resolve each one to its
        # authoritative corpus stem first, since B.27-era ids contain no FP
        # number and a bare 'fp152' filter would match nothing (_registry_doc).
        search_queries = _resolve_doc_tags(search_queries)

    # A one-document plan (its partner identifier resolves nowhere) still ships
    # the comparison rules: the answer has to say so, document by document.
    decomposed = len(search_queries) > 1 or plan is not None
    if decomposed or search_queries[0]["q"] != message.content:
        async with cl.Step(name="retrieval query") as step:
            step.output = "\n".join(
                f"{sq['q']}" + (f"   [{sq['doc']}]" if sq.get("doc") else "")
                for sq in search_queries)

    per_query = config.TOP_K if not decomposed else max(3, config.TOP_K // len(search_queries))
    weak_signal = True
    per_lists = []
    # The raw message rides along as `original`: whenever sq["q"] is a rewrite
    # of it (conductor translation, noise cleanup, pronoun resolution, or a
    # planner field query), the retriever lets the user's own wording help rank
    # pages INSIDE the document the rewrite chose. It never chooses documents —
    # see Retriever._probes. Only a turn that stayed on ONE query sends it: a
    # message that fanned out names every document it compares, so inside any
    # one of them it is a probe for the OTHERS' names and figures (measured:
    # it cost the three-way and two-way comparisons the registry-cited page
    # they had, and bought them nothing).
    original = message.content if len(search_queries) == 1 else None
    for sq in search_queries:
        got, conf = await cl.make_async(retriever.search_with_confidence)(
            sq["q"], per_query, sq.get("doc"), original)
        if conf >= config.MIN_DENSE_SCORE:
            weak_signal = False
        per_lists.append(got)
    # round-robin across queries so the global cap cannot starve the later
    # documents of a multi-doc turn (review cross-cutting #2)
    from itertools import zip_longest
    seen, hits = set(), []
    for tier in zip_longest(*per_lists):
        for h in tier:
            if h is None:
                continue
            key = (h.doc_id, h.page, h.text[:120])
            if key not in seen:
                seen.add(key)
                hits.append(h)
    hits = hits[:15]
    hits, year_note = _year_assist(message.content, hits)
    board_note = _board_range_note(message.content)
    if board_note:
        year_note = f"{year_note} {board_note}" if year_note else board_note
    coverage_note = _corpus_coverage_note(message.content)
    if coverage_note:
        year_note = (f"{year_note} {coverage_note}" if year_note
                     else coverage_note)
    context = "\n\n".join(
        f"[{_doc_label(h.doc_id, h.page)}] (score {h.score:.2f})\n{h.text}"
        for h in hits)
    if year_note:
        context = year_note + "\n\n" + context
    if weak_signal:
        context = ("Note: retrieval confidence for this question is LOW — the "
                   "excerpts below may not actually be relevant. Do not force an "
                   "answer from marginal matches; say plainly that the corpus "
                   "does not appear to cover this.\n\n") + context
    reg_note = None
    try:
        from gcf_qna.rag.registry import registry_note
        reg_note = registry_note(message.content)
        # …and a line for whatever else THIS turn resolved to. The question's
        # own words are not the only evidence of which document a turn is
        # about: a follow-up names none, and its resolved query names one.
        reg_note = _extend_registry_note(reg_note, search_queries)
        if reg_note:
            context = reg_note + "\n\n" + context
    except Exception:
        pass    # the registry is an enhancement, never a blocker
    if matrix_block:
        # ABOVE the registry note and the excerpts: the matrix is the complete
        # half of the evidence (every named document, every asked field, empty
        # cells included) and has to be read before the retrieved sample.
        context = matrix_block + "\n\n" + context
    # What the answer is allowed to have used: this turn's excerpts plus the
    # computed blocks prepended above (registry line, year/board note, evidence
    # matrix) — built here, from the same strings the context got, so the audit
    # cannot drift from what the model actually read.
    evidence = None
    if config.VERIFY:
        try:
            evidence = verify.build_evidence(
                hits, [n for n in (reg_note, year_note, matrix_block) if n])
        except Exception as e:                    # noqa: BLE001
            evidence = None                       # never blocks the answer
            print(f"verification evidence unavailable: {e}", flush=True)
    system_prompt = assemble(year=bool(year_note), registry=bool(reg_note),
                             comparison=decomposed, matrix=bool(matrix_block),
                             lang=_detect_lang(message.content))
    # Evidence isolation (plan step 1): this call gets the question, the
    # resolved-reference ids and THIS turn's excerpts — never prior answers.
    messages = _answer_messages(system_prompt, context, message.content,
                                _resolved_refs_note(search_queries, message.content))

    reply = cl.Message(content="")
    stream = await client.chat.completions.create(
        model=config.CHAT_MODEL,
        max_completion_tokens=config.MAX_ANSWER_TOKENS,
        messages=messages,
        stream=True,
    )
    async for part in stream:
        if part.choices and part.choices[0].delta.content:
            await reply.stream_token(part.choices[0].delta.content)
    await reply.send()

    # Claim-level audit of what was just written, BEFORE it becomes history: an
    # abstained answer enters history banner and all, so the next turn's
    # conductor remembers the caveat rather than a bare unverified claim.
    res = await _verify_reply(reply, evidence) if evidence is not None else None

    history += [
        {"role": "user", "content": message.content},
        {"role": "assistant", "content": reply.content},
    ]
    cl.user_session.set("history", history[-12:])  # keep the last 6 exchanges

    if hits:
        # order sources/evidence by what the answer actually cited (review #6):
        # cited (doc,page) pairs first, uncited retrieved hits after. With
        # verification on, the citations are re-parsed from the answer by
        # verify.cited_sources, which is page-aware; with it off, the same
        # regex over the answer as before.
        if res is not None:
            cited_docs = {d for d, _ in res.sources}
            cited_keys = {(d, p) for d, p in res.sources if p}
        else:
            cited_docs = set(re.findall(r"\[([0-9]{1,3}_[\w.\-]+)", reply.content or ""))
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
        # the raw answer. Whatever the verifier already flagged is dropped, so
        # the same page is never reported twice in two wordings.
        bad_cites = _invalid_citations(
            reply.content or "", hits,
            _note_pages([reg_note, year_note, matrix_block]))
        if res is not None:
            flagged = _verifier_flagged_cites(res)
            bad_cites = [b for b in bad_cites if _cite_key(b) not in flagged]
        if bad_cites:
            sources += ("\n⚠️ cited but not among retrieved pages (treat with "
                        "caution): " + "; ".join(bad_cites[:4]))
        for line in _verification_lines(res):
            sources += "\n" + line
        # Ground the citations: annotated page images with the cited passage
        # highlighted (green lines / blue table region). Dedupe by (doc, page),
        # cap at 3 pages so answers stay scannable.
        elements, seen = [], set()
        for h in hits:
            if not h.page or (h.doc_id, h.page) in seen:
                continue
            seen.add((h.doc_id, h.page))
            try:
                g = await cl.make_async(ground_chunk)(
                    {"doc_id": h.doc_id, "page": h.page, "text": h.text})
                img = await cl.make_async(annotated_page)(g) if g else None
            except Exception:
                g, img = None, None
            if img is not None:
                label = f"{h.doc_id} — p. {h.page}"
                if g and g.kind == "page":
                    label += " (page-level match)"
                elements.append(cl.Image(name=label, path=str(img), display="inline"))
            if len(elements) >= 3:
                break
        await cl.Message(content=f"📎 Sources: {sources}", elements=elements).send()
    else:
        # No excerpts means no sources line to hang the verdict on, and a
        # note-only answer is exactly the kind that needs one. It goes out in
        # the sources message's own shape: the leading 📎 is what keeps UI
        # furniture out of the conversation when a thread is resumed
        # (_history_from_thread), so a bare cl.Message here would come back as
        # an assistant turn.
        lines = _verification_lines(res)
        if lines:
            await cl.Message(
                content="📎 Sources: none retrieved\n" + "\n".join(lines)).send()
