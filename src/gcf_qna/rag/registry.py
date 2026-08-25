"""Document metadata registry: deterministic lookups for cover-page facts.

Built once by scripts/build_registry.py (LLM over cover pages + financing
section; board/year derived from doc ids). Serves the fact classes that
chunk retrieval handles worst: 'which entity implements FPnnn',
'proposals of 2023', financing-at-a-glance. Financing values are RAW
quoted strings from the documents — never normalized numbers.

Schema 2 (data/registry_v2.json, built by scripts/build_registry_v2.py) adds
provenance-aware facts on top, reached through facts()/canonical()/conflicts().
Each fact is a list of candidates carrying raw source text, normalized value,
currency, unit, page and template section, with one marked canonical and any
disagreeing figure elsewhere in the document marked conflicting. It is a
separate file with a separate cache: the v1 LOOKUPS (load/by_fp/by_year/
resolve_board_code) never read it.

``registry_note()`` is the one exception, and it is a presentation-layer one:
the note it writes for the answer model keeps every v1 cover-page field
(title, entity, countries, board, year — clean in schema 1) and prints the
MONEY fields from schema 2 instead, with the page and template section the
figure was read from, plus a warning line per figure the same document
contradicts elsewhere. Schema 2's optional ``meta_provenance`` adds the page
for the text fields too, printed as a section-less '(p.N)' beside the value.
When registry_v2.json is absent or unreadable — or simply predates either
addition — the note falls back, field by field, to the exact v1 string it
always printed.

On a SINGLE-identifier turn that line also SERVES the fields the question asks
for (``_served_bits``). ``planner.detect`` builds its evidence matrix only when
a question names two documents, so at arity one every field beyond the five
this line prints was reachable through retrieval alone: the matrix served
eighteen fields, the line served five, and the same fact came back page-cited
or not depending on how many documents the question happened to name. The
fields come from ``planner.fields_for`` — one detector for both arities — the
values and pointers from schema 2's own candidates, and three guards keep the
line a note rather than a dump: only the fields the question asks for, nothing
at all for a field the store does not hold (a confirmed absence is data the
registry does not yet have, and inventing one here would be the exact failure
this file exists to prevent), and nothing for a field the document contradicts
itself on — that one belongs to the conflict warning, which now covers every
field rather than only the money ones. A document whose extraction the builder
itself flagged (``suspect``, ``llm_fallback``) says so on its own line, in a
marker that publishes no page and no document, the way a cut list says it was
cut.

Three lookups run in the OTHER direction: ``by_country``, ``by_entity`` and
``by_board``, the inverse index H1/H2/H16 of docs/l1-l2-coverage-review.md
found missing, each with a ``registry_note`` trigger that fires only on a
question asking for a SET of proposals. The entity side normalises at lookup
time — 126 stored spellings, 70 organisations, three of the clusters ruled on
by the corpus owner — and never edits data/: the rows come back exactly as the
documents print them. The board side is the one the review calls an
ASYMMETRY rather than a gap: an out-of-range meeting was answered definitively
by the app while an in-range one, whose documents this registry holds, was
answered not at all.

``_extrema_note`` is the only note here that COMPUTES over the corpus rather
than listing it (H3/H3b): the smallest or largest figure of one money field,
ranked over schema 2's normalised values, refusing every figure whose currency
the document does not state — 'do not assume it is USD' is ``planner``'s own
comparability law — counting what it excluded and naming the excluded figures
that fall beyond its answer. It is also the only note here that prints a page
and a document stem, because an extremum is ONE figure in ONE document and
that is the shape the note-page readers can credit.

Every list any of these notes prints carries its true length, and a list cut
by a cap says it was cut (``_list_bit``, ``_listing``, ``_conflict_lines``).
That is F13's fix: the countries fragment used to print five of FP151's 44
values with no count and no ellipsis inside a note the prompt calls
authoritative, and the answer that said 'five' verified.
"""
from __future__ import annotations

import json
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from gcf_qna import config

_lock = threading.Lock()
_cache: Optional[Dict[str, dict]] = None


def load() -> Dict[str, dict]:
    global _cache
    with _lock:
        if _cache is None:
            p = config.DATA_DIR / "registry.json"
            _cache = (json.loads(p.read_text(encoding="utf-8")).get("documents", {})
                      if p.exists() else {})
    return _cache


# --- schema 2: provenance-aware facts (additive; v1 paths never read this) ---
_cache_v2: Optional[Dict[str, dict]] = None


def load_v2() -> Dict[str, dict]:
    """documents of data/registry_v2.json, or {} when it has not been built.

    Separate cache from load(): every v1 code path keeps reading registry.json
    even when the v2 file exists.
    """
    global _cache_v2
    with _lock:
        if _cache_v2 is None:
            p = config.DATA_DIR / "registry_v2.json"
            _cache_v2 = (json.loads(p.read_text(encoding="utf-8")).get("documents", {})
                         if p.exists() else {})
    return _cache_v2


def _row_v2(fp_or_stem) -> Optional[dict]:
    """Accepts a doc id, an FP number, 'FP274' or '274'."""
    rows = load_v2()
    if isinstance(fp_or_stem, str):
        if fp_or_stem in rows:
            return {"doc_id": fp_or_stem, **rows[fp_or_stem]}
        m = _FP_RE.search(fp_or_stem) or re.fullmatch(r"\s*0*(\d{1,3})\s*", fp_or_stem)
        if not m:
            return None
        fp_or_stem = int(m.group(1))
    hits = [{"doc_id": k, **v} for k, v in rows.items() if v.get("fp") == fp_or_stem]
    if not hits:
        return None
    hits.sort(key=lambda r: (f"fp{fp_or_stem}" not in r["doc_id"].lower(), r["doc_id"]))
    return hits[0]


def facts(fp_or_stem) -> Dict[str, List[dict]]:
    """{field: [candidate, ...]} for a document, or {} when it has no v2 row.

    Candidates keep their source text and page: {"raw", "value", "currency",
    "unit", "page", "section", "status"}.
    """
    row = _row_v2(fp_or_stem)
    return (row or {}).get("facts") or {}


def canonical(fp_or_stem, field: str) -> Optional[dict]:
    """The candidate parsed from the template section for `field`, or None.

    None also means "the document states it somewhere but not in a template
    section" — call facts() to see the other candidates.
    """
    return next((c for c in facts(fp_or_stem).get(field, [])
                 if c.get("status") == "canonical"), None)


def conflicts(fp_or_stem) -> Dict[str, List[dict]]:
    """{field: [conflicting candidate, ...]} — fields whose document prints a
    figure that disagrees with the canonical one. Empty dict when consistent."""
    out = {}
    for field, cands in facts(fp_or_stem).items():
        bad = [c for c in cands if c.get("status") == "conflicting"]
        if bad:
            out[field] = bad
    return out


def by_fp(n: int) -> Optional[dict]:
    rows = [{"doc_id": k, **v} for k, v in load().items() if v.get("fp") == n]
    if not rows:
        return None
    # prefer the package document named after the FP over status/mention docs
    # (case-folded: some stems are '72_GCF_B.35_..._FP203')
    rows.sort(key=lambda r: (f"fp{n}" not in r["doc_id"].lower(), r["doc_id"]))
    return rows[0]


def by_year(y: int) -> List[dict]:
    return sorted(({"doc_id": k, **v} for k, v in load().items() if v.get("year") == y),
                  key=lambda r: (r.get("fp") or 0, r["doc_id"]))


def by_board(n: int) -> List[dict]:
    """Every registry row whose document belongs to board meeting B.``n``.

    by_year's twin — the board is on every row — and the lookup H16 of
    docs/l1-l2-coverage-review.md calls an ASYMMETRY rather than a gap:
    'Which funding proposals were approved at B.44?' already gets a definitive
    answer, because 44 is outside the corpus and
    ``chainlit_app._board_range_note`` says so in as many words, while the same
    question about B.35 — seven documents this registry holds — got nothing at
    all. The out-of-range arm is left exactly where it is; this is the arm that
    was missing.

    The isinstance guard is ``_collect``'s: an error row holds no 'board', and
    a row can be null outright.
    """
    return sorted(({"doc_id": k, **v} for k, v in load().items()
                   if isinstance(v, dict) and v.get("board") == n),
                  key=lambda r: (r.get("fp") or 0, r["doc_id"]))


# --- note enrichment: v1 text fields, v2 money with provenance --------------

# Money fields a note may carry, in the order a conflict warning prefers them:
# the figures a reader asks about first, and the ones a wrong answer costs most.
_MONEY_FIELDS = ("gcf_funding_requested", "total_financing", "co_financing")
# The worst documents print four or five disagreeing figures. A note that lists
# them all stops being a note; two warnings — one per field, each naming at
# most two disagreeing prints — say 'this document contradicts itself, here is
# where' just as well.
_MAX_CONFLICT_LINES = 2
_MAX_CONFLICT_ALTS = 2

# --- fields the line SERVES at arity one ------------------------------------
# H7 / campaign Phase 1. A single-identifier turn gets no evidence matrix —
# `planner.detect` needs two identifiers — so every field registry v2 holds
# beyond the five this line already prints was reachable only through
# retrieval, at arity one, for the whole corpus: the matrix serves 18 fields,
# the line printed 5. The fields the QUESTION asks for are appended to the line
# below, read off the SAME v2 candidates the matrix cell would quote and
# carrying the SAME '(p.N, SECTION)' pointer, so a fact answered at both
# arities is answered with one value and one citation.
#
# Three guards, and each one is a defect this pass is not allowed to create:
#   * only the asked fields (`_asked_fields`), never every field the store
#     holds — a kitchen-sink line is not evidence, it is noise the model must
#     sift, and this line is what the prompt calls authoritative;
#   * an absent field appends NOTHING. Silence here is not an answer to the
#     ask; recording a CONFIRMED absence is Phase 3's data work, and a note
#     that invented 'not stated' from an empty candidate list would be
#     asserting a fact the registry does not hold;
#   * a field the document CONTRADICTS itself on is left to `_conflict_lines`,
#     which prints every disagreeing figure with its page. Printing the
#     canonical figure here as well would be the silent choice between two
#     figures that the conflict machinery exists to refuse.
#: Fields `_fmt` already prints; serving them again would double-print a fact.
_LINE_FIELDS = frozenset({
    "title", "countries", "accredited_entity",
    "gcf_funding_requested", "total_financing"})
#: field -> the label that HEADS its segment. Not decoration: `verify.
#: _field_lines` reads the value of a field off this one-line note by finding
#: the label at the head of its semicolon-separated part, and `verify.
#: _FIELD_LABELS` is the vocabulary that anchors it — so the label a field is
#: served under here is the label the verifier looks for there.
_SERVED_LABELS = {
    "executing_entity": "executing entity",
    "national_designated_authority": "national designated authority",
    "project_size": "project size",
    "co_financing": "co-financing",
    "instruments": "instruments",
    "financial_instruments": "financial instruments",
    "implementation_period": "implementation period",
    "lifespan": "lifespan",
    "ess_category": "ESS category",
    "mitigation_outcome": "mitigation outcome",
    "adaptation_outcome": "adaptation outcome",
    "beneficiaries_direct": "direct beneficiaries",
    "beneficiaries_indirect": "indirect beneficiaries",
}
#: Characters of one served value, and of one disagreeing print of a text
#: field. The money fields print short figures; `executing_entity` prints
#: whole paragraphs of A.20 (the longest in the corpus runs past 300
#: characters), and a note is one line per document. Cut values carry the
#: marker `_clip` puts on them.
_MAX_FIELD_CHARS = 120
#: Values a served LIST field prints before it truncates. Set ABOVE the
#: largest list the corpus actually holds (four instruments, FP176) on the same
#: reasoning as `_MAX_INVERSE_ROWS`: it is a backstop against a future corpus,
#: not a working cap — and when it does bite it says so.
_MAX_FIELD_VALUES = 6

# ---------------------------------------------------------------------------
# lists a note prints: a cut one must SAY it is cut, and carry the true count
# ---------------------------------------------------------------------------
# F13/P5 of docs/l1-l2-coverage-review.md §7.1. The countries fragment was
# ', '.join(r["countries"][:5]) — no count, no ellipsis — inside a note the
# prompt calls authoritative. Asked "FP151: how many countries does it cover?
# List them all", the system answered FIVE, cited the note, and
# classify_deterministic scored the claim SUPPORTED, because the evidence
# really did say it. The truth is 44. Nothing in the stack asked whether the
# evidence was COMPLETE, so the fix is to make the note say so itself, in both
# directions: every list the note prints carries its length, and a cut list
# says it is cut.
#: Values a `_fmt` list field prints before it truncates.
_MAX_LIST_VALUES = 5
#: Rows the year listing prints (unchanged), and rows an inverse listing does.
#: The inverse cap is set ABOVE the largest group the corpus actually holds
#: (UNDP, 41 documents) on purpose: an inverse note exists to answer 'which
#: proposals', and a listing that stopped one row short of complete would be
#: the very defect this pass is fixing. It is a backstop against a future
#: corpus, not a working cap — and when it does bite it says so.
_MAX_YEAR_ROWS = 12
_MAX_INVERSE_ROWS = 50
#: Characters of a title a listing keeps.
_MAX_TITLE_CHARS = 45
#: Tokens the longest indexed country/entity name can have.
_MAX_NAME_TOKENS = 8


def _clip(text: Optional[str], limit: int = _MAX_TITLE_CHARS) -> str:
    """A title shortened for a listing, with '…' when it really was cut."""
    text = str(text or "?")
    return text if len(text) <= limit else text[:limit] + "…"


def _list_bit(label: str, values: Sequence[str],
              cap: int = _MAX_LIST_VALUES) -> str:
    """'countries (2): Angola, Benin' — or, cut, the count and the marker.

    The count is printed in BOTH directions on purpose. A complete list reads
    'countries (2): ...' and a cut one reads 'countries (5 of 44 — list
    truncated): ..., …', so neither can be mistaken for the other and the TRUE
    count is on the line whichever way it went. Everything else about the
    fragment is unchanged: the label still HEADS its semicolon-separated
    segment, which is what ``verify._field_lines`` requires to read the value
    of a field off a one-line registry note, and what
    ``verify._FIELD_LABELS``' 'countr(y|ies)' anchor matches.
    """
    vals = [str(v).strip() for v in values if str(v or "").strip()]
    if len(vals) <= cap:
        return f"{label} ({len(vals)}): " + ", ".join(vals)
    return (f"{label} ({cap} of {len(vals)} — list truncated): "
            + ", ".join(vals[:cap]) + ", …")


def _listing(rows: Sequence[dict], cap: int) -> Tuple[str, str]:
    """(the 'FPn "title"; ...' listing, the '(+N more)' tail).

    One formatter for every listing a note prints — the year one and both
    inverse ones — so a cap can never be added to one and forgotten in
    another. The tail carries the number NOT printed; the caller carries the
    total, so the two together state the truth exactly once each.
    """
    shown = rows[:cap]
    listing = "; ".join(f'FP{r["fp"]} "{_clip(r.get("title"))}"' for r in shown)
    return listing, (f" (+{len(rows) - cap} more)" if len(rows) > cap else "")


def _v2_facts(doc_id: Optional[str]) -> Dict[str, List[dict]]:
    """facts() that cannot break a note.

    An absent, half-written or corrupt registry_v2.json must leave the note
    byte-identical to what schema 1 alone produced — the provenance is an
    upgrade, never a dependency.
    """
    if not doc_id:
        return {}
    try:
        return facts(doc_id) or {}
    except Exception:
        return {}


def _v2_meta(doc_id: Optional[str]) -> Dict[str, dict]:
    """``meta_provenance`` for a document, or ``{}`` — same never-break contract
    as ``_v2_facts``.

    The key is ADDITIVE and OPTIONAL: a registry_v2.json built before the
    cover-page provenance pass carries no ``meta_provenance`` at all, and an
    entry carries only the fields the builder actually found. Absent, partial,
    or the wrong type, the note must come out byte-identical to the one
    schema 1 alone produced — the provenance is an upgrade, never a
    dependency.
    """
    if not doc_id:
        return {}
    try:
        mp = (_row_v2(doc_id) or {}).get("meta_provenance")
        return mp if isinstance(mp, dict) else {}
    except Exception:
        return {}


def _meta_page(meta: Dict[str, dict], field: str) -> str:
    r"""``' (p.3)'`` for a cover-page fact the builder sourced, else ``''``.

    EXACTLY '(p.N)', with no section part, because that is the shape the two
    consumers credit. ``chainlit_app._note_pages`` and
    ``verify.note_page_scopes`` read a note line's pages with the SAME regex,
    byte for byte — ``r"\(p\.(\d{1,3})[,)]"`` — whose ``[,)]`` arm exists so a
    pointer can end at the closing paren instead of a ', SECTION' tail. So
    'accredited entity: X (p.3)' publishes (doc, 3) exactly as
    'GCF funding requested: 18.5 M USD (p.5, A.8)' publishes (doc, 5), and the
    model told to cite the page printed beside THAT fact cites a page the
    checker holds instead of guessing one (release-6 guessed p.3 on both merge
    traps: right page, invented citation, flagged everywhere).

    A page outside 1..999 prints nothing: those regexes cap at three digits,
    so a four-digit page would publish no scope and the printed pointer would
    invite exactly the flagged citation this is here to prevent. ``bool`` is
    rejected too — ``isinstance(True, int)`` is True, and 'p.True' is not a
    page.
    """
    hit = meta.get(field)
    if not isinstance(hit, dict):
        return ""
    page = hit.get("page")
    if not isinstance(page, int) or isinstance(page, bool) or not 1 <= page <= 999:
        return ""
    return f" (p.{page})"


def _usable(c: Optional[dict]) -> Optional[dict]:
    """A candidate is printable only with the two things it is here for: the
    source text and the page it was read from."""
    if not c or not c.get("raw") or not isinstance(c.get("page"), int):
        return None
    return c


def _canon2(f2: Dict[str, List[dict]], field: str) -> Optional[dict]:
    return _usable(next((c for c in f2.get(field, [])
                         if c.get("status") == "canonical"), None))


def _section(c: dict) -> str:
    """'rule:B.2(a)' -> 'B.2(a)'. The 'rule:' marker is builder bookkeeping
    (the page carried the figure but not the heading), not part of a pointer."""
    return str(c.get("section") or "?").split("rule:")[-1] or "?"


def _where(c: dict) -> str:
    """'(p.5, A.8)' — the shape verify.build_evidence keys a note line's page
    on, and the shape the answer model is expected to cite back."""
    return f"(p.{c['page']}, {_section(c)})"


def _money_bit(label: str, c: dict) -> str:
    if c.get("value") is None:
        # The page prints a scale word its own mantissa contradicts ('28,654
        # million USD'). The registry publishes no number for it and neither
        # does the note: the print is quoted and left ambiguous on purpose.
        return f'{label}: "{c["raw"]}" {_where(c)} (unit as printed is ambiguous)'
    return f"{label}: {c['raw']} {_where(c)}"


def _fig(c: dict) -> str:
    """'40,511,264 USD (p.7, A.8)' — a printed figure and where it is printed."""
    return f"{c['raw']} {_where(c)}"


def _amount(c: dict) -> str:
    """The source text of a figure, with the currency schema 2 recorded for it
    when the text itself carries none ('22,953' -> '22,953 USD').

    Only the extrema lines use this. Everywhere else a figure is quoted beside
    the field it answers, and the field says what the money is; a figure quoted
    inside a RANKING has to carry the currency it was ranked under, or the one
    line that must never be read as unit-free is exactly the one that is. The
    currency is schema 2's record, not a re-reading of the page — and a
    candidate whose currency schema 2 did NOT record prints bare, which is the
    ``NOT RANKED`` line's whole point.
    """
    raw = str(c.get("raw") or "")
    cur = c.get("currency")
    if (not cur or cur.casefold() in _fold(raw)
            or any(sym in raw for sym in "$\u20ac\u00a3")):
        return raw
    return f"{raw} {cur}"


def _planner():
    """The planner module, imported at CALL time.

    `planner` imports this module at import time — the board-code pattern and
    the resolver live here — so the reuse this file needs in the other
    direction (`fields_for`, `FIELD_ORDER`, `LIST_FIELDS`) can only be a
    deferred import. It is also the never-break contract every other v2
    reader here honours: a planner that failed to import must leave the note
    exactly as schema 1 alone produced it, not raise inside a note.
    """
    try:
        from gcf_qna.rag import planner
        return planner
    except Exception:                                          # noqa: BLE001
        return None


# A descriptive apposition after an identifier NAMES the document; it does not
# ask for a field. Measured on the 126-case gold set, where two questions ask
# for exactly one field each and mention a second field's word inside the
# phrase that names the document: 'FP152, the Global Subnational Climate Fund
# equity proposal' (asks: accredited entity) and 'FP259, the Pacific tuna
# adaptation programme' (asks: total financing). `fields_for` is calibrated for
# a MATRIX, where a spare column costs one line of a block that is already a
# block; on the one registry line the prompt calls authoritative, a spare
# segment is a fact asserted about a document nobody asked about. So the span
# is dropped from the question text before the fields are read off it.
#
# Narrowly: only a span that FOLLOWS an identifier, only up to the first
# sentence break, and only in this module — `fields_for`'s own rules and every
# matrix the planner builds are untouched by it.
_APPOSITION_RE = re.compile(
    r",\s*(?:the|a|an|le|la|les|un|une)\s+[^,?.;!]{0,80}", re.I)
def _drop_document_apposition(question: str) -> str:
    """'FP259, the Pacific tuna adaptation programme?' -> 'FP259?'"""
    out: List[str] = []
    pos = 0
    for m in _APPOSITION_RE.finditer(question or ""):
        if m.start() < pos or not _ENDS_WITH_ID_RE.search(question[:m.start()]):
            continue
        out.append(question[pos:m.start()])
        pos = m.end()
    out.append((question or "")[pos:])
    return "".join(out)


def _asked_fields(question: str) -> List[str]:
    """The fields THIS question asks for that the line does not already print.

    `planner.fields_for` is the detector, unchanged and unduplicated: the same
    keyword map, the same EN+FR rules, the same FIELD_ORDER output order that
    makes two spellings of one ask produce one answer. Its default set is
    refused here — 'tell me about FP254' names no field, and the four defaults
    are the four this line already prints — so a question with no field word
    appends nothing.
    """
    p = _planner()
    if p is None or not (question or "").strip():
        return []
    try:
        fields, used_default = p.fields_for(_drop_document_apposition(question))
    except Exception:                                          # noqa: BLE001
        return []
    if used_default:
        return []
    return [f for f in fields if f not in _LINE_FIELDS]


def _text_fig(c: dict) -> str:
    """`_fig` for a field whose value is text: the print, clipped with the
    marker a cut string carries, and where it is printed."""
    return f"{_clip(c['raw'], _MAX_FIELD_CHARS)} {_where(c)}"


def _instructional(raw: Optional[str]) -> bool:
    """Is this print the TEMPLATE talking rather than the document answering?

    'Indicate the number of years and months the project is expected to...',
    'If not the Accredited Entity, please indicate the full legal name of...'
    — 14 candidates over 11 documents where the builder captured the prompt
    instead of the answer. Two reasons not to serve one, and the second is the
    load-bearing one:

    * it is not a value. A note line the prompt calls authoritative would be
      stating the template's question as the document's answer.
    * `verify._field_lines` skips a WHOLE LINE that carries an instruction
      phrase — those phrases were the source of every false contradiction
      measured on the recorded answers — and a registry line is one line per
      document. Serving one would take that document's money segments off the
      checker with it: the field service would be buying one field by blinding
      the verifier on the rest of the line.

    ONE definition of the phrase, imported from the reader that acts on it, so
    the two cannot drift; and never a blocker, like every other cross-module
    read in this file.
    """
    try:
        from gcf_qna.rag import verify
        return bool(verify._INSTRUCTION_RE.search(raw or ""))
    except Exception:                                          # noqa: BLE001
        return False


def _served_value(label: str, field: str, c: dict) -> str:
    """One served value: '<label>: <as printed> (p.N, SECTION)'.

    Money formatting for a money print — `_money_bit`, which quotes a figure
    whose scale word its own mantissa contradicts and says so — and plain,
    clipped text for everything else. A print is money here when the field is
    one of the money fields or when schema 2 recorded a currency for it, which
    is what makes a number an amount; `mitigation_outcome`'s '37.6 million
    CO2e' is a figure and not an amount, and does not get an amount's wording.
    A print too long for the cap always takes the text form: `_money_bit` does
    not clip, and no money print in this corpus is that long.
    """
    raw = str(c.get("raw") or "")
    short = _clip(raw, _MAX_FIELD_CHARS)
    if (field in _MONEY_FIELDS or c.get("currency")) and short == raw:
        return _money_bit(label, c)
    return f"{label}: {short} {_where(c)}"


def _served_bits(doc_id: Optional[str], question: str) -> List[str]:
    """One segment per asked-for field this document states, or []."""
    fields = _asked_fields(question)
    if not fields:
        return []
    f2 = _v2_facts(doc_id)
    p = _planner()
    lists = getattr(p, "LIST_FIELDS", frozenset()) if p else frozenset()
    bits: List[str] = []
    for field in fields:
        usable = [c for c in (f2.get(field) or []) if _usable(c)]
        if not usable:
            continue                  # absent: the note says nothing about it
        if any(c.get("status") == "conflicting" for c in usable):
            continue                  # _conflict_lines already prints them all
        cands = [c for c in usable if not _instructional(c.get("raw"))]
        if not cands:
            continue                  # every print of it is the template's own
        label = _SERVED_LABELS.get(field, field.replace("_", " "))
        if field in lists:
            # ONE SEGMENT PER VALUE, and deliberately not `_list_bit`'s
            # count-first shape. `verify._field_lines` reads the value of a
            # field as the first amount AFTER its label, so
            # 'financial instruments (2): $5 million USD ...' publishes the
            # COUNT as the field's value — measured: a claim correctly stating
            # the loan's $46 million came back as a contradiction of '2'. The
            # count-first shape is safe for a list of names (`countries`) and
            # unsafe for a list of figures, so a served list repeats its label
            # instead, which also puts every value under a label the reader can
            # find rather than only the first. One candidate per instrument,
            # and the SECTION ('A.10 Loan') is what says which instrument a
            # value belongs to.
            seen, vals = set(), []
            for c in cands:
                key = (c.get("raw") or "").strip().casefold()
                if key and key not in seen:
                    seen.add(key)
                    vals.append(c)
            bits += [_served_value(label, field, c)
                     for c in vals[:_MAX_FIELD_VALUES]]
            if len(vals) > _MAX_FIELD_VALUES:
                bits.append(f"{label}: not every value is listed — "
                            f"{_MAX_FIELD_VALUES} of {len(vals)} shown above, "
                            "list truncated")
            continue
        # canonical first, then the earliest page that states it — the same
        # order `planner._registry_cell` fills a matrix cell in, so the two
        # arities cannot quote different prints of one fact
        canon = _canon2(f2, field)
        bits.append(_served_value(
            label, field,
            canon if canon in cands else min(cands, key=lambda c: c["page"])))
    return bits


# --- extraction honesty -----------------------------------------------------
# 16 documents are flagged `suspect` by the v2 builder's own consistency check
# and 19 were read by its LLM fallback rather than by the template rules, and
# until now the line printed their values exactly as confidently as any other.
# The fix is the one `_list_bit` already established for a cut list: the line
# says so itself. Adjudicating the 35 is Phase 3's work; saying which 35 they
# are is this line's.
#
# The marker must publish NO page and NO document. `chainlit_app._note_pages`
# and `verify.note_page_scopes` read a line's pointers with
# `[\[(](\d{1,3}_[\w.\-]+)` and `\(p\.(\d{1,3})[,)]`, so a parenthesis
# opening on a letter is inert to both, and an inert marker cannot become the
# invented citation this file spends `_meta_page` preventing.
_EXTRACTION_FLAGS = {
    "gcf>total": "the GCF figure on this line is larger than the total",
}


def _extraction_flags(doc_id: Optional[str]) -> List[str]:
    """The 'this extraction is flagged' segments for a document, or []."""
    try:
        cov = (_row_v2(doc_id) or {}).get("coverage")
    except Exception:                                          # noqa: BLE001
        return []
    if not isinstance(cov, dict):
        return []
    bits: List[str] = []
    if cov.get("llm_fallback"):
        bits.append("extraction flagged (llm_fallback): the values on this line "
                    "were read by model fallback, not by the template rules — "
                    "verify each against the page it cites")
    reason = cov.get("suspect")
    if reason:
        why = _EXTRACTION_FLAGS.get(str(reason))
        bits.append(f"extraction flagged ({reason}): "
                    + (f"{why} — " if why else "")
                    + "verify against the pages this line cites")
    return bits


def _conflict_field_order(f2: Dict[str, List[dict]]) -> List[str]:
    """Every field of a document that a conflict warning may cover, in order.

    Money first, in the order the warnings have always preferred them, then
    every other field the document holds in the planner's own column order,
    then anything neither list knows, alphabetically. Derived rather than
    listed so that a field the extractor grows tomorrow warns without an edit
    here — the defect being closed is precisely a field whose conflicts no
    mechanism looked at.
    """
    p = _planner()
    known = list(_MONEY_FIELDS) + [
        f for f in (getattr(p, "FIELD_ORDER", ()) if p else ())
        if f not in _MONEY_FIELDS]
    return ([f for f in known if f in f2]
            + sorted(f for f in f2 if f not in known))


def _conflict_lines(r: dict) -> List[str]:
    """One warning line per field the document contradicts itself on.

    Each line is its own line, names the document id, and leads with the
    CANONICAL figure's '(p.N, SECTION)': verify.build_evidence keys a note line
    on the first such pointer it finds, so the warning becomes page-level
    evidence at the page an answer actually cites, holding every figure of the
    field — which is exactly what an answer that 'reports both figures with
    their pages' has to verify against.

    EVERY field, not only the money ones. The money restriction was a scope
    decision taken when only money conflicts had been measured; the corpus
    holds four more, in three documents, and every one of them is now askable
    at arity one (`_served_bits`): FP139's implementation period (5 years vs
    25, both on p.5 under A.11), FP240's mitigation and adaptation outcomes,
    FP202's direct beneficiaries (81,551 under A.6 vs 1,251,769 under A.7).
    Serving those fields on the line while the only machinery that knows they
    are contradicted looked at money alone would have printed one of two
    disagreeing figures as though it were the fact.

    Money keeps money's formatting — `_fig`, the whole print, no clipping —
    and a text field takes `_text_fig`, which clips a paragraph-length print
    with the marker a cut string carries. The caps are unchanged and still
    announced: at most `_MAX_CONFLICT_LINES` lines, at most
    `_MAX_CONFLICT_ALTS` disagreeing prints each, and a final line naming
    every field held back.
    """
    f2 = _v2_facts(r.get("doc_id"))
    out: List[str] = []
    held: List[str] = []
    for field in _conflict_field_order(f2):
        alts = [c for c in f2.get(field, [])
                if c.get("status") == "conflicting" and _usable(c)]
        if not alts:
            continue
        if len(out) >= _MAX_CONFLICT_LINES:
            held.append(field)               # the cap, said out loud below
            continue
        money = field in _MONEY_FIELDS
        show = _fig if money else _text_fig
        canon = _canon2(f2, field)
        printed = [show(c) for c in ([canon] if canon else [])
                   + alts[:_MAX_CONFLICT_ALTS]]
        cut = len(alts) - _MAX_CONFLICT_ALTS
        more = (f" (+{cut} more disagreeing print{'s' if cut > 1 else ''} of "
                f"this field in the document, not listed — list truncated)"
                if cut > 0 else "")
        # 'figures' is what a money conflict prints; a contradicted text field
        # prints values, and calling '5 years, 0 months' a figure would be the
        # note describing its own evidence wrongly
        ask = (f"report both {'figures' if money else 'values'} with their "
               "pages." if len(printed) == 2
               else "report all of them with their pages.")
        out.append(f"Registry — CONFLICT in this document ({r['doc_id']}): {field} "
                   f"is printed as {'; also as '.join(printed)}{more} — {ask}")
    if held:
        kind = "figures" if all(f in _MONEY_FIELDS for f in held) else "prints"
        out.append(f"Registry — CONFLICT in this document ({r['doc_id']}): "
                   f"{len(held)} further field"
                   f"{'s' if len(held) > 1 else ''} ({', '.join(held)}) also "
                   f"print{'' if len(held) > 1 else 's'} disagreeing {kind}, "
                   f"not listed above — list truncated.")
    return out


def _fmt(r: dict, question: Optional[str] = None) -> str:
    """The one registry line for a document — every emitter goes through here.

    ``registry_note`` prints it for an FP id and, prefixed '<CODE> resolves
    to:', for a board code; ``chainlit_app._extend_registry_note`` prints it
    for a document only this TURN resolved to. One formatter, so the three can
    never drift, and so the cover-page pages below reach all three at once.

    The v1 text fields (title, entity, countries) now carry the page schema 2
    read them on, when it recorded one: same '(p.N)' pointer the money fields
    have always carried, minus the template section a cover-page fact has no
    equivalent of. Value and page come from DIFFERENT files by design —
    registry.json is the clean source for the text, registry_v2.json for the
    provenance — so a field is printed page-less unless v2 states a page for
    it, which is also what happens for every document built before the
    provenance pass.

    `question` is the arity-one field service (`_served_bits`): pass the turn's
    question and the fields it asks for are appended, from v2's own candidates
    and with v2's own pointers. It is optional and defaults to OFF, so the
    caller that has no question — `chainlit_app._extend_registry_note`, which
    fires on a document this TURN resolved to rather than one the question
    named — keeps the line it has always printed. The extraction flags do NOT
    depend on it: a flagged extraction is flagged on every line it appears on,
    asked about or not.

    Everything new is APPENDED. The existing bits keep their order and their
    bytes, so today's line is a prefix of tomorrow's and a diff of the two is
    exactly what was added.
    """
    f2 = _v2_facts(r.get("doc_id"))
    mp = _v2_meta(r.get("doc_id"))
    gcf, total = _canon2(f2, "gcf_funding_requested"), _canon2(f2, "total_financing")
    bits = []
    if r.get("title"):
        bits.append(f'"{r["title"]}"' + _meta_page(mp, "title"))
    if r.get("accredited_entity"):
        bits.append(f"accredited entity: {r['accredited_entity']}"
                    + _meta_page(mp, "accredited_entity"))
    if r.get("countries"):
        bits.append(_list_bit("countries", r["countries"])
                    + _meta_page(mp, "countries"))
    if gcf:
        bits.append(_money_bit("GCF funding requested", gcf))
    elif r.get("gcf_financing"):
        bits.append(f"GCF financing (as printed): {r['gcf_financing']}")
    if total:
        bits.append(_money_bit("total financing", total))
    elif r.get("total_financing"):
        bits.append(f"total financing (as printed): {r['total_financing']}")
    if r.get("board"):
        bits.append(f"board B.{r['board']}, {r.get('year')}")
    if question:
        bits += _served_bits(r.get("doc_id"), question)
    bits += _extraction_flags(r.get("doc_id"))
    fp = f"FP{r['fp']}: " if r.get("fp") else ""
    return f"{fp}{'; '.join(bits)} [{r['doc_id']}, cover pages]"


# groups: board, agenda item (optional), addendum
_BOARD_CODE_RE = re.compile(
    r"b\.?\s?(\d{2})\s*[/.\-]\s*(?:(\d{2})\s*[/.\-]\s*)?add\.?\s?(\d{2})", re.I)


def resolve_board_code(board: int, add: int, item: Optional[int] = None) -> Optional[dict]:
    """Deterministic board-code -> document resolution (GCF/B.42/02/Add.16).

    The agenda item (the middle '/02/') is part of the identifier: one board
    carries several series (b30-02-* and b30-03-* both exist), so a stated item
    MUST match or the code resolves to nothing. Codes without an item part fall
    back to add-only matching.
    """
    from gcf_qna.boards import board_of
    tag = f"add{add:02d}" if item is None else f"{item:02d}add{add:02d}"
    for k, v in load().items():
        if board_of(k) == board and tag in re.sub(r"[^a-z0-9]", "", k.lower()):
            return {"doc_id": k, **v}
    return None


def _board_code_text(b_: str, item_: str, add_: str) -> str:
    """The code as the user wrote it (zero-padding kept), canonically prefixed."""
    return f"GCF/B.{b_}/" + (f"{item_}/" if item_ else "") + f"Add.{add_}"


# FP ids as users write them: FP274, FP 274, FP-274, FP0086. Leading zeros are
# stripped INSIDE the group, otherwise 'FP0086' captures '008' and confidently
# resolves to FP8 — a different proposal. The trailing (?!\d) stops 'fp2023' (a
# year) from being read as FP202; \b would also reject the real doc stems that
# end '...-fp272_0', since '2' and '_' are both word characters.
#
# H10 of docs/l1-l2-coverage-review.md measured what the strict shape misses:
# 'FP#220', 'FP.220', 'FP no. 220', 'proposal 220' and 'funding proposal number
# 220' all resolved to NOTHING, and a miss here is silent — no registry note,
# no identifier routing, no guard, just open retrieval with no signal that an
# identifier was lost. Two arms are added, and the boundaries they keep are the
# point of the pattern, not decoration:
#
#   * the FP arm accepts the punctuation and the number word ('#', '.', 'no.',
#     'n°', 'number') between 'FP' and the digits;
#   * the PROSE arm accepts a SINGULAR 'proposal'/'proposition' (with an
#     optional 'funding') and needs either a number word or TWO digits.
#
# Both refusals are deliberate. **Singular only**: 'proposals 220 and 203'
# would bind 220 and lose 203, and a lone id is worse than none — chainlit's
# _prescope_single_fp hard-scopes retrieval to it and starves the partner
# document ('FPs 12 and 74' binds nothing today for the same reason).
# **Two digits without a number word**: 'proposal 2' is an enumeration far more
# often than an id, and FP1-FP9 are still reachable as 'FP2' or 'proposal no.
# 2'. The (?!\d) tail keeps 'proposal 2020' (a year) out, and no arm has a head
# for a bare '220' or for 'Add.220', so '220 countries' binds nothing.
#
# ONE capturing group, in a shared prefix rather than per-arm: every consumer
# reads `findall` as a list of numbers (registry_note, resolve_fps,
# chainlit_app's four call sites, planner.detect, eval_answers' _FP_TOKEN_RE),
# and a second group would hand all of them tuples.
_FP_RE = re.compile(
    r"(?:"
    r"fp[\s\-.#]{0,2}(?:n[o\u00b0\u00ba]s?\.?|num(?:ber|ero|\u00e9ro)?\.?)?[\s.#]{0,2}"
    r"|(?:funding[\s\-]?)?(?:proposal|proposition)[\s\-]?"
    r"(?:(?:n[o\u00b0\u00ba]s?\.?|num(?:ber|ero|\u00e9ro)?\.?)[\s.#]{0,2}|#\s?|(?=\d\d))"
    r")0*(\d{1,3})(?!\d)", re.I)
# public alias: the app imports this so the FP pattern has ONE definition
FP_RE = _FP_RE
BOARD_CODE_RE = _BOARD_CODE_RE

#: either identifier pattern, anchored at the END of a span — the test
#: `_drop_document_apposition` runs to decide whether a ', the ...' phrase
#: follows an identifier (and so names the document) or opens the ask itself.
#: Built from the two patterns above rather than beside its caller, because
#: they are defined below it: one definition of what an identifier is, here as
#: everywhere else in this file.
_ENDS_WITH_ID_RE = re.compile(
    r"(?:" + _FP_RE.pattern + r"|" + _BOARD_CODE_RE.pattern + r")\s*$", re.I)


def resolve_fps(question: str):
    """(resolved rows, missing fp numbers) for every FP id in the question."""
    resolved, missing = [], []
    for n in dict.fromkeys(_FP_RE.findall(question.lower())):
        row = by_fp(int(n))
        (resolved if row else missing).append(row or int(n))
    return resolved, missing


# ---------------------------------------------------------------------------
# inverse lookups: by_country / by_entity, and the normalisation they need
# ---------------------------------------------------------------------------
# H1/H2 of docs/l1-l2-coverage-review.md §4.2: the registry answers 'which
# countries does FP151 cover' and answers nothing at all in the other
# direction. The live probes measured what that costs — P2 named 6 of Kenya's
# 25 and never said the list was partial; P10, in French, named 6 of 13 with
# one category error (FP183 credited to the World Bank; it is IFAD's).
#
# Both directions are a lookup over rows the registry already holds. The
# COUNTRY field is nearly clean (178 distinct values); the ENTITY field is not
# (126 distinct strings for far fewer organisations — 'United Nations
# Development Programme', 'United Nations Development Programme (UNDP)',
# 'United Nations Development Program, UNDP' and 'UNDP' are one entity written
# four ways). The normalisation below runs at LOOKUP time and never touches
# data/: the stored strings stay exactly as the documents print them, and a
# note prints the spelling the corpus uses most.

#: Trailing corporate forms that do not distinguish two entities.
_CORP_SUFFIX = frozenset(
    "gmbh ltd limited inc incorporated llc llp lp plc pte ag".split())
#: Leading articles (EN/FR/ES) dropped from a lookup key.
_ARTICLES = frozenset("the la le les el los".split())
#: Political qualifiers a country's long form carries and its short form does
#: not ('Republic of Serbia' -> 'serbia'). Stripped from the FRONT only, so
#: 'South Sudan' and 'North Macedonia' keep their heads.
_COUNTRY_QUALIFIERS = frozenset(
    "republic of the kingdom state states federated plurinational islamic "
    "hashemite democratic people s united federation commonwealth".split())
#: Values of the countries field that name no country: they are left out of
#: the index and out of the question vocabulary rather than half-matched.
_NOT_A_COUNTRY_RE = re.compile(r"\b(?:region|regional|programme|program|global)\b",
                               re.I)
#: One country the corpus spells several ways. Each tuple becomes ONE group,
#: reachable by every key in it. Derived from the 178 stored values, not from
#: a world list: only the collisions this corpus actually contains.
_COUNTRY_SYNONYMS = (
    ("cote d ivoire", "ivory coast"),
    ("lao pdr", "laos", "laos pdr", "lao people s democratic republic"),
    ("viet nam", "vietnam"),
    ("kyrgyz republic", "kyrgyzstan"),
    ("dr congo", "drc", "democratic republic of congo",
     "democratic republic of the congo"),
    ("timor leste", "democratic republic of timor leste"),
)
#: French names for entities the corpus records only in English. The data
#: merges 'Agence Française de Développement' with 'French Development Agency'
#: and 'Banque Ouest Africaine de Développement' with 'West African
#: Development Bank' by itself — both pairs publish the same acronym — so
#: these are the organisations whose French name appears in no row at all,
#: which is why probe P10 asked for 'Banque mondiale' and got a category
#: error. Multi-word and unambiguous on purpose: an acronym like 'BAD'
#: (Banque africaine de développement) is an English word and is NOT indexed.
_FR_ENTITY_ALIASES = {
    "banque mondiale": "world bank",
    "pnud": "united nations development programme",
    "pnue": "united nations environment programme",
    "programme des nations unies pour le developpement":
        "united nations development programme",
    "programme des nations unies pour l environnement":
        "united nations environment programme",
    "banque africaine de developpement": "african development bank",
    "banque asiatique de developpement": "asian development bank",
    "banque interamericaine de developpement": "inter american development bank",
    "fonds international de developpement agricole":
        "international fund for agricultural development",
    "organisation des nations unies pour l alimentation et l agriculture":
        "food and agriculture organization of the united nations",
}

_PAREN_RE = re.compile(r"\(([^()]*)\)")


def _fold(s) -> str:
    """Casefold and strip accents: "Côte d'Ivoire" -> "cote d'ivoire".

    The 'oe' ligature is spelled out by hand: NFKD has no decomposition for
    U+0153, so 'mise en œuvre' would otherwise never match a pattern written
    'oeuvre' — which is the commonest French phrasing of 'implemented by'.
    """
    folded = "".join(c for c in unicodedata.normalize("NFKD", str(s or ""))
                     if not unicodedata.combining(c)).casefold()
    return folded.replace("\u0153", "oe").replace("\u00e6", "ae")


def _norm_key(s) -> str:
    """The lookup key for a stored value, or for a phrase out of a question.

    Fold case and accents, drop parentheticals (an acronym gloss is indexed
    separately, never as part of the name), '&' -> 'and', punctuation ->
    space, then drop a leading article and any TRAILING corporate form. Two
    strings with the same key are the same name as far as a lookup is
    concerned — which is what merges 'The World Bank' with 'World Bank',
    'Acumen Fund, Inc.' with 'Acumen Fund', and 'UGANDA' with 'Uganda'.
    """
    t = _PAREN_RE.sub(" ", _fold(s)).replace("&", " and ")
    toks = [w for w in re.split(r"[^a-z0-9]+", t) if w]
    while toks and toks[-1] in _CORP_SUFFIX:
        toks.pop()
    if len(toks) > 1 and toks[0] in _ARTICLES:
        toks.pop(0)
    return " ".join(toks)


def _short_country_key(key: str) -> str:
    """'republic of serbia' -> 'serbia'; '' when nothing was stripped.

    Front-anchored, so a name whose first word is a real part of it ('South
    Sudan') keeps it. A short name two DIFFERENT long forms reduce to is
    dropped by the index rather than resolved (see ``_build_country_index``):
    'Congo' is both 'Republic of Congo' and 'Democratic Republic of the
    Congo', and an authoritative note must not pick one silently.
    """
    toks = key.split()
    i = 0
    while i < len(toks) and toks[i] in _COUNTRY_QUALIFIERS:
        i += 1
    return " ".join(toks[i:]) if i and toks[i:] else ""


def _is_acronym(tok: str) -> bool:
    """'UNDP', 'KfW', 'CABEI' — yes. 'The', 'Ltd', 'AG', 'Pegasus' — no.

    Two capitals minimum (a capitalised word is not an acronym), three
    characters minimum (so 'CI' and 'AG' can never claim a question), never a
    corporate form.
    """
    return (3 <= len(tok) <= 10 and tok.isalpha() and tok[0].isupper()
            and sum(c.isupper() for c in tok) >= 2
            and tok.lower() not in _CORP_SUFFIX)


def _acronyms(raw: str) -> List[str]:
    """The acronyms a stored name publishes for itself, upper-cased.

    Three shapes, all of them in the corpus: a parenthetical gloss ('... (GIZ)
    GmbH'), a leading acronym before a dash ('IUCN - International Union
    ...'), and a trailing one ('International Fund for Agricultural
    Development - IFAD', 'United Nations Development Program, UNDP'). A name
    that is NOTHING but an acronym ('FAO') publishes itself, which is how the
    six rows spelled 'FAO' reach the seven that spell it out.
    """
    out = []
    for inner in _PAREN_RE.findall(raw or ""):
        tok = inner.strip().strip(".,")
        if _is_acronym(tok):
            out.append(tok)
    toks = [t.strip(".,") for t in
            re.split(r"[\s,\-–—]+", _PAREN_RE.sub(" ", raw or "")) if t.strip(".,")]
    if toks and _is_acronym(toks[0]):
        out.append(toks[0])
    if len(toks) > 1 and _is_acronym(toks[-1]):
        out.append(toks[-1])
    return list(dict.fromkeys(t.upper() for t in out))


class _Groups:
    """Union-find over the raw strings of one field.

    Entities merge on TWO relations — same normalised name, and same published
    acronym — and the two together pull 'UNDP', 'United Nations Development
    Programme' and 'United Nations Development Program, UNDP' into one group
    with no hand-written mapping.
    """

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass(frozen=True)
class _Index:
    """One field's inverse index."""
    keys: Dict[str, str]                 # lookup key -> group id
    rows: Dict[str, List[str]]           # group id -> doc ids
    display: Dict[str, str]              # group id -> the spelling to print
    variants: Dict[str, List[str]]       # group id -> the spellings merged


def _finish_index(keys, by_value, raw_of, groups, prefer_long: bool) -> _Index:
    rows: Dict[str, List[str]] = {}
    variants: Dict[str, List[str]] = {}
    counts: Dict[str, Dict[str, int]] = {}
    for key, doc_ids in by_value.items():
        root = groups.find(key)
        rows.setdefault(root, []).extend(doc_ids)
        for raw in raw_of[key]:
            counts.setdefault(root, {})
            counts[root][raw] = counts[root].get(raw, 0) + 1
    for root, seen in counts.items():
        variants[root] = sorted(seen)
    # The spelling the corpus uses most — 'United Nations Development
    # Programme', not the four rows that say 'UNDP'. A tie goes to the LONGER
    # string for an entity (six IUCN spellings, one document each: the note
    # should print the name, not the acronym) and to the SHORTER one for a
    # country ('Marshall Islands', not 'Republic of the Marshall Islands').
    display = {}
    for root, seen in counts.items():
        display[root] = max(seen, key=lambda r: (seen[r], len(r) if prefer_long
                                                 else -len(r)))
    rows = {k: sorted(dict.fromkeys(v)) for k, v in rows.items()}
    return _Index(keys={k: groups.find(v) for k, v in keys.items()},
                  rows=rows, display=display, variants=variants)


def _collect(docs: Dict[str, dict], field: str, is_list: bool, skip=None):
    """({lookup key: [doc id, ...]}, {lookup key: [raw value, ...]}).

    Reads one field of every row and tolerates everything the registry can
    hold there: an error row with no such field, a null, a number where a
    string belongs. ``skip`` drops values that are not a name at all.
    """
    by_value: Dict[str, List[str]] = {}
    raw_of: Dict[str, List[str]] = {}
    for doc_id, row in docs.items():
        if not isinstance(row, dict):
            continue
        got = row.get(field)
        values = (got or []) if is_list else ([got] if got else [])
        for raw in values if isinstance(values, list) else []:
            if not isinstance(raw, str) or not raw.strip():
                continue
            if skip is not None and skip.search(raw):
                continue
            key = _norm_key(raw)
            if not key:
                continue
            by_value.setdefault(key, []).append(doc_id)
            raw_of.setdefault(key, []).append(raw)
    return by_value, raw_of


def _build_country_index(docs: Dict[str, dict]) -> _Index:
    by_value, raw_of = _collect(docs, "countries", True,
                                skip=_NOT_A_COUNTRY_RE)
    g = _Groups()
    for key in by_value:
        g.find(key)
    for group in _COUNTRY_SYNONYMS:
        present = [k for k in group if k in by_value]
        for other in present[1:]:
            g.union(present[0], other)
    # 'Viet Nam' and 'Vietnam' are one country and two keys; the squeezed form
    # is the only difference between them.
    owner: Dict[str, str] = {}
    for key in by_value:
        squeezed = key.replace(" ", "")
        if squeezed == key:
            continue
        if squeezed in by_value:
            g.union(squeezed, key)
        elif squeezed in owner:
            g.union(owner[squeezed], key)
        else:
            owner[squeezed] = key
    keys = {k: k for k in by_value}
    keys.update({sq: k for sq, k in owner.items()})
    # short names, and the ambiguity rule: a short name TWO unmerged long
    # forms answer to is not indexed at all.
    short_owner: Dict[str, str] = {}
    ambiguous = set()
    for key in by_value:
        short = _short_country_key(key)
        if not short:
            continue
        if short in by_value:
            g.union(short, key)
        elif short in short_owner:
            if g.find(short_owner[short]) != g.find(key):
                ambiguous.add(short)
        else:
            short_owner[short] = key
    for short, key in short_owner.items():
        if short not in ambiguous and short not in keys:
            keys[short] = key
    return _finish_index(keys, by_value, raw_of, g, prefer_long=False)


#: Organisational tails that do not make a different organisation OF A NAME
#: THE CORPUS ALREADY RECORDS: 'World Bank Group' is merged into 'World Bank'
#: only because the shorter name is itself a stored value. A tail whose
#: remainder is not a stored value is left alone, so 'World Wildlife Fund'
#: never becomes 'World Wildlife'.
_ORG_TAIL = frozenset("group fund foundation".split())

# ---------------------------------------------------------------------------
# owner rulings on three clusters (decided 2026-08-26, recorded here)
# ---------------------------------------------------------------------------
# The tail rule above is a GUESS about the world made from two strings, and
# entity_clusters() exists so a human can read those guesses back. Three were
# put to the corpus owner, who ruled:
#
#   (a) 'World Bank Group' = 'The World Bank' — MERGED, ruling upheld. The
#       corpus records 'The World Bank' (8), 'World Bank' (3) and 'World Bank
#       Group' (1) plus one parenthetical gloss; one organisation, 13
#       documents, and probe P10 asked for 'Banque mondiale' and was answered
#       out of 13. Pinned by test_the_world_bank_group_stays_merged.
#
#   (b) 'Ministry of Environment' / 'Ministry of Environment (MOE)' — MERGED,
#       ruling upheld. Three documents, two spellings of one national body;
#       the merge is _norm_key's parenthetical rule, not the tail rule.
#       (The ruling covers THESE spellings only: 'Ministry of Environment and
#       Sustainable Development, Cali, Colombia' is a different string and a
#       different key, and stays where it is.) Pinned by
#       test_ministry_of_environment_stays_merged.
#
#   (c) 'Acumen' vs 'Acumen Fund' — SPLIT, merge DECLINED. The tail rule had
#       pulled the single 'Acumen' row into the six-document 'Acumen Fund'
#       family; the owner ruled they are distinct canonical entities, so the
#       alias index must keep them apart. Pinned by
#       test_acumen_and_acumen_fund_stay_split.
#
# A declined merge is data the clustering cannot derive, so it is written down
# rather than inferred — the same shape as the non-merge the corpus produces on
# its own for 'Inter-American Investment Corporation (IDB Invest)', which stays
# distinct from 'Inter-American Development Bank (IDB)' because 'IDB Invest' is
# two tokens and _is_acronym refuses it. That one is a happy accident of the
# rules; this one is a decision, and it is enforced explicitly.
#:
#: (head key, full key) pairs ``_ORG_TAIL`` must NOT merge, by owner ruling.
_OWNER_KEEP_DISTINCT = frozenset({("acumen", "acumen fund")})


def _build_entity_index(docs: Dict[str, dict]) -> _Index:
    by_value, raw_of = _collect(docs, "accredited_entity", False)
    acr_of = {key: list(dict.fromkeys(a for raw in raws for a in _acronyms(raw)))
              for key, raws in raw_of.items()}
    g = _Groups()
    first_of: Dict[str, str] = {}
    for key in by_value:
        g.find(key)
        for acr in acr_of.get(key) or []:
            if acr in first_of:
                g.union(first_of[acr], key)
            else:
                first_of[acr] = key
    for key, raws in raw_of.items():
        # a parenthetical that is not an acronym but IS a name the corpus
        # records elsewhere: 'International Reconstruction and Development and
        # International Development Association (World Bank)'.
        for raw in raws:
            for inner in _PAREN_RE.findall(raw):
                gloss = _norm_key(inner)
                if gloss and gloss != key and gloss in by_value:
                    g.union(gloss, key)
        head = " ".join(key.split()[:-1])
        if (key.split()[-1:] and key.split()[-1] in _ORG_TAIL and head in by_value
                and (head, key) not in _OWNER_KEEP_DISTINCT):
            g.union(head, key)
    keys = {k: k for k in by_value}
    for acr, key in first_of.items():
        keys.setdefault(acr.casefold(), key)
    for alias, target in _FR_ENTITY_ALIASES.items():
        owner = keys.get(_norm_key(target))
        if owner:
            keys.setdefault(alias, owner)
    return _finish_index(keys, by_value, raw_of, g, prefer_long=True)


_index_cache: Optional[Tuple[Dict[str, dict], _Index, _Index]] = None


def _indexes() -> Tuple[_Index, _Index]:
    """(country index, entity index) for the CURRENT registry rows.

    Keyed on the rows object itself rather than on a built flag: the suites
    monkeypatch ``registry._cache`` with four-document fixtures, and an index
    built from the real 273 rows must not survive into one of them.
    """
    global _index_cache
    docs = load()
    if _index_cache is None or _index_cache[0] is not docs:
        _index_cache = (docs, _build_country_index(docs), _build_entity_index(docs))
    return _index_cache[1], _index_cache[2]


def _rows_of(index: _Index, name: str) -> List[dict]:
    docs = load()
    root = index.keys.get(_norm_key(name))
    if root is None:
        return []
    return sorted(({"doc_id": d, **docs[d]} for d in index.rows.get(root, [])
                   if isinstance(docs.get(d), dict)),
                  key=lambda r: (r.get("fp") or 0, r["doc_id"]))


def by_country(name: str) -> List[dict]:
    """Every registry row whose countries field names ``name``.

    Case- and accent-insensitive over whole VALUES, never over substrings:
    'Mali' resolves to the Mali rows and not to Malawi, 'Niger' not to
    Nigeria, 'Guinea' neither to 'Guinea-Bissau' nor to 'Papua New Guinea'. A
    long official form is reachable by its short name ('Serbia' ->
    'Republic of Serbia') unless that short name is ambiguous in this corpus
    ('Congo'), and the spellings one country is written several ways in are
    merged (``_COUNTRY_SYNONYMS``, plus 'UGANDA'/'Uganda' by folding).
    """
    return _rows_of(_indexes()[0], name)


def by_entity(name_or_acronym: str) -> List[dict]:
    """Every registry row whose accredited entity is ``name_or_acronym``.

    The argument may be any spelling the corpus prints, or a published
    acronym: by_entity('UNDP'), by_entity('undp') and by_entity('United
    Nations Development Programme') return the same rows.
    """
    return _rows_of(_indexes()[1], name_or_acronym)


def entity_clusters() -> Dict[str, List[str]]:
    """{printed name: [the raw spellings merged into it]}.

    The normalisation, exposed so a review can read what it merged instead of
    trusting it — 126 stored strings, far fewer organisations.
    """
    idx = _indexes()[1]
    return {idx.display[root]: idx.variants.get(root, [])
            for root in sorted(idx.rows, key=lambda r: (-len(idx.rows[r]),
                                                        idx.display[r]))}


def country_clusters() -> Dict[str, List[str]]:
    """{printed name: [the raw spellings merged into it]} for the countries."""
    idx = _indexes()[0]
    return {idx.display[root]: idx.variants.get(root, [])
            for root in sorted(idx.rows, key=lambda r: (-len(idx.rows[r]),
                                                        idx.display[r]))}


# --- the question side: the ask shape, and the names a question spells ------

#: The plural noun an inverse ask names, or a singular one a set quantifier
#: makes plural in meaning ('list EVERY proposal from IFAD' is H1's own
#: wording). 'Which proposal restores mangrove ecosystems in Ecuador?' — the
#: shape of six gold discovery cases — asks retrieval to identify ONE
#: document and is none of this note's business, so the bare singular arm is
#: deliberately absent. The FP arm excludes an FP IDENTIFIER ('FPs 12 and
#: 74'): 'FPs' is the plural ask, never a document id.
_SET_NOUN = (r"(?:fps(?![\s\-]{0,2}\d)|proposals|projects|programmes|documents|"
             r"propositions|projets|"
             r"(?:every|all|each|toutes?\s+les|tous\s+les)\s+(?:\w+\s+){0,2}?"
             r"(?:proposal|project|programme|document|proposition|projet)s?)")
#: The ask shape an inverse note answers: a request for a SET of proposals.
#: An interrogative or an imperative, then a set noun — and, at the call site,
#: a name the registry indexes. The three together are the guard against
#: 'What does FP123 do in Kenya?', a document-scoped question that merely
#: mentions a country. Matched against the FOLDED question, so 'énumère' and
#: 'enumere' are one pattern.
_SET_ASK_RE = re.compile(
    r"\b(?:which|what|list|name|show|give\s+me|how\s+many)\b[^?!.]{0,60}?"
    r"\b" + _SET_NOUN + r"\b"
    r"|\b(?:quel|quels|quelle|quelles|combien|liste[rz]?|listes|enumere[rz]?|"
    r"enumerer)\b[^?!.]{0,60}?"
    r"\b" + _SET_NOUN + r"\b")
#: 'UNDP proposals', "IFAD's projects" — the attributive form, which carries
#: no relation word at all. Probe F12 ('how many UNDP proposals?') is exactly
#: this shape, so it has to count as a relation or the note never reaches the
#: question the review measured.
_SET_NOUN_TOKENS = frozenset(
    "fps proposals projects programmes documents propositions projets".split())
#: A relation between the asked-for set and the entity. Bare 'of'/'de' count:
#: what this excludes is a name mentioned for some OTHER reason ('which
#: proposals cite the World Bank's methodology?'), not a particular phrasing.
_ENTITY_RELATION_RE = re.compile(
    r"\b(?:by|from|of|with|implement\w*|financ\w*|fund\w*|submitted|approved|"
    r"accredited|entity|entities|par|de|du|des|dont|soumises?|presentees?|"
    r"accreditees?|accredite|entite|entites|mises?\s+en\s+oeuvre)\b")


def _detect(index: _Index, question: str) -> Optional[str]:
    """The longest indexed name the question spells, or None.

    Word boundaries by construction: both sides are tokenised and whole
    n-gram runs are matched, so 'Mali' can never be found inside 'Malawi' nor
    'Niger' inside 'Nigeria' — the failure a substring scan would have.
    Longest first, so 'Papua New Guinea' wins over 'Guinea' and 'South
    Africa' over nothing at all.
    """
    toks = [t for t in re.split(r"[^a-z0-9]+", _fold(question)) if t]
    for size in range(min(_MAX_NAME_TOKENS, len(toks)), 0, -1):
        for i in range(len(toks) - size + 1):
            phrase = " ".join(toks[i:i + size])
            if phrase in index.keys:
                return phrase
    return None


def _inverse_note(rows: List[dict], lead: str, scope: str = "",
                  tail: str = "") -> Optional[str]:
    """One authoritative listing line for an inverse lookup, or None.

    The year listing's shape, above: the count first, FP ids and shortened
    titles after, no document stems and no page pointers — so
    ``_note_pages``/``note_page_scopes`` publish nothing uncreditable from it.

    ``tail`` is a sentence the listing itself needs and the count cannot say
    (the board listing uses it to state that the registry records no approval
    DECISION); it carries no pointer, for the reason above.

    Whether the listing is COMPLETE is stated either way. That is the half
    probe P2 was missing — it named 6 of Kenya's 25 and never said the list
    was partial — and ``scope`` is the other half: an entity listing is
    complete over the SPELLINGS the normalisation merged, not over every
    string in the corpus that might mean the same organisation, and an
    authoritative note that overstated that would be the same defect wearing
    the opposite sign.
    """
    rows = [r for r in rows if r.get("fp")]
    if not rows:
        return None
    listing, more = _listing(rows, _MAX_INVERSE_ROWS)
    n = len(rows)
    state = (f"{_MAX_INVERSE_ROWS} of {n} listed — LIST TRUNCATED, but the "
             f"count {n} is complete" if more else
             f"complete listing over the {len(load())} corpus documents{scope}")
    return f"Registry — {n} {lead} ({state}): {listing}{more}{tail}"


def _country_note(question: str) -> Optional[str]:
    """The country-inverse note (H2), or None when the question is not that ask."""
    q = _fold(question)
    if not _SET_ASK_RE.search(q):
        return None
    idx = _indexes()[0]
    key = _detect(idx, q)
    if not key:
        return None
    name = idx.display[idx.keys[key]]
    return _inverse_note(by_country(key),
                         f"funding proposals in the corpus name {name} in "
                         f"their countries field")


def _attributive(question: str, key: str) -> bool:
    """True for 'UNDP proposals' / "UNDP's projects" — the name used as the
    modifier of the asked-for set, which needs no relation word."""
    toks = [t for t in re.split(r"[^a-z0-9]+", question) if t]
    name = key.split()
    for i in range(len(toks) - len(name) + 1):
        if toks[i:i + len(name)] != name:
            continue
        tail = toks[i + len(name):]
        if tail[:1] == ["s"]:                      # "UNDP's proposals"
            tail = tail[1:]
        if tail[:1] and tail[0] in _SET_NOUN_TOKENS:
            return True
    return False


def _entity_note(question: str) -> Optional[str]:
    """The entity-inverse note (H1), or None when the question is not that ask."""
    q = _fold(question)
    if not _SET_ASK_RE.search(q):
        return None
    idx = _indexes()[1]
    key = _detect(idx, q)
    if not key:
        return None
    if not (_ENTITY_RELATION_RE.search(q) or _attributive(q, key)):
        return None
    root = idx.keys[key]
    spelled = len(idx.variants.get(root, []))
    scope = (f", covering the {spelled} spellings of that name the registry "
             f"holds" if spelled > 1 else "")
    return _inverse_note(by_entity(key),
                         f"funding proposals in the corpus record "
                         f"{idx.display[root]} as the accredited entity", scope)



# ---------------------------------------------------------------------------
# the board inverse (H16): the asymmetry, not the gap
# ---------------------------------------------------------------------------
# 'Which funding proposals were approved at B.44?' is answered definitively
# today — 44 is outside the corpus and chainlit_app._board_range_note says so —
# while the same question about B.35, seven documents this registry holds,
# received nothing at all. §4.3 counts that pair among the THREE false-
# authority misfires in the review, because the system is most confident
# exactly where it knows least.
#
# Out of range stays where it is. _board_range_note owns B.1-B.10 and B.44+,
# including the template-heading ambiguity it was rewritten around (H5/P6), and
# nothing here duplicates it: this note fires only for a board IN
# BOARD_YEARS, which is the one case that regex deliberately skips.

#: A board written in prose: 'at B.35', 'board B.35', 'B35'. Two digits, so a
#: template heading ('section B.3', 'B.2(a)') can never reach this pattern at
#: all — the corpus boards are B.11-B.43 and the proposal template numbers its
#: own headings B.1-B.10. Named apart from ``chainlit_app._BOARD_TOKEN_RE`` on
#: purpose: that one is deliberately WIDER (\d{1,2} plus a paragraph-letter
#: group) because it has to judge the ambiguous low numbers this one never
#: sees, and two patterns with one name would read as one pattern.
_PROSE_BOARD_RE = re.compile(r"\bb\.?\s?(\d{2})\b", re.I)

#: 'What was approved at B.35?' — the bare form, with no set noun in it.
#:
#: JUDGEMENT CALL, and the answer is YES, this is a set ask. A board approves a
#: BATCH of funding proposals, so 'what was approved' names a set the way
#: 'list EVERY proposal' does: _SET_NOUN already accepts a singular noun that a
#: quantifier makes plural, and here the verb carries the quantifier instead of
#: a word. The decisive argument is the asymmetry itself — the out-of-range arm
#: answers THIS EXACT PHRASING definitively ('What was approved at B.44?' ->
#: 'B.44 is not in this corpus ... State this definitively'), so refusing it in
#: range would leave H16's misfire standing in the very wording that produced
#: it. The risk is bounded three ways: the question must also name an in-range
#: board, the note is a COMPLETE counted listing rather than a sample, and the
#: note says outright that the registry records no approval decision. Both
#: readings are pinned in the suite (test_a_bare_what_was_approved_is_a_set_ask
#: and its NEGATIVE twin, which fixes what must NOT fire alongside it).
_APPROVAL_ASK_RE = re.compile(
    r"\b(?:what|which)\b[^?!.]{0,40}?\b(?:was|were)\s+approved\b"
    r"|\bwhat\s+did\s+the\s+board\s+approve\b"
    r"|\bqu['\s]?est[-\s]ce\s+qui\s+a\s+ete\s+approuve"
    r"|\bqu['\s]?a[-\s]t[-\s]il\s+approuve")


def _board_years() -> Dict[int, int]:
    """{board: year}, imported at CALL time like ``resolve_board_code``'s
    ``board_of`` — the module graph stays registry -> config only."""
    from gcf_qna.boards import BOARD_YEARS
    return BOARD_YEARS


def _prose_boards(question: str) -> List[int]:
    """The board numbers a question names OUTSIDE a full board code.

    'GCF/B.35/02/Add.05' names ONE document, which ``registry_note`` already
    resolves and prints; answering it with the whole of B.35 would be a
    different question. So a token inside a code span is dropped, and
    'bc-b35-02-add05' — a gold case — keeps the single-document note it has
    always had.
    """
    spans = [m.span() for m in _BOARD_CODE_RE.finditer(question)]
    out: List[int] = []
    for m in _PROSE_BOARD_RE.finditer(question):
        if not any(lo <= m.start() < hi for lo, hi in spans):
            out.append(int(m.group(1)))
    return list(dict.fromkeys(out))


def _board_note(question: str) -> List[str]:
    """The in-range board listings (H16), one line per board, or []."""
    q = _fold(question)
    if not (_SET_ASK_RE.search(q) or _APPROVAL_ASK_RE.search(q)):
        return []
    years = _board_years()
    out: List[str] = []
    for n in _prose_boards(question):
        if n not in years:
            continue                    # out of range: _board_range_note owns it
        note = _inverse_note(
            by_board(n),
            f"funding-proposal documents in the corpus are from board meeting "
            f"B.{n} ({years[n]})",
            tail=" — the board is read from each document's own identifier; "
                 "the registry records no approval DECISION, so do not report "
                 "this listing as what the board approved")
        if note:
            out.append(note)
    return out


# ---------------------------------------------------------------------------
# corpus extrema (H3/H3b, probes P3 and F15): the money question the registry
# can settle, and the comparison it must refuse
# ---------------------------------------------------------------------------
# P3 asked for the smallest GCF funding request in the corpus and was refused —
# honestly, but the corpus knows. F15 asked the year-scoped form and got a
# WRONG value out of a note that printed the right one: the model compared
# '18.5 M USD' against '17,198,843 USD' by mantissa, which is what an
# unnormalised list of thirty raw strings invites. Schema 2 holds the
# normalised value, the currency, the unit and the page for every figure it
# read, so the ranking is a computation, not a judgement — and the refusals
# around it are computations too:
#
#   * only the CANONICAL candidate of a field is ranked (the template cell, not
#     every print in the document);
#   * only figures whose currency the document actually states as USD, because
#     'currency not printed for X - do not assume it is USD/EUR' is the
#     planner's own comparability law (planner.Comparability), and a note that
#     assumed it would be the same false authority in a new place;
#   * a figure whose unit the document's own mantissa contradicts ('28,654
#     million USD') has no value at all in schema 2 and cannot be ranked;
#   * every excluded document is COUNTED in the note, and an excluded figure
#     that falls beyond the answer is NAMED there, with the reason it is not
#     ranked. That is the F13 discipline applied to a comparison: a ranking
#     over a subset must say what the subset left out.

#: (field, label, the phrasings that name it). Total financing is tested first:
#: 'the largest total financing' also contains 'financing'.
_EXTREMA_FIELDS = (
    ("total_financing", "total financing", re.compile(
        r"\btotal\s+(?:financing|finance|cost|budget|project\s+(?:cost|value|"
        r"size))\b|\bfinancement\s+total\b|\bcout\s+total\b")),
    ("gcf_funding_requested", "GCF funding requested", re.compile(
        r"\bgcf\s+(?:funding|financing|grant|contribution|request\w*)\b"
        r"|\bfunding\s+request(?:ed|s)?\b|\brequests?\b|\brequested\b"
        r"|\bamount\s+(?:of\s+)?(?:gcf\s+)?(?:funding|financing)\b"
        r"|\bdemande\w*\s+de\s+financement\b"
        r"|\bfinancement\s+(?:du|de\s+la)\s+gcf\b"
        r"|\bmontant\s+(?:demande|du\s+financement|de\s+la\s+demande)\b")),
)
#: 'at least' is not a superlative, and it is the commonest 'least' in English.
_MIN_ASK_RE = re.compile(
    r"\b(?:smallest|lowest|minimum|min|cheapest)\b|(?<!at\s)\bleast\b"
    r"|\bplus\s+(?:petite?s?|faibles?|basse?s?)\b|\bmoindre\b")
_MAX_ASK_RE = re.compile(
    r"\b(?:largest|biggest|highest|greatest|maximum|max|most)\b"
    r"|\bplus\s+(?:grande?s?|eleve\w*|important\w*|haute?s?)\b")
#: The licence for a corpus-wide answer: the question has to ASK for one. Same
#: discipline as _SET_ASK_RE and as chainlit_app._CORPUS_TOKEN_RE, and the
#: reason 'which proposal requested the most?' — which could as easily mean
#: 'of the two above' — gets no authoritative ranking from here.
_CORPUS_SCOPE_RE = re.compile(
    r"\bcorpus\b|\bcollection\b|\bdataset\b|\bbase\s+documentaire\b"
    r"|\bever\b|\bof\s+all\b|\boverall\b|\bacross\s+all\b"
    r"|\ball\s+(?:the\s+)?(?:proposals|projects|documents)\b"
    r"|\bjamais\b|\bde\s+tous\b|\bde\s+toutes\b|\btout\s+le\s+corpus\b")
#: A superlative whose SUBJECT is not a proposal. 'Which country received the
#: most GCF funding in the corpus?' is a SUM across documents, and summing is
#: the failure F11 measured (totals 21-35x too high) and the one operation
#: `planner` refuses outright ("there is deliberately no 'calculated'
#: status"). The largest figure any single document states is not an answer to
#: it — it is P10's category error with a number attached — so the note stays
#: out and CORE's refusal stands.
_OTHER_SUBJECT_RE = re.compile(
    r"\b(?:countr(?:y|ies)|entit(?:y|ies|e|es)|organi[sz]ations?|agenc(?:y|ies)|"
    r"regions?|sectors?|themes?|years?|boards?|pays|annees?|secteurs?)\b")
_MIN, _MAX = "smallest", "largest"
#: Excluded figures NAMED beside the answer (each on its own line, so no line
#: ever pairs one document's stem with another document's page).
_MAX_NAMED_EXCLUSIONS = 2


def _asked_year(q: str) -> Optional[int]:
    """The one corpus year a question names, or None (two years are a range,
    and a range is _scan_years' question, not this one)."""
    years = {int(y) for y in re.findall(r"\b(20[12]\d)\b", q)}
    years &= set(_board_years().values())
    return next(iter(years)) if len(years) == 1 else None


def _extrema_ask(question: str):
    """(field, label, direction, year) for an extremum ask, or None.

    The guards, in order:

      * an FP id or a full board code -> the question names its documents, and
        a ranking over named documents is ``planner``'s, with its own
        comparability verdict and its own refusal text. This note never
        competes with it.
      * both a min word and a max word, or neither -> not a clean ask.
      * no money field named -> 'which years have the MOST funding proposals in
        this corpus?' is a count of documents, not of money (and is a gold
        case, `agg-year-most`).
      * a subject that is not a proposal -> 'which COUNTRY received the most
        GCF funding?' asks for a sum this corpus never computes.
      * no corpus word -> a YEAR-scoped superlative only fires for the MINIMUM.
        The reason is measured, not stylistic: the max arm of that shape is
        `agg-2020-largest`, which chainlit's _year_assist already answers at
        1.00 in every recorded run, and covering a passing case with a second
        authoritative note buys nothing and risks it. The min arm is F15, which
        is measured WRONG — the ranking is where the model actually fails,
        because a maximum can be eyeballed out of raw strings and a minimum
        cannot ('18.5 M USD' looks bigger than '17,198,843 USD'). A year-scoped
        MAXIMUM is available here the moment the question says 'in the corpus'.
    """
    q = _fold(question)
    if _FP_RE.search(question) or _BOARD_CODE_RE.search(question):
        return None
    lo, hi = bool(_MIN_ASK_RE.search(q)), bool(_MAX_ASK_RE.search(q))
    if lo == hi or _OTHER_SUBJECT_RE.search(q):
        return None
    hit = next(((f, label) for f, label, rx in _EXTREMA_FIELDS if rx.search(q)),
               None)
    if hit is None:
        return None
    year = _asked_year(q)
    if not _CORPUS_SCOPE_RE.search(q) and (year is None or not lo):
        return None
    return (*hit, _MIN if lo else _MAX, year)


def _money_figures(field: str, year: Optional[int]):
    """([the USD figures a ranking may use], {reason: [excluded item, ...]}).

    An item is {doc_id, fp, title, cand, why}: ``cand`` is the canonical schema
    2 candidate (``_canon2``, so it carries a source string and an integer
    page) and ``why`` is the sentence the note prints if that figure has to be
    named beside the answer.
    """
    ranked: List[dict] = []
    excluded: Dict[str, List[dict]] = {}
    for doc_id, row in load().items():
        if not isinstance(row, dict) or not row.get("fp"):
            continue
        if year is not None and row.get("year") != year:
            continue
        c = _canon2(_v2_facts(doc_id), field)
        item = {"doc_id": doc_id, "fp": row["fp"], "title": row.get("title"),
                "cand": c, "why": ""}
        if c is None:
            reason = "state no figure this registry could read for the field"
        elif c.get("value") is None:
            reason = ("print a figure whose unit the document's own mantissa "
                      "contradicts")
            item["why"] = "the unit as printed is ambiguous, so it has no value"
        elif not c.get("currency"):
            reason = "print no currency at all"
            item["why"] = ("the document states NO currency for this figure — "
                           "do not assume USD")
        elif c["currency"] != "USD":
            reason = f"state {c['currency']}"
            item["why"] = (f"the figure is stated in {c['currency']}, and this "
                           f"corpus carries no conversion rule")
        else:
            ranked.append(item)
            continue
        excluded.setdefault(reason, []).append(item)
    ranked.sort(key=lambda i: (i["cand"]["value"], i["fp"]))
    return ranked, excluded


def _beyond(excluded: Dict[str, List[dict]], direction: str, edge: float):
    """Excluded figures that fall past the answer, most extreme first.

    The ones that would have CHANGED the answer if the corpus had printed a
    currency for them. Naming them is not a cross-currency ranking: no
    conversion is claimed, the nominal relation is stated as nominal, and each
    line says why the figure is not comparable. The whole list is returned; the
    caller names the most extreme few and says how many there were, because a
    cut list that does not say it was cut is F13 all over again.
    """
    out = [i for v in excluded.values() for i in v
           if i["cand"] and i["cand"].get("value") is not None
           and ((i["cand"]["value"] < edge) if direction == _MIN
                else (i["cand"]["value"] > edge))]
    out.sort(key=lambda i: (i["cand"]["value"], i["fp"]),
             reverse=direction == _MAX)
    return out


def _extrema_note(question: str) -> List[str]:
    """The corpus/year extremum lines (H3/H3b), or [].

    The sentences avoid ever putting a FIELD LABEL at the head of a segment:
    ``verify._field_lines`` reads the value of a field as the first amount
    after its label when the label heads a ';'- or '. '-separated segment, and
    an instruction that ended '...the smallest GCF funding requested in the
    corpus.' put one there. It carried no amount and so could not manufacture a
    conflict, but a note line is read by three parsers and none of them should
    have to be lucky.

    These lines DO carry provenance — '(p.10, B.2(b)) [240_gcf-b15-13-add08]' —
    unlike every listing note in this module, and the choice is deliberate: a
    listing has no one page to point at, an extremum names exactly ONE figure
    in exactly ONE document, and that is the shape ``_note_pages`` and
    ``verify.note_page_scopes`` turn into a citable scope. So the model is told
    a page it may cite and the checker holds the same page. Each such line
    names ONE document, because those readers pair every page on a line with
    every stem on it.
    """
    ask = _extrema_ask(question)
    if ask is None:
        return []
    field, label, direction, year = ask
    ranked, excluded = _money_figures(field, year)
    total = len(ranked) + sum(len(v) for v in excluded.values())
    if not total:
        return []
    scope = f"the {year} funding proposals" if year else "the corpus"
    if not ranked:
        return [f"Registry — no document in {scope} states a {label} figure in "
                f"USD that this registry could read ({total} checked), so the "
                f"registry supports NO {direction}-figure answer here. Say so; "
                f"do not rank the figures the documents state in other "
                f"currencies against one another."]
    edge = (ranked[0] if direction == _MIN else ranked[-1])["cand"]["value"]
    tied = [i for i in ranked if i["cand"]["value"] == edge]
    best = tied[0]
    c = best["cand"]
    tie = ("" if len(tied) == 1 else
           " (TIED at this figure with "
           + ", ".join(f"FP{i['fp']}" for i in tied[1:]) + ")")
    # A canonical read from a 'rule:' section is a figure the builder found on
    # the page WITHOUT the template heading above it — the weaker provenance of
    # the two, and the corpus minimum for GCF funding requested is one of them
    # (22,953 on a p.10 financing table). Saying so is the difference between a
    # citable figure and a manufactured superlative.
    caution = ("" if not str(c.get("section") or "").startswith("rule:") else
               f" CAUTION: this figure was read from p.{c['page']} without a "
               f"labelled template heading above it (the section is inferred), "
               f"so check the page before repeating it as the {direction} "
               f"figure in {scope}.")
    lines = [f"Registry — {direction.upper()} {label} in {scope}: FP{best['fp']} "
             f'"{_clip(best.get("title"))}" — {_amount(c)} {_where(c)} '
             f"[{best['doc_id']}]{tie}. Ranked over the {len(ranked)} of {total} "
             f"documents in {scope} whose registry figure is an unambiguous USD "
             f"amount.{caution}"]
    beyond = _beyond(excluded, direction, edge)
    named = beyond[:_MAX_NAMED_EXCLUSIONS]
    word = "smaller" if direction == _MIN else "larger"
    if not beyond:
        reach = (f" None of the excluded figures is nominally {word} than the "
                 f"ranked answer.")
    elif len(beyond) == len(named):
        reach = (f" {len(beyond)} of the excluded figures "
                 f"{'is' if len(beyond) == 1 else 'are'} nominally {word} than "
                 f"the ranked answer, named below.")
    else:
        reach = (f" {len(beyond)} of the excluded figures are nominally {word} "
                 f"than the ranked answer; the {len(named)} most extreme are "
                 f"named below — LIST TRUNCATED.")
    if excluded:
        parts = "; ".join(
            f"{len(v)} {reason}" for reason, v in
            sorted(excluded.items(), key=lambda kv: (-len(kv[1]), kv[0])))
        lines.append(
            f"Registry — excluded from that comparison: {total - len(ranked)} "
            f"of {total} documents in {scope} — {parts}. Figures in different "
            f"currencies are never ranked against one another, and a figure "
            f"printed without a currency is not assumed to be USD.{reach}")
    for i in named:
        d = i["cand"]
        lines.append(
            f"Registry — NOT RANKED, though nominally "
            f"{'smaller' if direction == _MIN else 'larger'} than the figure "
            f"above: FP{i['fp']} prints {_amount(d)} {_where(d)} "
            f"[{i['doc_id']}] — {i['why']}. Report it only with that caveat, "
            f"and never as the {direction} figure in {scope}.")
    return lines


def _single_document(question: str) -> bool:
    """Does this question name exactly ONE document?

    The planner's own count, not a second one: `planner.detect` builds a matrix
    at two identifiers and returns None at one, and `_identifiers` is what it
    counts — including its collapse of two names for one document ('FP274 and
    GCF/B.42/02/Add.16' is one document, one row, one arity). Asking it here is
    what makes the field service the exact complement of the matrix: every turn
    is served by one mechanism or the other, and never by both.
    """
    p = _planner()
    if p is None:
        return False
    try:
        return len(p._identifiers(question)) == 1
    except Exception:                                          # noqa: BLE001
        return False


def registry_note(question: str) -> Optional[str]:
    """Computed corpus-metadata note for the answer model, or None."""
    if not load():
        return None
    q = question.lower()
    notes: List[str] = []
    resolved_docs: List[str] = []
    # The asked-for fields are served on the ONE document this turn names, and
    # on its first line only: a question that names the same document twice
    # (an FP id and its board code) prints two lines for it, and the fact
    # belongs on one of them, not on both.
    ask: Optional[str] = question if _single_document(question) else None
    for n in dict.fromkeys(_FP_RE.findall(q)):
        row = by_fp(int(n))
        if row:
            notes.append("Registry — " + _fmt(row, ask))
            ask = None
            notes += _conflict_lines(row)
            resolved_docs.append(row["doc_id"])
        else:
            notes.append(f"Registry — FP{n}: NOT FOUND in the 273-document corpus "
                         "registry. Do not infer details for it from other documents.")
    for b_, item_, add_ in dict.fromkeys(_BOARD_CODE_RE.findall(question)):
        row = resolve_board_code(int(b_), int(add_), int(item_) if item_ else None)
        code = _board_code_text(b_, item_, add_)
        if row:
            notes.append(f"Registry — {code} resolves to: " + _fmt(row, ask))
            ask = None
            notes += _conflict_lines(row)          # same enrichment, same helper
            resolved_docs.append(row["doc_id"])
        else:
            notes.append(f"Registry — {code}: NOT FOUND in the 273-document corpus "
                         "registry. Do not infer details for it from other documents.")
    if len(set(resolved_docs)) > 1:
        notes.append("Registry — the identifiers above resolve to DIFFERENT "
                     "documents. Never merge them or treat them as the same proposal.")
    # The year listing stays page-less on purpose: it names FP numbers, never
    # document stems, so _note_pages/note_page_scopes (which credit a page only
    # to a document named on the SAME line) would publish no scope for a page
    # printed here — and an uncreditable pointer is precisely the invented
    # citation the provenance exists to stop.
    for y in dict.fromkeys(re.findall(r"\b(20[12]\d)\b", q)):
        rows = [r for r in by_year(int(y)) if r.get("fp")]
        if rows:
            listing, more = _listing(rows, _MAX_YEAR_ROWS)
            notes.append(f"Registry — {len(rows)} funding-proposal documents from {y} "
                         f"in the corpus: {listing}{more}")
    # The inverse lookups (H1/H2), on the same page-less terms as the year
    # listing: FP numbers and titles, never a document stem, so no page a
    # reader could not credit is ever published from one of these lines.
    for inverse in (_country_note(question), _entity_note(question)):
        if inverse:
            notes.append(inverse)
    # The board listing (H16) is one of those, on the same page-less terms.
    notes += _board_note(question)
    # The extrema lines are NOT: an extremum is one figure in one document, so
    # each of those lines carries the page and the stem the readers can credit.
    notes += _extrema_note(question)
    return "\n".join(notes) or None
