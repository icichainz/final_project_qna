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
"""
from __future__ import annotations

import json
import re
import threading
from typing import Dict, List, Optional

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


def _conflict_lines(r: dict) -> List[str]:
    """One warning line per money field the document contradicts itself on.

    Each line is its own line, names the document id, and leads with the
    CANONICAL figure's '(p.N, SECTION)': verify.build_evidence keys a note line
    on the first such pointer it finds, so the warning becomes page-level
    evidence at the page an answer actually cites, holding every figure of the
    field — which is exactly what an answer that 'reports both figures with
    their pages' has to verify against.
    """
    f2 = _v2_facts(r.get("doc_id"))
    out: List[str] = []
    for field in _MONEY_FIELDS:
        if len(out) >= _MAX_CONFLICT_LINES:
            break
        alts = [c for c in f2.get(field, [])
                if c.get("status") == "conflicting" and _usable(c)]
        if not alts:
            continue
        canon = _canon2(f2, field)
        printed = [_fig(c) for c in ([canon] if canon else [])
                   + alts[:_MAX_CONFLICT_ALTS]]
        ask = ("report both figures with their pages." if len(printed) == 2
               else "report all of them with their pages.")
        out.append(f"Registry — CONFLICT in this document ({r['doc_id']}): {field} "
                   f"is printed as {'; also as '.join(printed)} — {ask}")
    return out


def _fmt(r: dict) -> str:
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
        bits.append("countries: " + ", ".join(r["countries"][:5])
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
_FP_RE = re.compile(r"fp[\s\-]?0*(\d{1,3})(?!\d)", re.I)
# public alias: the app imports this so the FP pattern has ONE definition
FP_RE = _FP_RE
BOARD_CODE_RE = _BOARD_CODE_RE


def resolve_fps(question: str):
    """(resolved rows, missing fp numbers) for every FP id in the question."""
    resolved, missing = [], []
    for n in dict.fromkeys(_FP_RE.findall(question.lower())):
        row = by_fp(int(n))
        (resolved if row else missing).append(row or int(n))
    return resolved, missing


def registry_note(question: str) -> Optional[str]:
    """Computed corpus-metadata note for the answer model, or None."""
    if not load():
        return None
    q = question.lower()
    notes: List[str] = []
    resolved_docs: List[str] = []
    for n in dict.fromkeys(_FP_RE.findall(q)):
        row = by_fp(int(n))
        if row:
            notes.append("Registry — " + _fmt(row))
            notes += _conflict_lines(row)
            resolved_docs.append(row["doc_id"])
        else:
            notes.append(f"Registry — FP{n}: NOT FOUND in the 273-document corpus "
                         "registry. Do not infer details for it from other documents.")
    for b_, item_, add_ in dict.fromkeys(_BOARD_CODE_RE.findall(question)):
        row = resolve_board_code(int(b_), int(add_), int(item_) if item_ else None)
        code = _board_code_text(b_, item_, add_)
        if row:
            notes.append(f"Registry — {code} resolves to: " + _fmt(row))
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
            listing = "; ".join(f"FP{r['fp']} \"{(r.get('title') or '?')[:45]}\"" for r in rows[:12])
            more = f" (+{len(rows) - 12} more)" if len(rows) > 12 else ""
            notes.append(f"Registry — {len(rows)} funding-proposal documents from {y} "
                         f"in the corpus: {listing}{more}")
    return "\n".join(notes) or None
