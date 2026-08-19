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
separate file with a separate cache: the v1 lookups above never read it.
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


def _fmt(r: dict) -> str:
    bits = []
    if r.get("title"):
        bits.append(f'"{r["title"]}"')
    if r.get("accredited_entity"):
        bits.append(f"accredited entity: {r['accredited_entity']}")
    if r.get("countries"):
        bits.append("countries: " + ", ".join(r["countries"][:5]))
    if r.get("gcf_financing"):
        bits.append(f"GCF financing (as printed): {r['gcf_financing']}")
    if r.get("total_financing"):
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
            resolved_docs.append(row["doc_id"])
        else:
            notes.append(f"Registry — FP{n}: NOT FOUND in the 273-document corpus "
                         "registry. Do not infer details for it from other documents.")
    for b_, item_, add_ in dict.fromkeys(_BOARD_CODE_RE.findall(question)):
        row = resolve_board_code(int(b_), int(add_), int(item_) if item_ else None)
        code = _board_code_text(b_, item_, add_)
        if row:
            notes.append(f"Registry — {code} resolves to: " + _fmt(row))
            resolved_docs.append(row["doc_id"])
        else:
            notes.append(f"Registry — {code}: NOT FOUND in the 273-document corpus "
                         "registry. Do not infer details for it from other documents.")
    if len(set(resolved_docs)) > 1:
        notes.append("Registry — the identifiers above resolve to DIFFERENT "
                     "documents. Never merge them or treat them as the same proposal.")
    for y in dict.fromkeys(re.findall(r"\b(20[12]\d)\b", q)):
        rows = [r for r in by_year(int(y)) if r.get("fp")]
        if rows:
            listing = "; ".join(f"FP{r['fp']} \"{(r.get('title') or '?')[:45]}\"" for r in rows[:12])
            more = f" (+{len(rows) - 12} more)" if len(rows) > 12 else ""
            notes.append(f"Registry — {len(rows)} funding-proposal documents from {y} "
                         f"in the corpus: {listing}{more}")
    return "\n".join(notes) or None
