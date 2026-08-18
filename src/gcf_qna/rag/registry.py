"""Document metadata registry: deterministic lookups for cover-page facts.

Built once by scripts/build_registry.py (LLM over cover pages + financing
section; board/year derived from doc ids). Serves the fact classes that
chunk retrieval handles worst: 'which entity implements FPnnn',
'proposals of 2023', financing-at-a-glance. Financing values are RAW
quoted strings from the documents — never normalized numbers.
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


def by_fp(n: int) -> Optional[dict]:
    rows = [{"doc_id": k, **v} for k, v in load().items() if v.get("fp") == n]
    if not rows:
        return None
    # prefer the package document named after the FP over status/mention docs
    rows.sort(key=lambda r: (f"fp{n}" not in r["doc_id"], r["doc_id"]))
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


def registry_note(question: str) -> Optional[str]:
    """Computed corpus-metadata note for the answer model, or None."""
    if not load():
        return None
    q = question.lower()
    notes: List[str] = []
    for n in dict.fromkeys(re.findall(r"fp\s?(\d{2,3})", q)):
        row = by_fp(int(n))
        if row:
            notes.append("Registry — " + _fmt(row))
    for y in dict.fromkeys(re.findall(r"\b(20[12]\d)\b", q)):
        rows = [r for r in by_year(int(y)) if r.get("fp")]
        if rows:
            listing = "; ".join(f"FP{r['fp']} \"{(r.get('title') or '?')[:45]}\"" for r in rows[:12])
            more = f" (+{len(rows) - 12} more)" if len(rows) > 12 else ""
            notes.append(f"Registry — {len(rows)} funding-proposal documents from {y} "
                         f"in the corpus: {listing}{more}")
    return "\n".join(notes) or None
