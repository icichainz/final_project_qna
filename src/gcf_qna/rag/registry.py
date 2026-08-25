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

Two lookups run in the OTHER direction: ``by_country`` and ``by_entity``, the
inverse index H1/H2 of docs/l1-l2-coverage-review.md found missing, each with
a ``registry_note`` trigger that fires only on a question asking for a SET of
proposals. The entity side normalises at lookup time — 126 stored spellings,
69 organisations — and never edits data/: the rows come back exactly as the
documents print them.

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
    held: List[str] = []
    for field in _MONEY_FIELDS:
        alts = [c for c in f2.get(field, [])
                if c.get("status") == "conflicting" and _usable(c)]
        if not alts:
            continue
        if len(out) >= _MAX_CONFLICT_LINES:
            held.append(field)               # the cap, said out loud below
            continue
        canon = _canon2(f2, field)
        printed = [_fig(c) for c in ([canon] if canon else [])
                   + alts[:_MAX_CONFLICT_ALTS]]
        cut = len(alts) - _MAX_CONFLICT_ALTS
        more = (f" (+{cut} more disagreeing print{'s' if cut > 1 else ''} of "
                f"this field in the document, not listed — list truncated)"
                if cut > 0 else "")
        ask = ("report both figures with their pages." if len(printed) == 2
               else "report all of them with their pages.")
        out.append(f"Registry — CONFLICT in this document ({r['doc_id']}): {field} "
                   f"is printed as {'; also as '.join(printed)}{more} — {ask}")
    if held:
        out.append(f"Registry — CONFLICT in this document ({r['doc_id']}): "
                   f"{len(held)} further field"
                   f"{'s' if len(held) > 1 else ''} ({', '.join(held)}) also "
                   f"print{'' if len(held) > 1 else 's'} disagreeing figures, "
                   f"not listed above — list truncated.")
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
#: and 'Acumen Fund' into 'Acumen' only because the shorter name is itself a
#: stored value. A tail whose remainder is not a stored value is left alone,
#: so 'World Wildlife Fund' never becomes 'World Wildlife'.
_ORG_TAIL = frozenset("group fund foundation".split())


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
        if key.split()[-1:] and key.split()[-1] in _ORG_TAIL and head in by_value:
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


def _inverse_note(rows: List[dict], lead: str, scope: str = "") -> Optional[str]:
    """One authoritative listing line for an inverse lookup, or None.

    The year listing's shape, above: the count first, FP ids and shortened
    titles after, no document stems and no page pointers — so
    ``_note_pages``/``note_page_scopes`` publish nothing uncreditable from it.

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
    return f"Registry — {n} {lead} ({state}): {listing}{more}"


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
            listing, more = _listing(rows, _MAX_YEAR_ROWS)
            notes.append(f"Registry — {len(rows)} funding-proposal documents from {y} "
                         f"in the corpus: {listing}{more}")
    # The inverse lookups (H1/H2), on the same page-less terms as the year
    # listing: FP numbers and titles, never a document stem, so no page a
    # reader could not credit is ever published from one of these lines.
    for inverse in (_country_note(question), _entity_note(question)):
        if inverse:
            notes.append(inverse)
    return "\n".join(notes) or None
