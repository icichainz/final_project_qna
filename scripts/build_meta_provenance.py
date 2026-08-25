#!/usr/bin/env python3
"""Derive page provenance for the cover-page META facts (entity/countries/title).

WHY
---
registry notes print page provenance beside the MONEY figures ("18.5 M USD
(p.5, A.8)") but the entity, countries and title lines carry no page. The
answer model is instructed to cite the page printed beside a fact, so for
those three it INVENTS one — right often enough to look fine, procedurally
fabricated every time, and flagged by every checker.

This script closes that gap at build time: for each document it finds the
cover page that carries a LABEL-ANCHORED line for the field ("Accredited
Entity:", "Country(ies):", "Project/Programme title:") whose content agrees
with the value registry.json already holds, and records that page plus the
source line.

OUTPUT CONTRACT (additive, optional, one key per doc entry)
-----------------------------------------------------------
    "meta_provenance": {
      "accredited_entity": {"page": <int>, "quote": "<source line, <=200 chars>"},
      "countries":         {...same...},
      "title":             {...same...}
    }

Only fields actually FOUND get an entry. A document with none found gets no
"meta_provenance" key at all — an absent page falls back to today's behaviour
(cite the document's cover pages) and costs nothing, while a WRONG page
poisons citations. Every other byte of registry_v2.json is preserved: the
file is re-serialized with the exact settings build_registry_v2.py uses
(ensure_ascii=False, indent=1), so the diff is pure insertion.

PRECISION OVER RECALL
---------------------
A page qualifies ONLY when a line on it parses as `LABEL: value` where LABEL
normalises to one of the accepted field labels. A bare value in prose is
never provenance: "Chile" appears in the body of dozens of proposals and
means nothing; "Country(ies): Chile" is the fact. Unanchored matching is not
implemented at all rather than gated, because a gate that can be tuned is a
gate that will eventually be loosened.

USAGE
-----
    python scripts/build_meta_provenance.py                  # dry run: report only
    python scripts/build_meta_provenance.py --out /tmp/x.json
    python scripts/build_meta_provenance.py --in-place       # rewrite registry_v2

Writing is opt-in because data/registry_v2.json is integrity-anchored in
data/eval/CHECKSUMS.sha256; regenerating it invalidates that line and the
anchor must be re-recorded deliberately, not as a side effect of a dry run.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_V1 = REPO_ROOT / "data" / "registry.json"
DEFAULT_REGISTRY_V2 = REPO_ROOT / "data" / "registry_v2.json"
DEFAULT_CORPUS = REPO_ROOT / "data" / "extracted" / "vlm" / "qwen_qwen2.5-vl-7b"

FIELDS: Tuple[str, ...] = ("accredited_entity", "countries", "title")
COVER_PAGES = 12          # how deep into the document a cover-page label may sit
QUOTE_MAX = 200           # contract: quote trimmed to <=200 chars
_MAX_VALUE_LINES = {"title": 4, "accredited_entity": 4, "countries": 10}


# --------------------------------------------------------------------------
# page splitting
# --------------------------------------------------------------------------

# The VLM extractor delimits pages with:   ---\n**Page 3**\n---
_PAGE_MARKER = re.compile(r"^---[ \t]*\n\*\*Page (\d+)\*\*[ \t]*\n---[ \t]*$", re.M)


def split_pages(text: str) -> List[Tuple[int, str]]:
    """[(page_number, page_body)] in document order, cover pages first."""
    marks = list(_PAGE_MARKER.finditer(text))
    pages: List[Tuple[int, str]] = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        try:
            num = int(m.group(1))
        except ValueError:  # pragma: no cover - regex guarantees digits
            continue
        pages.append((num, text[m.end():end]))
    return pages


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def norm(s: str) -> str:
    """Casefold, drop accents, reduce every non-alphanumeric run to one space."""
    s = _strip_accents(str(s or "")).lower()
    s = re.sub(r"[^0-9a-z]+", " ", s)
    return s.strip()


def tokens(s: str) -> List[str]:
    return [t for t in norm(s).split() if t]


# Words that carry no identity: "The World Bank" and "World Bank" are the same
# accredited entity, and a French title's articles say nothing about which
# project it is.
_STOPWORDS = frozenset({
    "the", "of", "and", "for", "a", "an", "to", "in", "on", "at", "by", "with",
    "de", "del", "la", "le", "les", "du", "des", "et", "y", "el", "los", "las",
    "da", "do", "dos", "das", "e", "il", "der", "die", "das", "und", "von",
})


def distinctive(s: str) -> List[str]:
    """Identity-bearing tokens: stopwords and single characters removed."""
    seen: List[str] = []
    for t in tokens(s):
        if t in _STOPWORDS or len(t) < 2:
            continue
        if t not in seen:
            seen.append(t)
    return seen


# --------------------------------------------------------------------------
# label recognition
# --------------------------------------------------------------------------

# Section numbering that prefixes the label in the funding-proposal template:
#   "A.1.5. Accredited entity"   "A1.1 Project / programme title"   "(a) Pays"
_SECTION_PREFIX = re.compile(
    r"^(?:[\(\[]?[a-z][\.\)\]]?\s*)?(?:[a-z]?\s*\d+(?:\s*[\.\-]\s*\d+)*[\.\)]?\s+)", re.I)
# Markdown furniture: heading hashes, bold/italic stars, list bullets, quotes.
_MD_FURNITURE = re.compile(r"^[\s>#*_\-•·]+|[\s*_:]+$")

# Filler tokens inside a label that carry no meaning: the "(ies)" of
# "Country(ies)", the "(s)" of "Countries(s)", the "/region" of
# "Country/Region", the "or" of "Project or programme title".
_LABEL_FILLER = frozenset({"ies", "es", "s", "region", "regions", "or"})

_ACCEPTED_LABELS: Dict[str, frozenset] = {
    "accredited_entity": frozenset({
        "accredited entity", "accredited entities",
        "entite accreditee", "entite accredite", "entites accreditees",
    }),
    "countries": frozenset({
        "country", "countries", "pays",
    }),
    "title": frozenset({
        "project programme title", "project program title",
        "project title", "programme title", "program title",
        "project programme name", "titre du projet", "titre",
        "titre du projet programme", "projet programme titre",
    }),
}


def normalise_label(raw: str) -> str:
    """Reduce a candidate label to its comparable core.

    "### A.1.3 Country (ies) / region:" -> "country"
    "| Accredited Entity: |"            -> "accredited entity"
    """
    s = _MD_FURNITURE.sub("", str(raw or "").strip())
    s = _SECTION_PREFIX.sub("", s.strip())
    parts = [t for t in norm(s).split() if t and t not in _LABEL_FILLER]
    return " ".join(parts)


def label_field(raw: str) -> Optional[str]:
    """Which field this label names, or None when it is not a field label."""
    core = normalise_label(raw)
    if not core:
        return None
    for field, accepted in _ACCEPTED_LABELS.items():
        if core in accepted:
            return field
    return None


# --------------------------------------------------------------------------
# line parsing:  label / value split
# --------------------------------------------------------------------------

def parse_label_line(line: str) -> Optional[Tuple[str, str, str]]:
    """(field, inline_value, label_text) for a label-anchored line, else None.

    Handles the three shapes the corpus actually uses:
        "## Accredited Entity:"                  -> value on following lines
        "Accredited Entity: FMO"                 -> value inline
        "| Country/Region: | Burkina Faso |"     -> value in the next table cell
    """
    if not line or not line.strip():
        return None

    stripped = line.strip()

    # --- markdown table row -------------------------------------------------
    if stripped.startswith("|"):
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) < 2:
            return None
        # A key/value row is two cells ("| Country/Region: | Burkina Faso |").
        # A wider row is the header of a data table ("| Country | Climate
        # Hazard addressed by this Annex | High |") whose second cell is
        # another column heading, not the country — unless the label cell
        # carries the colon that marks it as a key. Requiring the colon
        # everywhere was tried and costs real coverage: several documents
        # write the key/value row without one.
        if len(cells) > 2 and not cells[0].endswith(":"):
            return None
        field = label_field(cells[0])
        if field:
            return field, cells[1], _clean_label(cells[0])
        return None

    # --- "Label: value" -----------------------------------------------------
    if ":" in stripped:
        head, _, tail = stripped.partition(":")
        field = label_field(head)
        if field:
            return field, tail.strip(), _clean_label(head)

    # --- bare heading, value on the next line --------------------------------
    field = label_field(stripped)
    if field:
        return field, "", _clean_label(stripped)
    return None


def _clean_label(raw: str) -> str:
    """The label as a reader should see it: markdown furniture removed.

    "### A.1.3 Country (ies) / region:" -> "A.1.3 Country (ies) / region"
    """
    return _MD_FURNITURE.sub("", str(raw or "").strip()).strip()


def _is_boundary(line: str) -> bool:
    """A line that ends the value region of the label above it."""
    s = line.strip()
    if s.startswith("#"):
        return True
    # A table row is self-contained: its value sits in its own cell, so the row
    # below is the NEXT fact ("| Accredited Entity: | The World Bank |" must not
    # swallow "| Date of Submission: | Jan 19th, 2017 |").
    if s.startswith("|"):
        return True
    if parse_label_line(line) is not None:
        return True
    return False


def collect_value(lines: Sequence[str], idx: int, inline: str, field: str) -> str:
    """The value text belonging to the label at `lines[idx]`.

    The inline value plus the continuation lines beneath it. Country lists in
    particular wrap over several lines and, in the multi-country programmes,
    are broken into blank-line-separated regional groups ("Africa: ...",
    "Asia-Pacific: ...") that all belong to the one field. Titles and entity
    names are single-valued, so for those the first blank line ends the value.
    """
    max_lines = _MAX_VALUE_LINES.get(field, 4)
    # Consecutive blank lines tolerated once the value has started.
    gap_allowance = 1 if field == "countries" else 0
    parts: List[str] = []

    if inline.strip():
        parts.append(inline.strip())

    blanks = 0
    taken = 0
    j = idx + 1
    while j < len(lines) and taken < max_lines:
        nxt = lines[j]
        if not nxt.strip():
            # Before the value starts, a blank is just spacing under the label.
            if parts and blanks >= gap_allowance:
                break
            blanks += 1
            if blanks > 2:
                break
            j += 1
            continue
        if _is_boundary(nxt):
            break
        blanks = 0
        parts.append(nxt.strip().strip("|").strip())
        taken += 1
        j += 1

    return " ".join(parts).strip()


def trim_quote(s: str) -> str:
    """Collapse whitespace and trim to the contract's 200-character ceiling."""
    s = re.sub(r"\s+", " ", str(s or "")).strip()
    return s[:QUOTE_MAX].strip()


# --------------------------------------------------------------------------
# value matching
# --------------------------------------------------------------------------

_PAREN = re.compile(r"\(([^)]*)\)")
_ACRONYM_TOKEN = re.compile(r"\b([A-Z]{2,8})\b")
_PREFIX_MIN = 5


def acronyms_of(registry_value: str) -> List[str]:
    """Short names the document may use in place of the full entity name.

    Both shapes occur in the corpus: parenthesised ("French Development Agency
    (AFD)") and prefixed ("IUCN - International Union for Conservation of
    Nature"). Either is the entity, so either identifies the page.
    """
    out: List[str] = []
    for m in _PAREN.findall(str(registry_value or "")):
        n = norm(m)
        if n and 2 <= len(n) <= 8 and not _is_roman(n):
            out.append(n)
    for m in _ACRONYM_TOKEN.findall(str(registry_value or "")):
        n = norm(m)
        # An unparenthesised run of capitals is only taken as an acronym at 3+
        # characters and never as a Roman numeral: the "II" of "Programme II"
        # would otherwise match any page whose entity cell carries a numeral.
        if n and len(n) >= 3 and not _is_roman(n) and n not in out:
            out.append(n)
    return out


def _is_roman(s: str) -> bool:
    return bool(s) and set(s) <= set("ivxlcdm")


def _token_hit(tok: str, pool: set) -> bool:
    """Token equality, tolerant of the morphology drift the VLM introduces.

    "Environment Investment Fund" and "Environmental Investment Fund" are the
    same accredited entity; a prefix relationship on tokens long enough to be
    distinctive absorbs that without letting short words collide.
    """
    if tok in pool:
        return True
    if len(tok) < _PREFIX_MIN:
        return False
    return any(len(p) >= _PREFIX_MIN and (p.startswith(tok) or tok.startswith(p))
               for p in pool)


def entity_matches(registry_value: str, blob: str) -> bool:
    """Does `blob` name the accredited entity the registry records?

    Accepts the full name, the name without its acronym, the acronym alone
    ("AFD" for "French Development Agency (AFD)"), and near-complete token
    coverage for the spelling drift the VLM introduces.
    """
    rv = norm(registry_value)
    bl = norm(blob)
    if not rv or not bl:
        return False
    if rv in bl:
        return True

    core = norm(_PAREN.sub(" ", str(registry_value)))
    if core and len(core) >= 6 and core in bl:
        return True
    bl_tokens = set(bl.split())
    for ac in acronyms_of(registry_value):
        if ac in bl_tokens:
            return True

    toks = distinctive(rv)
    if len(toks) < 2:
        return False

    # The page may print the initialism the registry spells out: "UNDP" for
    # "United Nations Development Programme". Three letters minimum, so that a
    # two-word entity cannot be claimed by an unrelated pair of capitals.
    if len(toks) >= 3:
        initials = "".join(t[0] for t in toks)
        if len(initials) >= 3 and initials in bl_tokens:
            return True

    # The page may print a shorter form of the same name: "Agency for
    # Agricultural Development" where the registry holds "Agency for
    # Agricultural Development - ADA". Everything on the page must be part of
    # the registry name, and there must be enough of it to identify anyone.
    reg_tokens = set(toks)
    blob_distinctive = distinctive(bl)
    if len(blob_distinctive) >= 3 and all(_token_hit(t, reg_tokens)
                                          for t in blob_distinctive):
        return True

    hits = sum(1 for t in toks if _token_hit(t, bl_tokens))
    return hits >= 2 and hits / len(toks) >= 0.8


def countries_match(registry_value, blob: str) -> bool:
    """Does `blob` list the countries the registry records?

    Short lists must match completely; long regional lists (the multi-country
    programmes run to 40+ entries) need a majority, because the VLM drops or
    garbles a few names in a long semicolon-separated run.
    """
    if isinstance(registry_value, str):
        names = [registry_value]
    else:
        names = list(registry_value or [])
    names = [n for n in names if norm(n)]
    if not names:
        return False

    bl = norm(blob)
    if not bl:
        return False
    present = 0
    for n in names:
        # Word-boundary match: "Chad" must not be satisfied by "Chadian", and
        # "Mali" must not be satisfied by "Malawi".
        if re.search(rf"(?<![a-z0-9]){re.escape(norm(n))}(?![a-z0-9])", bl):
            present += 1
    if len(names) <= 2:
        return present == len(names)
    return present / len(names) >= 0.5


def title_matches(registry_value: str, blob: str) -> bool:
    """Does `blob` carry the project title the registry records?

    Compared in whichever direction is the shorter string: the cover page often
    prints a clipped title ("Scaling up the Deployment of Integrated Utility
    Services (IUS)") where the registry holds the full one, and vice versa.
    """
    a = distinctive(registry_value)
    b = distinctive(blob)
    if len(a) < 2 or len(b) < 2:
        return False
    shared = set(a) & set(b)
    if len(shared) < 3:
        # Very short titles cannot clear a 3-token bar; require them whole.
        if len(a) <= 3 and set(a) <= set(b):
            return True
        return False
    return max(len(shared) / len(a), len(shared) / len(b)) >= 0.6


_MATCHERS = {
    "accredited_entity": entity_matches,
    "countries": countries_match,
    "title": title_matches,
}


# --------------------------------------------------------------------------
# per-document extraction
# --------------------------------------------------------------------------

def extract_document(text: str, meta: dict,
                     cover_pages: int = COVER_PAGES) -> Dict[str, dict]:
    """meta_provenance for one document: {field: {"page": int, "quote": str}}.

    Walks the cover region in page order and returns the FIRST page whose
    label-anchored line for a field agrees with that field's registry value.
    Fields with no qualifying page are simply absent.
    """
    found: Dict[str, dict] = {}
    for page_no, body in split_pages(text):
        if page_no < 1 or page_no > cover_pages:
            continue
        lines = body.split("\n")
        for idx, line in enumerate(lines):
            parsed = parse_label_line(line)
            if parsed is None:
                continue
            field, inline, label_text = parsed
            if field in found or field not in FIELDS:
                continue
            registry_value = meta.get(field)
            if registry_value in (None, "", []):
                continue
            blob = collect_value(lines, idx, inline, field)
            if not blob:
                continue
            if not _MATCHERS[field](registry_value, blob):
                continue
            quote = trim_quote(f"{label_text}: {blob}" if label_text else blob)
            if not quote:
                continue
            found[field] = {"page": int(page_no), "quote": quote}
        if len(found) == len(FIELDS):
            break
    # Stable key order so the serialized file is byte-deterministic.
    return {f: found[f] for f in FIELDS if f in found}


def corpus_path_for(doc_id: str, corpus_dir: Path) -> Optional[Path]:
    p = corpus_dir / f"{doc_id}.md"
    return p if p.exists() else None


def build(registry_v1: Dict[str, dict], corpus_dir: Path,
          only: Optional[Iterable[str]] = None,
          cover_pages: int = COVER_PAGES) -> Dict[str, Dict[str, dict]]:
    """{doc_id: meta_provenance} for every doc that yielded at least one field."""
    wanted = set(only) if only else None
    out: Dict[str, Dict[str, dict]] = {}
    for doc_id in sorted(registry_v1):
        if wanted is not None and doc_id not in wanted:
            continue
        path = corpus_path_for(doc_id, corpus_dir)
        if path is None:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        prov = extract_document(text, registry_v1[doc_id], cover_pages=cover_pages)
        if prov:
            out[doc_id] = prov
    return out


# --------------------------------------------------------------------------
# registry_v2 augmentation
# --------------------------------------------------------------------------

def strip_meta_provenance(registry_v2: dict) -> dict:
    """A copy of the registry with every meta_provenance key removed.

    The additive-only invariant is proved by deep-comparing this against the
    pre-existing file: if anything else moved, the comparison fails.
    """
    out = json.loads(json.dumps(registry_v2))
    for entry in out.get("documents", {}).values():
        entry.pop("meta_provenance", None)
    return out


def augment(registry_v2: dict, provenance: Dict[str, Dict[str, dict]]) -> dict:
    """registry_v2 with meta_provenance added; nothing else touched.

    Any meta_provenance already present is replaced wholesale, so a re-run is
    idempotent rather than cumulative.
    """
    out = json.loads(json.dumps(registry_v2))
    for doc_id, entry in out.get("documents", {}).items():
        entry.pop("meta_provenance", None)
        prov = provenance.get(doc_id)
        if prov:
            entry["meta_provenance"] = prov
    return out


def dumps(registry: dict) -> str:
    """Serialize exactly as scripts/build_registry_v2.py does (byte-faithful)."""
    return json.dumps(registry, ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _report(registry_v1: Dict[str, dict], provenance: Dict[str, Dict[str, dict]],
            stream=sys.stdout) -> None:
    total = len(registry_v1)
    w = stream.write
    w(f"documents in registry: {total}\n")
    w(f"documents with any provenance: {len(provenance)} "
      f"({len(provenance) / total:.1%})\n\n")
    for field in FIELDS:
        hits = {d: p[field] for d, p in provenance.items() if field in p}
        pages = Counter(v["page"] for v in hits.values())
        w(f"{field}: {len(hits)}/{total} ({len(hits) / total:.1%})\n")
        w("  pages: " + ", ".join(f"p{p}={n}" for p, n in sorted(pages.items())) + "\n")
    missing_all = sorted(d for d in registry_v1 if d not in provenance)
    w(f"\ndocuments with NO provenance at all ({len(missing_all)}):\n")
    for d in missing_all:
        w(f"  {d}\n")
    for field in FIELDS:
        gap = sorted(d for d in registry_v1 if field not in provenance.get(d, {}))
        w(f"\nmissing {field} ({len(gap)}):\n")
        for d in gap:
            w(f"  {d}\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_V1,
                    help="registry.json (v1 cover-page values to match against)")
    ap.add_argument("--registry-v2", type=Path, default=DEFAULT_REGISTRY_V2,
                    help="registry_v2.json to augment")
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS,
                    help="directory of extracted per-document markdown")
    ap.add_argument("--cover-pages", type=int, default=COVER_PAGES,
                    help="how many leading pages count as the cover region")
    ap.add_argument("--only", action="append", default=None,
                    help="restrict to this doc id (repeatable)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the augmented registry here")
    ap.add_argument("--in-place", action="store_true",
                    help="rewrite --registry-v2 in place (invalidates its "
                         "line in data/eval/CHECKSUMS.sha256)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    registry_v1 = json.loads(args.registry.read_text(encoding="utf-8")).get("documents", {})
    provenance = build(registry_v1, args.corpus, only=args.only,
                       cover_pages=args.cover_pages)

    if not args.quiet:
        _report(registry_v1, provenance)

    target = args.registry_v2 if args.in_place else args.out
    if target is not None:
        registry_v2 = json.loads(args.registry_v2.read_text(encoding="utf-8"))
        augmented = augment(registry_v2, provenance)
        assert strip_meta_provenance(augmented) == strip_meta_provenance(registry_v2), \
            "additive-only invariant violated: something other than meta_provenance changed"
        target.write_text(dumps(augmented), encoding="utf-8")
        if not args.quiet:
            sys.stdout.write(f"\nwrote {target}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
