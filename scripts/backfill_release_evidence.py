#!/usr/bin/env python3
"""Reconstruct the evidence a recorded release run actually held, as a sidecar.

`scripts/eval_answers.py` records, per case, the *keys* of the evidence the
answer path held (`claims.evidence_keys`) and a lossy hit list
(`{doc, page, score}` — no text).  The adjudication labels are defined over
the evidence itself, so a human cannot separate `missing_citation` from
`missing_retrieval_evidence` from `verifier_false_positive` out of a release
row alone.  This script rebuilds the missing half and writes it beside the
release as an immutable sidecar.  It NEVER writes to the release.

What is exactly recoverable
    * the evidence KEY structure — `verify.build_evidence` is re-run over the
      recorded hits and note blocks and its keys are compared, in order,
      against the recorded `claims.evidence_keys`; a mismatch is reported per
      case and marks that case's reconstruction incomplete;
    * the note blocks (`notes_used.registry` / `.year`) — recorded verbatim;
    * the full claim text, its char span in the answer, and its citations —
      `verify.extract_claims` and `verify.parse_citations` are byte-identical
      (AST) to the revision that produced the release, and re-extraction
      reproduces the recorded claim counts exactly.

What is only approximately recoverable
    * hit TEXT.  A hit is one chunk; the record keeps only `(doc, page)`, and
      a page can hold up to 27 chunks.  Text is recovered from the index by
      `(doc_id, page)` and is therefore a PAGE SUPERSET of what the prompt
      held whenever the page has more than one chunk.  Every evidence entry
      carries `fidelity: "exact" | "page-superset"` and `chunks_on_page`, and
      every claim carries `evidence_fidelity`.  A superset can only make
      evidence look MORE complete than it was, so "term present" is an upper
      bound on what the prompt held.

The presence probe is a literal / hyphen-tolerant / separator-agnostic search
written out in this file.  It calls no model, and it deliberately does not
reuse the verifier's disputed matchers: it reports WHERE a term occurs and
WHAT SPELLING the page uses, and leaves the label to the human.

The probe's own result is never used to rule a label out.  An earlier version
excluded `missing_citation` whenever a term was absent, on the (sound)
superset argument that "absent from the reconstruction" implies "absent at run
time".  The argument was sound and the probe was not: the corpus prints a
comma decimal mark ('USD 49,312 million') where the answers print a period, so
every one of release-1's five absent amounts was a false absence and three
uncited claims had their correct label struck off.  Exclusions are now
structural only; probe results are reported, and every held page shows the
reviewer its text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gcf_qna.rag import verify  # noqa: E402
from adjudicate_claims import LABELS, _claim_id  # noqa: E402  (one identity function)


SCHEMA_VERSION = 1
NOTES_DOC = verify.NOTES_DOC

#: Only these note blocks reached ``verify.build_evidence`` — ``eval_answers``
#: passes ``[registry, year, matrix]`` while ``notes_used`` records every
#: truthy note, ``board`` included.  Reconstructing from ``notes_used`` wholesale
#: would invent evidence the answer never held.
EVIDENCE_NOTE_KEYS = ("registry", "year", "matrix")

DEFAULT_RELEASE = ROOT / "data" / "eval" / "release_release-1.jsonl"
DEFAULT_CHUNKS = ROOT / "data" / "index" / "default" / "chunks.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "eval" / "release_release-1-evidence.jsonl"
CHECKSUMS = ROOT / "data" / "eval" / "CHECKSUMS.sha256"


class BackfillError(ValueError):
    """Raised when a release, an index or an output path is unusable."""


# ---------------------------------------------------------------------------
# io helpers
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise BackfillError(f"cannot read {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BackfillError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise BackfillError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise BackfillError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def protected_paths(checksums: Path = CHECKSUMS) -> set[Path]:
    """Every file the measurement-anchor checksum list covers, resolved.

    The sidecar is derived from these; writing one on top of one would
    destroy the anchor the whole rollout is measured against.
    """
    protected: set[Path] = set()
    try:
        lines = checksums.read_text(encoding="utf-8").splitlines()
    except OSError:
        return protected
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        name = parts[1].strip().lstrip("*")
        candidate = Path(name)
        if not candidate.is_absolute():
            candidate = checksums.resolve().parents[2] / candidate
        protected.add(candidate.resolve())
    return protected


def assert_writable_output(
    output: Path, *, inputs: Iterable[Path], force: bool = False
) -> None:
    resolved = output.resolve()
    for source in inputs:
        if resolved == source.resolve():
            raise BackfillError(
                f"refusing to write the sidecar over its own input {source}"
            )
    if resolved in protected_paths():
        raise BackfillError(
            f"refusing to write {output}: it is a checksummed measurement anchor"
        )
    if resolved.exists() and not force:
        raise BackfillError(f"refusing to overwrite {output}; pass --force")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    temporary: Optional[Path] = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        temporary.chmod(0o644)          # NamedTemporaryFile defaults to 0600
        temporary.replace(path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise BackfillError(f"cannot write {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# chunk text recovery
# ---------------------------------------------------------------------------


def wanted_pages(release_rows: Iterable[dict[str, Any]]) -> set[tuple[str, Any]]:
    wanted: set[tuple[str, Any]] = set()
    for row in release_rows:
        for hit in row.get("hits") or ():
            if isinstance(hit, dict) and hit.get("doc"):
                wanted.add((hit["doc"], hit.get("page")))
    return wanted


def load_page_texts(
    chunks_path: Path, wanted: set[tuple[str, Any]]
) -> dict[tuple[str, Any], list[str]]:
    """``{(doc_id, page): [chunk_text, ...]}`` in file order, for wanted pages."""
    pages: dict[tuple[str, Any], list[str]] = defaultdict(list)
    if not wanted:
        return dict(pages)
    try:
        handle = chunks_path.open("r", encoding="utf-8")
    except OSError as exc:
        raise BackfillError(f"cannot read {chunks_path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BackfillError(
                    f"{chunks_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            key = (chunk.get("doc_id"), chunk.get("page"))
            if key in wanted:
                pages[key].append(chunk.get("text") or "")
    return dict(pages)


# ---------------------------------------------------------------------------
# presence probe  (literal, local, model-free)
# ---------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_MAP = str.maketrans(
    {
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "−": "-", "‑": "-",
        " ": " ", " ": " ", " ": " ",
    }
)


def display(text: str) -> str:
    """Punctuation- and whitespace-normalized text, case preserved.

    Excerpts are cut out of this string, and ``normalize`` only lowercases it,
    so a match offset found in the normalized text indexes the display text at
    the same position — no offset mapping, nothing to get subtly wrong.
    """
    text = unicodedata.normalize("NFKC", text or "").translate(_PUNCT_MAP)
    return _WS_RE.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Lowercased :func:`display` text — the literal-search index."""
    return display(text).lower()


def loose(text: str) -> str:
    """:func:`normalize` text with hyphens read as spaces.

    'the relevant Board-meeting figures' and 'date of GCF's board meeting' are
    the same phrase to a reader, and release-1 has a claim whose *only* probed
    term is absent for exactly that reason.  The substitution is one character
    for one character, so a match offset found here still indexes the
    :func:`display` string at the same position — no offset mapping.
    """
    return normalize(text).replace("-", " ")


#: one printed figure: digit runs joined by '.' or ','  ('49,312', '6,876,446.27').
#: A space between digits is deliberately NOT a separator: the corpus prints
#: 'Direct beneficiary sec. 5 155,000 direct beneficiaries' and 'total cost
#: (USD) - 240.0 80.0 160.0', so reading a space as a grouping mark would fuse
#: adjacent figures and DELETE them from the index — the same false-absence
#: failure this probe exists to fix, in the opposite direction.
_NUMBER_TOKEN_RE = re.compile(r"(?<![\d.,])\d+(?:[.,]\d+)*")


def _digits_of(token: str) -> str:
    return re.sub(r"\D", "", token)


def figure_key(token: str) -> str:
    """A printed figure with its separator characters made interchangeable.

    '49,312' and '49.312' both become '49|312'; the GROUPING survives, only
    the choice of mark is erased.  '4,931.2' stays '4|931|2' and therefore
    does not collide, which digits-only comparison ('49312' for all three)
    would not have caught.
    """
    return "|".join(re.split(r"[.,]", token))


def number_index(normalized: str) -> dict[str, list[str]]:
    """``{figure key: [spelling, ...]}`` for every figure in normalized text.

    The spellings are what the page actually prints, so a row can show the
    reviewer '49,312 million' beside the answer's '49.312 million' instead of
    asserting one of them is absent.
    """
    index: dict[str, list[str]] = defaultdict(list)
    for match in _NUMBER_TOKEN_RE.finditer(normalized):
        token = match.group(0)
        if len(_digits_of(token)) >= 3:
            key = figure_key(token)
            if token not in index[key]:
                index[key].append(token)
    return dict(index)


def separator_agnostic_key(term: str) -> str:
    """:func:`figure_key` of the first internally-separated figure, else ``''``.

    THE BUG THIS EXISTS FOR: the corpus prints a comma where the answers print
    a period — 'USD 49,312 million' against the answer's 'USD 49.312 million'.
    A literal search misses it, and :func:`verify.amounts` misses it twice
    over: 49,312 parses as a US thousands group whose printed scale word
    contradicts it, so ``value`` is None and ``bare`` is 49312.0 against the
    answer's 49312000.0 / 49.312.  Neither ``value`` nor ``mantissa`` nor the
    literal fallback can see it, and every one of release-1's five "absent"
    amounts was absent for this reason alone.

    Reading the separators as interchangeable is the only reading true under
    either convention.  It is deliberately a PRESENCE probe and never a value
    claim: whether 49,312 and 49.312 are the same magnitude is precisely the
    question the row now puts in front of the human, spelling included.

    Only figures with an INTERNAL separator qualify.  'FP172', 'B.30' and
    'p.45' have none, and probing the corpus for a bare '172' would report
    presence from any page number that happened to be 172.
    """
    for match in _NUMBER_TOKEN_RE.finditer(normalize(term)):
        token = match.group(0)
        if re.search(r"\d[.,]\d", token) and len(_digits_of(token)) >= 3:
            return figure_key(token)
    return ""


#: A match strong enough to rule a label out.  ``digits`` is not: it says the
#: page prints the same digits, not that it prints the same figure.
STRICT_MATCH_KINDS = frozenset({"literal", "hyphen-variant", "value", "mantissa"})


def _amount_signature(amount: Any) -> tuple[Optional[float], float]:
    return (getattr(amount, "value", None), getattr(amount, "bare", 0.0))


def _amounts_agree(a: Any, b: Any) -> str:
    """"value" | "mantissa" | "" — deliberately explicit, not verify's matcher.

    ``value`` compares the fully-scaled figures inside the coarser of the two
    printed grains ('18.5 million' and '18,500,000' agree).  ``mantissa`` is
    the weaker fallback used when a printed scale word contradicts its own
    mantissa and ``value`` is therefore unknown.

    Both sides come from :func:`verify.amounts`, which reads a comma as a
    thousands separator.  Under the corpus's comma decimal mark that is wrong
    by a factor of 1000 ('USD 1,066 million' parses as 1.066 billion), so
    neither branch fires on a separator mismatch — see
    :func:`separator_agnostic_key`, which is what covers that case.
    """
    av, ab = _amount_signature(a)
    bv, bb = _amount_signature(b)
    if av is not None and bv is not None:
        grain = max(getattr(a, "grain", 1.0) or 1.0, getattr(b, "grain", 1.0) or 1.0)
        if abs(av - bv) <= grain / 2:
            return "value"
    if ab and bb and ab == bb:
        return "mantissa"
    return ""


_NOT_FOUND_RE = re.compile(r"not found in the cited evidence:\s*(.+)$", re.I)
#: the verifier joins several missing targets with ', '; a thousands separator
#: inside one figure ('49,751,264') never carries the space, so splitting on
#: comma-plus-space keeps figures whole.
_TARGET_SPLIT_RE = re.compile(r",\s+")
_ANSWER_SIDE_RE = re.compile(r"the answer states '([^']{2,80})'", re.I)
_EVIDENCE_SIDE_RE = re.compile(
    r"(?:cited evidence states|the cited document also prints|"
    r"the corpus registry records[^']*|registry states)\s*'([^']{2,80})'",
    re.I,
)
_QUOTED_RE = re.compile(r"'([^']{2,80})'")


def reason_targets(reason: str) -> list[dict[str, Any]]:
    """The concrete strings the recorded verifier reason names, with their side.

    These are the verifier's own words about the evidence it held, so probing
    them says directly whether the recorded reason still describes the
    evidence we reconstructed:  an ``evidence-side`` value that the
    reconstruction cannot find means the reconstruction is not faithful; an
    ``answer-side`` value it cannot find is the expected shape of a real
    contradiction.
    """
    reason = reason or ""
    truncated = len(reason) >= 160
    out: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(term: str, source: str, *, maybe_truncated: bool = False) -> None:
        term = term.strip().strip(".,;")
        if len(term) < 2 or term.casefold() in seen:
            return
        seen.add(term.casefold())
        out.append(
            {"term": term, "source": source, "term_truncated": maybe_truncated}
        )

    match = _NOT_FOUND_RE.search(reason)
    if match:
        pieces = _TARGET_SPLIT_RE.split(match.group(1))
        for position, piece in enumerate(pieces):
            add(
                piece,
                "not-found-target",
                maybe_truncated=truncated and position == len(pieces) - 1,
            )
    for quoted in _ANSWER_SIDE_RE.finditer(reason):
        add(quoted.group(1), "conflicting-value:answer")
    for quoted in _EVIDENCE_SIDE_RE.finditer(reason):
        add(quoted.group(1), "conflicting-value:evidence")
    for quoted in _QUOTED_RE.finditer(reason):
        add(quoted.group(1), "quoted-value")
    return out


# ---------------------------------------------------------------------------
# reconstruction
# ---------------------------------------------------------------------------


def _key_string(doc: str, page: Any) -> str:
    return f"{doc}|{page if page is not None else '-'}"


def _note_blocks(notes_used: dict[str, Any]) -> list[str]:
    return [
        notes_used[key]
        for key in EVIDENCE_NOTE_KEYS
        if isinstance(notes_used.get(key), str) and notes_used[key]
    ]


def _context(answer: str, start: int, end: int, width: int = 400) -> dict[str, str]:
    return {
        "before": answer[max(0, start - width):start],
        "after": answer[end:end + width],
    }


def _citation_entries(
    claim: Any, evidence_keys: list[str], hit_keys: set[str]
) -> list[dict[str, Any]]:
    docs = sorted({key.split("|", 1)[0] for key in evidence_keys})
    entries: list[dict[str, Any]] = []
    for citation in claim.citations:
        resolved_doc = (
            verify._resolve_doc(citation.doc, docs) if citation.doc else NOTES_DOC
        )
        key = (
            _key_string(resolved_doc, citation.page) if resolved_doc else None
        )
        entries.append(
            {
                "raw": citation.raw,
                "doc": citation.doc,
                "page": citation.page,
                "kind": citation.kind,
                "resolved_doc": resolved_doc,
                "evidence_key": key,
                "held": bool(key and key in evidence_keys),
                "retrieved": bool(key and key in hit_keys),
            }
        )
    return entries


class EvidenceIndex:
    """The three searchable views of one case's held evidence, built once.

    ``norm`` is the literal-search index, ``loose`` reads hyphens as spaces,
    and ``numbers`` maps a figure's digits to the spellings the page prints.
    All three are keyed by evidence key and share :func:`display` offsets.
    """

    __slots__ = ("norm", "loose", "numbers", "amounts")

    def __init__(self, entries: Iterable[dict[str, Any]]) -> None:
        self.norm: dict[str, str] = {}
        self.loose: dict[str, str] = {}
        self.numbers: dict[str, dict[str, list[str]]] = {}
        self.amounts: dict[str, list[Any]] = {}
        for entry in entries:
            key, text = entry["key"], entry["text"]
            self.norm[key] = normalize(text)
            self.loose[key] = loose(text)
            self.numbers[key] = number_index(self.norm[key])
            self.amounts[key] = verify.amounts(text)


def _find_term(
    term: str, index: EvidenceIndex
) -> tuple[dict[str, set[str]], dict[str, list[str]]]:
    """``({key: {match kind}}, {key: [spelling the page prints]})``."""
    needle = normalize(term)
    loose_needle = loose(term)
    digits = separator_agnostic_key(term)
    kinds: dict[str, set[str]] = {}
    spellings: dict[str, list[str]] = {}

    def hit(key: str, kind: str, spelling: str) -> None:
        kinds.setdefault(key, set()).add(kind)
        bucket = spellings.setdefault(key, [])
        if spelling and spelling not in bucket:
            bucket.append(spelling)

    for key, text in index.norm.items():
        if needle and needle in text:
            hit(key, "literal", needle)
        elif loose_needle and loose_needle in index.loose[key]:
            at = index.loose[key].find(loose_needle)
            hit(key, "hyphen-variant", text[at:at + len(loose_needle)])
        if digits:
            for spelling in index.numbers[key].get(digits, ()):
                hit(key, "digits", spelling)
    return kinds, spellings


def _locate(
    kinds: dict[str, set[str]],
    spellings: dict[str, list[str]],
    cited_keys: set[str],
    exact_keys: set[str],
) -> dict[str, Any]:
    keys = sorted(kinds)
    strict = [k for k in keys if kinds[k] & STRICT_MATCH_KINDS]
    all_kinds = sorted({kind for found in kinds.values() for kind in found})
    return {
        "found_in_keys": keys,
        "found_in_cited_keys": sorted(k for k in keys if k in cited_keys),
        "found_in_exact_keys": sorted(k for k in keys if k in exact_keys),
        # A strict hit is the term's own spelling, or the same figure by value.
        # A 'digits'-only hit is the same digits punctuated differently, which
        # may or may not be the same magnitude — reported, never relied on.
        "found_in_strict_keys": strict,
        "found_in_exact_strict_keys": sorted(k for k in strict if k in exact_keys),
        "match_kinds": all_kinds,
        "matches_by_key": {
            key: {"kinds": sorted(kinds[key]), "spellings": spellings.get(key, [])}
            for key in keys
        },
        "present_in_held_evidence": bool(keys),
        "separator_convention_differs": bool(keys) and all_kinds == ["digits"],
        # a page-superset key holds text the prompt may never have carried, so
        # a hit there is an upper bound; a STRICT hit in an exact key is the
        # real thing.
        "presence_is_upper_bound": bool(keys) and not any(
            k in exact_keys for k in strict
        ),
    }


def _probe_term(
    term: str, index: EvidenceIndex, cited_keys: set[str], exact_keys: set[str]
) -> dict[str, Any]:
    kinds, spellings = _find_term(term, index)
    return {"term": term, **_locate(kinds, spellings, cited_keys, exact_keys)}


def _probe_amount(
    amount: Any, index: EvidenceIndex, cited_keys: set[str], exact_keys: set[str]
) -> dict[str, Any]:
    kinds, spellings = _find_term(amount.raw, index)
    for key, held in index.amounts.items():
        for candidate in held:
            agreement = _amounts_agree(amount, candidate)
            if agreement:
                kinds.setdefault(key, set()).add(agreement)
                bucket = spellings.setdefault(key, [])
                spelling = normalize(candidate.raw)
                if spelling and spelling not in bucket:
                    bucket.append(spelling)
                break
    return {
        "term": amount.raw,
        "value": amount.value,
        "mantissa": amount.bare,
        "currency": amount.currency,
        **_locate(kinds, spellings, cited_keys, exact_keys),
    }


EXCERPT_WINDOW = 240
EXCERPT_MAX_WINDOWS = 3
#: Every held key shows at least this much text, matched or not.  A row that
#: shows the reviewer no evidence TEXT cannot separate missing_citation from
#: missing_retrieval_evidence from verifier_false_positive, whatever else it
#: carries — 70% of entries and 12 whole rows were blank before this.
HEAD_EXCERPT_CHARS = 480
#: A page the claim actually cites is the thing under review; show all of it.
CITED_EXCERPT_CHARS = 6000
#: A claim with nothing probeable has no matched windows to centre on, so its
#: held pages get a larger head instead.
PROBE_FREE_EXCERPT_CHARS = 1200


def build_excerpt(
    text: str,
    terms: Iterable[str],
    *,
    window: int = EXCERPT_WINDOW,
    head: int = HEAD_EXCERPT_CHARS,
    max_windows: int = EXCERPT_MAX_WINDOWS,
) -> str:
    """Match-centred excerpt of one evidence page, for a human to read.

    Windows around the first occurrence of each matched spelling, joined by an
    ellipsis; the head of the page when nothing matched.  Never empty and
    never ``None``: a page shorter than ``head`` is returned whole.  The full
    text stays in the sidecar's ``evidence`` block, digest and all.
    """
    shown = display(text)
    if len(shown) <= head:
        return shown
    index = shown.lower()
    if len(index) != len(shown):                 # a length-changing lowercase
        return shown[:head] + " …"
    spans: list[tuple[int, int]] = []
    for term in terms:
        needle = normalize(term)
        if not needle:
            continue
        at = index.find(needle)
        if at < 0:
            continue
        spans.append((max(0, at - window), min(len(shown), at + len(needle) + window)))
        if len(spans) >= max_windows:
            break
    if not spans:
        return shown[:head] + " …"
    spans.sort()
    merged: list[list[int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    parts = []
    for start, end in merged:
        piece = shown[start:end]
        parts.append(
            ("… " if start else "") + piece + (" …" if end < len(shown) else "")
        )
    return " […] ".join(parts)


_HEADING_RE = re.compile(r"^\s*#{1,6}\s")
_BOLD_ONLY_RE = re.compile(r"^\s*\*\*[^*]+\*\*[\s:.]*$")


def _decision_inputs(
    claim: Any, terms: list[dict[str, Any]], cited: bool, fidelity: str
) -> dict[str, Any]:
    """The mechanical facts that separate the labels — reported, not decided.

    Only STRUCTURAL exclusions are taken: facts about the claim and its own
    brackets, which the reconstruction cannot get wrong.  A probe result never
    rules a label out on its own any more.

    WHY, in full, because the previous version printed the opposite claim in
    every row.  It excluded ``missing_citation`` whenever a probed term was
    absent, arguing that the reconstruction searches a page SUPERSET of what
    the prompt held, so "absent here" implies "absent at run time".  The
    superset argument is sound.  The probe was not: the corpus prints
    'USD 49,312 million' where the answer prints 'USD 49.312 million', and a
    literal-plus-parsed-value probe cannot see through the separator in either
    direction.  All five "absent" amounts in release-1 were false absences and
    three uncited claims had their correct label — ``missing_citation`` —
    struck off the list a reviewer was told to choose from.

    The probe is separator-agnostic and hyphen-tolerant now (see
    :func:`separator_agnostic_key` and :func:`loose`), but "this probe finds
    every spelling of every term" is not a property this file can prove.  So
    absence is REPORTED (``terms_absent_from_held_evidence``) and excludes
    nothing, and the reviewer reads the excerpts.

    The one surviving evidence-driven exclusion is
    ``missing_retrieval_evidence``, and only on a STRICT match — the term's own
    spelling, or the same figure by value or mantissa — inside evidence
    recovered exactly (a single-chunk page or a verbatim note block).  A
    ``digits``-only match is not strict: it says the page prints the same
    digits, not the same figure.  ``verifier_false_positive`` is never
    mechanically excludable and always stays live.
    """
    checkable = [t for t in terms if t.get("term")]
    present = [t for t in checkable if t["present_in_held_evidence"]]
    absent = [t for t in checkable if not t["present_in_held_evidence"]]
    upper_bound = [t for t in present if t["presence_is_upper_bound"]]
    exact_present = [t for t in present if t["found_in_exact_strict_keys"]]
    variant_only = [t for t in present if t["separator_convention_differs"]]
    heading = bool(
        _HEADING_RE.match(claim.text) or _BOLD_ONLY_RE.match(claim.text)
    )
    lead_in = claim.text.rstrip().endswith(":")

    excluded: dict[str, str] = {}
    # An INHERITED bracket is the block's, not the sentence's; whether that
    # counts as citing the claim is precisely one of the things Wave 1 is being
    # asked to decide, so it excludes nothing.
    if cited and not claim.inherited:
        excluded["missing_citation"] = (
            "the claim carries its own citation "
            f"({len(claim.citations)} bracket(s))"
        )
    if not checkable:
        # NOTHING is excluded here. The term probe extracts amounts and named
        # entities; it does not read years or board ids, so "no checkable
        # term" meant "nothing this probe knows how to look for" and struck
        # `missing_citation` off 5 of 71 rows whose claim asserts exactly a
        # year or a board code (Wave 0c finding F1). A probe's blind spot is
        # not evidence about the world: it is reported as a signal below and
        # rules out no label.
        pass
    elif not absent and len(exact_present) == len(checkable):
        excluded["missing_retrieval_evidence"] = (
            "every term is present, on its own spelling or by value, in "
            "evidence recovered exactly (single-chunk pages / verbatim notes)"
        )

    return {
        "cited": cited,
        "has_checkable_content": bool(checkable),
        "n_terms": len(checkable),
        "n_terms_present_in_held_evidence": len(present),
        "n_terms_absent_from_held_evidence": len(absent),
        "terms_absent_from_held_evidence": [t["term"] for t in absent],
        "all_terms_present_in_held_evidence": bool(checkable and not absent),
        "any_term_present_in_cited_evidence": any(
            t["found_in_cited_keys"] for t in checkable
        ),
        "n_terms_present_in_exact_evidence": len(exact_present),
        "n_terms_matched_only_by_separator_variant": len(variant_only),
        "terms_matched_only_by_separator_variant": [t["term"] for t in variant_only],
        "presence_is_upper_bound": bool(upper_bound),
        "looks_like_heading": heading,
        "is_lead_in_to_a_list": lead_in,
        "unit_kind": claim.unit_kind,
        "citations_inherited_from_block": bool(claim.inherited),
        "evidence_fidelity": fidelity,
        "labels_excluded_by_evidence": dict(sorted(excluded.items())),
        "labels_remaining": [label for label in LABELS if label not in excluded],
        "note": (
            "Exclusions here are STRUCTURAL (the claim's own bracket; nothing "
            "citable in the sentence), plus missing_retrieval_evidence on a "
            "strict match in exactly-recovered evidence. Probe results are "
            "otherwise REPORTED, never used to rule a label out: an absent "
            "term is only as good as the probe that looked for it. Presence "
            "is an upper bound wherever presence_is_upper_bound is true, and "
            "a term listed in terms_matched_only_by_separator_variant was "
            "found by its digits alone — read the excerpt and the spelling "
            "the page prints before treating it as the same figure."
        ),
    }


def reconstruct_case(
    row: dict[str, Any],
    page_texts: dict[tuple[str, Any], list[str]],
    *,
    release_sha256: str = "",
    release_path: str = "",
) -> dict[str, Any]:
    """One sidecar row: the evidence a single recorded case actually held."""
    case_id = row.get("id")
    if not isinstance(case_id, str) or not case_id:
        raise BackfillError("release row has no id")
    answer = row.get("answer") or ""
    claims_block = row.get("claims") if isinstance(row.get("claims"), dict) else {}
    recorded_keys = list(claims_block.get("evidence_keys") or [])
    notes_used = row.get("notes_used") if isinstance(row.get("notes_used"), dict) else {}
    hits = [h for h in (row.get("hits") or ()) if isinstance(h, dict)]

    incomplete: list[str] = []

    # The full claim text is recovered by re-running verify.extract_claims over
    # the recorded answer. That is only sound while extraction still sees the
    # same claims the release recorded, so check the count the release wrote
    # down rather than assuming the module never moves.
    extracted = verify.extract_claims(answer)
    # A RECONCILIATION, NOT A TOLERANCE. Extraction no longer mints
    # heading-form lead-ins (adjudication ruling 6) or the conversational
    # offer the hedge list now catches, so a release recorded before those
    # rules lists units the extractor will never return.
    # `verify.unminted_units` names them — from the same walk that produced
    # the claims, so the two cannot disagree — and each is recovered below
    # with its own text, marked `resolved_by_extraction`. A unit counts
    # towards the recorded total only when it really was a claim: either it
    # still triggers a kind (the lead-ins), or the release itself listed it as
    # a failure. Ordinary prose the extractor never minted is not excused, so
    # any OTHER drift in the count still fails this check.
    failure_texts = {(f.get("text") or "")
                     for f in (claims_block.get("failures") or ())
                     if isinstance(f, dict)}
    resolved_units = [c for c in verify.unminted_units(answer)
                      if c.kind or c.text[:160] in failure_texts]
    recorded_claims = claims_block.get("claims")
    if isinstance(recorded_claims, int) and \
            recorded_claims != len(extracted) + len(resolved_units):
        incomplete.append(
            f"verify.extract_claims now yields {len(extracted)} claims (plus "
            f"{len(resolved_units)} units extraction no longer mints) where the "
            f"release recorded {recorded_claims}; claim recovery is not faithful"
        )

    # --- rebuild the evidence exactly as build_evidence saw it -------------
    rebuilt_hits = []
    hit_meta: dict[str, dict[str, Any]] = {}
    for hit in hits:
        key_pair = (hit.get("doc"), hit.get("page"))
        chunks = page_texts.get(key_pair)
        key = _key_string(hit.get("doc"), hit.get("page"))
        if chunks is None:
            incomplete.append(f"no chunk text in the index for {key}")
            chunks = []
        text = "\n".join(c for c in chunks if c)
        rebuilt_hits.append(
            {"doc_id": hit.get("doc"), "page": hit.get("page"), "text": text}
        )
        meta = hit_meta.setdefault(
            key,
            {
                "key": key,
                "doc": hit.get("doc"),
                "page": hit.get("page"),
                "kind": "hit",
                "scores": [],
                "chunks_on_page": len(chunks),
                "fidelity": "exact" if len(chunks) == 1 else "page-superset",
            },
        )
        meta["scores"].append(hit.get("score"))
        if len(chunks) != 1:
            meta["fidelity"] = "page-superset" if chunks else "missing"

    blocks = _note_blocks(notes_used)
    evidence = verify.build_evidence(rebuilt_hits, blocks)
    rebuilt_keys = [_key_string(doc, page) for doc, page in evidence]
    keys_match = rebuilt_keys == recorded_keys
    if not keys_match:
        incomplete.append(
            "reconstructed evidence keys differ from claims.evidence_keys"
        )

    entries: list[dict[str, Any]] = []
    for (doc, page), text in evidence.items():
        key = _key_string(doc, page)
        meta = hit_meta.get(key)
        entry = {
            "key": key,
            "doc": doc,
            "page": page,
            "kind": meta["kind"] if meta else ("notes" if doc == NOTES_DOC else "note-derived"),
            "scores": meta["scores"] if meta else [],
            "chunks_on_page": meta["chunks_on_page"] if meta else None,
            "fidelity": meta["fidelity"] if meta else "exact",
            "chars": len(text),
            "text_sha256": sha256_text(text),
            "text": text,
        }
        entries.append(entry)

    search = EvidenceIndex(entries)
    hit_keys = set(hit_meta)
    exact_keys = {e["key"] for e in entries if e["fidelity"] == "exact"}

    # --- per-failed-claim reconstruction ----------------------------------
    by_prefix: dict[str, list[Any]] = defaultdict(list)
    for claim in extracted:
        by_prefix[claim.text[:160]].append(claim)
    # the dropped lead-ins join the lookup so their rows keep their text,
    # citations and evidence view; `resolved_by_extraction` on the row is what
    # tells a reader the unit is no longer a claim at all.
    resolved_texts = set()
    for unit in resolved_units:
        by_prefix[unit.text[:160]].append(unit)
        resolved_texts.add(unit.text[:160])

    claim_rows: list[dict[str, Any]] = []
    for index, failure in enumerate(claims_block.get("failures") or ()):
        if not isinstance(failure, dict):
            incomplete.append(f"failure {index + 1} is not an object")
            continue
        text = failure.get("text") or ""
        matches = by_prefix.get(text) or []
        claim_id = _claim_id(
            case_id,
            {
                "status": failure.get("status"),
                "kind": failure.get("kind"),
                "text": text,
                "reason": failure.get("reason"),
            },
        )
        if len(matches) != 1:
            incomplete.append(
                f"{claim_id}: recorded claim text matches {len(matches)} "
                "extracted claims; full text not recoverable"
            )
            claim_rows.append(
                {
                    "claim_id": claim_id,
                    "failure_index": index,
                    "reconstructed": False,
                    "reason_not_reconstructed": (
                        f"recorded text matched {len(matches)} extracted claims"
                    ),
                    "source_status": failure.get("status"),
                    "source_kind": failure.get("kind"),
                    "source_reason": failure.get("reason"),
                    "claim_text_recorded": text,
                }
            )
            continue

        claim = matches[0]
        start = answer.find(claim.text)
        end = start + len(claim.text) if start >= 0 else -1
        citations = _citation_entries(claim, rebuilt_keys, hit_keys)
        cited_keys = {c["evidence_key"] for c in citations if c["evidence_key"]}
        fidelity = (
            "exact"
            if all(e["fidelity"] == "exact" for e in entries)
            else "page-superset"
        )

        terms: list[dict[str, Any]] = []
        for amount in claim.amounts:
            probe = _probe_amount(amount, search, cited_keys, exact_keys)
            probe["term_kind"] = "amount"
            terms.append(probe)
        for group in claim.entities:
            variants = [v for v in group if v]
            kinds: dict[str, set[str]] = {}
            spellings: dict[str, list[str]] = {}
            for variant in variants:
                found_kinds, found_spellings = _find_term(variant, search)
                for key, found in found_kinds.items():
                    kinds.setdefault(key, set()).update(found)
                for key, found in found_spellings.items():
                    bucket = spellings.setdefault(key, [])
                    bucket.extend(s for s in found if s not in bucket)
            terms.append(
                {
                    "term": variants[0] if variants else "",
                    "term_kind": "entity",
                    "variants": variants,
                    **_locate(kinds, spellings, cited_keys, exact_keys),
                }
            )
        reason_probe = []
        for target in reason_targets(failure.get("reason") or ""):
            probe = _probe_term(target["term"], search, cited_keys, exact_keys)
            probe["source"] = target["source"]
            reason_probe.append(probe)

        # --- the evidence view a reviewer reads for THIS claim -------------
        # Both sides of every match: what the CLAIM says, and what the PAGE
        # prints.  Showing 'USD 49.312 million' matched by '49,312' is the
        # whole point — the reviewer, not the probe, decides whether those
        # are the same figure.
        matched_by_key: dict[str, list[str]] = defaultdict(list)
        spelling_by_key: dict[str, list[str]] = defaultdict(list)
        for probe in terms + reason_probe:
            for key, match in probe["matches_by_key"].items():
                if probe["term"] not in matched_by_key[key]:
                    matched_by_key[key].append(probe["term"])
                bucket = spelling_by_key[key]
                bucket.extend(s for s in match["spellings"] if s not in bucket)
        probe_free = not any(t.get("term") for t in terms)
        evidence_view = []
        for entry in entries:
            key = entry["key"]
            matched = matched_by_key.get(key, [])
            spelled = spelling_by_key.get(key, [])
            is_cited = key in cited_keys
            head = (
                CITED_EXCERPT_CHARS if is_cited
                else PROBE_FREE_EXCERPT_CHARS if probe_free
                else HEAD_EXCERPT_CHARS
            )
            shown = display(entry["text"])
            excerpt = build_excerpt(
                entry["text"],
                spelled + matched,
                head=head,
                max_windows=6 if is_cited else EXCERPT_MAX_WINDOWS,
            )
            evidence_view.append(
                {
                    "key": key,
                    "doc": entry["doc"],
                    "page": entry["page"],
                    "kind": entry["kind"],
                    "scores": entry["scores"],
                    "chunks_on_page": entry["chunks_on_page"],
                    "fidelity": entry["fidelity"],
                    "chars": entry["chars"],
                    "text_sha256": entry["text_sha256"],
                    "cited_by_claim": is_cited,
                    "matched_terms": matched,
                    "matched_spellings": spelled,
                    "excerpt": excerpt,
                    "excerpt_chars": len(excerpt),
                    "excerpt_is_full_text": excerpt == shown,
                }
            )
        evidence_view.sort(
            key=lambda e: (not e["cited_by_claim"], not e["matched_terms"], e["key"])
        )

        recomputed_kind = verify.claim_kind(claim.text)
        claim_rows.append(
            {
                "claim_id": claim_id,
                "failure_index": index,
                "reconstructed": True,
                "resolved_by_extraction": claim.text[:160] in resolved_texts,
                "source_status": failure.get("status"),
                "source_kind": failure.get("kind"),
                "source_reason": failure.get("reason"),
                "claim_text_recorded": text,
                "claim_text_full": claim.text,
                "claim_text_truncated": len(claim.text) > len(text),
                "claim_char_start": start,
                "claim_char_end": end,
                "claim_context": _context(answer, start, end) if start >= 0 else None,
                "claim_kind_recomputed": recomputed_kind,
                "claim_kind_agrees_with_record": recomputed_kind == failure.get("kind"),
                "cited": bool(claim.citations),
                "citations": citations,
                "cited_evidence_keys": sorted(cited_keys),
                "evidence_fidelity": fidelity,
                "evidence_view": evidence_view,
                "term_probe": terms,
                "reason_probe": reason_probe,
                "decision_inputs": _decision_inputs(
                    claim, terms, bool(claim.citations), fidelity
                ),
            }
        )

    reconstructed = sum(1 for c in claim_rows if c.get("reconstructed"))
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "release_path": release_path,
        "release_sha256": release_sha256,
        "question": row.get("question"),
        "answer": answer,
        "expect_docs": list((row.get("expect") or {}).get("docs") or []),
        "expect_pages": list((row.get("expect") or {}).get("pages") or []),
        "expect_notes": (row.get("expect") or {}).get("notes"),
        "hits": [
            {"doc": h.get("doc"), "page": h.get("page"), "score": h.get("score")}
            for h in hits
        ],
        "notes_used": dict(notes_used),
        "notes_in_evidence": [
            k for k in EVIDENCE_NOTE_KEYS if isinstance(notes_used.get(k), str) and notes_used[k]
        ],
        "notes_recorded_but_not_evidence": sorted(
            k for k in notes_used
            if k not in EVIDENCE_NOTE_KEYS and notes_used.get(k)
        ),
        "evidence_keys_recorded": recorded_keys,
        "evidence_keys_reconstructed": rebuilt_keys,
        "evidence_keys_match": keys_match,
        "evidence": entries,
        "claims": claim_rows,
        "reconstruction": {
            "complete": keys_match and not incomplete and reconstructed == len(claim_rows),
            "failures_total": len(claim_rows),
            "failures_reconstructed": reconstructed,
            "exact_evidence": all(e["fidelity"] == "exact" for e in entries),
            "problems": sorted(dict.fromkeys(incomplete)),
        },
    }


def provenance() -> dict[str, Any]:
    """Which revision of the verifier's parsers produced this reconstruction.

    ``extract_claims`` / ``build_evidence`` / ``parse_citations`` are what makes
    the reconstruction faithful; ``claim_kind`` is what the wave is actively
    changing.  Recording their AST digests makes a stale sidecar visible
    instead of silently wrong.
    """
    import ast

    source = (ROOT / "src" / "gcf_qna" / "rag" / "verify.py")
    digests: dict[str, str] = {}
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        tree = None
    if tree is not None:
        wanted = {
            "extract_claims", "build_evidence", "parse_citations",
            "claim_kind", "amounts", "entities",
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                digests[node.name] = hashlib.sha1(
                    ast.dump(node).encode("utf-8")
                ).hexdigest()[:12]
    return {
        "verify_sha256": sha256_file(source) if source.exists() else None,
        "verify_ast": dict(sorted(digests.items())),
    }


def build_sidecar(
    release_path: Path, chunks_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    release_rows = _read_jsonl(release_path)
    if not release_rows:
        raise BackfillError(f"{release_path}: no rows")
    release_sha = sha256_file(release_path)
    page_texts = load_page_texts(chunks_path, wanted_pages(release_rows))
    origin = provenance()

    rows = []
    for row in release_rows:
        case = reconstruct_case(
            row,
            page_texts,
            release_sha256=release_sha,
            release_path=str(release_path),
        )
        case["provenance"] = origin
        rows.append(case)

    claims = [c for row in rows for c in row["claims"]]
    done = [c for c in claims if c.get("reconstructed")]
    fidelity = Counter(c.get("evidence_fidelity", "unknown") for c in done)
    problems = Counter(p for row in rows for p in row["reconstruction"]["problems"])
    inputs = [c["decision_inputs"] for c in done]
    no_citation = [c for c in done if not c["decision_inputs"]["cited"]]
    probes = [p for c in done for p in c["reason_probe"]]

    def _side(source: str) -> tuple[int, int]:
        subset = [p for p in probes if p["source"] == source]
        return sum(1 for p in subset if p["present_in_held_evidence"]), len(subset)

    stats = {
        "schema_version": SCHEMA_VERSION,
        "release": str(release_path),
        "release_sha256": release_sha,
        "chunks": str(chunks_path),
        "cases": len(rows),
        "cases_with_matching_evidence_keys": sum(
            1 for row in rows if row["evidence_keys_match"]
        ),
        "failed_claims": len(claims),
        "claims_reconstructed": len(done),
        "claims_not_reconstructed": sum(1 for c in claims if not c.get("reconstructed")),
        "claims_in_fully_exact_cases": fidelity.get("exact", 0),
        "claims_in_page_superset_cases": fidelity.get("page-superset", 0),
        "evidence_keys": sum(len(row["evidence"]) for row in rows),
        "evidence_keys_exact": sum(
            1 for row in rows for e in row["evidence"] if e["fidelity"] == "exact"
        ),
        "evidence_chars": sum(
            e["chars"] for row in rows for e in row["evidence"]
        ),
        "claim_kind_disagreements": sum(
            1 for c in done if not c["claim_kind_agrees_with_record"]
        ),
        "claims_uncited": len(no_citation),
        "claims_without_checkable_terms": sum(
            1 for i in inputs if not i["has_checkable_content"]
        ),
        "claims_with_a_term_absent_from_held_evidence": sum(
            1 for i in inputs if i["n_terms_absent_from_held_evidence"]
        ),
        "claims_with_a_separator_variant_match": sum(
            1 for i in inputs if i["n_terms_matched_only_by_separator_variant"]
        ),
        # N14: a row that shows no evidence TEXT cannot be labelled from the
        # row alone, whatever else it carries. This must stay 0.
        "claims_with_no_visible_evidence_text": sum(
            1 for c in done
            if not any(e["excerpt"] for e in c["evidence_view"])
            and c["evidence_view"]
        ),
        "evidence_view_entries": sum(len(c["evidence_view"]) for c in done),
        "evidence_view_entries_without_text": sum(
            1 for c in done for e in c["evidence_view"] if not e["excerpt"]
        ),
        "evidence_view_chars_shown": sum(
            e["excerpt_chars"] for c in done for e in c["evidence_view"]
        ),
        "evidence_view_chars_total": sum(
            e["chars"] for c in done for e in c["evidence_view"]
        ),
        "claims_with_every_term_in_exact_evidence": sum(
            1 for i in inputs
            if i["has_checkable_content"]
            and i["n_terms_present_in_exact_evidence"] == i["n_terms"]
        ),
        "claims_with_no_label_excluded": sum(
            1 for i in inputs if not i["labels_excluded_by_evidence"]
        ),
        "uncited_claims_by_exclusion": dict(
            sorted(
                Counter(
                    "+".join(sorted(c["decision_inputs"]["labels_excluded_by_evidence"]))
                    or "(none)"
                    for c in no_citation
                ).items()
            )
        ),
        # fidelity cross-check: every figure the recorded reason says it read
        # OUT OF the held evidence must be findable in the reconstruction.
        "reason_evidence_side_confirmed": "%d/%d" % _side("conflicting-value:evidence"),
        "reason_answer_side_confirmed": "%d/%d" % _side("conflicting-value:answer"),
        "provenance": provenance(),
        "problems": dict(sorted(problems.items())),
    }
    return rows, stats


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="reconstruct and report, write nothing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.dry_run:
            assert_writable_output(
                args.output,
                inputs=(args.release, args.chunks),
                force=args.force,
            )
        rows, stats = build_sidecar(args.release, args.chunks)
        if not args.dry_run:
            assert_writable_output(
                args.output,
                inputs=(args.release, args.chunks),
                force=args.force,
            )
            write_jsonl(args.output, rows)
            stats["output"] = str(args.output)
        else:
            stats["output"] = None
        print(json.dumps(stats, indent=2, sort_keys=True))
        return 0
    except BackfillError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
