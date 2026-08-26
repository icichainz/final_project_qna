#!/usr/bin/env python3
"""Refresh the REGISTRY-DERIVED note text of the frozen release-1 evidence
sidecar, and nothing else.

    venv/bin/python scripts/refresh_release_evidence_notes.py            # report only
    venv/bin/python scripts/refresh_release_evidence_notes.py --apply --anchor

WHY A FROZEN FILE IS REGENERATED AT ALL
---------------------------------------
``data/eval/release_release-1-evidence.jsonl`` is the scorer's baseline: it
carries, per recorded case, the evidence text that turn held, and
``scripts/score_verifier.py`` rebuilds ``{(doc, page): text}`` from it verbatim
and re-runs the deterministic verifier over it.  Two halves of that text have
completely different provenance:

  * HIT TEXT — the corpus chunks retrieval returned.  This is a RECORDING.  It
    can never be regenerated from anything (the run is not reproducible), and
    this script never touches a byte of it.
  * NOTE TEXT — the 'Registry — …' block the answer path COMPUTED from the
    corpus fact registry at run time.  This is not a recording of the world,
    it is a rendering of a store, and the store is versioned.

The 117 ratified corrections (`data/registry_corrections.json`, owner-ratified
2026-08-26) moved the store out from under the recording.  The verifier reads
the LIVE registry (``verify._field_conflict`` and the deference checks call
``registry.facts``/``registry.load``), so a replay now judges an answer that
states a now-CORRECT figure — FP173's USD 279 million, FP254's 258 million,
FP153's 26,654 — against a note block that still prints the pre-correction
one, and returns CONTRADICTED against the baseline's own stale text.  That is
the instrument disagreeing with itself, not a finding about the answer.

So the note halves are re-rendered from the corrected store by the SAME
emitter the answer path uses (``registry.registry_note``), the hit halves are
left byte-identical, and the whole transform is scripted, deterministic and
diffed.  This is the registry-rebuild precedent applied to the scorer
baseline: loud, ratified, never silent.  Zero API calls.

WHAT IS REFRESHED AND WHAT IS NOT
---------------------------------
REFRESHED   ``notes_used["registry"]`` and every evidence entry text that
            ``verify.build_evidence`` derives from it — the ``__notes__|-``
            block, the per-document ``note-derived`` entries, and the note
            lines build_evidence APPENDS to a hit page it also points at.
            ``evidence_keys_reconstructed`` / ``evidence_keys_match`` /
            ``notes_in_evidence`` follow, recomputed rather than asserted.

FROZEN      ``answer``, ``question``, ``hits``, ``claims`` (the per-claim
            adjudication view a human ruled on — rewriting it would rewrite
            the record the adjudication was made against), ``expect_*``,
            ``provenance``, ``release_path``/``release_sha256``,
            ``schema_version``, ``evidence_keys_recorded`` (the key list the
            RELEASE recorded; it is release-1's own record, not a rendering),
            and the chunk text of every ``hit`` entry.

NOT REFRESHED, DELIBERATELY  ``notes_used["year"]`` / ``["board"]``.  These
            come from ``chainlit_app._year_assist`` / ``_board_range_note``,
            whose output depends on the TURN'S HITS (which excerpts were
            dated, which were sorted first), not on the registry alone.
            Re-rendering them would mean replaying retrieval, which is not a
            deterministic transform of the recording — the very property this
            script exists to have.  Their staleness is measured instead and
            printed in the report (`year_note_rows`), so it is named rather
            than assumed away.

WHAT THIS DOES TO THE SCORER, AND WHY IT IS TRANSITIONAL
--------------------------------------------------------
The corrections were adjudicated against an INDEPENDENT pymupdf extraction,
and the qwen markdown under ``data/index/`` still prints the old digits — p.7
of FP274's package still reads '40,511,264' where the store now reads
40,751,264.  So refreshing the notes makes the baseline agree with the STORE
and, on those documents, disagree with the SERVED TEXT the recorded answers
were written from.  Twenty-one Wave-1 adjudications move as a result; they are
superseded in bulk by owner ruling 12 (2026-08-26) and named row by row in
``tests/test_verify.py``.

That divergence is TRANSITIONAL, not the new steady state: the corpus cure —
re-extracting every flagged page and refreshing the index — is queued for the
next LM Studio window.  When it lands, the served text and the store agree
again and those pins are expected to move back, loudly, in one traced step.
Re-running this script after the cure is the first half of that step; it is
idempotent, so a run that changes nothing prints exactly that.

HOW THE HIT TEXT IS PROVEN UNTOUCHED
------------------------------------
``build_evidence`` merges: a note line that prints '(p.5, A.8)' is appended to
the SAME key a hit for page 5 occupies, so 23 of this file's 640 hit entries
hold 'chunk text + \\n + note line' in one string.  A refresh that only
regexed the note text would risk cutting into the chunk.

Instead the file is DECOMPOSED and the decomposition is proven before anything
is written:

  1. replay build_evidence's own note routing over the RECORDED note blocks to
     learn, per key, which lines it appended and in what order;
  2. strip exactly those lines back off each recorded entry, yielding the pure
     hit text per key;
  3. GATE — re-run ``verify.build_evidence(pure_hits, recorded_blocks)`` and
     require the result to reproduce the recorded evidence dict EXACTLY: same
     keys, same order, same bytes.  If it does not, the decomposition is not
     faithful and the script writes nothing.

Only then is ``build_evidence(pure_hits, refreshed_blocks)`` taken as the new
evidence.  The pure hit text is carried through untouched by construction, and
the report re-states it as a sha256 per hit key, before and after.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gcf_qna.rag import registry, verify  # noqa: E402

EVIDENCE = ROOT / "data" / "eval" / "release_release-1-evidence.jsonl"
CHECKSUMS = ROOT / "data" / "eval" / "CHECKSUMS.sha256"

#: the blocks ``eval_answers`` passes to ``build_evidence``, in its order.
EVIDENCE_NOTE_KEYS = ("registry", "year", "matrix")
#: the one block this script re-renders.
REFRESHED_NOTE_KEY = "registry"

#: fields that must come out of the transform byte-identical.
FROZEN_FIELDS = ("answer", "case_id", "claims", "expect_docs", "expect_notes",
                 "expect_pages", "hits", "provenance", "question",
                 "reconstruction", "release_path", "release_sha256",
                 "schema_version", "evidence_keys_recorded")


class RefreshError(RuntimeError):
    """Raised when the transform cannot be proven faithful."""


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RefreshError(f"{path}:{n}: {exc}") from None
    return rows


def serialize(rows: list[dict[str, Any]]) -> str:
    """The sidecar's own serialization — ``backfill_release_evidence`` writes
    exactly this, and the committed file round-trips through it byte for byte,
    so an untouched row is untouched in the FILE and not merely in memory."""
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                   for r in rows)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_atomic(path: Path, content: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", delete=False) as fh:
        fh.write(content)
        tmp = Path(fh.name)
    tmp.chmod(0o644)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# decomposition
# ---------------------------------------------------------------------------


def key_string(doc: Any, page: Any) -> str:
    return f"{doc}|{page if page else '-'}"


def note_routing(blocks: list[str]) -> list[tuple[tuple[Any, Any], str]]:
    """``[(evidence key, line)]`` in the order ``build_evidence`` adds them.

    A REPLICA of the note half of ``verify.build_evidence``, and deliberately
    written against its own module constants (``_MATRIX_DOC_RE``,
    ``_REG_DOC_RE``, ``_DOC_RE``, ``_MATRIX_PAGE_RE``) rather than re-derived,
    so a routing change there is a routing change here.  The gate in
    ``decompose`` is what makes the replica safe: if it ever stops agreeing
    with the real function, the reproduction check fails and nothing is
    written.
    """
    out: list[tuple[tuple[Any, Any], str]] = []
    for block in blocks:
        if not block:
            continue
        out.append((verify.NOTES_KEY, str(block)))
        current: Optional[str] = None
        for line in str(block).splitlines():
            matrix = verify._MATRIX_DOC_RE.match(line.strip())
            if matrix:
                current = matrix.group("doc")
                out.append(((current, None), line.strip()))
                continue
            reg = verify._REG_DOC_RE.search(line)
            if reg:
                out.append(((reg.group(1), None), line.strip()))
                continue
            docs = re.findall(verify._DOC_RE, line)
            owner = docs[0] if docs else current
            if not owner:
                continue
            page = verify._MATRIX_PAGE_RE.search(line)
            out.append(((owner, int(page.group(1)) if page else None), line.strip()))
    return out


def blocks_of(notes_used: dict[str, Any]) -> list[str]:
    return [notes_used[k] for k in EVIDENCE_NOTE_KEYS
            if isinstance(notes_used.get(k), str) and notes_used[k]]


def recorded_evidence(row: dict[str, Any]) -> dict[tuple[Any, Any], str]:
    return {(e["doc"], e["page"] if e["page"] else None): e["text"]
            for e in row.get("evidence") or ()}


def decompose(row: dict[str, Any]) -> dict[tuple[Any, Any], str]:
    """The pure hit text per key — recorded text minus the note lines
    build_evidence appended to it.  Raises unless the decomposition
    reproduces the recorded evidence exactly."""
    recorded = recorded_evidence(row)
    blocks = blocks_of(row.get("notes_used") or {})
    pure = dict(recorded)
    # reverse order: the last line appended is the last one to come off
    for key, line in reversed(note_routing(blocks)):
        text = (line or "").strip()
        if not text or key not in pure:
            continue
        current = pure[key]
        if current == text:
            del pure[key]
        elif current.endswith("\n" + text):
            pure[key] = current[:-(len(text) + 1)]
        # otherwise build_evidence deduplicated it ('text not in prev') and
        # appended nothing, so there is nothing to take off

    rebuilt = verify.build_evidence(
        [{"doc_id": doc, "page": page, "text": text}
         for (doc, page), text in pure.items()], blocks)
    if list(rebuilt.items()) != list(recorded.items()):
        raise RefreshError(
            f"{row.get('case_id')}: the note decomposition does not reproduce "
            f"the recorded evidence; refusing to refresh")
    return pure


# ---------------------------------------------------------------------------
# the refresh
# ---------------------------------------------------------------------------


def note_line_shape(line: str) -> str:
    """A line's EMITTER identity, for the diff report: two renderings of the
    same registry line are 'changed', not 'one removed and one added'."""
    for pattern, name in (
            (r"^Registry — FP(\d+): NOT FOUND", "fp-absent"),
            (r"^Registry — FP(\d+): ", "fp"),
            (r"^Registry — CONFLICT in this document \(([^)]+)\): (\d+) further",
             "conflict-truncated"),
            (r"^Registry — CONFLICT in this document \(([^)]+)\): (\w+) ", "conflict"),
            (r"^Registry — (GCF/B[^ ]+) resolves to: ", "board-code"),
            (r"^Registry — (\d+) funding-proposal documents from (20\d\d)", "year-listing"),
            (r"^Registry — the identifiers above resolve", "multi-document"),
    ):
        found = re.match(pattern, line)
        if found:
            return f"{name}({', '.join(found.groups())})"
    return f"other({line[:60]})"


def refresh_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """One refreshed row plus its diff record."""
    case_id = row.get("case_id")
    pure = decompose(row)
    recorded = recorded_evidence(row)
    hit_keys = {(e["doc"], e["page"] if e["page"] else None)
                for e in row["evidence"] if e["kind"] == "hit"}
    meta = {e["key"]: e for e in row["evidence"]}

    notes_used = dict(row.get("notes_used") or {})
    before = notes_used.get(REFRESHED_NOTE_KEY) or ""
    after = registry.registry_note(row.get("question") or "") or ""
    if after:
        notes_used[REFRESHED_NOTE_KEY] = after
    else:
        notes_used.pop(REFRESHED_NOTE_KEY, None)

    new_evidence = verify.build_evidence(
        [{"doc_id": doc, "page": page, "text": text}
         for (doc, page), text in pure.items()], blocks_of(notes_used))

    entries: list[dict[str, Any]] = []
    for (doc, page), text in new_evidence.items():
        key = key_string(doc, page)
        old = meta.get(key)
        is_hit = (doc, page) in hit_keys
        entries.append({
            "key": key, "doc": doc, "page": page,
            "kind": (old["kind"] if old else
                     ("notes" if doc == verify.NOTES_DOC else "note-derived")),
            "scores": old["scores"] if old else [],
            "chunks_on_page": old["chunks_on_page"] if old else None,
            "fidelity": old["fidelity"] if old else "exact",
            "chars": len(text),
            "text_sha256": sha256_text(text),
            "text": text,
        })
        if is_hit and old and old["kind"] != "hit":       # cannot happen; loud if it does
            raise RefreshError(f"{case_id}: hit entry {key} changed kind")

    new_row = dict(row)
    new_row["notes_used"] = notes_used
    new_row["evidence"] = entries
    new_row["evidence_keys_reconstructed"] = [e["key"] for e in entries]
    new_row["evidence_keys_match"] = (
        new_row["evidence_keys_reconstructed"] == list(row.get("evidence_keys_recorded") or ()))
    new_row["notes_in_evidence"] = [
        k for k in EVIDENCE_NOTE_KEYS
        if isinstance(notes_used.get(k), str) and notes_used[k]]
    new_row["notes_recorded_but_not_evidence"] = [
        k for k in notes_used
        if k not in EVIDENCE_NOTE_KEYS and notes_used.get(k)]

    # --- the per-row proof, and the diff -----------------------------------
    for key in hit_keys:
        head = pure.get(key, "")
        got = new_evidence.get(key)
        if got is None or not got.startswith(head):
            raise RefreshError(
                f"{case_id}: hit text at {key_string(*key)} did not survive the "
                f"refresh unchanged")

    old_lines = [ln for ln in before.splitlines() if ln.strip()]
    new_lines = [ln for ln in after.splitlines() if ln.strip()]
    old_by_shape = {note_line_shape(ln): ln for ln in old_lines}
    new_by_shape = {note_line_shape(ln): ln for ln in new_lines}
    changed = {s: {"before": old_by_shape[s], "after": new_by_shape[s]}
               for s in old_by_shape if s in new_by_shape
               and old_by_shape[s] != new_by_shape[s]}
    diff = {
        "case_id": case_id,
        "question": row.get("question"),
        "registry_note_sha256_before": sha256_text(before) if before else None,
        "registry_note_sha256_after": sha256_text(after) if after else None,
        "note_lines_removed": [old_by_shape[s] for s in old_by_shape
                               if s not in new_by_shape],
        "note_lines_added": [new_by_shape[s] for s in new_by_shape
                             if s not in old_by_shape],
        "note_lines_changed": changed,
        "evidence_keys_added": [key_string(*k) for k in new_evidence if k not in recorded],
        "evidence_keys_removed": [key_string(*k) for k in recorded if k not in new_evidence],
        "hit_entries": len(hit_keys),
        "hit_text_sha256_identical": True,          # proven above, entry by entry
        "hit_entries_whose_note_tail_moved": sum(
            1 for k in hit_keys if new_evidence.get(k) != recorded.get(k)),
        "note_entries_changed": sum(
            1 for k in new_evidence
            if k not in hit_keys and new_evidence[k] != recorded.get(k)),
        "year_note_present": bool((row.get("notes_used") or {}).get("year")),
    }
    diff["changed"] = bool(
        diff["note_lines_removed"] or diff["note_lines_added"]
        or changed or diff["evidence_keys_added"] or diff["evidence_keys_removed"]
        or diff["hit_entries_whose_note_tail_moved"] or diff["note_entries_changed"])
    for field in FROZEN_FIELDS:
        if json.dumps(new_row.get(field), sort_keys=True) != \
                json.dumps(row.get(field), sort_keys=True):
            raise RefreshError(f"{case_id}: frozen field {field!r} was modified")
    return new_row, diff


def refresh(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out, diffs = [], []
    for row in rows:
        new_row, diff = refresh_row(row)
        out.append(new_row)
        diffs.append(diff)
    changed = [d for d in diffs if d["changed"]]
    report = {
        "cases": len(rows),
        "cases_refreshed": len(changed),
        "cases_unchanged": len(rows) - len(changed),
        "registry_notes_recorded": sum(
            1 for r in rows if (r.get("notes_used") or {}).get("registry")),
        "registry_notes_after": sum(
            1 for r in out if (r.get("notes_used") or {}).get("registry")),
        "note_lines_removed": Counter(
            note_line_shape(l) for d in diffs for l in d["note_lines_removed"]),
        "note_lines_added": Counter(
            note_line_shape(l) for d in diffs for l in d["note_lines_added"]),
        "note_lines_changed": sum(len(d["note_lines_changed"]) for d in diffs),
        "evidence_keys_added": Counter(
            k for d in diffs for k in d["evidence_keys_added"]),
        "evidence_keys_removed": Counter(
            k for d in diffs for k in d["evidence_keys_removed"]),
        "hit_entries_total": sum(d["hit_entries"] for d in diffs),
        "hit_entries_with_unchanged_text_sha256": sum(d["hit_entries"] for d in diffs),
        "hit_entries_whose_appended_note_tail_moved": sum(
            d["hit_entries_whose_note_tail_moved"] for d in diffs),
        "note_entries_changed": sum(d["note_entries_changed"] for d in diffs),
        "rows_whose_keys_no_longer_match_the_release": sum(
            1 for r in out if not r["evidence_keys_match"]),
        "year_note_rows_not_refreshed": sum(1 for d in diffs if d["year_note_present"]),
        "cases": len(rows),
        "per_case": diffs,
    }
    return out, report


# ---------------------------------------------------------------------------
# the anchor
# ---------------------------------------------------------------------------


def reanchor(path: Path, digest: str, checksums: Path = CHECKSUMS) -> bool:
    """Rewrite the ONE line of the checksum anchor that covers ``path``."""
    rel = str(path.resolve().relative_to(ROOT))
    lines = checksums.read_text(encoding="utf-8").splitlines(keepends=True)
    hit = [i for i, line in enumerate(lines)
           if not line.startswith("#") and line.rstrip("\n").endswith("  " + rel)]
    if len(hit) != 1:
        raise RefreshError(f"{checksums}: expected exactly one line for {rel}, "
                           f"found {len(hit)}")
    if lines[hit[0]] == f"{digest}  {rel}\n":
        return False
    lines[hit[0]] = f"{digest}  {rel}\n"
    write_atomic(checksums, "".join(lines))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence", type=Path, default=EVIDENCE)
    parser.add_argument("--output", type=Path, default=None,
                        help="write here instead of over the input")
    parser.add_argument("--report", type=Path, default=None,
                        help="write the full per-case diff report as JSON")
    parser.add_argument("--apply", action="store_true",
                        help="write the refreshed sidecar (default: report only)")
    parser.add_argument("--anchor", action="store_true",
                        help="rewrite this file's line in data/eval/CHECKSUMS.sha256")
    args = parser.parse_args(argv)

    source = args.evidence
    original = source.read_text(encoding="utf-8")
    rows = read_jsonl(source)
    if serialize(rows) != original:
        raise RefreshError(
            f"{source} does not round-trip through its own serialization; a "
            f"refresh could not be proven byte-minimal")
    refreshed, report = refresh(rows)
    content = serialize(refreshed)

    print(f"cases                                  {report['cases']}")
    print(f"cases refreshed                        {report['cases_refreshed']}")
    print(f"registry note lines changed            {report['note_lines_changed']}")
    print(f"registry note lines removed            "
          f"{sum(report['note_lines_removed'].values())} "
          f"{dict(report['note_lines_removed'])}")
    print(f"registry note lines added              "
          f"{sum(report['note_lines_added'].values())} "
          f"{dict(report['note_lines_added'])}")
    print(f"evidence keys added                    "
          f"{sum(report['evidence_keys_added'].values())}")
    print(f"evidence keys removed                  "
          f"{sum(report['evidence_keys_removed'].values())}")
    print(f"hit entries                            {report['hit_entries_total']}")
    print(f"hit entries, chunk text unchanged      "
          f"{report['hit_entries_with_unchanged_text_sha256']}  (proven per entry)")
    print(f"hit entries whose note tail moved      "
          f"{report['hit_entries_whose_appended_note_tail_moved']}")
    print(f"note entries changed                   {report['note_entries_changed']}")
    print(f"rows no longer key-matching release-1  "
          f"{report['rows_whose_keys_no_longer_match_the_release']}")
    print(f"year-note rows left stale on purpose   "
          f"{report['year_note_rows_not_refreshed']}")
    print(f"sha256 before                          {sha256_text(original)}")
    print(f"sha256 after                           {sha256_text(content)}")

    if args.report:
        args.report.write_text(json.dumps(
            report, indent=1, ensure_ascii=False,
            default=lambda o: dict(o) if isinstance(o, Counter) else str(o)),
            encoding="utf-8")
        print(f"report written to {args.report}")

    if not args.apply:
        print("(report only; pass --apply to write)")
        return 0

    target = args.output or source
    write_atomic(target, content)
    print(f"wrote {target}")
    if args.anchor:
        moved = reanchor(target, sha256_text(content))
        print(f"anchor {'rewritten' if moved else 'already current'} in {CHECKSUMS}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RefreshError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
