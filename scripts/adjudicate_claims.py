#!/usr/bin/env python3
"""Export, validate, import, and summarize claim-failure adjudications.

An exported row must be sufficient on its own: a reviewer picks a label by
reading it, never by re-running the model or the retriever.  That means the
row carries not only the claim and the verifier's recorded verdict but the
EVIDENCE THE LABELS ARE DEFINED OVER — the retrieved hits with their page
numbers and the text that was on those pages, the registry/year note blocks,
the evidence keys the turn held, the claim's citations and where they resolve,
and the claim's surrounding context in the answer.  Without that,
``missing_citation`` / ``missing_retrieval_evidence`` / ``verifier_false_positive``
are indistinguishable for any claim whose whole signal is "no citation on a
factual claim".

The release record keeps the evidence KEYS but not the hit TEXT, so the text
half is reconstructed by ``scripts/backfill_release_evidence.py`` into a
sidecar and joined here by ``claim_id``.  Pass ``--evidence <sidecar>`` to
``export`` (and to every later ``validate``/``import``/``summary`` of that
inventory, so the immutable fields line up).

Tamper detection is over three digests: ``release_sha256`` fingerprints the
release the row came from, ``evidence_source_sha256`` the sidecar, and
``evidence_digest`` the row's own evidence payload — recomputed from the row
AND compared against the canonical value, so neither editing the evidence nor
editing the evidence and its digest together survives ``validate``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Optional


SCHEMA_VERSION = 2
LABELS = (
    "verifier_false_positive",
    "genuine_answer_error",
    "missing_retrieval_evidence",
    "missing_citation",
    "registry_conflict",
    "ambiguous_unscorable",
)

#: Copied verbatim from the release; identity of the claim under review.
RELEASE_FIELDS = (
    "schema_version",
    "claim_id",
    "case_id",
    "release_sha256",
    "question",
    "answer",
    "source_status",
    "source_kind",
    "claim_text",
    "source_reason",
    "evidence_keys",
    "hits",
    "notes_used",
    "expect_docs",
    "expect_pages",
)
#: Reconstructed by scripts/backfill_release_evidence.py, joined on claim_id.
EVIDENCE_FIELDS = (
    "evidence_status",
    "evidence_source_sha256",
    "claim_text_full",
    "claim_text_truncated",
    "claim_char_start",
    "claim_char_end",
    "context_before",
    "context_after",
    "claim_kind_recomputed",
    "claim_kind_agrees_with_record",
    "cited",
    "citations",
    "cited_evidence_keys",
    "evidence_fidelity",
    "evidence",
    "term_probe",
    "reason_probe",
    "decision_inputs",
)
#: sha256 over EVIDENCE_FIELDS; an immutable field in its own right.
DIGEST_FIELD = "evidence_digest"
ANNOTATION_FIELDS = ("label", "reviewer", "notes")

IMMUTABLE_FIELDS = RELEASE_FIELDS + EVIDENCE_FIELDS + (DIGEST_FIELD,)
RECORD_FIELDS = IMMUTABLE_FIELDS + ANNOTATION_FIELDS

EVIDENCE_ABSENT = "absent"
EVIDENCE_RECONSTRUCTED = "reconstructed"

#: zero-width characters that make an "is this non-empty?" check theatre
_INVISIBLE = "​‌‍⁠﻿"


class AdjudicationError(ValueError):
    """Raised when a release or adjudication inventory is invalid."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AdjudicationError(f"cannot read {path}: {exc}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AdjudicationError(
                f"{path}:{line_number}: invalid JSON: {exc.msg}"
            ) from exc
        if not isinstance(row, dict):
            raise AdjudicationError(
                f"{path}:{line_number}: expected a JSON object"
            )
        rows.append(row)
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise AdjudicationError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _visible(value: Any) -> str:
    """``value`` with zero-width characters removed, for emptiness checks."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value)
    return "".join(c for c in text if c not in _INVISIBLE).strip()


def _require_text(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdjudicationError(f"{location}: expected non-empty text")
    return value


def _claim_id(case_id: str, failure: dict[str, str]) -> str:
    identity = {
        "case_id": case_id,
        "source_status": failure["status"],
        "source_kind": failure["kind"],
        "claim_text": failure["text"],
        "source_reason": failure["reason"],
    }
    return f"claim-{hashlib.sha256(_canonical(identity)).hexdigest()[:24]}"


def evidence_digest(row: dict[str, Any]) -> str:
    """sha256 over the row's evidence payload, keyed to the claim it describes.

    ``claim_id`` and ``release_sha256`` are inside the digest so a labelled row
    cannot be repointed at a different claim by swapping the identity fields
    while keeping a valid-looking evidence block, or vice versa.
    """
    payload = {
        "claim_id": row.get("claim_id"),
        "release_sha256": row.get("release_sha256"),
        **{field: row.get(field) for field in EVIDENCE_FIELDS},
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()


# ---------------------------------------------------------------------------
# evidence sidecar
# ---------------------------------------------------------------------------


def read_evidence(
    path: Path, release_sha256: Optional[str] = None
) -> tuple[dict[str, dict[str, Any]], str]:
    """``{claim_id: reconstructed evidence}`` from a backfill sidecar.

    The sidecar stamps the release it was reconstructed from; a sidecar built
    against a different release describes evidence that never belonged to
    these claims, so it is refused rather than joined.
    """
    sha = _sha256_file(path)
    by_claim: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(_read_jsonl(path), start=1):
        stamped = row.get("release_sha256")
        if release_sha256 is not None and stamped != release_sha256:
            raise AdjudicationError(
                f"{path}:{row_number}: sidecar was built from release "
                f"{stamped!r}, not {release_sha256!r}; rebuild it with "
                "scripts/backfill_release_evidence.py"
            )
        claims = row.get("claims")
        if not isinstance(claims, list):
            raise AdjudicationError(
                f"{path}:{row_number}: evidence row has no claims list"
            )
        for claim in claims:
            if not isinstance(claim, dict) or not claim.get("claim_id"):
                raise AdjudicationError(
                    f"{path}:{row_number}: evidence claim has no claim_id"
                )
            claim_id = claim["claim_id"]
            if claim_id in by_claim:
                raise AdjudicationError(
                    f"{path}:{row_number}: duplicate claim_id {claim_id!r}"
                )
            by_claim[claim_id] = claim
    if not by_claim:
        raise AdjudicationError(f"{path}: sidecar carries no reconstructed claims")
    return by_claim, sha


_ABSENT_EVIDENCE = {
    "evidence_status": EVIDENCE_ABSENT,
    "evidence_source_sha256": None,
    "claim_text_full": None,
    "claim_text_truncated": None,
    "claim_char_start": None,
    "claim_char_end": None,
    "context_before": None,
    "context_after": None,
    "claim_kind_recomputed": None,
    "claim_kind_agrees_with_record": None,
    "cited": None,
    "citations": None,
    "cited_evidence_keys": None,
    "evidence_fidelity": None,
    "evidence": None,
    "term_probe": None,
    "reason_probe": None,
    "decision_inputs": None,
}


def _evidence_fields(
    claim: Optional[dict[str, Any]], sidecar_sha: Optional[str]
) -> dict[str, Any]:
    if claim is None:
        return dict(_ABSENT_EVIDENCE)
    if not claim.get("reconstructed"):
        fields = dict(_ABSENT_EVIDENCE)
        fields["evidence_status"] = "not-reconstructed"
        fields["evidence_source_sha256"] = sidecar_sha
        return fields
    context = claim.get("claim_context") or {}
    return {
        "evidence_status": EVIDENCE_RECONSTRUCTED,
        "evidence_source_sha256": sidecar_sha,
        "claim_text_full": claim.get("claim_text_full"),
        "claim_text_truncated": claim.get("claim_text_truncated"),
        "claim_char_start": claim.get("claim_char_start"),
        "claim_char_end": claim.get("claim_char_end"),
        "context_before": context.get("before"),
        "context_after": context.get("after"),
        "claim_kind_recomputed": claim.get("claim_kind_recomputed"),
        "claim_kind_agrees_with_record": claim.get("claim_kind_agrees_with_record"),
        "cited": claim.get("cited"),
        "citations": claim.get("citations"),
        "cited_evidence_keys": claim.get("cited_evidence_keys"),
        "evidence_fidelity": claim.get("evidence_fidelity"),
        "evidence": claim.get("evidence_view"),
        "term_probe": claim.get("term_probe"),
        "reason_probe": claim.get("reason_probe"),
        "decision_inputs": claim.get("decision_inputs"),
    }


# ---------------------------------------------------------------------------
# inventory
# ---------------------------------------------------------------------------


def build_inventory(
    release_path: Path, evidence_path: Optional[Path] = None
) -> list[dict[str, Any]]:
    """Build an unreviewed, deterministic inventory from release failures."""
    inventory: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    claim_ids: set[str] = set()
    release_sha = _sha256_file(release_path)
    evidence_by_claim: dict[str, dict[str, Any]] = {}
    evidence_sha: Optional[str] = None
    if evidence_path is not None:
        evidence_by_claim, evidence_sha = read_evidence(evidence_path, release_sha)

    for row_number, row in enumerate(_read_jsonl(release_path), start=1):
        location = f"{release_path}:row {row_number}"
        case_id = _require_text(row.get("id"), f"{location} id")
        if case_id in case_ids:
            raise AdjudicationError(f"{location}: duplicate case id {case_id!r}")
        case_ids.add(case_id)

        claims = row.get("claims")
        if not isinstance(claims, dict):
            raise AdjudicationError(f"{location}: missing claims object")
        failures = claims.get("failures")
        if not isinstance(failures, list):
            raise AdjudicationError(f"{location}: claims.failures must be a list")
        if not failures:
            # A clean case contributes nothing; its question/answer are never
            # read, so a malformed one must not abort the whole export.
            continue

        question = _require_text(row.get("question"), f"{location} question")
        answer = _require_text(row.get("answer"), f"{location} answer")
        expect = row.get("expect") if isinstance(row.get("expect"), dict) else {}
        notes_used = (
            row.get("notes_used") if isinstance(row.get("notes_used"), dict) else {}
        )

        for failure_number, raw_failure in enumerate(failures, start=1):
            failure_location = f"{location} failure {failure_number}"
            if not isinstance(raw_failure, dict):
                raise AdjudicationError(
                    f"{failure_location}: expected a JSON object"
                )
            failure = {
                key: _require_text(raw_failure.get(key), f"{failure_location} {key}")
                for key in ("status", "kind", "text", "reason")
            }
            claim_id = _claim_id(case_id, failure)
            if claim_id in claim_ids:
                raise AdjudicationError(
                    f"{failure_location}: duplicate claim identity {claim_id}; "
                    "the source failures are indistinguishable"
                )
            claim_ids.add(claim_id)
            record = {
                "schema_version": SCHEMA_VERSION,
                "claim_id": claim_id,
                "case_id": case_id,
                "release_sha256": release_sha,
                "question": question,
                "answer": answer,
                "source_status": failure["status"],
                "source_kind": failure["kind"],
                "claim_text": failure["text"],
                "source_reason": failure["reason"],
                "evidence_keys": list(claims.get("evidence_keys") or []),
                "hits": [
                    {
                        "doc": hit.get("doc"),
                        "page": hit.get("page"),
                        "score": hit.get("score"),
                    }
                    for hit in (row.get("hits") or [])
                    if isinstance(hit, dict)
                ],
                "notes_used": dict(notes_used),
                "expect_docs": list(expect.get("docs") or []),
                "expect_pages": list(expect.get("pages") or []),
                **_evidence_fields(evidence_by_claim.get(claim_id), evidence_sha),
                "label": None,
                "reviewer": "",
                "notes": "",
            }
            record[DIGEST_FIELD] = evidence_digest(record)
            inventory.append({field: record[field] for field in RECORD_FIELDS})
    if evidence_by_claim:
        unknown = set(evidence_by_claim) - claim_ids
        if unknown:
            raise AdjudicationError(
                f"{evidence_path}: {len(unknown)} reconstructed claim(s) are not "
                f"in {release_path}: {sorted(unknown)[:5]}"
            )
    return inventory


def visible_evidence_chars(row: dict[str, Any]) -> int:
    """How many characters of evidence TEXT this row actually shows a reader.

    Evidence keys, hit scores and probe booleans are not evidence: a reviewer
    separates ``missing_citation`` from ``missing_retrieval_evidence`` from
    ``verifier_false_positive`` by reading what the page says.  A row that
    shows zero characters of it is not labellable however many fields it has,
    so it is counted here and refused at ``export``.
    """
    entries = row.get("evidence")
    if not isinstance(entries, list):
        return 0
    return sum(
        len(entry["excerpt"])
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("excerpt"), str)
    )


def evidence_coverage(inventory: Iterable[dict[str, Any]]) -> dict[str, int]:
    rows = list(inventory)
    statuses = Counter(row.get("evidence_status") for row in rows)
    reconstructed = [
        row for row in rows if row.get("evidence_status") == EVIDENCE_RECONSTRUCTED
    ]
    return {
        "total": len(rows),
        "with_evidence": statuses.get(EVIDENCE_RECONSTRUCTED, 0),
        "without_evidence": len(rows) - statuses.get(EVIDENCE_RECONSTRUCTED, 0),
        "without_evidence_text": sum(
            1 for row in reconstructed if visible_evidence_chars(row) == 0
        ),
        "evidence_text_chars": sum(
            visible_evidence_chars(row) for row in reconstructed
        ),
    }


def read_inventory(path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows:
        raise AdjudicationError(f"{path}: inventory is empty")
    return rows


def validate_inventory(
    inventory: Iterable[dict[str, Any]],
    expected: Iterable[dict[str, Any]],
    *,
    allow_unreviewed: bool = False,
) -> dict[str, Any]:
    """Validate an inventory against its release and return deterministic counts."""
    rows = list(inventory)
    expected_rows = list(expected)
    expected_by_id = {row["claim_id"]: row for row in expected_rows}
    seen: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    evidence_hint = False

    for row_number, row in enumerate(rows, start=1):
        prefix = f"inventory row {row_number}"
        fields = set(row)
        missing_fields = set(RECORD_FIELDS) - fields
        extra_fields = fields - set(RECORD_FIELDS)
        if missing_fields:
            errors.append(f"{prefix}: missing fields {sorted(missing_fields)}")
        if extra_fields:
            errors.append(f"{prefix}: unknown fields {sorted(extra_fields)}")
        if missing_fields:
            continue

        claim_id = row["claim_id"]
        if not isinstance(claim_id, str) or not claim_id:
            errors.append(f"{prefix}: claim_id must be non-empty text")
            continue
        if claim_id in seen:
            errors.append(f"{prefix}: duplicate claim_id {claim_id!r}")
            continue
        seen[claim_id] = row

        # Self-consistency first: a row whose evidence no longer hashes to its
        # own digest has been edited, whatever the release says.
        recomputed = evidence_digest(row)
        if row[DIGEST_FIELD] != recomputed:
            errors.append(
                f"{prefix}: {DIGEST_FIELD} does not match the row's own evidence "
                "(the evidence block was edited)"
            )

        canonical = expected_by_id.get(claim_id)
        if canonical is None:
            errors.append(f"{prefix}: unknown claim_id {claim_id!r}")
        else:
            for field in IMMUTABLE_FIELDS:
                if row[field] != canonical[field]:
                    errors.append(
                        f"{prefix}: immutable field {field!r} differs from release"
                    )
                    if field in EVIDENCE_FIELDS or field == DIGEST_FIELD:
                        evidence_hint = evidence_hint or (
                            canonical["evidence_status"] != row["evidence_status"]
                        )

        label = row["label"]
        reviewer = row["reviewer"]
        notes = row["notes"]
        if label is not None and label not in LABELS:
            errors.append(f"{prefix}: invalid label {label!r}")
        if not isinstance(reviewer, str):
            errors.append(f"{prefix}: reviewer must be text")
        if not isinstance(notes, str):
            errors.append(f"{prefix}: notes must be text")
        if label in LABELS and isinstance(reviewer, str) and not _visible(reviewer):
            errors.append(f"{prefix}: reviewed claim requires a reviewer")
        if (
            label == "ambiguous_unscorable"
            and isinstance(notes, str)
            and not _visible(notes)
        ):
            errors.append(
                f"{prefix}: ambiguous_unscorable claim requires explanatory notes"
            )

    missing_ids = set(expected_by_id) - set(seen)
    if missing_ids:
        errors.append(
            f"inventory is missing {len(missing_ids)} release claims: "
            f"{sorted(missing_ids)[:5]}"
        )
    if evidence_hint:
        errors.append(
            "evidence fields differ because the inventory and this check were "
            "built with different --evidence inputs; pass the same sidecar "
            "that produced the inventory"
        )
    if errors:
        raise AdjudicationError("\n".join(errors))

    labels = Counter(row["label"] for row in rows if row["label"] is not None)
    unreviewed = sum(row["label"] is None for row in rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "total": len(rows),
        "reviewed": len(rows) - unreviewed,
        "unreviewed": unreviewed,
        "complete": unreviewed == 0,
        "labels": {label: labels.get(label, 0) for label in LABELS},
        "source_statuses": dict(
            sorted(Counter(row["source_status"] for row in rows).items())
        ),
        "source_kinds": dict(
            sorted(Counter(row["source_kind"] for row in rows).items())
        ),
        "evidence": evidence_coverage(rows),
    }
    if unreviewed and not allow_unreviewed:
        raise AdjudicationError(
            f"inventory has {unreviewed} unreviewed claim(s); "
            "use --allow-unreviewed only for an explicit work-in-progress check"
        )
    return summary


def import_inventory(
    inventory: Iterable[dict[str, Any]],
    expected: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate annotations and return them in canonical release order.

    ``import`` writes the gated canonical artifact, so it has no
    work-in-progress escape hatch: an incomplete inventory is a hard error.
    """
    rows = list(inventory)
    expected_rows = list(expected)
    summary = validate_inventory(rows, expected_rows, allow_unreviewed=False)
    annotations = {
        row["claim_id"]: {field: row[field] for field in ANNOTATION_FIELDS}
        for row in rows
    }
    merged: list[dict[str, Any]] = []
    for canonical in expected_rows:
        row = dict(canonical)
        row.update(annotations[row["claim_id"]])
        merged.append(row)
    return merged, summary


def write_inventory(
    path: Path, rows: Iterable[dict[str, Any]], *, force: bool = False
) -> None:
    if path.exists():
        if not force:
            raise AdjudicationError(f"refusing to overwrite {path}; pass --force")
        labelled = _labelled_rows(path)
        if labelled:
            raise AdjudicationError(
                f"refusing to overwrite {path} even with --force: it holds "
                f"{labelled} human label(s); move it aside first"
            )
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
        temporary.chmod(0o644)
        temporary.replace(path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise AdjudicationError(f"cannot write {path}: {exc}") from exc


def _labelled_rows(path: Path) -> int:
    """How many human labels an existing file at ``path`` would destroy.

    A file we cannot parse as an inventory is not evidence of safety: it is
    counted as one label so ``--force`` refuses rather than guessing.
    """
    try:
        rows = _read_jsonl(path)
    except AdjudicationError:
        return 1
    if not rows:
        return 0
    if not all("label" in row for row in rows):
        return 1
    return sum(1 for row in rows if row.get("label") is not None)


def _assert_distinct_outputs(output: Path, **inputs: Optional[Path]) -> None:
    resolved = output.resolve()
    for name, source in inputs.items():
        if source is not None and resolved == source.resolve():
            raise AdjudicationError(
                f"refusing to write --output over --{name} ({source})"
            )


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def _add_release_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--release",
        type=Path,
        required=True,
        help="release JSONL containing claims.failures",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=None,
        help="evidence sidecar from scripts/backfill_release_evidence.py",
    )


def _add_work_in_progress_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="explicitly permit null labels for a work-in-progress check",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export", help="create an unreviewed inventory from a release"
    )
    _add_release_argument(export_parser)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--force", action="store_true")
    export_parser.add_argument(
        "--allow-missing-evidence",
        action="store_true",
        help="export rows a human cannot label; for inspection only",
    )

    validate_parser = subparsers.add_parser(
        "validate", help="validate an inventory against its release"
    )
    _add_release_argument(validate_parser)
    validate_parser.add_argument("--inventory", type=Path, required=True)
    _add_work_in_progress_argument(validate_parser)

    import_parser = subparsers.add_parser(
        "import", help="validate and canonicalize a completed inventory"
    )
    _add_release_argument(import_parser)
    import_parser.add_argument("--inventory", type=Path, required=True)
    import_parser.add_argument("--output", type=Path, required=True)
    import_parser.add_argument("--force", action="store_true")

    summary_parser = subparsers.add_parser(
        "summary", help="print label and source-failure counts"
    )
    _add_release_argument(summary_parser)
    summary_parser.add_argument("--inventory", type=Path, required=True)
    _add_work_in_progress_argument(summary_parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if getattr(args, "output", None) is not None:
            _assert_distinct_outputs(
                args.output,
                release=args.release,
                evidence=args.evidence,
                inventory=getattr(args, "inventory", None),
            )
        expected = build_inventory(args.release, args.evidence)
        coverage = evidence_coverage(expected)
        if args.command == "export":
            if coverage["without_evidence"] and not args.allow_missing_evidence:
                raise AdjudicationError(
                    f"{coverage['without_evidence']} of {coverage['total']} claims "
                    "would export without the evidence their labels are defined "
                    "over; build the sidecar with "
                    "scripts/backfill_release_evidence.py and pass --evidence "
                    "(or --allow-missing-evidence for a non-labellable dump)"
                )
            if coverage["without_evidence_text"] and not args.allow_missing_evidence:
                raise AdjudicationError(
                    f"{coverage['without_evidence_text']} of {coverage['total']} "
                    "claims carry evidence keys but no evidence TEXT; a reviewer "
                    "cannot separate missing_citation from "
                    "missing_retrieval_evidence from verifier_false_positive "
                    "without reading what the page says. Rebuild the sidecar "
                    "with scripts/backfill_release_evidence.py "
                    "(or --allow-missing-evidence for a non-labellable dump)"
                )
            write_inventory(args.output, expected, force=args.force)
            print(
                f"exported {len(expected)} unreviewed claims to {args.output} "
                f"({coverage['with_evidence']} with reconstructed evidence, "
                f"{coverage['without_evidence']} without); "
                "complete human labels before validation"
            )
            return 0

        inventory = read_inventory(args.inventory)
        if args.command == "import":
            merged, summary = import_inventory(inventory, expected)
            write_inventory(args.output, merged, force=args.force)
        else:
            summary = validate_inventory(
                inventory,
                expected,
                allow_unreviewed=args.allow_unreviewed,
            )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except AdjudicationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
