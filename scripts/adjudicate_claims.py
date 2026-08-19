#!/usr/bin/env python3
"""Export, validate, import, and summarize claim-failure adjudications."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
LABELS = (
    "verifier_false_positive",
    "genuine_answer_error",
    "missing_retrieval_evidence",
    "missing_citation",
    "registry_conflict",
    "ambiguous_unscorable",
)
RECORD_FIELDS = (
    "schema_version",
    "claim_id",
    "case_id",
    "question",
    "answer",
    "source_status",
    "source_kind",
    "claim_text",
    "source_reason",
    "label",
    "reviewer",
    "notes",
)
IMMUTABLE_FIELDS = RECORD_FIELDS[:9]
ANNOTATION_FIELDS = RECORD_FIELDS[9:]


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
    encoded = json.dumps(
        identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"claim-{hashlib.sha256(encoded).hexdigest()[:24]}"


def build_inventory(release_path: Path) -> list[dict[str, Any]]:
    """Build an unreviewed, deterministic inventory from release failures."""
    inventory: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    claim_ids: set[str] = set()

    for row_number, row in enumerate(_read_jsonl(release_path), start=1):
        location = f"{release_path}:row {row_number}"
        case_id = _require_text(row.get("id"), f"{location} id")
        if case_id in case_ids:
            raise AdjudicationError(f"{location}: duplicate case id {case_id!r}")
        case_ids.add(case_id)

        question = _require_text(row.get("question"), f"{location} question")
        answer = _require_text(row.get("answer"), f"{location} answer")
        claims = row.get("claims")
        if not isinstance(claims, dict):
            raise AdjudicationError(f"{location}: missing claims object")
        failures = claims.get("failures")
        if not isinstance(failures, list):
            raise AdjudicationError(f"{location}: claims.failures must be a list")

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
            inventory.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "claim_id": claim_id,
                    "case_id": case_id,
                    "question": question,
                    "answer": answer,
                    "source_status": failure["status"],
                    "source_kind": failure["kind"],
                    "claim_text": failure["text"],
                    "source_reason": failure["reason"],
                    "label": None,
                    "reviewer": "",
                    "notes": "",
                }
            )
    return inventory


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

        canonical = expected_by_id.get(claim_id)
        if canonical is None:
            errors.append(f"{prefix}: unknown claim_id {claim_id!r}")
        else:
            for field in IMMUTABLE_FIELDS:
                if row[field] != canonical[field]:
                    errors.append(
                        f"{prefix}: immutable field {field!r} differs from release"
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
        if label in LABELS and isinstance(reviewer, str) and not reviewer.strip():
            errors.append(f"{prefix}: reviewed claim requires a reviewer")
        if (
            label == "ambiguous_unscorable"
            and isinstance(notes, str)
            and not notes.strip()
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
    *,
    allow_unreviewed: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate annotations and return them in canonical release order."""
    rows = list(inventory)
    expected_rows = list(expected)
    summary = validate_inventory(
        rows, expected_rows, allow_unreviewed=allow_unreviewed
    )
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
    if path.exists() and not force:
        raise AdjudicationError(f"refusing to overwrite {path}; pass --force")
    content = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    temporary: Path | None = None
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
        temporary.replace(path)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise AdjudicationError(f"cannot write {path}: {exc}") from exc


def _add_release_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--release",
        type=Path,
        required=True,
        help="release JSONL containing claims.failures",
    )


def _add_work_in_progress_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="explicitly permit null labels for a work-in-progress check",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export", help="create an unreviewed inventory from a release"
    )
    _add_release_argument(export_parser)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--force", action="store_true")

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
    _add_work_in_progress_argument(import_parser)

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
        expected = build_inventory(args.release)
        if args.command == "export":
            write_inventory(args.output, expected, force=args.force)
            print(
                f"exported {len(expected)} unreviewed claims to {args.output}; "
                "complete human labels before validation"
            )
            return 0

        inventory = read_inventory(args.inventory)
        if args.command == "import":
            merged, summary = import_inventory(
                inventory,
                expected,
                allow_unreviewed=args.allow_unreviewed,
            )
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
