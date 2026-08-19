"""Tests for the claim-failure adjudication workflow."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import adjudicate_claims as ac  # noqa: E402


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def _release_row(case_id, *failures):
    return {
        "id": case_id,
        "question": f"Question for {case_id}?",
        "answer": f"Answer for {case_id} [doc-{case_id}, p.1].",
        "claims": {"failures": list(failures)},
    }


def _failure(text, *, status="unsupported", kind="entity"):
    return {
        "status": status,
        "kind": kind,
        "text": text,
        "reason": f"Reason for {text}",
    }


def _sample_release(tmp_path):
    rows = [
        _release_row("case-b", _failure("Claim two", kind="money")),
        _release_row(
            "case-a",
            _failure("Claim one"),
            _failure("Claim three", status="contradicted", kind="year"),
        ),
    ]
    return _write_jsonl(tmp_path / "release.jsonl", rows), rows


def _complete(inventory):
    completed = copy.deepcopy(inventory)
    for index, row in enumerate(completed):
        row["label"] = ac.LABELS[index % len(ac.LABELS)]
        row["reviewer"] = "reviewer@example.test"
        row["notes"] = "Reviewed against the recorded answer and evidence."
    return completed


def test_real_release_has_deterministic_unreviewed_inventory():
    release = ROOT / "data" / "eval" / "release_release-1.jsonl"

    first = ac.build_inventory(release)
    second = ac.build_inventory(release)

    assert first == second
    assert len(first) == 71
    assert len({row["claim_id"] for row in first}) == 71
    assert all(row["label"] is None for row in first)
    assert all(row["reviewer"] == "" for row in first)


def test_claim_ids_are_stable_when_release_rows_are_reordered(tmp_path):
    release, rows = _sample_release(tmp_path)
    normal = ac.build_inventory(release)
    reversed_release = _write_jsonl(tmp_path / "reversed.jsonl", reversed(rows))
    reordered = ac.build_inventory(reversed_release)

    normal_ids = {row["claim_text"]: row["claim_id"] for row in normal}
    reordered_ids = {row["claim_text"]: row["claim_id"] for row in reordered}
    assert normal_ids == reordered_ids


def test_validation_fails_closed_on_unreviewed_claims(tmp_path):
    release, _ = _sample_release(tmp_path)
    inventory = ac.build_inventory(release)

    with pytest.raises(ac.AdjudicationError, match="3 unreviewed"):
        ac.validate_inventory(inventory, inventory)

    summary = ac.validate_inventory(
        inventory, inventory, allow_unreviewed=True
    )
    assert summary["complete"] is False
    assert summary["reviewed"] == 0
    assert summary["unreviewed"] == 3


def test_validation_requires_an_allowed_label_and_named_reviewer(tmp_path):
    release, _ = _sample_release(tmp_path)
    expected = ac.build_inventory(release)
    inventory = _complete(expected)

    inventory[0]["label"] = "matcher_bug"
    with pytest.raises(ac.AdjudicationError, match="invalid label"):
        ac.validate_inventory(inventory, expected)

    inventory = _complete(expected)
    inventory[0]["reviewer"] = ""
    with pytest.raises(ac.AdjudicationError, match="requires a reviewer"):
        ac.validate_inventory(inventory, expected)

    inventory = _complete(expected)
    inventory[0]["label"] = "ambiguous_unscorable"
    inventory[0]["notes"] = ""
    with pytest.raises(ac.AdjudicationError, match="requires explanatory notes"):
        ac.validate_inventory(inventory, expected)


@pytest.mark.parametrize("mutation, message", [
    ("alter", "immutable field 'claim_text'"),
    ("remove", "inventory is missing 1 release claims"),
    ("duplicate", "duplicate claim_id"),
])
def test_validation_detects_drift_missing_and_duplicate_claims(
    tmp_path, mutation, message
):
    release, _ = _sample_release(tmp_path)
    expected = ac.build_inventory(release)
    inventory = _complete(expected)

    if mutation == "alter":
        inventory[0]["claim_text"] = "Edited source claim"
    elif mutation == "remove":
        inventory.pop()
    else:
        inventory.append(copy.deepcopy(inventory[0]))

    with pytest.raises(ac.AdjudicationError, match=message):
        ac.validate_inventory(inventory, expected)


def test_import_restores_canonical_order_and_summarizes_labels(tmp_path):
    release, _ = _sample_release(tmp_path)
    expected = ac.build_inventory(release)
    inventory = list(reversed(_complete(expected)))

    merged, summary = ac.import_inventory(inventory, expected)

    assert [row["claim_id"] for row in merged] == [
        row["claim_id"] for row in expected
    ]
    assert summary["complete"] is True
    assert summary["total"] == 3
    assert summary["source_statuses"] == {
        "contradicted": 1,
        "unsupported": 2,
    }
    assert sum(summary["labels"].values()) == 3


def test_cli_export_is_unreviewed_and_default_summary_returns_failure(
    tmp_path, capsys
):
    release, _ = _sample_release(tmp_path)
    inventory_path = tmp_path / "inventory.jsonl"

    assert ac.main([
        "export",
        "--release", str(release),
        "--output", str(inventory_path),
    ]) == 0
    exported = ac.read_inventory(inventory_path)
    assert all(row["label"] is None for row in exported)

    assert ac.main([
        "summary",
        "--release", str(release),
        "--inventory", str(inventory_path),
    ]) == 2
    assert "unreviewed claim(s)" in capsys.readouterr().err

    assert ac.main([
        "summary",
        "--release", str(release),
        "--inventory", str(inventory_path),
        "--allow-unreviewed",
    ]) == 0
    assert '"unreviewed": 3' in capsys.readouterr().out


def test_duplicate_source_failures_are_rejected_as_indistinguishable(tmp_path):
    failure = _failure("Same claim")
    release = _write_jsonl(
        tmp_path / "release.jsonl",
        [_release_row("case-a", failure, copy.deepcopy(failure))],
    )

    with pytest.raises(ac.AdjudicationError, match="indistinguishable"):
        ac.build_inventory(release)
