"""Tests for the claim-failure adjudication workflow and its evidence backfill.

Two halves:

* ``scripts/backfill_release_evidence.py`` reconstructs the evidence a recorded
  release run actually held (hit text recovered from the index, note blocks,
  claim spans, citations) into a sidecar;
* ``scripts/adjudicate_claims.py`` exports that evidence into the rows a human
  labels, and refuses to validate a row whose evidence has been edited.

Every assertion here is paired: an adversarial case that must be REFUSED and a
permissive case that must still be ACCEPTED, so a later widening of any check
fails the suite instead of passing it.
"""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import adjudicate_claims as ac  # noqa: E402
import backfill_release_evidence as bf  # noqa: E402

RELEASE = ROOT / "data" / "eval" / "release_release-1.jsonl"
CHUNKS = ROOT / "data" / "index" / "default" / "chunks.jsonl"

#: The release-1 case and claim that established finding N3: the corpus prints
#: 'USD 49,312 million' where the answer prints 'USD 49.312 million'.
N3_CASE = "bc-b30-03-add04"
N3_CLAIM = "claim-ce3be1a81f0ac4568cbbbab4"
N3_PAGE_KEY = "103_gcf-b30-03-add04|140"

# The release run and the index are large, gitignored-by-default artifacts:
# a clean checkout has neither, and a test that errors there is finding 26 all
# over again.
needs_release = pytest.mark.skipif(
    not RELEASE.exists(), reason="recorded release run not present in this checkout"
)
needs_index = pytest.mark.skipif(
    not (RELEASE.exists() and CHUNKS.exists()),
    reason="chunk index not present in this checkout",
)


@pytest.fixture(scope="session")
def real_sidecar(tmp_path_factory):
    """The sidecar for the real release, REBUILT from the release and the index.

    Deliberately not the committed ``release_release-1-evidence.jsonl``: a
    checked-in sidecar is a snapshot of whatever the backfill did when someone
    last ran it, so asserting against it tests the artifact and not the code —
    exactly the staleness N20 flags on the sidecar's own provenance stamp.
    """
    if not (RELEASE.exists() and CHUNKS.exists()):
        pytest.skip("release run or chunk index not present in this checkout")
    rows, _ = bf.build_sidecar(RELEASE, CHUNKS)
    path = tmp_path_factory.mktemp("sidecar") / "evidence.jsonl"
    return _write_jsonl(path, rows)


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


#: document ids must look like the corpus's ('01_gcf-b27-…'); verify's _DOC_RE
#: is what decides whether a registry note attaches to a document at all.
def _doc(case_id):
    return f"01_{case_id}"


#: sentences verify.extract_claims really does return as claims
CLAIM_ONE = "Claim one is implemented by the Ruritania Agency."
CLAIM_TWO = "Claim two requests USD 5.0 million."
CLAIM_THREE = "Claim three was approved at B.30 in 2019."


def _failure(text, *, status="unsupported", kind="entity", reason=None):
    return {
        "status": status,
        "kind": kind,
        "text": text,
        "reason": reason or f"Reason for {text}",
    }


def _release_row(case_id, *failures, answer=None, **overrides):
    texts = [failure["text"] for failure in failures]
    row = {
        "id": case_id,
        "question": f"Question for {case_id}?",
        "answer": answer if answer is not None else (
            " ".join(texts) or f"Nothing to report for {case_id}."
        ),
        "claims": {"failures": list(failures), "evidence_keys": [f"{_doc(case_id)}|1"]},
        "hits": [{"doc": _doc(case_id), "page": 1, "score": 0.5}],
        "notes_used": {},
        "expect": {"docs": [_doc(case_id)], "pages": [1]},
    }
    row.update(overrides)
    return row


def _sample_release(tmp_path):
    rows = [
        _release_row("case-b", _failure(CLAIM_TWO, kind="money")),
        _release_row(
            "case-a",
            _failure(CLAIM_ONE),
            _failure(CLAIM_THREE, status="contradicted", kind="year"),
        ),
    ]
    return _write_jsonl(tmp_path / "release.jsonl", rows), rows


def _sample_chunks(tmp_path, name="chunks.jsonl"):
    return _write_jsonl(
        tmp_path / name,
        [
            {
                "doc_id": _doc("case-a"),
                "page": 1,
                "text": "The Ruritania Agency implements it; approved at B.30 in 2019.",
            },
            {
                "doc_id": _doc("case-b"),
                "page": 1,
                "text": "The request is USD 5.0 million.",
            },
        ],
    )


def _sample_sidecar(tmp_path, release):
    """A real sidecar for the synthetic release, via the real backfill."""
    rows, _ = bf.build_sidecar(release, _sample_chunks(tmp_path))
    return _write_jsonl(tmp_path / "evidence.jsonl", rows)


def _inventory(tmp_path):
    release, _ = _sample_release(tmp_path)
    sidecar = _sample_sidecar(tmp_path, release)
    return release, sidecar, ac.build_inventory(release, sidecar)


def _complete(inventory):
    completed = copy.deepcopy(inventory)
    for index, row in enumerate(completed):
        row["label"] = ac.LABELS[index % len(ac.LABELS)]
        row["reviewer"] = "reviewer@example.test"
        row["notes"] = "Reviewed against the recorded answer and evidence."
    return completed


# ---------------------------------------------------------------------------
# finding 6 — the export must carry the evidence the labels are defined over
# ---------------------------------------------------------------------------


def test_export_carries_hits_evidence_keys_and_registry_notes(real_sidecar):
    """The three fields finding 6 names as dropped, on every real row."""
    inventory = ac.build_inventory(RELEASE, real_sidecar)

    assert len(inventory) == len({row["claim_id"] for row in inventory})
    assert all(row["evidence_keys"] for row in inventory)
    assert all(isinstance(row["notes_used"], dict) for row in inventory)
    with_registry = [row for row in inventory if row["notes_used"].get("registry")]
    assert len(with_registry) >= 50, "registry notes must survive the export"
    # Two release cases retrieved nothing and held only the registry note; every
    # other row must carry its hits, with page numbers and not just doc ids.
    hitless = [row for row in inventory if not row["hits"]]
    assert len(hitless) <= 2
    assert all(row["notes_used"] for row in hitless)
    assert all(
        any(hit.get("page") is not None for hit in row["hits"])
        for row in inventory if row["hits"]
    )


def test_export_row_is_self_sufficient_for_a_human_label(real_sidecar):
    inventory = ac.build_inventory(RELEASE, real_sidecar)

    for row in inventory:
        assert row["evidence_status"] == "reconstructed"
        assert row["claim_text_full"] and row["claim_text_full"].startswith(
            row["claim_text"][:80]
        )
        assert row["claim_char_start"] >= 0
        assert row["context_before"] is not None
        assert row["context_after"] is not None
        assert isinstance(row["citations"], list)
        assert row["evidence"], "no evidence passages on the row"
        assert all("excerpt" in entry for entry in row["evidence"])
        assert row["decision_inputs"]["n_terms"] >= 0
        assert row["source_status"] and row["source_reason"]


# ---------------------------------------------------------------------------
# N14 — the row must show the reviewer the evidence TEXT, not just its keys
# ---------------------------------------------------------------------------


def test_every_real_row_shows_the_reviewer_evidence_text(real_sidecar):
    """No row, and no held page inside a row, may be text-free.

    The dimension: a page that matched NOTHING and is cited by nothing. The
    previous policy emitted an excerpt only when a probe matched or the page
    was cited, which left 601 of 853 entries and 12 whole rows blank —
    precisely the rows whose labels turn on what the pages do not say.
    """
    inventory = ac.build_inventory(RELEASE, real_sidecar)

    blank_entries = [
        (row["claim_id"], entry["key"])
        for row in inventory
        for entry in row["evidence"]
        if not entry["excerpt"]
    ]
    assert blank_entries == []

    blank_rows = [
        row["claim_id"] for row in inventory
        if not any(entry["excerpt"] for entry in row["evidence"])
    ]
    assert blank_rows == []
    assert all(ac.visible_evidence_chars(row) > 0 for row in inventory)

    # and the unmatched, uncited pages — the ones that used to be null — are
    # the majority of what is now visible, not a rounding error
    unmatched = [
        entry
        for row in inventory
        for entry in row["evidence"]
        if not entry["matched_terms"] and not entry["cited_by_claim"]
    ]
    assert len(unmatched) > 300
    assert all(entry["excerpt_chars"] > 0 for entry in unmatched)


def test_no_citation_claims_carry_the_evidence_that_separates_their_labels(
    real_sidecar,
):
    """Finding 6's undecidable set: 24 uncited claims, every one with text.

    The labels turn on what the held pages say, so each such row must show the
    pages, name which of them the probe matched, and leave
    ``verifier_false_positive`` live. What it must NOT do is rule a label out
    on the strength of the probe (N3) — the only exclusions permitted here are
    structural.
    """
    inventory = ac.build_inventory(RELEASE, real_sidecar)
    uncited = [row for row in inventory if row["decision_inputs"]["cited"] is False]

    assert len(uncited) >= 20, "the release's uncited failures should dominate"
    for row in uncited:
        inputs = row["decision_inputs"]
        # missing_citation may only be excluded structurally: the claim has
        # nothing citable in it. Never because a probe failed to find a term.
        reason = inputs["labels_excluded_by_evidence"].get("missing_citation")
        if reason is not None:
            assert inputs["n_terms"] == 0, (row["claim_id"], reason)
            assert "no citation could have supported" not in reason
        assert "verifier_false_positive" in inputs["labels_remaining"]

    text_free = [
        row["claim_id"] for row in uncited
        if not any(entry["excerpt"] for entry in row["evidence"])
    ]
    assert text_free == []
    assert all(ac.visible_evidence_chars(row) > 0 for row in uncited)


@needs_index
def test_the_recorded_n3_claim_is_labellable_from_its_own_row(real_sidecar):
    """The row that made a reviewer land on the wrong label.

    ``bc-b30-03-add04`` answers 'USD 49.312 million'; page 140 of the cited
    document prints 'USD 49,312 million'. Before the fix the probe reported
    the figure ABSENT, ruled ``missing_citation`` — the correct label — off
    the list, and showed no excerpt at all, so nothing in the row could
    contradict it.
    """
    inventory = ac.build_inventory(RELEASE, real_sidecar)
    (row,) = [r for r in inventory if r["claim_id"] == N3_CLAIM]
    assert row["case_id"] == N3_CASE
    assert "49.312" in row["claim_text_full"]

    inputs = row["decision_inputs"]
    (amount,) = [t for t in row["term_probe"] if t["term_kind"] == "amount"]
    assert amount["present_in_held_evidence"] is True
    assert amount["match_kinds"] == ["digits"]
    assert amount["found_in_keys"] == [N3_PAGE_KEY]
    assert amount["matches_by_key"][N3_PAGE_KEY]["spellings"] == ["49,312"]

    assert inputs["terms_absent_from_held_evidence"] == []
    assert inputs["terms_matched_only_by_separator_variant"] == [amount["term"]]
    assert "missing_citation" in inputs["labels_remaining"]
    assert "missing_citation" not in inputs["labels_excluded_by_evidence"]

    # and the reviewer can see the corpus's own spelling to judge it
    (page,) = [e for e in row["evidence"] if e["key"] == N3_PAGE_KEY]
    assert "49,312" in page["excerpt"]
    assert page["matched_spellings"] == ["49,312"]


def test_export_refuses_rows_without_evidence_but_allows_an_explicit_dump(tmp_path):
    release, _ = _sample_release(tmp_path)
    output = tmp_path / "inventory.jsonl"

    assert ac.main([
        "export", "--release", str(release), "--output", str(output),
    ]) == 2
    assert not output.exists()

    # permissive direction: the escape hatch still works and marks the rows
    assert ac.main([
        "export", "--release", str(release), "--output", str(output),
        "--allow-missing-evidence",
    ]) == 0
    rows = ac.read_inventory(output)
    assert {row["evidence_status"] for row in rows} == {"absent"}


# ---------------------------------------------------------------------------
# finding 13 — tamper detection over the evidence, not just nine copied fields
# ---------------------------------------------------------------------------


def test_fabricated_evidence_is_rejected_with_and_without_a_fresh_digest(tmp_path):
    _, _, expected = _inventory(tmp_path)
    inventory = _complete(expected)

    # (a) evidence swapped, digest left behind
    tampered = copy.deepcopy(inventory)
    tampered[0]["evidence"] = [{"key": "made-up|1", "excerpt": "fabricated"}]
    with pytest.raises(ac.AdjudicationError, match="evidence_digest does not match"):
        ac.validate_inventory(tampered, expected)

    # (b) evidence swapped AND the digest recomputed to match
    tampered = copy.deepcopy(inventory)
    tampered[0]["evidence"] = [{"key": "made-up|1", "excerpt": "fabricated"}]
    tampered[0]["evidence_digest"] = ac.evidence_digest(tampered[0])
    with pytest.raises(ac.AdjudicationError, match="immutable field 'evidence'"):
        ac.validate_inventory(tampered, expected)

    # permissive direction: the untouched inventory still validates
    summary = ac.validate_inventory(inventory, expected)
    assert summary["complete"] is True
    assert summary["evidence"]["with_evidence"] == len(inventory)


def test_a_labelled_row_repointed_at_another_claim_is_rejected(tmp_path):
    _, _, expected = _inventory(tmp_path)
    inventory = _complete(expected)

    repointed = copy.deepcopy(inventory)
    repointed[0]["claim_id"] = repointed[1]["claim_id"]
    with pytest.raises(ac.AdjudicationError, match="duplicate claim_id"):
        ac.validate_inventory(repointed, expected)

    # keep the id, move the evidence of a different claim onto it
    borrowed = copy.deepcopy(inventory)
    for field in ac.EVIDENCE_FIELDS:
        borrowed[0][field] = copy.deepcopy(borrowed[1][field])
    borrowed[0]["evidence_digest"] = ac.evidence_digest(borrowed[0])
    with pytest.raises(ac.AdjudicationError, match="immutable field"):
        ac.validate_inventory(borrowed, expected)

    # permissive: reordering rows is not tampering
    ac.validate_inventory(list(reversed(inventory)), expected)


def test_a_modified_release_source_is_detected(tmp_path):
    release, sidecar, expected = _inventory(tmp_path)
    inventory = _complete(expected)
    assert {row["release_sha256"] for row in inventory} == {
        ac._sha256_file(release)
    }

    rows = [json.loads(line) for line in release.read_text().splitlines() if line]
    rows[0]["answer"] += " Silently added sentence."
    _write_jsonl(release, rows)

    # the sidecar still names the release it was built from
    with pytest.raises(ac.AdjudicationError, match="sidecar was built from release"):
        ac.build_inventory(release, sidecar)

    # and rebuilding the sidecar does not launder it either: the labelled rows
    # still carry the fingerprint of the release they were exported from
    rebuilt = _sample_sidecar(tmp_path, release)
    edited = ac.build_inventory(release, rebuilt)
    with pytest.raises(ac.AdjudicationError, match="immutable field 'release_sha256'"):
        ac.validate_inventory(inventory, edited)


def test_unknown_and_missing_claim_ids_are_rejected(tmp_path):
    _, _, expected = _inventory(tmp_path)
    inventory = _complete(expected)

    unknown = copy.deepcopy(inventory)
    unknown[0]["claim_id"] = "claim-000000000000000000000000"
    with pytest.raises(ac.AdjudicationError, match="unknown claim_id"):
        ac.validate_inventory(unknown, expected)

    short = copy.deepcopy(inventory)
    short.pop()
    with pytest.raises(ac.AdjudicationError, match="missing 1 release claims"):
        ac.validate_inventory(short, expected)

    duplicated = copy.deepcopy(inventory)
    duplicated.append(copy.deepcopy(duplicated[0]))
    with pytest.raises(ac.AdjudicationError, match="duplicate claim_id"):
        ac.validate_inventory(duplicated, expected)


def test_unknown_and_missing_fields_are_rejected(tmp_path):
    _, _, expected = _inventory(tmp_path)

    extra = copy.deepcopy(_complete(expected))
    extra[0]["confidence"] = 0.9
    with pytest.raises(ac.AdjudicationError, match=r"unknown fields \['confidence'\]"):
        ac.validate_inventory(extra, expected)

    dropped = copy.deepcopy(_complete(expected))
    del dropped[0]["hits"]
    with pytest.raises(ac.AdjudicationError, match=r"missing fields \['hits'\]"):
        ac.validate_inventory(dropped, expected)


def test_immutable_field_set_covers_everything_but_the_annotations():
    assert set(ac.RECORD_FIELDS) == set(ac.IMMUTABLE_FIELDS) | set(
        ac.ANNOTATION_FIELDS
    )
    assert not set(ac.IMMUTABLE_FIELDS) & set(ac.ANNOTATION_FIELDS)
    for field in ("hits", "evidence_keys", "notes_used", "evidence", "decision_inputs"):
        assert field in ac.IMMUTABLE_FIELDS


# ---------------------------------------------------------------------------
# findings 14, 15, 24 — gates that must not be bypassable
# ---------------------------------------------------------------------------


def test_validate_needs_an_explicit_allow_unreviewed(tmp_path):
    _, _, expected = _inventory(tmp_path)

    with pytest.raises(ac.AdjudicationError, match="3 unreviewed"):
        ac.validate_inventory(expected, expected)

    summary = ac.validate_inventory(expected, expected, allow_unreviewed=True)
    assert summary["complete"] is False
    assert summary["reviewed"] == 0
    assert summary["unreviewed"] == 3


def test_import_has_no_unreviewed_escape_hatch(tmp_path):
    release, sidecar, expected = _inventory(tmp_path)
    inventory_path = _write_jsonl(tmp_path / "inventory.jsonl", expected)

    with pytest.raises(SystemExit):
        ac.build_parser().parse_args([
            "import",
            "--release", str(release),
            "--inventory", str(inventory_path),
            "--output", str(tmp_path / "out.jsonl"),
            "--allow-unreviewed",
        ])
    with pytest.raises(ac.AdjudicationError, match="unreviewed claim"):
        ac.import_inventory(expected, expected)
    assert not (tmp_path / "out.jsonl").exists()

    # permissive: a complete inventory imports and canonicalizes
    merged, summary = ac.import_inventory(
        list(reversed(_complete(expected))), expected
    )
    assert [row["claim_id"] for row in merged] == [
        row["claim_id"] for row in expected
    ]
    assert summary["complete"] is True


def test_force_never_destroys_human_labels_but_still_replaces_a_draft(tmp_path):
    _, _, expected = _inventory(tmp_path)
    path = tmp_path / "inventory.jsonl"

    _write_jsonl(path, _complete(expected))
    with pytest.raises(ac.AdjudicationError, match="3 human label"):
        ac.write_inventory(path, expected, force=True)
    assert [row["label"] for row in ac.read_inventory(path)] != [None] * 3

    _write_jsonl(path, expected)               # unreviewed draft
    ac.write_inventory(path, expected, force=True)
    assert all(row["label"] is None for row in ac.read_inventory(path))


def test_output_may_not_be_the_release_or_the_evidence_sidecar(tmp_path):
    release, sidecar, _ = _inventory(tmp_path)
    before = release.read_bytes(), sidecar.read_bytes()

    assert ac.main([
        "export", "--release", str(release), "--evidence", str(sidecar),
        "--output", str(release), "--force",
    ]) == 2
    assert ac.main([
        "export", "--release", str(release), "--evidence", str(sidecar),
        "--output", str(sidecar), "--force",
    ]) == 2
    assert (release.read_bytes(), sidecar.read_bytes()) == before


def test_zero_width_reviewer_and_notes_do_not_pass_as_attribution(tmp_path):
    _, _, expected = _inventory(tmp_path)

    blank = _complete(expected)
    blank[0]["reviewer"] = "​​"
    with pytest.raises(ac.AdjudicationError, match="requires a reviewer"):
        ac.validate_inventory(blank, expected)

    blank = _complete(expected)
    blank[0]["label"] = "ambiguous_unscorable"
    blank[0]["notes"] = "⁠ ﻿"
    with pytest.raises(ac.AdjudicationError, match="requires explanatory notes"):
        ac.validate_inventory(blank, expected)

    # permissive: a real name that merely contains a zero-width joiner is fine
    joined = _complete(expected)
    joined[0]["reviewer"] = "Ada‍Lovelace"
    ac.validate_inventory(joined, expected)


def test_invalid_labels_are_rejected_and_every_declared_label_is_accepted(tmp_path):
    _, _, expected = _inventory(tmp_path)

    inventory = _complete(expected)
    inventory[0]["label"] = "matcher_bug"
    with pytest.raises(ac.AdjudicationError, match="invalid label"):
        ac.validate_inventory(inventory, expected)

    for label in ac.LABELS:
        inventory = _complete(expected)
        for row in inventory:
            row["label"] = label
            row["notes"] = "explanatory notes"
        summary = ac.validate_inventory(inventory, expected)
        assert summary["labels"][label] == len(inventory)


# ---------------------------------------------------------------------------
# finding 22 — a clean case must not abort the export
# ---------------------------------------------------------------------------


def test_a_clean_case_with_a_bad_question_does_not_abort_the_export(tmp_path):
    release = _write_jsonl(
        tmp_path / "release.jsonl",
        [
            _release_row("case-clean", question=None, claims={
                "claims": 0, "failures": [], "evidence_keys": []}),
            _release_row("case-bad", _failure(CLAIM_ONE)),
        ],
    )
    inventory = ac.build_inventory(release, None)
    assert [row["case_id"] for row in inventory] == ["case-bad"]

    # permissive direction stays strict: a FAILING case still needs its text
    broken = _write_jsonl(
        tmp_path / "broken.jsonl",
        [_release_row("case-bad", _failure(CLAIM_ONE), question=None)],
    )
    with pytest.raises(ac.AdjudicationError, match="question"):
        ac.build_inventory(broken, None)


def test_duplicate_source_failures_are_rejected_as_indistinguishable(tmp_path):
    failure = _failure("Same claim")
    release = _write_jsonl(
        tmp_path / "release.jsonl",
        [_release_row("case-a", failure, copy.deepcopy(failure))],
    )

    with pytest.raises(ac.AdjudicationError, match="indistinguishable"):
        ac.build_inventory(release, None)


def test_claim_ids_are_stable_when_release_rows_are_reordered(tmp_path):
    release, rows = _sample_release(tmp_path)
    normal = ac.build_inventory(release, None)
    reversed_release = _write_jsonl(tmp_path / "reversed.jsonl", reversed(rows))
    reordered = ac.build_inventory(reversed_release, None)

    normal_ids = {row["claim_text"]: row["claim_id"] for row in normal}
    reordered_ids = {row["claim_text"]: row["claim_id"] for row in reordered}
    assert normal_ids == reordered_ids


def test_a_sidecar_built_from_a_different_release_is_rejected(tmp_path):
    release, sidecar, _ = _inventory(tmp_path)

    rows = [json.loads(line) for line in sidecar.read_text().splitlines() if line]
    for row in rows:
        row["release_sha256"] = "0" * 64
    _write_jsonl(sidecar, rows)
    with pytest.raises(ac.AdjudicationError, match="sidecar was built from release"):
        ac.build_inventory(release, sidecar)

    # permissive direction: the sidecar the backfill actually produced joins
    fresh = _sample_sidecar(tmp_path, release)
    assert len(ac.build_inventory(release, fresh)) == 3


def test_a_sidecar_claim_absent_from_the_release_is_rejected(tmp_path):
    release, sidecar, _ = _inventory(tmp_path)
    rows = [json.loads(line) for line in sidecar.read_text().splitlines() if line]
    rows[0]["claims"].append(
        {"claim_id": "claim-deadbeefdeadbeefdeadbeef", "reconstructed": True}
    )
    _write_jsonl(sidecar, rows)

    with pytest.raises(ac.AdjudicationError, match="not in"):
        ac.build_inventory(release, sidecar)


# ---------------------------------------------------------------------------
# the backfill: what it reconstructs, and how honestly
# ---------------------------------------------------------------------------


@needs_index
def test_reconstruction_reproduces_the_recorded_evidence_keys_exactly():
    """The reconstruction is checked against what the run actually recorded."""
    rows, stats = bf.build_sidecar(RELEASE, CHUNKS)

    assert stats["cases"] == stats["cases_with_matching_evidence_keys"]
    assert stats["claims_not_reconstructed"] == 0
    assert stats["problems"] == {}
    for row in rows:
        assert row["evidence_keys_reconstructed"] == row["evidence_keys_recorded"]


@needs_index
def test_recorded_conflict_values_are_findable_in_the_reconstruction():
    """Every figure the recorded reason read OUT OF the evidence is still there.

    If the reconstruction had drifted, the verifier's own quoted evidence-side
    values would stop resolving; the answer-side values are expected to be
    missing, which is what a contradiction means.
    """
    _, stats = bf.build_sidecar(RELEASE, CHUNKS)
    found, total = (int(x) for x in stats["reason_evidence_side_confirmed"].split("/"))

    assert total >= 20
    assert found == total


def test_hit_text_is_recovered_from_the_index_by_doc_and_page(tmp_path):
    release = _write_jsonl(
        tmp_path / "release.jsonl",
        [_release_row("case-a", _failure(CLAIM_ONE))],
    )
    chunks = _write_jsonl(
        tmp_path / "chunks.jsonl",
        [
            {"doc_id": _doc("case-a"), "page": 1, "text": "Claim one is stated here."},
            {"doc_id": _doc("case-a"), "page": 9, "text": "An unrelated page."},
            {"doc_id": "other", "page": 1, "text": "A different document."},
        ],
    )
    rows, _ = bf.build_sidecar(release, chunks)

    (entry,) = rows[0]["evidence"]
    assert entry["key"] == f"{_doc('case-a')}|1"
    assert entry["text"] == "Claim one is stated here."
    assert entry["fidelity"] == "exact"
    assert entry["chunks_on_page"] == 1


def test_a_multi_chunk_page_is_marked_a_superset_and_never_excludes_retrieval(
    tmp_path,
):
    """The soundness rule: presence in a superset key may not rule anything out."""
    claim = "The programme is run by Ruritania."
    release = _write_jsonl(
        tmp_path / "release.jsonl",
        [_release_row("case-a", _failure(claim, reason="no citation"))],
    )

    two_chunks = [
        {"doc_id": _doc("case-a"), "page": 1, "text": "Unrelated preamble."},
        {"doc_id": _doc("case-a"), "page": 1, "text": "Implemented by Ruritania."},
    ]
    rows, _ = bf.build_sidecar(
        release, _write_jsonl(tmp_path / "chunks-two.jsonl", two_chunks)
    )
    reconstructed = rows[0]["claims"][0]
    assert rows[0]["evidence"][0]["fidelity"] == "page-superset"
    assert reconstructed["decision_inputs"]["presence_is_upper_bound"] is True
    assert "missing_retrieval_evidence" not in (
        reconstructed["decision_inputs"]["labels_excluded_by_evidence"]
    )

    # permissive direction: on a single-chunk page the same term IS exact, and
    # missing_retrieval_evidence is then correctly ruled out.
    one_chunk = [
        {"doc_id": _doc("case-a"), "page": 1, "text": "Implemented by Ruritania."}
    ]
    rows, _ = bf.build_sidecar(
        release, _write_jsonl(tmp_path / "chunks-one.jsonl", one_chunk)
    )
    reconstructed = rows[0]["claims"][0]
    assert rows[0]["evidence"][0]["fidelity"] == "exact"
    assert reconstructed["decision_inputs"]["presence_is_upper_bound"] is False
    assert "missing_retrieval_evidence" in (
        reconstructed["decision_inputs"]["labels_excluded_by_evidence"]
    )


def test_an_absent_term_is_reported_but_rules_out_nothing(tmp_path):
    """N3: absence is a signal about the PROBE, not a fact about the evidence.

    The old rule ("a term the probe cannot find could not have been cited, so
    ``missing_citation`` is impossible") is only as good as the probe, and the
    probe was wrong about every amount in release-1. Absence must therefore be
    reported and exclude nothing, in BOTH directions: an absent term leaves
    every label live, and a present term does too.
    """
    claim = "The programme is run by Freedonia."
    release = _write_jsonl(
        tmp_path / "release.jsonl", [_release_row("case-a", _failure(claim))]
    )
    chunks = _write_jsonl(
        tmp_path / "chunks.jsonl",
        [{"doc_id": _doc("case-a"), "page": 1, "text": "Implemented by Ruritania."}],
    )

    built, _ = bf.build_sidecar(release, chunks)
    inputs = built[0]["claims"][0]["decision_inputs"]
    assert inputs["terms_absent_from_held_evidence"] == ["Freedonia"]
    assert inputs["labels_excluded_by_evidence"] == {}
    assert "missing_citation" in inputs["labels_remaining"]
    assert "missing_retrieval_evidence" in inputs["labels_remaining"]

    # permissive direction: presence in EXACT evidence still rules the one
    # thing out that the reconstruction can actually vouch for.
    chunks = _write_jsonl(
        tmp_path / "chunks2.jsonl",
        [{"doc_id": _doc("case-a"), "page": 1, "text": "Implemented by Freedonia."}],
    )
    built, _ = bf.build_sidecar(release, chunks)
    inputs = built[0]["claims"][0]["decision_inputs"]
    assert inputs["terms_absent_from_held_evidence"] == []
    assert list(inputs["labels_excluded_by_evidence"]) == [
        "missing_retrieval_evidence"
    ]
    assert "missing_citation" in inputs["labels_remaining"]


def test_no_exported_row_claims_its_exclusions_are_sound(tmp_path):
    """The sentence printed in every row must not outrun what the probe knows."""
    claim = "The programme is run by Freedonia."
    release = _write_jsonl(
        tmp_path / "release.jsonl", [_release_row("case-a", _failure(claim))]
    )
    chunks = _write_jsonl(
        tmp_path / "chunks.jsonl",
        [{"doc_id": _doc("case-a"), "page": 1, "text": "Implemented by Ruritania."}],
    )
    built, _ = bf.build_sidecar(release, chunks)
    note = built[0]["claims"][0]["decision_inputs"]["note"]

    assert "exclusions are sound" not in note
    assert "REPORTED" in note and "STRUCTURAL" in note


# ---------------------------------------------------------------------------
# N3 — the presence probe, on the one dimension the finding turns on:
# the SEPARATOR CHARACTER, with the digits held fixed
# ---------------------------------------------------------------------------


def _probe_amount_against(tmp_path, name, answer_figure, page_text):
    """Probe ``answer_figure`` (as written in an answer) against ``page_text``."""
    claim = f"The project requests {answer_figure}."
    release = _write_jsonl(
        tmp_path / f"release-{name}.jsonl",
        [_release_row("case-a", _failure(claim, kind="money"))],
    )
    chunks = _write_jsonl(
        tmp_path / f"chunks-{name}.jsonl",
        [{"doc_id": _doc("case-a"), "page": 1, "text": page_text}],
    )
    built, _ = bf.build_sidecar(release, chunks)
    reconstructed = built[0]["claims"][0]
    (amount,) = [t for t in reconstructed["term_probe"] if t["term_kind"] == "amount"]
    return reconstructed, amount


@pytest.mark.parametrize("name, page_text, present, kind", [
    # THE FINDING: the same figure, comma where the answer writes a period.
    ("comma-decimal", "The total is USD 49,312 million.", True, "digits"),
    # same spelling as the answer — the case that always worked
    ("same-spelling", "The total is USD 49.312 million.", True, "literal"),
    # ADVERSARIAL: one digit different is a different figure, whatever the
    # separator. A relaxation that matched this would be worthless.
    ("other-figure", "The total is USD 49,313 million.", False, None),
    # ADVERSARIAL: the same leading digits inside a longer figure are not it.
    ("longer-figure", "The total is USD 49,312,817.", False, None),
    # ADVERSARIAL: the same DIGITS grouped differently are a different figure
    # 100x away — this is why the probe compares '49|312', not '49312'.
    ("regrouped", "The total is USD 4,931.2 million.", False, None),
    # ADVERSARIAL: the digits as a bare run, with no grouping at all.
    ("ungrouped", "The total is USD 49312 million.", False, None),
])
def test_a_comma_decimal_mark_is_found_and_a_different_figure_is_not(
    tmp_path, name, page_text, present, kind
):
    """Vary ONLY the separator character; hold the figure fixed.

    Prior rounds' probes varied the document and the wording but always wrote
    the figure the same way on both sides, so the corpus's comma decimal mark
    never appeared in a fixture at all. ``verify.amounts`` reads 'USD 49,312
    million' as a US thousands group whose scale word contradicts it, giving
    value=None / bare=49312.0 against the answer's 49312000.0 / 49.312 —
    neither ``value`` nor ``mantissa`` nor the literal fallback can bridge it,
    and the probe reported the figure ABSENT from a page that prints it.
    """
    _, amount = _probe_amount_against(
        tmp_path, name, "USD 49.312 million", page_text
    )

    assert amount["present_in_held_evidence"] is present
    if kind is None:
        assert amount["match_kinds"] == []
    else:
        assert kind in amount["match_kinds"]


def test_adjacent_figures_are_not_fused_into_one(tmp_path):
    """Why a space is not read as a grouping mark.

    The corpus prints 'Direct beneficiary sec. 5 155,000 direct beneficiaries'.
    Reading the space as a thousands separator would index that as the single
    figure 5|155|000 and DELETE 155,000 from the page — a false absence, which
    is the exact failure this probe exists to fix.
    """
    index = bf.number_index(
        bf.normalize("Direct beneficiary sec. 5 155,000 direct beneficiaries")
    )

    assert index == {"155|000": ["155,000"]}
    assert bf.figure_key("5 155,000") != "155|000"


def test_a_separator_variant_match_is_reported_but_never_strict(tmp_path):
    """A digits match must not buy an exclusion a literal match would.

    '49,312' and '49.312' are the same digits; whether they are the same
    MAGNITUDE depends on a convention the reconstruction cannot read. So the
    relaxation is allowed to report presence and must not be allowed to rule
    ``missing_retrieval_evidence`` out — even on a single-chunk page, where a
    literal hit on the same page does exactly that.
    """
    variant, amount = _probe_amount_against(
        tmp_path, "variant", "USD 49.312 million", "The total is USD 49,312 million."
    )
    assert variant["evidence_view"][0]["fidelity"] == "exact"
    assert amount["found_in_exact_keys"] == [f"{_doc('case-a')}|1"]
    assert amount["found_in_strict_keys"] == []
    assert amount["found_in_exact_strict_keys"] == []
    assert amount["separator_convention_differs"] is True
    assert amount["presence_is_upper_bound"] is True
    assert variant["decision_inputs"]["labels_excluded_by_evidence"] == {}

    # permissive direction: the SAME page, the SAME figure, written the answer's
    # way, is a strict hit and does rule missing_retrieval_evidence out.
    literal, amount = _probe_amount_against(
        tmp_path, "literal", "USD 49.312 million", "The total is USD 49.312 million."
    )
    assert amount["found_in_strict_keys"] == [f"{_doc('case-a')}|1"]
    assert amount["separator_convention_differs"] is False
    assert amount["presence_is_upper_bound"] is False
    assert list(literal["decision_inputs"]["labels_excluded_by_evidence"]) == [
        "missing_retrieval_evidence"
    ]


@pytest.mark.parametrize("page_text, present, kinds", [
    # THE FINDING: a hyphen in the answer, a space on the page.
    ("Held at the board meeting in Bonn.", True, ["hyphen-variant"]),
    ("Held at the Board-meeting in Bonn.", True, ["literal"]),
    # ADVERSARIAL: a hyphen reads as a space, not as nothing and not as a
    # licence to match a different phrase.
    ("Held at the boardmeeting in Bonn.", False, []),
    ("Held at the board of meetings in Bonn.", False, []),
])
def test_a_hyphen_is_read_as_a_space_but_nothing_else_is(
    tmp_path, page_text, present, kinds
):
    release = _write_jsonl(
        tmp_path / "release.jsonl",
        [_release_row("case-a", _failure("The Board-meeting decided it."))],
    )
    chunks = _write_jsonl(
        tmp_path / "chunks.jsonl",
        [{"doc_id": _doc("case-a"), "page": 1, "text": page_text}],
    )
    built, _ = bf.build_sidecar(release, chunks)

    probe = bf._probe_term(
        "Board-meeting",
        bf.EvidenceIndex(built[0]["evidence"]),
        set(),
        set(),
    )
    assert probe["present_in_held_evidence"] is present
    assert probe["match_kinds"] == kinds


@pytest.mark.parametrize("term, key", [
    # the two spellings of one figure collapse onto one key ...
    ("USD 49.312 million", "49|312"),
    ("USD 49,312 million", "49|312"),
    ("USD 6,876,446.27", "6|876|446|27"),
    # ... while a different GROUPING of the same digits does not
    ("USD 4,931.2 million", "4|931|2"),
    # no INTERNAL separator: a document code or a page pointer is not a figure,
    # and probing the corpus for a bare '172' would report presence from any
    # page number that happened to be 172.
    ("FP172", ""),
    ("B.30", ""),
    ("p.45", ""),
    ("2021", ""),
])
def test_only_internally_separated_figures_get_the_relaxed_probe(term, key):
    assert bf.separator_agnostic_key(term) == key


def test_only_the_note_blocks_that_reached_the_evidence_are_reconstructed(tmp_path):
    """``notes_used`` records ``board``; ``build_evidence`` never saw it."""
    doc = _doc("case-a")
    release = _write_jsonl(
        tmp_path / "release.jsonl",
        [
            _release_row(
                "case-a",
                _failure(CLAIM_ONE),
                notes_used={
                    "registry": f"Registry - {doc}: financing 5 M USD "
                                f"[{doc}, cover page]",
                    "board": "Board note that never entered the prompt evidence.",
                },
                claims={
                    "failures": [_failure(CLAIM_ONE)],
                    "evidence_keys": [f"{doc}|1", "__notes__|-", f"{doc}|-"],
                },
            )
        ],
    )
    chunks = _write_jsonl(
        tmp_path / "chunks.jsonl",
        [{"doc_id": doc, "page": 1, "text": "Claim one is stated here."}],
    )
    rows, _ = bf.build_sidecar(release, chunks)

    assert rows[0]["evidence_keys_match"] is True
    assert rows[0]["notes_in_evidence"] == ["registry"]
    assert rows[0]["notes_recorded_but_not_evidence"] == ["board"]
    joined = " ".join(entry["text"] for entry in rows[0]["evidence"])
    assert f"Registry - {doc}" in joined
    assert "never entered the prompt evidence" not in joined


def test_extraction_drift_is_reported_not_silently_absorbed(tmp_path):
    """A release whose recorded claim count no longer reproduces is flagged."""
    release = _write_jsonl(
        tmp_path / "release.jsonl", [_release_row("case-a", _failure(CLAIM_ONE))]
    )
    rows = [json.loads(line) for line in release.read_text().splitlines() if line]
    rows[0]["claims"]["claims"] = 99
    _write_jsonl(release, rows)
    chunks = _write_jsonl(
        tmp_path / "chunks.jsonl",
        [{"doc_id": _doc("case-a"), "page": 1, "text": "Claim one is stated here."}],
    )

    built, stats = bf.build_sidecar(release, chunks)
    assert built[0]["reconstruction"]["complete"] is False
    assert any("not faithful" in p for p in built[0]["reconstruction"]["problems"])
    assert stats["problems"]


@pytest.mark.parametrize("reason, expected", [
    # adversarial: a thousands separator must not split one figure into three
    (
        "not found in the cited evidence: USD 49,751,264",
        ["USD 49,751,264"],
    ),
    # permissive: a genuine list of targets must still split
    (
        "not found in the cited evidence: Antarctica, Ruritania",
        ["Antarctica", "Ruritania"],
    ),
])
def test_reason_targets_split_on_lists_but_not_inside_figures(reason, expected):
    targets = [
        t["term"] for t in bf.reason_targets(reason)
        if t["source"] == "not-found-target"
    ]
    assert targets == expected


def test_reason_targets_separate_the_evidence_side_from_the_answer_side():
    reason = (
        "cited evidence states '40,511,264 USD' for this field, "
        "the answer states 'USD 40,751,254'"
    )
    by_source = {t["source"]: t["term"] for t in bf.reason_targets(reason)}

    assert by_source["conflicting-value:evidence"] == "40,511,264 USD"
    assert by_source["conflicting-value:answer"] == "USD 40,751,254"


def test_backfill_refuses_to_write_over_its_inputs(tmp_path):
    release, _ = _sample_release(tmp_path)
    chunks = _write_jsonl(tmp_path / "chunks.jsonl", [])
    before = release.read_bytes()

    with pytest.raises(bf.BackfillError, match="over its own input"):
        bf.assert_writable_output(release, inputs=(release, chunks), force=True)
    with pytest.raises(bf.BackfillError, match="over its own input"):
        bf.assert_writable_output(chunks, inputs=(release, chunks), force=True)
    assert release.read_bytes() == before

    # permissive: a genuinely new path is writable
    bf.assert_writable_output(
        tmp_path / "sidecar.jsonl", inputs=(release, chunks)
    )


@needs_release
def test_backfill_refuses_to_write_over_a_checksummed_anchor(tmp_path):
    anchors = bf.protected_paths()
    assert RELEASE.resolve() in anchors

    with pytest.raises(bf.BackfillError, match="measurement anchor"):
        bf.assert_writable_output(
            RELEASE, inputs=(tmp_path / "other.jsonl",), force=True
        )


def test_sidecar_and_inventory_are_deterministic(tmp_path):
    release, sidecar, first = _inventory(tmp_path)
    second = ac.build_inventory(release, sidecar)

    assert first == second
    rows_a, _ = bf.build_sidecar(release, tmp_path / "chunks.jsonl")
    rows_b, _ = bf.build_sidecar(release, tmp_path / "chunks.jsonl")
    assert rows_a == rows_b


def test_excerpt_is_centred_on_the_match_and_falls_back_to_the_head():
    text = ("filler " * 200) + "the Amazon Bioeconomy Fund " + ("filler " * 200)

    centred = bf.build_excerpt(text, ["Amazon Bioeconomy Fund"])
    assert "Amazon Bioeconomy Fund" in centred
    assert len(centred) < len(text)

    head = bf.build_excerpt(text, ["nothing here"])
    assert "Amazon Bioeconomy Fund" not in head
    assert head.startswith("filler")
    assert head, "an unmatched page must still show its head, not nothing"

    # a page shorter than the head budget is shown whole
    assert bf.build_excerpt("A short page.", []) == "A short page."


def test_an_unmatched_uncited_page_still_shows_text_and_a_cited_page_shows_more(
    tmp_path,
):
    """N14, on the dimension it turns on: whether the page matched anything.

    Three pages, one claim: the cited page, a page that matches a probed term,
    and a page that matches nothing and is cited by nothing. The old policy
    emitted ``excerpt: null`` for the third, which is the entire signal for
    ``missing_retrieval_evidence`` — 'the turn held these pages and none of
    them says it'.
    """
    doc = _doc("case-a")
    claim = f"Claim one is implemented by the Ruritania Agency. [{doc}, p.1]"
    release = _write_jsonl(
        tmp_path / "release.jsonl",
        [
            _release_row(
                "case-a",
                _failure(claim),
                claims={
                    "failures": [_failure(claim)],
                    "evidence_keys": [f"{doc}|1", f"{doc}|2", f"{doc}|3"],
                },
                hits=[
                    {"doc": doc, "page": 1, "score": 0.9},
                    {"doc": doc, "page": 2, "score": 0.8},
                    {"doc": doc, "page": 3, "score": 0.7},
                ],
            )
        ],
    )
    chunks = _write_jsonl(
        tmp_path / "chunks.jsonl",
        [
            {"doc_id": doc, "page": 1, "text": "Cited page. " + ("body " * 400)},
            {"doc_id": doc, "page": 2, "text": ("pad " * 300) + "the Ruritania Agency"},
            {"doc_id": doc, "page": 3, "text": "Wholly unrelated. " + ("x " * 400)},
        ],
    )
    rows, stats = bf.build_sidecar(release, chunks)
    view = {e["key"]: e for e in rows[0]["claims"][0]["evidence_view"]}

    assert all(entry["excerpt"] for entry in view.values())
    assert view[f"{doc}|3"]["matched_terms"] == []
    assert view[f"{doc}|3"]["cited_by_claim"] is False
    assert view[f"{doc}|3"]["excerpt"].startswith("Wholly unrelated.")

    # the cited page is the thing under review, so it is shown whole
    assert view[f"{doc}|1"]["cited_by_claim"] is True
    assert view[f"{doc}|1"]["excerpt_is_full_text"] is True
    # the matched page is centred on the match, not merely headed
    assert "Ruritania Agency" in view[f"{doc}|2"]["excerpt"]
    assert view[f"{doc}|2"]["excerpt_is_full_text"] is False

    assert stats["evidence_view_entries_without_text"] == 0
    assert stats["claims_with_no_visible_evidence_text"] == 0


def test_a_claim_with_nothing_probeable_still_shows_its_pages(tmp_path):
    """The `n_terms == 0` row: no term to centre on, so a larger head instead.

    These were the rows most likely to be blank — nothing matched, so nothing
    was shown — and they are exactly the rows where the reviewer has only the
    evidence text to go on.
    """
    doc = _doc("case-a")
    # the shape release-1 actually has six of: a colon lead-in with no amount,
    # year or named entity in it (this is bc-b42-02-add16's, de-specified)
    claim = "There are inconsistent printed figures for the funding requested:"
    release = _write_jsonl(
        tmp_path / "release.jsonl", [_release_row("case-a", _failure(claim))]
    )
    chunks = _write_jsonl(
        tmp_path / "chunks.jsonl",
        [{"doc_id": doc, "page": 1, "text": "Page body. " + ("word " * 500)}],
    )
    rows, _ = bf.build_sidecar(release, chunks)
    (reconstructed,) = rows[0]["claims"]

    assert reconstructed["reconstructed"] is True
    assert reconstructed["decision_inputs"]["n_terms"] == 0
    assert reconstructed["decision_inputs"]["is_lead_in_to_a_list"] is True
    (entry,) = reconstructed["evidence_view"]
    assert entry["matched_terms"] == []
    assert entry["excerpt"].startswith("Page body.")
    assert entry["excerpt_chars"] > bf.HEAD_EXCERPT_CHARS


def test_export_refuses_an_evidence_block_with_no_readable_text(tmp_path):
    """The gate: keys and probes are not evidence, text is."""
    release, sidecar, expected = _inventory(tmp_path)
    output = tmp_path / "inventory.jsonl"

    blanked = [json.loads(line) for line in sidecar.read_text().splitlines() if line]
    for row in blanked:
        for claim in row["claims"]:
            for entry in claim.get("evidence_view") or ():
                entry["excerpt"] = None
    _write_jsonl(sidecar, blanked)

    assert ac.main([
        "export", "--release", str(release), "--evidence", str(sidecar),
        "--output", str(output),
    ]) == 2
    assert not output.exists()

    # permissive: the sidecar the backfill actually produces exports cleanly,
    # and reports how much evidence text a reviewer is getting
    fresh = _sample_sidecar(tmp_path, release)
    assert ac.main([
        "export", "--release", str(release), "--evidence", str(fresh),
        "--output", str(output),
    ]) == 0
    coverage = ac.evidence_coverage(ac.read_inventory(output))
    assert coverage["without_evidence_text"] == 0
    assert coverage["evidence_text_chars"] > 0


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def test_cli_export_validate_import_round_trip(tmp_path, capsys):
    release, sidecar, expected = _inventory(tmp_path)
    inventory_path = tmp_path / "inventory.jsonl"

    assert ac.main([
        "export", "--release", str(release), "--evidence", str(sidecar),
        "--output", str(inventory_path),
    ]) == 0
    assert "71" not in capsys.readouterr().out

    assert ac.main([
        "summary", "--release", str(release), "--evidence", str(sidecar),
        "--inventory", str(inventory_path),
    ]) == 2
    assert "unreviewed claim(s)" in capsys.readouterr().err

    assert ac.main([
        "summary", "--release", str(release), "--evidence", str(sidecar),
        "--inventory", str(inventory_path), "--allow-unreviewed",
    ]) == 0
    assert '"unreviewed": 3' in capsys.readouterr().out

    _write_jsonl(inventory_path, _complete(ac.read_inventory(inventory_path)))
    assert ac.main([
        "import", "--release", str(release), "--evidence", str(sidecar),
        "--inventory", str(inventory_path), "--output", str(tmp_path / "final.jsonl"),
    ]) == 0
    final = ac.read_inventory(tmp_path / "final.jsonl")
    assert all(row["label"] is not None for row in final)


def test_validate_points_at_the_missing_sidecar_instead_of_drowning_in_diffs(
    tmp_path, capsys
):
    release, sidecar, expected = _inventory(tmp_path)
    inventory_path = _write_jsonl(tmp_path / "inventory.jsonl", _complete(expected))

    assert ac.main([
        "validate", "--release", str(release), "--inventory", str(inventory_path),
    ]) == 2
    assert "same sidecar" in capsys.readouterr().err

    assert ac.main([
        "validate", "--release", str(release), "--evidence", str(sidecar),
        "--inventory", str(inventory_path),
    ]) == 0
