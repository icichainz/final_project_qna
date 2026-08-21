"""Self-tests for scripts/judge_audit.py — the LLM-judge promotion audit.

The audit reconstructs what ``verify.adjudicate`` accepted in an ALREADY
RECORDED run by set-differencing the deterministic verdicts against the run's
recorded failure list. Nothing here calls a model, touches the network or
loads the FAISS index: every fixture is a hand-built record dict.

The suite is paired throughout — for each thing the audit must FIND there is a
corruption of the same fixture it must REFUSE (report UNDECIDABLE for) rather
than answer wrongly. A reconstruction that silently guesses is worse than one
that abstains, because the number it prints is the one a switch decision gets
made on.
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import judge_audit as ja                                       # noqa: E402
from gcf_qna.rag import verify                                 # noqa: E402

PARITY = ROOT / "data" / "eval" / "release_parity-baseline.jsonl"
ADJUDICATION = ROOT / "data" / "eval" / "judge_audit_adjudication_parity-baseline.jsonl"

# Fake document ids: they parse as ids but are in no registry, so the
# fixtures exercise the matcher alone and cannot be rescued by a registry row.
DOC = "999_test-doc-a"
OTHER = "998_test-doc-b"
VERIFY_SHA = hashlib.sha256(ja.VERIFY_PY.read_bytes()).hexdigest()


# ---------------------------------------------------------------- fixtures --
def _record(answer, hits, claims_block, notes=None, case="c1", cls="identifier"):
    """A minimal release record shaped like the ones eval_answers writes."""
    return {
        "id": case, "class": cls, "lang": "en", "question": "q?",
        "raw_answer": answer, "answer": answer,
        "hits": [{"doc": d, "page": p, "score": 0.5, "text": t}
                 for d, p, t in hits],
        "notes_used": notes or {},
        "claims": claims_block,
        "verify_blob_sha": VERIFY_SHA,
        "usage": {"calls": [
            {"role": "answer", "prompt_tokens": 100, "completion_tokens": 10,
             "latency_s": 1.0},
            {"role": "judge", "prompt_tokens": 400, "completion_tokens": 40,
             "latency_s": 2.0}], "turn_latency_s": 3.0},
    }


def _block(record_stub, **over):
    """The claims block production would have written for `record_stub`."""
    ev = ja.evidence_for(record_stub)
    claims = verify.extract_claims(record_stub["raw_answer"])
    out = {"claims": len(claims), "supported": len(claims), "contradicted": 0,
           "unsupported": 0, "grounded": 0, "citation_supported": 0, "cited": 0,
           "judge_promotions": 0, "n_failures": 0, "failures": [],
           "evidence_keys": ja._evidence_keys(ev), "verify_status": "verified",
           "verifier_mode": "production"}
    out.update(over)
    return out


@pytest.fixture
def rescue_record():
    """One claim the judge promoted, and the fact IS in the held evidence.

    The claim cites p.5 of one document; the figure is printed in a SECOND
    held document. The deterministic pass fails it inside the cited scope (and
    inside the cited document, so the citation-page-mismatch branch cannot
    save it), while the union of the turn's evidence entails it — exactly a
    RESCUE.
    """
    answer = f"FP151 requests **USD 18.5 million** [{DOC}, p.5]."
    hits = [(DOC, 5, "A.8 Requested amount: see the financing annex."),
            (OTHER, 1, "The GCF grant is USD 18,500,000 in total.")]
    stub = _record(answer, hits, {})
    stub["claims"] = _block(stub, judge_promotions=1)
    return stub


@pytest.fixture
def fabrication_record():
    """One promoted claim whose name appears in NO held evidence."""
    answer = f"FP151 is implemented by **Banco Fantasma** [{DOC}, p.5]."
    hits = [(DOC, 5, "A.8 Requested amount: 18.5 M USD. Accredited entity: IUCN."),
            (OTHER, 1, "The GCF grant is USD 18,500,000 in total.")]
    stub = _record(answer, hits, {})
    stub["claims"] = _block(stub, judge_promotions=1)
    return stub


# ------------------------------------------------------------- replay core --
def test_evidence_uses_productions_note_order_not_notes_used(rescue_record):
    """A `board` note is recorded but never handed to build_evidence.

    `run_case` passes [registry, year, matrix]. Reading `notes_used.values()`
    instead would build an evidence set the turn did not hold, and every
    verdict downstream would be against evidence production never saw.
    """
    rec = copy.deepcopy(rescue_record)
    rec["notes_used"] = {"registry": "Registry — FP151: entity IUCN.",
                         "board": "Note: B.99 is not in this corpus."}
    keys = ja.evidence_for(rec)
    blob = verify._text_of(keys, list(keys))
    assert "Registry — FP151" in blob
    assert "B.99" not in blob, "a board note reached the evidence; production's is [registry, year, matrix]"


def test_promotion_is_reconstructed_from_the_set_difference(rescue_record):
    got = ja.replay_case(rescue_record)
    assert got["gates"] == []
    promoted = [r for r in got["rows"] if r["verdict_source"] == "llm-promoted"]
    assert len(promoted) == 1
    assert promoted[0]["deterministic_status"] == verify.UNSUPPORTED
    assert promoted[0]["production_status"] == verify.SUPPORTED


def test_a_claim_the_judge_left_failing_is_not_a_promotion(rescue_record):
    """The judge answered and said 'still unsupported' — that is not a promotion."""
    rec = copy.deepcopy(rescue_record)
    claims = verify.extract_claims(rec["raw_answer"])
    rec["claims"].update(
        judge_promotions=0, supported=0, unsupported=1, n_failures=1,
        failures=[{"status": verify.UNSUPPORTED, "kind": claims[0].kind,
                   "text": claims[0].text[:160], "reason": "judge verdict",
                   "cited": True, "grounded": True, "source": "llm"}])
    got = ja.replay_case(rec)
    assert got["gates"] == []
    assert [r for r in got["rows"] if r["verdict_source"] == "llm-promoted"] == []
    assert [r["verdict_source"] for r in got["rows"]] == ["llm"]


# ----------------------------------------------------------- classification --
def test_grounded_promotion_classifies_as_rescue(rescue_record):
    result = ja.audit([rescue_record])
    assert [p["verdict"] for p in result["promotions"]] == [ja.RESCUE]
    assert result["promotions"][0]["grounded"] is True


def test_ungrounded_promotion_classifies_as_fabrication_pass(fabrication_record):
    result = ja.audit([fabrication_record])
    p, = result["promotions"]
    assert p["verdict"] == ja.FABRICATION
    assert p["grounded"] is False
    assert "Banco Fantasma" in p["elements_missing"], p["elements_missing"]


def test_fabrication_row_reports_the_exact_unmatched_substrings(fabrication_record):
    """'the matcher says no' and 'the evidence does not say it' differ.

    The residue list is what lets a human tell a real fabrication from a
    cross-lingual or acronym miss, so it must never be empty on a
    FABRICATION-PASS row.
    """
    p, = ja.audit([fabrication_record])["promotions"]
    assert p["elements_missing"], "a FABRICATION-PASS with no named residue is unreviewable"
    assert p["why"].startswith("not matched anywhere in the held evidence")


# ------------------------------------------------------------------- gates --
def test_stale_verify_py_makes_every_promotion_undecidable(rescue_record):
    rec = copy.deepcopy(rescue_record)
    rec["verify_blob_sha"] = "0" * 64
    result = ja.audit([rec])
    assert [p["verdict"] for p in result["promotions"]] == [ja.UNDECIDABLE]
    assert "verify_blob_sha" in result["promotions"][0]["why"]
    assert result["totals"]["verify_sha_matches"] is False


def test_truncated_failure_list_is_a_gate_not_a_guess(rescue_record):
    """Without --release the list is capped at 6, so the set difference lies."""
    rec = copy.deepcopy(rescue_record)
    rec["claims"]["n_failures"] = 9          # 9 failures happened, 0 were kept
    result = ja.audit([rec])
    assert [p["verdict"] for p in result["promotions"]] == [ja.UNDECIDABLE]
    assert "truncated failure list" in result["promotions"][0]["why"]


def test_claim_count_drift_is_a_gate(rescue_record):
    rec = copy.deepcopy(rescue_record)
    rec["claims"]["claims"] = 99
    got = ja.replay_case(rec)
    assert any("claim count" in g for g in got["gates"])


def test_evidence_key_drift_is_a_gate(rescue_record):
    rec = copy.deepcopy(rescue_record)
    rec["claims"]["evidence_keys"] = ["some_other_doc|1"]
    got = ja.replay_case(rec)
    assert any("evidence keys" in g for g in got["gates"])


def test_promotion_count_disagreeing_with_the_record_is_a_gate(rescue_record):
    rec = copy.deepcopy(rescue_record)
    rec["claims"]["judge_promotions"] = 7
    got = ja.replay_case(rec)
    assert any("reconstructed promotions" in g for g in got["gates"])


def test_unmatched_recorded_failure_is_a_gate(rescue_record):
    """A recorded failure the replay cannot place means the inputs moved."""
    rec = copy.deepcopy(rescue_record)
    rec["claims"].update(
        judge_promotions=1, n_failures=1,
        failures=[{"status": verify.UNSUPPORTED, "kind": "money",
                   "text": "a claim this answer never made", "reason": "x",
                   "cited": True, "grounded": False, "source": "llm"}])
    got = ja.replay_case(rec)
    assert any("no replayed counterpart" in g for g in got["gates"])


def test_pre_f7_record_without_hit_text_is_a_gate(rescue_record):
    rec = copy.deepcopy(rescue_record)
    for h in rec["hits"]:
        h.pop("text")
    got = ja.replay_case(rec)
    assert any("no hit text" in g for g in got["gates"])


def test_guard_and_chat_turns_are_skipped_not_scored(rescue_record):
    rec = copy.deepcopy(rescue_record)
    rec["claims"] = None
    rec["claims_skipped"] = {"reason": "guard-answer: production returns before "
                                       "verification"}
    result = ja.audit([rec])
    assert result["totals"]["cases_verified"] == 0
    assert result["totals"]["cases_skipped"] == 1
    assert result["promotions"] == []


# ------------------------------------------------------------ counterfactual --
def test_counterfactual_flips_verified_to_abstain_when_the_judge_carried_it(
        rescue_record):
    """One required claim, supported only by the judge: judge off -> abstain."""
    result = ja.audit([rescue_record])
    cf = ja.counterfactual(result, [rescue_record])
    assert result["totals"]["supported_judge_on"] == 1
    assert result["totals"]["supported_judge_off"] == 0
    assert [(f["status_on"], f["status_off"]) for f in cf["flips"]] == \
        [("verified", "abstain")]


def test_counterfactual_reports_no_flip_when_the_judge_changed_nothing():
    """A turn the deterministic pass already cleared must not appear."""
    answer = f"FP151 requests **USD 18.5 million** [{DOC}, p.5]."
    hits = [(DOC, 5, "A.8 Requested amount: 18.5 M USD.")]
    rec = _record(answer, hits, {})
    rec["claims"] = _block(rec)
    result = ja.audit([rec])
    assert result["promotions"] == []
    assert result["totals"]["supported_judge_on"] == \
        result["totals"]["supported_judge_off"] == 1
    assert ja.counterfactual(result, [rec])["flips"] == []


# -------------------------------------------------------------- accounting --
def test_cost_uses_the_harness_rate_constant_and_never_a_local_copy():
    """Restating the rate here is how two reports drift apart."""
    import eval_answers as ev
    src = ja.VERIFY_PY.parent  # noqa: F841  (kept: locates the tree in failures)
    text = Path(ja.__file__).read_text(encoding="utf-8")
    assert "TOKEN_COST_USD" in text
    assert "ev.TOKEN_COST_USD" in text, "the audit must import the rate, not restate it"
    rec = _record("no claims here.", [], {"claims": 0})
    rec["claims"] = None
    usage = ja.judge_usage([rec])
    assert usage["judge"]["calls"] == 1
    assert usage["judge"]["cost_usd"] == round(
        400 * ev.TOKEN_COST_USD["prompt"] + 40 * ev.TOKEN_COST_USD["completion"], 4)


def test_judge_usage_separates_the_judge_from_the_answer_call(rescue_record):
    usage = ja.judge_usage([rescue_record])
    assert usage["judge"]["prompt_tokens"] == 400
    assert usage["answer"]["prompt_tokens"] == 100
    assert usage["judge"]["latency_s"] == 2.0


# ------------------------------------------------------------ adjudications --
def test_adjudications_annotate_but_never_overwrite_the_machine_verdict(
        fabrication_record, tmp_path):
    p = tmp_path / "adj.jsonl"
    key = ja.claim_key("c1", verify.extract_claims(
        fabrication_record["raw_answer"])[0].text)
    p.write_text(json.dumps({"key": key, "verdict": "MATCHER-ARTIFACT",
                             "note": "reviewed"}) + "\n", encoding="utf-8")
    result = ja.audit([fabrication_record], ja.load_adjudications(p))
    row, = result["promotions"]
    assert row["verdict"] == ja.FABRICATION, "the machine column must survive review"
    assert row["adjudicated"] == "MATCHER-ARTIFACT"
    assert row["adjudication_note"] == "reviewed"


def test_claim_key_is_stable_and_case_scoped():
    a = ja.claim_key("case-a", "some claim")
    assert a == ja.claim_key("case-a", "some claim")
    assert a != ja.claim_key("case-b", "some claim")
    assert a != ja.claim_key("case-a", "some other claim")


# ------------------------------------------------- the recorded parity run --
@pytest.mark.skipif(not PARITY.exists(), reason="parity baseline not in this checkout")
def test_parity_baseline_reconstructs_exactly():
    """The finding this script exists to support, pinned as a regression.

    28 promotions over 21 cases, no case tripping a gate. If verify.py or the
    recording moves, this fails loudly instead of quietly reporting a
    different table.
    """
    records = ja.load_run(PARITY)
    result = ja.audit(records)
    t = result["totals"]
    assert t["verify_sha_matches"] is True
    assert [c["case"] for c in result["cases"] if c["gates"]] == []
    assert t["promotions"] == 28 and t["promotion_cases"] == 21
    assert t["claims"] == 154
    assert t["supported_judge_on"] == 140
    assert t["supported_judge_off"] == 112
    assert t["grounded"] == 139
    # every recorded per-case count is reproduced, not just the run total
    for case in result["cases"]:
        promoted = sum(1 for r in case["rows"] if r["verdict_source"] == "llm-promoted")
        assert promoted == case["recorded"]["judge_promotions"], case["case"]


@pytest.mark.skipif(not PARITY.exists(), reason="parity baseline not in this checkout")
def test_parity_baseline_promotion_split_and_supported_over_grounded():
    records = ja.load_run(PARITY)
    result = ja.audit(records)
    verdicts = {}
    for p in result["promotions"]:
        verdicts[p["verdict"]] = verdicts.get(p["verdict"], 0) + 1
    assert verdicts == {ja.RESCUE: 17, ja.FABRICATION: 11}
    S = verify.SUPPORTED
    supported_ungrounded = [r for r in result["rows"]
                            if r["production_status"] == S and not r["grounded"]]
    # 11 judge promotions + 2 deterministic registry-backed claims
    assert len(supported_ungrounded) == 13
    assert sum(1 for r in supported_ungrounded
               if r["verdict_source"] == "llm-promoted") == 11
    assert result["totals"]["supported_judge_on"] > result["totals"]["grounded"]


@pytest.mark.skipif(not (PARITY.exists() and ADJUDICATION.exists()),
                    reason="recorded artefacts not in this checkout")
def test_every_fabrication_pass_carries_a_human_adjudication():
    """The 11 machine FABRICATION-PASS rows are all reviewed on file.

    A FABRICATION-PASS is a statement about the MATCHER; leaving one
    unreviewed in the recorded artefact would let it be read as a finding of
    fabrication.
    """
    records = ja.load_run(PARITY)
    result = ja.audit(records, ja.load_adjudications(ADJUDICATION))
    unreviewed = [p["key"] for p in result["promotions"]
                  if p["verdict"] == ja.FABRICATION and not p["adjudicated"]]
    assert unreviewed == []
