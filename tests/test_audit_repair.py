"""Fixture tests for scripts/audit_repair.py — the Wave 4 repair A/B.

NO TEST HERE MAKES AN API CALL. Every judge and repair completion is served
by a stub client whose text is written in the test, and ``test_zero_api``
proves it: ``verify._client`` is replaced by a function that raises, so any
path that reached for a real client would fail the suite rather than bill it.

The properties pinned, in the order the plan asks for them:

  * the OFF arm is byte-identical to ``verify.verify_answer(allow_repair=0)``
    on a fixed fixture (standing rule 3: fallback proofs, not assertions);
  * one judge call and one repair call per case, shared between the arms —
    the whole methodological point;
  * the carry-off scoring re-uses the SAME repaired text, and can disagree
    with carry-on about adoption (an instrument that cannot see the
    difference is the Wave-2 blind-spot failure repeated);
  * every invented-source shape is DETECTED, one test per shape, and the
    negative case (a figure that really is in the evidence) is not;
  * a lost supported claim gates, and a documented one does not;
  * corrected / deleted / qualified are three outcomes, not one.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import audit_repair as ar                                        # noqa: E402
from gcf_qna.rag import verify                                   # noqa: E402


# ---------------------------------------------------------------------------
# stub client
# ---------------------------------------------------------------------------
class _Usage:
    prompt_tokens = 100
    completion_tokens = 20
    total_tokens = 120


class _Resp:
    model = "stub-snapshot"

    def __init__(self, content):
        self.choices = [ar._Choice(content)]
        self.usage = _Usage()


class _Completions:
    def __init__(self, owner):
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._create(**kwargs)


class _Chat:
    def __init__(self, owner):
        self.completions = _Completions(owner)


class StubClient:
    """Serves canned judge/repair completions and counts what was asked."""

    def __init__(self, judge=None, repair=None):
        self.chat = _Chat(self)
        self.roles = []
        self._judge = judge
        self._repair = repair

    def _create(self, **kwargs):
        role = ar._call_role(kwargs)
        self.roles.append(role)
        if role == "judge":
            if self._judge is None:
                raise AssertionError("judge called but no judge reply staged")
            return _Resp(json.dumps(self._judge))
        if role == "repair":
            if self._repair is None:
                raise AssertionError("repair called but no repair reply staged")
            return _Resp(self._repair)
        raise AssertionError(f"unexpected call role {role}")


class ExplodingClient:
    """Any call at all is a test failure."""

    def __init__(self):
        self.chat = _Chat(self)

    def _create(self, **kwargs):                                 # pragma: no cover
        raise AssertionError("an API call was attempted")


# ---------------------------------------------------------------------------
# fixtures: one document, one page, and answers over it
# ---------------------------------------------------------------------------
DOC = "124_gcf-b27-02-add11"
OTHER_DOC = "125_gcf-b30-02-add03"
PAGE = 45

EVIDENCE = (
    "# FINANCING INFORMATION\n"
    "## C.1 Total amount and currency\n"
    "### (a) Requested GCF funding\n"
    "Requested GCF funding: USD 18,500,000\n"
    "Total project financing: USD 50,000,000\n"
    "Number of countries: 9\n"
    "Accredited Entity: Global Green Growth Institute\n")

OTHER_EVIDENCE = (
    "# PROJECT SUMMARY\n"
    "Requested GCF funding: USD 42,000,000\n"
    "Accredited Entity: Asian Development Bank\n")


def record(answer, hits=None, case_id="t-case", lang="en", expect=None,
           **extra):
    hits = hits if hits is not None else [
        {"doc": DOC, "page": PAGE, "score": 0.9, "text": EVIDENCE}]
    rec = {
        "id": case_id, "class": "identifier", "lang": lang,
        "question": "What is the total GCF funding requested?",
        "turns": 0, "mode": "answers", "guard": False, "chat": False,
        "raw_answer": answer, "answer": answer, "answer_source": "model",
        "hits": hits, "notes_used": {},
        "expect": expect or {"behavior": "answer", "docs": [DOC],
                             "must_contain": [], "must_not_contain": []},
        "score": 1.0,
    }
    rec.update(extra)
    return rec


def evidence_of(rec):
    return ar.evidence_for(rec)


def replay_to(tmp_path, records, client, **kw):
    src = tmp_path / "release_fixture.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    got = ar.replay(src, tmp_path / "replay", client=client, **kw)
    arms = {a: {r["id"]: r for r in ar.read_jsonl(got["paths"][a])}
            for a in ar.ARMS}
    return got, arms


# ---------------------------------------------------------------------------
# the instrument agrees with the release harness's own scorers
# ---------------------------------------------------------------------------
def test_grounded_flags_matches_eval_answers():
    """The audit must not carry a second groundedness definition."""
    import eval_answers as ev
    rec = record(f"FP151 requests **USD 18.5 million** in GCF funding "
                 f"[{DOC}, p. {PAGE}]. It also names Global Green Growth "
                 f"Institute [{DOC}, p. {PAGE}].")
    ev_ = evidence_of(rec)
    claims = verify.extract_claims(rec["raw_answer"])
    assert claims
    assert ar._grounded_flags(claims, ev_) == ev.grounded_flags(claims, ev_)


def test_claims_block_matches_claim_metrics():
    """An arm row's `claims` block is the release record's block, key for key."""
    import eval_answers as ev
    rec = record(f"FP151 requests **USD 18.5 million** in GCF funding "
                 f"[{DOC}, p. {PAGE}].")
    ev_ = evidence_of(rec)
    claims = verify.extract_claims(rec["raw_answer"])
    verdicts = verify.classify_deterministic(claims, ev_)
    mine = ar._claims_block(verdicts, ev_)
    theirs = ev.claim_metrics(claims, verdicts, ev_, full_failures=True)
    for key in theirs:
        assert mine[key] == theirs[key], key


def test_nd_threshold_is_the_exact_integer():
    """`>= 95%` is a count, not a rounded percentage (metric contract 5)."""
    assert ar._nd(154, 154)["threshold_95"] == 147     # ceil(0.95 * 154)
    assert ar._nd(0, 100)["threshold_95"] == 95
    assert ar._nd(0, 0)["rate"] is None


# ---------------------------------------------------------------------------
# the replay: what differs between the arms, and what does not
# ---------------------------------------------------------------------------
def test_off_arm_is_byte_identical_to_verify_answer_with_repair_off(tmp_path):
    """Standing rule 3: prove the off-path equals the pre-change path."""
    answer = (f"FP151 requests **USD 18.5 million** in GCF funding "
              f"[{DOC}, p. {PAGE}]. The total project cost is "
              f"**USD 99,900,000** [{DOC}, p. {PAGE}].")
    rec = record(answer)
    ev_ = evidence_of(rec)
    direct = verify.verify_answer(answer, ev_, client=StubClient(
        judge={"verdicts": []}), use_llm=True, allow_repair=False)

    _got, arms = replay_to(tmp_path, [rec],
                           StubClient(judge={"verdicts": []}, repair=answer))
    row = arms["off"]["t-case"]
    assert row["answer"] == direct.answer == answer
    assert row["verify_status"] == direct.status
    assert [c["status"] for c in row["claim_rows"]] == [v.status for v in direct.verdicts]
    assert row["repaired"] is False and row["repair_rejected"] is False


def test_one_judge_call_and_one_repair_call_shared(tmp_path):
    """The judge sample is COMMON to both arms; only repair differs."""
    answer = (f"FP151 requests **USD 18.5 million** [{DOC}, p. {PAGE}]. "
              f"The total project cost is **USD 99,900,000** [{DOC}, p. {PAGE}].")
    fixed = f"FP151 requests **USD 18.5 million** [{DOC}, p. {PAGE}]."
    client = StubClient(judge={"verdicts": []}, repair=fixed)
    got, arms = replay_to(tmp_path, [record(answer)], client)
    assert client.roles.count("repair") == 1, client.roles
    assert client.roles.count("judge") <= 1, client.roles
    on = arms["on"]["t-case"]["repair_replay"]
    co = arms["on-carryoff"]["t-case"]["repair_replay"]
    # the carry-off scoring made NO extra call and scored the SAME text
    assert on["repair_text_sha256"] == co["repair_text_sha256"]
    assert on["repair_calls"] == co["repair_calls"] == 1
    assert got["summary"]["cases"] == 1


def test_no_failures_means_no_repair_call(tmp_path):
    """A clean answer must not reach the repair model at all."""
    answer = f"FP151 requests **USD 18.5 million** in GCF funding [{DOC}, p. {PAGE}]."
    client = StubClient(judge={"verdicts": []})       # no repair reply staged
    _got, arms = replay_to(tmp_path, [record(answer)], client)
    assert "repair" not in client.roles
    for arm in ar.ARMS:
        assert arms[arm]["t-case"]["answer"] == answer
        assert arms[arm]["t-case"]["verify_status"] == "verified"


def test_guard_and_error_rows_are_carried_through_every_arm(tmp_path):
    """Production returns before verification on a guard turn; so must the
    replay, in all three arms, so the rows cancel instead of moving a rate."""
    rows = [record("guarded", case_id="g", guard=True),
            record("chatty", case_id="c", chat=True),
            dict(record("x", case_id="e"), error="boom")]
    _got, arms = replay_to(tmp_path, rows, ExplodingClient())
    for arm in ar.ARMS:
        for cid in ("g", "c", "e"):
            assert arms[arm][cid]["repair_replay"]["replayed"] is False
            assert arms[arm][cid]["claim_rows"] == []


def test_carry_off_context_manager_restores_the_binding():
    saved = verify._carry_cleared
    with ar.no_carry_cleared():
        assert verify._carry_cleared is not saved
        assert verify._carry_cleared([1, 2], ["ignored"]) == [1, 2]
    assert verify._carry_cleared is saved


def test_carry_off_can_reject_what_carry_on_adopts(tmp_path):
    """The number the plan GATES on must be able to differ from the one
    production ships. A scoring that cannot disagree is not a second opinion.

    Setup: the judge clears a claim the deterministic matcher fails, and the
    repair leaves that sentence untouched. `_carry_cleared` then re-clears it
    after the rewrite (carry-on adopts); with the carry disabled the recheck
    still fails and the repair is rejected (carry-off keeps the original).
    """
    bad = f"The programme covers **12 countries** [{DOC}, p. {PAGE}]."
    answer = (f"FP151 requests **USD 18.5 million** [{DOC}, p. {PAGE}]. {bad} "
              f"The total project cost is **USD 99,900,000** [{DOC}, p. {PAGE}].")
    # the repair drops only the 99.9m sentence and keeps the judge-cleared one
    repaired = f"FP151 requests **USD 18.5 million** [{DOC}, p. {PAGE}]. {bad}"

    ev_ = evidence_of(record(answer))
    det = verify.classify_deterministic(verify.extract_claims(answer), ev_)
    plausible = [v for v in det if v.status == verify.UNSUPPORTED and v.plausible]
    judge = {"verdicts": [{"id": v.claim.index, "status": "supported",
                           "reason": "paraphrase"} for v in plausible
                          if "countries" in v.claim.text]}
    if not judge["verdicts"]:
        pytest.skip("fixture no longer produces a judge-clearable claim")

    _got, arms = replay_to(tmp_path, [record(answer)],
                           StubClient(judge=judge, repair=repaired))
    on, co = arms["on"]["t-case"], arms["on-carryoff"]["t-case"]
    assert on["repair_replay"]["repair_text_sha256"] == \
        co["repair_replay"]["repair_text_sha256"]
    assert on["repaired"] != co["repaired"] or on["answer"] != co["answer"], (
        "carry-on and carry-off produced the identical outcome on a fixture "
        "built to separate them — the carry-off arm is not a second opinion")


# ---------------------------------------------------------------------------
# audit: rows built by hand, so each shape is checked in isolation
# ---------------------------------------------------------------------------
def arm_row(answer, claims, case_id="t-case", arm="off", hits=None,
            allowed=None, lang="en", checks=None, keys=None):
    """One replayed arm row. `claims` is a list of (text, status, grounded)."""
    hits = hits if hits is not None else [
        {"doc": DOC, "page": PAGE, "score": 0.9, "text": EVIDENCE}]
    rows = []
    for i, (text, status, grounded) in enumerate(claims):
        cits = verify.parse_citations(text)
        rows.append({
            "key": ar.claim_key(case_id, text), "norm": verify.norm_text(text),
            "index": i, "text": text, "kind": "money", "required": True,
            "cited": bool(cits), "citations": [[c.doc, c.page] for c in cits],
            "status": status, "source": "deterministic", "reason": "",
            "flags": [], "grounded": grounded})
    n = len(rows)
    sup = sum(1 for r in rows if r["status"] == verify.SUPPORTED)
    gro = sum(1 for r in rows if r["grounded"])
    cs = sum(1 for r in rows if r["status"] == verify.SUPPORTED and r["cited"])
    return {
        "id": case_id, "class": "identifier", "lang": lang, "repair_arm": arm,
        "answer": answer, "raw_answer": answer, "hits": hits,
        "notes_used": {},
        "evidence_keys": keys if keys is not None else [f"{DOC}|{PAGE}"],
        "claim_rows": rows,
        "claims": {"claims": n, "supported": sup, "grounded": gro,
                   "citation_supported": cs,
                   "cited": sum(1 for r in rows if r["cited"]),
                   "contradicted": sum(1 for r in rows
                                       if r["status"] == verify.CONTRADICTED),
                   "unsupported": sum(1 for r in rows
                                      if r["status"] == verify.UNSUPPORTED)},
        "checks": checks if checks is not None else {
            "pass": True, "score": 1.0, "behavior": True, "language": True,
            "citations": True, "must_contain": {}, "must_not_contain": {}},
        "fields": None, "score": 1.0, "verify_status": "verified",
        "repaired": arm != "off", "repair_rejected": False, "repair_notes": [],
        "usage": {"calls": []},
        "repair_replay": {"replayed": True,
                          "repair_allowed_docs": allowed if allowed is not None
                          else [DOC]},
    }


def run_audit(tmp_path, off_rows, on_rows, **kw):
    a = tmp_path / "off.jsonl"
    b = tmp_path / "on.jsonl"
    a.write_text("".join(json.dumps(r) + "\n" for r in off_rows), encoding="utf-8")
    b.write_text("".join(json.dumps(r) + "\n" for r in on_rows), encoding="utf-8")
    return ar.audit(a, b, **kw)


SUP, UNS, CON = verify.SUPPORTED, verify.UNSUPPORTED, verify.CONTRADICTED


def test_invented_document_is_caught(tmp_path):
    before = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    after = f"GCF funding is **USD 18.5 million** [999_not-a-doc, p. 3]."
    rep = run_audit(tmp_path,
                    [arm_row(before, [(before, UNS, True)])],
                    [arm_row(after, [(after, SUP, True)], arm="on")])
    assert [d["doc"] for d in rep["invented_docs"]] == ["999_not-a-doc"]
    assert rep["gate_pass"] is False
    assert any("invented document" in b for b in rep["breaches"])


def test_invented_page_is_caught(tmp_path):
    before = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    after = f"GCF funding is **USD 18.5 million** [{DOC}, p. 7]."
    rep = run_audit(tmp_path,
                    [arm_row(before, [(before, UNS, True)])],
                    [arm_row(after, [(after, SUP, True)], arm="on")])
    assert rep["invented_pages"] == [{"doc": DOC, "page": 7, "case": "t-case"}]
    assert rep["gate_pass"] is False


def test_document_not_shown_to_the_repair_pass_is_caught(tmp_path):
    """The hard case: the doc IS in the evidence, so verify's own exact-match
    gate passes, and the attribution is still invented."""
    before = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    after = f"GCF funding is **USD 42 million** [{OTHER_DOC}, p. 2]."
    hits = [{"doc": DOC, "page": PAGE, "score": 0.9, "text": EVIDENCE},
            {"doc": OTHER_DOC, "page": 2, "score": 0.5, "text": OTHER_EVIDENCE}]
    keys = [f"{DOC}|{PAGE}", f"{OTHER_DOC}|2"]
    rep = run_audit(
        tmp_path,
        [arm_row(before, [(before, UNS, True)], hits=hits, keys=keys)],
        [arm_row(after, [(after, SUP, True)], arm="on", hits=hits, keys=keys,
                 allowed=[DOC])])
    assert rep["invented_docs"] == [] and rep["invented_pages"] == []
    assert [d["doc"] for d in rep["sources_not_shown_to_repair"]] == [OTHER_DOC]
    assert rep["gate_pass"] is False


def test_invented_figure_is_caught(tmp_path):
    before = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    after = f"GCF funding is **USD 23,700,000** [{DOC}, p. {PAGE}]."
    rep = run_audit(tmp_path,
                    [arm_row(before, [(before, UNS, True)])],
                    [arm_row(after, [(after, SUP, True)], arm="on")])
    digits = {f["digits"] for f in rep["invented_figures"]}
    assert "23700000" in digits
    assert rep["invented_figures_matcher_cannot_place"]
    assert rep["gate_pass"] is False


def test_a_figure_that_is_in_the_evidence_is_not_invented(tmp_path):
    """The negative. 50,000,000 is printed on the held page; moving to it is a
    correction, and calling it an invention would make the gate useless."""
    before = f"Total project financing is **USD 99,900,000** [{DOC}, p. {PAGE}]."
    after = f"Total project financing is **USD 50,000,000** [{DOC}, p. {PAGE}]."
    rep = run_audit(tmp_path,
                    [arm_row(before, [(before, CON, False)])],
                    [arm_row(after, [(after, SUP, True)], arm="on")])
    assert rep["invented_figures"] == []
    assert rep["invented_docs"] == [] and rep["invented_pages"] == []
    assert rep["gate_pass"] is True


def test_lost_supported_claim_gates_and_a_documented_one_does_not(tmp_path):
    """Isolated from the supported-count gate on purpose: the repair replaces
    a failing claim with a correct one, so `supported` does NOT fall, and the
    only thing left to gate is the supported claim that went missing."""
    keep = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    lost = f"Total project financing is **USD 50 million** [{DOC}, p. {PAGE}]."
    bad = f"The programme covers **12 countries** [{DOC}, p. {PAGE}]."
    fixed = f"The programme covers **9 countries** [{DOC}, p. {PAGE}]."
    off_rows = [arm_row(keep + " " + lost + " " + bad,
                        [(keep, SUP, True), (lost, SUP, True), (bad, UNS, False)])]
    on_rows = [arm_row(keep + " " + fixed,
                       [(keep, SUP, True), (fixed, SUP, True)], arm="on")]
    rep = run_audit(tmp_path, off_rows, on_rows)
    assert rep["supported_before"] == rep["supported_after"] == 2
    assert rep["lost_supported_claims"] == 1
    assert rep["lost_supported_undocumented"] == 1
    assert rep["gate_pass"] is False
    assert rep["breaches"] == ["1 supported claim(s) lost with no documented "
                               "necessity"]
    entry = [r for r in rep["lost_claims"] if r["status_before"] == SUP][0]
    assert entry["text"] == lost and entry["grounded_before"] is True

    j = tmp_path / "just.jsonl"
    j.write_text(json.dumps({"key": entry["key"],
                             "justification": "the page prints a rival total"})
                 + "\n", encoding="utf-8")
    rep2 = run_audit(tmp_path, off_rows, on_rows, justifications=j)
    assert rep2["lost_supported_claims"] == 1
    assert rep2["lost_supported_undocumented"] == 0
    assert rep2["gate_pass"] is True


def test_every_lost_claim_is_listed_not_only_the_supported_ones(tmp_path):
    keep = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    bad = f"The total project cost is **USD 99,900,000** [{DOC}, p. {PAGE}]."
    rep = run_audit(tmp_path,
                    [arm_row(keep + " " + bad, [(keep, SUP, True), (bad, UNS, False)])],
                    [arm_row(keep, [(keep, SUP, True)], arm="on")])
    assert len(rep["lost_claims"]) == 1
    assert rep["lost_claims"][0]["status_before"] == UNS
    assert rep["lost_supported_claims"] == 0
    assert rep["gate_pass"] is True                  # a profitable deletion


def test_corrected_deleted_and_qualified_are_three_outcomes(tmp_path):
    wrong = f"Total project financing is **USD 99,900,000** [{DOC}, p. {PAGE}]."
    fixed = f"Total project financing is **USD 50,000,000** [{DOC}, p. {PAGE}]."
    gone = f"The programme covers **12 countries** [{DOC}, p. {PAGE}]."
    hedge_before = f"The co-financing is **USD 7,000,000** [{DOC}, p. {PAGE}]."
    hedge_after = ("The co-financing is not stated in the retrieved evidence "
                   f"[{DOC}, p. {PAGE}].")
    off_rows = [
        arm_row(wrong, [(wrong, CON, False)], case_id="c1"),
        arm_row(gone, [(gone, UNS, False)], case_id="c2"),
        arm_row(hedge_before, [(hedge_before, UNS, False)], case_id="c3"),
    ]
    on_rows = [
        arm_row(fixed, [(fixed, SUP, True)], case_id="c1", arm="on"),
        arm_row("No figure is available.", [], case_id="c2", arm="on"),
        arm_row(hedge_after, [], case_id="c3", arm="on"),
    ]
    rep = run_audit(tmp_path, off_rows, on_rows)
    fates = {r["case"]: r["fate"] for r in rep["lost_claims"]}
    assert fates["c1"] == "corrected"
    assert fates["c2"] == "deleted"
    assert fates["c3"] == "qualified"
    assert rep["corrected"] == 1 and rep["deleted"] == 1 and rep["qualified"] == 1


def test_deletion_of_a_grounded_claim_is_called_out(tmp_path):
    """'corrected rather than merely deleted WHERE EVIDENCE PERMITS' — the
    evidence permitting it is exactly `grounded_before`."""
    keep = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    miscited = f"Total project financing is **USD 50 million** [{DOC}, p. 99]."
    rep = run_audit(
        tmp_path,
        [arm_row(keep + " " + miscited,
                 [(keep, SUP, True), (miscited, UNS, True)])],
        [arm_row(keep, [(keep, SUP, True)], arm="on")])
    assert len(rep["deleted_though_grounded"]) == 1
    assert rep["deleted_though_grounded"][0]["text"] == miscited
    # reported, not gated: an honest deletion is still allowed
    assert rep["gate_pass"] is True


def test_answer_check_pass_to_fail_gates(tmp_path):
    a = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    off = arm_row(a, [(a, SUP, True)])
    on = arm_row(a, [(a, SUP, True)], arm="on")
    on["answer"] = a + " extra"
    on["checks"] = dict(off["checks"], **{"pass": False, "behavior": False})
    rep = run_audit(tmp_path, [off], [on])
    assert rep["answer_checks_pass_to_fail"] == 1
    assert rep["answer_checks_pass_to_fail_cases"] == ["t-case"]
    assert any(r["check"] == "behavior" for r in rep["check_regressions"])
    assert rep["gate_pass"] is False


def test_a_french_answer_turned_english_is_a_language_regression(tmp_path):
    fr = f"Le financement du FCV est de **18,5 millions USD** [{DOC}, p. {PAGE}]."
    en = f"The GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    off = arm_row(fr, [(fr, UNS, True)], lang="fr")
    on = arm_row(en, [(en, SUP, True)], lang="fr", arm="on")
    on["checks"] = dict(on["checks"], language=False, **{"pass": False})
    rep = run_audit(tmp_path, [off], [on])
    assert len(rep["language_regressions"]) == 1
    assert rep["language_regressions"][0]["case"] == "t-case"
    assert rep["gate_pass"] is False


def test_supported_falling_is_a_breach(tmp_path):
    a = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    b = f"GCF funding is not stated [{DOC}, p. {PAGE}]."
    rep = run_audit(tmp_path, [arm_row(a, [(a, SUP, True)])],
                    [arm_row(b, [(b, UNS, False)], arm="on")])
    assert rep["supported_before"] == 1 and rep["supported_after"] == 0
    assert any("supported fell" in x for x in rep["breaches"])


def test_unchanged_answers_move_nothing(tmp_path):
    a = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    rep = run_audit(tmp_path, [arm_row(a, [(a, SUP, True)])],
                    [arm_row(a, [(a, SUP, True)], arm="on")])
    assert rep["answers_rewritten"] == 0
    assert rep["lost_claims"] == [] and rep["added_claims"] == []
    assert rep["invented_figures"] == []
    assert rep["gate_pass"] is True


def test_novel_sentences_are_listed_with_groundedness(tmp_path):
    a = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    extra = f"The accredited entity is Global Green Growth Institute [{DOC}, p. {PAGE}]."
    rep = run_audit(tmp_path, [arm_row(a, [(a, SUP, True)])],
                    [arm_row(a + " " + extra,
                             [(a, SUP, True), (extra, SUP, True)], arm="on")])
    assert [n["text"] for n in rep["novel_sentences"]] == [extra]
    assert rep["novel_sentences"][0]["grounded"] is True
    assert len(rep["added_claims"]) == 1


def test_main_exits_1_on_a_breach_and_0_otherwise(tmp_path, capsys):
    a = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    bad = "GCF funding is **USD 18.5 million** [999_not-a-doc, p. 1]."
    off = tmp_path / "off.jsonl"
    on = tmp_path / "on.jsonl"
    off.write_text(json.dumps(arm_row(a, [(a, SUP, True)])) + "\n", encoding="utf-8")
    on.write_text(json.dumps(arm_row(bad, [(bad, SUP, True)], arm="on")) + "\n",
                  encoding="utf-8")
    assert ar.main(["--off", str(off), "--on", str(on), "--quiet"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["gate_pass"] is False
    assert payload["invented_docs"]

    on.write_text(json.dumps(arm_row(a, [(a, SUP, True)], arm="on")) + "\n",
                  encoding="utf-8")
    assert ar.main(["--off", str(off), "--on", str(on), "--quiet"]) == 0


def test_the_contract_keys_are_all_present(tmp_path, capsys):
    """The plan names the keys this script must put on stdout."""
    a = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    off = tmp_path / "off.jsonl"
    on = tmp_path / "on.jsonl"
    off.write_text(json.dumps(arm_row(a, [(a, SUP, True)])) + "\n", encoding="utf-8")
    on.write_text(json.dumps(arm_row(a, [(a, SUP, True)], arm="on")) + "\n",
                  encoding="utf-8")
    ar.main(["--off", str(off), "--on", str(on), "--quiet"])
    payload = json.loads(capsys.readouterr().out)
    for key in ("invented_docs", "invented_pages", "invented_figures",
                "lost_claims", "lost_supported_claims",
                "answer_checks_pass_to_fail", "claims_before", "claims_after",
                "supported_before", "supported_after"):
        assert key in payload, key


def test_zero_api(tmp_path, monkeypatch):
    """Zero-API is TESTED, not asserted (standing rule 3)."""
    def boom():                                                  # pragma: no cover
        raise AssertionError("verify._client() was called")
    monkeypatch.setattr(verify, "_client", boom)
    a = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    rep = run_audit(tmp_path, [arm_row(a, [(a, SUP, True)])],
                    [arm_row(a, [(a, SUP, True)], arm="on")])
    assert rep["gate_pass"] is True
    # the replay refuses rather than silently skipping the judge
    src = tmp_path / "rel.jsonl"
    src.write_text(json.dumps(record(a)) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        ar.replay(src, tmp_path / "rp")


def test_write_refuses_to_overwrite_a_recorded_arm(tmp_path):
    p = tmp_path / "x.jsonl"
    ar.write_jsonl(p, [{"id": "a"}])
    with pytest.raises(SystemExit):
        ar.write_jsonl(p, [{"id": "b"}])
    ar.write_jsonl(p, [{"id": "b"}], force=True)
    assert ar.read_jsonl(p) == [{"id": "b"}]


def test_a_page_number_inside_a_citation_is_not_an_invented_figure(tmp_path):
    """The instrument's own false positive, pinned. A re-citation to another
    HELD page is judged by `invented_pages`; counting its digits as an
    invented figure fires on every legitimate move and hides the real shape."""
    before = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    after = f"GCF funding is **USD 18.5 million** [{DOC}, p. 87]."
    keys = [f"{DOC}|{PAGE}", f"{DOC}|87"]
    hits = [{"doc": DOC, "page": PAGE, "score": 0.9, "text": EVIDENCE},
            {"doc": DOC, "page": 87, "score": 0.4, "text": "Annex text."}]
    rep = run_audit(
        tmp_path,
        [arm_row(before, [(before, UNS, True)], keys=keys, hits=hits)],
        [arm_row(after, [(after, SUP, True)], arm="on", keys=keys, hits=hits)])
    assert rep["invented_figures"] == []
    assert rep["invented_pages"] == []
    assert rep["newly_cited_pages"] == [{"doc": DOC, "page": 87, "case": "t-case"}]
    assert rep["gate_pass"] is True


def test_repair_call_spread_is_reported_when_a_second_sample_exists(tmp_path):
    a = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    off = arm_row(a, [(a, UNS, True)])
    on = arm_row(a, [(a, SUP, True)], arm="on")
    on["repaired"] = True
    on["repair_replay"]["repair_second_sample"] = {
        "identical_text": False, "adopted": False, "status": "partial",
        "notes": ["repair rejected"]}
    rep = run_audit(tmp_path, [off], [on])
    sp = rep["repair_call_spread"]
    assert sp["cases_sampled_twice"] == 1
    assert sp["identical_repair_text"] == 0
    assert sp["adoption_flipped"] == 1
    assert sp["flips"][0]["case"] == "t-case"


def test_cost_excludes_the_generation_calls_of_a_skipped_row(tmp_path):
    """A guard row carries the ORIGINAL run's usage; booking its answer call
    as a repair cost would put the generation bill in the repair envelope."""
    a = f"GCF funding is **USD 18.5 million** [{DOC}, p. {PAGE}]."
    skipped = arm_row(a, [(a, SUP, True)], case_id="g", arm="on")
    skipped["repair_replay"] = {"replayed": False, "reason": "guard-answer"}
    skipped["usage"] = {"calls": [{"role": "answer", "latency_s": 3.0,
                                   "prompt_tokens": 5000,
                                   "completion_tokens": 400,
                                   "total_tokens": 5400}]}
    live = arm_row(a, [(a, SUP, True)], case_id="t", arm="on")
    live["usage"] = {"calls": [{"role": "repair", "latency_s": 2.0,
                                "prompt_tokens": 100, "completion_tokens": 10,
                                "total_tokens": 110}]}
    off_rows = [arm_row(a, [(a, SUP, True)], case_id="g"),
                arm_row(a, [(a, SUP, True)], case_id="t")]
    rep = run_audit(tmp_path, off_rows, [skipped, live])
    assert "answer" not in rep["cost"]
    assert rep["cost"]["repair"]["calls"] == 1


def test_pre_gate_probe_sees_a_rewrite_the_gate_threw_away(tmp_path):
    """The adversarial probe. A repair that invents a citation AND leaves a
    claim failing is rejected for the second reason; counting only ADOPTED
    rewrites would report zero inventions from a pass that proposed one."""
    answer = (f"FP151 requests **USD 18.5 million** [{DOC}, p. {PAGE}]. "
              f"The total project cost is **USD 99,900,000** [{DOC}, p. {PAGE}].")
    proposed = (f"FP151 requests **USD 18.5 million** [{DOC}, p. {PAGE}]. "
                f"The total project cost is **USD 77,700,000** [999_invented, p. 3].")
    client = StubClient(judge={"verdicts": []}, repair=proposed)
    _got, arms = replay_to(tmp_path, [record(answer)], client)
    on = arms["on"]["t-case"]
    assert on["repaired"] is False and on["repair_rejected"] is True
    pg = on["repair_replay"]["repair_pre_gate"]
    assert pg["introduced_sources"], "verify's own gate saw nothing"
    assert pg["would_adopt_without_carry"] is False

    a = tmp_path / "off.jsonl"
    b = tmp_path / "on.jsonl"
    a.write_text(json.dumps(arms["off"]["t-case"]) + "\n", encoding="utf-8")
    b.write_text(json.dumps(on) + "\n", encoding="utf-8")
    rep = ar.audit(a, b)
    probe = rep["pre_gate_probe"]
    assert probe["proposals"] == 1
    assert probe["verify_gate_caught_introduced_sources"]
    assert probe["audit_found"]["invented_docs"] == 1
    assert probe["audit_found"]["invented_figures"] >= 1
    # the ADOPTED tables stay clean: the rewrite never reached the user
    assert rep["invented_docs"] == [] and rep["answers_rewritten"] == 0
