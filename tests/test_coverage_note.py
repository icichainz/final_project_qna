"""The corpus-coverage note: the L2 closer for corpus-level aggregates.

Measured defect (release-4, both RERANK arms, agg-corpus-boards 0.43):
"Which board meetings does this corpus cover?" names no year and no board
code, so neither the year note nor _board_range_note fired, notes_used came
back empty, and the model — correctly forbidden from stating corpus-wide
facts from a retrieved sample — described portfolio-company "board meetings"
it found in excerpts. The registry is complete and can answer; this note
delivers it, with the same authority framing the year note uses.
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gcf_qna.app import chainlit_app as app  # noqa: E402
from gcf_qna.boards import BOARD_YEARS  # noqa: E402
from gcf_qna.rag import registry  # noqa: E402


# ------------------------------------------------------------- trigger ---
@pytest.mark.parametrize("q", [
    "Which board meetings does this corpus cover?",
    "What years does the corpus cover?",
    "How many funding proposals are in the corpus?",
    "How many documents does this corpus contain in total?",
    "Combien de propositions de financement ce corpus contient-il ?",
    "Quelles réunions du Conseil du GCF ce corpus couvre-t-il ?",
])
def test_coverage_questions_fire(q):
    assert app._corpus_coverage_note(q) is not None


@pytest.mark.parametrize("q,reason", [
    ("What was approved at B.42?", "board-code paths own board codes"),
    ("How many proposals from 2020 are in this corpus?",
     "the year note owns year questions - no turn gets two authorities"),
    ("Who sits on the board of directors of the accredited entity for FP151?",
     "'board' without a corpus token is an ordinary retrieval question"),
    ("What is the total GCF funding requested in FP151?", "ordinary question"),
    ("Does this corpus cover Tanzania?",
     "country coverage is answered from excerpts, not the boards table"),
])
def test_non_coverage_questions_do_not_fire(q, reason):
    assert app._corpus_coverage_note(q) is None, reason


# ---------------------------------------------- computed, never hardcoded ---
def test_the_numbers_are_recomputed_from_the_registry():
    """Independent recomputation: the note's totals must equal what the
    registry actually holds, so a corpus rebuild moves the note with it."""
    note = app._corpus_coverage_note("Which board meetings does this corpus cover?")
    lo, hi = min(BOARD_YEARS), max(BOARD_YEARS)
    assert f"B.{lo} ({BOARD_YEARS[lo]})" in note
    assert f"B.{hi} ({BOARD_YEARS[hi]})" in note
    counts = {y: len([r for r in registry.by_year(y) if r.get("fp")])
              for y in sorted(set(BOARD_YEARS.values()))}
    assert f"{sum(counts.values())} funding-proposal documents" in note
    for y, n in counts.items():
        assert f"{y}: {n}" in note


def test_the_note_carries_the_authority_framing():
    """The corpus-scope prompt rule forbids corpus-wide claims from excerpts;
    the note must carry the same authority override the year note uses, or
    the model will refuse to use it."""
    note = app._corpus_coverage_note("What years does the corpus cover?")
    assert "complete" in note and "authoritative, unlike the excerpts" in note


# ------------------------------------------------------------- wiring ---
def test_app_and_harness_emit_the_same_note():
    import eval_answers as ev  # noqa: F401  (parity: harness calls app's fn)
    src = (ROOT / "scripts" / "eval_answers.py").read_text()
    assert "app._corpus_coverage_note(question)" in src
    app_src = (ROOT / "src" / "gcf_qna" / "app" / "chainlit_app.py").read_text()
    assert "_corpus_coverage_note(message.content)" in app_src


# ----------------------------------------------- the recorded failing turn ---
def test_the_recorded_release_4_turn_is_now_answerable_from_the_note():
    """Replay of the recorded agg-corpus-boards turn: every must_contain
    regex of the case must be satisfiable from the note alone."""
    rec = None
    for line in (ROOT / "data" / "eval" / "release_release-4.jsonl").open():
        r = json.loads(line)
        if r["id"] == "agg-corpus-boards":
            rec = r
            break
    assert rec is not None
    assert (rec.get("notes_used") or {}) == {}, \
        "premise: the recorded run had no note for this turn"
    note = app._corpus_coverage_note(rec["question"])
    assert note is not None
    for pat in rec["expect"]["must_contain"]:
        assert re.search(pat[3:], note), f"{pat} not answerable from the note"


# ------------------------------------------------------- new gold cases ---
def test_the_new_gold_cases_load_and_their_expectations_hold():
    import eval_answers as ev
    cases = {c["id"]: c for c in ev.load_cases(ev.DEFAULT_CASES)}
    for cid in ("agg-corpus-years", "agg-corpus-total", "fr-agg-corpus-boards"):
        assert cid in cases, f"{cid} missing from the gold set"
        note = app._corpus_coverage_note(cases[cid]["question"])
        assert note is not None, f"{cid}: the coverage note must fire"
        for pat in cases[cid]["expect"]["must_contain"]:
            assert re.search(pat[3:], note), \
                f"{cid}: {pat} not answerable from the note"
