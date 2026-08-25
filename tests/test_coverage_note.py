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


def test_the_note_forbids_totalling_the_figures_it_licenses():
    """The coverage note is the one note a 'how many / how much in total'
    question reaches, and the year note it hands off to prints one amount per
    proposal in mixed currencies. Measured (F11): asked to total the 2020
    listing the model returned $29.0B against a truth near $1.36B. The rule
    rides on the note, not the prompt - notes fire per trigger, the prompt is
    paid for on every turn."""
    note = app._corpus_coverage_note("How many funding proposals are in the corpus?")
    assert app._NO_SUM_RULE in note
    assert "MUST NOT be summed" in note
    # ... and it stays a rule, not evidence: no page and no document to cite
    assert app._note_pages([app._NO_SUM_RULE]) == set()


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


# --------------------------------------------- the thematic fence (H12/P7) ---
#
# `_COVERAGE_ASK_RE` matches "how many <up to two words> proposals", which is
# also the shape of "how many AGRICULTURE proposals are in the corpus?" — and
# of "how many proposals in the corpus CONCERN AGRICULTURE?", where the
# restriction follows the noun instead of preceding it. Both fired a note that
# holds per-year counts and ends "Answer corpus-coverage questions from this
# note", in front of a question the registry has no field for. Probe P7
# measured the model coping (it said it has no theme field and refused the
# count), which is why this is hygiene — and why the fix must not cost the
# cases that DO belong to the note.

@pytest.mark.parametrize("q,restriction", [
    ("How many proposals in the corpus concern agriculture?", "agriculture"),
    ("How many agriculture proposals are in the corpus?", "agriculture"),
    ("How many adaptation proposals does this corpus contain?", "adaptation"),
    ("Combien de propositions du corpus concernent l'agriculture ?",
     "agriculture"),
    ("How many proposals in the corpus are implemented by UNDP?", "undp"),
    ("Which years does the corpus cover for Kenya?", "kenya"),
    ("How many mitigation and adaptation proposals are in this dataset?",
     "mitigation"),
])
def test_a_restricted_count_gets_no_coverage_note(q, restriction):
    assert app._corpus_coverage_note(q) is None, q
    assert restriction in app._off_vocabulary(q)


@pytest.mark.parametrize("q", [
    "Which board meetings does this corpus cover?",
    "What years does the corpus cover?",
    "How many funding proposals are in the corpus?",
    "How many documents does this corpus contain in total?",
    "Combien de propositions de financement ce corpus contient-il ?",
    "Quelles réunions du Conseil du GCF ce corpus couvre-t-il ?",
    "Which years have the most funding proposals in this corpus?",
    "Hi, could you tell me how many funding proposals are in the corpus?",
    "Bonjour, pouvez-vous me dire combien de propositions ce corpus contient ?",
    "Quick question: what years does this corpus cover?",
])
def test_the_corpus_own_vocabulary_still_fires(q):
    """The fence is an allowlist, so every word of a real coverage question
    has to be in it — including the ranking words of `agg-year-most`, which
    rank the note's OWN dimension and restrict nothing."""
    assert app._off_vocabulary(q) == []
    assert app._corpus_coverage_note(q) is not None


def test_every_gold_coverage_case_still_fires():
    """The allowlist is exact, and an exact fence is one edit away from
    silencing a case it was never aimed at. Every gold case that recorded a
    coverage note in release-10 is pinned here by its own question."""
    import eval_answers as ev
    cases = {c["id"]: c for c in ev.load_cases(ev.DEFAULT_CASES)}
    for cid in ("agg-corpus-boards", "agg-corpus-years", "agg-corpus-total",
                "fr-agg-corpus-boards", "agg-year-most"):
        q = cases[cid]["question"]
        assert app._corpus_coverage_note(q) is not None, \
            f"{cid}: {app._off_vocabulary(q)}"


def test_the_fence_folds_accents_and_ignores_elisions():
    """'réunions' and 'reunions' decide the same way, and the single letters
    French elision and hyphen glue leave behind ("l'agriculture",
    "couvre-t-il") are never the word that restricts a count."""
    assert app._deaccent("Quelles réunions") == "Quelles reunions"
    assert app._off_vocabulary("Quelles reunions ce corpus couvre-t-il ?") == []
    assert app._off_vocabulary("l'agriculture") == ["agriculture"]


# ----------------------------------------------------- ruling 10, here too ---
def test_the_coverage_note_names_the_boards_of_every_year_it_counts():
    """The same fact the year note now prints, for the note that fires when
    the question names no year at all — computed from BOARD_YEARS."""
    note = app._corpus_coverage_note("What years does the corpus cover?")
    for y in sorted(set(BOARD_YEARS.values())):
        boards = ", ".join(f"B.{b}" for b, yy in sorted(BOARD_YEARS.items())
                           if yy == y)
        n = len([r for r in registry.by_year(y) if r.get("fp")])
        assert f"{y}: {n} ({boards})" in note
