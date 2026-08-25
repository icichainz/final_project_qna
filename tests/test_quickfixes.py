"""Three production-observed failures, fixed and pinned.

1. Registry-resolved doc tags: B.27-era filenames carry no FP number
   (FP152 -> '123_gcf-b27-02-add12'), so a 'fp152' doc filter matched nothing
   and the scoped search degraded to an unscoped phrase — live, a question
   about FP152's approval conditions came back with FP242 pages.
2. Year ranges: "approved after 2023" fired for 2023 only, so the registry
   note listed 2023 documents and the model never saw FP274 (2025).
3. Prompt lines: the board-year IS the approval year (the model refused to
   call FP274 'approved after 2023'), and the excerpts are injected by the
   system, not pasted by the user (answers asked the user to share pages).
"""
import pytest

from gcf_qna.app.chainlit_app import (_registry_doc, _rescope_items,
                                      _resolve_doc_tags, _year_assist,
                                      _year_scope)
from gcf_qna.app.prompts import assemble, assemble_chat
from gcf_qna.rag import registry
from gcf_qna.rag.retrieve import _doc_match

FP152 = "123_gcf-b27-02-add12"          # no 'fp152' anywhere in the stem
FP086 = "189_12-status-approved-fps-adding-host-countries-respect-fp086-gcf-b37-07"
FP202 = "73_gcf-b35-02-add04"           # the doc a truncated 'fp2023' hit
FP214 = "61_gcf-b37-02-add05-funding-proposal-package-fp214"
FP274 = "02_gcf-b42-02-add16-funding-proposal-package-fp274"
FP008 = "205_gcf-b11-02-add08-funding-proposal-package-fp008"

FAKE_REGISTRY = {
    FP152: {"fp": 152, "year": 2020, "board": 27, "gcf_financing": "150 M USD"},
    FP086: {"fp": 86, "year": 2023, "board": 37},
    FP202: {"fp": 202, "year": 2023, "board": 35},
    FP214: {"fp": 214, "year": 2023, "board": 37},
    FP274: {"fp": 274, "year": 2025, "board": 42},
    FP008: {"fp": 8, "year": 2015, "board": 11},
}


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(registry, "_cache", FAKE_REGISTRY)
    # ... and an EMPTY v2: the year note now prefers the v2 canonical figure,
    # so a v1-only fixture that left _cache_v2 alone would quietly read the
    # real data/registry_v2.json and print the corpus's money for these
    # stems. Empty means "v2 knows nothing here" -> the v1 fallback, which is
    # what these cases are pinning.
    monkeypatch.setattr(registry, "_cache_v2", {})
    yield


@pytest.fixture
def no_registry(monkeypatch):
    monkeypatch.setattr(registry, "load",
                        lambda: (_ for _ in ()).throw(FileNotFoundError()))
    yield


# --- 1: registry-resolved doc tags -----------------------------------------

def test_fp_tag_resolves_to_the_authoritative_stem(fake_registry):
    """The verified failure: 'fp152' matches no document by itself."""
    assert not _doc_match(FP152, "fp152")
    assert _registry_doc("fp152") == FP152
    assert _registry_doc("FP152") == FP152
    assert _registry_doc("02_fp152") == FP152       # fabricated conductor tag
    assert _doc_match(FP152, _registry_doc("fp152"))


def test_zero_padded_tag_resolves_through_by_fp(fake_registry):
    """'fp086' -> by_fp(86): the registry key is an int, the tag is padded."""
    assert _registry_doc("fp086") == FP086
    assert _registry_doc("fp86") == FP086
    assert _registry_doc("fp0086") == FP086      # was FP8's doc (greedy 008)


def test_truncated_fp_token_resolves_to_nothing(fake_registry):
    """The regex regression: 'fp2023' must not read as FP202.

    A doc tag that resolves to a real but WRONG document HARD-SCOPES the
    sub-query to it — strictly worse than the unscoped degrade this fix
    exists to prevent.
    """
    assert _registry_doc("fp2023") == "fp2023"
    assert _registry_doc("fp2023") != FP202
    assert _registry_doc("fp1520") == "fp1520"


def test_single_digit_fp_resolves(fake_registry):
    """The shared pattern accepts 1-3 digits; the old local copy needed 2."""
    assert _registry_doc("fp8") == FP008
    assert _registry_doc("fp008") == FP008


def test_one_fp_pattern_in_the_app():
    """No second definition to drift from the registry's."""
    from gcf_qna.app import chainlit_app
    assert chainlit_app._FP_RE is registry._FP_RE


def test_known_corpus_id_is_left_alone(fake_registry):
    """A tag the guard pinned to a cited document keeps pointing there."""
    assert _registry_doc(FP274) == FP274


def test_unknown_fp_and_tagless_input_pass_through(fake_registry):
    assert _registry_doc("fp999") == "fp999"
    assert _registry_doc("13_gcf-b41-02-add03-fp250") == "13_gcf-b41-02-add03-fp250"
    assert _registry_doc("") == ""
    assert _registry_doc(None) is None


def test_registry_missing_keeps_the_tag(no_registry):
    assert _registry_doc("fp152") == "fp152"
    items = _resolve_doc_tags([{"q": "conditions", "doc": "fp152"}])
    assert items[0]["doc"] == "fp152"


def test_empty_registry_keeps_the_tag(monkeypatch):
    monkeypatch.setattr(registry, "_cache", {})
    assert _registry_doc("fp152") == "fp152"


def test_resolution_composes_with_the_guard(fake_registry):
    """The guard rewrites a fabricated tag to the plain 'fp214' token; the
    resolution pass then upgrades that token to the real stem."""
    items = _rescope_items([{"q": "total financing", "doc": "02_fp214"}],
                           "What is FP214's total financing?", [FP274])
    assert items[0]["doc"] == "fp214"                 # guard's decision intact
    items = _resolve_doc_tags(items)
    assert items[0]["doc"] == FP214


def test_stripped_tags_are_not_resurrected(fake_registry):
    """An unrelated tag is stripped by the guard; the FP number inside it must
    not bring the scope back through the registry."""
    items = _rescope_items(
        [{"q": "total financing", "doc": "13_gcf-b41-02-add03-fp250"}],
        "What is FP214's total financing?", [FP274])
    items = _resolve_doc_tags(items)
    assert items[0]["doc"] is None
    assert items[0]["q"] == "total financing FP214"


def test_history_slack_tag_survives_resolution(fake_registry):
    items = _resolve_doc_tags(_rescope_items(
        [{"q": "total financing", "doc": FP214},
         {"q": "total financing", "doc": FP274}],
        "How does FP214's financing compare to it?", [FP274]))
    assert [i["doc"] for i in items] == [FP214, FP274]


# --- 2: year-range semantics ------------------------------------------------

def test_year_scope_expands_ranges():
    assert _year_scope("proposals approved after 2023") == {2024, 2025}
    assert _year_scope("anything before 2018") == {2015, 2016, 2017}
    assert _year_scope("proposals since 2023") == {2023, 2024, 2025}
    assert _year_scope("projets approuvés après 2023") == {2024, 2025}
    assert _year_scope("projets depuis 2023") == {2023, 2024, 2025}
    assert _year_scope("projets avant 2018") == {2015, 2016, 2017}
    assert _year_scope("until 2017") == {2015, 2016, 2017}
    assert _year_scope("from 2019 to 2021") == {2019, 2020, 2021}
    assert _year_scope("à partir de 2023") == {2023, 2024, 2025}
    assert _year_scope("proposals from 2023 onwards") == {2023, 2024, 2025}


def test_bare_years_keep_exact_behaviour():
    assert _year_scope("Which proposals were approved in 2020?") == {2020}
    assert _year_scope("compare 2019 and 2021") == {2019, 2021}
    assert _year_scope("proposals from 2020?") == {2020}      # 'of 2020'
    assert _year_scope("no year here") == set()


def test_range_stays_inside_the_corpus_span():
    assert _year_scope("after 2014") == set(range(2015, 2026))
    # a range with no overlap contributes NO years: listing 2015's proposals
    # under "before 2015" would be a wrong answer stated with confidence
    assert _year_scope("before 2015") == set()
    assert _year_scope("after 2025") == set()


def test_out_of_range_gets_a_definitive_note(fake_registry):
    hits, note = _year_assist("Which proposals were approved before 2015?", [])
    assert "no proposals before 2015" in note
    assert "B.11 (2015) through B.43 (2025)" in note
    assert "State this definitively." in note
    assert "FP8" not in note                 # never the nearest year's listing
    assert hits == []


def test_out_of_range_appended_to_a_real_year_note(fake_registry):
    _, note = _year_assist("proposals in 2020 or after 2025?", [])
    assert "FP152" in note                   # the 2020 half still answered
    assert "no proposals after 2025" in note


def test_after_2023_note_covers_2024_and_2025(fake_registry):
    _, note = _year_assist("Which proposals were approved after 2023?", [])
    assert "2025 — 1 proposals: FP274." in note
    assert "2024 — no registered proposals" in note
    assert "FP214" not in note and "2023 —" not in note   # bound excluded


def test_before_2018_note_covers_2015_to_2017(fake_registry):
    _, note = _year_assist("Anything before 2018?", [])
    assert "2015 — 1 proposals: FP8." in note
    for y in (2016, 2017):
        assert f"{y} — no registered proposals" in note
    assert "2018" not in note


def test_wide_span_emits_counts_not_financing(monkeypatch):
    """Counts instead of per-FP financing — and no invented contiguity.

    2021's numbers here are deliberately gappy (156..173 minus two), the way
    the real corpus is: 2023 holds FP86 and FP224 with 110 non-members in
    between. 2022's are consecutive, so the range form is still exercised.
    """
    gappy = [n for n in range(156, 174) if n not in (160, 168)]
    wide = {f"{n}_gcf-b28-02-add{n:03d}": {"fp": n, "year": 2021, "board": 28,
                                           "gcf_financing": "9 M USD"}
            for n in gappy}
    wide.update({f"{n}_gcf-b31-02-add{n:03d}": {"fp": n, "year": 2022, "board": 31}
                 for n in range(182, 185)})
    wide.update(FAKE_REGISTRY)
    monkeypatch.setattr(registry, "_cache", wide)
    _, note = _year_assist("proposals approved since 2018", [])
    assert "2021 — 16 proposals (FP156 … FP173, not contiguous)." in note
    assert "FP156–FP173" not in note              # the false contiguity claim
    assert "2022 — 3 proposals: FP182–FP184." in note   # genuinely consecutive
    assert "9 M USD" not in note                  # no per-FP financing listing
    assert "2018–2025" in note                    # compact span in the tail


def test_non_contiguous_year_never_claims_a_range(fake_registry):
    """2023 in the fixture is FP86 + FP202 + FP214 — the real corpus shape."""
    _, note = _year_assist("proposals approved since 2015", [])
    assert "2023 — 3 proposals (FP86 … FP214, not contiguous)." in note
    assert "FP86–FP214" not in note
    assert "2025 — 1 proposals: FP274." in note   # single row needs no range


def test_three_year_span_keeps_the_detailed_listing(fake_registry):
    _, note = _year_assist("proposals since 2023", [])
    assert "2023 — 3 proposals: FP86; FP202; FP214." in note
    assert "authoritative" in note


def test_range_fallback_without_registry(no_registry):
    _, note = _year_assist("proposals approved after 2023", [])
    assert "2024, 2025" in note
    assert "B.38, B.39, B.40, B.41, B.42, B.43" in note or "B.38–B.43" in note
    assert "state this limit" in note


# --- 3: prompt lines --------------------------------------------------------

def test_year_block_states_the_approval_convention():
    p = assemble(year=True)
    assert "the board meeting year is the approval" in p
    assert "do not demand separate approval-decision text" in p
    # scoped to packages: the one status document states its own earlier
    # original approval ('at the twenty-first meeting') and must be usable
    assert "Status and addendum documents are different" in p
    assert "approval year" not in assemble(year=False)


def test_core_says_the_context_is_not_user_supplied():
    p = assemble()
    assert "not supplied by the user" in p
    assert "never ask the user to paste or share pages" in p
    assert "retrieval did not surface it" in p
    assert "not supplied by the user" in assemble_chat("French")
