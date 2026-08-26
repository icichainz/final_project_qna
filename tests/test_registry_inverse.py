"""The registry's answer to docs/l1-l2-coverage-review.md §4.2 and §7.

**Truncation markers (F13/P5).** `_fmt` printed `', '.join(countries[:5])` with
no count and no ellipsis inside a note the prompt calls authoritative. Asked
"FP151: how many countries does it cover? List them all", the system said
FIVE, cited it, and `classify_deterministic` scored the claim SUPPORTED —
because the evidence really did say five. The truth is 44. Nothing in the
stack asks whether evidence is COMPLETE, so the note has to say so itself:
every list it prints now carries its true length, and a cut one says it was
cut. Pinned here in both directions, on every list the module emits, and
through the consumers that read those lines (`verify._field_lines`'s label
anchor, `_note_pages`/`note_page_scopes`, and the gold cases whose
`must_contain` regexes read the countries fragment).

**Inverse lookups (H1/H2).** `by_country` and `by_entity` did not exist, so
'which proposals are in Kenya' fell to open retrieval: probe P2 named 6 of 25
and never said the list was partial; P10, in French, named 6 of 13 with one
category error. The lookups are here, with the entity normalisation the
review's H1 says is a data prerequisite (126 stored strings, far fewer
organisations), and both are wired to a note that fires ONLY on a set ask —
'What does FP123 do in Kenya?' must never produce a corpus-wide listing.

**Owner rulings (§7).** Three clusters the normalisation had guessed at were
read back to the corpus owner and ruled on (2026-08-26): 'World Bank Group'
and 'Ministry of Environment' stay MERGED, 'Acumen' and 'Acumen Fund' are
SPLIT. A ruling that lives only in a comment is one the next refactor loses,
so each is pinned.

**The board inverse (§8, H16).** 'Which proposals were approved at B.44?' has
always been answered definitively — 44 is outside the corpus — while B.35, a
meeting whose seven documents the registry holds, was answered not at all.
§4.3 counts that pair among the review's three false-authority misfires.

**Corpus extrema (§9, H3/H3b).** P3 asked for the smallest GCF request in the
corpus and was refused; F15 asked the year-scoped form and named the wrong
proposal out of a note that printed the right figure. The ranking is computed
from schema 2's normalised values — and every refusal around it is computed
too: no cross-currency comparison, no assumed currency, no unrankable unit,
and a stated count for everything the ranking left out.
"""
import pytest

from gcf_qna.app import chainlit_app as app
from gcf_qna.rag import registry
from gcf_qna.rag import verify as V

FP151 = "124_gcf-b27-02-add11"

# --------------------------------------------------------------------------
# a synthetic corpus: the collisions the real one contains, in miniature
# --------------------------------------------------------------------------
SYN = {
    "01_a-fp1": {"fp": 1, "title": "Mali Hydromet", "board": 13, "year": 2016,
                 "accredited_entity": "United Nations Development Programme",
                 "countries": ["Mali"]},
    "02_a-fp2": {"fp": 2, "title": "Malawi Resilience", "board": 13, "year": 2016,
                 "accredited_entity": "United Nations Development Programme (UNDP)",
                 "countries": ["Malawi"]},
    "03_a-fp3": {"fp": 3, "title": "Niger Solar", "board": 14, "year": 2016,
                 "accredited_entity": "UNDP", "countries": ["Niger"]},
    "04_a-fp4": {"fp": 4, "title": "Nigeria Grid", "board": 14, "year": 2016,
                 "accredited_entity": "United Nations Development Program, UNDP",
                 "countries": ["Nigeria"]},
    "05_a-fp5": {"fp": 5, "title": "Uganda Landscapes", "board": 15, "year": 2017,
                 "accredited_entity": "The World Bank", "countries": ["Uganda"]},
    "06_a-fp6": {"fp": 6, "title": "Uganda Water", "board": 15, "year": 2017,
                 "accredited_entity": "World Bank", "countries": ["UGANDA"]},
    "07_a-fp7": {"fp": 7, "title": "Serbia Cities", "board": 16, "year": 2017,
                 "accredited_entity": "World Bank Group",
                 "countries": ["Republic of Serbia"]},
    "08_a-fp8": {"fp": 8, "title": "Serbia Forests", "board": 16, "year": 2017,
                 "accredited_entity": "International Finance Corporation (IFC)",
                 "countries": ["Serbia"]},
    "09_a-fp9": {"fp": 9, "title": "Mekong Delta", "board": 17, "year": 2018,
                 "accredited_entity": "Agence Française de Développement (AFD)",
                 "countries": ["Viet Nam"]},
    "10_a-fp10": {"fp": 10, "title": "Coastal Vietnam", "board": 17, "year": 2018,
                  "accredited_entity": "French Development Agency (AFD)",
                  "countries": ["Vietnam"]},
    "11_a-fp11": {"fp": 11, "title": "Congo Basin", "board": 18, "year": 2018,
                  "accredited_entity": "IUCN",
                  "countries": ["Republic of Congo"]},
    "12_a-fp12": {"fp": 12, "title": "Congo Forests", "board": 18, "year": 2018,
                  "accredited_entity":
                      "International Union for Conservation of Nature (IUCN)",
                  "countries": ["Democratic Republic of the Congo"]},
    # a many-country row: the F13 shape, in miniature
    "13_a-fp13": {"fp": 13, "title": "Regional Facility", "board": 19, "year": 2019,
                  "accredited_entity": "Acumen Fund, Inc.",
                  "countries": ["Angola", "Benin", "Botswana", "Burkina Faso",
                                "Burundi", "Kenya", "Mali"]},
    # a row with no FP number: it is in the registry, never in a listing
    "14_a-status": {"fp": None, "title": "Status of approved proposals",
                    "board": 19, "year": 2019,
                    "accredited_entity": "The World Bank", "countries": ["Kenya"]},
    # values that name no country
    "15_a-fp15": {"fp": 15, "title": "Global Markets", "board": 20, "year": 2019,
                  "accredited_entity": "FMO",
                  "countries": ["Asia and Pacific Region", "Kenya"]},
}


@pytest.fixture
def syn(monkeypatch):
    monkeypatch.setattr(registry, "_cache", SYN)
    monkeypatch.setattr(registry, "_cache_v2", {})
    return registry


@pytest.fixture
def real(monkeypatch):
    """The real 273-document registry, as the probes met it."""
    monkeypatch.setattr(registry, "_cache", None)
    monkeypatch.setattr(registry, "_cache_v2", None)
    assert len(registry.load()) == 273
    return registry


# ==========================================================================
# 1. truncation markers — F13/P5
# ==========================================================================

def test_a_complete_list_prints_its_length_and_no_marker():
    assert registry._list_bit("countries", ["Angola", "Benin"]) == \
        "countries (2): Angola, Benin"
    assert registry._list_bit("countries", ["Mali"]) == "countries (1): Mali"
    assert "truncated" not in registry._list_bit("countries", ["Mali"])


def test_a_cut_list_says_it_was_cut_and_carries_the_true_count():
    got = registry._list_bit("countries", [f"C{i}" for i in range(44)])
    assert got.startswith("countries (5 of 44 — list truncated): C0, C1, C2, C3, C4")
    assert got.endswith(", …")


@pytest.mark.parametrize("n,truncated", [(4, False), (5, False), (6, True)])
def test_the_boundary_between_complete_and_cut_is_exact(n, truncated):
    got = registry._list_bit("countries", [f"C{i}" for i in range(n)])
    assert ("list truncated" in got) is truncated
    assert (f"countries ({n}):" in got) is not truncated
    # whichever side of the cap it falls, the TRUE count is on the line
    assert str(n) in got.split(":")[0]


def test_blank_values_never_inflate_the_count():
    assert registry._list_bit("countries", ["Mali", "", None, "  "]) == \
        "countries (1): Mali"


def test_the_f13_probe_note_now_states_44_and_reads_as_truncated(real):
    """F13, reproduced offline. Same question, same registry, new note.

    The probe answered 'five', cited it, and was scored supported. What the
    model now reads on the countries fragment is the count 44 and the word
    truncated — the five names can no longer be read as the whole list.
    """
    note = real.registry_note(
        "FP151: how many countries does it cover? List them all.")
    line = note.splitlines()[0]
    frag = [s for s in line.split("; ") if s.startswith("countries")][0]
    assert frag.startswith("countries (5 of 44 — list truncated): ")
    assert "44" in frag and "list truncated" in frag
    assert frag.split(" (p.")[0].endswith(", …")
    assert "Angola, Benin, Botswana, Burkina Faso, Burundi" in frag
    # the exact string the probe read, gone
    assert "countries: Angola, Benin, Botswana, Burkina Faso, Burundi;" not in note
    # and the true count is extractable from the line by itself
    assert "5 of 44" in frag


def test_the_countries_label_still_heads_its_segment_for_verify(real):
    """The consumer sweep, executed: `verify._field_lines` reads a field's
    value off the one-line note by requiring the LABEL to head its
    semicolon-separated segment, and `_FIELD_LABELS` anchors 'countr(y|ies)'.
    A count in parentheses sits AFTER the label, so both still match."""
    line = real.registry_note("What is FP151?").splitlines()[0]
    rx = dict(V._FIELD_RES)["countries"]
    segs = list(V._field_lines(line, rx))
    assert len(segs) == 1
    seg, at = segs[0]
    assert seg.strip().startswith("countries (5 of 44")
    assert seg[at:].startswith(" (5 of 44")
    assert V.claim_field("FP151 covers 44 countries") == "countries"


def test_the_page_pointer_still_follows_the_countries_value(real):
    """`_meta_page` appends '(p.N)' after the list, and both readers of a note
    line's pages still credit it to the document on that line."""
    line = real.registry_note("What is FP151?").splitlines()[0]
    assert "…" in line and " (p.3);" in line
    assert (FP151, 3) in app._note_pages([line])
    assert (V.note_scope_doc(FP151), 3) in V.note_page_scopes(line)


def test_a_clipped_title_in_a_listing_gets_an_ellipsis():
    assert registry._clip("short") == "short"
    assert registry._clip("x" * 45) == "x" * 45
    assert registry._clip("x" * 46) == "x" * 45 + "…"
    assert registry._clip(None) == "?"


def test_the_year_listing_keeps_its_count_and_its_more_tail(real):
    note = real.registry_note("Which proposals are from 2020?")
    year = [ln for ln in note.splitlines() if "documents from 2020" in ln][0]
    assert year.startswith("Registry — 30 funding-proposal documents from 2020")
    assert year.endswith("(+18 more)")           # 12 printed, 30 stated, 18 held


def test_a_capped_conflict_field_says_how_many_prints_it_held():
    """The other silently-cut list in this module: `_MAX_CONFLICT_ALTS`."""
    row = {"doc_id": "99_x", "fp": 99}
    alts = [{"raw": f"{i},000,000 USD", "page": 10 + i, "section": "B.2",
             "status": "conflicting"} for i in range(4)]
    registry._cache_v2 = {"99_x": {"fp": 99, "facts": {"total_financing": [
        {"raw": "1 M USD", "page": 5, "section": "A.7", "status": "canonical"}]
        + alts}}}
    try:
        (line,) = registry._conflict_lines(row)
    finally:
        registry._cache_v2 = None
    assert "(+2 more disagreeing prints of this field in the document, not " \
           "listed — list truncated)" in line
    assert "report all of them with their pages." in line


def test_a_conflict_field_dropped_by_the_line_cap_is_named(real):
    """`_MAX_CONFLICT_LINES` drops whole fields; the drop is now announced."""
    rows = [r for r in (real.by_fp(n) for n in range(1, 274)) if r]
    lines = [ln for r in rows for ln in real._conflict_lines(r)
             if "further field" in ln]
    for ln in lines:
        assert ln.endswith("— list truncated.")
        assert "also print" in ln


# ==========================================================================
# 2. by_country — H2
# ==========================================================================

def test_by_country_matches_whole_values_not_substrings(syn):
    assert [r["fp"] for r in syn.by_country("Mali")] == [1, 13]
    assert [r["fp"] for r in syn.by_country("Malawi")] == [2]
    assert [r["fp"] for r in syn.by_country("Niger")] == [3]
    assert [r["fp"] for r in syn.by_country("Nigeria")] == [4]


def test_by_country_is_case_and_accent_insensitive(syn):
    assert [r["fp"] for r in syn.by_country("uganda")] == [5, 6]
    assert [r["fp"] for r in syn.by_country("UGANDA")] == [5, 6]
    assert [r["fp"] for r in syn.by_country("  Uganda  ")] == [5, 6]


def test_a_long_official_form_is_reachable_by_its_short_name(syn):
    assert [r["fp"] for r in syn.by_country("Serbia")] == [7, 8]
    assert [r["fp"] for r in syn.by_country("Republic of Serbia")] == [7, 8]


def test_an_ambiguous_short_name_resolves_to_nothing(syn):
    """'Congo' is two different countries in this corpus. An authoritative
    note must not pick one silently, so the short key is not indexed at all
    and both long forms stay reachable."""
    assert syn.by_country("Congo") == []
    assert [r["fp"] for r in syn.by_country("Republic of Congo")] == [11]
    assert [r["fp"] for r in
            syn.by_country("Democratic Republic of the Congo")] == [12]


def test_one_country_spelled_two_ways_is_one_group(syn):
    assert [r["fp"] for r in syn.by_country("Viet Nam")] == [9, 10]
    assert [r["fp"] for r in syn.by_country("Vietnam")] == [9, 10]


def test_a_value_that_names_no_country_is_not_indexed(syn):
    assert syn.by_country("Asia and Pacific Region") == []
    assert syn.by_country("Atlantis") == []
    assert syn.by_country("") == []


def test_the_real_corpus_numbers_the_review_measured(real):
    assert len(real.by_country("Kenya")) == 25            # H2's own figure
    assert len(real.by_country("Mali")) == 16
    assert len(real.by_country("Malawi")) == 8
    assert real.by_country("Guinea-Bissau") != real.by_country("Guinea")
    assert len(real.by_country("Papua New Guinea")) == 3
    # 178 stored values, 137 countries once the spellings are merged
    assert len(real.country_clusters()) == 137


# ==========================================================================
# 3. by_entity and the aliasing — H1
# ==========================================================================

def test_the_spellings_of_one_entity_are_one_group(syn):
    for spelling in ("UNDP", "undp",
                     "United Nations Development Programme",
                     "United Nations Development Programme (UNDP)",
                     "United Nations Development Program, UNDP"):
        assert [r["fp"] for r in syn.by_entity(spelling)] == [1, 2, 3, 4], spelling


def test_an_article_a_corporate_form_and_an_org_tail_do_not_split_a_group(syn):
    want = ["14_a-status", "05_a-fp5", "06_a-fp6", "07_a-fp7"]
    for spelling in ("The World Bank", "World Bank", "world bank",
                     "World Bank Group"):
        assert [r["doc_id"] for r in syn.by_entity(spelling)] == want, spelling


def test_a_shared_acronym_merges_an_english_and_a_french_name(syn):
    assert [r["fp"] for r in syn.by_entity("AFD")] == [9, 10]
    assert [r["fp"] for r in syn.by_entity("French Development Agency")] == [9, 10]
    assert [r["fp"] for r in
            syn.by_entity("Agence Française de Développement")] == [9, 10]


def test_a_different_entity_is_not_swept_into_the_group(syn):
    assert [r["fp"] for r in syn.by_entity("IFC")] == [8]
    assert [r["fp"] for r in syn.by_entity("International Finance Corporation")] == [8]
    assert syn.by_entity("Asian Development Bank") == []


def test_two_short_acronyms_can_never_claim_a_question(syn):
    """'CI' and 'AG' are two characters; an alias that short would match half
    the questions in the corpus. Three characters minimum, two capitals
    minimum, and never a corporate form."""
    assert registry._is_acronym("UNDP") and registry._is_acronym("KfW")
    assert not registry._is_acronym("CI")
    assert not registry._is_acronym("AG")
    assert not registry._is_acronym("Ltd")
    assert not registry._is_acronym("Pegasus")
    assert registry._acronyms("Acumen Fund, LLC.") == []
    assert registry._acronyms("Deutsche Bank AG") == []
    assert registry._acronyms("IUCN - International Union for Conservation") == ["IUCN"]
    assert registry._acronyms("International Fund for Agricultural Development - "
                              "IFAD") == ["IFAD"]
    assert registry._acronyms("Nederlandse Financierings-Maatschappij N.V. (FMO)") \
        == ["FMO"]


def test_the_normalisation_never_edits_the_stored_value(syn):
    """The rows come back exactly as data/registry.json holds them."""
    got = {r["accredited_entity"] for r in syn.by_entity("UNDP")}
    assert got == {"United Nations Development Programme",
                   "United Nations Development Programme (UNDP)",
                   "UNDP", "United Nations Development Program, UNDP"}


def test_the_real_corpus_clusters_the_review_counted(real):
    """H1: 127 unnormalised strings, 'UNDP under at least two spellings'."""
    stored = {r["accredited_entity"] for r in real.load().values()
              if isinstance(r, dict) and r.get("accredited_entity")}
    assert len(stored) == 126
    # 70, not the 69 of the first pass: the owner ruled 'Acumen' and 'Acumen
    # Fund' are two entities, and the alias index now keeps them apart
    # (test_acumen_and_acumen_fund_stay_split).
    assert len(real.entity_clusters()) == 70
    assert len(real.by_entity("UNDP")) == 41           # P1's own denominator
    assert len(real.by_entity("United Nations Development Programme")) == 41
    assert len(real.by_entity("The World Bank")) == 13  # P10's denominator
    assert len(real.by_entity("FAO")) == 21
    assert len(real.by_entity("IFAD")) == 10
    assert len(real.by_entity("IUCN")) == 6


def test_a_french_exonym_reaches_the_entity_the_corpus_records_in_english(real):
    assert real.by_entity("Banque mondiale") == real.by_entity("The World Bank")
    assert real.by_entity("PNUD") == real.by_entity("UNDP")
    assert real.by_entity("Banque africaine de développement") == \
        real.by_entity("African Development Bank")


# ==========================================================================
# 4. the triggers — the guard against a false authoritative note
# ==========================================================================

COUNTRY_POSITIVE = [
    "Which proposals are in Kenya?",
    "How many proposals target Kenya?",
    "List all proposals covering Kenya.",
    "Which FPs cover Kenya?",
    "What projects does the corpus hold for Kenya?",
    "Quelles propositions ciblent le Kenya ?",
    "Combien de propositions concernent le Kenya ?",
    "Liste des propositions au Kenya",
]
COUNTRY_NEGATIVE = [
    "What does FP123 do in Kenya?",                 # document-scoped
    "Which proposal restores mangrove ecosystems in Kenya?",   # singular: find one
    "Kenya is mentioned in FP151.",                 # a statement
    "How much GCF funding went to Kenya?",          # a money ask, not a set ask
    "Which countries does FP151 cover?",            # the OTHER direction
    "Which proposals restore mangroves?",           # a set ask, no country
]


@pytest.mark.parametrize("q", COUNTRY_POSITIVE)
def test_the_country_note_fires_on_a_set_ask(real, q):
    note = real._country_note(q)
    assert note and note.startswith("Registry — 25 funding proposals in the "
                                    "corpus name Kenya in their countries field")


@pytest.mark.parametrize("q", COUNTRY_NEGATIVE)
def test_the_country_note_does_not_fire_on_anything_else(real, q):
    assert real._country_note(q) is None


ENTITY_POSITIVE = [
    "Which proposals does UNDP implement?",
    "How many UNDP proposals are in the corpus?",
    "List every proposal from UNDP.",
    "Which projects are implemented by the United Nations Development Programme?",
    "Quelles propositions sont mises en oeuvre par le PNUD ?",
    "Combien de propositions du PNUD ?",
]
ENTITY_NEGATIVE = [
    "Which proposals cite UNDP's methodology?",     # a mention, no relation
    "What does FP234 say about UNDP?",              # document-scoped
    "UNDP is the accredited entity of FP234.",      # a statement
    "Which proposal does UNDP implement in Tonga?",  # singular: find one
]


@pytest.mark.parametrize("q", ENTITY_POSITIVE)
def test_the_entity_note_fires_on_a_set_ask(real, q):
    note = real._entity_note(q)
    assert note and note.startswith(
        "Registry — 41 funding proposals in the corpus record United Nations "
        "Development Programme as the accredited entity (complete listing over "
        "the 273 corpus documents, covering the 5 spellings of that name the "
        "registry holds): ")


@pytest.mark.parametrize("q", ENTITY_NEGATIVE)
def test_the_entity_note_does_not_fire_on_anything_else(real, q):
    assert real._entity_note(q) is None


def test_an_incidental_world_bank_never_fires_the_inverse_note(real):
    """H1's own guard, spelled out: the name of an entity is not an ask."""
    for q in ["The World Bank is the accredited entity of FP012.",
              "Which proposals cite the World Bank's methodology?",
              "Do FP012 and FP074 of the Africa Hydromet Program target the "
              "same country?",
              "What is the World Bank's role in FP012?"]:
        assert real._entity_note(q) is None, q
    assert real._entity_note("Which proposals are implemented by the World "
                             "Bank?") is not None


# The 2026-08-26 wave added twelve gold cases WRITTEN TO exercise the new
# notes; the fence becomes an exact allowlist so any OTHER question firing
# still fails loudly.
_INVERSE_GOLD = {"agg-inv-undp", "fr-inv-banque-mondiale",
                 "agg-inv-kenya", "fr-inv-senegal",
                 "l2x-inv-iucn-count"}  # 2026-08-26 L2 fill-in wave
_NEW_NOTE_GOLD = _INVERSE_GOLD | {"agg-inv-b35", "agg-corpus-largest",
                                  "agg-2020-least"}


def test_no_gold_case_question_fires_an_inverse_note(real):
    """The pre-wave cases are the regression fence: exactly the four
    inverse-authored cases fire, and nothing else may."""
    import json
    from pathlib import Path
    gold = Path(__file__).resolve().parents[1] / "scripts" / "answer_gold.jsonl"
    fired = [json.loads(ln)["id"] for ln in gold.read_text().splitlines()
             if ln.strip() and (real._country_note(json.loads(ln)["question"])
                                or real._entity_note(json.loads(ln)["question"]))]
    assert sorted(fired) == sorted(_INVERSE_GOLD)


# ==========================================================================
# 5. what the inverse note says
# ==========================================================================

def test_the_country_note_lists_every_match_with_the_count(syn):
    note = syn._country_note("Which proposals are in Kenya?")
    assert note == (
        "Registry — 2 funding proposals in the corpus name Kenya in their "
        "countries field (complete listing over the 15 corpus documents): "
        'FP13 "Regional Facility"; FP15 "Global Markets"')


def test_a_row_without_an_fp_number_is_counted_by_neither_side(syn):
    """`14_a-status` names Kenya and has no FP: the year listing has always
    skipped such rows, and an inverse listing that counted them would state a
    number it could not print."""
    assert len(syn.by_country("Kenya")) == 3
    note = syn._country_note("Which proposals are in Kenya?")
    assert note.startswith("Registry — 2 funding proposals")
    assert "Status of approved proposals" not in note


def test_the_entity_note_names_the_spellings_it_merged(syn):
    note = syn._entity_note("Which proposals does UNDP implement?")
    assert note.startswith(
        "Registry — 4 funding proposals in the corpus record United Nations "
        "Development Programme (UNDP) as the accredited entity (complete "
        "listing over the 15 corpus documents, covering the 4 spellings of "
        "that name the registry holds): ")
    assert 'FP1 "Mali Hydromet"; FP2 "Malawi Resilience"; FP3 "Niger Solar"; ' \
           'FP4 "Nigeria Grid"' in note


def test_a_single_spelling_says_nothing_about_spellings(syn):
    note = syn._entity_note("Which proposals are implemented by IFC?")
    assert "spellings" not in note
    assert note.endswith("(complete listing over the 15 corpus documents): "
                         'FP8 "Serbia Forests"')


def test_an_inverse_listing_over_the_cap_says_so_and_keeps_the_true_count(
        monkeypatch):
    big = {f"{i:03d}_x-fp{i}": {"fp": i, "title": f"Project {i}", "board": 20,
                                "year": 2019, "accredited_entity": "FAO",
                                "countries": ["Kenya"]}
           for i in range(1, 56)}
    monkeypatch.setattr(registry, "_cache", big)
    monkeypatch.setattr(registry, "_cache_v2", {})
    note = registry._country_note("Which proposals are in Kenya?")
    assert note.startswith("Registry — 55 funding proposals in the corpus name "
                           "Kenya in their countries field (50 of 55 listed — "
                           "LIST TRUNCATED, but the count 55 is complete)")
    assert note.endswith("(+5 more)")
    assert 'FP50 "Project 50"' in note and 'FP51 "Project 51"' not in note


def test_the_largest_real_group_is_listed_whole(real):
    """UNDP is 41 documents — the biggest inverse answer this corpus can give,
    and probe P1 named three of them. It arrives complete, or the note is back
    to the defect it was written to fix."""
    note = real._entity_note("Which proposals does UNDP implement?")
    assert "LIST TRUNCATED" not in note and "more)" not in note
    assert note.count('; FP') == 40 and note.startswith("Registry — 41 ")


def test_the_inverse_note_publishes_no_page_and_no_document_stem(real):
    """Same discipline as the year listing: FP numbers only. A stem plus a page
    on one line is what `_note_pages` and `note_page_scopes` turn into a
    citable scope, and a listing has no page to give."""
    note = real._country_note("Which proposals are in Kenya?")
    assert app._note_pages([note]) == set()
    assert V.note_page_scopes(note) == []
    assert "cover pages]" not in note and "(p." not in note


def test_the_inverse_notes_reach_registry_note_itself(real):
    note = real.registry_note("Which proposals does UNDP implement in Kenya?")
    lines = note.splitlines()
    assert any(ln.startswith("Registry — 25 funding proposals") for ln in lines)
    assert any(ln.startswith("Registry — 41 funding proposals") for ln in lines)


def test_an_empty_registry_still_produces_no_note(monkeypatch):
    monkeypatch.setattr(registry, "_cache", {})
    monkeypatch.setattr(registry, "_cache_v2", {})
    assert registry.registry_note("Which proposals are in Kenya?") is None
    assert registry.by_country("Kenya") == [] and registry.by_entity("UNDP") == []


def test_an_error_row_cannot_break_an_index(monkeypatch):
    monkeypatch.setattr(registry, "_cache", {
        "1_ok": {"fp": 1, "title": "T", "countries": ["Kenya"],
                 "accredited_entity": "UNDP"},
        "2_err": {"error": "Unterminated string ..."},
        "3_junk": None,
        "4_types": {"fp": 4, "countries": [None, 7, "Kenya"],
                    "accredited_entity": 12},
    })
    monkeypatch.setattr(registry, "_cache_v2", {})
    assert [r["doc_id"] for r in registry.by_country("Kenya")] == ["1_ok", "4_types"]
    assert [r["doc_id"] for r in registry.by_entity("UNDP")] == ["1_ok"]


def test_the_index_follows_the_rows_it_was_built_from(monkeypatch):
    """The suites monkeypatch `_cache`; an index cached across that swap would
    answer questions about the wrong corpus."""
    monkeypatch.setattr(registry, "_cache", SYN)
    assert len(registry.by_country("Kenya")) == 3
    monkeypatch.setattr(registry, "_cache", {"z_1": {
        "fp": 1, "title": "T", "countries": ["Kenya"], "accredited_entity": "X"}})
    assert len(registry.by_country("Kenya")) == 1


# ==========================================================================
# 6. the consumer sweep, as tests
# ==========================================================================

def test_a_country_name_in_the_question_is_matched_on_word_boundaries(real):
    """The detection side of the Mali/Malawi rule: a note that answered a
    Malawi question with Mali's 16 proposals would be a new false-authority
    misfire, which is the failure class §4.3 counts as the most expensive."""
    mali = real._country_note("Which proposals are in Mali?")
    malawi = real._country_note("Which proposals are in Malawi?")
    assert mali.startswith("Registry — 16 funding proposals in the corpus name Mali ")
    assert malawi.startswith("Registry — 8 funding proposals in the corpus name "
                             "Malawi ")
    niger = real._country_note("Which proposals target Niger?")
    nigeria = real._country_note("Which proposals target Nigeria?")
    assert niger.startswith("Registry — 11 ") and "name Niger in" in niger
    assert nigeria.startswith("Registry — 17 ") and "name Nigeria in" in nigeria


def test_the_longest_name_the_question_spells_wins(real):
    png = real._country_note("Which proposals are in Papua New Guinea?")
    assert "name Papua New Guinea in" in png
    assert real._country_note("Which proposals are in Guinea?").startswith(
        "Registry — 6 funding proposals in the corpus name Guinea ")


def test_the_new_countries_fragment_contradicts_no_claim(real):
    """The count in the fragment is two bare integers ('5 of 44') sitting under
    a field label. `verify._field_conflict` reads the first amount AFTER a
    label as that field's value, so the sweep has to show the marker cannot
    manufacture a contradiction — and that the true count still verifies."""
    ev = V.build_evidence([], real.registry_note("What is FP151?"))
    answer = ("FP151 covers 44 countries. [124_gcf-b27-02-add11, cover pages]\n\n"
              "Its GCF funding requested is 18.5 M USD. "
              "[124_gcf-b27-02-add11, p. 5]")
    claims = V.extract_claims(answer)
    verdicts = V.classify_deterministic(claims, ev)
    assert [v.status for v in verdicts] == [V.SUPPORTED] * len(verdicts)
    assert all(v.status != V.CONTRADICTED for v in verdicts)


GOLD_COUNTRY_CASES = ["txt-cmp-fp012-fp074-country", "cid-fp-86-spaced",
                      "id-fp203-objective", "noisy-typo-fp267", "id-fp234-entity"]


def test_the_gold_cases_that_read_countries_off_the_note_still_match(real):
    """Every gold case whose `expect.fields` names `countries`, replayed
    against the NOTE the new format produces. `txt-cmp-fp012-fp074-country`
    wants Mali and Burkina Faso; `cid-fp-86-spaced` reads a country out of a
    list that is now truncated (8 values, 5 printed) and its alternation has
    to survive the cut."""
    import json
    import re as _re
    from pathlib import Path
    gold = Path(__file__).resolve().parents[1] / "scripts" / "answer_gold.jsonl"
    cases = {c["id"]: c for c in
             (json.loads(ln) for ln in gold.read_text().splitlines() if ln.strip())}
    for cid in GOLD_COUNTRY_CASES:
        case = cases[cid]
        note = real.registry_note(case["question"]) or ""
        for pat in case["expect"]["must_contain"]:
            rx = pat[3:] if pat.startswith("re:") else _re.escape(pat)
            assert _re.search(rx, note), f"{cid}: {pat}"


def test_the_truncated_note_still_prints_the_country_the_case_needs(real):
    """FP086 has eight countries and the case asks for one of them: the cut
    keeps the first five, and 'Kazakhstan' and 'Moldova' are among them."""
    line = real.registry_note("What is FP 86 about, and which countries does "
                              "it cover?").splitlines()[0]
    assert "countries (5 of 8 — list truncated): Bosnia and Herzegovina, " \
           "Kazakhstan, Kyrgyzstan, Moldova, Tajikistan, …" in line
    assert "Ukraine" not in line and "Republic of Serbia" not in line


def test_the_french_ligature_survives_the_folding(real):
    """NFKD has no decomposition for 'œ', so 'mise en œuvre' — the commonest
    French phrasing of 'implemented by' — needs the ligature spelled out."""
    assert registry._fold("mises en œuvre") == "mises en oeuvre"
    for q in ["Quelles propositions sont mises en œuvre par le PNUD ?",
              "Quelles propositions sont mises en oeuvre par le PNUD ?"]:
        assert real._entity_note(q).startswith("Registry — 41 funding proposals")
    note = real._country_note("Quelles propositions ciblent la Côte d'Ivoire ?")
    assert note.startswith("Registry — 13 funding proposals in the corpus name "
                           "Côte d'Ivoire")


# ==========================================================================
# 7. owner rulings on three clusters (decided 2026-08-26)
# ==========================================================================
# entity_clusters() exists so a human can read the normalisation's guesses
# back. Three were read back and ruled on; all three are pinned here, because
# a ruling that lives only in a comment is a ruling the next refactor loses.

def test_the_world_bank_group_stays_merged(real):
    """Ruling (a): 'World Bank Group' = 'The World Bank' — MERGED, upheld.

    P10's denominator, and the merge that made a French question answerable:
    one organisation, 13 documents, four spellings.
    """
    rows = real.by_entity("The World Bank")
    assert len(rows) == 13
    assert real.by_entity("World Bank Group") == rows
    assert real.by_entity("World Bank") == rows
    assert real.by_entity("Banque mondiale") == rows
    assert "World Bank Group" in [v for v in real.entity_clusters()["The World Bank"]]


def test_ministry_of_environment_stays_merged(real):
    """Ruling (b): 'Ministry of Environment' (3 documents) — MERGED, upheld.

    Two spellings of one body, merged by _norm_key's parenthetical rule rather
    than by the org-tail rule. The ruling covers THESE spellings: the long
    Colombian string is a different name and stays where it is.
    """
    rows = real.by_entity("Ministry of Environment")
    assert len(rows) == 3
    assert real.by_entity("Ministry of Environment (MOE)") == rows
    assert {r["accredited_entity"] for r in rows} == \
        {"Ministry of Environment", "Ministry of Environment (MOE)"}
    assert real.by_entity("Ministry of Finance") != rows


def test_acumen_and_acumen_fund_stay_split(real):
    """Ruling (c): 'Acumen' != 'Acumen Fund' — SPLIT, the merge was DECLINED.

    The org-tail rule had pulled the single 'Acumen' row into the six-document
    'Acumen Fund' family on the strength of one word. The owner ruled they are
    distinct canonical entities, so `_OWNER_KEEP_DISTINCT` stops that one
    union — and only that one: the same rule still merges 'World Bank Group'
    above.
    """
    acumen = real.by_entity("Acumen")
    fund = real.by_entity("Acumen Fund")
    assert len(acumen) == 1 and len(fund) == 6
    assert not ({r["doc_id"] for r in acumen} & {r["doc_id"] for r in fund})
    assert [r["accredited_entity"] for r in acumen] == ["Acumen"]
    assert {r["accredited_entity"] for r in fund} == {
        "Acumen Fund", "Acumen Fund Inc", "Acumen Fund, Inc.", "Acumen Fund, LLC."}
    # the corporate-suffix and parenthetical rules keep working INSIDE the
    # family the split created
    assert real.by_entity("Acumen Fund, Inc.") == fund
    assert ("acumen", "acumen fund") in registry._OWNER_KEEP_DISTINCT


def test_the_split_is_an_exception_not_a_new_rule(real):
    """Every other org-tail merge the corpus makes is untouched by the
    exception list — and the non-merge the RULES already produce on their own
    still stands beside the one the owner had to decide.

    'Inter-American Investment Corporation (IDB Invest)' is distinct from
    'Inter-American Development Bank (IDB)' for free: 'IDB Invest' is two
    tokens, so `_is_acronym` refuses it and no acronym union ever fires. That
    is the shape `_OWNER_KEEP_DISTINCT` mirrors — with the difference that a
    declined merge cannot be derived from the strings and has to be written
    down.
    """
    assert len(real.by_entity("World Bank Group")) == 13      # 'group' tail
    assert len(registry._OWNER_KEEP_DISTINCT) == 1
    assert not registry._is_acronym("IDB Invest")
    idb_invest = real.by_entity("Inter-American Investment Corporation")
    assert len(idb_invest) == 1 and len(real.by_entity("IDB")) == 12
    assert idb_invest[0]["doc_id"] not in {r["doc_id"] for r in real.by_entity("IDB")}


# ==========================================================================
# 8. the board inverse (H16) — the asymmetry, closed on the in-range side
# ==========================================================================

def test_by_board_returns_the_rows_of_that_meeting(syn):
    assert [r["fp"] for r in syn.by_board(13)] == [1, 2]
    # the lookup returns every ROW of the meeting, FP-less ones included (the
    # note is what drops them, exactly as it does for a year listing)
    assert [r["doc_id"] for r in syn.by_board(19)] == ["14_a-status", "13_a-fp13"]
    assert syn.by_board(99) == []


def test_by_board_survives_the_rows_a_registry_can_hold(monkeypatch):
    monkeypatch.setattr(registry, "_cache", {
        "1_ok": {"fp": 1, "title": "T", "board": 35},
        "2_err": {"error": "Unterminated string ..."},
        "3_junk": None,
    })
    monkeypatch.setattr(registry, "_cache_v2", {})
    assert [r["doc_id"] for r in registry.by_board(35)] == ["1_ok"]


BOARD_POSITIVE = [
    "Which funding proposals were approved at B.35?",
    "Which proposals were approved at B.35?",
    "List the proposals from B.35.",
    "How many proposals were approved at B.35?",
    "Quelles propositions ont été approuvées à la B.35 ?",
]


@pytest.mark.parametrize("q", BOARD_POSITIVE)
def test_the_board_note_fires_on_a_set_ask(real, q):
    note = real._board_note(q)
    assert len(note) == 1
    assert note[0].startswith(
        "Registry — 7 funding-proposal documents in the corpus are from board "
        "meeting B.35 (2023) (complete listing over the 273 corpus documents): "
        'FP199 "')
    assert 'FP205 "Infrastructure Climate Resilient Fund (ICRF)"' in note[0]


def test_a_bare_what_was_approved_is_a_set_ask(real):
    """The judgement call H16's guard question asks for, decided YES.

    'What was approved at B.42?' carries no set noun at all, and it is still a
    set ask: a board approves a BATCH, so the verb carries the quantifier the
    way 'list EVERY proposal' carries it in a word. The decisive argument is
    the asymmetry — chainlit's out-of-range arm answers this EXACT phrasing
    definitively, so refusing it in range would leave the misfire standing in
    the wording that produced it.
    """
    note = real._board_note("What was approved at B.42?")
    assert len(note) == 1 and note[0].startswith(
        "Registry — 11 funding-proposal documents in the corpus are from board "
        "meeting B.42 (2025)")
    assert real._board_note("What was approved at B.35?")
    assert real._board_note("What did the board approve at B.35?")


def test_the_bare_form_still_needs_the_board_and_the_ask(real):
    """The other half of that call: what the widened trigger must NOT take."""
    for q in ["What was approved yesterday?",              # no board
              "B.35 approved a lot of money.",             # a statement
              "What was approved at B.44?",                # out of range
              "What does section B.3 of FP151 say?",       # a template heading
              "How much was approved at B.35?"]:           # money, not a set
        assert real._board_note(q) == [], q


def test_an_out_of_range_board_still_belongs_to_the_app_note(real):
    """The two arms never both fire, and the definitive one is untouched."""
    q = "Which funding proposals were approved at B.44?"
    assert real._board_note(q) == []
    assert "B.44 is not in this corpus" in app._board_range_note(q)
    inrange = "Which funding proposals were approved at B.35?"
    assert app._board_range_note(inrange) is None
    assert real._board_note(inrange)


def test_a_full_board_code_never_answers_with_the_whole_meeting(real):
    """'GCF/B.35/02/Add.05' names ONE document, and registry_note already
    resolves it. Both gold cases that spell a code keep the note they had."""
    for q in ["What proposal does GCF/B.35/02/Add.05 carry?",
              "Which funding proposals are in GCF/B.35/02/Add.05?",
              "What is in GCF/B.30/02/Add.03?"]:
        assert real._board_note(q) == [], q
    note = real.registry_note("What proposal does GCF/B.35/02/Add.05 carry?")
    assert "GCF/B.35/02/Add.05 resolves to" in note
    assert "are from board meeting" not in note


def test_the_board_note_says_the_registry_records_no_approval(real):
    """The listing answers 'which documents', not 'what was approved', and it
    says so: the board is read off each document's identifier."""
    note = real._board_note("What was approved at B.35?")[0]
    assert note.endswith(
        "— the board is read from each document's own identifier; the registry "
        "records no approval DECISION, so do not report this listing as what "
        "the board approved")


def test_two_boards_get_two_lines(real):
    notes = real._board_note("Which proposals were approved at B.35 and B.36?")
    assert len(notes) == 2
    assert "B.35 (2023)" in notes[0] and "B.36 (2023)" in notes[1]


def test_the_board_listing_publishes_no_page_and_no_stem(real):
    """Same discipline as the year and inverse listings: FP numbers only."""
    note = real._board_note("Which proposals were approved at B.35?")[0]
    assert app._note_pages([note]) == set()
    assert V.note_page_scopes(note) == []
    assert "(p." not in note and "cover pages]" not in note


def test_the_board_note_reaches_registry_note(real):
    lines = real.registry_note("Which proposals were approved at B.35?").splitlines()
    assert any(ln.startswith("Registry — 7 funding-proposal documents in the "
                             "corpus are from board meeting B.35") for ln in lines)


def test_the_board_listing_states_its_own_completeness(syn):
    note = syn._board_note("Which proposals were approved at B.19?")[0]
    assert note.startswith(
        "Registry — 1 funding-proposal documents in the corpus are from board "
        "meeting B.19 (2018) (complete listing over the 15 corpus documents): "
        'FP13 "Regional Facility"')          # the FP-less status row is not listed


# ==========================================================================
# 9. corpus extrema (H3/H3b, probes P3 and F15)
# ==========================================================================

def test_the_corpus_minimum_answers_p3(real):
    """P3 asked for the smallest GCF request in the corpus and was refused.

    The note now answers it — over the population it can rank, with the
    denominator, and with the figure that is SMALLER still named rather than
    silently dropped or silently called USD.
    """
    lines = real._extrema_note("What is the smallest GCF funding request in "
                               "the corpus?")
    assert lines[0].startswith(
        'Registry — SMALLEST GCF funding requested in the corpus: FP35 '
        '"Climate Information Services for Resilient De…" — 22,953 USD '
        "(p.10, B.2(b)) [240_gcf-b15-13-add08]. Ranked over the 188 of 273 "
        "documents in the corpus whose registry figure is an unambiguous USD "
        "amount.")
    assert lines[1] == (
        "Registry — excluded from that comparison: 85 of 273 documents in the "
        "corpus — 29 state no figure this registry could read for the field; "
        "28 print a figure whose unit the document's own mantissa contradicts; "
        "23 state EUR; 5 print no currency at all. Figures in different "
        "currencies are never ranked against one another, and a figure printed "
        "without a currency is not assumed to be USD. 1 of the excluded "
        "figures is nominally smaller than the ranked answer, named below.")
    assert lines[2] == (
        "Registry — NOT RANKED, though nominally smaller than the figure "
        "above: FP2 prints 16,265 (p.8, B.2(b)) [272_gcf-b11-04-add02] — the "
        "document states NO currency for this figure — do not assume USD. "
        "Report it only with that caveat, and never as the smallest figure in "
        "the corpus.")
    assert len(lines) == 3


def test_the_review_s_own_truth_value_is_the_one_the_note_publishes(real):
    """§7's P3 row records the truth as 'FP2 = 16,265'. It is in the note —
    and it is NOT the ranked answer, because the page that prints it prints no
    currency and 'do not assume it is USD' is planner.Comparability's own law.
    Naming it beside the answer is the honest half of that refusal: the reader
    sees the number, the reason, and the page."""
    note = "\n".join(real._extrema_note("smallest GCF request in the corpus?"))
    assert "FP2 prints 16,265 (p.8, B.2(b))" in note
    assert "USD 16,265" not in note


def test_the_corpus_maximum(real):
    lines = real._extrema_note("What is the largest GCF funding request ever?")
    assert lines[0].startswith(
        'Registry — LARGEST GCF funding requested in the corpus: FP241 ')
    assert "— 2149 million USD (p.5, A.8) " \
           "[35_gcf-b39-02-add16-rev01-funding-proposal-package-fp241]" in lines[0]
    assert "CAUTION" not in lines[0]         # read from a labelled A.8 cell
    assert len(lines) == 2                   # nothing excluded ranks above it


def test_a_figure_read_without_a_template_heading_says_so(real):
    """The corpus minimum comes from a 'rule:' section — a figure the builder
    found on the page without the heading above it. An authoritative
    superlative built on the weaker provenance has to disclose it, or the note
    manufactures the very kind of certainty F13 was about."""
    line = real._extrema_note("smallest GCF funding request in the corpus?")[0]
    assert ("CAUTION: this figure was read from p.10 without a labelled "
            "template heading above it") in line


def test_the_extrema_lines_publish_one_creditable_scope_each(real):
    """The one note family in this module that DOES carry provenance: an
    extremum is one figure in one document, which is exactly what
    `_note_pages` and `note_page_scopes` can credit. One document per line, so
    no line pairs one document's stem with another's page."""
    lines = real._extrema_note("What is the smallest GCF funding request in "
                               "the corpus?")
    assert app._note_pages(lines) == {("240_gcf-b15-13-add08", 10),
                                      ("272_gcf-b11-04-add02", 8)}
    assert V.note_page_scopes(lines[0]) == [
        (V.note_scope_doc("240_gcf-b15-13-add08"), 10)]
    assert V.note_page_scopes(lines[1]) == []
    assert V.note_page_scopes(lines[2]) == [
        (V.note_scope_doc("272_gcf-b11-04-add02"), 8)]


def test_the_extremum_claim_verifies_against_its_own_note(real):
    """The point of carrying the pointer: an answer that cites the page the
    note prints is grounded, not flagged."""
    note = real.registry_note("What is the smallest GCF funding request in "
                              "the corpus?")
    ev = V.build_evidence([], note)
    claims = V.extract_claims(
        "The smallest GCF funding requested the registry can rank is "
        "22,953 USD, for FP35. [240_gcf-b15-13-add08, p. 10]")
    verdicts = V.classify_deterministic(claims, ev)
    assert verdicts and all(v.status != V.CONTRADICTED for v in verdicts)


EXTREMA_NEGATIVE = [
    ("Which of FP012 and FP074 requested the largest GCF funding?",
     "a named comparison is planner's, with its own comparability verdict"),
    ("Compare the GCF funding requested by FP151 and FP152 — which is larger?",
     "same, in the shape S8 actually tests"),
    ("Which 2020 proposal requested the largest GCF funding?",
     "agg-2020-largest: a year-scoped MAXIMUM already answered at 1.00"),
    ("Which proposal requested the most GCF funding?",
     "no scope word: 'the most' could mean 'of the two above'"),
    ("Which years have the most funding proposals in this corpus?",
     "agg-year-most: a count of documents, not of money"),
    ("How many proposals in the corpus request GCF funding?",
     "no superlative at all"),
    ("Does the corpus hold at least one proposal over 100 million USD?",
     "'at least' is not a superlative"),
    ("What is the largest and smallest GCF request in the corpus?",
     "both directions at once is not a clean ask"),
    ("What is the smallest country in the corpus?",
     "no money field named"),
    ("Which country received the most GCF funding in the corpus?",
     "a sum across documents — the operation F11 got 21-35x wrong and the one "
     "planner refuses outright; the biggest single figure is not an answer"),
    ("Which entity has the largest GCF funding request in the corpus?",
     "same: an entity total is a sum, and this registry computes none"),
    ("Quel pays a reçu le plus de financement du GCF dans le corpus ?",
     "the French twin of the same category error (P10's shape, with a number)"),
]


@pytest.mark.parametrize("q,why", EXTREMA_NEGATIVE)
def test_the_extrema_note_does_not_fire_on_anything_else(real, q, why):
    assert real._extrema_note(q) == [], why


def test_the_year_arm_fires_only_for_the_minimum(real):
    """F15 asked the year-scoped MINIMUM and got a wrong value out of a note
    that printed the right one — the model compared '18.5 M USD' with
    '17,198,843 USD' by mantissa. That arm is computed here. The MAXIMUM arm
    of the same shape is `agg-2020-largest`, answered at 1.00 in every recorded
    run by chainlit's year note, and a second authoritative note over a passing
    case buys nothing; it is available the moment the question says 'corpus'.
    """
    lines = real._extrema_note("Which 2020 proposal requested the least?")
    assert lines[0].startswith(
        "Registry — SMALLEST GCF funding requested in the 2020 funding "
        'proposals: FP129 "Afghanistan Rural Energy Market Transformatio…" — '
        "USD 17,198,843 (p.5, A8) [146_gcf-b26-02-add01]. Ranked over the 21 "
        "of 30 documents in the 2020 funding proposals whose registry figure "
        "is an unambiguous USD amount.")
    assert lines[1].startswith("Registry — excluded from that comparison: 9 "
                               "of 30 documents in the 2020 funding proposals")
    assert real._extrema_note("Which 2020 proposal requested the largest GCF "
                              "funding?") == []
    assert real._extrema_note("Which 2020 proposal in the corpus requested the "
                              "largest GCF funding?")[0].startswith(
        "Registry — LARGEST GCF funding requested in the 2020 funding proposals")


def test_a_second_year_makes_it_a_range_and_a_range_is_not_this_question(real):
    assert real._extrema_note("smallest GCF request between 2019 and 2021?") == []


def test_total_financing_is_a_different_field(real):
    line = real._extrema_note("What is the largest total financing in the "
                              "corpus?")[0]
    assert line.startswith("Registry — LARGEST total financing in the corpus: "
                           "FP241 ")
    assert "3762 million USD (p.5, A.7)" in line


def test_the_french_form_answers_the_same_question(real):
    en = real._extrema_note("What is the smallest GCF funding request in the "
                            "corpus?")
    fr = real._extrema_note("Quelle est la plus petite demande de financement "
                            "du GCF dans le corpus ?")
    assert fr == en


def test_a_corpus_with_no_readable_figure_refuses_out_loud(syn):
    """The SYN corpus has no schema-2 facts at all: there is nothing to rank,
    and the note says so rather than falling silent — a silent note is what
    lets a model invent the superlative itself."""
    lines = syn._extrema_note("What is the smallest GCF funding request in the "
                              "corpus?")
    # 14, not 15: the FP-less status row is not a funding proposal and is no
    # more countable here than it is listable in an inverse note.
    assert lines == [
        "Registry — no document in the corpus states a GCF funding requested "
        "figure in USD that this registry could read (14 checked), so the "
        "registry supports NO smallest-figure answer here. Say so; do not rank "
        "the figures the documents state in other currencies against one "
        "another."]


MONEY = {
    "a1": {"fp": 1, "title": "Alpha", "board": 35, "year": 2023},
    "a2": {"fp": 2, "title": "Beta", "board": 35, "year": 2023},
    "a3": {"fp": 3, "title": "Gamma", "board": 35, "year": 2023},
    "a4": {"fp": 4, "title": "Delta", "board": 35, "year": 2023},
}


def _cand(raw, value, currency, page, section="A.8"):
    return {"raw": raw, "value": value, "currency": currency, "unit": None,
            "page": page, "section": section, "status": "canonical"}


MONEY_V2 = {
    "a1": {"fp": 1, "facts": {"gcf_funding_requested": [
        _cand("1,000,000 USD", 1000000.0, "USD", 5)]}},
    "a2": {"fp": 2, "facts": {"gcf_funding_requested": [
        _cand("1,000,000 USD", 1000000.0, "USD", 6)]}},
    "a3": {"fp": 3, "facts": {"gcf_funding_requested": [
        _cand("900,000 Eur", 900000.0, "EUR", 7)]}},
    "a4": {"fp": 4, "facts": {"gcf_funding_requested": [
        _cand("28,654 million USD", None, "USD", 8)]}},
}


@pytest.fixture
def money(monkeypatch):
    monkeypatch.setattr(registry, "_cache", MONEY)
    monkeypatch.setattr(registry, "_cache_v2", MONEY_V2)
    return registry


def test_a_tie_is_stated_not_broken_silently(money):
    lines = money._extrema_note("smallest GCF request in the corpus?")
    assert lines[0].startswith(
        'Registry — SMALLEST GCF funding requested in the corpus: FP1 "Alpha" '
        "— 1,000,000 USD (p.5, A.8) [a1] (TIED at this figure with FP2). "
        "Ranked over the 2 of 4 documents")


def test_a_cheaper_euro_figure_is_named_and_never_ranked(money):
    """The planner's law, in the one place a note could quietly break it: 900k
    EUR is nominally below the USD minimum and is NOT the answer."""
    lines = money._extrema_note("smallest GCF request in the corpus?")
    assert lines[1] == (
        "Registry — excluded from that comparison: 2 of 4 documents in the "
        "corpus — 1 print a figure whose unit the document's own mantissa "
        "contradicts; 1 state EUR. Figures in different currencies are never "
        "ranked against one another, and a figure printed without a currency "
        "is not assumed to be USD. 1 of the excluded figures is nominally "
        "smaller than the ranked answer, named below.")
    assert lines[2] == (
        "Registry — NOT RANKED, though nominally smaller than the figure "
        "above: FP3 prints 900,000 Eur (p.7, A.8) [a3] — the figure is stated "
        "in EUR, and this corpus carries no conversion rule. Report it only "
        "with that caveat, and never as the smallest figure in the corpus.")


def test_the_maximum_states_that_nothing_excluded_reaches_it(money):
    """The count is printed in BOTH directions, like every other list this
    module emits: 'None of the excluded figures is nominally larger' is a
    fact about the exclusions, and its absence would leave a reader guessing
    whether any had been dropped."""
    lines = money._extrema_note("largest GCF request in the corpus?")
    assert lines[1].endswith("None of the excluded figures is nominally larger "
                             "than the ranked answer.")
    assert len(lines) == 2


def test_more_named_exclusions_than_the_cap_say_they_were_cut(monkeypatch):
    """A cut list that does not say it was cut is F13 in a new place."""
    rows = {"u1": {"fp": 1, "title": "Ranked", "year": 2023, "board": 35}}
    facts = {"u1": {"fp": 1, "facts": {"gcf_funding_requested": [
        _cand("9,000,000 USD", 9000000.0, "USD", 5)]}}}
    for i in range(2, 6):                     # four EUR figures, all smaller
        rows[f"e{i}"] = {"fp": i, "title": f"Euro {i}", "year": 2023, "board": 35}
        facts[f"e{i}"] = {"fp": i, "facts": {"gcf_funding_requested": [
            _cand(f"{i} million Eur", i * 1000000.0, "EUR", 5)]}}
    monkeypatch.setattr(registry, "_cache", rows)
    monkeypatch.setattr(registry, "_cache_v2", facts)
    lines = registry._extrema_note("smallest GCF request in the corpus?")
    assert lines[1].endswith(
        "4 of the excluded figures are nominally smaller than the ranked "
        "answer; the 2 most extreme are named below — LIST TRUNCATED.")
    assert len(lines) == 4                    # answer + exclusions + the 2 named
    assert "FP2 prints 2 million Eur" in lines[2]      # most extreme first
    assert "FP3 prints 3 million Eur" in lines[3]


def test_an_unrankable_unit_is_counted_and_never_named_as_a_rival(money):
    """'28,654 million USD' has no value in schema 2 — it cannot be smaller or
    larger than anything, so it is counted among the exclusions and never
    appears on a NOT RANKED line."""
    note = "\n".join(money._extrema_note("largest GCF request in the corpus?"))
    assert "1 print a figure whose unit the document's own mantissa contradicts" \
        in note
    assert "FP4" not in note


def test_the_extrema_notes_reach_registry_note_itself(real):
    note = real.registry_note("What is the smallest GCF funding request in the "
                              "corpus?")
    assert note.count("\n") == 2
    assert note.startswith("Registry — SMALLEST GCF funding requested")


def test_an_empty_registry_produces_none_of_the_new_notes(monkeypatch):
    monkeypatch.setattr(registry, "_cache", {})
    monkeypatch.setattr(registry, "_cache_v2", {})
    assert registry.registry_note("What was approved at B.35?") is None
    assert registry.registry_note("smallest GCF request in the corpus?") is None
    assert registry.by_board(35) == []


def test_no_gold_case_question_fires_any_new_note(real):
    """The fence, extended to the two notes this pass adds. The 89 recorded
    cases measure the system's behaviour; a case that silently gained an
    authoritative note would be a behaviour change no run had measured."""
    import json
    from pathlib import Path
    gold = Path(__file__).resolve().parents[1] / "scripts" / "answer_gold.jsonl"
    fired = []
    for ln in gold.read_text().splitlines():
        if not ln.strip():
            continue
        case = json.loads(ln)
        q = case["question"]
        if (real._country_note(q) or real._entity_note(q) or real._board_note(q)
                or real._extrema_note(q)):
            fired.append(case["id"])
    assert sorted(fired) == sorted(_NEW_NOTE_GOLD)
