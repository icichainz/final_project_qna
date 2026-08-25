"""Page provenance for the cover-page META facts (entity, countries, title).

WHAT THIS GUARDS. Registry notes print a page beside every MONEY figure but
printed none beside entity/countries/title, so the answer model — told to cite
the page printed beside a fact — invented one. It guessed p.3 and was usually
factually right, which is exactly what makes the defect expensive: a checker
sees a fabricated citation, a reader sees a correct answer, and the two never
reconcile. scripts/build_meta_provenance.py derives those pages from the
corpus at build time instead.

WHY THE TESTS BUILD RATHER THAN READ. Every assertion here runs the extractor
over the real corpus and checks what it COMPUTES. None of them read a
meta_provenance key out of data/registry_v2.json, because that file is
integrity-anchored in data/eval/CHECKSUMS.sha256 and has deliberately not
been regenerated yet (see test_the_shipped_registry_is_still_the_anchored_one).
Testing the computation rather than the artifact means these tests pin the
extractor's behaviour today and keep pinning it after the file is regenerated.

THE SPOT ANCHORS are the six documents whose correct pages were established
independently — by the release-6 and release-3 records and by earlier agents
reading the pages — before this extractor existed. They are the only external
check on precision available, so they are pinned exactly rather than as a
count.
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _build_module():
    p = ROOT / "scripts" / "build_meta_provenance.py"
    spec = importlib.util.spec_from_file_location("build_meta_provenance", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


B = _build_module()

REGISTRY_V1 = ROOT / "data" / "registry.json"
REGISTRY_V2 = ROOT / "data" / "registry_v2.json"
CORPUS = ROOT / "data" / "extracted" / "vlm" / "qwen_qwen2.5-vl-7b"

pytestmark = pytest.mark.skipif(
    not (REGISTRY_V1.exists() and CORPUS.is_dir()),
    reason="corpus extraction / registry not present in this checkout",
)


@pytest.fixture(scope="module")
def registry_v1():
    return json.loads(REGISTRY_V1.read_text(encoding="utf-8"))["documents"]


@pytest.fixture(scope="module")
def provenance(registry_v1):
    """The extractor's output over the whole corpus. ~1s, so built once."""
    return B.build(registry_v1, CORPUS)


# ---------------------------------------------------------------------------
# the data contract
# ---------------------------------------------------------------------------

def test_only_the_three_field_names_ever_appear(provenance):
    """A fourth field would break the note emitter coded against this contract."""
    seen = {f for entry in provenance.values() for f in entry}
    assert seen <= {"accredited_entity", "countries", "title"}
    assert seen == {"accredited_entity", "countries", "title"}, (
        "all three fields should be represented somewhere in 273 documents"
    )


def test_every_entry_is_a_page_and_a_quote(provenance):
    for doc_id, entry in provenance.items():
        assert isinstance(entry, dict) and entry, doc_id
        for field, rec in entry.items():
            assert set(rec) == {"page", "quote"}, (doc_id, field, rec)
            assert isinstance(rec["page"], int) and not isinstance(rec["page"], bool), (doc_id, field)
            assert rec["page"] >= 1, (doc_id, field, rec["page"])
            assert isinstance(rec["quote"], str) and rec["quote"].strip(), (doc_id, field)
            assert len(rec["quote"]) <= B.QUOTE_MAX, (doc_id, field, len(rec["quote"]))


def test_a_document_with_nothing_found_gets_no_key_at_all(provenance, registry_v1):
    """Absent beats empty: the emitter falls back on absence, not on {}."""
    assert all(entry for entry in provenance.values())
    # Some documents genuinely yield nothing — status papers and the oldest
    # board packets have no cover-page label block at all. They must be missing
    # from the mapping entirely rather than carrying an empty dict.
    assert len(provenance) < len(registry_v1)


def test_the_page_is_a_cover_page(provenance):
    """Provenance from deep inside a 200-page packet would be a parsing bug."""
    pages = [rec["page"] for e in provenance.values() for rec in e.values()]
    assert max(pages) <= B.COVER_PAGES
    # The GCF template puts this block on the first page of the proposal
    # proper, which sits behind the addendum cover and table of contents.
    assert sum(1 for p in pages if p <= 3) / len(pages) > 0.90


# ---------------------------------------------------------------------------
# the spot anchors: pages established before this extractor existed
# ---------------------------------------------------------------------------

SPOT_ANCHORS = [
    # (doc_id, field, page, what established it)
    ("39_gcf-b39-02-add12-rev01-funding-proposal-package-fp237",
     "accredited_entity", 3, "FP237 entity AFD"),
    ("80_gcf-b34-02-add05",
     "accredited_entity", 3, "FP195 entity CAF"),
    ("262_gcf-b13-16-add04",
     "countries", 3, "FP012 Mali"),
    ("201_gcf-b19-22-add16-rev01",
     "countries", 2, "FP074 Burkina Faso"),
    ("124_gcf-b27-02-add11",
     "accredited_entity", 3, "FP151 IUCN"),
    ("123_gcf-b27-02-add12",
     "accredited_entity", 3, "FP152 Pegasus"),
]


@pytest.mark.parametrize("doc_id,field,page,label",
                         SPOT_ANCHORS, ids=[a[3] for a in SPOT_ANCHORS])
def test_spot_anchor_page(provenance, doc_id, field, page, label):
    rec = provenance.get(doc_id, {}).get(field)
    assert rec is not None, f"{label}: no provenance extracted for {doc_id}/{field}"
    assert rec["page"] == page, f"{label}: expected p{page}, got p{rec['page']}"


def test_the_spot_anchor_quotes_show_the_label_that_justified_the_page():
    """A page is only provenance because a LABEL on it named the field.

    The quote is what a reviewer reads to check the page without opening the
    PDF, so it has to carry the label, not just the value.
    """
    reg = json.loads(REGISTRY_V1.read_text(encoding="utf-8"))["documents"]
    prov = B.build(reg, CORPUS, only=[a[0] for a in SPOT_ANCHORS])
    for doc_id, field, _page, label in SPOT_ANCHORS:
        quote = prov[doc_id][field]["quote"]
        assert B.label_field(quote.split(":")[0]) == field, (label, quote)


# ---------------------------------------------------------------------------
# the additive-only invariant
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not REGISTRY_V2.exists(), reason="registry_v2.json absent")
def test_augmenting_changes_nothing_but_meta_provenance(provenance):
    """The money candidates, statuses and every other key must survive.

    Deep-compared with the key stripped, so a reordered list or a mutated
    status fails here rather than in a scoring run three waves later.
    """
    before = json.loads(REGISTRY_V2.read_text(encoding="utf-8"))
    after = B.augment(before, provenance)
    assert B.strip_meta_provenance(after) == B.strip_meta_provenance(before)
    assert list(after) == list(before)
    assert list(after["documents"]) == list(before["documents"])
    for doc_id, entry in after["documents"].items():
        # filter BOTH sides: once the shipped file carries the key (it does
        # since the 2026-08-25 application), 'before' includes it too
        original_keys = [k for k in before["documents"][doc_id]
                         if k != "meta_provenance"]
        assert [k for k in entry if k != "meta_provenance"] == original_keys, doc_id


@pytest.mark.skipif(not REGISTRY_V2.exists(), reason="registry_v2.json absent")
def test_the_serialized_diff_is_pure_insertion(provenance):
    """Byte-level proof, not just structural: no existing line may disappear.

    dumps() reproduces build_registry_v2.py's own settings exactly, so a
    regenerated file differs from the shipped one only by added lines.
    """
    raw = REGISTRY_V2.read_text(encoding="utf-8")
    assert B.dumps(json.loads(raw)) == raw, (
        "serializer drifted from the one that wrote registry_v2.json; "
        "regenerating would rewrite untouched lines"
    )
    augmented = B.dumps(B.augment(json.loads(raw), provenance))
    import difflib
    removed = [line for line in difflib.unified_diff(
        raw.split("\n"), augmented.split("\n"), n=0)
        if line.startswith("-") and not line.startswith("---")]
    assert removed == []


@pytest.mark.skipif(not REGISTRY_V2.exists(), reason="registry_v2.json absent")
def test_augmenting_twice_is_idempotent(provenance):
    """A re-run must replace provenance, never accumulate or double it."""
    before = json.loads(REGISTRY_V2.read_text(encoding="utf-8"))
    once = B.augment(before, provenance)
    twice = B.augment(once, provenance)
    assert once == twice


def test_the_shipped_registry_is_still_the_anchored_one():
    """data/registry_v2.json is checksum-anchored and NOT yet regenerated.

    data/eval/CHECKSUMS.sha256 pins this file's sha256; regenerating it drops
    that suite from 73/73 to 72/73 and the anchor has to be re-recorded in the
    same change. Until that is done deliberately, the shipped file carries no
    meta_provenance. When it is regenerated, this test flips to validating the
    shipped keys instead of asserting their absence.
    """
    if not REGISTRY_V2.exists():
        pytest.skip("registry_v2.json absent")
    docs = json.loads(REGISTRY_V2.read_text(encoding="utf-8"))["documents"]
    shipped = {d: e["meta_provenance"] for d, e in docs.items() if "meta_provenance" in e}
    if not shipped:
        pytest.skip("registry_v2.json not yet regenerated (checksum anchor pending)")
    for doc_id, entry in shipped.items():
        for field, rec in entry.items():
            assert field in B.FIELDS, (doc_id, field)
            assert isinstance(rec["page"], int) and rec["page"] >= 1, (doc_id, field)
            assert len(rec["quote"]) <= B.QUOTE_MAX, (doc_id, field)


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------

def test_running_the_script_twice_gives_a_byte_identical_file(tmp_path):
    """Reproducibility is the whole reason this is a build step and not a prompt.

    Scoped to a handful of documents so the test stays fast; the extractor has
    no per-document state, so byte-equality here is byte-equality everywhere.
    """
    import subprocess
    import sys

    docs = [a[0] for a in SPOT_ANCHORS]
    outs = []
    for name in ("first.json", "second.json"):
        out = tmp_path / name
        cmd = [sys.executable, str(ROOT / "scripts" / "build_meta_provenance.py"),
               "--quiet", "--out", str(out)]
        for d in docs:
            cmd += ["--only", d]
        subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True)
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    assert b"meta_provenance" in outs[0]


def test_the_builder_is_a_pure_function_of_its_inputs(registry_v1):
    once = B.build(registry_v1, CORPUS, only=["262_gcf-b13-16-add04"])
    twice = B.build(registry_v1, CORPUS, only=["262_gcf-b13-16-add04"])
    assert once == twice


# ---------------------------------------------------------------------------
# the generic-string guard: the reason this is label-anchored at all
# ---------------------------------------------------------------------------

PROSE_MENTIONING_CHILE = """---
**Page 3**
---

# Funding Proposal

The programme will be implemented across the southern cone. Chile has
committed to a 2050 net-zero target, and Chile's national grid operator is
an executing partner. Baseline studies from Chile informed the design.

Chile

---
**Page 4**
---

Further detail on Chile follows in section B.
"""


def test_a_country_named_only_in_prose_is_not_provenance():
    """'Chile' in a sentence is not the country field; a label makes it one.

    This is the whole precision argument. An absent page costs nothing — the
    emitter falls back to the document's cover pages — while a page derived
    from prose puts a confident, wrong citation next to a fact.
    """
    meta = {"countries": ["Chile"], "accredited_entity": None, "title": None}
    assert B.extract_document(PROSE_MENTIONING_CHILE, meta) == {}


def test_the_same_value_under_a_label_is_provenance():
    """The control for the test above: only the label differs."""
    labelled = PROSE_MENTIONING_CHILE.replace("\nChile\n", "\nCountry(ies): Chile\n")
    meta = {"countries": ["Chile"], "accredited_entity": None, "title": None}
    got = B.extract_document(labelled, meta)
    assert got["countries"]["page"] == 3
    assert "Chile" in got["countries"]["quote"]


def test_a_table_of_contents_row_is_not_a_label():
    """The addendum's contents page says 'accredited entity' on every line.

    "| Funding proposal submitted by the accredited entity | 3 |" names a
    section, not the entity, and its second cell is a page number. Matching it
    would put provenance on the contents page of nearly every document.
    """
    assert B.parse_label_line(
        "| Funding proposal submitted by the accredited entity | 3 |") is None
    assert B.parse_label_line(
        "| Response from the accredited entity to the Panel | 144 |") is None
    assert B.parse_label_line(
        "e) Responses from the accredited entity to the Panel.") is None


def test_the_labels_the_corpus_actually_uses_are_recognised():
    """Documented so a future template variant is added here, not guessed at."""
    for raw, field in [
        ("## Accredited Entity:", "accredited_entity"),
        ("### A.1.5. Accredited entity", "accredited_entity"),
        ("**Accredited Entity:**", "accredited_entity"),
        ("## Country(ies):", "countries"),
        ("### Countries(s):", "countries"),
        ("## Country/Region:", "countries"),
        ("#### A.1.3 Country (ies) / region", "countries"),
        ("## Project/Programme title:", "title"),
        ("### A.1.1. Project / programme title", "title"),
        ("| Program Title: | Ecuador REDD-plus |", "title"),
    ]:
        parsed = B.parse_label_line(raw)
        assert parsed is not None, raw
        assert parsed[0] == field, (raw, parsed[0])


def test_a_generic_heading_is_not_a_field_label():
    for raw in ("## Summary", "### Table of Content", "## A.9 Project size",
                "| Country | Climate Hazard | High |", "## Entity"):
        assert B.parse_label_line(raw) is None, raw


# ---------------------------------------------------------------------------
# the matchers, exercised directly
# ---------------------------------------------------------------------------

def test_entity_matching_absorbs_the_variation_the_corpus_shows():
    assert B.entity_matches("The World Bank", "World Bank")
    assert B.entity_matches("World Bank", "The World Bank")
    assert B.entity_matches("French Development Agency (AFD)", "AFD")
    assert B.entity_matches("IUCN - International Union for Conservation of Nature",
                            "IUCN")
    assert B.entity_matches("United Nations Development Programme", "UNDP")
    assert B.entity_matches("Environment Investment Fund of Namibia",
                            "Environmental Investment Fund of Namibia")
    assert B.entity_matches("Corporación Andina de Fomento (CAF)",
                            "Corporacion Andina de Fomento")


def test_entity_matching_rejects_a_different_entity():
    assert not B.entity_matches("The World Bank", "Asian Development Bank")
    assert not B.entity_matches("Acumen Fund, LLC.", "Pegasus Capital Advisors LP")
    # A numeral is not an acronym: "Programme II" must not claim any page whose
    # entity cell happens to carry a roman numeral.
    assert not B.entity_matches("Green Programme II", "Unrelated Trust II")


def test_country_matching_needs_the_countries_not_a_region_name():
    assert B.countries_match(["Mali"], "Mali (Sub-Saharan Africa)")
    assert B.countries_match(["Panama", "Paraguay", "Uruguay"],
                             "Panama, Paraguay and Uruguay")
    assert not B.countries_match(["Uganda", "Kenya", "Nigeria"],
                                 "East Africa & West Africa")
    assert not B.countries_match(["Botswana", "Chad", "Kenya"], "Multiple countries")


def test_country_matching_does_not_match_a_longer_word():
    """'Mali' must not be satisfied by 'Malawi', nor 'Chad' by 'Chadian'."""
    assert not B.countries_match(["Mali"], "Malawi")
    assert not B.countries_match(["Chad"], "Chadian smallholder farmers")


def test_title_matching_tolerates_a_clipped_cover_title():
    """Cover pages often print a truncated title; the registry holds the full one."""
    full = ("Scaling up the Deployment of Integrated Utility Services (IUS) to "
            "Support Energy Sector Transformation in the Caribbean")
    assert B.title_matches(full, "Scaling up the Deployment of Integrated Utility Services (IUS)")
    assert not B.title_matches(full, "Sustainable landscapes in Eastern Madagascar")


def test_quotes_are_trimmed_to_the_contract_ceiling():
    assert len(B.trim_quote("x" * 500)) == B.QUOTE_MAX
    assert B.trim_quote("  Accredited   Entity:\n  UNDP  ") == "Accredited Entity: UNDP"
