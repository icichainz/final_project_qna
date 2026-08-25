"""Board-code resolution (agenda item is part of the identifier), the registry
build's resume filter, and the schema-2 provenance the note prints.

The note tests pin the exact strings the answer model and the verifier read:
the v2 money fragment with its page and section, the conflict warnings, and —
byte for byte — the v1 note that survives when registry_v2.json does not.
"""
import importlib.util
from pathlib import Path

import pytest

from gcf_qna.rag import registry

# b30 carries two series in the real corpus: b30-02-* and b30-03-*.
REG = {
    "104_gcf-b30-03-add03-funding-proposal-package-fp172":
        {"fp": 172, "title": "Bhutan Glacial", "accredited_entity": "UNDP",
         "countries": ["Bhutan"], "board": 30, "year": 2021,
         "gcf_financing": None, "total_financing": None},
    "103_gcf-b30-03-add04-funding-proposal-package-fp173":
        {"fp": 173, "title": "Sahel Resilience", "accredited_entity": "IFAD",
         "countries": ["Mali"], "board": 30, "year": 2021,
         "gcf_financing": None, "total_financing": None},
    "106_gcf-b30-02-add01":
        {"fp": None, "title": "Consideration of funding proposals",
         "accredited_entity": None, "countries": [], "board": 30, "year": 2021,
         "gcf_financing": None, "total_financing": None},
    "02_gcf-b42-02-add16-funding-proposal-package-fp274":
        {"fp": 274, "title": "BRACE", "accredited_entity": "SC Australia",
         "countries": [], "board": 42, "year": 2025,
         "gcf_financing": None, "total_financing": None},
}


@pytest.fixture
def reg(monkeypatch):
    monkeypatch.setattr(registry, "_cache", REG)
    return registry


def test_stated_item_must_match(reg):
    # GCF/B.30/02/Add.03 does not exist: the b30-03-add03 doc is NOT it
    assert reg.resolve_board_code(30, 3, 2) is None
    n = reg.registry_note("Summarize GCF/B.30/02/Add.03")
    assert "NOT FOUND" in n and "GCF/B.30/02/Add.03" in n
    assert "Bhutan Glacial" not in n


def test_item_and_padding_preserved_in_note(reg):
    row = reg.resolve_board_code(30, 4, 3)
    assert row["doc_id"] == "103_gcf-b30-03-add04-funding-proposal-package-fp173"
    n = reg.registry_note("Summarize GCF/B.30/03/Add.04")
    assert "GCF/B.30/03/Add.04" in n
    assert "/02/" not in n
    assert "FP173" in n and "NOT FOUND" not in n


def test_no_item_falls_back_to_add_only(reg):
    row = reg.resolve_board_code(30, 3)
    assert row["doc_id"] == "104_gcf-b30-03-add03-funding-proposal-package-fp172"
    n = reg.registry_note("What is in B.30/Add.03?")
    assert "GCF/B.30/Add.03 resolves to" in n and "FP172" in n


def test_existing_style_code_still_resolves(reg):
    n = reg.registry_note("Are FP172 and GCF/B.42/02/Add.16 the same?")
    assert "DIFFERENT" in n and "BRACE" in n and "Bhutan Glacial" in n


def test_error_row_is_not_a_usable_row():
    monkey = {"123_gcf-b27-02-add12": {"error": "Unterminated string ..."}}
    registry._cache = monkey
    try:
        assert registry.by_fp(152) is None
    finally:
        registry._cache = None


# --------------------------------------------------------------------------
# schema-2 provenance in the note (FP151 / FP153 / FP274 shaped)
# --------------------------------------------------------------------------

DOC151, DOC153 = "124_gcf-b27-02-add11", "122_gcf-b27-02-add13"
DOC274 = "02_gcf-b42-02-add16-funding-proposal-package-fp274"

# v1 rows as the cover-page builder writes them: clean text fields, financing
# as one raw string with no page. FP274's row has no financing at all — the
# case where v2 does not upgrade a fragment but supplies one v1 never had.
REGV1 = {
    DOC151: {"fp": 151, "title": "TA Facility", "accredited_entity": "IUCN",
             "countries": ["Angola", "Benin"], "board": 27, "year": 2020,
             "gcf_financing": "18.5 M USD",
             "total_financing": "28 M USD for Technical Assistance"},
    DOC153: {"fp": 153, "title": "Mongolian Green Finance Corporation",
             "accredited_entity": "XacBank LLC", "countries": ["Mongolia"],
             "board": 27, "year": 2020, "gcf_financing": "28,654 million USD",
             "total_financing": "49,654 million USD"},
    DOC274: {"fp": 274, "title": "BRACE", "accredited_entity": "SC Australia",
             "countries": ["Zambia"], "board": 42, "year": 2025,
             "gcf_financing": None, "total_financing": None},
}


def _c(raw, value, page, section, status, currency="USD", unit=None):
    return {"raw": raw, "value": value, "currency": currency, "unit": unit,
            "page": page, "section": section, "status": status}


REGV2 = {
    DOC151: {"fp": 151, "facts": {
        "gcf_funding_requested": [
            _c("18.5 M USD", 18_500_000.0, 5, "A.8", "canonical", unit="million")],
        "total_financing": [
            _c("28 M USD", 28_000_000.0, 5, "A.7", "canonical", unit="million"),
            _c("$720000000", 720_000_000.0, 60, "rule:B.2(a)", "conflicting")],
    }},
    DOC153: {"fp": 153, "facts": {
        # the mantissa and the printed scale word cannot both be true, so the
        # builder published the print with no value
        "gcf_funding_requested": [
            _c("28,654 million USD", None, 5, "A.8", "canonical"),
            _c("26,654 million USD", None, 48, "rule:B.2(b)", "conflicting")],
        "total_financing": [_c("49,654 million USD", None, 5, "A.7", "canonical")],
    }},
    DOC274: {"fp": 274, "facts": {
        "gcf_funding_requested": [
            _c("40,511,264 USD", 40_511_264.0, 7, "A.8", "canonical"),
            _c("49,751,264", 49_751_264.0, 8, "A.10 Grant", "conflicting",
               currency=None),
            _c("40,751,254", 40_751_254.0, 40, "rule:C.1(a)", "conflicting"),
            _c("40,000,000", 40_000_000.0, 55, "rule:C.1(a)", "conflicting")],
        "total_financing": [
            _c("46,737,340 USD", 46_737_340.0, 7, "A.7", "canonical"),
            _c("48,000,000 USD", 48_000_000.0, 41, "rule:C.1(a)", "conflicting")],
        # a third conflicting field: the one the two-line cap has to drop
        "co_financing": [_c("6,226,076 USD", 6_226_076.0, 7, "A.7", "canonical"),
                         _c("9,000,000 USD", 9_000_000.0, 41, "rule:C.1(b)",
                            "conflicting")],
    }},
}

# The note FP151 got before schema 2 existed, byte for byte.
V1_NOTE_151 = (
    'Registry — FP151: "TA Facility"; accredited entity: IUCN; '
    'countries (2): Angola, Benin; GCF financing (as printed): 18.5 M USD; '
    'total financing (as printed): 28 M USD for Technical Assistance; '
    'board B.27, 2020 [124_gcf-b27-02-add11, cover pages]')


@pytest.fixture
def regv2(monkeypatch):
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_cache_v2", REGV2)
    return registry


def test_v2_money_replaces_the_v1_fragment_and_keeps_the_v1_text(regv2):
    line = regv2.registry_note("What is FP151's GCF financing?").splitlines()[0]
    assert "GCF funding requested: 18.5 M USD (p.5, A.8)" in line
    assert "total financing: 28 M USD (p.5, A.7)" in line
    assert "GCF financing (as printed)" not in line        # the page-less v1 fragment
    assert "for Technical Assistance" not in line          # ... and its unsourced tail
    # title/entity/countries/board stay v1: schema 1 is the clean source there
    assert '"TA Facility"' in line and "accredited entity: IUCN" in line
    assert "countries (2): Angola, Benin" in line and "board B.27, 2020" in line
    assert line.endswith(f"[{DOC151}, cover pages]")


def test_v2_supplies_money_the_v1_row_never_had(regv2):
    line = regv2.registry_note("What is FP274's GCF financing?").splitlines()[0]
    assert "GCF funding requested: 40,511,264 USD (p.7, A.8)" in line
    assert "total financing: 46,737,340 USD (p.7, A.7)" in line


def test_v1_raw_survives_when_v2_has_no_canonical(monkeypatch):
    """A supporting-only field is 'stated somewhere but not in a template
    section': the note keeps printing the v1 string rather than promoting it."""
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_cache_v2", {DOC151: {"fp": 151, "facts": {
        "gcf_funding_requested": [_c("18.5 M USD", 18_500_000.0, 90, "rule:A.8",
                                     "supporting")]}}})
    line = registry.registry_note("FP151?").splitlines()[0]
    assert "GCF financing (as printed): 18.5 M USD;" in line
    assert "(p.90" not in line
    assert "total financing (as printed): 28 M USD for Technical Assistance" in line


def test_conflict_warning_is_appended_capped_and_page_bearing(regv2):
    """Both caps still hold — and both now SAY they held (F13's rule applied to
    the other truncated list in this module: what is dropped is announced, with
    the number dropped, so a note can never read as the whole story)."""
    n = regv2.registry_note("What is FP274's GCF financing?")
    conf = [ln for ln in n.splitlines() if "CONFLICT" in ln]
    assert len(conf) == 3                      # two field lines + the cap line
    assert conf[0] == (
        f"Registry — CONFLICT in this document ({DOC274}): gcf_funding_requested "
        "is printed as 40,511,264 USD (p.7, A.8); also as 49,751,264 (p.8, "
        "A.10 Grant); also as 40,751,254 (p.40, C.1(a)) (+1 more disagreeing "
        "print of this field in the document, not listed — list truncated) — "
        "report all of them with their pages.")
    assert "40,000,000 " not in n               # a fourth print is over the cap
    assert conf[1].startswith(
        f"Registry — CONFLICT in this document ({DOC274}): total_financing")
    assert conf[2] == (
        f"Registry — CONFLICT in this document ({DOC274}): 1 further field "
        "(co_financing) also prints disagreeing figures, not listed above — "
        "list truncated.")
    assert "9,000,000" not in n                 # named, not printed


def test_conflict_line_leads_with_the_canonical_and_strips_the_rule_marker(regv2):
    """The canonical figure and page come first (that is the pointer the answer
    cites, and what verify keys the line on). 'rule:B.2(a)' means the page
    printed the figure but not the heading — builder bookkeeping, not a pointer
    the answer model should be handed."""
    conf = [ln for ln in regv2.registry_note("FP151?").splitlines() if "CONFLICT" in ln]
    assert conf == [
        f"Registry — CONFLICT in this document ({DOC151}): total_financing "
        "is printed as 28 M USD (p.5, A.7); also as $720000000 (p.60, B.2(a)) "
        "— report both figures with their pages."]


def test_value_nulled_canonical_quotes_the_raw_and_flags_the_unit(regv2):
    n = regv2.registry_note("What is FP153's GCF financing?")
    line = n.splitlines()[0]
    assert ('GCF funding requested: "28,654 million USD" (p.5, A.8) '
            "(unit as printed is ambiguous)") in line
    assert ('total financing: "49,654 million USD" (p.5, A.7) '
            "(unit as printed is ambiguous)") in line
    # the conflicting print of the same unreadable scale is still reported
    assert "26,654 million USD (p.48, B.2(b))" in n


def test_a_document_without_conflicts_stays_one_line(monkeypatch):
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_cache_v2", {DOC274: {"fp": 274, "facts": {
        "gcf_funding_requested": [_c("40,511,264 USD", 40_511_264.0, 7, "A.8",
                                     "canonical")]}}})
    n = registry.registry_note("What is FP274's GCF financing?")
    assert len(n.splitlines()) == 1 and "CONFLICT" not in n


def test_enrichment_does_not_double_the_line(monkeypatch):
    """Provenance is an annotation, not a candidate dump."""
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_cache_v2", {})
    v1 = registry.registry_note("FP151?").splitlines()[0]
    monkeypatch.setattr(registry, "_cache_v2", REGV2)
    v2 = registry.registry_note("FP151?").splitlines()[0]
    assert len(v2) < 2 * len(v1)


def test_board_code_note_gets_the_same_enrichment(regv2):
    n = regv2.registry_note("Summarize GCF/B.42/02/Add.16")
    assert n.splitlines()[0].startswith("Registry — GCF/B.42/02/Add.16 resolves to: ")
    assert "GCF funding requested: 40,511,264 USD (p.7, A.8)" in n
    assert len([ln for ln in n.splitlines() if "CONFLICT" in ln]) == 3


def test_missing_v2_registry_leaves_the_v1_note_byte_identical(monkeypatch):
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_cache_v2", {})
    assert registry.registry_note("What is FP151's GCF financing?") == V1_NOTE_151


def test_unbuilt_v2_file_leaves_the_v1_note_byte_identical(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_cache_v2", None)
    monkeypatch.setattr(registry.config, "DATA_DIR", tmp_path)
    assert registry.registry_note("What is FP151's GCF financing?") == V1_NOTE_151


def test_corrupt_v2_registry_leaves_the_v1_note_byte_identical(monkeypatch):
    """A half-written registry_v2.json raises inside json.loads. The note is
    the answer model's most trusted context: it degrades, it never fails.

    The corruption is simulated at load_v2 — the real choke point every v2
    consumer shares — not at `facts`: patching only `facts` left `_v2_meta`
    reading a warm cache, which is how this test once passed while a
    corrupt file would still have printed provenance pages (caught live
    when meta_provenance landed in the shipped registry)."""
    def boom(*a, **k):
        raise ValueError("Expecting ',' delimiter: line 41210 column 9")
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_cache_v2", None)
    monkeypatch.setattr(registry, "load_v2", boom)
    assert registry.registry_note("What is FP151's GCF financing?") == V1_NOTE_151


def test_note_lines_land_in_verify_evidence_with_their_pages(regv2):
    """The verifier reads these lines back: the cover-page line is document
    scope (its own bracket says 'cover pages'), each conflict warning is page
    scope at the canonical page, and it holds every figure of its field — so
    the answer the warning asks for is the answer that verifies."""
    from gcf_qna.rag import verify

    ev = verify.build_evidence([], regv2.registry_note("What is FP274's GCF financing?"))
    assert verify.NOTES_KEY in ev
    assert "40,511,264 USD (p.7, A.8)" in ev[(DOC274, None)]
    page7 = ev[(DOC274, 7)]
    assert "40,511,264" in page7 and "49,751,264" in page7 and "40,751,254" in page7
    for answer in (f"FP274 requests USD 40,511,264 [{DOC274}, cover pages].",
                   f"FP274 requests USD 40,511,264 [{DOC274}, p. 7].",
                   f"FP274 requests USD 40,511,264; the document also prints "
                   f"49,751,264 and 40,751,254 [{DOC274}, p. 7]."):
        claims = verify.extract_claims(answer)
        assert [v.status for v in verify.classify_deterministic(claims, ev)] \
            == [verify.SUPPORTED], answer
    wrong = verify.extract_claims(f"FP274 requests USD 44,000,000 [{DOC274}, p. 7].")
    assert [v.status for v in verify.classify_deterministic(wrong, ev)] \
        == [verify.UNSUPPORTED]


# --------------------------------------------------------------------------
# cover-page provenance (schema 2's optional `meta_provenance`)
# --------------------------------------------------------------------------
# The key the v2 builder adds for the fields that are NOT money:
#
#   "meta_provenance": {"accredited_entity": {"page": 3, "quote": "..."},
#                       "countries": {...}, "title": {...}}
#
# Additive and optional — only the fields the builder found are present, a row
# built before the pass has no key at all — so every test below has a mirror
# proving the page-less output is unchanged.
#
# WHY the note prints it: told to cite the page beside a fact, the answer model
# GUESSED one for the facts that had none. release-6's two merge traps both
# came back '[124_gcf-b27-02-add11, p. 3]' for the accredited entity — the
# right page, an invented citation, flagged by every instrument. The page was
# in the registry the whole time; nothing printed it.

PROV = {
    DOC151: {"title": {"page": 1, "quote": "TA Facility for the Global ..."},
             "accredited_entity": {"page": 3, "quote": "International Union ..."},
             "countries": {"page": 2, "quote": "Angola, Benin"}},
    DOC274: {"accredited_entity": {"page": 3,
                                   "quote": "Conservation International ..."}},
}


def _with_prov(rows=None):
    """REGV2, plus meta_provenance — nothing else about the row changes."""
    out = {k: dict(v) for k, v in REGV2.items()}
    for doc, meta in (rows or PROV).items():
        out.setdefault(doc, {"fp": REGV1[doc]["fp"]})
        out[doc] = {**out[doc], "meta_provenance": meta}
    return out


@pytest.fixture
def regprov(monkeypatch):
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_cache_v2", _with_prov())
    return registry


#: FP151's line with provenance, byte for byte. Every page is a section-less
#: '(p.N)' sitting immediately after its own field's value; the money keeps the
#: '(p.N, SECTION)' it always had, because a template section is a thing a
#: cover-page fact has no equivalent of.
PROV_NOTE_151 = (
    'Registry — FP151: "TA Facility" (p.1); accredited entity: IUCN (p.3); '
    'countries (2): Angola, Benin (p.2); GCF funding requested: 18.5 M USD '
    '(p.5, A.8); total financing: 28 M USD (p.5, A.7); board B.27, 2020 '
    '[124_gcf-b27-02-add11, cover pages]')


def test_entity_title_and_countries_print_the_page_v2_read_them_on(regprov):
    line = regprov.registry_note("What is FP151's GCF financing?").splitlines()[0]
    assert line == PROV_NOTE_151
    assert "accredited entity: IUCN (p.3)" in line
    assert '"TA Facility" (p.1)' in line
    assert "countries (2): Angola, Benin (p.2)" in line
    # the money fragment is untouched: it already had a pointer, with a section
    assert "GCF funding requested: 18.5 M USD (p.5, A.8)" in line


def test_the_pointer_is_exactly_the_shape_both_consumers_credit():
    r"""Not a formatting preference — the emitted shape is dictated by the two
    regexes that decide which cited pages are legal, and they are the same
    pattern byte for byte:

        verify._NOTE_PAGE_RE  = re.compile(r"\(p\.(\d{1,3})[,)]")
        chainlit_app._note_pages: re.findall(r"\(p\.(\d{1,3})[,)]", line)

    The ')' arm of that character class is what lets a pointer end at the
    closing paren instead of a ', SECTION' tail, so '(p.3)' is creditable and
    'p.3' or '(page 3)' or '(p. 3)' would not be. Pinned against the pattern
    ITSELF (and against the app's own source, which holds the second copy
    inline) so a future edit to either regex fails here rather than silently
    making every entity page uncitable.
    """
    from gcf_qna.rag import verify
    pattern = r"\(p\.(\d{1,3})[,)]"
    assert verify._NOTE_PAGE_RE.pattern == pattern
    app_src = (Path(__file__).resolve().parents[1] / "src" / "gcf_qna" / "app"
               / "chainlit_app.py").read_text(encoding="utf-8")
    assert f're.findall(r"{pattern}", line)' in app_src
    import re as _re
    assert _re.findall(pattern, "accredited entity: IUCN (p.3);") == ["3"]
    for rejected in ("p.3", "(page 3)", "(p. 3)", "(p.3 A.8)"):
        assert not _re.findall(pattern, rejected), rejected


def test_the_entity_page_reaches_both_note_page_readers(regprov):
    """End of the wire: the app decides an answer's citation is legal from
    `_note_pages`, the verifier scopes a claim from `note_page_scopes`, and the
    harness scorer calls the app's function. All three must now hold FP151's
    entity page — the one release-6 was flagged for citing."""
    from gcf_qna.app import chainlit_app as app
    from gcf_qna.rag import verify

    note = regprov.registry_note("What is FP151's GCF financing?")
    line = note.splitlines()[0]
    assert (DOC151, 3) in app._note_pages([note])          # the entity page
    assert (DOC151, 5) in app._note_pages([note])          # ... and the money page
    assert (verify.note_scope_doc(DOC151), 3) in verify.note_page_scopes(line)


def test_only_the_fields_v2_actually_sourced_get_a_page(regprov):
    """FP274's builder found the entity and nothing else: partial provenance is
    the normal case, not a defect."""
    line = regprov.registry_note("What is FP274's GCF financing?").splitlines()[0]
    assert "accredited entity: SC Australia (p.3)" in line
    assert '"BRACE";' in line                       # title: no page, no pointer
    assert "countries (1): Zambia;" in line          # countries: likewise


def test_a_row_with_no_meta_provenance_is_byte_identical(monkeypatch):
    """The contract's absent arm, against the string the note printed before
    this key existed. Their file may land after ours; ours must not care."""
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_cache_v2", REGV2)          # no meta_provenance
    n = registry.registry_note("What is FP151's GCF financing?")
    assert n.splitlines()[0] == (
        'Registry — FP151: "TA Facility"; accredited entity: IUCN; '
        'countries (2): Angola, Benin; GCF funding requested: 18.5 M USD (p.5, A.8); '
        'total financing: 28 M USD (p.5, A.7); board B.27, 2020 '
        '[124_gcf-b27-02-add11, cover pages]')
    assert "(p.3)" not in n and "(p.1)" not in n
    monkeypatch.setattr(registry, "_cache_v2", {})
    assert registry.registry_note("What is FP151's GCF financing?") == V1_NOTE_151


@pytest.mark.parametrize("meta,why", [
    ({"accredited_entity": {"quote": "IUCN"}}, "a quote with no page"),
    ({"accredited_entity": {"page": None}}, "an explicit null page"),
    ({"accredited_entity": {"page": "3"}}, "a page as a string"),
    ({"accredited_entity": {"page": 3.0}}, "a page as a float"),
    ({"accredited_entity": {"page": True}}, "bool is an int in Python"),
    ({"accredited_entity": {"page": 0}}, "page 0 is not a page"),
    ({"accredited_entity": {"page": -3}}, "negative"),
    ({"accredited_entity": {"page": 1000}}, "past the readers' 3-digit cap"),
    ({"accredited_entity": "p.3"}, "the field is not an object"),
    ({"accredited_entity": None}, "an explicit null field"),
])
def test_a_page_the_readers_could_not_credit_is_never_printed(monkeypatch, meta, why):
    """A pointer `_note_pages`/`note_page_scopes` cannot parse would be worse
    than none: the model would cite it and the checker would flag the citation
    as invented — release-6's defect, re-created by the fix meant to end it.
    So an unusable page prints nothing at all, and the line falls back to the
    exact page-less form."""
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_cache_v2", _with_prov({DOC151: meta}))
    line = registry.registry_note("What is FP151's GCF financing?").splitlines()[0]
    assert "accredited entity: IUCN;" in line, why
    from gcf_qna.app import chainlit_app as app
    assert {p for d, p in app._note_pages([line]) if d == DOC151} == {5}, why


@pytest.mark.parametrize("bad", [[], "meta", 3, None])
def test_a_meta_provenance_of_the_wrong_type_degrades_silently(monkeypatch, bad):
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_cache_v2",
                        {DOC151: {"fp": 151, "meta_provenance": bad}})
    assert registry.registry_note("What is FP151's GCF financing?") == V1_NOTE_151


def test_a_raising_v2_lookup_leaves_the_provenance_out_not_the_note(monkeypatch):
    """`_v2_meta` holds `_v2_facts`'s contract: the note is the answer model's
    most trusted context, so it degrades and never fails."""
    def boom(*a, **k):
        raise ValueError("Expecting ',' delimiter: line 41210 column 9")
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_row_v2", boom)
    assert registry.registry_note("What is FP151's GCF financing?") == V1_NOTE_151


def test_conflict_lines_are_untouched_by_cover_page_provenance(regprov):
    """The warnings are money-only (`_MONEY_FIELDS`) and are emitted by a
    different helper: a document that contradicts itself prints exactly the
    lines it printed before, byte for byte."""
    n = regprov.registry_note("What is FP274's GCF financing?")
    conf = [ln for ln in n.splitlines() if "CONFLICT" in ln]
    assert len(conf) == 3
    assert conf[0] == (
        f"Registry — CONFLICT in this document ({DOC274}): gcf_funding_requested "
        "is printed as 40,511,264 USD (p.7, A.8); also as 49,751,264 (p.8, "
        "A.10 Grant); also as 40,751,254 (p.40, C.1(a)) (+1 more disagreeing "
        "print of this field in the document, not listed — list truncated) — "
        "report all of them with their pages.")
    assert conf[1].startswith(
        f"Registry — CONFLICT in this document ({DOC274}): total_financing")
    assert "(p.3)" not in "\n".join(conf)


def test_the_board_code_resolves_to_line_inherits_the_pages(regprov):
    """The second emitter. It prefixes '<CODE> resolves to:' and then calls the
    same `_fmt`, so it gets the provenance for free — and its page must be
    creditable through the same reader, since the line still ends
    '[stem, cover pages]'."""
    from gcf_qna.app import chainlit_app as app
    n = regprov.registry_note("Summarize GCF/B.42/02/Add.16")
    line = n.splitlines()[0]
    assert line.startswith("Registry — GCF/B.42/02/Add.16 resolves to: ")
    assert "accredited entity: SC Australia (p.3)" in line
    assert (DOC274, 3) in app._note_pages([line])


def test_accented_values_survive_the_formatting(monkeypatch):
    """French entity and country names run through the same f-strings; the
    pointer is appended after the value, never inside it."""
    from gcf_qna.app import chainlit_app as app
    doc = "77_gcf-b30-02-add09-funding-proposal-package-fp180"
    monkeypatch.setattr(registry, "_cache", {doc: {
        "fp": 180, "board": 30, "year": 2021,
        "title": "Résilience des systèmes agroalimentaires au Sahel",
        "accredited_entity": "Agence Française de Développement (AFD)",
        "countries": ["Côte d'Ivoire", "Bénin", "Sénégal"]}})
    monkeypatch.setattr(registry, "_cache_v2", {doc: {"fp": 180, "meta_provenance": {
        "title": {"page": 1}, "accredited_entity": {"page": 3},
        "countries": {"page": 2}}}})
    line = registry.registry_note("Résume le FP180").splitlines()[0]
    assert line == (
        'Registry — FP180: "Résilience des systèmes agroalimentaires au Sahel" '
        '(p.1); accredited entity: Agence Française de Développement (AFD) '
        "(p.3); countries (3): Côte d'Ivoire, Bénin, Sénégal (p.2); "
        f"board B.30, 2021 [{doc}, cover pages]")
    # the parenthesised '(AFD)' inside the value does not shadow the pointer
    assert {p for d, p in app._note_pages([line]) if d == doc} == {1, 2, 3}


def test_the_year_listing_stays_page_less(regprov):
    """Deliberately excluded. The listing names FP numbers and no document
    stems, and both readers credit a page only to a document named on ITS OWN
    line — so a page printed here would publish no scope, which is exactly the
    uncitable pointer this whole change exists to remove."""
    from gcf_qna.app import chainlit_app as app
    n = regprov.registry_note("Which proposals were approved in 2020?")
    listing = [ln for ln in n.splitlines() if "documents from 2020" in ln]
    assert len(listing) == 1
    assert "(p." not in listing[0]
    assert app._note_pages(listing) == set()


def test_a_v2_row_with_provenance_but_no_money_keeps_the_v1_fragments(monkeypatch):
    """The upgrade is per-field: a row that carries `meta_provenance` and no
    `facts` gets its cover-page pages and keeps the page-less v1 money strings,
    which must stay page-less — the v1 registry never recorded where it read
    them."""
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "_cache_v2",
                        {DOC151: {"fp": 151, "meta_provenance": PROV[DOC151]}})
    line = registry.registry_note("What is FP151's GCF financing?").splitlines()[0]
    assert line == (
        'Registry — FP151: "TA Facility" (p.1); accredited entity: IUCN (p.3); '
        'countries (2): Angola, Benin (p.2); GCF financing (as printed): 18.5 M USD; '
        'total financing (as printed): 28 M USD for Technical Assistance; '
        'board B.27, 2020 [124_gcf-b27-02-add11, cover pages]')


def test_note_lines_still_land_in_verify_evidence_at_document_scope(regprov):
    """`verify.build_evidence` files a main registry line by its
    '[stem, cover pages]' bracket, at DOCUMENT scope — it never reads a page
    off that line — so adding cover-page pointers does not move any evidence
    key. The pages reach the verifier through `note_scopes`, the derived path,
    exactly as the money pages already did."""
    from gcf_qna.rag import verify
    note = regprov.registry_note("What is FP151's GCF financing?")
    ev = verify.build_evidence([], note)
    assert "accredited entity: IUCN (p.3)" in ev[(DOC151, None)]
    assert (DOC151, 3) not in ev                  # not a retrieved-page key
    assert (verify.note_scope_doc(DOC151), 3) in verify.note_scopes(
        {verify.NOTES_KEY: note})


def _build_registry_module():
    p = Path(__file__).resolve().parents[1] / "scripts" / "build_registry.py"
    spec = importlib.util.spec_from_file_location("build_registry", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_needs_extraction_retries_error_rows():
    needs_extraction = _build_registry_module().needs_extraction
    reg = {
        "ok_doc": {"fp": 1, "title": "t", "board": 27, "year": 2020},
        "err_doc": {"error": "Unterminated string starting at: line 51"},
        "healed_doc": {"fp": 2, "title": None, "board": 27, "year": 2020},
    }
    assert needs_extraction("err_doc", reg) is True      # retried, not cached forever
    assert needs_extraction("absent_doc", reg) is True
    assert needs_extraction("ok_doc", reg) is False
    assert needs_extraction("healed_doc", reg) is False  # LLM nulls are a real result
