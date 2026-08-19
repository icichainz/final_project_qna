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
    'countries: Angola, Benin; GCF financing (as printed): 18.5 M USD; '
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
    assert "countries: Angola, Benin" in line and "board B.27, 2020" in line
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
    n = regv2.registry_note("What is FP274's GCF financing?")
    conf = [ln for ln in n.splitlines() if "CONFLICT" in ln]
    assert len(conf) == 2                      # one line per field, two fields
    assert conf[0] == (
        f"Registry — CONFLICT in this document ({DOC274}): gcf_funding_requested "
        "is printed as 40,511,264 USD (p.7, A.8); also as 49,751,264 (p.8, "
        "A.10 Grant); also as 40,751,254 (p.40, C.1(a)) — report all of them "
        "with their pages.")
    assert "40,000,000" not in n                # a third print is over the cap
    assert conf[1].startswith(
        f"Registry — CONFLICT in this document ({DOC274}): total_financing")
    assert "co_financing" not in n              # ... and so is a third field
    assert "9,000,000" not in n


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
    assert len([ln for ln in n.splitlines() if "CONFLICT" in ln]) == 2


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
    the answer model's most trusted context: it degrades, it never fails."""
    def boom(*a, **k):
        raise ValueError("Expecting ',' delimiter: line 41210 column 9")
    monkeypatch.setattr(registry, "_cache", REGV1)
    monkeypatch.setattr(registry, "facts", boom)
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
