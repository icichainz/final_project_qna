"""Board-code resolution (agenda item is part of the identifier) and the
registry build's resume filter."""
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
