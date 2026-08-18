from gcf_qna.app.chainlit_app import _detect_lang, _doc_label, _invalid_citations


def test_detect_lang():
    assert _detect_lang("Quel est le financement total ?") == "French"
    assert _detect_lang("Which entity implements FP218?") == "English"
    assert _detect_lang("je veux un tableau par rapport a ca") == "French"
    assert _detect_lang("FP274?") is None


def test_doc_label_formats():                         # review finding #4
    assert "B.37, 2023" in _doc_label("61_gcf-b37-02-add05-x", 3)
    assert "B.35, 2023" in _doc_label("72_GCF_B.35_02_Add.05_x", 3)
    assert "—" not in _doc_label("weird_doc", 1)


class H:
    def __init__(self, d, p):
        self.doc_id, self.page = d, p


def test_invalid_citations():
    hits = [H("02_gcf-b42-02-add16-funding-proposal-package-fp274", 8)]
    bad = _invalid_citations("See [02_gcf-b42-02-add16-funding-proposal-package-fp274, p. 35].", hits)
    assert bad and "p.35" in bad[0]
    assert _invalid_citations("See [02_gcf-b42-02-add16-funding-proposal-package-fp274, p. 8].", hits) == []
    assert _invalid_citations("AE [57_gcf-x, cover pages].", hits) == []
