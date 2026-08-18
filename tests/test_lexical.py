from gcf_qna.rag.lexical import LexicalIndex, tokenize


def test_tokenizer_identifiers():
    assert tokenize("FP274") == ["fp274"]
    assert "b42" in tokenize("GCF/B.42/02/Add.16") and "add16" in tokenize("GCF/B.42/02/Add.16")
    assert "fp274" in tokenize("What about FP 274?")
    assert tokenize("") == []


def _chunks(tag):
    return [{"doc_id": f"01_gcf-b42-02-add16-{tag}", "text": f"alpha bravo {tag} one"},
            {"doc_id": f"02_gcf-b37-02-add05-{tag}", "text": f"charlie delta {tag} two"}]


def test_staleness_rebuild(tmp_path):     # review finding #1
    a = _chunks("first")
    lx = LexicalIndex(tmp_path)
    lx.ensure(a)
    assert lx.search("charlie", 5) == [1]
    # simulate an in-place index rebuild with DIFFERENT chunks
    b = list(reversed(_chunks("second")))
    lx2 = LexicalIndex(tmp_path)
    lx2.ensure(b)                          # fingerprint mismatch -> rebuild
    assert lx2.search("charlie", 5) == [0], "stale rowids must not survive a chunk-store rebuild"


def test_same_chunks_no_rebuild(tmp_path):
    a = _chunks("same")
    LexicalIndex(tmp_path).ensure(a)
    mtime = (tmp_path / "lexical.db").stat().st_mtime_ns
    LexicalIndex(tmp_path).ensure(a)
    assert (tmp_path / "lexical.db").stat().st_mtime_ns == mtime
