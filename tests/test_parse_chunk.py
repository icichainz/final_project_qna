from gcf_qna.rag.chunk import chunk_text
from gcf_qna.rag.parse import split_pages


def test_split_pages_markers():
    md = "---\n**Page 1**\n---\n\nalpha\n\n---\n**Page 2**\n---\n\nbravo"
    assert split_pages(md) == [(1, "alpha"), (2, "bravo")]


def test_split_pages_fallback():
    assert split_pages("no markers here") == [(0, "no markers here")]


def test_chunk_bounds():
    pieces = chunk_text("word " * 600, size=1000, overlap=200)
    assert pieces and all(len(p) <= 1000 for p in pieces)
