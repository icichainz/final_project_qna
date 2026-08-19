"""Section-aware chunking and the step-3 retrieval pipeline.

Three properties are load-bearing here and each has a test that fails loudly
if it stops holding:

  * the section path is a RETRIEVAL artefact — it is embedded and lexically
    indexed, and it never reaches the answer prompt or ground.py;
  * a markdown table small enough to keep is never split, because half a table
    has no header row and answers nothing;
  * a document that IS an identifier outranks a document that merely mentions
    it when the two-stage router picks target documents.
"""
import hashlib

import faiss
import numpy as np

from gcf_qna.rag.parse import (SectionTracker, chunk_document, chunk_page,
                               make_record, retrieval_text, section_id,
                               source_text)
from gcf_qna.rag.retrieve import Reranker, Retriever, doc_is_identifier


# --------------------------------------------------------------- fixtures ---
class FakeEmbedder:
    """Deterministic unit vectors from text hashes — no torch needed."""
    def encode(self, texts, **kw):
        out = []
        for t in texts:
            rng = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16))
            v = rng.standard_normal(16).astype("float32")
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def build(chunks, tmp_path, **kw):
    emb = FakeEmbedder()
    index = faiss.IndexFlatIP(16)
    index.add(emb.encode([retrieval_text(c) for c in chunks]))
    return Retriever(index, chunks, emb, index_dir=tmp_path, **kw)


PAGE = """# A PROJECT/PROGRAMME SUMMARY

## A.7 Total financing (GCF + co-finance)
46,737,340 USD

## A.8 Total GCF funding requested
40,511,264 USD

### A.9 Project size
Small (Up to USD 50 million)
"""


# ------------------------------------------------------- section detection ---
def test_section_id_spellings():
    assert section_id("A.8 Total GCF funding requested") == "A.8"
    assert section_id("A. 8 Total financing") == "A.8"
    assert section_id("A PROJECT/PROGRAMME SUMMARY") == "A"
    assert section_id("Section B PROJECT INFORMATION") == "B"
    assert section_id("Annex II Feasibility study") == "ANNEX II"
    assert section_id("C.1 Financing information") == "C.1"


def test_section_id_rejects_prose_and_running_headers():
    """'A funding proposal titled ...' is a sentence, not section A."""
    for line in ("A funding proposal titled BRACE", "Summary", "Disclaimer",
                 "Table of Contents",
                 "GREEN CLIMATE FUND FUNDING PROPOSAL V3.0 | PAGE 2"):
        assert section_id(line) is None, line


def test_section_path_uses_printed_ids_not_hash_depth():
    """The VLM writes A.8 as '##' and A.9 as '###' on the same page; A.9 is a
    sibling of A.8, so it must not nest under it."""
    chunks = chunk_page(PAGE, size=120, overlap=20)
    paths = [c.section_path for c in chunks]
    assert "A PROJECT/PROGRAMME SUMMARY > A.8 Total GCF funding requested" in paths
    assert not any("A.8" in p and "A.9" in p for p in paths if p)


def test_running_header_ends_a_carried_section():
    t = SectionTracker()
    t.heading("## A.19. Complementarity and coherence")
    t.page_break()
    assert t.path and t.path.startswith("A.19")      # ids survive a page break
    t.heading("# GREEN CLIMATE FUND FUNDING PROPOSAL V3.0 | PAGE 2")
    assert t.path is None


def test_unnumbered_heading_does_not_cross_pages():
    t = SectionTracker()
    t.heading("## Disclaimer")
    assert t.path == "Disclaimer"
    t.page_break()
    assert t.path is None


# ------------------------------------------------ retrieval / source split ---
def test_retrieval_text_prefixes_and_source_text_stays_pure():
    chunks = chunk_page(PAGE, size=120, overlap=20)
    src = next(c for c in chunks if c.section_path and "A.8" in c.section_path)
    rec = make_record("doc", 7, src, section_prefix=True)
    assert rec["section_path"].endswith("A.8 Total GCF funding requested")
    assert retrieval_text(rec).startswith(rec["section_path"] + "\n\n")
    # what the model and the grounder read is the page text, unprefixed
    assert source_text(rec) == rec["text"] == rec["retrieval_text"].split("\n\n", 1)[1]
    assert not source_text(rec).startswith(rec["section_path"])
    assert "40,511,264 USD" in source_text(rec)


def test_section_prefix_is_opt_in():
    """Default builds embed the page text: the prefix measured WORSE on
    all-mpnet-base-v2 (gold evidence page fell from rank 2 to 14 in its own
    document). The path is still stored, and still drives expansion."""
    rec = make_record("doc", 7, chunk_page(PAGE, 120, 20)[0])
    assert rec["section_path"] and "retrieval_text" not in rec
    assert retrieval_text(rec) == source_text(rec) == rec["text"]


def test_hits_carry_source_text_never_the_prefix(tmp_path):
    chunks = [make_record("doc-a", 7, c) for c in chunk_page(PAGE, 120, 20)]
    r = build(chunks, tmp_path)
    for h in r.search("total gcf funding requested", top_k=3):
        assert not h.text.startswith(h.section_path or "\0")
        assert h.text == source_text(r.chunks[h.chunk_index])


def test_old_schema_chunks_still_work(tmp_path):
    """data/index/default predates the schema: no source_text, no
    retrieval_text, no section_path. It must load and search unchanged."""
    old = [{"doc_id": "02_gcf-b42-02-add16-package-fp274", "page": 8,
            "text": "total financing information for the project"},
           {"doc_id": "61_gcf-b37-02-add05-package-fp214", "page": 3,
            "text": "gender action plan budget"}]
    assert retrieval_text(old[0]) == source_text(old[0]) == old[0]["text"]
    r = build(old, tmp_path)
    hits = r.search("gender action plan budget", top_k=2)
    assert hits and all(h.text and h.section_path is None for h in hits)


def test_source_text_key_is_honored_when_present():
    """A future writer may spell the field out; both spellings resolve."""
    rec = {"text": "ignored", "source_text": "page text", "section_path": "A.8"}
    assert source_text(rec) == "page text"
    # no retrieval_text on the row means the page text IS what was embedded —
    # never re-derive a prefix the build did not apply
    assert retrieval_text(rec) == "page text"
    assert retrieval_text(dict(rec, retrieval_text="A.8\n\npage text")) == "A.8\n\npage text"


# ---------------------------------------------------------------- tables ----
TABLE_PAGE = "## A.4 Result area(s)\n\n" + "\n".join(
    ["| Area | GCF | Co-financiers |", "|---|---|---|"]
    + [f"| Result area {i} | {i}0% | {i}5% |" for i in range(1, 15)])


def test_table_stays_atomic_and_keeps_its_heading():
    assert 300 < len(TABLE_PAGE) <= 2 * 300, "fixture must exercise the 2x rule"
    chunks = chunk_page(TABLE_PAGE, size=300, overlap=60)
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1, "a table under 2x chunk_size must not be split"
    assert "Result area 1 " in tables[0].text and "Result area 14 " in tables[0].text
    assert tables[0].section_path.endswith("A.4 Result area(s)")
    prefixed = make_record("d", 7, tables[0], section_prefix=True)
    assert retrieval_text(prefixed).startswith("A.4 Result area(s)")


def test_oversized_table_still_splits():
    big = "| Area | GCF |\n" + "\n".join(f"| Result area {i} | {i}0% |"
                                         for i in range(400))
    chunks = chunk_page(big, size=500, overlap=100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1000 for c in chunks)


def test_chunk_document_tracks_pages_and_sections():
    doc = (f"---\n**Page 1**\n---\n\n{PAGE}\n"
           f"---\n**Page 2**\n---\n\ncontinued prose under A.9\n")
    got = list(chunk_document(doc, size=200, overlap=40))
    assert {p for p, _ in got} == {1, 2}
    tail = [c for p, c in got if p == 2][0]
    # A.9 is a printed id, so it carries onto the continuation page
    assert tail.section_path and "A.9" in tail.section_path


# ------------------------------------------------------- mention magnet -----
IS_DOC = "124_gcf-b27-02-add11-funding-proposal-package-fp151"
MENTIONS_DOC = "34_gcf-b30-02-add07-funding-proposal-package-fp242"

# The real shape of the defect: the magnet is a SHORT list of identifiers
# (BM25 rewards that) inside an unrelated package, while the document that IS
# FP151 carries the code only in its filename and pages of ordinary prose.
_PROSE = ("The programme will strengthen coastal resilience through mangrove "
          "restoration, community early warning systems and revised building "
          "codes, with disbursement against agreed milestones over five years "
          "in the participating regions and their surrounding catchments. ")
MAGNET_CHUNKS = [
    {"doc_id": MENTIONS_DOC, "page": 12,
     "text": "Other GCF programs in Guyana include: FP189, FP203, FP152, FP151"},
    {"doc_id": MENTIONS_DOC, "page": 13,
     "text": "FP151 FP152 FP189 FP203 coordination arrangements"},
    {"doc_id": IS_DOC, "page": 4, "text": "Total GCF financing requested. " + _PROSE},
    {"doc_id": IS_DOC, "page": 5, "text": "Implementation arrangements. " + _PROSE},
]


def test_doc_is_identifier():
    assert doc_is_identifier(IS_DOC, ["fp151"])
    assert not doc_is_identifier(MENTIONS_DOC, ["fp151"])
    assert doc_is_identifier("189_12-status-approved-fps-fp086-gcf-b37-07", ["fp86"])


def test_target_docs_prefer_the_document_that_is_the_identifier(tmp_path):
    r = build(MAGNET_CHUNKS, tmp_path)
    assert r.hybrid_enabled
    lex = r.lexical.search("fp151", 20)
    assert r.chunks[lex[0]]["doc_id"] == MENTIONS_DOC, "premise: the magnet wins BM25"
    assert r._target_docs(lex, ["fp151"])[0] == IS_DOC


def test_identifier_query_puts_the_real_document_first(tmp_path):
    r = build(MAGNET_CHUNKS, tmp_path)
    hits = r.search("What does FP151 finance?", top_k=4)
    assert hits[0].doc_id == IS_DOC


# ------------------------------------------- dedup / quota / expansion ------
def test_dedup_runs_before_quotas(tmp_path):
    """Two copies of one passage must not spend two of a document's slots."""
    dupes = [{"doc_id": "doc-a", "page": 1, "text": "identical evidence   text"},
             {"doc_id": "doc-a", "page": 1, "text": "identical evidence text"},
             {"doc_id": "doc-b", "page": 1, "text": "other document evidence"}]
    r = build(dupes, tmp_path)
    kept = r._dedup([(0, 1.0), (1, 0.9), (2, 0.8)])
    assert [i for i, _ in kept] == [0, 2], "whitespace-only variants are one passage"
    quota = r._quota(kept, ["doc-a", "doc-b"], 4)
    assert [r.chunks[i]["doc_id"] for i, _ in quota] == ["doc-a", "doc-b"]


def test_quota_round_robins_documents(tmp_path):
    chunks = [{"doc_id": f"doc-{d}", "page": p, "text": f"passage {d}{p}"}
              for d in "ab" for p in range(1, 5)]
    r = build(chunks, tmp_path)
    ranked = [(i, 1.0 - i / 100) for i in range(8)]      # all of doc-a first
    got = r._quota(ranked, ["doc-a", "doc-b"], 4)
    assert [r.chunks[i]["doc_id"] for i, _ in got] == ["doc-a", "doc-b"] * 2


def test_expansion_adds_neighbouring_same_section_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("SECTION_EXPAND", "1")        # opt-in: see the flag table
    chunks = [
        {"doc_id": "d", "page": 4, "text": "financing head", "section_path": "C.1"},
        {"doc_id": "d", "page": 5, "text": "financing tail", "section_path": "C.1"},
        {"doc_id": "d", "page": 9, "text": "unrelated annex", "section_path": "H"},
    ]
    r = build(chunks, tmp_path)
    assert r._neighbors(0) == [1]
    got = [i for i, _ in r._expand([(0, 1.0), (2, 0.5)], top_k=3)]
    # seeds keep their order; the neighbour is appended, never inserted above one
    assert got == [0, 2, 1]
    assert r._neighbors(2) == [], "a different section is not a neighbour"


def test_expansion_never_displaces_the_top_seeds(tmp_path, monkeypatch):
    monkeypatch.setenv("SECTION_EXPAND", "1")
    chunks = [{"doc_id": "d", "page": p, "text": f"passage {p}", "section_path": "C.1"}
              for p in range(1, 11)]
    r = build(chunks, tmp_path)
    ranked = [(i, 1.0 - i / 100) for i in range(10)]
    got = r._expand(ranked, top_k=10)
    assert [i for i, _ in got][:6] == [0, 1, 2, 3, 4, 5]
    monkeypatch.setenv("SECTION_EXPAND", "0")
    assert r._expand(ranked, top_k=10) == ranked


# ---------------------------------------------------------------- rerank ----
def test_rerank_interface_reorders_with_a_mock_scorer(tmp_path, monkeypatch):
    monkeypatch.setenv("RERANK", "1")                # opt-in: see the flag table
    chunks = [{"doc_id": "d", "page": p, "text": t} for p, t in
              enumerate(["alpha", "bravo", "charlie", "delta", "echo",
                         "foxtrot", "golf", "hotel", "india", "juliet"], 1)]
    seen = {}

    def scorer(query, texts):
        seen["query"], seen["n"] = query, len(texts)
        return [1.0 if "juliet" in t else 0.0 for t in texts]

    r = build(chunks, tmp_path, reranker=Reranker(scorer=scorer))
    ranked = [(i, 1.0 - i / 100) for i in range(10)]
    got = r._rerank("which is last?", ranked, top_k=3)
    assert got[0][0] == 9 and seen["query"] == "which is last?" and seen["n"] == 10
    assert got[0][1] == ranked[9][1], "rerank decides order, not the printed score"


def test_rerank_scores_retrieval_text_including_the_section_path(tmp_path, monkeypatch):
    monkeypatch.setenv("RERANK", "1")
    chunks = [make_record("d", 7, c, section_prefix=True)
              for c in chunk_page(PAGE, 60, 10)]
    while len(chunks) < 10:                                  # clear the pool floor
        chunks.append({"doc_id": "d", "page": 99, "text": f"filler {len(chunks)}"})
    scored = {}

    def scorer(query, texts):
        scored["texts"] = texts
        return list(range(len(texts)))[::-1]

    r = build(chunks, tmp_path, reranker=Reranker(scorer=scorer))
    r._rerank("q", [(i, 0.5) for i in range(len(chunks))], top_k=3)
    assert any(t.startswith("A PROJECT/PROGRAMME SUMMARY") for t in scored["texts"])


def test_rerank_falls_back_when_the_model_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setenv("RERANK", "1")
    rr = Reranker(model_name="cross-encoder/does-not-exist-anywhere")
    assert rr.score("q", ["a", "b"]) is None and rr.failed
    chunks = [{"doc_id": "d", "page": p, "text": f"passage {p}"} for p in range(1, 11)]
    r = build(chunks, tmp_path, reranker=rr)
    ranked = [(i, 1.0 - i / 100) for i in range(10)]
    assert r._rerank("q", ranked, top_k=3) == ranked      # fusion order survives


def test_rerank_flag_off_keeps_fusion_order(tmp_path, monkeypatch):
    monkeypatch.setenv("RERANK", "0")               # also the default
    calls = []
    r = build([{"doc_id": "d", "page": p, "text": f"p{p}"} for p in range(1, 11)],
              tmp_path, reranker=Reranker(scorer=lambda q, t: calls.append(t) or
                                          [0.0] * len(t)))
    ranked = [(i, 0.5) for i in range(10)]
    assert r._rerank("q", ranked, top_k=3) == ranked and not calls


def test_small_pools_skip_the_reranker(tmp_path, monkeypatch):
    """A model load to reorder three passages is not worth its startup cost."""
    monkeypatch.setenv("RERANK", "1")
    calls = []
    chunks = [{"doc_id": "d", "page": p, "text": f"p{p}"} for p in range(1, 4)]
    r = build(chunks, tmp_path,
              reranker=Reranker(scorer=lambda q, t: calls.append(t) or [0.0] * len(t)))
    r.search("anything", top_k=2)
    assert not calls


# ------------------------------------------------------- page diversity -----
def test_page_diversity_defers_repeat_pages(tmp_path, monkeypatch):
    chunks = [{"doc_id": "d", "page": p, "text": f"chunk {p}.{n}"}
              for p, n in ((41, 1), (41, 2), (41, 3), (56, 1), (40, 1))]
    r = build(chunks, tmp_path)
    ranked = [(i, 1.0 - i / 100) for i in range(5)]
    got = [r.chunks[i]["page"] for i, _ in r._diversify(ranked)]
    assert got == [41, 56, 40, 41, 41], "each page is served once before twice"
    monkeypatch.setenv("PAGE_DIVERSITY", "0")
    assert r._diversify(ranked) == ranked


# --------------------------------------------- one head per identifier -----
def test_each_identifier_nominates_its_own_document(tmp_path):
    """A joined 'fp220 OR fp203' BM25 head can be won outright by one code, and
    the other proposal then never enters the routing candidates at all."""
    # BM25 normalizes by length, so the proposal whose pages are terse takes
    # the whole head even though both codes are equally common in the corpus.
    loud = [{"doc_id": "55_gcf-b37-02-add11-package-fp220", "page": p,
             "text": f"FP220 activity {p}"} for p in range(1, 61)]
    quiet = [{"doc_id": "72_GCF_B.35_02_Add.05_package_for_FP203", "page": p,
              "text": f"FP203 page {p}: " + "programme ranking and disbursement "
                      "schedule for the reporting period " * 14}
             for p in range(1, 61)]
    r = build(loud + quiet, tmp_path)
    joined = r.lexical.search("fp220 fp203", 40)
    assert r._docs_of(joined) == ["55_gcf-b37-02-add11-package-fp220"], "premise"
    targets = r._target_docs(joined, ["fp203", "fp220"], limit=3)
    assert set(targets) == {"55_gcf-b37-02-add11-package-fp220",
                            "72_GCF_B.35_02_Add.05_package_for_FP203"}
    hits = r.search("Compare FP220 and FP203 rankings", top_k=6)
    assert len({h.doc_id for h in hits}) == 2, "both proposals must be served"
