#!/usr/bin/env python3
"""Build a FAISS index from an extracted corpus.

Examples:
  python scripts/build_index.py --source data/extracted/vlm/qwen_qwen2.5-vl-7b
  python scripts/build_index.py --source data/extracted/docling --name docling \
      --embedding-model sentence-transformers/all-MiniLM-L6-v2
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gcf_qna import config
from gcf_qna.rag import Embedder, build_index, iter_documents, save_index
from gcf_qna.rag.parse import chunk_document, make_record, retrieval_text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, required=True,
                    help="extraction dir, e.g. data/extracted/vlm/<model>")
    ap.add_argument("--name", default="default", help="index name under data/index/")
    ap.add_argument("--embedding-model", default=config.EMBEDDING_MODEL)
    ap.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    ap.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    ap.add_argument("--limit", type=int, default=None, help="index only the first N documents")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing index of that name")
    ap.add_argument("--section-prefix", action="store_true",
                    help="embed 'section path + page text' instead of the page "
                         "text alone (measured to hurt all-mpnet-base-v2; see "
                         "parse.make_record)")
    a = ap.parse_args()

    out = config.INDEX_DIR / a.name
    if (out / "index.faiss").exists() and not a.force:
        raise SystemExit(f"{out} already holds an index — pass --force to replace it "
                         "(data/index/default is the production fallback)")

    import hashlib
    chunks, n_docs, seen, dropped = [], 0, set(), 0
    for doc_id, text in iter_documents(a.source):
        for page_no, chunk in chunk_document(text, a.chunk_size, a.chunk_overlap):
            # GCF packages repeat proposal pages inside annexes -> exact
            # duplicate chunks (11,577 measured) that crowd out distinct
            # evidence in top-k. Dedup by normalized doc/page/text.
            key = (doc_id, page_no,
                   hashlib.sha1(" ".join(chunk.text.split()).encode()).hexdigest())
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            chunks.append(make_record(doc_id, page_no, chunk,
                                      section_prefix=a.section_prefix))
        n_docs += 1
        if a.limit and n_docs >= a.limit:
            break
    if dropped:
        print(f"deduplicated: {dropped} exact duplicate chunks dropped")
    if not chunks:
        raise SystemExit(f"No documents found under {a.source}")
    sectioned = sum(1 for c in chunks if c.get("section_path"))
    tables = sum(1 for c in chunks if c.get("kind") == "table")
    print(f"{n_docs} documents -> {len(chunks)} chunks "
          f"({sectioned} with a section path, {tables} table chunks) "
          f"| embedding: {a.embedding_model}")

    t0 = time.time()
    # EMBED retrieval_text — the page text, or the section path plus the page
    # text under --section-prefix. chunk['text'] stays the source text the
    # answer prompt and ground.py read either way.
    emb = Embedder(a.embedding_model).encode([retrieval_text(c) for c in chunks])
    index = build_index(emb)
    save_index(index, chunks, out, embedding_model=a.embedding_model,
               chunk_size=a.chunk_size, chunk_overlap=a.chunk_overlap,
               section_prefix=bool(a.section_prefix),
               source=str(a.source), n_documents=n_docs, n_table_chunks=tables)
    print(f"wrote {out}  ({emb.shape[0]} vectors x {emb.shape[1]} dims)  in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
