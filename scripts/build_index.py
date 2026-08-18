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
from gcf_qna.rag import (Embedder, build_index, chunk_text, iter_documents,
                         save_index, split_pages)


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
    a = ap.parse_args()

    import hashlib
    chunks, n_docs, seen, dropped = [], 0, set(), 0
    for doc_id, text in iter_documents(a.source):
        for page_no, body in split_pages(text):
            for piece in chunk_text(body, a.chunk_size, a.chunk_overlap):
                # GCF packages repeat proposal pages inside annexes -> exact
                # duplicate chunks (11,577 measured) that crowd out distinct
                # evidence in top-k. Dedup by normalized doc/page/text.
                key = (doc_id, page_no,
                       hashlib.sha1(" ".join(piece.split()).encode()).hexdigest())
                if key in seen:
                    dropped += 1
                    continue
                seen.add(key)
                chunks.append({"doc_id": doc_id, "page": page_no, "text": piece})
        n_docs += 1
        if a.limit and n_docs >= a.limit:
            break
    if dropped:
        print(f"deduplicated: {dropped} exact duplicate chunks dropped")
    if not chunks:
        raise SystemExit(f"No documents found under {a.source}")
    print(f"{n_docs} documents -> {len(chunks)} chunks | embedding: {a.embedding_model}")

    t0 = time.time()
    emb = Embedder(a.embedding_model).encode([c["text"] for c in chunks])
    index = build_index(emb)
    out = config.INDEX_DIR / a.name
    save_index(index, chunks, out, embedding_model=a.embedding_model)
    print(f"wrote {out}  ({emb.shape[0]} vectors x {emb.shape[1]} dims)  in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
