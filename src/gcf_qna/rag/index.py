"""Build, save, and load the FAISS index.

Metadata is JSON (config.json + chunks.jsonl), replacing the old pickle
sidecar: safe to load, diffable, language-agnostic. Vectors are normalized,
so IndexFlatIP gives cosine similarity.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def build_index(embeddings):
    import faiss
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index, chunks: List[Dict[str, Any]], out_dir: Path, *,
               embedding_model: str, **meta: Any) -> None:
    import faiss
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lexical.db").unlink(missing_ok=True)   # sidecar must not outlive the chunks
    faiss.write_index(index, str(out_dir / "index.faiss"))
    with (out_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    sectioned = sum(1 for c in chunks if c.get("section_path"))
    prefixed = sum(1 for c in chunks if c.get("retrieval_text"))
    (out_dir / "config.json").write_text(json.dumps({
        "embedding_model": embedding_model,
        "metric": "cosine (IndexFlatIP over normalized vectors)",
        "n_chunks": len(chunks),
        # Schema 2 chunks carry section_path and, where it differs, the
        # retrieval_text that was actually embedded. 'text' is the source text
        # in BOTH schemas, which is why a schema-1 index still loads.
        "chunk_schema": 2 if sectioned else 1,
        "n_sectioned": sectioned,
        "embedded_field": ("retrieval_text: source text with a section-path "
                           f"prefix on {prefixed} chunks" if prefixed else
                           "retrieval_text: the source text itself (no prefix)"),
        "created_at": int(time.time()),
        **meta,
    }, indent=2), encoding="utf-8")


def load_index(out_dir: Path) -> Tuple[Any, List[Dict[str, Any]], Dict[str, Any]]:
    import faiss
    out_dir = Path(out_dir)
    index = faiss.read_index(str(out_dir / "index.faiss"))
    lines = (out_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(ln) for ln in lines if ln]
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    return index, chunks, cfg
