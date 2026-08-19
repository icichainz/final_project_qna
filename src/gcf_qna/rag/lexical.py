"""Lexical BM25 search over the chunk store, via SQLite FTS5.

FTS5 over rank-bm25 deliberately: disk-backed (no ~1-2 GB of in-memory
token lists on the deployed server), stdlib-only, and ships BM25 ranking
natively. Identifier-preserving tokenization is done in Python and stored
as a pre-tokenized column, so FTS5's own tokenizer never touches codes
like FP274 or B.42. The index is a sidecar (lexical.db) beside the FAISS
files, built lazily from chunks.jsonl on first use — existing indexes
upgrade themselves without a rebuild.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

# Keeps fp274, b.42, add.16 intact; splits on everything else; lowercases.
TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[a-z0-9]+)*")
TOKENIZER_VERSION = 2   # bump on any change: the FTS index self-rebuilds


def tokenize(text: str) -> List[str]:
    """Identifier-normalizing tokenizer.

    Queries write 'B.42'/'Add.16'; filenames spell 'b42'/'add16' — emit the
    dotless variant alongside any dotted token so both spellings meet in the
    index. 'FP 274' additionally joins into 'fp274'.
    """
    raw = TOKEN_RE.findall(text.lower())
    out: List[str] = []
    for i, t in enumerate(raw):
        out.append(t)
        if "." in t:
            out.append(t.replace(".", ""))
        if t == "fp" and i + 1 < len(raw) and raw[i + 1].isdigit():
            out.append("fp" + raw[i + 1])
    return out


def fp_variants(tok: str) -> List[str]:
    """fp86 <-> fp086: the corpus zero-pads one FP number (fp086), so an
    unpadded query token must also try the padded spelling and vice versa.
    Non-FP tokens pass through unchanged."""
    m = re.fullmatch(r"fp0*(\d{1,3})", tok)
    if not m:
        return [tok]
    n = int(m.group(1))
    return sorted({tok, f"fp{n}", f"fp{n:03d}"})


def _doc_tokens(doc_id: str) -> List[str]:
    """Filenames carry the proposal number even when the chunk body doesn't."""
    return tokenize(doc_id.replace("-", " ").replace("_", " "))


def _chunks_fingerprint(chunks: List[Dict[str, Any]]) -> str:
    """Cheap identity of the chunk store: FAISS/chunks.jsonl can be rebuilt
    in place, and a lexical.db from the previous build would map BM25 rowids
    to the wrong chunks (review finding #1)."""
    h = hashlib.sha1(str(len(chunks)).encode())
    for c in (chunks[0], chunks[-1]) if chunks else ():
        h.update(c.get("doc_id", "").encode())
        h.update(c.get("text", "")[:80].encode())
    return h.hexdigest()


class LexicalIndex:
    def __init__(self, index_dir: Path):
        self.path = Path(index_dir) / "lexical.db"
        self._con: Optional[sqlite3.Connection] = None

    def ensure(self, chunks: List[Dict[str, Any]]) -> None:
        """Open the index, (re)building it when absent OR when the tokenizer
        version or the chunk-store fingerprint no longer match."""
        fp = _chunks_fingerprint(chunks)
        build = not self.path.exists()
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        if not build:
            ver = self._con.execute("PRAGMA user_version").fetchone()[0]
            stored_fp = None
            try:
                stored_fp = self._con.execute(
                    "SELECT value FROM meta WHERE key='chunks_fp'").fetchone()
                stored_fp = stored_fp[0] if stored_fp else None
            except sqlite3.OperationalError:
                pass
            if ver != TOKENIZER_VERSION or stored_fp != fp:   # stale: rebuild
                self._con.close()
                self.path.unlink()
                build = True
                self._con = sqlite3.connect(self.path, check_same_thread=False)
        if build:
            con = self._con
            con.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5"
                        "(toks, content='', columnsize=1, tokenize=\"unicode61 tokenchars '.'\")")
            rows = ((i, " ".join(tokenize(c["text"]) + _doc_tokens(c.get("doc_id", ""))))
                    for i, c in enumerate(chunks))
            con.executemany("INSERT INTO chunks_fts(rowid, toks) VALUES (?, ?)", rows)
            con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
            con.execute("INSERT INTO meta VALUES ('chunks_fp', ?)", (fp,))
            con.execute(f"PRAGMA user_version = {TOKENIZER_VERSION}")
            con.commit()

    def search(self, query: str, n: int) -> List[int]:
        """Chunk indices ranked by BM25. OR-semantics: subsets may match."""
        toks = tokenize(query)
        if not toks or self._con is None:
            return []
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(toks))
        try:
            rows = self._con.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY bm25(chunks_fts) LIMIT ?", (match, n)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [r[0] for r in rows]
