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


def _doc_tokens(doc_id: str) -> List[str]:
    """Filenames carry the proposal number even when the chunk body doesn't."""
    return tokenize(doc_id.replace("-", " ").replace("_", " "))


class LexicalIndex:
    def __init__(self, index_dir: Path):
        self.path = Path(index_dir) / "lexical.db"
        self._con: Optional[sqlite3.Connection] = None

    def ensure(self, chunks: List[Dict[str, Any]]) -> None:
        """Open the index, building it from the chunk list if absent."""
        build = not self.path.exists()
        self._con = sqlite3.connect(self.path, check_same_thread=False)
        if not build:
            ver = self._con.execute("PRAGMA user_version").fetchone()[0]
            if ver != TOKENIZER_VERSION:       # stale tokenizer: rebuild
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
