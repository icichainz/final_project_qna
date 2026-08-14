"""Central configuration: every path and tunable, env-overridable.

This is the single place constants live. The main2/main3/main4 forks of the
old codebase existed because these values were hardcoded per file — anything
you would have varied by copying a file belongs here instead.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("GCF_QNA_ROOT", str(Path(__file__).resolve().parents[2])))

# --- data layout (see data/README.md for the regeneration contract) ---
DATA_DIR = Path(os.getenv("GCF_QNA_DATA", str(PROJECT_ROOT / "data")))
RAW_PDF_DIR = DATA_DIR / "raw" / "pdfs"
FUNDING_DIR = DATA_DIR / "raw" / "funding"
PAGE_CACHE_DIR = DATA_DIR / "cache" / "pages"
EXTRACTED_DIR = DATA_DIR / "extracted"
INDEX_DIR = DATA_DIR / "index"

# --- RAG ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
TOP_K = int(os.getenv("TOP_K", "5"))

# --- chat ---
CHAT_MODEL = os.getenv("CHAT_MODEL", "claude-sonnet-5")
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "1024"))

# --- local inference endpoint (VLM extraction) ---
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://192.168.56.1:12345/v1").rstrip("/")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "")
