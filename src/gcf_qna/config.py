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
TOP_K = int(os.getenv("TOP_K", "10"))
# hybrid retrieval (BM25 + dense fused by reciprocal rank)
HYBRID = os.getenv("HYBRID", "1") == "1"
CANDIDATES_PER_RETRIEVER = int(os.getenv("CANDIDATES_PER_RETRIEVER", "30"))
RRF_K = int(os.getenv("RRF_K", "60"))
# below this best dense cosine, retrieval is flagged low-confidence (step 4
# of the hybrid plan): the answer model gets an explicit weak-signal note
MIN_DENSE_SCORE = float(os.getenv("MIN_DENSE_SCORE", "0.5"))
# per-turn conductor call (mode routing + English queries); 0 restores the
# history-only condensation behavior
CONDUCTOR = os.getenv("CONDUCTOR", "1") == "1"
# deterministic comparison planner (plan step 4): a question that names >= 2
# documents AND asks a comparison/field question builds its evidence matrix
# from the registry before retrieval, instead of routing through the LLM
# conductor. Independent switch, DEFAULT OFF for this deploy — the conductor
# path is the measured one; flip after the eval run.
PLANNER = os.getenv("PLANNER", "0") == "1"
# claim-level verification of the finished answer against the pages it cites,
# plus at most one constrained repair pass (plan step 5, gcf_qna.rag.verify).
# Master switch, DEFAULT OFF for this deploy: it adds up to two LLM calls per
# turn and rewrites answers, so it ships behind the same discipline as PLANNER.
VERIFY = os.getenv("VERIFY", "0") == "1"
# The two optional LLM calls inside that pass, consulted ONLY when VERIFY=1
# and INDEPENDENT of each other: the batched judge over claims the
# deterministic checks could not confirm (VERIFY_LLM), and the constrained
# repair rewrite (VERIFY_REPAIR). VERIFY_LLM=0 does not disable the repair —
# it leaves the repair working from deterministic verdicts, which is a
# supported deployment. VERIFY_REPAIR=0 is what guarantees the answer text is
# never touched; both off makes the whole pass pure python.
VERIFY_LLM = os.getenv("VERIFY_LLM", "1") == "1"
VERIFY_REPAIR = os.getenv("VERIFY_REPAIR", "1") == "1"

# --- chat (OpenAI-compatible endpoint) ---
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.2")
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "1024"))
# Optional: point the OpenAI client elsewhere (e.g. LM Studio) instead of api.openai.com
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

# --- app persistence (conversation threads) ---
APP_DB = Path(os.getenv("APP_DB", str(DATA_DIR / "app.db")))
PUBLIC_DIR = PROJECT_ROOT / "public"

# --- local inference endpoint (VLM extraction) ---
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://192.168.56.1:12345/v1").rstrip("/")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "")
