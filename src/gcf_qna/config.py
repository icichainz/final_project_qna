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
# Default 1 since 2026-08-26 (owner decision, coverage-campaign plan Phase 0):
# every release measured PLANNER=1 and production deploys it; a lost .env line
# must not silently un-planner the system.
PLANNER = os.getenv("PLANNER", "1") == "1"
# claim-level verification of the finished answer against the pages it cites
# (plan step 5, gcf_qna.rag.verify). Master switch, DEFAULT OFF for this
# deploy: it adds an LLM call per turn, so it ships behind the same discipline
# as PLANNER. The pass DETECTS only — it reports what the cited pages do not
# support and never edits the answer.
VERIFY = os.getenv("VERIFY", "0") == "1"
# The one optional LLM call inside that pass, consulted ONLY when VERIFY=1: the
# batched judge over claims the deterministic checks could not confirm.
# VERIFY_LLM=0 leaves the pass pure python — fewer claims are adjudicated, none
# of the reporting changes.
VERIFY_LLM = os.getenv("VERIFY_LLM", "1") == "1"
# There is deliberately no VERIFY_REPAIR here. The repair pass — rewrite the
# answer, adopt the rewrite if it re-verifies clean — was removed in eac4c94:
# adopting a rewrite deletes the evidence of what the model got wrong, and the
# measured reproducibility of the adopt decision never justified that (see
# docs/claim-support-rollout-plan.md). A switch that can only ever be 0 is a
# trap: it reads as a live capability one env edit away, and the code behind it
# is gone. VERIFY_REPAIR left in an .env is inert — nothing reads it.

# --- chat (OpenAI-compatible endpoint) ---
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5.2")
# None (the default) means the answer calls send no output cap at all. The old
# 1024 budget silently truncated long multi-section answers mid-sentence: the
# API stops at the cap, the stream loop never sees a finish_reason, and on a
# reasoning model the cap covers reasoning tokens too, so the visible budget
# was smaller still. Set a positive number to re-impose a budget.
MAX_ANSWER_TOKENS = int(os.getenv("MAX_ANSWER_TOKENS", "0")) or None
# Optional: point the OpenAI client elsewhere (e.g. LM Studio) instead of api.openai.com
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")

# --- app persistence (conversation threads) ---
APP_DB = Path(os.getenv("APP_DB", str(DATA_DIR / "app.db")))
PUBLIC_DIR = PROJECT_ROOT / "public"

# --- local inference endpoint (VLM extraction) ---
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://192.168.56.1:12345/v1").rstrip("/")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "")
