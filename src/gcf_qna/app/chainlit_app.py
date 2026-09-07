"""Chainlit chat over the indexed GCF corpus — the UI, and only the UI.

Run:   chainlit run src/gcf_qna/app/chainlit_app.py
Needs: pip install -e ".[app]"  and OPENAI_API_KEY in the environment
       (never hardcoded — the old repo leaked five keys that way).
       Set OPENAI_BASE_URL to target any OpenAI-compatible server instead
       (e.g. LM Studio); then the key may be empty.

What a turn DOES lives in `gcf_qna.pipeline`, which imports no chainlit and
is the same code `scripts/eval_answers.py` measures. What is left here is
Chainlit: auth, the SQLite thread store, the retriever singleton, and the
rendering of one `pipeline.run_turn` — its steps, its streamed tokens, its
sources line and the annotated page images.

Every helper this module used to define is re-exported below, so the names
tests and the harness already look up here keep resolving.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import threading
import time
from typing import Optional

import chainlit as cl
from chainlit.types import ThreadDict

from gcf_qna import config, pipeline
from gcf_qna.rag import registry  # noqa: F401  (re-export)
from gcf_qna.app.highlight import annotated_page
from gcf_qna.rag import Embedder, Retriever, load_index
from gcf_qna.rag import planner, verify  # noqa: F401  (re-export)
from gcf_qna.rag.ground import ground_chunk

from gcf_qna.app.prompts import (  # noqa: F401  (re-export)
    CONDUCTOR_PROMPT, SYSTEM_PROMPT, assemble, assemble_chat)

# The turn's own helpers. Explicitly, never a wildcard: this list IS the
# module's compatibility surface — `tests/` and `scripts/eval_answers.py` look
# these names up on `chainlit_app`, and a wildcard would let one silently
# disappear from the pipeline module without anything failing here.
from gcf_qna.pipeline import (  # noqa: F401
    _abstain_banner, _answer_cap, _answer_messages, _asks_about_both,
    _BOARD_CODE_RE, _board_range_note, _BOARD_TOKEN_RE, _BOARD_WORD_RE,
    _boards_in, _cite_key, _cited_docs, _claim_texts, _CLOSED_RANGE_RE,
    _COMPARE_INTENT_RE, _COMPARE_RULE, _CONFLICT_DOC_RE, _CONFLICT_PAGE_RE,
    _conflict_probe, _conflict_probe_asks, _context_block,
    ConversationMemory, _corpus_coverage_note, _CORPUS_TOKEN_RE,
    _COVERAGE_ASK_RE, _COVERAGE_VOCAB, _deaccent, _detect_lang, _doc_label,
    _EN_WORDS, _extend_registry_note, _finish_reason, _fp_list, _fp_of,
    _FP_RE, _FR_WORDS, _ids_in, _invalid_citations, _MAX_PROBE_HITS,
    _MAX_PROBE_PAGES, _MAX_TURN_NOTE_DOCS, MemoryEntry, _money,
    _NO_SUM_RULE, _NOTE_DOC_RE, _note_pages, _off_vocabulary, _ONWARD_RE,
    _OPEN_RANGE_RE, _outside_corpus_note, _plan_query, _planner_intent,
    _prescope_single_fp, _registry_doc, _registry_fp, _rescope_items,
    _resolve_doc, _resolve_doc_tags, _resolved_refs_note, _scan_years,
    _SECTION_ASK_RE, _section_probe, _section_probe_asks, _SECTION_WORD_RE,
    _span_text, _TEMPLATE_SECTION_MAX, _TOTAL_YEARS_MAX, _truncation_marker,
    _TRUNCATION_STEP_NOTE, _turn_doc_ids, TurnContext, TurnHooks, TurnPlan,
    TurnResult, TurnSettings, _usd_amount, _usd_total, _v2_money,
    _verification_lines, _verifier_flagged_cites, _with_ids, _year_assist,
    _YEAR_RE, _year_scope, _year_total_line)

# Logging, configured once and only when nothing else has: `basicConfig` is a
# no-op once handlers exist, so an operator's own setup (and chainlit's server)
# wins. The ROOT stays at WARNING and LOG_LEVEL is applied to the `gcf_qna`
# tree alone: at INFO on the root, faiss, sentence-transformers and httpx bury
# the one line per turn this exists to publish — and every importer of this
# module, the eval harness included, would inherit that.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
logging.getLogger("gcf_qna").setLevel(
    getattr(logging, (config.LOG_LEVEL or "INFO").upper(), logging.INFO))

log = logging.getLogger("gcf_qna.app")


async def _verify_reply(reply, evidence: dict, truncated: bool = False):
    """Audit a finished answer and render its step; returns the verification
    result, or None when the verification could not run.

    Kept as a name (and as the one place the audit meets a `cl.Message`) while
    the audit itself is `pipeline.run_verification`: the verifier is a pure
    detector, so the only thing this can add to the message is the abstain
    banner above the model's own text.
    """
    original = reply.content
    async with cl.Step(name="verification") as step:
        res, output, banner = await cl.make_async(pipeline.run_verification)(
            original, evidence, TurnSettings.from_config(), truncated)
        step.output = output
    if banner:
        reply.content = banner + "\n\n" + original
        await reply.update()
    return res


def _index_dir():
    return config.INDEX_DIR / os.getenv("INDEX_NAME", "default")


# One retriever per process, shared by every chat session: the FAISS index is
# ~730 MB on disk and the embedder holds GPU state — loading them per session
# made the first question of every chat pay ~1 min of cold start.
_retriever: Optional[Retriever] = None
_retriever_meta: dict = {}
_retriever_lock = threading.Lock()


def get_retriever() -> Optional[Retriever]:
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            idx_dir = _index_dir()
            if not (idx_dir / "index.faiss").exists():
                return None
            t0 = time.perf_counter()
            index, chunks, cfg = load_index(idx_dir)
            embedder = Embedder(cfg.get("embedding_model"))
            embedder.encode(["warmup"])   # load weights + CUDA context now
            _retriever = Retriever(index, chunks, embedder, index_dir=idx_dir)
            _retriever_meta.update(cfg)
            print(f"retriever ready: {cfg.get('n_chunks')} chunks, "
                  f"{cfg.get('embedding_model')} in {time.perf_counter() - t0:.1f}s",
                  flush=True)
    return _retriever


# Warm up in the background at server start, so even the first session's first
# question hits a hot retriever. PRELOAD=0 disables (e.g. for quick UI work).
if os.getenv("PRELOAD", "1") == "1":
    threading.Thread(target=get_retriever, daemon=True).start()


# ---------------------------------------------------------------------------
# Conversation history: SQLite-backed thread persistence + auth.
# Chainlit's sidebar (threads, resume, feedback) activates when a data layer
# AND authentication are configured. Threads live in data/app.db; element
# files (evidence images) are copied under public/app_files/ so resumed
# threads render across restarts. Schema: scripts/init_appdb.py.
# ---------------------------------------------------------------------------
_data_layer_instance = None


def _make_sqlite_layer():
    """SQLAlchemyDataLayer with SQLite-shape normalization.

    The layer targets Postgres, whose driver auto-parses JSONB -> dict and
    BOOLEAN -> bool. aiosqlite returns raw TEXT/int, and chainlit's frontend
    is written against the Postgres shape — replayed assistant messages
    render blank when step.metadata arrives as a string. Normalize at the
    one read path both the sidebar and thread replay flow through.
    """
    from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

    class SqliteNormalizedLayer(SQLAlchemyDataLayer):
        async def get_all_user_threads(self, user_id=None, thread_id=None):
            threads = await super().get_all_user_threads(user_id, thread_id)
            for t in threads or []:
                if isinstance(t.get("metadata"), str):
                    try:
                        t["metadata"] = json.loads(t["metadata"] or "{}")
                    except ValueError:
                        t["metadata"] = {}
                for st in t.get("steps") or []:
                    if isinstance(st.get("metadata"), str):
                        try:
                            st["metadata"] = json.loads(st["metadata"] or "{}")
                        except ValueError:
                            st["metadata"] = {}
                    for k in ("streaming", "waitForAnswer", "isError", "defaultOpen"):
                        if isinstance(st.get(k), int):
                            st[k] = bool(st[k])
            return threads

    return SqliteNormalizedLayer


@cl.data_layer
def _data_layer():
    global _data_layer_instance
    if _data_layer_instance is None:
        from gcf_qna.app.storage_local import LocalStorageClient
        SQLAlchemyDataLayer = _make_sqlite_layer()
        if not config.APP_DB.exists():
            # first boot: create the schema (idempotent DDL)
            import runpy
            runpy.run_path(str(config.PROJECT_ROOT / "scripts" / "init_appdb.py"),
                           run_name="__main__")
        _data_layer_instance = SQLAlchemyDataLayer(
            conninfo=f"sqlite+aiosqlite:///{config.APP_DB}",
            storage_provider=LocalStorageClient(),
        )
    return _data_layer_instance


@cl.password_auth_callback
async def auth(username: str, password: str) -> Optional[cl.User]:
    from gcf_qna.app import accounts
    users = accounts.parse_env_users()
    expected = users.get(username.strip())
    if expected and hmac.compare_digest(password, expected):
        return cl.User(identifier=username.strip())
    # Self-registered accounts (scrypt-hashed, data/app.db). Both halves of
    # the check block: scrypt is ~100 ms of deliberate CPU and sqlite3 is
    # synchronous. Run on the event loop they freeze token streaming and
    # websocket heartbeats for EVERY connected session, so offload to a
    # worker thread — and throttle, since /login (unlike /register) has no
    # rate limit of its own to stop a login flood from renting that CPU.
    if not accounts.login_allowed(username):
        return None
    if await cl.make_async(accounts.check_login)(username, password):
        return cl.User(identifier=username.strip())
    return None


# Self-registration page + API (/register); ALLOW_SIGNUP=0 disables.
try:
    from gcf_qna.app.register import mount as _mount_register
    _mount_register()
except Exception as _e:   # never let signup wiring break the chat app
    print(f"signup routes not mounted: {_e}", flush=True)


# ---------------------------------------------------------------------------
# Per-thread state: the replayed history, and the conversation's memory.
# ---------------------------------------------------------------------------
#: Where the thread's `ConversationMemory` is kept in the thread's own
#: metadata. Persisting it is an optimisation, never the contract: every read
#: falls back to rebuilding from the persisted steps.
_MEMORY_KEY = "gcf_qna_memory"


def _history_from_thread(thread: ThreadDict) -> list:
    """The replayed conversation of a resumed thread (pipeline.history_from_steps).

    Without this, the first follow-up in a resumed thread regresses to the
    starved-decomposer bug: pronouns unresolvable, cited doc ids invisible.
    """
    return pipeline.history_from_steps(thread.get("steps"))


def _thread_metadata(thread: ThreadDict) -> dict:
    """The thread's metadata as a dict — aiosqlite hands JSON back as TEXT."""
    meta = thread.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta or "{}")
        except ValueError:
            meta = {}
    return meta if isinstance(meta, dict) else {}


def _session_memory() -> pipeline.ConversationMemory:
    mem = cl.user_session.get("memory")
    if not isinstance(mem, pipeline.ConversationMemory):
        mem = pipeline.ConversationMemory()
        cl.user_session.set("memory", mem)
    return mem


async def _persist_memory(memory) -> None:
    """Keep the memory in the thread's metadata, best effort.

    The resume path rebuilds it from the persisted steps regardless, so this
    is a shortcut and never a dependency: no thread store, no thread id yet,
    or a layer that refuses the write all leave the rebuild in charge.
    """
    try:
        thread_id = getattr(getattr(cl.context, "session", None), "thread_id", None)
    except Exception:                                  # noqa: BLE001
        return                     # no chainlit context: nothing to persist to
    if not thread_id or _data_layer_instance is None:
        return
    try:
        meta = dict(cl.user_session.get("thread_metadata") or {})
        meta[_MEMORY_KEY] = memory.to_metadata()
        cl.user_session.set("thread_metadata", meta)
        await _data_layer_instance.update_thread(thread_id, metadata=meta)
    except Exception:                                  # noqa: BLE001
        log.warning("conversation memory not persisted to thread metadata; "
                    "resume will rebuild it from the thread's steps",
                    exc_info=True)


@cl.on_chat_resume
async def on_resume(thread: ThreadDict):
    retriever = await cl.make_async(get_retriever)()
    cl.user_session.set("retriever", retriever)
    history = _history_from_thread(thread)
    cl.user_session.set("history", history)
    meta = _thread_metadata(thread)
    cl.user_session.set("thread_metadata", meta)
    memory = pipeline.ConversationMemory.from_metadata(meta.get(_MEMORY_KEY))
    if memory is None:
        # the fallback that is always available: the citations the assistant
        # turns actually made, registry-resolved turn by turn
        memory = await cl.make_async(pipeline.ConversationMemory.from_history)(
            history)
    cl.user_session.set("memory", memory)
    log.info("thread resumed: %d replayed messages, %d remembered documents",
             len(history), len(memory))


@cl.on_chat_start
async def start():
    if not os.getenv("OPENAI_API_KEY") and not config.OPENAI_BASE_URL:
        await cl.Message(
            content="⚠️ `OPENAI_API_KEY` is not set (and no `OPENAI_BASE_URL` for a "
                    "local server). Copy `.env.example` to `.env`, fill it in, restart."
        ).send()
        return
    idx_dir = _index_dir()
    if not (idx_dir / "index.faiss").exists():
        await cl.Message(
            content=f"⚠️ No index found at `{idx_dir}`.\n"
                    "Build one first:\n```\npython scripts/build_index.py "
                    "--source data/extracted/vlm/qwen_qwen2.5-vl-7b --name default\n```"
        ).send()
        return

    retriever = await cl.make_async(get_retriever)()
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("history", [])
    cl.user_session.set("memory", pipeline.ConversationMemory())
    cl.user_session.set("thread_metadata", {})
    await cl.Message(
        content="👋 **Welcome to SSA CHATBOT.** Ask questions, compare projects, and explore "
                "insights across Green Climate Fund proposals. Every answer is grounded in the "
                "source documents, with citations you can review and verify."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """One `pipeline.run_turn`, rendered.

    Everything this function decides is a rendering decision: which steps to
    open, where the tokens go, whether the finished text needs an update, and
    which pages to ground as images. What the turn is — conductor, guards,
    planner, retrieval, notes, probes, prompt, verification — is in
    `gcf_qna.pipeline`, and `scripts/eval_answers.py` measures that same code.
    """
    retriever = cl.user_session.get("retriever")
    if retriever is None:
        await cl.Message(content="Session not initialised — fix the startup warning first.").send()
        return

    import openai

    client = openai.AsyncOpenAI(base_url=config.OPENAI_BASE_URL or None)
    history = cl.user_session.get("history") or []
    memory = _session_memory()
    reply = cl.Message(content="")

    async def _step(name, output):
        async with cl.Step(name=name) as step:
            step.output = output

    async def _answer_complete(_text):
        await reply.send()

    result = await pipeline.run_turn(
        message.content, history=history, retriever=retriever, llm=client,
        settings=pipeline.TurnSettings.from_config(), memory=memory,
        hooks=pipeline.TurnHooks(step=_step, on_token=reply.stream_token,
                                 answer_complete=_answer_complete,
                                 offload=cl.make_async))

    if result.mode == "guard":
        # answered from the registry: no model call, no sources, no evidence
        await cl.Message(content=result.answer).send()
    elif result.answer != reply.content:
        # The abstain banner and the truncation marker are the system's own
        # lines, added after the stream ended — sync them onto what was sent.
        # An update that throws (a closed websocket) must not leave the user
        # with one text and the history with another: what is on screen wins,
        # and the turn is recorded as the model wrote it.
        shown = reply.content
        try:
            reply.content = result.answer
            await reply.update()
        except Exception:                              # noqa: BLE001
            reply.content = shown
            result.answer = shown
            log.warning("the answer message could not be updated; the abstain "
                        "banner / truncation marker was not applied",
                        exc_info=True)

    history += [
        {"role": "user", "content": message.content},
        {"role": "assistant", "content": result.answer},
    ]
    cl.user_session.set("history", history[-12:])  # keep the last 6 exchanges

    if result.mode != "retrieve":
        return

    # The conversation's documents, from THIS turn's own resolved plan and its
    # answer's citations — never by parsing prose for facts.
    memory.update(**(result.memory_update or {}))
    cl.user_session.set("memory", memory)
    await _persist_memory(memory)

    rep = result.sources
    if result.context.hits:
        # Ground the citations: annotated page images with the cited passage
        # highlighted (green lines / blue table region). Dedupe by (doc, page),
        # cap at 3 pages so answers stay scannable.
        elements, seen = [], set()
        for h in rep.hits:
            if not h.page or (h.doc_id, h.page) in seen:
                continue
            seen.add((h.doc_id, h.page))
            try:
                g = await cl.make_async(ground_chunk)(
                    {"doc_id": h.doc_id, "page": h.page, "text": h.text})
                img = await cl.make_async(annotated_page)(g) if g else None
            except Exception:
                g, img = None, None
            if img is not None:
                label = f"{h.doc_id} — p. {h.page}"
                if g and g.kind == "page":
                    label += " (page-level match)"
                elements.append(cl.Image(name=label, path=str(img), display="inline"))
            if len(elements) >= 3:
                break
        await cl.Message(content=f"📎 Sources: {rep.rendered()}",
                         elements=elements).send()
    elif rep.lines:
        # No excerpts means no sources line to hang the verdict on, and a
        # note-only answer is exactly the kind that needs one. It goes out in
        # the sources message's own shape: the leading 📎 is what keeps UI
        # furniture out of the conversation when a thread is resumed
        # (pipeline.history_from_steps), so a bare cl.Message here would come
        # back as an assistant turn.
        await cl.Message(
            content="📎 Sources: none retrieved\n" + "\n".join(rep.lines)).send()
