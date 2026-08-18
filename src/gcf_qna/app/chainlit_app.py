"""Chainlit chat over the indexed GCF corpus.

Run:   chainlit run src/gcf_qna/app/chainlit_app.py
Needs: pip install -e ".[app]"  and OPENAI_API_KEY in the environment
       (never hardcoded — the old repo leaked five keys that way).
       Set OPENAI_BASE_URL to target any OpenAI-compatible server instead
       (e.g. LM Studio); then the key may be empty.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import threading
import time
from typing import Optional

import chainlit as cl
from chainlit.types import ThreadDict

from gcf_qna import config
from gcf_qna.app.highlight import annotated_page
from gcf_qna.rag import Embedder, Retriever, load_index
from gcf_qna.rag.ground import ground_chunk

SYSTEM_PROMPT = (
    "You answer questions about Green Climate Fund (GCF) funding proposals.\n"
    "Ground every answer in the provided context excerpts and cite document\n"
    "ids in brackets, e.g. [01_gcf-b42-02-add17]. If the context does not\n"
    "contain the answer, say so plainly instead of guessing.\n"
    "The excerpts are a small retrieved sample of a 273-document corpus: never\n"
    "state corpus-wide totals, counts, rankings or superlatives (most, largest,\n"
    "all, only) as fact — explicitly scope such claims to 'among the retrieved\n"
    "excerpts' and say the full corpus may contain larger/other cases.\n"
    "When the user compares specific documents, report what each document's\n"
    "excerpts state, item by item — including 'no figure stated in the\n"
    "excerpts' for a document — never refuse the whole comparison because\n"
    "some items lack data.\n"
    "If only part of a question is answerable from the excerpts, answer that\n"
    "part and state plainly which part the excerpts cannot support — do not\n"
    "refuse the whole question.\n"
    "Document ids encode the GCF board meeting: '...-b42-02-...' means B.42.\n"
    "Board-meeting years (verified from the corpus): B.11=2015; B.13-B.15=2016;\n"
    "B.16-B.18=2017; B.19-B.21=2018; B.22-B.24=2019; B.25-B.27=2020;\n"
    "B.28-B.30=2021; B.31-B.34=2022; B.35-B.37=2023; B.38-B.40=2024;\n"
    "B.41-B.43=2025.\n"
    "When a question mentions a year: (1) translate the year to its board\n"
    "range with the table, (2) check each excerpt's document id against that\n"
    "range, (3) present the excerpts that match (naming their FP numbers and\n"
    "boards), and say the corpus may hold more documents from that year than\n"
    "were retrieved. Never claim there is no year information while holding\n"
    "document ids you can date with the table."
)


# Board-meeting years, verified from 271 corpus page-1 dates. Injected into
# every excerpt header so the answer model never has to do table lookups.
BOARD_YEARS = {11: 2015, 12: 2016, 13: 2016, 14: 2016, 15: 2016, 16: 2017,
               17: 2017, 18: 2017, 19: 2018, 20: 2018, 21: 2018, 22: 2019,
               23: 2019, 24: 2019, 25: 2020, 26: 2020, 27: 2020, 28: 2021,
               29: 2021, 30: 2021, 31: 2022, 32: 2022, 33: 2022, 34: 2022,
               35: 2023, 36: 2023, 37: 2023, 38: 2024, 39: 2024, 40: 2024,
               41: 2025, 42: 2025, 43: 2025}
_BOARD_RE = re.compile(r"-b(\d+)-")


def _doc_label(doc_id: str, page) -> str:
    """Citation header with precomputed board/year: the model reads dates
    instead of deriving them."""
    label = doc_id + (f", p. {page}" if page else "")
    m = _BOARD_RE.search(doc_id)
    if m and int(m.group(1)) in BOARD_YEARS:
        b = int(m.group(1))
        label += f" — B.{b}, {BOARD_YEARS[b]}"
    return label


def _year_assist(question: str, hits: list):
    """Code-side year matching: if the question names a year, sort matching
    excerpts first and emit a computed note. Removes the lookup burden the
    answer model repeatedly failed to carry."""
    years = {int(y) for y in re.findall(r"\b(20[12]\d)\b", question)}
    if not years:
        return hits, None
    def doc_year(doc_id):
        m = _BOARD_RE.search(doc_id)
        return BOARD_YEARS.get(int(m.group(1))) if m else None
    matched = [h for h in hits if doc_year(h.doc_id) in years]
    rest = [h for h in hits if h not in matched]
    ys = ", ".join(str(y) for y in sorted(years))
    if matched:
        ids = "; ".join(_doc_label(h.doc_id, h.page) for h in matched)
        note = (f"Note (computed from document ids): excerpts dated {ys}: {ids}. "
                f"The corpus may contain more documents from {ys} than were retrieved.")
    else:
        boards = ", ".join(f"B.{b}" for b, y in sorted(BOARD_YEARS.items()) if y in years)
        note = (f"Note (computed from document ids): none of the retrieved excerpts are "
                f"from {ys} (that year corresponds to boards {boards}). Answer what the "
                f"excerpts do support and state this limit.")
    return matched + rest, note


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
def auth(username: str, password: str) -> Optional[cl.User]:
    from gcf_qna.app import accounts
    users = accounts.parse_env_users()
    expected = users.get(username.strip())
    if expected and hmac.compare_digest(password, expected):
        return cl.User(identifier=username.strip())
    # self-registered accounts (scrypt-hashed, data/app.db)
    from gcf_qna.app import accounts
    if accounts.check_login(username, password):
        return cl.User(identifier=username.strip())
    return None


# Self-registration page + API (/register); ALLOW_SIGNUP=0 disables.
try:
    from gcf_qna.app.register import mount as _mount_register
    _mount_register()
except Exception as _e:   # never let signup wiring break the chat app
    print(f"signup routes not mounted: {_e}", flush=True)


def _history_from_thread(thread: ThreadDict) -> list:
    """Rebuild the condenser's memory from persisted steps.

    Without this, the first follow-up in a resumed thread regresses to the
    starved-decomposer bug: pronouns unresolvable, cited doc ids invisible.
    Sources lines (📎) are UI furniture, not conversation — skipped.
    """
    steps = sorted(thread.get("steps") or [], key=lambda s: s.get("createdAt") or "")
    history = []
    for st in steps:
        out = (st.get("output") or "").strip()
        if not out or out.startswith("📎"):
            continue
        if st.get("type") == "user_message":
            history.append({"role": "user", "content": out})
        elif st.get("type") == "assistant_message":
            history.append({"role": "assistant", "content": out})
    return history[-12:]


@cl.on_chat_resume
async def on_resume(thread: ThreadDict):
    retriever = await cl.make_async(get_retriever)()
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("history", _history_from_thread(thread))


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
    await cl.Message(
        content=f"Ready — {_retriever_meta.get('n_chunks')} chunks indexed "
                f"({_retriever_meta.get('embedding_model')}). Ask about the GCF proposals."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    retriever = cl.user_session.get("retriever")
    if retriever is None:
        await cl.Message(content="Session not initialised — fix the startup warning first.").send()
        return

    import openai

    client = openai.AsyncOpenAI(base_url=config.OPENAI_BASE_URL or None)
    history = cl.user_session.get("history") or []

    # Follow-ups carry references the embedder cannot resolve ("how does THAT
    # compare..."), so retrieval on the raw message fetches noise. One LLM call
    # rewrites the message against recent history — into a single standalone
    # query, or, for comparisons/aggregations over documents named in the
    # conversation, one doc-scoped sub-query per entity (per-document quota).
    # See docs/query-decomposition.html. Best-effort: any failure falls back
    # to the raw message; the original wording still goes to the answer model.
    search_queries = [{"q": message.content, "doc": None}]
    if history:
        try:
            # 500-char truncation used to cut off the citation lists, leaving
            # the decomposer blind to most cited docs — extract them explicitly.
            convo = "\n".join(f"{m['role']}: {m['content'][:1200]}" for m in history[-6:])
            cited: list = []
            for m in history:
                for d in re.findall(r"\[(\d{1,3}_gcf-[\w.\-]+)", m["content"]):
                    if d not in cited:
                        cited.append(d)
            if cited:
                convo += "\nDocuments cited in conversation: " + ", ".join(cited[-12:])
            resp = await client.chat.completions.create(
                model=config.CHAT_MODEL,
                max_completion_tokens=300,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content":
                        "Turn the user's latest message into retrieval queries for a "
                        "document index. Respond with JSON only. Decide in this order:\n"
                        "1. If the message asks about the corpus or 'the proposals' in "
                        "general and does NOT refer back to specific items from earlier "
                        "answers, return it UNCHANGED as one query with no doc: "
                        '{"queries": [{"q": "<message>"}]}\n'
                        "2. If it refers back to SPECIFIC items from earlier answers "
                        "('those', 'the ones you mentioned', named projects/ids) and "
                        "compares or aggregates them, emit one query per item (max 6), "
                        "each tagged with its document id from the conversation: "
                        '{"queries": [{"q": "...", "doc": "<id>"}, ...]} — each q a short '
                        "phrase for ONE item's attribute, never a comparative question.\n"
                        "3. Otherwise (a follow-up on one topic), return ONE standalone "
                        "rewritten query with pronouns resolved, no doc tag."},
                    {"role": "user", "content":
                        f"Conversation:\n{convo}\n\nLatest message: {message.content}"},
                ],
            )
            data = json.loads(resp.choices[0].message.content or "{}")
            parsed = []
            for item in (data.get("queries") or [])[:6]:
                if isinstance(item, str) and item.strip():
                    parsed.append({"q": item.strip(), "doc": None})
                elif isinstance(item, dict) and (item.get("q") or "").strip():
                    parsed.append({"q": item["q"].strip(), "doc": item.get("doc") or None})
            # Deterministic guards against rewrite contamination:
            # 1. When the MESSAGE names its own identifiers (FP numbers /
            #    board codes), every doc tag must match one of them —
            #    history documents must never override explicit ids
            #    (observed: 'FP214... FP265?' answered entirely from the
            #    previous turn's FP274).
            # 2. A single query may only carry a doc tag if the message
            #    names that document; fan-outs without message ids keep
            #    their history-derived tags (the 'compare those' case).
            msg_l = message.content.lower()
            msg_ids = set(re.findall(r"fp\s?(\d{2,3})", msg_l))
            for item in parsed:
                d = str(item.get("doc") or "").lower()
                if not d:
                    continue
                if msg_ids:
                    # The message names its own ids: never trust decomposer
                    # tags at all (observed fabrication: '02_fp214' — history
                    # prefix mashed onto a message id). Two-stage routing
                    # resolves the real doc from the sub-query text.
                    item["doc"] = None
                elif len(parsed) == 1:
                    fp = re.search(r"fp(\d{2,3})", d)
                    if d[:20] not in msg_l and not (fp and "fp" + fp.group(1) in msg_l):
                        item["doc"] = None
            if parsed:
                search_queries = parsed
        except Exception:
            pass

    decomposed = len(search_queries) > 1
    if decomposed or search_queries[0]["q"] != message.content:
        async with cl.Step(name="retrieval query") as step:
            step.output = "\n".join(
                f"{sq['q']}" + (f"   [{sq['doc']}]" if sq.get("doc") else "")
                for sq in search_queries)

    per_query = config.TOP_K if not decomposed else max(3, config.TOP_K // len(search_queries))
    seen, hits = set(), []
    for sq in search_queries:
        got = await cl.make_async(retriever.search)(sq["q"], per_query, sq.get("doc"))
        for h in got:
            key = (h.doc_id, h.page, h.text[:120])
            if key not in seen:
                seen.add(key)
                hits.append(h)
    hits = hits[:15]
    hits, year_note = _year_assist(message.content, hits)
    context = "\n\n".join(
        f"[{_doc_label(h.doc_id, h.page)}] (score {h.score:.2f})\n{h.text}"
        for h in hits)
    if year_note:
        context = year_note + "\n\n" + context
    messages = history + [{
        "role": "user",
        "content": f"Context excerpts:\n{context}\n\nQuestion: {message.content}",
    }]

    reply = cl.Message(content="")
    stream = await client.chat.completions.create(
        model=config.CHAT_MODEL,
        max_completion_tokens=config.MAX_ANSWER_TOKENS,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        stream=True,
    )
    async for part in stream:
        if part.choices and part.choices[0].delta.content:
            await reply.stream_token(part.choices[0].delta.content)
    await reply.send()

    history += [
        {"role": "user", "content": message.content},
        {"role": "assistant", "content": reply.content},
    ]
    cl.user_session.set("history", history[-12:])  # keep the last 6 exchanges

    if hits:
        sources = ", ".join(sorted({f"{h.doc_id} p.{h.page}" if h.page else h.doc_id
                                    for h in hits}))
        # Ground the citations: annotated page images with the cited passage
        # highlighted (green lines / blue table region). Dedupe by (doc, page),
        # cap at 3 pages so answers stay scannable.
        elements, seen = [], set()
        for h in hits:
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
        await cl.Message(content=f"📎 Sources: {sources}", elements=elements).send()
