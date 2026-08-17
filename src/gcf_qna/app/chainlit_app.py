"""Chainlit chat over the indexed GCF corpus.

Run:   chainlit run src/gcf_qna/app/chainlit_app.py
Needs: pip install -e ".[app]"  and OPENAI_API_KEY in the environment
       (never hardcoded — the old repo leaked five keys that way).
       Set OPENAI_BASE_URL to target any OpenAI-compatible server instead
       (e.g. LM Studio); then the key may be empty.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Optional

import chainlit as cl

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
    "excerpts' and say the full corpus may contain larger/other cases."
)


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
            _retriever = Retriever(index, chunks, embedder)
            _retriever_meta.update(cfg)
            print(f"retriever ready: {cfg.get('n_chunks')} chunks, "
                  f"{cfg.get('embedding_model')} in {time.perf_counter() - t0:.1f}s",
                  flush=True)
    return _retriever


# Warm up in the background at server start, so even the first session's first
# question hits a hot retriever. PRELOAD=0 disables (e.g. for quick UI work).
if os.getenv("PRELOAD", "1") == "1":
    threading.Thread(target=get_retriever, daemon=True).start()


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
            convo = "\n".join(f"{m['role']}: {m['content'][:500]}" for m in history[-6:])
            resp = await client.chat.completions.create(
                model=config.CHAT_MODEL,
                max_completion_tokens=300,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content":
                        "Turn the user's latest message into retrieval queries for a "
                        "document index, resolving pronouns and references from the "
                        "conversation. Respond with JSON only.\n"
                        'Normally: {"queries": [{"q": "<one standalone query>"}]}\n'
                        "If the message compares or aggregates over multiple specific "
                        "documents/projects cited in the conversation (ids look like "
                        "'40_gcf-b39-02-add11-...'), emit one query per item, max 6, "
                        'each with its document id: {"queries": [{"q": "...", '
                        '"doc": "<id>"}, ...]}\\n'
                        "Each sub-query must be a short search phrase about ONE "
                        "item's attribute (e.g. 'indigenous peoples plan dedicated "
                        "budget'), never a comparative question."},
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
    context = "\n\n".join(
        f"[{h.doc_id}{f', p. {h.page}' if h.page else ''}] (score {h.score:.2f})\n{h.text}"
        for h in hits)
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
