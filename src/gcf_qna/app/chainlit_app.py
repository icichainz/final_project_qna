"""Chainlit chat over the indexed GCF corpus.

Run:   chainlit run src/gcf_qna/app/chainlit_app.py
Needs: pip install -e ".[app]"  and ANTHROPIC_API_KEY in the environment
       (never hardcoded — the old repo leaked five keys that way).
"""
from __future__ import annotations

import os

import chainlit as cl

from gcf_qna import config
from gcf_qna.rag import Embedder, Retriever, load_index

SYSTEM_PROMPT = (
    "You answer questions about Green Climate Fund (GCF) funding proposals.\n"
    "Ground every answer in the provided context excerpts and cite document\n"
    "ids in brackets, e.g. [01_gcf-b42-02-add17]. If the context does not\n"
    "contain the answer, say so plainly instead of guessing."
)


def _index_dir():
    return config.INDEX_DIR / os.getenv("INDEX_NAME", "default")


@cl.on_chat_start
async def start():
    if not os.getenv("ANTHROPIC_API_KEY"):
        await cl.Message(
            content="⚠️ `ANTHROPIC_API_KEY` is not set. Copy `.env.example` to `.env`, "
                    "fill in the key, and restart."
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

    index, chunks, cfg = await cl.make_async(load_index)(idx_dir)
    retriever = Retriever(index, chunks, Embedder(cfg.get("embedding_model")))
    cl.user_session.set("retriever", retriever)
    cl.user_session.set("history", [])
    await cl.Message(
        content=f"Ready — {cfg['n_chunks']} chunks indexed "
                f"({cfg.get('embedding_model')}). Ask about the GCF proposals."
    ).send()


@cl.on_message
async def main(message: cl.Message):
    retriever = cl.user_session.get("retriever")
    if retriever is None:
        await cl.Message(content="Session not initialised — fix the startup warning first.").send()
        return

    import anthropic

    hits = await cl.make_async(retriever.search)(message.content, config.TOP_K)
    context = "\n\n".join(f"[{h.doc_id}] (score {h.score:.2f})\n{h.text}" for h in hits)

    history = cl.user_session.get("history") or []
    messages = history + [{
        "role": "user",
        "content": f"Context excerpts:\n{context}\n\nQuestion: {message.content}",
    }]

    client = anthropic.AsyncAnthropic()
    reply = cl.Message(content="")
    async with client.messages.stream(
        model=config.CHAT_MODEL,
        max_tokens=config.MAX_ANSWER_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
    ) as stream:
        async for token in stream.text_stream:
            await reply.stream_token(token)
    await reply.send()

    history += [
        {"role": "user", "content": message.content},
        {"role": "assistant", "content": reply.content},
    ]
    cl.user_session.set("history", history[-12:])  # keep the last 6 exchanges

    if hits:
        sources = ", ".join(sorted({h.doc_id for h in hits}))
        await cl.Message(content=f"📎 Sources: {sources}").send()
