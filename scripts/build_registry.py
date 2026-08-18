#!/usr/bin/env python3
"""Build the document metadata registry from cover pages.

For each document in the extracted corpus, an LLM reads the first pages
(templated GCF cover + summary) and emits structured metadata: FP number,
title, accredited entity, countries, financing (kept as RAW quoted strings —
the corpus templates contain 'Enter amount' noise, so numbers are stored as
written, never normalized). Board and year are derived from the doc id, not
the LLM. Resumable: existing rows are kept unless --force, except rows that only
recorded an LLM error — those are retried on every run.

  python scripts/build_registry.py                # all docs -> data/registry.json
  python scripts/build_registry.py --limit 5      # smoke test
"""
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from gcf_qna import config
from gcf_qna.boards import board_of, year_of
from gcf_qna.rag import split_pages

REGISTRY_PATH = config.DATA_DIR / "registry.json"
SOURCE_DIR = config.EXTRACTED_DIR / "vlm" / "qwen_qwen2.5-vl-7b"
MAX_CHARS = 9000
CONCURRENCY = 8
# 300 truncated the countries list mid-string on long proposals -> unparseable
# JSON -> permanent error rows; retries widen the cap further.
MAX_TOKENS = 700

PROMPT = (
    "You extract metadata from the cover/summary pages of a Green Climate Fund "
    "board document. Respond with JSON only:\n"
    '{"fp_number": <int or null>, "title": "<proposal/document title or null>", '
    '"accredited_entity": "<name or null>", "countries": ["<country>", ...], '
    '"gcf_financing": "<amount AS WRITTEN, with currency/units, or null>", '
    '"total_financing": "<amount AS WRITTEN, or null>"}\n'
    "Quote financing amounts exactly as printed (the templates contain "
    "placeholder noise like 'Enter amount' — if a field only shows a "
    "placeholder, use null). Use null for anything not stated."
)


def needs_extraction(stem: str, registry: dict) -> bool:
    """Resume filter. A row that only carries an LLM error is NOT registered:
    it must be retried, otherwise a truncated response is cached forever."""
    row = registry.get(stem)
    return row is None or "error" in row


async def extract_one(client, doc_id: str, text: str, sem):
    row, err = {}, None
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=config.CHAT_MODEL,
                    max_completion_tokens=MAX_TOKENS * (attempt + 1),   # widen on retry
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": PROMPT},
                              {"role": "user", "content": text}])
                row = json.loads(resp.choices[0].message.content or "{}")
                if not isinstance(row, dict):
                    raise ValueError("JSON is not an object")
                err = None
                break
            except Exception as e:
                row, err = {}, f"{type(e).__name__}: {e}"[:120]
                if attempt == 2:
                    break
                await asyncio.sleep(2 * (attempt + 1))
    fp = row.get("fp_number")
    m = re.search(r"fp(\d{2,3})", doc_id)      # the doc id is authoritative
    if m:
        fp = int(m.group(1))
    out = {
        "fp": fp if isinstance(fp, int) else None,
        "title": row.get("title"),
        "accredited_entity": row.get("accredited_entity"),
        "countries": row.get("countries") or [],
        "gcf_financing": row.get("gcf_financing"),
        "total_financing": row.get("total_financing"),
        # derived from the doc id: written even when the LLM call failed, so a
        # failed row still resolves by FP/board/year instead of looking absent
        "board": board_of(doc_id),
        "year": year_of(doc_id),
    }
    if err:
        out["error"] = err
    return doc_id, out


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    registry = {}
    if REGISTRY_PATH.exists() and not a.force:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8")).get("documents", {})

    docs = sorted(SOURCE_DIR.glob("*.md"))
    todo = [d for d in docs if needs_extraction(d.stem, registry)][: a.limit]
    print(f"{len(docs)} docs | {len(registry)} already registered | {len(todo)} to extract")
    if todo:
        import openai
        client = openai.AsyncOpenAI(base_url=config.OPENAI_BASE_URL or None)
        sem = asyncio.Semaphore(CONCURRENCY)

        async def run(path):
            full = path.read_text(encoding="utf-8")
            pages = split_pages(full)[:6]
            text = "\n\n".join(body for _, body in pages)[:MAX_CHARS]
            # financing figures live in section C, far past the cover pages —
            # append that section explicitly
            m = re.search(r"C\.?1?\.?\s*(?:FINANCING INFORMATION|Total Financing)", full, re.I)
            if m:
                text += "\n\n[FINANCING SECTION]\n" + full[m.start():m.start() + 2500]
            return await extract_one(client, path.stem, text, sem)

        done = 0
        for coro in asyncio.as_completed([run(p) for p in todo]):
            doc_id, row = await coro
            registry[doc_id] = row
            done += 1
            if done % 25 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}")

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(
        {"schema_version": 1, "source": str(SOURCE_DIR.name),
         "documents": dict(sorted(registry.items()))},
        ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(1 for r in registry.values() if r.get("title"))
    ae = sum(1 for r in registry.values() if r.get("accredited_entity"))
    fin = sum(1 for r in registry.values() if r.get("gcf_financing"))
    bad = sum(1 for r in registry.values() if r.get("error"))
    print(f"registry: {len(registry)} docs | title {ok} | accredited_entity {ae} "
          f"| gcf_financing {fin} | errors {bad} (retried next run)")
    print(f"-> {REGISTRY_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
