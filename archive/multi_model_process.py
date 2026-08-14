#!/usr/bin/env python3
import os
import tempfile
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import aiohttp
import json
import random
from enum import Enum

try:
    from pypdf import PdfReader, PdfWriter  # pip install pypdf
except ImportError:
    raise SystemExit("Please: pip install pypdf")

try:
    import aiofiles  # pip install aiofiles
except ImportError:
    raise SystemExit("Please: pip install aiofiles")

# --------------------------------------------------------------------
# Models list (from your message)
# --------------------------------------------------------------------
MODELS_JSON = r"""{
  "data": [
    {"id": "qwen/qwen3-coder-30b","object":"model","owned_by":"organization_owner"},
    {"id": "qwen/qwq-32b","object":"model","owned_by":"organization_owner"},
    {"id": "mistralai/magistral-small-2509","object":"model","owned_by":"organization_owner"},
    {"id": "google/gemma-3-12b","object":"model","owned_by":"organization_owner"},
    {"id": "ibm/granite-4-h-tiny","object":"model","owned_by":"organization_owner"},
    {"id": "microsoft/phi-4","object":"model","owned_by":"organization_owner"},
    {"id": "deepseek/deepseek-r1-0528-qwen3-8b","object":"model","owned_by":"organization_owner"},
    {"id": "microsoft/phi-4-mini-reasoning","object":"model","owned_by":"organization_owner"},
    {"id": "qwen/qwen3-4b-2507","object":"model","owned_by":"organization_owner"},
    {"id": "meta/llama-3.3-70b","object":"model","owned_by":"organization_owner"},
    {"id": "google/gemma-3n-e4b","object":"model","owned_by":"organization_owner"},
    {"id": "google/gemma-3-4b","object":"model","owned_by":"organization_owner"},
    {"id": "qwen3-vl-30b-a3b-thinking","object":"model","owned_by":"organization_owner"},
    {"id": "pixtral-12b","object":"model","owned_by":"organization_owner"},
    {"id": "granite-vision-3.2-2b","object":"model","owned_by":"organization_owner"},
    {"id": "qwen/qwen2.5-vl-7b","object":"model","owned_by":"organization_owner"},
    {"id": "kimi-dev-72b","object":"model","owned_by":"organization_owner"},
    {"id": "codestral-22b-v0.1","object":"model","owned_by":"organization_owner"},
    {"id": "openai/gpt-oss-20b","object":"model","owned_by":"organization_owner"},
    {"id": "microsoft/phi-4-reasoning-plus","object":"model","owned_by":"organization_owner"},
    {"id": "text-embedding-nomic-embed-text-v1.5","object":"model","owned_by":"organization_owner"},
    {"id": "smollm3-3b-128k","object":"model","owned_by":"organization_owner"},
    {"id": "qwen3-8b","object":"model","owned_by":"organization_owner"},
    {"id": "gemma-3-27b-it","object":"model","owned_by":"organization_owner"},
    {"id": "deepseek-r1-distill-qwen-7b","object":"model","owned_by":"organization_owner"}
  ],
  "object": "list"
}"""

# ---------------- CONFIG ----------------
class ModelConfig(Enum):
    LOCAL_DEFAULT = "qwen/qwen3-coder-30b"

class OutputRoot(Enum):
    BASE = "lmstudio_output_md"  # we'll create BASE/<sanitized_model_id>/...

FOLDER_PATH = "pdfs_20_2"             # Folder containing PDFs
PAGES_PER_CHUNK = 1
TEMPERATURE = 0.0
MAX_CONCURRENT = 10                  # LM-Studio parallelism

# LM Studio local server (OpenAI-compatible)
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://192.168.56.1:12345/v1")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "")

# Token/accounting knobs
CONTEXT_WINDOW_TOKENS = int(os.getenv("CTX_TOKENS", "8192"))   # adjust to your model’s context
MAX_OUTPUT_TOKENS     = int(os.getenv("MAX_OUTPUT_TOKENS", "1024"))
TOKEN_SAFETY_MARGIN   = int(os.getenv("TOKEN_SAFETY_MARGIN", "200"))  # guard for system/meta tokens

# To avoid blowing context windows; coarse char clamp (still useful even with token counting)
MAX_CHARS_PER_REQUEST = 15000
# ----------------------------------------


@dataclass
class ChunkInfo:
    pdf_name: str
    chunk_index: int
    chunk_path: str
    page_range_label: str


class TokenCounter:
    """
    Prefers tiktoken (accurate, fast). If unavailable, uses a chars→tokens heuristic (~4 chars/token).
    """
    def __init__(self):
        self._enc = None
        self._mode = "heuristic"
        try:
            import tiktoken  # type: ignore
            # cl100k_base works well for most chat models; it’s close enough for counting budget
            self._enc = tiktoken.get_encoding("cl100k_base")
            self._mode = "tiktoken:cl100k_base"
        except Exception:
            self._enc = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._enc:
            try:
                return len(self._enc.encode(text))
            except Exception:
                pass
        # Heuristic fallback: 4 chars ≈ 1 token (conservative)
        return max(1, len(text) // 4)

    def __repr__(self) -> str:
        return f"<TokenCounter mode={self._mode}>"


def sanitize_model_id(model_id: str) -> str:
    # make filesystem-safe-ish paths
    safe = model_id.replace("/", "_").replace(":", "_")
    for ch in ['\\', ' ', '*', '?', '"', '<', '>', '|']:
      safe = safe.replace(ch, "_")
    return safe


class AsyncPDFProcessor:
    def __init__(self, base_url: str, api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or None
        self.tok = TokenCounter()

    # ---------- FS & splitting ----------

    def get_pdf_files(self, folder_path: str, f_limit: Optional[int] = None) -> List[Path]:
        """
        f_limit: int or None. If None, do not limit; otherwise slice to f_limit.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise SystemExit(f"Folder {folder_path} does not exist.")
        all_pdfs = list(folder.glob("*.pdf"))
        if not all_pdfs:
            raise SystemExit(f"No PDF files found in {folder_path}")
        return all_pdfs[:f_limit] if isinstance(f_limit, int) and f_limit > 0 else all_pdfs

    def split_pdf_to_chunks(self, pdf_path: str, pages_per_chunk: int) -> List[ChunkInfo]:
        reader = PdfReader(pdf_path)
        n_pages = len(reader.pages)
        chunks: List[ChunkInfo] = []
        pdf_name = Path(pdf_path).stem

        for chunk_idx, start in enumerate(range(0, n_pages, pages_per_chunk)):
            end = min(start + pages_per_chunk, n_pages)
            writer = PdfWriter()
            for i in range(start, end):
                writer.add_page(reader.pages[i])

            fd, tmp_pdf = tempfile.mkstemp(
                suffix=f".{pdf_name}.chunk{chunk_idx:03d}.{start+1}-{end}.pdf"
            )
            os.close(fd)
            with open(tmp_pdf, "wb") as f:
                writer.write(f)

            page_range_label = f"pages {start+1}–{end}"
            chunks.append(ChunkInfo(
                pdf_name=pdf_name,
                chunk_index=chunk_idx,
                chunk_path=tmp_pdf,
                page_range_label=page_range_label
            ))
        return chunks

    # ---------- helpers ----------

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            reader = PdfReader(pdf_path)
            parts = []
            for i, page in enumerate(reader.pages, start=1):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                if txt.strip():
                    parts.append(f"\n\n[Page {i}]\n{txt.strip()}")
                else:
                    parts.append(f"\n\n[Page {i}]")
            return "".join(parts)
        except Exception as e:
            return f"[EXTRACTION_ERROR] {e}"

    def clamp(self, s: str, max_chars: int) -> str:
        return s if len(s) <= max_chars else s[:max_chars]

    async def _with_retries(self, coro_factory, *, tries=5, base=0.7, max_sleep=10.0):
        last = None
        for i in range(tries):
            try:
                return await coro_factory()
            except Exception as e:
                last = e
                msg = str(e).lower()
                retryable = any(s in msg for s in ("429", "timeout", "temporarily", "5", "rate", "connection", "reset", "refused"))
                if retryable and i < tries - 1:
                    sleep_for = min(max_sleep, base * (2 ** i)) + random.random()
                    await asyncio.sleep(sleep_for)
                else:
                    break
        raise last

    # ---------- LM Studio chat/completions ----------

    async def chat_markdownify(self, session: aiohttp.ClientSession, model: str, system_prompt: str, user_text: str, temperature: float, max_tokens: int) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        url = f"{self.base_url}/chat/completions"
        async with session.post(url, json=payload, headers=headers) as r:
            if r.status != 200:
                t = await r.text()
                raise Exception(f"LM Studio API error {r.status}: {t}")
            data = await r.json()
            return data["choices"][0]["message"]["content"].strip()

    # ---------- per-chunk pipeline ----------

    async def process_one_chunk(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        model_id: str,
        chunk: ChunkInfo
    ) -> Tuple[int, str]:
        async with semaphore:
            try:
                raw_text = self.extract_text_from_pdf(chunk.chunk_path)
                clamped = self.clamp(raw_text, MAX_CHARS_PER_REQUEST)

                system_prompt = (
                    "You convert PDFs into clean, GitHub-flavored Markdown for RAG.\n"
                    "Rules:\n"
                    "1) Preserve structure with proper #/##/### headings.\n"
                    "2) Convert tables to well-formatted Markdown tables when possible.\n"
                    "3) Keep lists and links.\n"
                    "4) Remove repeated headers/footers and OCR noise.\n"
                    "5) No summaries—output the full text you can read.\n"
                    "6) Output ONLY Markdown (no extra commentary).\n"
                    "7) Keep the original page numbers visible."
                )
                user_prompt = (
                    f"Convert this extracted PDF chunk ({chunk.page_range_label}) to Markdown.\n\n"
                    f"--- BEGIN EXTRACTED TEXT ---\n{clamped}\n--- END EXTRACTED TEXT ---"
                )

                # ---- token counting ----
                sys_toks = self.tok.count(system_prompt)
                usr_toks = self.tok.count(user_prompt)
                in_toks  = sys_toks + usr_toks

                # Budget for output within the context window
                available = max(256, CONTEXT_WINDOW_TOKENS - in_toks - TOKEN_SAFETY_MARGIN)
                max_out = max(64, min(MAX_OUTPUT_TOKENS, available))

                print(
                    f"🔢 {chunk.pdf_name} | {chunk.page_range_label} | "
                    f"model={model_id} | tokens: in={in_toks} (sys={sys_toks}, user={usr_toks}) | "
                    f"max_out={max_out} | ctx={CONTEXT_WINDOW_TOKENS} | {self.tok}"
                )

                md = await self._with_retries(
                    lambda: self.chat_markdownify(
                        session,
                        model_id,
                        system_prompt,
                        user_prompt,
                        temperature=TEMPERATURE,
                        max_tokens=int(max_out)
                    )
                )
                header = (
                    f"*Model:* `{model_id}`  \n"
                    f"*Token usage:* input={in_toks} (sys={sys_toks}, user={usr_toks}), "
                    f"max_out={max_out}, ctx={CONTEXT_WINDOW_TOKENS}\n"
                )
                formatted = f"\n\n---\n*Chunk {chunk.page_range_label}*\n{header}---\n\n{md.strip()}\n"
                print(f"✅ {chunk.pdf_name} | {chunk.page_range_label} | model={model_id}")
                return chunk.chunk_index, formatted
            except Exception as e:
                print(f"⚠️ {chunk.pdf_name} | {chunk.page_range_label} | model={model_id} failed: {e}")
                return chunk.chunk_index, f"\n\n---\n*Error processing {chunk.page_range_label} with {model_id}: {e}*\n---\n\n"
            finally:
                try:
                    os.remove(chunk.chunk_path)
                except OSError:
                    pass

    async def process_single_pdf(self, session: aiohttp.ClientSession, model_id: str, pdf_path: str, output_dir: str):
        pdf_name = Path(pdf_path).stem
        print(f"\n📄 Processing PDF: {Path(pdf_path).name} | model={model_id}")
        chunks = self.split_pdf_to_chunks(pdf_path, PAGES_PER_CHUNK)
        print(f"🔧 Split into {len(chunks)} chunks (window={MAX_CONCURRENT})")

        semaphore = asyncio.Semaphore(MAX_CONCURRENT*2)
        tasks = [asyncio.create_task(self.process_one_chunk(session, semaphore, model_id, ch)) for ch in chunks]
        results: List[Tuple[int, str]] = await asyncio.gather(*tasks)

        results.sort(key=lambda x: x[0])
        merged_md = "".join(md for _, md in results).strip() + "\n"

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{pdf_name}.md")
        async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
            await f.write(merged_md)
        print(f"✅ Wrote {out_path}")

    async def process_folder_for_model(self, model_id: str, folder_path: str, output_root: str, f_limit: Optional[int] = None):
        print(f"\n==============================")
        print(f"🎯 Model: {model_id}")
        print(f"==============================")
        print(f"🔍 Scanning: {folder_path}")
        pdf_files = self.get_pdf_files(folder_path, f_limit=f_limit)
        print(f"📚 Found {len(pdf_files)} PDFs")

        safe_model = sanitize_model_id(model_id)
        output_dir = os.path.join(output_root, safe_model)
        os.makedirs(output_dir, exist_ok=True)

        connector = aiohttp.TCPConnector(
            limit=MAX_CONCURRENT * 2,
            limit_per_host=MAX_CONCURRENT*2,
            ttl_dns_cache=3000,
            use_dns_cache=True,
        )
        timeout = aiohttp.ClientTimeout(total=9000)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for idx, pdf in enumerate(pdf_files, 1):
                print(f"\n🚀 [{idx}/{len(pdf_files)}] Start {Path(pdf).name} | model={model_id}")
                await self.process_single_pdf(session, model_id, pdf, output_dir)
                print(f"🏁 [{idx}/{len(pdf_files)}] Done {Path(pdf).name} | model={model_id}")


async def main_async():
    # Parse the models list
    try:
        models_payload = json.loads(MODELS_JSON)
        model_ids = [m["id"] for m in models_payload.get("data", []) if isinstance(m, dict) and "id" in m]
    except Exception as e:
        raise SystemExit(f"Invalid MODELS_JSON: {e}")

    if not model_ids:
        raise SystemExit("No model IDs found in MODELS_JSON")

    # Optional: skip obviously non-chat endpoints (embeddings). We'll just attempt all but skip common embedding ids.
    def should_skip(mid: str) -> bool:
        return "embedding" in mid.lower()

    output_root = OutputRoot.BASE.value
    os.makedirs(output_root, exist_ok=True)

    processor = AsyncPDFProcessor(LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY or "")

    # Process models sequentially (safer for a single local LM Studio server).
    for mid in model_ids:
        if should_skip(mid):
            print(f"⏭️ Skipping model (embedding detected): {mid}")
            continue
        try:
            await processor.process_folder_for_model(
                model_id=mid,
                folder_path=FOLDER_PATH,
                output_root=output_root,
                f_limit=5  # set to an int to limit number of PDFs per model
            )
        except Exception as e:
            print(f"❌ Fatal error while processing model {mid}: {e}")
            # continue with the next model


if __name__ == "__main__":
    asyncio.run(main_async())
