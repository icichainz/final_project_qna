
#!/usr/bin/env python3
"""
PDF -> page PNG -> Qwen2.5-VL -> Markdown per page -> merged <pdf>.md

Env:
  LMSTUDIO_BASE_URL (default: http://localhost:1234/v1)
  LMSTUDIO_API_KEY  (default: "")
  PAGE_DPI          (default: 250)
  TEMPERATURE       (default: 0.0)
  MAX_OUTPUT_TOKENS (default: 2048)
"""

import os, io, json, base64, asyncio, random, tempfile
import hashlib
import time
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from turtle import mode
from typing import List, Optional, Tuple, Dict, Any
import concurrent.futures
from multiprocessing import cpu_count

import aiohttp, aiofiles
from pypdf import PdfReader
from pdf2image import convert_from_path
from PIL import Image

# ---------- Config ----------
MODEL_ID = "qwen/qwen3-vl-8b"
#MODEL_ID = "qwen/qwen2.5-vl-7b"
class OutputRoot(Enum):
    BASE = "output_qwen_vl_markdown"

FOLDER_PATH = "pdfs_20_2"
PAGE_DPI = int(os.getenv("PAGE_DPI", "250"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "16000"))
MAX_CONCURRENT = 10
PDFS_LIMIT: Optional[int] = None

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://192.168.56.1:12345/v1").rstrip("/")
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "")

SYSTEM_PROMPT = (
    "You convert document page IMAGES into clean, faithful Markdown.\n"
    "Rules:\n"
    "1) Output ONLY Markdown (no code fences, no explanations).\n"
    "2) Preserve structure with proper #/##/### headings, lists, and emphasis.\n"
    "3) Convert tables to Markdown tables when possible.\n"
    "4) Remove repeated headers/footers and OCR noise.\n"
    "5) Keep page numbers if visible in the image.\n"
    "6) Transcribe exactly what's present; do not hallucinate.\n"
)

def sanitize_model_id(model_id: str) -> str:
    safe = model_id.replace("/", "_").replace(":", "_")
    for ch in ['\\',' ','*','?','"','<','>','|']:
        safe = safe.replace(ch, "_")
    return safe

def ensure_png(img: Image.Image) -> Image.Image:
    return img if img.mode in ("RGB", "RGBA") else img.convert("RGB")

def encode_image_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

@dataclass
class PageJob:
    page_index: int
    image_b64: Optional[str] = None
    image_path: Optional[str] = None

# Disk cache root for PNG pages
CACHE_ROOT = Path(".image_cache")
CACHE_ROOT.mkdir(exist_ok=True)

def _pdf_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def _write_atomic_json(path: Path, data: Dict[str, Any]):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)

def encode_png_file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def pdf_to_png_cache_worker(pdf_path: str, cache_dir: str, dpi: int = PAGE_DPI) -> Dict[str, Any]:
    """
    Process-pool worker: convert PDF pages -> PNG files and atomically place them under cache_dir.
    Returns metadata dict.
    """
    pdf_path = Path(pdf_path)
    cache_dir = Path(cache_dir)
    tmp_out = cache_dir.with_suffix(".tmp_" + str(int(time.time()*1000)))
    tmp_out.mkdir(parents=True, exist_ok=True)
    try:
        # validate & convert
        _ = PdfReader(str(pdf_path))
        pages = convert_from_path(str(pdf_path), dpi=dpi, fmt="png", output_folder=str(tmp_out))
        page_files = []
        for i, page in enumerate(pages, start=1):
            img = ensure_png(page)
            name = f"page_{i:03d}.png"
            path = tmp_out / name
            img.save(path, format="PNG")
            page_files.append(name)
        pdf_hash = _pdf_hash(pdf_path)
        meta = {
            "pdf_name": pdf_path.name,
            "pdf_hash": pdf_hash,
            "dpi": dpi,
            "created_at": int(time.time()),
            "pages": page_files,
        }
        # replace existing cache dir atomically
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        tmp_out.replace(cache_dir)
        _write_atomic_json(cache_dir / "metadata.json", meta)
        return meta
    except Exception:
        if tmp_out.exists():
            shutil.rmtree(tmp_out)
        raise


class QwenVLMarkdown:
    def __init__(self, base_url: str, api_key: str , model_id: str ):
        self.base_url = base_url
        self.api_key = api_key or None
        self.model_id = model_id

    def get_pdf_files(self, folder: str, f_limit: Optional[int] = None) -> List[Path]:
        p = Path(folder)
        if not p.exists():
            raise SystemExit(f"Folder {folder} does not exist.")
        pdfs = sorted(p.glob("*.pdf"))
        if not pdfs:
            raise SystemExit(f"No PDF files found in {folder}")
        return pdfs[:f_limit] if isinstance(f_limit, int) and f_limit > 0 else pdfs

    def pdf_to_page_jobs(self, pdf_path: Path, dpi: int = PAGE_DPI) -> List[PageJob]:
        # Validate open
        try:
            _ = PdfReader(str(pdf_path))
        except Exception as e:
            raise SystemExit(f"Invalid PDF {pdf_path.name}: {e}")
        with tempfile.TemporaryDirectory() as _tmp:
            pages = convert_from_path(str(pdf_path), dpi=dpi, fmt="png", output_folder=_tmp)
            jobs: List[PageJob] = []
            for i, page in enumerate(pages):
                img = ensure_png(page)
                jobs.append(PageJob(i, image_b64=encode_image_b64(img)))
            return jobs

    async def _page_to_markdown(self, session: aiohttp.ClientSession, job: PageJob) -> Tuple[int, str]:
        # ensure job.image_b64 is populated; if only image_path present, encode via process pool
        if not job.image_b64 and job.image_path:
            loop = asyncio.get_running_loop()
            job.image_b64 = await loop.run_in_executor(_PROCESS_POOL, encode_png_file_to_b64, job.image_path)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        content = [
            {"type": "text", "text": "Transcribe this page image into clean Markdown, following the rules."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{job.image_b64}"}},
        ]
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_OUTPUT_TOKENS,
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        async with session.post(url, json=payload, headers=headers) as r:
            txt = await r.text()
            if r.status != 200:
                raise RuntimeError(f"LM Studio API {r.status}: {txt}")
            
            # Parse JSON with error handling
            try:
                data = json.loads(txt)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON response from API: {txt[:500]}")
            
            # Debug: Check response structure
            if "choices" not in data:
                print(f"\n⚠️  Unexpected API response structure for page {job.page_index + 1}")
                print(f"Response keys: {list(data.keys())}")
                print(f"Full response (first 1000 chars):")
                print(json.dumps(data, indent=2)[:1000])
                raise RuntimeError(f"API response missing 'choices' key. Keys present: {list(data.keys())}")
            
            if not data["choices"]:
                raise RuntimeError(f"API returned empty choices array")
            
            if "message" not in data["choices"][0]:
                raise RuntimeError(f"API response missing 'message' in choices[0]. Keys: {list(data['choices'][0].keys())}")
            
            if "content" not in data["choices"][0]["message"]:
                raise RuntimeError(f"API response missing 'content' in message. Keys: {list(data['choices'][0]['message'].keys())}")
            
            md = data["choices"][0]["message"]["content"].strip()
        
        # strip accidental fences
        if md.startswith("```"):
            md = md.strip("`")
            md = md.split("\n", 1)[1] if "\n" in md else md
        return job.page_index, md

    async def process_pdf(self, session: aiohttp.ClientSession, pdf_path: Path, out_dir: Path, cache_dir: Optional[Path] = None):
        pdf_name = pdf_path.stem
        print(f"\n📄 {pdf_path.name} (model={self.model_id})")

        # If a cache_dir is provided, build PageJob list from on-disk PNGs (metadata.json)
        if cache_dir is not None and (cache_dir / "metadata.json").exists():
            meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
            jobs = [PageJob(i, image_path=str(cache_dir / name)) for i, name in enumerate(meta.get("pages", []))]
        else:
            # OFFLOAD cpu-bound PDF->PNG+BASE64 into process pool to avoid blocking the event loop
            loop = asyncio.get_running_loop()
            jobs = await loop.run_in_executor(_PROCESS_POOL, self.pdf_to_page_jobs, pdf_path, PAGE_DPI)

        print(f"🖼️  {len(jobs)} page image(s) @ {PAGE_DPI} DPI")

        # Use a small fixed pool of async workers instead of creating one task per page immediately.
        worker_count = min(MAX_CONCURRENT, max(1, len(jobs)))
        queue: asyncio.Queue = asyncio.Queue()
        for j in jobs:
            await queue.put(j)

        results: List[Tuple[int, str]] = []
        results_lock = asyncio.Lock()  # protect results append and completed_count
        completed_count = 0
        total_pages = len(jobs)

        async def worker():
            nonlocal completed_count
            while True:
                try:
                    job: PageJob = await queue.get()
                except asyncio.CancelledError:
                    return
                try:
                    for attempt in range(5):
                        try:
                            page_idx, markdown = await self._page_to_markdown(session, job)

                            async with results_lock:
                                results.append((page_idx, markdown))
                                completed_count += 1

                                # Print truncated output
                                truncated = markdown[:200].replace('\n', ' ')
                                if len(markdown) > 200:
                                    truncated += "..."
                                print(f"✓ Page {page_idx + 1}/{total_pages} [{completed_count}/{total_pages}] | {truncated}")

                            break
                        except Exception as e:
                            error_str = str(e).lower()
                            is_retryable = any(s in error_str for s in ("timeout","temporar","reset","refused","5","429"))
                            if attempt < 4 and is_retryable:
                                wait_time = min(8.0, 0.7*(2**attempt)) + random.random()
                                print(f"⚠️  Page {job.page_index + 1} attempt {attempt + 1} failed: {str(e)[:100]}")
                                print(f"   Retrying in {wait_time:.1f}s...")
                                await asyncio.sleep(wait_time)
                                continue
                            print(f"❌ Page {job.page_index + 1} failed after {attempt + 1} attempts")
                            raise
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        await queue.join()

        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

        results.sort(key=lambda x: x[0])

        # Merge -> add page separators
        merged = []
        for idx, md in results:
            merged.append(f"\n\n---\n**Page {idx+1}**\n---\n\n{md.strip()}")
        final_md = "\n".join(merged).strip() + "\n"

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pdf_name}.md"
        async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
            await f.write(final_md)
        print(f"✅ Wrote {out_path}")

    async def process_folder(self, folder_path: str, output_root: str, f_limit: Optional[int] = None):
        pdfs = self.get_pdf_files(folder_path, f_limit=f_limit)
        safe_model = sanitize_model_id(self.model_id)
        out_dir = Path(output_root) / safe_model

        connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT*4, limit_per_host=MAX_CONCURRENT*4, ttl_dns_cache=300)
        timeout = aiohttp.ClientTimeout(total=3600)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            for i, pdf in enumerate(pdfs, 1):
                print(f"\n🚀 [{i}/{len(pdfs)}] {pdf.name}")
                # try to find a cache for this pdf
                cache_dir = CACHE_ROOT / _pdf_hash(pdf)
                if (cache_dir / "metadata.json").exists():
                    await self.process_pdf(session, pdf, out_dir, cache_dir=cache_dir)
                else:
                    await self.process_pdf(session, pdf, out_dir, cache_dir=None)
                print(f"🏁 [{i}/{len(pdfs)}] Done {pdf.name}")

# Process pool will be created at runtime inside main_async to avoid spawning pools at import time
_PROCESS_POOL: Optional[concurrent.futures.ProcessPoolExecutor] = None

# ---------- Main ----------
async def main_async():
    global _PROCESS_POOL
    # create a process pool here to avoid worker processes spawning their own pools
    _PROCESS_POOL = concurrent.futures.ProcessPoolExecutor(max_workers=max(1, cpu_count() - 1))
    try:
        # build disk cache for all PDFs once (concurrent in process pool)
        master = QwenVLMarkdown(LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY, model_id=MODEL_ID)
        pdfs = master.get_pdf_files(FOLDER_PATH, f_limit=PDFS_LIMIT)
        loop = asyncio.get_running_loop()
        tasks = []
        to_run = []
        for pdf in pdfs:
            pdf_hash = _pdf_hash(pdf)
            cache_dir = CACHE_ROOT / pdf_hash
            if not (cache_dir / "metadata.json").exists():
                to_run.append((str(pdf), str(cache_dir)))
        if to_run:
            # Run cache conversions in batches to avoid overwhelming the machine
            batch_size = max(1, min(4, cpu_count() - 1))
            print(f"🔁 Building cache for {len(to_run)} PDFs in batches of {batch_size} using process pool")
            for i in range(0, len(to_run), batch_size):
                batch = to_run[i : i + batch_size]
                futs = [loop.run_in_executor(_PROCESS_POOL, pdf_to_png_cache_worker, pdf, cdir, PAGE_DPI) for pdf, cdir in batch]
                results = await asyncio.gather(*futs, return_exceptions=True)
                # handle exceptions per item; if a worker failed, retry synchronously to surface the exception
                for (pdf, cdir), res in zip(batch, results):
                    if isinstance(res, Exception):
                        print(f"⚠️  Cache worker failed for {pdf}: {res}. Retrying synchronously to show full traceback...")
                        try:
                            pdf_to_png_cache_worker(pdf, cdir, PAGE_DPI)
                        except Exception as e:
                            # If synchronous retry also fails, raise to stop execution (user can inspect)
                            print(f"❌ Synchronous cache build also failed for {pdf}: {e}")
                            raise

        # run model tasks concurrently; each will read cached PNGs
        model_ids = [
            "pixtral-12b",
            "granite-vision-3.2-2b",
            "microsoft/phi-4",
        ]
        for mid in model_ids:
            app = QwenVLMarkdown(LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY, model_id=mid)
            tasks.append(asyncio.create_task(app.process_folder(FOLDER_PATH, OutputRoot.BASE.value, f_limit=PDFS_LIMIT)))
        await asyncio.gather(*tasks)
    finally:
        # cleanly shutdown process pool
        if _PROCESS_POOL is not None:
            _PROCESS_POOL.shutdown(wait=True)

if __name__ == "__main__":
    asyncio.run(main_async())