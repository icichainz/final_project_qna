#!/usr/bin/env python3
"""
PDF -> page image -> VLM -> Markdown per page -> merged <pdf>.md

Run:  python scripts/extract_corpus.py   (or python -m gcf_qna.extraction.vlm)
Paths come from gcf_qna.config: pdfs in data/raw/pdfs, page cache in
data/cache/pages, output under data/extracted/vlm/<model>/.

Optimised rewrite of qwen_vlm_image_processing.py.

Key differences vs. the original:
  * PyMuPDF rendering straight to the model's useful resolution (no 250-DPI
    intermediate, no poppler subprocess, no duplicate PNGs on disk).
  * JPEG cache instead of PNG  (~49 GB -> ~18 GB for this corpus).
  * Per-page markdown sidecars => resume after a crash, no page ever silently lost.
  * Models run sequentially (LM Studio can only hold one VLM at a time).
  * Real retry classification, truncation detection, repetition-loop detection.
  * Optional PDF text layer fed to the model as a transcription reference.

Env:
  LMSTUDIO_BASE_URL   default http://192.168.56.1:12345/v1
  LMSTUDIO_API_KEY    default ""
  PAGE_MEGAPIXELS     default 2.0     target render size (accuracy/speed dial)
  JPEG_QUALITY        default 92
  TEMPERATURE         default 0.0
  MAX_OUTPUT_TOKENS   default 4096
  MAX_CONCURRENT      default 10       in-flight requests against the endpoint.
                                      KEEP AT 1 for LM Studio: two concurrent
                                      multimodal requests crash the model process
                                      (verified 2026-08-14 against qwen3-vl-8b).
  CRASH_COOLDOWN      default 45      seconds to wait for LM Studio to reload a
                                      model after it crashes
  USE_TEXT_LAYER      default 1       feed embedded PDF text as a reference
  PDFS_LIMIT          default ""      process only the first N PDFs
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import io
import json
import os
import random
import re
import shutil
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import pymupdf
from PIL import Image

from gcf_qna import config

# ---------------------------------------------------------------- config ----

FOLDER_PATH = os.getenv("FOLDER_PATH", str(config.RAW_PDF_DIR))
OUTPUT_ROOT = Path(os.getenv("OUTPUT_ROOT", str(config.EXTRACTED_DIR / "vlm")))
CACHE_ROOT = Path(os.getenv("CACHE_ROOT", str(config.PAGE_CACHE_DIR)))

# Built-in roster for a no-argument run: every listed model is processed
# sequentially, resuming wherever each left off.
DEFAULT_MODELS = [
    "qwen/qwen2.5-vl-7b",
    "pixtral-12b",
    "qwen/qwen3-vl-8b",
    "zai-org/glm-4.6v-flash",
]

# Precedence: CLI positionals > VLM_MODELS env > DEFAULT_MODELS
_env_models = [m.strip() for m in os.getenv("VLM_MODELS", "").split(",") if m.strip()]
MODEL_IDS = _env_models or DEFAULT_MODELS

LMSTUDIO_BASE_URL = config.LMSTUDIO_BASE_URL
LMSTUDIO_API_KEY = config.LMSTUDIO_API_KEY

PAGE_MEGAPIXELS = float(os.getenv("PAGE_MEGAPIXELS", "2.0"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "92"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "320000"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "10"))
USE_TEXT_LAYER = os.getenv("USE_TEXT_LAYER", "1") == "1"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "9999999999"))
CRASH_COOLDOWN = float(os.getenv("CRASH_COOLDOWN", "45"))
CIRCUIT_TRIP = int(os.getenv("CIRCUIT_TRIP", "20"))
# Documents still broken after this many full passes are skipped unless
# --retry-exhausted is given: one poisoned page must not tax every future run.
ATTEMPT_CEILING = int(os.getenv("ATTEMPT_CEILING", "3"))
MAX_ATTEMPTS = 6
RENDER_AHEAD = 2                      # PDFs rendered ahead of the LLM stage
_limit = os.getenv("PDFS_LIMIT", "")
PDFS_LIMIT: Optional[int] = int(_limit) if _limit.strip().isdigit() else None

SYSTEM_PROMPT = (
    "You convert document page IMAGES into clean, faithful Markdown.\n"
    "Rules:\n"
    "1) Output ONLY Markdown. Never wrap the answer in code fences.\n"
    "2) Preserve structure with #/##/### headings, lists, and emphasis.\n"
    "3) Render every table as a Markdown table WITH a header row and a\n"
    "   |---| separator. If the table has no header, use empty header cells.\n"
    "4) Drop running headers/footers, watermarks and OCR noise.\n"
    "5) Replace figures/charts/photos with ![<short literal caption>]() -- an\n"
    "   EMPTY link target. Never invent a URL or a placeholder image service. If\n"
    "   the figure carries data, transcribe its axis labels and legend below it.\n"
    "6) Transcribe exactly what is present. Never invent content. If a region is\n"
    "   illegible write [illegible].\n"
    "7) If the page is blank, output exactly: [blank page]\n"
)

USER_PROMPT = "Transcribe this page image into clean Markdown, following the rules."

# ------------------------------------------------------------- rendering ----

# Pages whose luminance barely varies are blank; skip the round trip entirely.
BLANK_STDDEV = 2.0


def _fingerprint(path: Path) -> str:
    """Cheap, collision-safe-enough PDF identity: size + head/tail content."""
    st = path.stat()
    h = zlib.crc32(path.name.encode())
    with path.open("rb") as f:
        head = f.read(1 << 20)
        if st.st_size > (1 << 21):
            f.seek(-(1 << 20), os.SEEK_END)
            tail = f.read(1 << 20)
        else:
            tail = b""
    import hashlib

    d = hashlib.sha256()
    d.update(str(st.st_size).encode())
    d.update(head)
    d.update(tail)
    return f"{d.hexdigest()[:32]}_{h:08x}"


def _write_atomic_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


def _overlap_frac(a: "pymupdf.Rect", b: "pymupdf.Rect") -> float:
    r = pymupdf.Rect(a)
    r.intersect(b)
    return (abs(r) / abs(a)) if abs(a) else 0.0


def extract_page_boxes(page, zoom: float, w: int, h: int) -> Dict[str, Any]:
    """Text lines, tables, and figures for one page, in PIXEL coordinates of
    the cached JPEG (zoom applied), so viewers can draw rects directly.

    Sources (all from the born-digital PDF, no OCR):
      lines   -> page.get_text("dict")  block/line hierarchy
      tables  -> page.find_tables()     ruled-table detector
      figures -> raster image rects + clustered vector drawings
    """
    def px(b) -> List[float]:
        return [round(b[0] * zoom, 1), round(b[1] * zoom, 1),
                round(b[2] * zoom, 1), round(b[3] * zoom, 1)]

    lines: List[Dict[str, Any]] = []
    for blk in page.get_text("dict").get("blocks", []):
        if blk.get("type") != 0:
            continue
        for ln in blk.get("lines", []):
            spans = [sp for sp in ln.get("spans", []) if sp.get("text", "").strip()]
            text = " ".join(sp["text"].strip() for sp in spans).strip()
            if not text:
                continue
            # Union of span boxes beats the raw line box: dotted leaders and
            # link annotations can inflate line bboxes across several rows.
            x0 = min(sp["bbox"][0] for sp in spans)
            y0 = min(sp["bbox"][1] for sp in spans)
            x1 = max(sp["bbox"][2] for sp in spans)
            y1 = max(sp["bbox"][3] for sp in spans)
            max_size = max((sp.get("size", 10.0) for sp in spans), default=10.0)
            y1 = min(y1, y0 + 1.6 * max_size)          # cap runaway heights
            lines.append({"bbox": px((x0, y0, x1, y1)), "text": text})

    tables: List[Dict[str, Any]] = []
    table_rects: List[pymupdf.Rect] = []
    try:
        for t in page.find_tables().tables:
            table_rects.append(pymupdf.Rect(t.bbox))
            tables.append({"bbox": px(t.bbox), "rows": t.row_count, "cols": t.col_count})
    except Exception:
        pass

    figures: List[Dict[str, Any]] = []
    try:
        for img in page.get_images(full=True):
            for r in page.get_image_rects(img):
                if r.width < 20 or r.height < 20:          # icons, bullets, logos
                    continue
                figures.append({"bbox": px(tuple(r)), "kind": "image"})
    except Exception:
        pass
    try:
        page_area = abs(page.rect)
        for r in page.cluster_drawings():
            if r.width < 30 or r.height < 30:
                continue
            if page_area and abs(r) > 0.9 * page_area:     # page border frames
                continue
            if (abs(r) < 0.05 * page_area
                    and any(_overlap_frac(r, tr) > 0.5 for tr in table_rects)):
                continue    # small cluster inside a table = ruling, not a figure
            figures.append({"bbox": px(tuple(r)), "kind": "drawing"})
    except Exception:
        pass

    return {"w": w, "h": h, "zoom": round(zoom, 4), "rotation": page.rotation,
            "lines": lines, "tables": tables, "figures": figures}


def render_pdf_worker(pdf_path: str, cache_dir: str) -> Dict[str, Any]:
    """Process-pool worker: render every page to a right-sized JPEG.

    Rendering happens directly at the target pixel budget, so no oversized
    bitmap is ever materialised. Returns the cache metadata dict.
    """
    pdf_p, cache_p = Path(pdf_path), Path(cache_dir)
    tmp_out = cache_p.with_name(cache_p.name + f".tmp{os.getpid()}")
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_out.mkdir(parents=True)
    try:
        doc = pymupdf.open(pdf_p)
        pages: List[Dict[str, Any]] = []
        for i, page in enumerate(doc):
            rect = page.rect
            area = max(1.0, rect.width * rect.height)
            zoom = min(4.0, (PAGE_MEGAPIXELS * 1e6 / area) ** 0.5)
            pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), colorspace=pymupdf.csRGB)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

            entry: Dict[str, Any] = {"n": i + 1, "w": pix.width, "h": pix.height}
            gray = img.convert("L")
            stat_min, stat_max = gray.getextrema()
            entry["blank"] = (stat_max - stat_min) < BLANK_STDDEV

            if not entry["blank"]:
                name = f"page_{i + 1:04d}.jpg"
                img.save(tmp_out / name, format="JPEG", quality=JPEG_QUALITY,
                         optimize=True, progressive=False, subsampling=0)
                entry["img"] = name
                boxes = extract_page_boxes(page, zoom, pix.width, pix.height)
                bname = f"page_{i + 1:04d}.boxes.json"
                (tmp_out / bname).write_text(
                    json.dumps(boxes, ensure_ascii=False), encoding="utf-8")
                entry["boxes"] = bname

            if USE_TEXT_LAYER:
                txt = page.get_text("text").strip()
                if 40 <= len(txt) <= 8000:
                    tname = f"page_{i + 1:04d}.txt"
                    (tmp_out / tname).write_text(txt, encoding="utf-8")
                    entry["txt"] = tname
            pages.append(entry)
        doc.close()

        meta = {
            "pdf_name": pdf_p.name,
            "megapixels": PAGE_MEGAPIXELS,
            "jpeg_quality": JPEG_QUALITY,
            "created_at": int(time.time()),
            "pages": pages,
        }
        if cache_p.exists():
            shutil.rmtree(cache_p)
        tmp_out.replace(cache_p)
        _write_atomic_json(cache_p / "metadata.json", meta)
        return meta
    except Exception:
        shutil.rmtree(tmp_out, ignore_errors=True)
        raise


@dataclass
class PageJob:
    index: int                  # 0-based
    image_path: Optional[Path]
    text_hint_path: Optional[Path]
    blank: bool


# ------------------------------------------------------------ post-check ----

_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*\n(.*?)\n?\s*```\s*$", re.S)


def strip_fences(md: str) -> str:
    m = _FENCE_RE.match(md)
    if m:
        return m.group(1).strip()
    if md.startswith("```"):
        md = md.split("\n", 1)[1] if "\n" in md else md[3:]
    if md.rstrip().endswith("```"):
        md = md.rstrip()[:-3]
    return md.strip()


def is_degenerate(md: str) -> bool:
    """Detect the VLM repetition loop failure mode (same line/phrase forever)."""
    if len(md) < 400:
        return False
    # Highly repetitive text compresses absurdly well.
    if len(zlib.compress(md.encode("utf-8"), 6)) / len(md) < 0.045:
        return True
    lines = [ln.strip() for ln in md.splitlines() if ln.strip()]
    if len(lines) >= 12:
        run = best = 1
        for a, b in zip(lines, lines[1:]):
            run = run + 1 if a == b else 1
            best = max(best, run)
        if best >= 12:
            return True
    return False


RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# LM Studio wraps backend failures in an Express HTML error page, so the useful
# text sits inside <pre> several hundred characters in. Dig it out before logging.
_PRE_RE = re.compile(r"<pre>(.*?)(?:</pre>|$)", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")

# A crashed model answers with this until LM Studio reloads the weights. Retrying
# immediately is pointless -- the backend needs time to come back up.
_CRASH_MARKERS = (
    "model has crashed",
    "internal server error",
    "model not loaded",
    "no model loaded",
    "model unloaded",
    "failed to load model",
    "model_not_found",
    "loading model",
)


def extract_error(body: str, limit: int = 220) -> str:
    """Reduce an LM Studio error body to the one line that matters."""
    txt = body
    m = _PRE_RE.search(body)
    if m:
        txt = m.group(1)
    elif body.lstrip().startswith("{"):
        try:
            data = json.loads(body)
            err = data.get("error", data)
            txt = err.get("message", json.dumps(err)) if isinstance(err, dict) else str(err)
        except (json.JSONDecodeError, AttributeError):
            pass
    txt = _TAG_RE.sub(" ", txt).replace("&nbsp;", " ")
    txt = " ".join(txt.split())
    return txt[:limit] if txt else f"empty body ({len(body)} bytes)"


def looks_like_crash(msg: str) -> bool:
    low = msg.lower()
    return any(mark in low for mark in _CRASH_MARKERS)


class RetryableError(RuntimeError):
    pass


class ServerCrash(RetryableError):
    """The backend model process died; it needs time, not another request."""


class CircuitOpen(RuntimeError):
    """Too many consecutive failures -- the endpoint is not coming back."""


class Endpoint:
    """Gate in front of the inference server.

    Two jobs beyond plain rate limiting. First, when the model process crashes,
    exactly one task drives the cooldown while every other task parks on the
    health event -- otherwise all in-flight pages would each pile another doomed
    request onto a server that is busy reloading weights. Second, it counts
    consecutive failures so a permanently dead endpoint aborts the run instead of
    grinding through the whole corpus.
    """

    def __init__(self, concurrency: int, cooldown: float = CRASH_COOLDOWN,
                 trip: int = CIRCUIT_TRIP):
        self.sem = asyncio.Semaphore(concurrency)
        self.healthy = asyncio.Event()
        self.healthy.set()
        self._recovery_lock = asyncio.Lock()
        self.cooldown = cooldown
        self.trip = trip
        self.consecutive_failures = 0
        self.aborted = asyncio.Event()

    def reset(self) -> None:
        """Clear state between model passes -- one bad model must not doom the next."""
        self.consecutive_failures = 0
        self.aborted.clear()
        self.healthy.set()

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.trip:
            raise CircuitOpen(
                f"{self.consecutive_failures} consecutive failures against "
                f"{LMSTUDIO_BASE_URL} -- stopping so the run does not burn the "
                f"whole corpus. Check that the model is loaded and serving."
            )

    async def quarantine(self, reason: str) -> None:
        """Hold every caller off while the backend reloads. Idempotent."""
        if self._recovery_lock.locked():
            await self.healthy.wait()      # someone else is already recovering
            return
        async with self._recovery_lock:
            if not self.healthy.is_set():
                return
            self.healthy.clear()
            print(f"   🛑 backend down ({reason}); pausing {self.cooldown:.0f}s "
                  f"for LM Studio to reload")
            try:
                await asyncio.sleep(self.cooldown)
            finally:
                self.healthy.set()
            print("   ▶️  resuming")


# ---------------------------------------------------------------- client ----


class VLMExtractor:
    def __init__(self, base_url: str, api_key: str, model_id: str, endpoint: "Endpoint",
                 io_pool: concurrent.futures.ThreadPoolExecutor):
        self.base_url = base_url
        self.api_key = api_key or None
        self.model_id = model_id
        self.ep = endpoint
        self.io_pool = io_pool
        self.stats = {"ok": 0, "blank": 0, "cached": 0, "failed": 0,
                      "truncated": 0, "retries": 0, "crashes": 0}

    # -- one page ------------------------------------------------------------

    async def _call(self, session: aiohttp.ClientSession, b64: str, hint: Optional[str],
                    max_tokens: int) -> Tuple[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        text = USER_PROMPT
        if hint:
            text += (
                "\n\nFor reference, the text layer embedded in this PDF page is below. "
                "Use it to resolve ambiguous characters, but trust the IMAGE for layout, "
                "reading order and tables, and ignore the reference where it disagrees "
                "with what you can see.\n\n<reference>\n" + hint + "\n</reference>"
            )
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": text},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ]},
            ],
            "temperature": TEMPERATURE,
            "top_p": 1.0,
            "max_tokens": max_tokens,
            "stream": False,
        }
        url = f"{self.base_url}/chat/completions"
        await self.ep.healthy.wait()
        async with self.ep.sem:
            async with session.post(url, json=payload, headers=headers) as r:
                body = await r.text()
                if r.status != 200:
                    detail = extract_error(body)
                    msg = f"HTTP {r.status}: {detail}"
                    # A crashed backend reports 400 as readily as 500, so trust the
                    # message over the status code here.
                    if looks_like_crash(detail):
                        raise ServerCrash(msg)
                    if r.status in RETRYABLE_STATUS:
                        raise RetryableError(msg)
                    raise RuntimeError(msg)
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            raise RetryableError(f"non-JSON response: {body[:300]}")
        choices = data.get("choices") or []
        if not choices or "message" not in choices[0]:
            raise RetryableError(f"malformed response: {json.dumps(data)[:300]}")
        content = choices[0]["message"].get("content") or ""
        return content.strip(), (choices[0].get("finish_reason") or "")

    async def page_to_markdown(self, session: aiohttp.ClientSession, job: PageJob) -> str:
        loop = asyncio.get_running_loop()
        b64 = await loop.run_in_executor(self.io_pool, _read_b64, job.image_path)
        hint = None
        if job.text_hint_path is not None:
            hint = await loop.run_in_executor(self.io_pool, _read_text, job.text_hint_path)

        budget = MAX_OUTPUT_TOKENS
        last_exc: Optional[BaseException] = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                md, finish = await self._call(session, b64, hint, budget)
                md = strip_fences(md)

                if finish == "length" and budget < 8192:
                    # Dense table/page: silently truncated output. Re-ask with room.
                    budget = min(8192, budget * 2)
                    self.stats["retries"] += 1
                    continue
                if is_degenerate(md):
                    # Repetition loop: retry without the hint, which is the usual trigger.
                    if hint is not None:
                        hint = None
                        self.stats["retries"] += 1
                        continue
                    raise RetryableError("degenerate repetition loop in output")
                if finish == "length":
                    self.stats["truncated"] += 1
                    md += "\n\n<!-- TRUNCATED: output hit the token limit -->"
                self.ep.record_success()
                return md
            except (RetryableError, aiohttp.ClientError, asyncio.TimeoutError) as e:
                last_exc = e
                self.ep.record_failure()        # raises CircuitOpen if the endpoint is dead
                if isinstance(e, ServerCrash):
                    # One task drives the cooldown; the rest park on the health
                    # event rather than each firing another request at a server
                    # that is mid-reload.
                    self.stats["crashes"] += 1
                    await self.ep.quarantine(str(e)[:80])
                    continue
                if attempt == MAX_ATTEMPTS - 1:
                    break
                self.stats["retries"] += 1
                delay = min(20.0, 0.8 * (2 ** attempt)) + random.random()
                # Only the first attempt is worth a line; the rest would flood the log.
                if attempt == 0:
                    print(f"   ⚠️  page {job.index + 1}: {str(e)[:140]} — retrying")
                await asyncio.sleep(delay)
        raise RuntimeError(f"page {job.index + 1} failed after {MAX_ATTEMPTS} attempts: {last_exc}")

    # -- one pdf -------------------------------------------------------------

    async def process_pdf(self, session: aiohttp.ClientSession, pdf_path: Path,
                          cache_dir: Path, out_dir: Path) -> Dict[str, Any]:
        t_doc = time.perf_counter()
        meta = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
        entries = meta["pages"]
        stem = pdf_path.stem
        final_path = out_dir / f"{stem}.md"
        page_dir = out_dir / ".pages" / stem
        page_dir.mkdir(parents=True, exist_ok=True)

        if final_path.exists() and final_path.stat().st_size > 0:
            print(f"   ⏭️  already merged: {final_path.name}")
            return {"status": "ok", "pages_total": len(entries),
                    "pages_done": len(entries), "pages_to_retry": [],
                    "last_error": None, "duration_s": 0.0}

        todo: List[PageJob] = []
        for i, e in enumerate(entries):
            sidecar = page_dir / f"page_{i + 1:04d}.md"
            if sidecar.exists():
                self.stats["cached"] += 1
                continue
            if e.get("blank") or not e.get("img"):
                sidecar.write_text("[blank page]\n", encoding="utf-8")
                self.stats["blank"] += 1
                continue
            todo.append(PageJob(
                index=i,
                image_path=cache_dir / e["img"],
                text_hint_path=cache_dir / e["txt"] if e.get("txt") else None,
                blank=False,
            ))

        total = len(entries)
        print(f"   🖼️  {total} pages | {len(todo)} to transcribe "
              f"({self.stats['cached']} cached, {self.stats['blank']} blank)")

        done = 0
        failures: List[int] = []
        last_error: Optional[str] = None
        t0 = time.perf_counter()

        async def run(job: PageJob) -> None:
            nonlocal done, last_error
            if self.ep.aborted.is_set():
                return
            try:
                md = await self.page_to_markdown(session, job)
                (page_dir / f"page_{job.index + 1:04d}.md").write_text(md + "\n", encoding="utf-8")
                self.stats["ok"] += 1
            except CircuitOpen:
                # Endpoint is gone. Stop the whole pass rather than let every
                # remaining page fail one by one.
                self.ep.aborted.set()
                raise
            except Exception as e:
                failures.append(job.index)
                last_error = f"page {job.index + 1}: {str(e)[:200]}"
                self.stats["failed"] += 1
                print(f"   ❌ page {job.index + 1}: {str(e)[:160]}")
            done += 1
            if done % 10 == 0 or done == len(todo):
                rate = done / max(1e-6, time.perf_counter() - t0)
                eta = (len(todo) - done) / max(1e-6, rate)
                print(f"   … {done}/{len(todo)} pages | {rate * 60:.1f} pages/min | ETA {eta / 60:.1f} min")

        # A flat task set bounded by the shared endpoint gate: no per-PDF worker
        # pool, so concurrency stays exactly MAX_CONCURRENT throughout.
        outcomes = await asyncio.gather(*(run(j) for j in todo), return_exceptions=True)
        for o in outcomes:
            if isinstance(o, CircuitOpen):
                raise o

        parts = []
        missing_pages: List[int] = []
        for i in range(total):
            sidecar = page_dir / f"page_{i + 1:04d}.md"
            if sidecar.exists():
                body = sidecar.read_text(encoding="utf-8").strip()
            else:
                body = "<!-- PAGE MISSING: extraction failed -->"
                missing_pages.append(i + 1)
            parts.append(f"\n\n---\n**Page {i + 1}**\n---\n\n{body}")

        duration = round(time.perf_counter() - t_doc, 1)
        if missing_pages:
            # Deliberately leave the merged file unwritten. Its presence is the
            # resume marker, so writing a hole-ridden version here would make the
            # next run skip the PDF and strand those pages forever. The sidecars
            # that did succeed are kept, so a re-run only retries the gaps.
            print(f"   ⚠️  {final_path.name}: {len(missing_pages)}/{total} page(s) missing — "
                  f"not merged; re-run to retry just those pages")
            return {"status": "partial", "pages_total": total,
                    "pages_done": total - len(missing_pages),
                    "pages_to_retry": missing_pages,
                    "last_error": last_error, "duration_s": duration}

        final_path.parent.mkdir(parents=True, exist_ok=True)
        final_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
        print(f"   ✅ {final_path}")
        return {"status": "ok", "pages_total": total, "pages_done": total,
                "pages_to_retry": [], "last_error": None, "duration_s": duration}


def _read_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ------------------------------------------------------------------ main ----


def get_pdf_files(folder: str, limit: Optional[int]) -> List[Path]:
    p = Path(folder)
    if not p.exists():
        raise SystemExit(f"Folder {folder} does not exist.")
    pdfs = sorted(p.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF files found in {folder}")
    return pdfs[:limit] if limit else pdfs


def sanitize_model_id(model_id: str) -> str:
    safe = model_id
    for ch in ['/', ':', '\\', ' ', '*', '?', '"', '<', '>', '|']:
        safe = safe.replace(ch, "_")
    return safe


UTC_FMT = "%Y-%m-%dT%H:%M:%SZ"

# A merged file's page separators, e.g. "**Page 17**" on its own line.
_PAGE_MARKER_RE = re.compile(r"(?m)^\*\*Page \d+\*\*$")


def _merged_page_count(path: Path) -> int:
    try:
        return len(_PAGE_MARKER_RE.findall(path.read_text(encoding="utf-8")))
    except OSError:
        return 0


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime(UTC_FMT)


class StatusBook:
    """Per-model ledger at <out_dir>/status.json.

    Disk is the source of truth for STATUS (merged file => ok, sidecars =>
    partial); reconcile() recomputes it every run, so deleting the file loses
    only history, never correctness. The book persists what disk cannot tell
    you: attempts, last error, failed pages, timings.
    """

    SCHEMA = 1
    PAGE_LIST_CAP = 50   # a doc where everything failed needs no 243-entry list

    def __init__(self, model_id: str, out_dir: Path):
        self.model_id = model_id
        self.out_dir = Path(out_dir)
        self.path = self.out_dir / "status.json"
        self.docs: Dict[str, Dict[str, Any]] = {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.docs = data.get("documents", {})
        except (OSError, json.JSONDecodeError):
            self.docs = {}

    def reconcile(self, pdfs: List[Path]) -> None:
        """Recompute every document's status from what is actually on disk."""
        for pdf in pdfs:
            stem = pdf.stem
            e = self.docs.setdefault(stem, {})
            merged = self.out_dir / f"{stem}.md"
            page_dir = self.out_dir / ".pages" / stem
            sidecars = len(list(page_dir.glob("page_*.md"))) if page_dir.exists() else 0
            if merged.exists() and merged.stat().st_size > 0:
                if not e.get("verified"):
                    # The current pipeline only merges complete documents, but
                    # files from the pre-ledger pipeline could merge with silent
                    # gaps. Verify once: page markers vs the PDF's true page
                    # count; incomplete files are quarantined and requeued.
                    pdf_pages = e.get("pdf_pages")
                    if pdf_pages is None:
                        try:
                            with pymupdf.open(pdf) as doc:
                                pdf_pages = doc.page_count
                        except Exception:
                            pdf_pages = None
                        if pdf_pages:
                            e["pdf_pages"] = pdf_pages
                    markers = _merged_page_count(merged)
                    e["pages_total"] = pdf_pages or markers or e.get("pages_total")
                    if pdf_pages and markers < pdf_pages:
                        stale_dir = self.out_dir / ".stale"
                        stale_dir.mkdir(exist_ok=True)
                        merged.rename(stale_dir / merged.name)
                        e["status"] = "partial" if sidecars else "pending"
                        e["pages_done"] = sidecars
                        e["last_error"] = (f"incomplete merge from old pipeline: "
                                           f"{markers}/{pdf_pages} pages — requeued")
                        print(f"   ♻️  {stem}: only {markers}/{pdf_pages} pages in merged file "
                              f"— requeued (old file kept in .stale/)")
                        continue
                    e["verified"] = True
                e["status"] = "ok"
                e["pages_done"] = e.get("pages_total") or sidecars or None
                e["pages_to_retry"] = []
                e["pages_to_retry_count"] = 0
                e.pop("last_error", None)
            elif e.get("status") == "failed":
                pass   # doc-level failure (e.g. unreadable PDF): disk can't see it
            elif sidecars:
                e["status"] = "partial"
                e["pages_done"] = sidecars
            else:
                e["status"] = "pending"
                e["pages_done"] = 0

    def should_process(self, stem: str, retry_exhausted: bool) -> Tuple[bool, str]:
        e = self.docs.get(stem, {})
        if e.get("status") == "ok":
            return False, "done"
        if not retry_exhausted and e.get("attempts", 0) >= ATTEMPT_CEILING:
            return False, "exhausted"
        return True, e.get("status", "pending")

    def record(self, stem: str, result: Dict[str, Any]) -> None:
        e = self.docs.setdefault(stem, {})
        pages = result.get("pages_to_retry") or []
        e["status"] = result["status"]
        if result.get("pages_total") is not None:
            e["pages_total"] = result["pages_total"]
        if result.get("pages_done") is not None:
            e["pages_done"] = result["pages_done"]
        e["pages_to_retry"] = pages[:self.PAGE_LIST_CAP]
        e["pages_to_retry_count"] = len(pages)
        if result.get("duration_s") is not None:
            e["duration_s"] = result["duration_s"]
        e["finished_at"] = _utcnow()
        e["attempts"] = e.get("attempts", 0) + 1
        if result.get("last_error"):
            e["last_error"] = str(result["last_error"])[:300]
        elif result["status"] == "ok":
            e.pop("last_error", None)

    def summary(self) -> Dict[str, int]:
        out = {"ok": 0, "partial": 0, "failed": 0, "pending": 0}
        for e in self.docs.values():
            st = e.get("status", "pending")
            out[st] = out.get(st, 0) + 1
        out["total"] = len(self.docs)
        return out

    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic_json(self.path, {
            "schema_version": self.SCHEMA,
            "model": self.model_id,
            "updated_at": _utcnow(),
            "summary": self.summary(),
            "documents": dict(sorted(self.docs.items())),
        })


async def main_async(models: Optional[List[str]] = None,
                     retry_exhausted: bool = False,
                     limit: Optional[int] = None) -> None:
    roster = models or MODEL_IDS
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    pdfs = get_pdf_files(FOLDER_PATH, limit if limit is not None else PDFS_LIMIT)
    print(f"📚 {len(pdfs)} PDFs | models: {', '.join(roster)}")
    print(f"   {PAGE_MEGAPIXELS} MP/page | {MAX_CONCURRENT} concurrent | "
          f"text-layer hint: {USE_TEXT_LAYER} | attempt ceiling: {ATTEMPT_CEILING}")

    loop = asyncio.get_running_loop()
    render_pool = concurrent.futures.ProcessPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 4) // 2)))
    io_pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(2, MAX_CONCURRENT * 2))
    endpoint = Endpoint(MAX_CONCURRENT)

    cache_dirs = {p: CACHE_ROOT / _fingerprint(p) for p in pdfs}
    render_errors: Dict[Path, str] = {}

    async def ensure_cache(pdf: Path) -> Optional[Path]:
        cdir = cache_dirs[pdf]
        if (cdir / "metadata.json").exists():
            return cdir
        try:
            await loop.run_in_executor(render_pool, render_pdf_worker, str(pdf), str(cdir))
            return cdir
        except Exception as e:
            render_errors[pdf] = str(e)[:300]
            print(f"⚠️  render failed for {pdf.name}: {e}")
            return None

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT * 2,
                                     limit_per_host=MAX_CONCURRENT * 2,
                                     ttl_dns_cache=300, keepalive_timeout=120)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=REQUEST_TIMEOUT,
                                    sock_connect=30)
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Models run one at a time: LM Studio evicts/reloads weights when asked
            # to serve several large VLMs at once, which costs far more than it saves.
            for model_id in roster:
                out_dir = OUTPUT_ROOT / sanitize_model_id(model_id)
                print(f"\n{'=' * 70}\n🤖 model: {model_id}\n{'=' * 70}")
                book = StatusBook(model_id, out_dir)
                book.reconcile(pdfs)

                work: List[Path] = []
                skipped_done = skipped_exhausted = 0
                for pdf in pdfs:
                    go, why = book.should_process(pdf.stem, retry_exhausted)
                    if go:
                        work.append(pdf)
                    elif why == "done":
                        skipped_done += 1
                    else:
                        skipped_exhausted += 1
                book.save()

                plan = f"   plan: {len(work)} to process | {skipped_done} already done"
                if skipped_exhausted:
                    plan += (f" | {skipped_exhausted} exhausted "
                             f"(≥{ATTEMPT_CEILING} attempts; use --retry-exhausted)")
                print(plan)
                if not work:
                    print(f"   ✅ nothing to do — see {book.path}")
                    continue

                endpoint.reset()
                app = VLMExtractor(LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, model_id, endpoint, io_pool)
                t_model = time.perf_counter()

                # Render RENDER_AHEAD PDFs ahead of the LLM stage so CPU rendering
                # and GPU inference overlap instead of alternating.
                pending: Dict[Path, asyncio.Task] = {}
                for pdf in work[:RENDER_AHEAD]:
                    pending[pdf] = asyncio.create_task(ensure_cache(pdf))

                aborted = False
                for i, pdf in enumerate(work, 1):
                    nxt = i - 1 + RENDER_AHEAD
                    if nxt < len(work) and work[nxt] not in pending:
                        pending[work[nxt]] = asyncio.create_task(ensure_cache(work[nxt]))
                    print(f"\n🚀 [{i}/{len(work)}] {pdf.name}")
                    cdir = await pending.pop(pdf)

                    if cdir is None:
                        book.record(pdf.stem, {
                            "status": "failed", "pages_to_retry": [],
                            "last_error": render_errors.get(pdf, "render failed"),
                            "duration_s": 0.0,
                        })
                    else:
                        try:
                            result = await app.process_pdf(session, pdf, cdir, out_dir)
                            book.record(pdf.stem, result)
                        except CircuitOpen as e:
                            book.record(pdf.stem, {
                                "status": "partial", "pages_to_retry": [],
                                "last_error": f"aborted mid-run: {str(e)[:200]}",
                            })
                            book.reconcile([pdf])   # true sidecar count from disk
                            print(f"\n🛑 aborting {model_id}: {e}")
                            aborted = True
                        except Exception as e:
                            book.record(pdf.stem, {
                                "status": "failed", "pages_to_retry": [],
                                "last_error": str(e)[:300],
                            })
                            print(f"❌ {pdf.name}: {e}")

                    book.save()   # after every document: a crash loses nothing
                    if aborted:
                        break

                for t in pending.values():
                    t.cancel()

                dt = time.perf_counter() - t_model
                st, bs = app.stats, book.summary()
                print(f"\n📊 {model_id}: {st['ok']} pages ok, {st['cached']} cached, "
                      f"{st['blank']} blank, {st['failed']} failed, {st['truncated']} truncated, "
                      f"{st['retries']} retries, {st['crashes']} backend crashes "
                      f"in {dt / 60:.1f} min")
                print(f"   docs: {bs['ok']} ok | {bs['partial']} partial | "
                      f"{bs['failed']} failed | {bs['pending']} pending  → {book.path}")
    finally:
        render_pool.shutdown(wait=True)
        io_pool.shutdown(wait=True)


def status_report(models: Optional[List[str]] = None,
                  limit: Optional[int] = None) -> None:
    """Print per-model progress from disk + status.json; runs nothing."""
    roster = models or MODEL_IDS
    pdfs = get_pdf_files(FOLDER_PATH, limit if limit is not None else PDFS_LIMIT)
    for model_id in roster:
        out_dir = OUTPUT_ROOT / sanitize_model_id(model_id)
        print(f"\n🤖 {model_id}")
        book = StatusBook(model_id, out_dir)
        book.reconcile(pdfs)
        book.save()
        smry = book.summary()
        print(f"   ok {smry['ok']} | partial {smry['partial']} | failed {smry['failed']} "
              f"| pending {smry['pending']} | total {smry['total']}   ({book.path})")
        problems = [(k, e) for k, e in sorted(book.docs.items())
                    if e.get("status") in ("partial", "failed")]
        for k, e in problems[:10]:
            pages = f"{e.get('pages_done', '?')}/{e.get('pages_total') or '?'} pages"
            retry = f", {e.get('pages_to_retry_count', 0)} to retry"
            err = f" | {e['last_error']}" if e.get("last_error") else ""
            print(f"   • {e['status']:7} {k}  [{e.get('attempts', 0)} attempt(s), {pages}{retry}]{err}")
        if len(problems) > 10:
            print(f"   … and {len(problems) - 10} more (see status.json)")


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        prog="vlm_pdf_to_markdown",
        description="PDF -> markdown extraction via local vision models, "
                    "with per-model status.json progress ledgers.",
        epilog="With no models given, the default roster runs sequentially:\n  "
               + "\n  ".join(DEFAULT_MODELS)
               + "\n\nEvery run resumes: finished documents are skipped, partial "
                 "documents retry\nonly their missing pages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("models", nargs="*",
                    help="model ids to run, in order (default: $VLM_MODELS or built-in roster)")
    ap.add_argument("--status", action="store_true",
                    help="print per-model progress and exit (runs nothing)")
    ap.add_argument("--limit", type=int, default=None, metavar="N",
                    help="process only the first N PDFs (smoke tests)")
    ap.add_argument("--retry-exhausted", action="store_true",
                    help=f"also retry documents with ≥{ATTEMPT_CEILING} recorded attempts")
    args = ap.parse_args(argv)

    models = args.models or None
    if args.status:
        status_report(models, args.limit)
        return
    asyncio.run(main_async(models, retry_exhausted=args.retry_exhausted, limit=args.limit))


if __name__ == "__main__":
    main()
