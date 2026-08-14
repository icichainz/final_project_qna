#!/usr/bin/env python3
"""Compatibility entrypoint — the pipeline itself lives in
src/gcf_qna/extraction/vlm.py; this keeps the original invocation working:

    python vlm_pdf_to_markdown.py

Same env-var controls as always (VLM_MODELS, PDFS_LIMIT, MAX_CONCURRENT, ...);
paths default to data/raw/pdfs -> data/cache/pages -> data/extracted/vlm/.
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gcf_qna.extraction.vlm import main_async

if __name__ == "__main__":
    asyncio.run(main_async())
