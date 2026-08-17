#!/usr/bin/env python3
"""Run the VLM PDF -> markdown extraction over the corpus.

Thin wrapper over gcf_qna.extraction.vlm — all settings are env vars
(see that module's docstring). Examples:
  python scripts/extract_corpus.py
  VLM_MODELS=qwen/qwen3-vl-8b PDFS_LIMIT=2 python scripts/extract_corpus.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gcf_qna.extraction.vlm import main

if __name__ == "__main__":
    main()
