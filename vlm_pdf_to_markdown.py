#!/usr/bin/env python3
"""Compatibility entrypoint — the pipeline itself lives in
src/gcf_qna/extraction/vlm.py. Usage:

    python vlm_pdf_to_markdown.py                       # default roster, sequentially
    python vlm_pdf_to_markdown.py pixtral-12b           # just this model
    python vlm_pdf_to_markdown.py m1 m2                 # these, in order
    python vlm_pdf_to_markdown.py --status              # progress report, runs nothing
    python vlm_pdf_to_markdown.py --limit 3 <model>     # smoke test
    python vlm_pdf_to_markdown.py -h                    # everything else

Each model keeps a ledger at data/extracted/vlm/<model>/status.json:
which documents finished, which need retry (and which pages), attempts,
and last errors. Runs always resume from that state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from gcf_qna.extraction.vlm import main

if __name__ == "__main__":
    main()
