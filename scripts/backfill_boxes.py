#!/usr/bin/env python3
"""Backfill page_NNNN.boxes.json sidecars for already-rendered page caches.

The render pass now writes boxes (text lines / tables / figures, in pixel
coords of the cached JPEG) alongside each page image; this fills them in for
caches rendered before the feature existed. Text extraction only — pages are
never re-rendered.

  python scripts/backfill_boxes.py            # all cached PDFs
  python scripts/backfill_boxes.py --limit 3  # first N cache dirs (smoke test)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pymupdf

from gcf_qna import config
from gcf_qna.extraction.vlm import _write_atomic_json, extract_page_boxes


def backfill(cache_dir: Path) -> str:
    meta_path = cache_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    entries = meta.get("pages", [])
    todo = [e for e in entries
            if e.get("img") and not (e.get("boxes") and (cache_dir / e["boxes"]).exists())]
    if not todo:
        return "complete"
    pdf = config.RAW_PDF_DIR / meta.get("pdf_name", "")
    if not pdf.exists():
        return f"skipped (no such pdf: {meta.get('pdf_name')})"
    with pymupdf.open(pdf) as doc:
        for e in todo:
            n = e["n"]
            if n > doc.page_count:
                continue
            page = doc[n - 1]
            zoom = e["w"] / page.rect.width if page.rect.width else 1.0
            boxes = extract_page_boxes(page, zoom, e["w"], e["h"])
            bname = f"page_{n:04d}.boxes.json"
            (cache_dir / bname).write_text(
                json.dumps(boxes, ensure_ascii=False), encoding="utf-8")
            e["boxes"] = bname
    _write_atomic_json(meta_path, meta)
    return f"{len(todo)} page(s) backfilled"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None, help="only the first N cache dirs")
    a = ap.parse_args()

    dirs = sorted(d for d in Path(config.PAGE_CACHE_DIR).iterdir()
                  if (d / "metadata.json").exists())
    if a.limit:
        dirs = dirs[:a.limit]
    t0 = time.time()
    done = 0
    for i, d in enumerate(dirs, 1):
        try:
            msg = backfill(d)
        except Exception as e:
            msg = f"ERROR: {e}"
        if msg != "complete":
            print(f"[{i}/{len(dirs)}] {d.name[:16]}… {msg}")
        done += 1
    print(f"\n{done} cache dir(s) processed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
