#!/usr/bin/env python3
"""Visual proof for citation grounding.

Takes real extracted pages, treats each page's markdown as a retrieved chunk,
grounds it back onto the cached page image, and writes JPEGs with the
highlight rects drawn on:  green = matched text lines, blue = table regions,
orange = detected figures (context). No model calls; PIL + the boxes sidecars.

  python scripts/demo_grounding.py --doc 152_gcf-b24-02-add09-rev01 --out /tmp/demo
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw

from gcf_qna import config
from gcf_qna.rag import split_pages
from gcf_qna.rag.ground import cache_dir_for, ground_chunk, load_page_assets

GREEN, BLUE, ORANGE = (26, 158, 90), (36, 98, 199), (233, 126, 22)


def draw(image_path: Path, grounding, figures, out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    dr = ImageDraw.Draw(img, "RGBA")
    for bbox in figures:
        dr.rectangle(bbox, outline=ORANGE, width=3)
    color = BLUE if grounding.kind == "table" else GREEN
    for bbox in grounding.rects:
        dr.rectangle(bbox, outline=color, width=3,
                     fill=(color[0], color[1], color[2], 40))
    img.save(out_path, quality=90)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", required=True, help="doc stem, e.g. 152_gcf-b24-02-add09-rev01")
    ap.add_argument("--source", type=Path,
                    default=config.EXTRACTED_DIR / "vlm" / "qwen_qwen2.5-vl-7b")
    ap.add_argument("--pages", type=int, nargs="*", default=None,
                    help="specific page numbers (default: first 3 + first table page)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    md = (a.source / f"{a.doc}.md").read_text(encoding="utf-8")
    pages = dict(split_pages(md))
    cdir = cache_dir_for(f"{a.doc}.pdf")
    if cdir is None:
        raise SystemExit(f"no page cache for {a.doc}.pdf")

    wanted = a.pages
    if not wanted:
        wanted = sorted(pages)[:3]
        # add the first page whose boxes contain a table
        for n in sorted(pages):
            assets = load_page_assets(a.doc, n, cdir)
            if assets and assets.get("boxes") and assets["boxes"]["tables"]:
                if n not in wanted:
                    wanted.append(n)
                break

    a.out.mkdir(parents=True, exist_ok=True)
    for n in wanted:
        if n not in pages:
            print(f"p{n}: not in extraction, skipped")
            continue
        chunk = {"doc_id": a.doc, "page": n, "text": pages[n]}
        g = ground_chunk(chunk, cache_dir=cdir)
        assets = load_page_assets(a.doc, n, cdir)
        if g is None or assets is None or assets["image"] is None:
            print(f"p{n}: no assets"); continue
        figures = [f["bbox"] for f in (assets.get("boxes") or {}).get("figures", [])]
        out = a.out / f"{a.doc}_p{n:03d}_{g.kind}_{g.confidence:.2f}.jpg"
        draw(assets["image"], g, figures, out)
        print(f"p{n}: kind={g.kind:5} confidence={g.confidence:.2f} "
              f"rects={len(g.rects)} figures={len(figures)} -> {out.name}")


if __name__ == "__main__":
    main()
