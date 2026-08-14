# data/ — regeneration contract

Everything here except `raw/` is reproducible by a named command. Nothing in
this directory is tracked by git (except this file and `raw/funding/*.csv`).

| layer | contents | regenerate with |
|---|---|---|
| `raw/pdfs/` | 273 GCF funding-proposal PDFs | **irreplaceable input** — back it up |
| `raw/funding/` | scraped funding CSVs (tracked in git) | `notebooks/GCF_FUND_DATA.ipynb` |
| `cache/pages/` | per-page JPEG + text-layer cache | auto-rebuilt by extraction (~3 s/PDF) |
| `cache/image_cache_legacy/` | old 250-DPI PNG cache (~58 GB) | superseded — safe to delete |
| `extracted/<method>/` | markdown per extraction method | `python scripts/extract_corpus.py` (vlm); notebooks for the rest |
| `index/<name>/` | FAISS index + chunks.jsonl + config.json | `python scripts/build_index.py --source ... --name <name>` |
| `index/legacy/` | old pickle metadata (meta.pkl) | superseded — rebuild instead |
