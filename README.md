# GCF Funding-Proposal Q&A

Question-answering over Green Climate Fund funding proposals: PDFs are
extracted to markdown by a local vision-language model, chunked and indexed
with FAISS, and served through a Chainlit chat UI backed by an
OpenAI-compatible LLM endpoint.

## Pipeline

```
acquire            extract                index                 serve
GCF scraping  ->   PDF -> markdown   ->   chunk/embed/FAISS ->  Chainlit chat
notebooks/         scripts/               scripts/              chainlit run
GCF_FUND_DATA      extract_corpus.py      build_index.py        src/gcf_qna/app/chainlit_app.py
```

## Layout

- `src/gcf_qna/` — the package: `config.py` (all paths/tunables, env-overridable),
  `extraction/vlm.py`, `rag/` (parse → chunk → embed → index → retrieve),
  `app/chainlit_app.py` (the **single** UI entrypoint)
- `scripts/` — CLIs over the package
- `data/` — corpus, caches, extraction outputs, indexes (gitignored; see
  `data/README.md` for the regeneration contract)
- `notebooks/` — data acquisition + extraction-method experiments
  (`notebooks/archive/` holds the per-LLM forks, frozen as a record)
- `archive/` — pre-re-architecture app variants, kept for reference
- `docs/` — project reports

## Quickstart

```bash
python -m venv venv && venv/bin/pip install -e ".[extraction,app]"
cp .env.example .env          # fill in OPENAI_API_KEY etc.

# 1. extract (needs LM Studio serving a vision model)
venv/bin/python scripts/extract_corpus.py

# 2. index
venv/bin/python scripts/build_index.py \
    --source data/extracted/vlm/qwen_qwen3-vl-8b --name default

# 3. chat
venv/bin/chainlit run src/gcf_qna/app/chainlit_app.py
```

Docker: `docker compose up --build` (expects `.env` and a built index in `data/index/`).

## Conventions

- **No keys in source, ever.** Config comes from the environment / `.env`.
  History prior to the re-architecture leaked keys and lives only on the
  GitHub remote; treat those keys as burned.
- **Vary by config, not by copy.** The old `main2/3/4` forks are in
  `archive/` as a cautionary tale — anything you'd change by copying a file
  belongs in `src/gcf_qna/config.py`, a flag, or an env var.
- Each `data/` layer is rebuildable by one command (`data/README.md`).
- Extraction throughput/accuracy notes: `docs/` and the VLM module docstring.
  On LM Studio keep `MAX_CONCURRENT=1` — two concurrent multimodal requests
  crash the model process (measured 2026-08-14).
