# GCF Q&A pipeline — one target per recurring operation.
#
#   make help                      list everything
#   make extract MODELS="pixtral-12b" LIMIT=3
#   make index SOURCE=data/extracted/vlm/qwen_qwen3-vl-8b NAME=qwen3
#
# Overridable variables (VAR=value on the command line):
PY      := venv/bin/python
MODELS  ?=                                             # empty = default roster
LIMIT   ?=                                             # empty = whole corpus
SOURCE  ?= data/extracted/vlm/qwen_qwen2.5-vl-7b       # extraction dir to index
NAME    ?= default                                     # index name under data/index/
DOC     ?= 152_gcf-b24-02-add09-rev01                  # doc stem for the demo
OUT     ?= /tmp/ground_demo                            # demo output dir

_limit  = $(if $(LIMIT),--limit $(LIMIT))

.DEFAULT_GOAL := help
.PHONY: help venv install extract status retry boxes index ground-demo chat docker-build docker-up

help:                ## list available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ \
	  {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

venv:                ## create the virtualenv
	python3 -m venv venv

install:             ## install the package + extraction + app stacks into venv
	venv/bin/pip install -e ".[extraction,app]"

extract:             ## run VLM extraction (MODELS="a b" LIMIT=N; resumes automatically)
	$(PY) vlm_pdf_to_markdown.py $(MODELS) $(_limit)

status:              ## per-model extraction progress from status.json (runs nothing)
	$(PY) vlm_pdf_to_markdown.py --status $(MODELS)

retry:               ## extraction including docs that exhausted their attempt ceiling
	$(PY) vlm_pdf_to_markdown.py --retry-exhausted $(MODELS) $(_limit)

boxes:               ## backfill page-geometry sidecars for cached pages (LIMIT=N dirs)
	$(PY) scripts/backfill_boxes.py $(_limit)

index:               ## build a FAISS index (SOURCE=<extraction dir> NAME=<index name>)
	$(PY) scripts/build_index.py --source $(SOURCE) --name $(NAME) $(_limit)

ground-demo:         ## draw citation highlights on real pages (DOC=<stem> OUT=<dir>)
	$(PY) scripts/demo_grounding.py --doc $(DOC) --out $(OUT)

chat:                ## run the Chainlit app (reads .env; INDEX_NAME picks the index)
	@if [ -f .env ]; then set -a; . ./.env; set +a; fi; \
	venv/bin/chainlit run src/gcf_qna/app/chainlit_app.py

docker-build:        ## build the app image
	docker compose build

docker-up:           ## run the app in Docker (expects .env + a built index in data/)
	docker compose up
