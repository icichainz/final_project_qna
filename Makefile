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
.PHONY: help venv install init-db extract status retry boxes index ground-demo chat docker-build docker-up push deploy remote-restart remote-logs remote-down remote-shell

help:                ## list available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ \
	  {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

init-db:             ## create/upgrade the conversation-history database
	$(PY) scripts/init_appdb.py

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
	@touch .env
	@grep -q '^CHAINLIT_AUTH_SECRET=..*' .env || { \
	  echo "CHAINLIT_AUTH_SECRET=$$(openssl rand -hex 32)" >> .env; \
	  echo "🔑 generated CHAINLIT_AUTH_SECRET into .env"; }
	@grep -q '^APP_USERS=..*' .env || \
	  echo "⚠️  APP_USERS is not set in .env — nobody will be able to log in. Add e.g.: APP_USERS=demo:choose-a-password"
	@set -a; . ./.env; set +a; \
	venv/bin/chainlit run src/gcf_qna/app/chainlit_app.py --headless --host 0.0.0.0 --port 8000

docker-build:        ## build the app image
	docker compose build

docker-up:           ## run the app in Docker (expects .env + a built index in data/)
	docker compose up

# --- Remote deployment (mirrors the from_server/ workflow) -------------------
# Override via environment:  REMOTE_HOST=1.2.3.4 make push
REMOTE_USER ?= root
REMOTE_HOST ?= 38.242.231.130
REMOTE_DIR  ?= /workspace/fp_gcf
COMPOSE_SERVICE ?= fp-gcf

# What ships: code + the data the app reads (index, raw pdfs for fingerprint
# lookup, page cache with geometry). What NEVER ships: server-side user state
# (app.db, evidence-image copies, HF cache) and regenerable/heavy leftovers.
DEPLOY_EXCLUDES := \
	--exclude '.git' \
	--exclude '__pycache__' \
	--exclude '*.pyc' \
	--exclude 'venv' \
	--exclude 'from_server' \
	--exclude 'data/app.db*' \
	--exclude 'public/app_files' \
	--exclude 'hf_cache' \
	--exclude 'data/cache/image_cache_legacy' \
	--exclude 'data/cache/highlights' \
	--exclude 'data/extracted' \
	--exclude 'data/index/legacy' \
	--exclude 'src/gcf_qna.egg-info' \
	--exclude '.stale'

.PHONY: push deploy remote-restart remote-logs remote-down remote-shell
push:                ## rsync code + serving data to the remote (first run syncs ~20 GB of page cache)
	@test -n "$(REMOTE_HOST)" || { echo "REMOTE_HOST is not set"; exit 1; }
	rsync -avz --progress $(DEPLOY_EXCLUDES) ./ $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/

deploy: push         ## push, rebuild the image on the server, start the stack
	ssh $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_DIR) && \
	  mkdir -p data public/app_files hf_cache && \
	  docker compose build $(COMPOSE_SERVICE) && \
	  docker compose up -d && \
	  docker compose ps'

remote-restart: push ## push code and restart the service (no rebuild)
	ssh $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_DIR) && \
	  docker compose restart $(COMPOSE_SERVICE)'

remote-logs:         ## tail the remote service logs
	ssh -t $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_DIR) && \
	  docker compose logs -f --tail=200 $(COMPOSE_SERVICE)'

remote-down:         ## stop the remote stack
	ssh $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_DIR) && \
	  docker compose down'

remote-shell:        ## shell into the running remote container
	ssh -t $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_DIR) && \
	  docker compose exec $(COMPOSE_SERVICE) sh'
