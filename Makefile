# GCF Q&A pipeline — one target per recurring operation.
#
#   make help                      list everything
#   make extract MODELS="pixtral-12b" LIMIT=3
#   make index SOURCE=data/extracted/vlm/qwen_qwen3-vl-8b NAME=qwen3
#   make preflight                 run the deploy interlock, touch nothing
#   make deployed-sha              what image production is actually running
#   make rollback SHA=<image tag>  put an earlier image back
#
# Overridable variables (VAR=value on the command line).
#
# `:=`, not `?=`: a command-line VAR=value still wins, but an exported shell
# variable no longer does. These are ordinary English words that tooling really
# does export — `NAME=Code` is set inside Claude Code, and under `?=` it made
# `make index` build data/index/Code instead of data/index/default, silently
# skipping the production warning below. Defaults now come from this file only.
#
# The comments sit on their own lines because `VAR := value  # note` puts the
# spaces before the `#` *inside* the value: NAME would be "default          ",
# which passes through the shell fine but fails a string compare like the
# NAME=default check in the index target below.
PY      := venv/bin/python
# MODELS: empty = the default extraction roster
MODELS  :=
# LIMIT: empty = the whole corpus
LIMIT   :=
# SOURCE: extraction dir to index
SOURCE  := data/extracted/vlm/qwen_qwen2.5-vl-7b
# NAME: index name under data/index/
NAME    := default
# DOC: doc stem for the grounding demo
DOC     := 152_gcf-b24-02-add09-rev01
# OUT: grounding-demo output dir
OUT     := /tmp/ground_demo

_limit  = $(if $(LIMIT),--limit $(LIMIT))

.DEFAULT_GOAL := help
.PHONY: help venv install test init-db extract status retry boxes index ground-demo chat docker-build docker-up \
        preflight push deploy rollback deployed-sha remote-images remote-restart remote-logs remote-down remote-shell \
        _check-clean-tree

help:                ## list available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z_-]+:.*##/ \
	  {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

test:                ## run the regression test suite
	$(PY) -m pytest tests/ -q

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

index:               ## build a FAISS index (SOURCE=<extraction dir> NAME=<index name> FORCE=1 to overwrite) — !! NAME=default writes the very artifact the next push ships to production
	@echo "index -> data/index/$(NAME)   (source: $(SOURCE))"
	@if [ "$(NAME)" = "default" ]; then \
	  echo "!! NAME=default -> data/index/$(NAME) is the index production loads (INDEX_NAME defaults to 'default')"; \
	  echo "!! and the artifact the next 'make push' rsyncs over the live one. Build under another"; \
	  echo "!! NAME and flip INDEX_NAME in .env unless you mean to replace production's index."; \
	  echo "!! (scripts/build_index.py:40 refuses to overwrite an existing index without FORCE=1.)"; \
	fi
	$(PY) scripts/build_index.py --source $(SOURCE) --name $(NAME) $(_limit) $(if $(FORCE),--force)

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
IMAGE ?= fp-gcf
# What deploy stamps onto the image it builds, so rollback has something to
# name. Meaningful only because push refuses a dirty tree: HEAD is the code.
# --short=7 matches docker-compose.yaml's `image: fp-gcf:${GIT_SHA:-latest}`
# and the sha column of docs/DEPLOYED.md — keep the three in step.
# `:=`, not `?=`: an exported GIT_SHA from the surrounding shell would otherwise
# name the image production runs and the tag rollback goes back to — the same
# environment footgun already fixed for NAME above, on the one variable that
# labels what is live. Override deliberately with `make deploy GIT_SHA=...`.
GIT_SHA := $(shell git rev-parse --short=7 HEAD 2>/dev/null || echo untagged)

# What ships: code + the data the app reads (index, raw pdfs for fingerprint
# lookup, page cache with geometry) + .env, which is how the operator flips
# PLANNER/VERIFY/VERIFY_LLM on production — .env MUST keep syncing.
# What NEVER ships: server-side user state (app.db, evidence-image copies, HF
# cache), regenerable/heavy leftovers, offline measurement output, and the
# half-written scratch that atomic writers leave behind mid-run.
#
# data/index is deliberately NOT excluded: production loads data/index/default.
# Note rsync runs without --delete, so an exclude added here stops a path from
# being *refreshed*; a copy already on the remote stays until removed by hand.
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
	--exclude '.stale' \
	--exclude 'data/eval' \
	--exclude 'data/canary' \
	--exclude '.pytest_cache' \
	--exclude '*.tmp' \
	--exclude '*.tmp.jpg' \
	--exclude '*.tmp[0-9]*' \
	--exclude '*.tmp_*' \
	--exclude '*.partial'

# --- Deploy interlock -------------------------------------------------------
# push rsyncs the WORKING TREE — not HEAD, not a build artifact. Whatever sits
# on disk at that moment is what production runs, so a speed bump guards it.
# It is overridable on purpose: it exists to make a risky push deliberate, not
# to make it impossible.
#
#   ALLOW_DIRTY=1   ship an uncommitted working tree anyway
#
# There was a second interlock here (FLIP=1), which refused to push an .env
# carrying VERIFY_REPAIR=1 because that flag let production rewrite answer
# text. eac4c94 removed the repair pathway and config.py no longer reads the
# variable, so the flag can no longer do anything: a guard over an .env line
# nothing reads teaches the operator that the line still matters. Removed with
# the code it guarded.

# Unreviewed code. `git diff` + `git diff --cached` is the exact dirty test;
# untracked files ship too, so they are reported as a note.
_check-clean-tree:
	@u=$$(git ls-files --others --exclude-standard | head -20); \
	  test -z "$$u" || { \
	    echo "NOTE: these untracked files are not ignored and would ship too:"; \
	    echo "$$u" | sed 's/^/      /'; }
	@git diff --quiet && git diff --cached --quiet || { \
	  if [ "$(ALLOW_DIRTY)" = "1" ]; then \
	    echo "WARNING: working tree is dirty and ALLOW_DIRTY=1 — shipping it as-is:"; \
	    git status --porcelain --untracked-files=no | sed 's/^/      /'; \
	  else \
	    echo "REFUSING TO PUSH: the working tree is dirty, and push rsyncs the working tree."; \
	    echo "  Uncommitted changes that would land on production:"; \
	    git status --porcelain --untracked-files=no | sed 's/^/      /'; \
	    echo "  Commit or stash them, or re-run with ALLOW_DIRTY=1 to ship them anyway."; \
	    exit 1; \
	  fi; }

preflight: _check-clean-tree ## run the push interlock locally; contacts nothing
	@echo "preflight OK — 'make push' would ship HEAD $(GIT_SHA) to $(REMOTE_HOST):$(REMOTE_DIR)"

push: _check-clean-tree ## rsync code + serving data to the remote (ALLOW_DIRTY=1 overrides the interlock)
	@test -n "$(REMOTE_HOST)" || { echo "REMOTE_HOST is not set"; exit 1; }
	rsync -avz --progress --chown=999:999 $(DEPLOY_EXCLUDES) ./ $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/

# deploy exports GIT_SHA before every compose call, so a docker-compose.yaml
# that interpolates it builds fp-gcf:<sha> directly; the explicit `docker tag`
# afterwards makes both fp-gcf:<sha> and fp-gcf:latest name that same build
# whichever way the compose file spells its image. That is what leaves rollback
# something to go back to.
deploy: push         ## push, rebuild the image on the server, tag it with the git SHA, start the stack
	ssh $(REMOTE_USER)@$(REMOTE_HOST) 'set -e; cd $(REMOTE_DIR) && \
	  export GIT_SHA=$(GIT_SHA) && \
	  mkdir -p data public/app_files hf_cache && \
	  chown -R 999:999 data public/app_files hf_cache && \
	  docker compose build $(COMPOSE_SERVICE) && \
	  built=$$(docker compose config --images $(COMPOSE_SERVICE) | head -1) && \
	  test -n "$$built" || { echo "cannot resolve the image compose just built"; exit 1; } && \
	  docker tag "$$built" $(IMAGE):$(GIT_SHA) && \
	  docker tag "$$built" $(IMAGE):latest && \
	  docker compose up -d && \
	  docker compose ps'
	@echo "deployed $(IMAGE):$(GIT_SHA) — undo with: make rollback SHA=<earlier tag>"
	@echo ""
	@echo "paste this row into docs/DEPLOYED.md (deploy does not write it: push"
	@echo "refuses a dirty tree, so a self-editing target would trip its own guard):"
	@printf '| %s | %s | %s |\n' "$$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(GIT_SHA)" \
	  "$$(grep -E '^(PLANNER|VERIFY|VERIFY_LLM)=' .env 2>/dev/null | tr '\n' ' ' | sed 's/ $$//')"

rollback:            ## put an earlier image back on the remote, no rebuild (SHA=<image tag>, required)
	@test -n "$(SHA)" || { \
	  echo "SHA is not set. Pick a tag with 'make remote-images', or 'make deployed-sha' for what runs now."; \
	  echo "Usage: make rollback SHA=<7-char git sha tagged by an earlier deploy>"; exit 1; }
	ssh $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_DIR) && \
	  export GIT_SHA=$(SHA) && \
	  docker image inspect $(IMAGE):$(SHA) >/dev/null 2>&1 || { \
	    echo "no image $(IMAGE):$(SHA) on the remote — available tags:"; \
	    docker images --format "      {{.Repository}}:{{.Tag}}  {{.CreatedSince}}" $(IMAGE); exit 1; }; \
	  want=$$(docker compose config --images $(COMPOSE_SERVICE) | head -1) && \
	  test -n "$$want" || { echo "cannot resolve the image compose expects"; exit 1; } && \
	  { [ "$$want" = "$(IMAGE):$(SHA)" ] || docker tag $(IMAGE):$(SHA) "$$want"; } && \
	  docker compose up -d --no-build --force-recreate $(COMPOSE_SERVICE) && \
	  docker compose ps && \
	  docker inspect --format "now running: {{.Config.Image}} ({{.Image}})" \
	    $$(docker compose ps -q $(COMPOSE_SERVICE))'
	@echo "rolled $(COMPOSE_SERVICE) back to $(IMAGE):$(SHA) — check the 'now running' line above."
	@echo "Code only: the remote .env (flags) and data/ (index, caches, app.db) are untouched,"
	@echo "and $(REMOTE_DIR) still holds the newer source — the next 'make deploy' rebuilds from it."

deployed-sha:        ## print the image tag + id the remote container is actually running
	@ssh $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_DIR) && \
	  cid=$$(docker compose ps -q $(COMPOSE_SERVICE)); \
	  test -n "$$cid" || { echo "$(COMPOSE_SERVICE) is not running on $(REMOTE_HOST)"; exit 1; }; \
	  docker inspect --format "container: {{.Name}}  started: {{.State.StartedAt}}" $$cid; \
	  docker inspect --format "image:     {{.Config.Image}}  ({{.Image}})" $$cid; \
	  docker image inspect --format "tags:      {{.RepoTags}}" $$(docker inspect --format "{{.Image}}" $$cid)'

remote-images:       ## list the image tags on the remote that rollback can name
	@ssh $(REMOTE_USER)@$(REMOTE_HOST) \
	  'docker images --format "      {{.Repository}}:{{.Tag}}  {{.CreatedSince}}  {{.Size}}" $(IMAGE)'

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
