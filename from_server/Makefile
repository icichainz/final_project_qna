# Deploy target — override these values via the environment, for example:
#   REMOTE_HOST=1.2.3.4 REMOTE_USER=deploy make push
REMOTE_USER ?= root
REMOTE_HOST ?= 38.242.231.130
REMOTE_DIR  ?= /workspace/pimento
LOCAL_DIR   := ./
COMPOSE_SERVICE ?= backend

EXCLUDES := \
	--exclude '.git' \
	--exclude '__pycache__' \
	--exclude '*.pyc' \
	--exclude '.venv' \
	--exclude '.pytest_cache' \
	--exclude '.DS_Store' \
	--exclude '.dev_session_secret' \
	--exclude 'static/uploads'

.PHONY: push deploy restart logs down ssh
push:
	@test -n "$(REMOTE_HOST)" || { echo "REMOTE_HOST is not set. Usage: REMOTE_HOST=<host> [REMOTE_USER=<user>] [REMOTE_DIR=<path>] make push"; exit 1; }
	rsync -avz --progress $(EXCLUDES) $(LOCAL_DIR) $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/

# Push the code, rebuild the backend image without cache, and start the stack.
# Use after changing dependencies, the Dockerfile, or an entrypoint.
deploy: push
	ssh $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_DIR) && \
	  docker compose build --no-cache $(COMPOSE_SERVICE) && \
	  docker compose up -d && \
	  docker compose ps'

# Push the code and restart only the backend service.
# Use for Python or template changes that do not alter the image.
restart: push
	ssh $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_DIR) && \
	  docker compose restart $(COMPOSE_SERVICE)'

# Tail logs from the remote backend service.
logs:
	ssh -t $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_DIR) && \
	  docker compose logs -f --tail=200 $(COMPOSE_SERVICE)'

# Stop the remote Compose stack.
down:
	ssh $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_DIR) && \
	  docker compose down'

# Open a shell in the running remote backend container.
ssh:
	ssh -t $(REMOTE_USER)@$(REMOTE_HOST) 'cd $(REMOTE_DIR) && \
	  docker compose exec $(COMPOSE_SERVICE) sh'

# --- Local Docker Compose ---------------------------------------------------
# These targets expect a compose.yaml, compose.yml, docker-compose.yaml, or
# docker-compose.yml file in this backend directory.
.PHONY: build up watch local-down local-logs
build:
	docker compose build

up:
	docker compose up -d --build

watch:
	docker compose up -d --build
	docker compose watch

local-down:
	docker compose down

local-logs:
	docker compose logs -f

# --- Database migrations (Alembic) ------------------------------------------
# Runtime/production schema is owned by Alembic migrations, NOT create_all.
# These run from this backend directory. Target a specific database by exporting
# DATABASE_URL, e.g. `DATABASE_URL=sqlite+aiosqlite:///./db/pimento.db make db-upgrade`
# (alembic/env.py reads DATABASE_URL and maps the async driver to a sync one).
# Author a migration with:  make db-revision m="describe the change"
ALEMBIC     ?= python3 -m alembic
DEV_DB_PATH ?= db/pimento.db

.PHONY: db-upgrade db-downgrade db-current db-history db-revision db-stamp db-reset
db-upgrade:            ## Apply all migrations up to head.
	$(ALEMBIC) upgrade head

db-downgrade:          ## Roll back exactly one migration.
	$(ALEMBIC) downgrade -1

db-current:            ## Show the revision the database is stamped at.
	$(ALEMBIC) current

db-history:            ## Show the migration history.
	$(ALEMBIC) history

db-revision:           ## Autogenerate a migration: make db-revision m="message"
	@test -n "$(m)" || { echo 'Usage: make db-revision m="describe the change"'; exit 1; }
	$(ALEMBIC) revision --autogenerate -m "$(m)"

db-stamp:              ## Stamp the database at head without running migrations.
	$(ALEMBIC) stamp head

# Dev only: delete the local sqlite dev database, then rebuild it from
# migrations. Only ever removes DEV_DB_PATH (default db/pimento.db) and its
# sqlite sidecar files, and pins the upgrade to that same dev database.
db-reset:              ## DEV ONLY: recreate the local sqlite dev DB from migrations.
	rm -f "$(DEV_DB_PATH)" "$(DEV_DB_PATH)-shm" "$(DEV_DB_PATH)-wal"
	DATABASE_URL="sqlite+aiosqlite:///./$(DEV_DB_PATH)" $(ALEMBIC) upgrade head
