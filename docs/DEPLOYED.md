# Deployed releases

Append-only record of what is (and was) running at <https://fp-gcf.ssa.tg>.

Its job is to answer two questions during an incident, without SSH access:
**which commit is live right now**, and **which commit do I roll back to**.

## How a release gets on the host

`docker-compose.yaml` pins `image: fp-gcf:${GIT_SHA:-latest}`, so every deploy
builds a *new* tag rather than overwriting a single `:latest`. The previous
release therefore stays on disk and can be started again without a rebuild:

```sh
docker image ls fp-gcf                  # tags still on the host
GIT_SHA=<sha-from-the-table-below> docker compose up -d
docker compose ps                       # confirm fp-gcf is healthy again
```

Rolling back does **not** roll back `.env`. The switch columns below record
what the switches were at deploy time; if a rollback is meant to undo a switch
flip as well, edit `.env` to match that row before bringing the stack up.
Rolling back also does not roll back `data/` — the bind-mounted index, page
cache and SQLite db are shared by every release.

## Format

One row per deploy, newest last. `make deploy` **prints** the row after the
remote `docker compose up -d` returns successfully; the operator pastes it in
and commits it with the next change. Deploy does not write the row itself on
purpose: `push` refuses a dirty tree, so a target that edited a tracked file
mid-deploy would either trip its own interlock on the following run or force
an unreviewed auto-commit.

| Column | Meaning |
| --- | --- |
| `deployed (UTC)` | `date -u +%Y-%m-%dT%H:%M:%SZ` at the moment the deploy finished |
| `sha` | `git rev-parse --short=7 HEAD` — also the image tag (`fp-gcf:<sha>`) |
| `switches` | the `PLANNER` / `VERIFY` / `VERIFY_LLM` / `RERANK` lines of the `.env` that shipped with this deploy |

A row is a fact about the past. Never edit or reorder existing rows; a bad
deploy is corrected by adding the row for the rollback, not by deleting the
row for the release that broke.

From the next deploy on, the switches column no longer carries `VERIFY_REPAIR`:
`eac4c94` removed the repair code path, so nothing reads that variable and a
`VERIFY_REPAIR=0` still sitting in a deployed `.env` records a switch that no
longer exists — the rows above keep it because it was true when they were
written.

## Log

Empty until the first deploy made with sha-tagged images. Releases that predate
this file were all built as `fp-gcf:latest` and overwrote each other, so no
earlier rollback target exists on the host.

Nothing may follow the table — the `deploy` target appends with `>>`, so a row
landing after a trailing paragraph would detach from the table.

| deployed (UTC) | sha | switches |
| --- | --- | --- |
| 2026-08-20T17:36:34Z | 4a04d32 | PLANNER=1 VERIFY=1 VERIFY_REPAIR=0 |
| 2026-08-21T19:35:53Z | e639915 | PLANNER=1 VERIFY=1 VERIFY_REPAIR=0 |
| 2026-08-24T11:45:46Z | 22f558b | PLANNER=1 VERIFY=1 VERIFY_REPAIR=0 |
| 2026-08-27T18:43:40Z | 337f7c7 | PLANNER=1 VERIFY=1 RERANK=1 |
| 2026-08-27T19:38:40Z | 94cea40 | PLANNER=1 VERIFY=1 RERANK=1 |
