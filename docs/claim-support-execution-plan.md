# Claim Support and Rollout Plan — Execution Plan (Waves −1–5)

**GCF Q&A | `/home/ssa/Workspace/final_project_qna` | branch `feat/claim-support-rollout`, tip `f204754`**

The first full-suite deterministic measurement (`data/eval/release_release-1.jsonl`, 66 cases, Aug 19) records **94 supported / 165 claims = 57.0% deterministic claim support against a >= 95% release gate**, with 71 failures (23 CONTRADICTED + 48 UNSUPPORTED; kinds money 34 / entity 29 / year 5 / existence 2 / number 1), required-field coverage 54/56, doc recall@5 30/30 (gold-30), answer recall@5 96%, evidence-page hit 94%, p50 3.4 s / p95 8.3 s, ~$0.33 (168,340 prompt + 11,478 completion tokens, alias `gpt-5.2`, snapshot unrecorded). **Groundedness was never measured at release-1** — the record's `claims` block contains only `{claims, supported, contradicted, unsupported, support_rate, evidence_keys, failures}` and its `hits` carry `{doc,page,score}` with no text, so no groundedness number exists and none can be derived by hand. F7 reconstructs it or declares release-1 un-replayable.

Production runs `PLANNER=1, VERIFY=1, VERIFY_REPAIR=0` with code defaults `CONDUCTOR=1` (`config.py:36`) and `VERIFY_LLM=1` (`config.py:55`); the `verify.py` repair layer is byte-identical to `f204754`. A **1,008-line uncommitted, unreviewed Codex wave** (`docs/claim-support-rollout-plan.md` +251, `scripts/eval_answers.py` +342, `src/gcf_qna/rag/verify.py` +235, `tests/test_eval_answers.py` +165, `tests/test_verify.py` +130, plus untracked `scripts/adjudicate_claims.py` 391 lines and `tests/test_adjudicate_claims.py` 202 lines) sits in the working tree and already contains matcher relaxations the plan it ships with forbids writing before adjudication.

**`make push`, `make deploy` and `make remote-restart` all rsync the working tree, not a commit** (`Makefile:96-100`), and `Dockerfile` does `COPY src/ src/`. Those 235 unreviewed `verify.py` lines are therefore deployable today, on the live `verify.verify_answer` path at `VERIFY=1`. Wave −1 exists to close that before any other work starts.

This document sequences the work so that nothing lands unreviewed, no matcher change is justified by inference, no measurement change is booked as an improvement, and no eval activity can reach production by accident.

---

## Metric contract

Three numbers, two gates. **Every claim metric is published as `n/d`, never as a bare rate** — in gate output, in the dashboard, and in the release report. A rate whose denominator moved is not a comparison.

| Name | Definition | Code | Gate |
|---|---|---|---|
| **Groundedness** | Some evidence held by the answer path entails the claim, whether or not the answer cites it | `_verify_against` run over **all** held evidence for every claim, cited or not (scoping change landed in 0b) | **>= 95% as `n/d`** (Wave 4, post-repair) |
| **Citation completeness** | The claim cites evidence that entails it | cited **AND** SUPPORTED, named `citation_completeness_rate` | **>= 95% as `n/d`** (Wave 4, post-repair) |
| Citation presence | Claim carries any citation, correct or not | `citation_presence_rate` (`bool(claim.citations)`) | reported, never gated |

Binding decisions:

1. **Scoped identity.** In `--verifier-mode deterministic` only, uncited claims are always UNSUPPORTED (`verify.py@HEAD` `classify_deterministic`), so `citation_completeness == supported == legacy support_rate`. **This identity does not hold in production mode**: `_judge` (`verify.py@HEAD` `_judge`, WT `:1636`) selects `status == UNSUPPORTED and plausible` and can only promote, never demote — a one-way ratchet that raises support with no verifier improvement. Every production-mode record therefore prints `supported`, `citation_supported` and `grounded` as **three separate `n/d`**, and no release-1 → release-2 comparison may cross modes.
2. The duplicate flag `grounded-without-citation` (emitted under the identical condition as `value-present-elsewhere`, consumed only by `tests/test_verify.py:382`) is **deleted**. One flag, one meaning.
3. **The groundedness definition is fixed in 0b, before any baseline is taken.** At HEAD, `value-present-elsewhere` is set only on the uncited branch, so a cited-but-wrong-page claim fails *both* metrics — precisely the miscitation case the split exists to isolate, and the split would carry no independent information (groundedness >= citation completeness by construction). Rescoping `_verify_against` to all held evidence is pure-python, relaxes no matcher, and creates no gaming surface. Baselining once under the correct definition beats re-baselining mid-plan.
4. `citation_support_rate` is **deleted in the same commit as the rename** — leaving both would give `compare_verifier_output` two identical pairs, double-counting one regression and showing four green deltas where three exist. 0b adds a startup assertion that the four compared metric keys are distinct.
5. **Gates are counts, not rounded percentages.** Where the release-1 artifact records only a rounded rate (answer recall@5, evidence-page hit), 0b recomputes the exact count from the record and the gate is fixed at that count. The `>= 95%` claim gates print `n/d` plus the exact integer threshold `ceil(0.95 * d)`.
6. The `>= 95%` threshold is a **release** gate only. Wave 2 gates on precision/recall/false-negatives vs adjudicated gold.
7. **Denominator discipline.** `extract_claims` decides the claim population and repair rewrites answers, so deleting an unsupportable sentence raises groundedness with zero new support: 63 deleted unsupported claims produce the same headline as 63 fixed ones. Every gate from Wave 2 on therefore additionally requires: absolute `supported` count non-decreasing; `claims` total within a stated band of the parity baseline with each excursion itemized per case; and **every claim present before and absent after listed with its status and a written justification** — not only the supported ones.

---

## Wave −1 — Production interlocks and the hold branch (blocking, single owner A0)

No review, labelling or measurement work begins until this wave's gate passes. A0 holds exclusive repo-wide write for its duration.

**1. Stop the tree from shipping.** In `Makefile:81-95`, add to `DEPLOY_EXCLUDES`: `--exclude '.env'`, `--exclude 'data/index'`, `--exclude 'data/registry*.json'`, `--exclude 'data/eval'`, `--exclude 'data/canary'`. `data/index/default` is the live 750 MB retrieval artifact and `registry*.json` is read by both `planner.py` and `verify.py` (the `registry_backed` rescue Wave 2 depends on); today any local `make index` (NAME defaults to `default`) silently replaces production's index on the next push, including via `make remote-restart`. Add `make push-artifacts` for the rare intentional index/registry ship.

**2. Guard `push`.** It aborts unless `git diff --quiet && git diff --cached --quiet` and `git rev-parse HEAD` matches a `deployed-*` tag; override only via `ALLOW_DIRTY=1`. It ssh-greps the remote `.env` for `APP_USERS`, `CHAINLIT_AUTH_SECRET`, `FP_GCF_DOMAIN`, `VERIFY_REPAIR` and aborts if any is missing. It greps the **local** `.env` for `VERIFY_REPAIR=1` and aborts unless `FLIP=1` is passed. Rationale: the local `.env` (13 keys) is missing five the deployed stack needs (`APP_USERS`, `ALLOW_SIGNUP`, `FP_GCF_DOMAIN`, `CONDUCTOR`, `VERIFY_LLM`) and `docker-compose.yaml` declares `env_file: .env`; the current rsync overwrites it, breaking static logins and, if `CHAINLIT_AUTH_SECRET` differs, invalidating every live session cookie at once. The remote `.env` is managed out-of-band over ssh from here on.

**3. Rollback path.** `docker-compose.yaml` pins `image: fp-gcf:latest` and `container_name: fp-gcf`; every deploy overwrites `latest` with no retained predecessor. Change to `image: fp-gcf:${GIT_SHA:-latest}` plus a `latest` alias, keep the last three tags on the server, record the deployed sha in a tracked file on every deploy, and add `make rollback SHA=<sha>` (`docker compose up -d --no-build`). **Rehearse it once and record the elapsed time as a gate artifact.**

**4. Protect the adjudication anchor.** `eval_answers.py` `record()` does `out.write_text(...)` unconditionally — one mistyped `--record release-1` destroys the anchor, and `.gitignore`'s `data/**` means `git clean -xfd` does too. Do now: add `!data/eval/` and `!data/eval/release_*.jsonl` to `.gitignore`; `git add -f data/eval/release_release-1.jsonl && git commit`; record its sha256 in a tracked file **in that commit**; patch `record()` to refuse an existing path without `--force-record`, with a unit test for the refusal.

**5. Move the held wave off the working tree — atomic, ordered, no other agent running.**

```bash
set -euo pipefail
cd /home/ssa/Workspace/final_project_qna
test "$(git rev-parse --abbrev-ref HEAD)" = feat/claim-support-rollout
test "$(git rev-parse HEAD)" = f204754177c839a2a839124712489fe61d77f34d
git stash push -u -- src scripts tests docs
git switch -c hold/verifier-calibration
git stash apply
git add -A && git commit -m 'verify: matcher calibration (HELD)'
git rev-parse --verify hold/verifier-calibration >/dev/null
test "$(git diff --stat main...hold/verifier-calibration | wc -l)" -gt 0
git switch feat/claim-support-rollout
git stash pop                                   # restores 0a/0b/0d paths only
git checkout HEAD -- src/gcf_qna/rag/verify.py tests/test_verify.py
git diff --quiet HEAD -- src/gcf_qna/rag/verify.py
```

The revert is the destructive step; the hold branch must exist and must **differ from HEAD on `verify.py`** before it runs. An emptiness check alone passes identically if the 235 lines were simply deleted.

**6. Capture the deployed configuration fingerprint** (gate artifact, committed next to the release records):

```bash
ssh root@38.242.231.130 'sed -E "s/=.*/=SET/" /workspace/fp_gcf/.env' > docs/deployed-env-fingerprint.txt
ssh root@38.242.231.130 'docker inspect -f {{.Config.Image}} fp-gcf; cd /workspace/fp_gcf && \
  sha256sum data/index/default/config.json data/index/default/index.faiss data/registry_v2.json' \
  >> docs/deployed-env-fingerprint.txt
sha256sum data/index/default/config.json data/index/default/index.faiss data/registry_v2.json \
  >> docs/deployed-env-fingerprint.txt
```

**7. Key rotation, in this order** (it depends on the deploy mechanism, so it is not wave-independent): (a) `.env` exclusion from step 1 landed; (b) rotate `OPENAI_API_KEY` on the server via ssh; (c) rotate locally; (d) one live smoke turn against `fp-gcf.ssa.tg` returning an answer with citations.

**Gate −1:**

```bash
set -euo pipefail
grep -q "exclude '.env'" Makefile && grep -q "exclude 'data/index'" Makefile
grep -q "exclude 'data/registry\*.json'" Makefile && grep -q "exclude 'data/eval'" Makefile
grep -q '!data/eval/' .gitignore
git ls-files --error-unmatch data/eval/release_release-1.jsonl
git rev-parse --verify hold/verifier-calibration >/dev/null
test "$(git diff --stat hold/verifier-calibration -- src/gcf_qna/rag/verify.py | wc -l)" -gt 0
git diff --quiet HEAD -- src/gcf_qna/rag/verify.py
test -s docs/deployed-env-fingerprint.txt
venv/bin/python -m pytest tests/ -k 'force_record or deploy_excludes' -q
```

Plus, on file: the rollback rehearsal elapsed time, and the standing rule **do not rebuild the index or registry during Waves 0–5**.

---

## Wave 0 — Adversarial review of the held Codex wave

**Nothing builds on unreviewed code, whoever wrote it.** Four commits, different fates.

| Commit | Contents | Owner | Fate |
|---|---|---|---|
| **0a** `tooling: claim adjudication CLI` | `scripts/adjudicate_claims.py`, `scripts/backfill_release_evidence.py`, `tests/test_adjudicate_claims.py` | A1 | lands after review — unblocks Wave 1a |
| **0b** `eval: metric split, groundedness rescoping, parity metadata` | `scripts/eval_answers.py`, `tests/test_eval_answers.py`, `src/gcf_qna/config.py`, `.env.example`, `tests/conftest.py` | A2 | lands after review — unblocks Wave 3A |
| **0c** `verify: matcher calibration (HELD)` | all `verify.py` hunks + the 12 held `tests/test_verify.py` tests + A4's two regression tests | A4 (tests only) | **on `hold/verifier-calibration`, never on the feature branch, never deployed.** Re-lands hunk-by-hunk in Wave 2 |
| **0d** `docs: execution plan` | `docs/claim-support-rollout-plan.md` (+251 uncommitted) | A3 | lands after review. **This document supersedes it**; 0d reduces it to a pointer plus the retracted-numbers note |

`verify.py` on the feature branch stays at HEAD for Waves 0–1 and for the Wave-3A parity baseline. That is what makes Wave 2's gate ("every matcher change maps to adjudicated false-positive anchors") satisfiable at all: code written before adjudication cannot cite adjudication.

### Mandatory fixes before 0a/0b/0d land

| # | Defect | Location | Fix |
|---|---|---|---|
| F1 | Parity metadata reports `production_single_id_prescope: False` while `Pipeline.plan` calls `app._rescope_items`, whose first statement is `_prescope_single_fp` | `eval_answers.py:726,735,786`; `chainlit_app.py:476` | report `True`, delete the limitation string |
| F2 | `compare_verifier_output` treats a broken metric as no-regression — **and the gates do not call it.** `--compare` calls `run_compare` (`eval_answers.py:1456`), which prints a table, returns `None`, exits 0 unconditionally and never emits `no_regression`; `_compare_extras` (`:1527`) silently drops any metric absent from either run | `eval_answers.py:1306-1337`, `:1456`, `:1527` | a `None` metric is a hard `no_regression: false`. Extend to `run_compare`/`_compare_extras`: build the same metric-pair table, return a dict, print it as JSON, `sys.exit(1)` on any regression or `None`. Add `--require-metrics groundedness_rate,citation_completeness_rate` so a missing key is a hard failure. Add the distinct-keys startup assertion (decision 4) |
| F3 | `config.VERIFY_REPAIR` defaults to `"1"`; repair stays off in production only because the rsynced `.env` says `0`. `.env.example` ships `VERIFY_REPAIR=1`; `tests/conftest.py:13` pins `("VERIFY_REPAIR", "1")` as "the shipped default" | `config.py:56`, `.env.example`, `tests/conftest.py:13` | **three-file atomic change**: default `"0"`; `.env.example` → `VERIFY_REPAIR=0`; conftest pin → `("VERIFY_REPAIR", "0")`. Add a mechanical drift test parsing `.env.example` and asserting, for every switch, example value == `config.py` default == conftest pin. Separately reconcile or annotate `.env.example`'s `VERIFY=0`/`PLANNER=0` against deployed `VERIFY=1`/`PLANNER=1` |
| F4 | Claim identity is derived from `extract_claims` — the component Wave 2 modifies (R4-B *is* an extraction-scope change) — so a matcher change silently alters the gold population; no collision handling for duplicate claim texts; 30/71 rows sit truncated at 160 chars in the frozen record | `eval_answers.py` `score_claims`; `adjudicate_claims.py:77-88` | **Freeze the claim inventory as a separately-versioned artifact.** Per case, record the ordered extracted claims with character spans into the recorded answer. Gold keys on `(case_id, span, sha256(normalized text))` with an explicit duplicate index. The release-1 inventory is regenerated by re-extracting from the recorded answers pinned to `verify.py@HEAD`, and the hash of that extraction output is recorded. **Any change to the extracted-claim population is its own gated, adjudicated number** — claims added / removed, per case — reported alongside precision/recall and never inside them |
| F5 | `build_inventory` hard-fails export of all claims if any release row lacks `question`/`answer` | `adjudicate_claims.py:104-105` | skip errored rows, count them, fail only if `errors > 0` and `--allow-errors` absent |
| F6 | `import --allow-unreviewed` can write a canonical adjudicated file containing unreviewed rows | `adjudicate_claims.py:346,370-376` | `--allow-unreviewed` accepted on `validate`/`summary` only |
| F7 | Records store hits as `{doc,page,score}` with no text, so adjudication cannot separate `verifier_false_positive` / `missing_retrieval_evidence` / `genuine_answer_error`, **and release-1 groundedness cannot be computed at all** | record shape `eval_answers.py:1425-1426` | Record `hit.text` going forward. For release-1, **`scripts/backfill_release_evidence.py` (owner A1)**: no re-retrieval — join each record's `evidence_keys` (`"<doc>\|<page>"`) against `data/index/default/chunks.jsonl` (`{doc_id, page, text}`). Index pinned to `data/index/default`; fingerprint defined as `sha256(data/index/default/chunks.jsonl)` (`config.json` holds only `{embedding_model, metric, n_chunks, created_at}` — no source dir). Page-less keys (`"__notes__\|-"`, `"<doc>\|-"`) have no chunk row and **resolve from the record's `notes_used` field**, or note-backed evidence is silently lost. **Precondition for trusting the reconstruction: every recorded `(doc,page,score)` triple across all 66 cases reproduces exactly.** Then replay `classify_deterministic` over the recorded answers to obtain the real release-1 grounded count. If reproduction fails, release-1 is declared un-replayable and re-run under the pinned config. **No groundedness delta may be published before that number exists** |
| F8 | Anchor mutability | — | done in Wave −1 |
| F9 | Two silent samplers one item below the data. `score_claims` records `failures[:6]` while aggregates count all failures (release-1 peaks at 4/case, so 71 currently matches — it diverges the moment a case crosses six). `_judge`'s `max_claims=12` caps LLM adjudication against an observed max of 11 claims/case, so production support is one claim from depending on a cost knob; Wave 3's conductor and decomposition push both | `score_claims`; `verify.py@HEAD` `_judge` | Record `n_failures` alongside the list and hard-fail the inventory export when `len(list) < n_failures`; drop the `[:6]` slice for release runs. Record `judge_candidates` and `judge_budget_exhausted` per case; **gate both release runs on `exhausted == 0` across all 66 cases**; if the cap binds, raise `max_claims` for release runs and record that production and release then differ |
| F10 | `ask_model` (`eval_answers.py:914`) sends no temperature and no seed and records the alias `gpt-5.2`, not the returned snapshot; every `--release` regenerates all 66 answers, so every run is an independent sample | `eval_answers.py:914` | Pin `temperature=0` and a seed where the endpoint supports it; record the **snapshot id returned per call**, not `config.CHAT_MODEL` |
| F11 | `eval_answers.py:77-78` calls `load_dotenv(ROOT / ".env")` — evals inherit production `INDEX_NAME`, keys and flags, and the natural way to get a repair-ON run is to edit the file that ships to production | `eval_answers.py:77` | Read `GCF_QNA_ENV`, defaulting to `data/eval/eval.env`, **never `./.env`**; only the API key is shared. Add tests asserting `--verifier-repair` turns repair on with `VERIFY_REPAIR=0` in the environment, and the converse |
| F12 | Release-1's `claims` block has no `groundedness_rate` / `citation_completeness_rate` / `citation_presence_rate`, so the mandated `release-1 → parity-baseline` compare fails by construction once F2 lands | — | `scripts/eval_answers.py --rescore-record data/eval/release_release-1.jsonl --out data/eval/release_release-1-rescored.jsonl` (owner A2), recomputing all three keys offline from the recorded claims/failures/evidence with **zero API calls**, asserting `support_rate == citation_completeness_rate` and printing groundedness from F7's reconstruction. **Every "release-1" comparison and every dashboard row points at the rescored file; raw `release_release-1.jsonl` is never a `--compare` operand** |

### Reviewer matrix and hunting grounds (default-refute)

Code citations are **tree-tagged**: symbols R1 and R4 probe (`_field_context_amounts`, `_scoped_field_conflict`, `_citation_context`, `_NEG_EXISTENCE_RE`, `_explicit_aliases`, `_claim_for_doc`) exist **only on the hold branch** — they are absent from `git show HEAD:src/gcf_qna/rag/verify.py`, and HEAD's equivalent code sits ~200 lines earlier (`value-present-elsewhere` is `verify.py@HEAD:1215` vs `verify.py@hold:1419`). Setup, mandatory: `git worktree add /tmp/hold hold/verifier-calibration`. **All `verify.py` probes run in `/tmp/hold`; `eval_answers.py` / `adjudicate_claims.py` probes run in the main tree.** Prefer symbol names to line numbers.

| Commit | Reviewers |
|---|---|
| 0a | R3 + two generalists |
| 0b | R2 + two generalists |
| 0d | R3 + one generalist |
| 0c (hold-branch review, not a feature-branch commit) | R1 + R4 |

- **R1 — repair-gate preservation.** Prove the three `817abdb` gates are semantically intact, not textually: `_introduced_sources` exact-match + `allowed_docs`, zero-remaining-failures adoption, gutted-answer protection via `_supported_required`. Probe the one place scoping weakens the recheck: `_field_conflict` early `return None` on empty `stated_amounts` (`verify.py@hold:1129-1131`) and `_scoped_field_conflict` skipping docs with no locally-attributed amounts (`@hold:1146`) — construct a contradiction the old union-text check caught and the new one misses. Dormant at `VERIFY_REPAIR=0`, load-bearing for Wave 5.
- **R2 — metric-split correctness.** Prove the deterministic-mode identity on rescored release-1; prove the rename does not move the printed PASS/FAIL line to a weaker number; prove the 0b groundedness rescoping changes no matcher; prove `_judge`'s promote-only asymmetry is surfaced in production-mode records; prove deleting `grounded-without-citation` and `citation_support_rate` changes no retained metric.
- **R3 — adjudication-tool tamper checks.** Attack `validate_inventory` (`adjudicate_claims.py:156-250`): immutable-field drift, duplicate/unknown/missing claim keys, `ambiguous_unscorable` without notes, unreviewed rows through `import`, atomic-write/`--force` semantics, anchor-file mutation. **Reviewer identity is out of scope — it is not enforceable in-file** (see Wave 1b); report any code path that pretends otherwise.
- **R4 — production-verifier impact.** Two CONFIRMED live regressions on `hold/verifier-calibration`, both pinned as failing tests before Wave 2 re-lands anything:
  - **A:** `FP151 requests **USD 10 million** in GCF financing and **USD 50 million** in total financing [doc, cover pages]` against agreeing evidence — HEAD SUPPORTED, hold **CONTRADICTED**, because `_field_context_amounts` (`@hold:1086-1103`) attributes amounts to the segment *after* each label and orphans amount-before-label figures. User-visible as `⚠️ treat with caution` (`chainlit_app.py:713-743`).
  - **B:** `The total project cost is not found in the retrieved excerpts [doc, p. 5]` — HEAD extracts zero claims, hold makes it an existence claim → UNSUPPORTED, and it consumes judge budget at `VERIFY_LLM=1`.
  - Also: `_citation_context` (`@hold:1061-1083`) drops all text after the last bracket, emptying context for citation-first styles and disabling the `registry_backed` rescue (`@hold:1276-1282`).

**A4** is a Wave-0 implementer owning `tests/test_verify.py` **on `hold/verifier-calibration` only**; sole deliverable is two xfail-free failing tests reproducing A and B. (Reviewers stay read-only; the tests need an author, and Wave 0 had none.)

**Gate 0** (asserting; a line whose expectation lives in a comment is not a gate):

```bash
set -euo pipefail
cd /home/ssa/Workspace/final_project_qna
test "$(git rev-parse --abbrev-ref HEAD)" = feat/claim-support-rollout
make test
test -z "$(git status --porcelain)"
git diff --quiet HEAD -- src/gcf_qna/rag/verify.py
grep -q '("VERIFY_REPAIR", "0")' tests/conftest.py
env -u VERIFY_REPAIR venv/bin/python -c \
  "import sys;sys.path.insert(0,'src');import gcf_qna.config as c;assert c.VERIFY_REPAIR is False"
venv/bin/python scripts/backfill_release_evidence.py --release data/eval/release_release-1.jsonl \
  --index data/index/default --out data/eval/release_release-1-backfilled.jsonl   # exits 1 on any triple mismatch
venv/bin/python scripts/eval_answers.py --rescore-record data/eval/release_release-1-backfilled.jsonl \
  --out data/eval/release_release-1-rescored.jsonl
jq -e '.claims.groundedness_rate != null' <(head -1 data/eval/release_release-1-rescored.jsonl) >/dev/null
python3 scripts/adjudicate_claims.py export --release data/eval/release_release-1-rescored.jsonl \
  --output /tmp/inv.jsonl
test "$(wc -l < /tmp/inv.jsonl)" -eq 71
git worktree add -f /tmp/hold hold/verifier-calibration
! venv/bin/python -m pytest /tmp/hold/tests/test_verify.py -k 'regression_a or regression_b' -q
```

Plus on file: reviewer reports per the matrix, each with a probe log; no reviewer reviewed a file it wrote (checked against `docs/authorship.jsonl`, appended at every wave close); the reconstructed release-1 groundedness `n/d`; the frozen claim-inventory hash.

---

## Wave 1a — Blinded labelling (agent), then STOP

Six mutually exclusive labels (`adjudicate_claims.py:17-24`): `verifier_false_positive`, `genuine_answer_error`, `missing_retrieval_evidence`, `missing_citation`, `registry_conflict`, `ambiguous_unscorable`. No score-based bypass; no tool inference of labels.

**The labelling instrument is blinded.** Handing the labeller the verifier's own `status` and `reason` ("not found in the cited evidence: X") and then asking whether the verifier was wrong builds the verifier's priors into the reference standard, and the anchoring suppresses `verifier_false_positive` rows from ever being proposed — which no amount of owner spot-checking on FP rows can repair. So: export a **labelling view** containing only `question`, full `answer`, the claim text, and the F7 held evidence text, with `status`, `reason` and `kind` **stripped**. The agent answers *"is this claim entailed by the held evidence? yes / no / ambiguous"* plus the cause taxonomy, with no knowledge of what the verifier concluded. Verifier verdicts are joined back afterwards to derive TP/FP.

**The gold set must contain negatives.** Adjudicating only the 71 flagged claims makes Gate 2's `recall == 1.000` mean "do not un-flag failures you already caught" and says nothing about false negatives — SUPPORTED claims that are actually wrong. Every hold-branch hunk is a relaxation, relaxations produce false negatives, and false negatives are the one mechanism that moves support toward 95%. So Wave 1a also labels **a seeded stratified sample of >= 40 of the 94 SUPPORTED claims** (stratified by kind: money / entity / year / number / existence) as gold negatives, through **the same blinded instrument, interleaved, with no indication of which rows were flagged**.

**Evidence resolution rule (state verbatim in the agent's prompt):** page text comes from the backfilled record itself (F7 inlines it, so the agent needs no file access). Where a lookup is unavoidable, it is `data/index/default/chunks.jsonl` filtered on `doc_id == <doc> and page == <page>`. **Reading `data/extracted/` is forbidden** — it holds five sibling, non-pinned extraction runs of whole-document markdown that is not page-addressed, so "page 48" cannot be resolved from it at all.

The agent writes `reviewer: "agent:<id>"` and a one-line rationale in `notes` for every row, to `data/eval/inventory_release-1.jsonl`.

**Wave 1a terminates the workflow run.** It exits 0 and prints the review-queue path and row counts per bucket. **Wave 1b is launched manually by the owner after signing.**

**Wave 1a preconditions:** `data/eval/release_release-1-backfilled.jsonl` exists and every non-`-` evidence key resolved; the seeded supported-claim sample file exists with its seed recorded.

---

## Wave 1b — Owner adoption (out of band), import, gate

**Owner adoption is a signed git commit, not a string field.** The anti-circularity mechanism cannot be a free-text `reviewer` value inside a JSONL that the labelling agent and the Wave-1 tooling agent can both write — `adjudicate_claims.py` cannot distinguish an owner-signed row from an agent that typed the owner's address, and every downstream gate authorizing matcher changes rests on it. Instead:

- The owner reviews **100% of `verifier_false_positive` rows** (the only rows that authorize Wave-2 code changes), **100% of `ambiguous_unscorable` rows** (each with a written resolution task), and a **seeded stratified 20% of every other bucket, minimum 3 rows per bucket** (>= 15 rows).
- The owner writes `data/eval/owner_signoff_release-1.jsonl` — **anchor keys and labels only** — and commits it **GPG-signed, as the commit author, touching nothing else**. Agents cannot produce a signed commit.
- **Disagreement rule:** if the owner overturns > 10% of a bucket's sampled rows, that bucket is re-adjudicated by a second agent through the blinded instrument and re-sampled.

**Gate 1:**

```bash
set -euo pipefail
command -v jq >/dev/null && command -v sha256sum >/dev/null
git log --show-signature -1 -- data/eval/owner_signoff_release-1.jsonl | grep -q 'Good signature'
test "$(git show --name-only --format= HEAD -- | wc -l)" -eq 1
python3 scripts/adjudicate_claims.py validate --release data/eval/release_release-1-rescored.jsonl \
  --inventory data/eval/inventory_release-1.jsonl                       # no --allow-unreviewed
python3 scripts/adjudicate_claims.py import --release data/eval/release_release-1-rescored.jsonl \
  --inventory data/eval/inventory_release-1.jsonl \
  --signoff data/eval/owner_signoff_release-1.jsonl \
  --output data/eval/adjudicated_release-1.jsonl                        # exits 1 on any unreviewed row
# the signed anchor set is exactly the FP set:
diff <(jq -r 'select(.label=="verifier_false_positive").anchor' data/eval/adjudicated_release-1.jsonl | sort) \
     <(jq -r '.anchor' data/eval/owner_signoff_release-1.jsonl | sort)
test "$(jq -r 'select(.label=="ambiguous_unscorable") | select((.notes|length)==0)' \
     data/eval/adjudicated_release-1.jsonl | wc -l)" -eq 0
test "$(jq -r 'select(.role=="gold_negative")' data/eval/adjudicated_release-1.jsonl | wc -l)" -ge 40
git ls-files --error-unmatch data/eval/adjudicated_release-1.jsonl
sha256sum data/eval/release_release-1-rescored.jsonl data/eval/adjudicated_release-1.jsonl
```

(`git add -f` happens in the wave's commit step, not inside the gate; the gate asserts the result.)

Deliverable: label distribution (label × status × kind), the gold-negative stratification, the owner sample and disagreement rate, the sha256 pair. **This is the critical path** — all of Wave 2 blocks on it; Wave 3A is deliberately scheduled in parallel so the machine is not idle.

---

## Wave 2 — Verifier calibration against adjudicated gold

**Sealed held-out slice, before any hunk is written.** Re-landing hunks mapped to the FP anchors they correct and then gating on those same anchors is a fit statistic dressed as a validation statistic — `FP == 0` on the calibration set is guaranteed by construction. So B3 seals a **seeded stratified held-out slice (~20 rows spanning every label and kind, plus the >= 40 gold negatives)** into a file **B1 cannot read**. **Gate 2 passes on the held-out slice**; calibration-set numbers are reported as fit and labelled as such. The five adversarial negatives are written by a reviewer who does not own `verify.py`.

Then re-land `hold/verifier-calibration` **hunk by hunk**, each carrying a git trailer `Adjudicated-anchors: <key>,<key>`. A hunk with no adjudicated FP behind it does not land.

**Tooling built here — `scripts/replay_claims.py` (owner B2).** It re-runs `verify.verify_answer` over the **recorded** answers and evidence and generates no model answers (the shipped harness always regenerates: `eval_answers.py:1374-1376`). Deterministic mode makes zero API calls. **B2 does not import from `eval_answers.py`** — it scores against the recorded JSON schema only, because C1 is concurrently rewriting `Pipeline.plan`, the metric names and `compare_verifier_output` in that file, and disjoint *file* ownership does not give disjoint *interface* ownership.

- `replay --release <rec> --verifier-mode {deterministic,production} [--verifier-repair] --out <jsonl>`
- `score --replay <jsonl> --gold data/eval/adjudicated_release-1.jsonl [--holdout <file>]`
- **Contract for both:** one JSON object on stdout with named keys — `score` emits `false_positives`, `recall`, `false_negative_rate`, `ambiguous`, `new_failures_on_previously_supported`, `claims_added`, `claims_removed`, `supported_before`, `supported_after` — and `sys.exit(1)` if any threshold is breached. `replay --verifier-repair` emits `original_answer` + `answer`; verified on a 2-case fixture as its own gate line.

Definitions: **true failure** = label in {`genuine_answer_error`, `missing_retrieval_evidence`, `missing_citation`, `registry_conflict`}; **false positive** = `verifier_false_positive`; **false negative** = a gold negative the calibrated verifier still calls SUPPORTED; `precision = TP/(TP+FP)` over emitted failures; `recall = TP still flagged / TP in gold`; `ambiguous_unscorable` excluded from both and reported separately.

**Per-relaxation adversarial negative** — one on the strict side, plus the R4 regressions on the permissive side:

| Relaxation | Adversarial negative | Regression pin |
|---|---|---|
| Negative-existence NOT-FOUND acceptance (`_NEG_EXISTENCE_RE`, `_check_existence`) | fabricated "FP999 does not exist" with no registry NOT FOUND line stays UNSUPPORTED | **hedge sentences without an FP/board/year token are not existence claims** (R4-B). `_NEG_EXISTENCE_RE` requires an identifier token; bare "not found in the retrieved excerpts" stays a non-claim as at HEAD |
| Explicit-alias gating (`_entity_present`, `_explicit_aliases`) | invented expansion and manufactured initials both rejected | registry-printed `Full Name (ACRO)` still rescues |
| Same-document / same-field conflict scoping (`_claim_for_doc`, `_scoped_field_conflict`) | cross-document and same-doc-different-field conflicts still suppressed | **amount-before-label phrasing stays SUPPORTED** (R4-A) |
| `_citation_context` trailing text | chained multi-doc brackets still drop foreign clauses | text after the final bracket is **retained**; `registry_backed` rescue fires for citation-first styles |
| Unit guard in `amount_matches` | "20 billion" never aliases "20 million" | EUR/billion counterparts unchanged |

**Wave 2 changes live user-visible behavior on deploy**, with no repair involved: production runs `VERIFY=1, VERIFY_REPAIR=0`, so every re-landed hunk changes which turns render `⚠️ treat with caution` (`chainlit_app.py:713-743`) — regression A *is* that banner. Wave 2 therefore ships behind a documented `VERIFY=0` kill switch with the rehearsed one-command revert from Wave −1, and is observed on the canary at `VERIFY_REPAIR=0` before it reaches production.

**Gate 2:**

```bash
set -euo pipefail
command -v jq >/dev/null && venv/bin/python -c 'import chainlit, faiss'
venv/bin/python scripts/replay_claims.py replay --release data/eval/release_release-1-rescored.jsonl \
  --verifier-mode deterministic --out data/eval/replay_release-1_calibrated.jsonl
venv/bin/python scripts/replay_claims.py score --replay data/eval/replay_release-1_calibrated.jsonl \
  --gold data/eval/adjudicated_release-1.jsonl --holdout data/eval/holdout_release-1.jsonl
venv/bin/python scripts/replay_claims.py replay --release data/eval/release_release-1-rescored.jsonl \
  --verifier-mode production --out data/eval/replay_release-1_prod.jsonl
venv/bin/python scripts/replay_claims.py score --replay data/eval/replay_release-1_prod.jsonl \
  --gold data/eval/adjudicated_release-1.jsonl --holdout data/eval/holdout_release-1.jsonl
make test
venv/bin/python scripts/check_anchor_trailers.py --range hold/verifier-calibration..HEAD \
  --gold data/eval/adjudicated_release-1.jsonl
```

Required, **on the held-out slice**: `false_positives == 0`; `recall == 1.000`; `false_negative_rate == 0` measured on the gold negatives **before and after** calibration on the same rows — a calibration with `FP == 0` and `recall == 1.0` that flips two gold negatives is rejected; `new_failures_on_previously_supported == 0` (or each adjudicated and none an FP). **In production mode: any judge promotion of an adjudicated true failure is a false negative and gates at 0.** Denominator lines: `supported` non-decreasing; `claims_added`/`claims_removed` itemized per case with each removed claim's status and a written justification. Warning-rate line: `warned_turns / total_turns`, HEAD verifier vs calibrated, **no more than +2 turns newly warned, each individually justified**. Then **re-sample fresh supported claims through the blinded instrument** to catch newly-created false negatives. `check_anchor_trailers.py` (owner B3) exits non-zero if any commit touching `verify.py` lacks the trailer or names a key that is absent or not labelled `verifier_false_positive`.

Explicitly **not** a gate here: the >= 95% release threshold. A calibration that raises claim support without satisfying recall and false-negative rate is rejected.

**Reviewer hunting grounds:** (a) hunk → anchor traceability; (b) the five adversarial negatives; (c) a new-FP-candidate sweep over the 94 previously-supported claims.

---

## Wave 3 — Production-parity release harness

**Three deltas, not two.** The `release-1 → parity-baseline` movement bundles eight independent pipeline changes plus fresh answer sampling, and one of them removes claims from the denominator — calling that one clean "measurement" bar is unattributable. The release report publishes:

1. **Definition change** (0b groundedness rescoping + metric rename), measured offline on the rescored release-1 record with zero API calls;
2. **Harness parity** (Half A), landed as an ordered sequence of gaps with a cheap deterministic-mode run **after each**, so every gap carries its own claim-population and rate delta;
3. **Calibration** (Half B) — the only bar that may be called an improvement.

Per-gap **claim-count deltas are published** so denominator effects — the guard-answer verification skip especially, which removes claims outright (2 cases in release-1) — are never inside a rate movement attributed to pipeline quality. The guard-skip claim count is its own line.

**Run-to-run spread, published before Half A.** With F10's pinning in place, run the pinned config **twice on the same tree** and publish the observed spread. **Any gate whose margin is narrower than that spread is reported as indeterminate, not passed.** Single-run numbers are never stated as exact.

Enumerated parity gaps, landed in this recorded order:

| # | Gap | Production | Harness today | Work |
|---|---|---|---|---|
| 1 | Prescope metadata | prescope **does** run | reported `False` | fixed in 0b (F1) |
| 2 | Comparison flag | `assemble(..., decomposed)` (`chainlit_app.py:1186,1254`) | proxied via `multi_identifier` (`eval_answers.py:879-881`) | use the real flag |
| 3 | Abstain | app keeps the **original** body (`chainlit_app.py:793`) | `final_answer = verifier_result.answer` unconditionally (`:1390`) | mirror the app, `or original` |
| 4 | Guard answers | FP-miss guard returns **before** verification (`chainlit_app.py:1116-1133`) | harness verifies guard answers (`:1374-1390`) | skip verification — **report the removed claim count as its own line** |
| 5 | History isolation | system + **one** user turn (`_answer_messages` `chainlit_app.py:806-826`), referents via `_resolved_refs_note` | fixture turns prepended raw (`:907-909`), no refs note | replicate isolation + refs note; the 4 multi-turn cases are the acceptance set |
| 6 | Planner / matrix | `planner.detect` + `_planner_intent` at `PLANNER=1` (`chainlit_app.py:1030-1032`), matrix `1139-1171` | opt-in, default OFF, non-production fallback (`:811-831`) | default ON for `--release`; fallback becomes the conductor |
| 7 | Conductor | `run_conductor` (`chainlit_app.py:1037-1089`) whenever `plan is None` and `CONDUCTOR=1` | absent (`Pipeline.plan` `:766-787`) | wire the real `run_conductor` behind `--conductor`, on by default for `--release` |
| 8 | Verifier config | `verify_answer(use_llm=1, allow_repair=0)` | deterministic default | pinned: `--production-planner --conductor --verifier-mode production` |
| 9 | Usage accounting | — | excludes judge + repair calls (`:1422`) | account them; latency/cost gates must reflect what production pays |

**Keep the deterministic harness.** `--verifier-mode deterministic` stays the cheap, zero-API per-commit instrument; production mode runs at wave boundaries only.

**Gate 3A** (Half A, in parallel with Wave 1, on HEAD `verify.py`):

```bash
set -euo pipefail
git diff --quiet HEAD -- src/gcf_qna/rag/verify.py
venv/bin/python scripts/eval_answers.py --release --production-planner --conductor \
  --verifier-mode production --record parity-baseline
git diff --quiet HEAD -- src/gcf_qna/rag/verify.py        # unchanged for the whole run
jq -e '.pipeline_parity.level=="full"' <(head -1 data/eval/release_parity-baseline.jsonl) >/dev/null
jq -e '.verify_blob_sha != null and .judge_budget_exhausted==0' \
  <(head -1 data/eval/release_parity-baseline.jsonl) >/dev/null
diff <(sha256sum data/index/default/config.json data/index/default/index.faiss data/registry_v2.json) \
     <(grep -A3 '^local-artifacts' docs/deployed-env-fingerprint.txt | tail -3)
```

The record stores the `verify.py` blob sha so Half B can prove Half A ran pre-calibration.

**Gate 3B** (Half B, after Wave 2):

```bash
set -euo pipefail
venv/bin/python scripts/eval_answers.py --release --production-planner --conductor \
  --verifier-mode production --record release-2
venv/bin/python scripts/eval_answers.py --compare data/eval/release_release-1-rescored.jsonl \
  data/eval/release_parity-baseline.jsonl \
  --require-metrics groundedness_rate,citation_completeness_rate
venv/bin/python scripts/eval_answers.py --compare data/eval/release_parity-baseline.jsonl \
  data/eval/release_release-2.jsonl \
  --require-metrics groundedness_rate,citation_completeness_rate
venv/bin/python scripts/eval_answers.py --gate                    # doc recall@5 == 30/30
```

Required: 66/66 complete, 0 harness errors; `judge_budget_exhausted == 0` for all 66; `pipeline_parity` shows `conductor.used=true`, `planner.used=true`, `answer_history_isolation=true`, `production_single_id_prescope=true`. **`level=full` may be asserted only when the Wave −1 deployment fingerprint matches the harness's pinned config field-for-field** (`CONDUCTOR/PLANNER/VERIFY/VERIFY_LLM/VERIFY_REPAIR/INDEX_NAME/CHAT_MODEL` + app commit sha); otherwise the record says `level=partial: unverified-deployment` and release-2 is not called production-representative. Local and remote artifact hashes match. Groundedness, citation completeness and supported all printed as `n/d`; answer recall@5 >= 64/66; evidence-page hit >= its 0b-recomputed count; field coverage >= 54/56.

---

## Wave 4 — Offline repair A/B over identical answers, 66 cases

**The A/B runs over one recorded answer set, not two generations.** Five independent `--release` runs are five samples of the answer distribution, so a generated A/B confounds the repair effect with generation variance. **Wave 3 Half B's recorded raw answers (`data/eval/release_release-2.jsonl`) are the repair input**; Wave 2 contributes the calibrated verifier, not answers. The second `--release` generation is deleted from this gate.

**Score twice.** `_carry_cleared` (`verify.py@HEAD`, WT `:1838-1847`) copies pre-repair judge SUPPORTED rulings onto post-repair verdicts by normalized claim text, upgrading any post-repair failure whose sentence repair left untouched — the headline number would carry forward the judge's verdicts on the *unrepaired* answer, which is the self-certification pattern this wave forbids elsewhere. So: score once with `_carry_cleared` **disabled** and a fresh classification over the repaired text, and once as production ships it. **Gate on the carry-off number; report carry-on as the production-behavior figure.** If they differ by more than a stated tolerance, that difference is a named release item, not a rounding note.

**`scripts/audit_repair.py` (owner D2) is the auditor.** `compare_verifier_output` is untouched in Wave 4 beyond F2. Contract: one JSON object on stdout with `invented_docs`, `invented_pages`, `invented_figures`, `lost_claims[]` (each with status and justification field), `lost_supported_claims`, `answer_checks_pass_to_fail`, `claims_before`, `claims_after`, `supported_before`, `supported_after`; `sys.exit(1)` on any breach. Checks: claim-set diff per case; cited doc/page set diff — any doc or page in the repaired answer absent from the raw answer's held evidence is an **invented source**, checked independently of `verify.py`'s own `_introduced_sources` gate; numeric-token diff; length / answer-check regression.

**Gate 4:**

```bash
set -euo pipefail
venv/bin/python scripts/replay_claims.py replay --release data/eval/release_release-2.jsonl \
  --verifier-mode production --out data/eval/replay_repair-off.jsonl
venv/bin/python scripts/replay_claims.py replay --release data/eval/release_release-2.jsonl \
  --verifier-mode production --verifier-repair --out data/eval/replay_repair-on.jsonl
venv/bin/python scripts/replay_claims.py replay --release data/eval/release_release-2.jsonl \
  --verifier-mode production --verifier-repair --no-carry-cleared --out data/eval/replay_repair-on-carryoff.jsonl
venv/bin/python scripts/eval_answers.py --compare data/eval/replay_repair-off.jsonl \
  data/eval/replay_repair-on-carryoff.jsonl \
  --require-metrics groundedness_rate,citation_completeness_rate
venv/bin/python scripts/audit_repair.py --off data/eval/replay_repair-off.jsonl \
  --on data/eval/replay_repair-on-carryoff.jsonl
```

Required, on the **carry-off** numbers: groundedness `n/d >= 95%` **AND** citation completeness `n/d >= 95%`, both printed with their exact integer thresholds; `no_regression: true` with no `None` metrics (F2); zero invented docs/pages/figures; answer-check pass→fail == 0; `supported` non-decreasing; every claim lost between off and on listed **with its status** and a written justification — the profitable deletions are the unsupported ones; latency p95 and cost recorded including judge and repair calls. Any gate whose margin is narrower than Wave 3's published run-to-run spread is reported **indeterminate**. Residue becomes a named release item with an owner.

---

## Wave 5 — Live canary, then the flip

`VERIFY_REPAIR` is a process-wide env flag (`config.py:56`, consumed `chainlit_app.py:785`), so "owner-controlled sessions" requires a second deployment. **It cannot be a second service in the production compose project:** `docker-compose.yaml` hardcodes `container_name: fp-gcf`, `image: fp-gcf:latest` and `labels: caddy: fp-gcf.ssa.tg`, publishes no host ports (caddy routes by label, so "separate port" is meaningless), and bringing a sibling up recreates `fp-gcf` — "production untouched" would be false — while both would bind-mount the same `./data`, so canary turns would write into the production `app.db` and `public/app_files`.

**Canary = a separate compose project in a separate remote directory:** `/workspace/fp_gcf_canary`, `COMPOSE_PROJECT_NAME=fp-gcf-canary`, `container_name: fp-gcf-canary`, its own caddy label on a canary hostname (DNS created in advance), its own `.env` with `VERIFY_REPAIR=1`, its own `APP_DB` and `public/app_files`. `data/index`, `data/raw`, `data/cache` bind-mounted **read-only** from the production dir. Up with `docker compose -p fp-gcf-canary up -d`; **never run production's `make deploy` while the canary is live**. Rollback: `docker compose -p fp-gcf-canary down` — production never touched.

The Wave-2 calibrated verifier is observed here first at `VERIFY_REPAIR=0` before it reaches production.

**Canary logs go to `data/canary/`** (excluded from `DEPLOY_EXCLUDES` in Wave −1, so a later push cannot overwrite them), JSONL per turn: question, query class, language, raw answer, repaired answer, cited evidence, full verifier result, reviewer decision. `make pull-canary` (`rsync -avz root@host:/workspace/fp_gcf_canary/data/canary/ data/canary/`) brings them back and hashes them on retrieval.

**Sample (>= 50 factual turns, >= 2 sessions), class quotas:** >= 10 comparison/conflict, >= 10 follow-up, >= 10 French, >= 5 abstention.

**Review protocol:** every turn that produced a warning or a repair, **plus >= 20 no-warning answers** sampled with a recorded seed — a verifier that never warns is not thereby correct. Each reviewed turn is labelled with the same six-label scheme through the blinded instrument, so canary findings feed straight into the gold set.

**Gate 5:** canary records pulled and sha256 recorded; zero false warnings, zero harmful repairs, zero invented citations, zero critical unsupported claims in the sample; no unresolved regression; latency p95 and per-turn cost within the Wave-4 envelope; post-deploy smoke turn passed and container healthy. Any failure brings the canary down and loops back to the owning wave.

**Flip:** update `VERIFY_REPAIR=1` in the **remote** `.env` over ssh (never by rsync — `.env` is excluded as of Wave −1), `make deploy` with `FLIP=1`, confirm the deployed sha and container health, run the smoke turn, keep the canary records, and publish the release report as **four bars**: definition change → harness parity → calibration → repair, each `n/d`, with the measurement/improvement split stated explicitly and any indeterminate gate named. Roll back via `make rollback SHA=<previous>` on harmful rewrite, invented source, false-warning spike, or latency/cost breach.

---

## Execution model (ultracode)

Each wave is one ultracode workflow run except Wave 1, which is two (1a ends the run; 1b is launched by the owner after signing). Implementation agents are Opus-class.

**Dependency graph:** `−1 → 0 → 1a → [owner sign-off] → 1b → 2 → 3B → 4 → 5`, with **3A branching off 0** and running concurrently with 1a/1b.

**Preflight, at the top of every wave:**

```bash
set -euo pipefail
command -v jq >/dev/null && command -v sha256sum >/dev/null
venv/bin/python -c 'import chainlit, faiss'      # skip for Waves −1, 0, 1
```

Interpreter rule, stated once: **`python3` for stdlib-only tools** (`adjudicate_claims.py`, `check_anchor_trailers.py`); **`venv/bin/python` for anything importing `gcf_qna.rag`**.

**Phase 1 — finders (read-only).** N agents, disjoint read targets, schema-forced JSON returns (`{summary, key_facts[], risks[], open_questions[]}`). Finders never write.

**Phase 2 — implementers (disjoint file ownership).** One agent owns one file set for the whole wave.

| Wave | `verify.py` | `eval_answers.py` | `config.py` / `.env.example` / `conftest.py` | `adjudicate_claims.py` / `backfill_*` / `replay_claims.py` / `audit_repair.py` / `check_anchor_trailers.py` | tests | Makefile / compose | docs / data |
|---|---|---|---|---|---|---|---|
| −1 | A0 (move to hold) | — | — | — | A0 | A0 | A0 |
| 0 | A4 (`hold` only) | A2 | A2 | A1 | A1 `test_adjudicate_claims.py`, A2 `test_eval_answers.py`, A4 `test_verify.py`@hold | — | A3 |
| 1a/1b | — | — | — | A1 (fixes only) | — | — | labelling agent (no code access) |
| 2 | B1 | — | — | B2 (`replay_claims.py`), B3 (`check_anchor_trailers.py`, held-out sealing) | B1 `test_verify.py` | — | B3 |
| 3 | — | C1 | C1 | — | C1 | — | C2 |
| 4 | — | — | — | D2 (`audit_repair.py`) | D2 | — | D3 |
| 5 | — | — | — | E1 (canary logger) | E1 | E1 | E2 |

**Interface freeze (Waves 2–3).** `replay_claims.py` may import **nothing** from `eval_answers.py`; it scores against the recorded JSON schema. `verify.verify_answer` and the `SUPPORTED` constant are frozen in signature for Waves 2–3; changing either requires a joint B1/C1 review.

**Phase 3 — adversarial verify (default-refute).** >= 3 reviewers per commit per the published matrix, each given the raw diff and the gate commands, **not** the implementer's summary. Each attempts refutation and reports the probes it ran; "looks correct" without a probe log is a rejected review. A reviewer may not review a file it wrote in any wave — enforced against `docs/authorship.jsonl`, appended at every wave close. Hunting grounds are published per wave (Wave 0: repair gates, metric split, tamper checks, production impact; Wave 2: hunk→anchor traceability, adversarial negatives, new-FP sweep; Wave 3: per-gap attribution and denominator movement; Wave 4: invented sources and carry-off vs carry-on; Wave 5: canary isolation).

**Phase 4 — gate.** The wave's gate block runs verbatim under `set -euo pipefail`. **A gate line whose expectation lives in a `#` comment is not a gate.** Non-zero exit or an out-of-tolerance number blocks the commit; there is no narrative override.

**Standing rules.**
1. **No self-certification.** No agent certifies its own output; no component certifies itself — including for headline numbers (Wave 4 gates carry-off, not carry-on).
2. **Tests pin fixed behavior, both directions.** Every relaxation gets an adversarial negative *and* a regression pin on the permissive side — the 12 held tests pin only the strict direction, which is exactly why regressions A and B survived them.
3. **Byte-identical fallback proofs where switches exist.** For every flag (`--conductor`, `--verifier-mode`, `--verifier-repair`, `VERIFY_REPAIR`), prove the off-path output is byte-identical to the pre-change path on a fixed fixture. Zero-API guarantees are tested, not asserted.
4. **Test isolation stays.** `tests/conftest.py:5-16` pins shipped defaults; changes require production-grade review and are owned by A2 in Wave 0, C1 thereafter. The drift test (F3) makes example/default/pin skew a test failure rather than a per-wave argument.
5. **Every wave ends in a commit or an explicit hold.** Held code lives on a named branch, never in the working tree.
6. **Do not rebuild the index or the registry during Waves 0–5.** Their hashes are gate lines in Waves 3, 4 and 5.

---

## Acceptance dashboard

All claim metrics are `n/d`. Gates are counts.

| Metric | Current | Gate | Measured by |
|---|---|---|---|
| Doc recall@5 (gold-30) | 30/30 | 30/30 | `eval_answers.py --gate` |
| Answer recall@5 (66) | 96% recorded; exact count from 0b | >= 64/66 | `--release --record <label>` |
| Evidence-page hit | 94% recorded; exact count from 0b | >= 0b-recomputed count | same |
| Required-field coverage | 54/56 | >= 54/56 | same |
| **Groundedness** | **unmeasured at release-1** (no key in the record; hits carry no text) — reconstructed by F7 or release-1 declared un-replayable | **>= 95% as `n/d`, carry-off, post-repair** | F7 backfill + replay; `groundedness_rate` |
| **Citation completeness** (cited AND supported) | 94/165 | **>= 95% as `n/d`, carry-off, post-repair** | `citation_completeness_rate` |
| Supported (deterministic) | 94/165 | non-decreasing at every wave | release record |
| Claim population | 165 | within stated band of parity baseline; excursions itemized per case | claim-inventory artifact (F4) |
| Guard-skip claims removed | n/a | reported as its own line, never inside a rate | Wave 3 Half A per-gap record |
| Citation presence (any citation) | unmeasured | reported, not gated | `citation_presence_rate` |
| Judge budget exhausted | unrecorded | 0/66 both release runs | release record (F9) |
| Adjudication coverage | 0/71 labelled | 71/71 + >= 40 gold negatives, 0 unreviewed, FP set == GPG-signed set | `adjudicate_claims.py` + `git log --show-signature` |
| Verifier FP / recall / FN vs gold (**held-out slice**) | unmeasured | 0 / 1.000 / 0 | `replay_claims.py score --holdout` |
| Judge promotions of adjudicated true failures | unmeasured | 0 | `score --verifier-mode production` |
| New failures on previously-supported claims | n/a | 0 unadjudicated | `replay_claims.py score` |
| Live warning rate (66-case run) | baseline from HEAD verifier | <= +2 turns newly warned, each justified | Wave 2 gate |
| Run-to-run spread (pinned config, same tree) | unmeasured | published; gates inside it read **indeterminate** | two Wave-3 pre-runs |
| Harness parity level | `partial` | `full`, and only with a matching deployment fingerprint | `pipeline_parity` block |
| Repair: answers degraded / invented sources | unmeasured | 0 / 0 | `audit_repair.py` |
| Carry-on vs carry-off gap | unmeasured | within stated tolerance, else a named release item | Wave 4 |
| Canary | 0 turns | >= 50 turns, quotas met, records pulled + hashed, 0 false warnings / harmful repairs / invented citations | `data/canary/` + review log |
| Test suite | 575 pass (~11 s) | 575+ pass | `make test` |
| Latency / cost per release run | p50 3.4 s, p95 8.3 s, ~$0.33 | p50 <= 4.0 s, p95 <= 10.0 s, <= $0.50 **including judge + repair** | `usage_accounting` (fixed in Wave 3) |
| **Deployed sha** | unrecorded | matches the intended release commit | `ssh … 'docker inspect -f {{.Config.Image}} fp-gcf'` |
| **Container health** | unchecked | healthy (`/auth/config` healthcheck) | `docker compose ps` |
| **Live warning rate, last N turns** | unmeasured | within the canary envelope | canary/production log tail |
| **Post-deploy smoke turn** | n/a | passes — mandatory in every wave gate that deploys | one live turn against the target host |
| Artifact hashes (index, registry) local == remote | unverified | equal | Waves 3, 4, 5 gate lines |

---

## Owner actions

1. **Revoke the five leaked API keys**, then rotate in the Wave −1 order: `.env` exclusion landed → rotate on the server via ssh → rotate locally → one live smoke turn against `fp-gcf.ssa.tg` returning an answer with citations. This is *not* wave-independent: before the exclusion lands, `make push` overwrites the remote `.env` from the local one, reverting a server-side rotation and shipping a file missing `APP_USERS`, `ALLOW_SIGNUP`, `FP_GCF_DOMAIN`, `CONDUCTOR` and `VERIFY_LLM`.
2. **Register the SSH key** so Wave 2/3 output can reach `fp-gcf.ssa.tg`.
3. **Confirm the venv has the app stack**: `venv/bin/python -c "import chainlit, faiss"`; if it fails, `make install`.
4. **Sign the adjudication out of band** (Wave 1b): review 100% of `verifier_false_positive` rows, 100% of `ambiguous_unscorable` rows, and a seeded stratified 20% of the rest, then commit `data/eval/owner_signoff_release-1.jsonl` **GPG-signed, touching nothing else**. This is the only step that cannot be delegated — a `reviewer` string in a JSONL is not a signature, and Wave 2's entire authority rests on this commit.
5. **Canary participation** (Wave 5): >= 50 factual turns across >= 2 sessions against `fp-gcf-canary`, quotas met, plus review of every warning/repair turn and >= 20 no-warning answers.
6. **Run the deployment fingerprint capture** (Wave −1 step 6) and commit its output. `level=full` in Wave 3 is asserted only against it.
7. **Decide the flip date to cite** in the final report: `.env:22` says 2026-08-21, the enabling commits and the release-1 run are dated Aug 19. Pick one and use it everywhere.
8. **Approve the rollback rehearsal result** (Wave −1 step 3) before Wave 2 ships anything.

**Retracted numbers.** The committed plan's failure buckets (24 note-backed + 36 field disagreements + 9 conflict double-counts + 9 abstentions + ~4 acronyms = 82 labels for 71 failures) overlap and must never be cited as adjudicated counts. Cite only 94/165 and the six-label scheme. **Any "57.0% groundedness" figure is fabricated** — release-1 groundedness is unmeasured and, before F7, bounded only by [94/165, 118/165] because 24 of the 71 failures are "no citation on a factual claim", the exact branch that sets `value-present-elsewhere`. The wave-3 "0 false positives on 24 recorded gpt-5.2 answers" figure (commit `817abdb`, `docs/build-report.html:110,423,431`) **cannot be re-run** — the artifact was never checked in. Its replacement is the Wave-2 replay over all 165 release-1 claims against adjudicated gold plus the sealed held-out slice; the old figure may be cited historically, never as a live gate.

---

## Non-goals

Not in this phase, and not because they are hard — because **no measured gap points at them**:

- **GraphRAG / knowledge-graph retrieval.** Doc recall@5 is 30/30 on gold-30 and answer recall@5 is 96%; the failures are verification and citation failures, not retrieval-topology failures.
- **Agentic / iterative search.** Evidence-page hit is 94%; search loops change the measurement surface without addressing the 71 adjudicated failures.
- **Long-context (whole-document prompting).** Field coverage is 54/56; the cost and latency envelope is a gate, not a budget to spend.
- **Vision-based verification of cited pages.** The grounded viewer already renders cited page regions; verifying pixels adds a model dependency where the failure mode is text-matcher calibration.
- **Activating `INDEX_NAME=v2`.** Section-aware retrieval code and the schema-v2 artifact exist; activation stays deferred behind its own retrieval and metadata gates and must not be entangled with a claim-support measurement — and rebuilding or switching an index is forbidden outright during Waves 0–5.

Revisit any of these only when a wave produces a measurement that names it.
