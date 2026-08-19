# Claim Support & Rollout Plan

**GCF Q&A · next implementation phase · 21 Aug 2026**

Successor to `rag-correctness-next-steps.html`, whose six steps are built, reviewed, and deployed. One definition-of-done clause remains unmet, and this plan exists to close it: *"every displayed factual claim is supported by its cited evidence."* First honest full-suite measurement: **57% claim support vs the ≥95% gate** (release run `release-1`, 66 answers, 165 claims). Everything below is grounded in that run's failure taxonomy — no speculative work.

## Where we stand

| Gate | Status |
|---|---|
| Document recall@5 (gold 30) | 100% — never regressed |
| Answer recall@5 (66 cases) | 96% |
| Evidence-page hit | 94% |
| Required-field coverage | 96% (54/56 scorable cells) |
| **Claim support** | **57% — FAIL (gate ≥95%)** |
| Latency / cost per release run | p50 3.4 s · p95 8.3 s · ~$0.33 |

Live in production: pre-scoping, history isolation, section-aware retrieval, provenance registry notes, `PLANNER=1`, `VERIFY=1` in observation mode (`VERIFY_REPAIR=0`).

---

## Step 1 — Make the verifier as evidence-aware as the answer path

**The finding:** the verifier models a claim as a simple sentence-citation pair. Real gpt-5.2 answers have structure the model doesn't cover, so *correct* answers score as failures. The release run's 71 failed claims decompose into:

| Bucket | Claims | Fix |
|---|---|---|
| Uncited but note-backed statements | 24 | Before declaring "no citation on a factual claim", check the claim's values/names against the notes evidence block (`NOTES_KEY`) — a figure the registry note supplied is grounded, cited or not |
| Field-value disagreements + absent values | 36 | Triage: some are genuine model errors (the repair pass's actual job); some are matcher gaps — fix only what inspection proves is a matcher gap, report the honest residue |
| Correct both-sides conflict reports double-counted | 9 | `registry_conflict` becomes answer-level: if the *answer* (any unit) carries the counterpart figure, no claim in it is contradicted for that field |
| Correct abstentions penalized | 9 | Negative-existence claims ("FP999 does not exist") match the registry NOT-FOUND note as supporting evidence |
| Acronym expansion scored as miss | ~4 | Entity matching accepts registered acronym ↔ full-name variants (IFAD case) |

**Touchpoints:** `src/gcf_qna/rag/verify.py`, `tests/test_verify.py`. One Opus agent + adversarial review, same cycle as every prior wave.

**Gate:** re-run `--release --record release-2`; claim support ≥95% **or** a documented residue where every remaining failure is a verified genuine model error (those are repair's job, not the matcher's). The 24-answer false-positive table must stay at 0. No regression in any other release metric.

## Step 2 — Re-measure and compare

Run `eval_answers.py --compare release_release-1.jsonl release_release-2.jsonl`. Every improvement must be attributable to a step-1 bucket; any unexplained movement is a finding, not a win. Commit + deploy (verifier changes reach production observation mode immediately).

## Step 3 — Live observation window

The fixtures have never been the whole story — FP86, the year aggregates, and the citation-bracket bug all came from live transcripts, not the gold set.

- Owner runs comparison-heavy and French sessions against production (planner path + verification warnings are live now).
- **Pass criterion:** zero ⚠️ verification lines on answers that are actually correct, across the sessions run. Any false flag becomes a step-1-style bucket and loops back.
- I score pasted transcripts against expectations, as established.

## Step 4 — Flip repair on

`VERIFY_REPAIR=1` (one-line `.env` change + redeploy) once steps 1–3 hold. Repair is already gated hard — adopted only when re-verification shows zero remaining failures, no introduced sources, supported claims retained — so the risk after a clean observation window is bounded. Verify with the adversarial probe set (wrong figure, invented page, gutted rewrite) against production.

## Step 5 — Deferred technical items (parallel, any order)

1. **Rotated-page line geometry** — re-extract `boxes.json` for the 205 rotated pages with a rotation-aware runaway-height cap (pure pymupdf, no VLM). Today highlights there are correctly *placed* but cover a median 22% of true line width.
2. **Section-path hygiene** — reject heading candidates over ~90 chars (28.7% of stored paths are bolded prose, not headings); rebuild v2 index metadata. Precondition for ever activating `SECTION_EXPAND`.
3. **Multilingual embedder A/B** — the deferred experiment the plan gated on labels that now exist: candidate model vs mpnet on gold recall, page-hit, and *French raw recall* (currently 12% raw / carried by conductor translation). Decides whether `INDEX_NAME=v2` + a re-embed ever ships.
4. **Remaining model corpora** — `make extract` resumes the three unextracted models, if the model-comparison angle still matters for the final report.

## Step 6 — Owner actions (only you can)

- **Revoke the five leaked API keys** — the old history is still public at the original repo.
- **Register the SSH key** (`github.com/settings/ssh/new`) so all branches push — `main` needs `--force-with-lease` to replace the leaked-key history.
- Decide whether the final academic deliverable needs anything beyond the build-report artifact (demo script, slides, defense notes) — say the word and it gets produced.

---

## Acceptance

| Metric | Now | Gate to close this plan |
|---|---|---|
| Claim support (full suite) | 57% | ≥95% or verified-genuine residue |
| Verifier false positives (recorded answers) | 0 | stays 0 |
| Live false ⚠️ flags | unmeasured | 0 across observation window |
| `VERIFY_REPAIR` | 0 | 1, with adversarial probes passing in production |
| All other release metrics | — | no regression |

**Definition of done:** the release report shows every gate green or a documented genuine-error residue; repair is live; the observation window produced no false flags; the deferred items are either done or explicitly re-deferred with reasons.

**Not in scope, still deferred:** GraphRAG, agentic search, long-context whole-document reading, vision verification of tables — same reasoning as the prior plan: no measured gap currently points at them.

---

*Companion to `rag-correctness-next-steps.html` (implemented) and `data/eval/release_release-1.jsonl` (the measurement this plan answers). Sequencing: step 1 is the critical path (~half a day); step 3 runs parallel from today; steps 5–6 are independent.*
