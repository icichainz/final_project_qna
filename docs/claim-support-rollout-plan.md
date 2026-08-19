# Claim Support and Rollout Plan

**GCF Q&A | next implementation phase | 19 Aug 2026**

This plan succeeds `rag-correctness-next-steps.html`. Its purpose is to close
the remaining definition-of-done gap: factual claims must be grounded in the
available evidence, and the answer must cite the evidence that supports them.
The first full-suite deterministic measurement (`release-1`) found 94 supported
claims and 71 failed claims across 66 answers: **57% grounded claim support
against a >=95% release gate**.

The 71 failures are not yet a human-labelled error taxonomy. Matcher changes
must not be justified by inferred or overlapping buckets.

## Current state

| Gate | Current measurement |
|---|---|
| Document recall@5 (gold 30) | 100% |
| Answer recall@5 (66 cases) | 96% |
| Evidence-page hit | 94% |
| Required-field coverage | 96% (54/56 scorable cells) |
| Deterministic grounded claim support | 57% (94/165) - FAIL |
| Release-run latency and cost | p50 3.4 s, p95 8.3 s, about $0.33 |

The configured runtime has `PLANNER=1`, `VERIFY=1`, and `VERIFY_REPAIR=0`.
It selects `INDEX_NAME=default`, whose index is the active schema-v1 artifact.
Section-aware retrieval code and the schema-v2 index artifact exist, but v2 is
not active. Activating `INDEX_NAME=v2` remains deferred until its own retrieval
and metadata gates pass.

## Metric contract

Two related metrics must remain separate:

1. **Groundedness:** the evidence available to the answer path entails the
   factual claim.
2. **Citation completeness:** each factual claim identifies the evidence that
   entails it.

A note-backed but uncited claim may be grounded, but it is citation-incomplete.
It must not be converted into a fully supported claim merely because the note
appears elsewhere in the prompt. Generation or repair must attach the note's
provenance citation.

## Step 1 - Human-adjudicate the release-1 failures

Export a deterministic inventory:

```bash
venv/bin/python scripts/adjudicate_claims.py export \
  --release data/eval/release_release-1.jsonl \
  --output data/eval/release_release-1-adjudication.work.jsonl
```

Every exported claim starts with `label: null`; the tool does not infer human
judgements. A reviewer assigns exactly one mutually exclusive root-cause label:

| Label | Meaning |
|---|---|
| `verifier_false_positive` | The recorded evidence and citation support the claim, but the verifier rejected it |
| `genuine_answer_error` | The answer states an incorrect, contradictory, or materially incomplete fact |
| `missing_retrieval_evidence` | The needed source evidence was not available in the recorded retrieval context |
| `missing_citation` | Supporting evidence was available, but the claim did not cite it |
| `registry_conflict` | Registry or document conflict metadata caused or exposes the failure and needs source-level resolution |
| `ambiguous_unscorable` | The record is insufficient for a defensible judgement; the ambiguity must be documented |

Each labelled row requires a reviewer identity. Notes should identify the
supporting page/evidence or explain why the claim cannot be scored. Validate and
canonicalize the completed review with:

```bash
venv/bin/python scripts/adjudicate_claims.py import \
  --release data/eval/release_release-1.jsonl \
  --inventory data/eval/release_release-1-adjudication.work.jsonl \
  --output data/eval/release_release-1-adjudicated.jsonl
```

`validate` and `summary` are also available. They fail when any claim remains
unreviewed unless `--allow-unreviewed` is explicitly supplied for a
work-in-progress check. Imports also reject missing, duplicate, unknown, or
source-modified claims.

**Gate:** all 71 failures have one valid human label and a named reviewer;
unreviewed count is zero; source fields validate against `release-1`; ambiguous
claims have written resolution tasks. There is no score-based bypass.

## Step 2 - Calibrate the verifier against the adjudicated gold

Change only behavior proven to be a `verifier_false_positive`. Keep genuine
answer errors, missing citations, missing retrieval evidence, unresolved
registry conflicts, and ambiguous claims visible. In particular:

- Do not accept uncited note-backed facts as citation-complete.
- Match conflict counterparts only within the same document, semantic field,
  currency/unit, and supported source.
- Accept acronym expansions only from registry aliases or explicit evidence.
- Add an adversarial negative for every matcher relaxation.

Report verifier precision and recall against the adjudicated claim set, with
`ambiguous_unscorable` reported separately rather than silently removed.

**Gate:** every matcher change maps to adjudicated false-positive claim IDs;
all known false positives are corrected; no adjudicated true failure changes to
supported; adversarial verifier tests and the existing suite pass. The release
claim-support threshold is not used as a substitute for calibration accuracy.

## Step 3 - Establish a production-parity release run

The release harness must exercise the same planning, conversational reference
resolution, retrieval, evidence construction, and verifier configuration as the
deployed answer path. It must record these separately:

- deterministic groundedness;
- citation completeness;
- complete verifier outcome, including any LLM adjudication;
- planner/conductor decisions needed to diagnose follow-up cases;
- latency, token use, retrieval, required fields, and answer checks.

Run a new parity baseline before claiming movement from `release-1`; harness
parity changes are measurement changes and must not be attributed to matcher
improvements. Then run the calibrated verifier under the same pinned setup.

**Gate:** all 66 cases complete without harness errors; planner and follow-up
fixtures demonstrably use the production path; groundedness and citation
completeness are both reported; existing retrieval, behavior, language,
required-field, latency, and cost gates do not regress.

## Step 4 - Evaluate repair offline on the full suite

Run all 66 cases with repair disabled and enabled under the same pinned release
configuration. Preserve both raw and repaired answers and compare them at claim
and answer level. Replay fixed raw answers through repair where possible to
separate repair behavior from model sampling variance.

The comparison must check:

- groundedness and citation completeness before and after repair;
- supported claims retained;
- no new claims, documents, pages, or invented sources;
- genuine errors corrected rather than merely deleted when evidence permits;
- required fields, conflict reporting, abstentions, language, and formatting;
- latency and cost impact.

**Gate:** repaired groundedness and citation completeness each meet the >=95%
release threshold; no previously correct answer becomes incorrect; no supported
claim is lost without a documented necessity; no citation or source is
invented; every other release metric remains within its existing gate. Any
residue remains a failed release item with an owner and follow-up action.

## Step 5 - Run a defined live canary

Deploy repair only to owner-controlled canary sessions, with
`VERIFY_REPAIR=0` as the immediate rollback. Review at least 50 factual turns
across at least two sessions. The set must include at least 10 comparison or
conflict turns, 10 follow-up/reference turns, 10 French turns, and 5 abstention
or not-found turns; categories may overlap.

Human-review every verifier warning or repair and a sample of at least 20
no-warning answers. Record the raw answer, repaired answer, cited evidence,
verifier result, query class, language, and reviewer decision.

**Gate:** zero false warnings or harmful repairs, zero invented citations,
zero critical unsupported claims in the reviewed no-warning sample, and no
unresolved regression in behavior, language, required fields, latency, or cost.
Any failure disables repair and returns to the relevant earlier step.

## Step 6 - Enable repair broadly

Set `VERIFY_REPAIR=1` for the normal deployment only after Steps 1-5 pass.
Preserve the canary records as rollout evidence and continue monitoring raw and
repaired verification outcomes. Roll back to `VERIFY_REPAIR=0` on a harmful
rewrite, invented source, false warning spike, or material latency/cost breach.

## Deferred technical work

These items are independent of the claim-support critical path unless
adjudication identifies them as a measured root cause:

1. Re-extract rotated-page line geometry with a rotation-aware height cap.
2. Reject heading candidates over about 90 characters and rebuild v2 metadata.
3. A/B a multilingual embedder against mpnet on recall, page hit, and French
   raw recall before considering activation of the v2 index.
4. Resume extraction of the remaining model corpora if model comparison remains
   part of the final report.

GraphRAG, agentic search, whole-document long-context reading, and vision table
verification remain out of scope until a measured, adjudicated failure points
to them.

## Owner actions

- Revoke the five API keys exposed in the old public history.
- Register the SSH key and replace the leaked-key history using the already
  documented repository procedure.
- Decide which final academic artifacts are required beyond the build report.

## Acceptance

| Metric | Gate |
|---|---|
| Adjudication completeness | 71/71 labelled, 0 unreviewed, source validation passes |
| Verifier calibration | Known false positives corrected; no true failures accepted |
| Production parity | Planner, conversation, retrieval, and full verifier paths exercised |
| Groundedness, full suite after repair | >=95% |
| Citation completeness, full suite after repair | >=95% |
| Repair regression | 0 previously correct answers made incorrect; 0 invented sources |
| Live canary | Defined 50-turn mix completed with all Step 5 gates passing |
| Normal deployment | `VERIFY_REPAIR=1`, with rollback and monitoring retained |
| Active index during this rollout | `default`; v2 remains separately gated |

**Definition of done:** adjudication is complete; verifier calibration is
measured against that gold; the production-parity release and full-suite repair
comparison pass without regression; the defined canary passes; and repair is
enabled with monitoring and rollback intact.

This document is a companion to `rag-correctness-next-steps.html` and
`data/eval/release_release-1.jsonl`.
