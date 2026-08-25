# L1/L2 coverage review — an audit of the coverage claim

**Target:** `final_project_qna`, `main @ 4db86c2`.
**Method:** record only. No API calls, no live probes, no git commands. Every number below
is computed from files in the working tree and is reproducible from them.
**Claim under audit:** Act XIII of `docs/build-report.html` marks levels **L1 (explicit facts)**
and **L2 (implicit facts)** of the Zhao et al. taxonomy (arXiv:2409.14924) `covered`; the
State-&-next section restates it flatly as "the L1/L2 query levels of the Zhao taxonomy are
covered (Act XIII)". The evidence offered is an 89-case gold set
(`scripts/answer_gold.jsonl`) and releases 3–7 (`data/eval/release_release-{3,4,4-off,5,6,7}.jsonl`).

A concurrent agent is live-probing several suspected gaps. Shapes that depend on live behaviour
were marked **probe pending** until the live probes landed; §7 carries the merged results.

---

## 0. Executive summary

| Finding | Evidence |
| --- | --- |
| The gold set maps cleanly onto 14 L1/L2 sub-shapes: **49 L1 cases, 40 L2 cases** | §1 |
| Release-7 behaviour: **83/89 at 1.00**, mean 0.9819. L1 mean 0.9746 (45/49 at 1.00), L2 mean 0.9908 (38/40) | §2 |
| **73/89 cases are always-1.00; 12 flap; 4 are standing defects.** But only **66/89** cases exist in release-3, **69/89** in release-5, and **20/89 have exactly two runs and 3 have exactly three** | §2 |
| **Zero same-configuration repeat runs exist in releases 3–7.** Every pair differs in `git_sha` and/or `RERANK`. "Sampling flapper" is therefore not separable from "code regression" in this evidence base | §2.3 |
| Claim mass: **173 claims / 89 cases**, 167 supported (96.5%), 0 contradicted, 6 unsupported. Mass is concentrated in S11 conflict (28) and S8 comparison (16); **S6 carries 1 claim from 1 case** | §3 |
| Evidence-page retrieval is the weak half: **26/36 target pages in release-7**, conflict class **8/14** — unchanged from release-6 | §2.5, §5 |
| The conflict capability is **entirely note-carried**: all 10 conflict-behaviour cases have a `Registry — CONFLICT` line; 3 of them score 1.00 while retrieving **0/2** of their target pages | §1.3, §4-H14 |
| **22 uncovered shapes** enumerated, 11 rated likely-fails (12 counting H9's two forms separately) | §4 |
| **6 findings are on no residue list**, including a false-authority note collision that no gold case can see | §5 |

**The single sharpest structural point.** The L1/L2 verdict is a *capability* argument, not a
measurement. Act XIII maps system components to levels; nothing in the report assigns a gold case
to L1 or L2, and no denominator is attached to the word "covered" — unlike every other coverage
number in the report (field 71/71, adjudication 71/71, page-level 64%). The gold set touches
**27 of 273 corpus documents (9.9%)**, with zero coverage of 2015, 2017 and 2019 (`data/registry.json`).
A shape inventory is the missing half of the claim, and this document is it.

**The three highest-severity holes** (details and code citations in §4):

1. **H5 — section-code collision.** `chainlit_app._board_range_note` matches `\bb\.?\s?(\d{1,2})\b`.
   "What does section **B.3** of FP151 say?" therefore injects
   *"B.3 is not in this corpus … State this definitively."* into a perfectly answerable turn.
   Collides on B.1–B.10, which are the standard GCF funding-proposal section headings.
   No gold case is section-scoped, so this is invisible to the suite.
2. **H1/H2 — inverse lookups.** `registry.py` exposes `by_fp`, `by_year`, `resolve_board_code`
   and nothing else. There is no `by_entity` and no `by_country`, and
   `retrieve.identifier_tokens` routes only on `fp\d{2,3}|b\.?\d{2}|add\.?\d{2}`. "Which proposals
   does UNDP implement?" and "Which proposals are in Kenya?" have **no mechanism at all**, and
   `prompts.CORE` forbids stating corpus-wide lists from a retrieved sample.
3. **H7/H16 — the single-document / in-range-board asymmetry.** `registry_v2.json` holds 19 fact
   fields over 273 documents; `registry._fmt` prints **5**. The planner matrix reaches 15 — but
   `planner.detect` returns `None` below two identifiers. So "compare the implementation periods of
   FP220 and FP203" gets a complete matrix and "what is the implementation period of FP220?" gets
   raw chunks. Symmetrically, `_board_range_note` gives an out-of-range board a definitive answer
   (`abs-b44`, `abs-b45`) while an in-range board question ("which proposals were approved at B.35?")
   fires no note at all.

---

## 1. Shape inventory of the 89-case gold set

### 1.1 The sub-shape map

Assignment is by *what the query demands of the data*, per the taxonomy's own criterion — L1 when
the answer is stated verbatim in a retrievable passage, L2 when it must be assembled across
documents or computed over the collection. Language (`fr`) and input degradation (`noisy`) are
treated as **attributes**, not shapes: they are the same demand on the data through a harder
channel, and folding them into their own classes is exactly what hides that S3, S4, S6, S8 and S10
have no French twin (§4-H11).

| # | Lvl | Sub-shape | n | Gold class origin |
| --- | --- | --- | --: | --- |
| S1 | L1 | id → one stated fact (money) | 9 | identifier 4, compact-id 1, french 2, noisy 2 |
| S2 | L1 | id → one stated fact (entity / country) | 7 | identifier 3, compact-id 1, french 1, noisy 2 |
| S3 | L1 | id → several facts, one document | 7 | identifier 5, compact-id 2 |
| S4 | L1 | board code → document | 5 | board-code 5 |
| S5 | L1 | description → the one document (discovery) | 10 | discovery 8, french 1, noisy 1 |
| S6 | L1 | description → a **set** of documents | 1 | discovery 1 |
| S7 | L1 | closed-world abstain | 10 | abstain 8, board-code 1, french 1 |
| S8 | L2 | cross-doc comparison, rankable | 4 | comparison 4 |
| S9 | L2 | cross-doc comparison, cross-currency refusal | 4 | comparison 3, french 1 |
| S10 | L2 | merge trap (twin documents) | 3 | comparison 3 |
| S11 | L2 | within-document conflict | 10 | conflict 7, french 2, noisy 1 |
| S12 | L2 | per-year aggregate | 8 | aggregate 6, french 2 |
| S13 | L2 | corpus-wide aggregate | 5 | aggregate 4, french 1 |
| S14 | L2 | follow-up reference resolution | 6 | followup 6 |
| | | **L1 total** | **49** | |
| | | **L2 total** | **40** | |

### 1.2 Shape × mechanism × outcome (release-7)

Mechanism is read from each release row's own `notes_used` keys plus `guard` / `matrix` /
`decomposed` flags — i.e. from what the turn actually held, not from inference.
`pages` is `retrieval.pages_hit / retrieval.pages_expected` summed over the shape.

| # | n | r7 mean | at 1.00 | runs/case | always-1.00 | flap | defect | claims | sup | unsup | pages | mechanism that answered it |
| --- | --: | --: | --: | --- | --: | --: | --: | --: | --: | --: | --- | --- |
| S1 | 9 | 1.000 | 9/9 | 2–6 | 9 | 0 | 0 | 18 | 18 | 0 | 8/9 | registry note ×9 |
| S2 | 7 | 1.000 | 7/7 | 6 | 7 | 0 | 0 | 7 | 7 | 0 | n/a | registry note ×7 |
| S3 | 7 | 1.000 | 7/7 | 2–6 | 7 | 0 | 0 | 23 | 23 | 0 | 2/2 | registry note ×7 |
| S4 | 5 | 0.960 | 4/5 | 6 | 4 | 1 | 0 | 10 | 10 | 0 | n/a | registry note ×5 |
| S5 | 10 | 1.000 | 10/10 | 2–6 | 10 | 0 | 0 | 16 | 16 | 0 | n/a | **retrieval only ×10** |
| S6 | 1 | **0.600** | 0/1 | 6 | 0 | 0 | **1** | 1 | 1 | 0 | n/a | **retrieval only ×1** |
| S7 | 10 | 0.936 | 8/10 | 2–6 | 7 | 1 | **2** | 9 | 7 | **2** | n/a | registry 2, registry+guard 2, board+year 2, year 2, **retrieval only 2** |
| S8 | 4 | 1.000 | 4/4 | 6 | 4 | 0 | 0 | 16 | 16 | 0 | 1/1 | planner matrix + registry ×4 |
| S9 | 4 | 1.000 | 4/4 | 6 | 2 | 2 | 0 | 8 | 8 | 0 | n/a | planner matrix + registry ×4 |
| S10 | 3 | 0.944 | 2/3 | 2–6 | 0 | 2 | **1** | 6 | 6 | 0 | 0/1 | planner matrix + registry ×3 |
| S11 | 10 | 1.000 | 10/10 | 2–6 | 9 | 1 | 0 | 28 | 28 | 0 | **13/20** | registry conflict lines ×10 |
| S12 | 8 | 0.975 | 7/8 | 2–6 | 6 | 2 | 0 | 11 | 8 | **3** | n/a | registry + year note ×8 |
| S13 | 5 | 1.000 | 5/5 | 2–6 | 4 | 1 | 0 | 7 | 7 | 0 | n/a | coverage/year note ×5 |
| S14 | 6 | 1.000 | 6/6 | 2–6 | 4 | 2 | 0 | 13 | 12 | **1** | 2/3 | registry note (incl. `_extend_registry_note`) ×6 |

Roll-up:

| Level | n | r7 mean | at 1.00 | always-1.00 (all its runs) | claims | supported | pages | present in release-3 |
| --- | --: | --: | --: | --: | --: | --: | --: | --: |
| L1 | 49 | 0.9746 | 45/49 | 44/49 | 84 | 82 (97.6%) | 10/11 | 40/49 |
| L2 | 40 | 0.9908 | 38/40 | 29/40 | 89 | 85 (95.5%) | 16/25 | 26/40 |

Note the inversion worth flagging: **L2 scores higher than L1 in release-7** (0.9908 vs 0.9746) but is
far less stable historically (always-1.00 29/40 vs 44/49) and has markedly worse evidence-page
retrieval (16/25 vs 10/11). The L2 number is newer and thinner, not stronger.

### 1.3 What actually answers each shape — the mechanism concentration

Across release-7's 89 cases (a case may hold more than one):

| Mechanism | cases | source |
| --- | --: | --- |
| `registry.registry_note` (FP id / board code / bare year) | 67 | `notes_used.registry` |
| `_year_assist` **or** `_corpus_coverage_note` (both land in `notes_used.year`) | 17 | `notes_used.year` |
| conductor decomposition | 14 | `decomposed: true` |
| **no note at all — pure chunk retrieval** | **13** | `notes_used == {}` |
| `planner` evidence matrix | 11 | `notes_used.matrix` |
| FP-miss guard (short-circuits before retrieval) | 2 | `guard: true` |
| `_board_range_note` | 2 | `notes_used.board` |

Two consequences the coverage claim has to live with:

- **75% of the suite (67/89) is carried by one function.** `registry_note` fires on exactly three
  triggers — an FP token, a board code, a bare `20[12]\d` year (`registry.py`). A user query that
  contains none of the three drops to the 13-case evidence base of pure retrieval.
- **The conflict class does not test conflict retrieval.** All 10 S11 cases carry a
  `Registry — CONFLICT in this document …` line printed by `registry._conflict_lines`, which already
  states both figures with both pages. `conf-fp153-gcf`, `conf-fp251-gcf` and `conf-fp201-gcf` score
  **1.00 while retrieving 0 of 2 target pages**. The measured capability is "the model copies a
  correct note", not "the system finds a contradiction".

---

## 2. Score stability across releases 3–7

### 2.1 The runs are not the same size

| release | rows | gold cases missing | notable config |
| --- | --: | --: | --- |
| release-3 | 66 | 23 | `RERANK` unset |
| release-4 | 66 | 23 | `RERANK=1` |
| release-4-off | 66 | 23 | `RERANK=0` (A/B arm of release-4, same `git_sha`) |
| release-5 | 69 | 20 | `RERANK=1`, coverage note added (`bd74abf`) |
| release-6 | 89 | 0 | 20 blind-authored cases added (`b5093c2`) |
| release-7 | 89 | 0 | `registry_v2` rebuilt (`meta_provenance`) |

So "stable across releases 3–7" holds for at most **66 of 89 cases**. Twenty of the 23 newest cases
(`abs-2026`, `abs-b45`, `agg-2022-count`, `agg-year-most`, `conf-fp201-gcf`, `conf-fp251-gcf`,
`conf-fp265-gcf`, `fr-agg-2018`, `fr-fp251-conflict`, `id-fp234-entity`, `id-fp246-financing`,
`noisy-typo-fp246`, `txt-cmp-fp012-fp074-country`, `txt-cmp-fp237-fp195-entity`,
`txt-disc-ews-timor`, `txt-disc-geothermal-indonesia`, `txt-disc-mangrove-ecuador`,
`txt-fu-fp171-total-fr`, `txt-fu-fp195-entity`, `txt-noisy-ews-timor`) have **two runs each**; the
other three (`agg-corpus-years`, `agg-corpus-total`, `fr-agg-corpus-boards`) have three.

**Verbatim consequence for the claim.** Of the 40 L2 cases, only 26 exist in release-3. The L2
"covered" verdict in Act XIII was written on the 69-case set (release-4/5) — *before* the 20
blind cases landed — and Act XIV, which added them, immediately recorded a new L2-shaped shortfall
(conflict-class page retrieval). The verdict precedes its strongest counter-evidence.

### 2.2 Per-case sequences

`—` = case absent from that release. Score is `checks.score`.

| case | shape | 3 | 4 | 4-off | 5 | 6 | 7 | verdict |
| --- | --- | --: | --: | --: | --: | --: | --: | --- |
| *the other 73 cases* (44 L1 + 29 L2) | — | — | — | — | — | — | — | **always 1.00** in every release they appear in — 53 of them over 6 runs, 3 over 3 runs, 17 over 2 runs |
| `bc-b30-02-add03-trap` | S7 | 0.83 | 0.83 | 0.83 | 0.83 | 0.50 | **1.00** | flapper, 1/6 at 1.00 |
| `bc-b27-02-add12` | S4 | 0.80 | 1.00 | 1.00 | 0.80 | 0.80 | **0.80** | flapper, 2/6 at 1.00 |
| `disc-subnational-pair` | S6 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 | **0.60** | **standing defect, 6 runs** |
| `cmp-fp086-fp220-currency` | S9 | 0.83 | 0.83 | 1.00 | 1.00 | 1.00 | 1.00 | flapper, 4/6 |
| `cmp-fp151-fp152-entity` | S10 | 0.67\* | 0.86 | 0.86 | 0.86 | 0.86 | **1.00** | flapper, 1/6 (\*different fixture, §2.4) |
| `cmp-fp254-fp248-currency` | S9 | 1.00 | 1.00 | 0.86 | 1.00 | 1.00 | 1.00 | flapper, 5/6 — the one `RERANK` casualty |
| `conf-fp274-consistency` | S11 | 1.00 | 0.80 | 0.80 | 1.00 | 1.00 | 1.00 | flapper, 4/6 |
| `fr-agg-2020` | S12 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | flapper, 5/6 |
| `agg-corpus-boards` | S13 | 0.71 | 0.43 | 0.43 | 1.00 | 1.00 | 1.00 | flapper, 3/6 — closed by `bd74abf` |
| `fu-compare-those` | S14 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | flapper, 5/6 |
| `fu-lang-switch` | S14 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | flapper, 5/6 — closed by `8203913` |
| `abs-offtopic` | S7 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | **0.86** | **standing defect, 6 runs** |
| `abs-antarctica` | S7 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | **0.50** | **standing defect, 6 runs** |
| `txt-cmp-fp237-fp195-entity` | S10 | — | — | — | — | 0.83 | **1.00** | flapper, 1/2 |
| `txt-cmp-fp012-fp074-country` | S10 | — | — | — | — | 0.83 | **0.83** | **standing defect, 2 runs** |
| `fr-agg-2018` | S12 | — | — | — | — | 1.00 | **0.80** | flapper, 1/2 — **regression in release-7** |

Aggregate: **73 solid / 12 flappers / 4 standing defects** out of 89.

### 2.3 The independence problem — the finding that most weakens "stable for N runs"

Every release row carries `harness` and `ambient_env`. Across all six:

- identical `harness`: `{"model_alias":"gpt-5.2","temperature":0.0,"seed":20260819,"top_k":10,"max_answer_tokens":1024}`;
- identical index (`index_chunks_sha256 = a2a6c5da…` in all six) and identical v1 registry (`559a3318…`);
- `git_sha` **differs in every pair**: `85a9ff7f` → `a8efbd78` (×2) → `bd74abf1` → `b5093c26` → `1d2905f6`;
- `RERANK` differs: unset (r3), `1` (r4), `0` (r4-off), `1` (r5/6/7);
- `registry_v2_sha256` changes once, at release-7 (`3476e554…` → `162ea177…`).

**There is no pair of releases in 3–7 run at the same configuration.** Release-4 vs release-4-off
comes closest and is deliberately an A/B on `RERANK`. That means the record cannot distinguish
run-to-run sampling variance from code drift for any of the 12 flappers. Act XIV calls
`bc-b30-02-add03-trap` and `bc-b27-02-add12` "sampling flappers" with "no code change attached",
but a code change *is* attached to every interval in the series — just not one whose author
believed it touched those cases. The only true same-configuration repeats in `data/eval/` are
`release_spread-a`/`-b` (`git_sha 1e8953cf`) and `release_spread-c`/`-d` (`git_sha 515eb392`),
both on the older 66-case set at a much earlier sha.

The `RERANK` A/B is the one interval where drift is isolated, and it moves two cases in **opposite**
directions: `cmp-fp086-fp220-currency` 0.83 (`RERANK=1`) → 1.00 (`RERANK=0`), and
`cmp-fp254-fp248-currency` 1.00 → 0.86. Both are S9. A one-switch flip that costs one case and buys
another is the size of the noise floor this suite is being read against.

### 2.4 Fixture drift, and the second instrument

The gold file's `expect` blocks were compared byte-for-byte against each release's embedded `expect`:
**one drift only** — `cmp-fp151-fp152-entity` in release-3, whose forbid regex was repaired on
2026-08-24 (`e371308`; the gold notes record it). Its 0.67 is therefore **not comparable** with the
0.86/1.00 that follow; releases 4–7 all match the current fixture exactly. That is a good result
and should be stated as such.

`checks.score` and claim support are two instruments and they disagree on three cases that
**pass at 1.00 while carrying unsupported claims**:

| case | shape | checks.score | claims | reason recorded in the artifact |
| --- | --- | --: | --- | --- |
| `agg-2020-largest` | S12 | 1.00 | 2, 1 unsupported | "Evidence lists 2020 registry items, not excerpt dates or basis" — a meta-sentence about the answer's own basis |
| `agg-2021-boards` | S12 | 1.00 | 3, 2 unsupported | "Evidence doesn't mention 'B.28 (2021)' explicitly" — board→year inference from `YEAR_BLOCK`, which the verifier cannot see |
| `fu-compare-those` | S14 | 1.00 | 6, 1 unsupported | "Evidence gives 150M and 18.5M, not 131.5M difference" — the derived-arithmetic limitation |

Only the third is on a residue list (§5).

### 2.5 Evidence-page retrieval (by gold class, the unit the release records)

`retrieval.pages_hit / pages_expected`, 36 target pages over 26 cases that assert one:

| class | release-6 | release-7 |
| --- | --- | --- |
| identifier | 6/6 | 6/6 |
| compact-id | 1/1 | 1/1 |
| conflict | **8/14** | **8/14** |
| french | 5/6 | 5/6 |
| noisy | 2/4 | 3/4 |
| comparison | 1/2 | 1/2 |
| followup | 1/3 | 2/3 |
| **total** | **24/36 (66.7%)** | **26/36 (72.2%)** |

Document recall@5 is 66/67 in both. The conflict shortfall is **exactly unchanged** between
release-6 and release-7 — `conf-fp153-gcf` 0/2, `conf-fp251-gcf` 0/2, `conf-fp201-gcf` 0/2,
`fr-fp251-conflict` 1/2 — while every one of those cases scores 1.00.

---

## 3. Claim-level coverage per shape

Release-7 totals: **173 claims from 89 cases** (1.94/case), 167 supported (**96.5%**),
0 contradicted, 6 unsupported, 171 cited (98.8%), grounded 163/173 (94.2%), judge budget
exhausted 0/89, `judge_candidates` 15 in total. Three cases produce zero claims by design
(`abs-fp999` and `fr-fp999-abstain` are guard answers that return before verification;
`abs-offtopic` is a chat-mode turn that builds no evidence — recorded verbatim in `claims_skipped`).

| # | shape | cases | claims | claims/case | supported | unsupported | support rate | evidence weight |
| --- | --- | --: | --: | --: | --: | --: | --: | --- |
| S11 | within-doc conflict | 10 | 28 | 2.80 | 28 | 0 | 100% | **heaviest** |
| S3 | id → several facts | 7 | 23 | 3.29 | 23 | 0 | 100% | heavy |
| S1 | id → one money fact | 9 | 18 | 2.00 | 18 | 0 | 100% | heavy |
| S5 | discovery | 10 | 16 | 1.60 | 16 | 0 | 100% | heavy |
| S8 | rankable comparison | 4 | 16 | 4.00 | 16 | 0 | 100% | heavy per case, 4 cases |
| S14 | follow-up | 6 | 13 | 2.17 | 12 | 1 | 92.3% | moderate |
| S12 | per-year aggregate | 8 | 11 | 1.38 | 8 | 3 | **72.7%** | **worst support rate** |
| S4 | board code → doc | 5 | 10 | 2.00 | 10 | 0 | 100% | moderate |
| S7 | closed-world abstain | 10 | 9 | 0.90 | 7 | 2 | 77.8% | **thin for 10 cases** |
| S9 | cross-currency refusal | 4 | 8 | 2.00 | 8 | 0 | 100% | moderate |
| S2 | id → one entity fact | 7 | 7 | 1.00 | 7 | 0 | 100% | thin per case |
| S13 | corpus-wide aggregate | 5 | 7 | 1.40 | 7 | 0 | 100% | moderate |
| S10 | merge trap | 3 | 6 | 2.00 | 6 | 0 | 100% | **thin — 3 cases** |
| S6 | description → set | **1** | **1** | 1.00 | 1 | 0 | 100% | **weakest: 1 case, 1 claim, 0.60 for 6 runs** |

Recurring failure kinds, all six of them:

| kind | n | shape | pattern |
| --- | --: | --- | --- |
| `entity` unsupported | 3 | S7 ×2, S12 ×1 | negative existence statements and meta-sentences about the answer's own basis; the extractor mints them as claims and no evidence key can entail them |
| `year` unsupported | 2 | S12 | "B.28 (2021)" — the board→year table lives in `prompts.YEAR_BLOCK`, not in evidence, so the mapping is unverifiable by construction |
| `money` unsupported | 1 | S14 | derived arithmetic (150 − 18.5 = 131.5) |

**Thinness verdicts.** S6 is one case carrying one claim and it has failed in six consecutive runs —
"multi-document discovery" is not evidenced at all, it is a known open defect with a sample size of 1.
S10 (merge trap) is three cases / six claims, and is the shape whose whole purpose is to catch the
most expensive failure mode this corpus has (twin proposals). S2 averages exactly one claim per
case — a single regex per case is a weak instrument for an entity fact whose registry string is
unnormalised (§4-H1). S7 has the second-worst support rate on the thinnest claim base, and two of
its ten cases are permanent 0.50/0.86 failures.

---

## 4. The uncovered-shape list

Every entry states the mechanism from the code, with the file and function named. Severity:

- **HIGH — likely-fails**: no mechanism, or a mechanism that actively injects a wrong authoritative note.
- **MED — probably works, unmeasured**: a mechanism plausibly covers it but zero gold cases measure it.
- **LOW — works by construction, unmeasured.**

### 4.1 The mechanism inventory these verdicts rest on

Read from the code, not inferred:

| Mechanism | Trigger | Reaches |
| --- | --- | --- |
| `registry.registry_note` | FP token (`fp[\s\-]?0*(\d{1,3})(?!\d)`), board code (`b.NN[/.-][NN/.-]add.NN`), bare `20[12]\d` | title, accredited entity, countries, `gcf_funding_requested`, `total_financing`, board+year, **and nothing else** — 5 of the 19 fact fields in `registry_v2.json` |
| `registry._conflict_lines` | as above | `gcf_funding_requested`, `total_financing`, `co_financing` only (`_MONEY_FIELDS`), max 2 lines × 2 alternates |
| `chainlit_app._year_assist` | `_scan_years`: bare years + 7 range forms | complete per-year FP listing with `gcf_financing` strings when ≤3 years; counts + FP span above that |
| `chainlit_app._board_range_note` | `\bb\.?\s?(\d{1,2})\b` where the number ∉ `BOARD_YEARS` (11–43) | a definitive "not in this corpus" |
| `chainlit_app._corpus_coverage_note` | corpus token **and** coverage/size ask **and** no year **and** no board number | board span + 273 total + per-year counts |
| `planner.detect` → `build_matrix` | **≥ 2 identifiers** | 15 fields from `FIELD_ORDER`, each cell page- and section-cited, with a comparability verdict |
| `retrieve` identifier routing | `fp\d{2,3}\|b\.?\d{2}\|add\.?\d{2}` | per-document BM25 heads + quotas |
| `chainlit_app._extend_registry_note` | documents this turn resolved (not the question's words) | the same `_fmt` line, for follow-ups |
| lexical/dense index | any text | `section_path` is embedded and lexically indexed (`lexical.py`, `test_section_retrieval.py`) |

Lookups that **do not exist**: `by_entity`, `by_country`, `by_board`, `by_theme`, any inverse index,
any cross-cell arithmetic (`planner`'s docstring: "There is deliberately no 'calculated' status").

### 4.2 The holes

| # | Shape a real user would ask | Mechanism analysis (from code) | Sev |
| --- | --- | --- | --- |
| **H1** | **Entity-inverse** — "Which proposals does UNDP implement?" / "List every proposal from IFAD." | No `by_entity` in `registry.py`; `registry_note` has no entity trigger; `identifier_tokens` does not route on names. Falls to open hybrid retrieval over 273 docs capped at 15 hits, under a `CORE` rule that forbids corpus-wide lists from a sample. Worse, the store itself is unnormalised: 127 distinct entity strings, with UNDP under at least *"United Nations Development Programme"* (24) and *"United Nations Development Programme (UNDP)"* (10), FAO under three spellings. Even the mechanism's data prerequisite is missing. **Probed (P1/P10/F12, §7): missing but honest** — 3 of 41 named with an explicit incompleteness statement in EN; the FR variant named 6 of 13 with one category error and no explicit incompleteness statement. | **HIGH** |
| **H2** | **Country-inverse** — "Which proposals are in Kenya?" | Same absence. Unlike H1 the data is clean: `registry.json` carries `countries` as arrays, 178 distinct values, Kenya in 25 documents. A `by_country()` note would be a ~10-line addition mirroring `by_year`. Today: retrieval only. **Probed (P2, §7): missing, honesty WEAK** — 6 of 25 named with only per-item caveats; no statement that the list is incomplete. | **HIGH** |
| **H3** | **Corpus-wide extremum** — "What is the smallest GCF request in the corpus?" / "the largest ever?" | `_corpus_coverage_note` prints counts, no money. `CORE` forbids corpus-wide superlatives outright. `agg-2020-largest` passes only because `_year_assist` prints per-FP `gcf_financing` for ≤3 years — that path needs a year token the question does not have. | **HIGH** |
| **H3b** | **Single-year MIN** — "Which 2020 proposal requested the least?" | Identical mechanism to the tested `agg-2020-largest` MAX. Complication: 8 of 30 2020 rows carry no figure, so a min over stated values is well-defined but a naive min is not. Untested. | **MED** |
| **H4** | **Cross-year arithmetic** — "Did 2020 request more GCF funding than 2021 combined?" | `_scan_years` yields both years; detailed mode (≤3 years) prints per-FP figures for both. But nothing sums: the planner never fires (no FP identifiers), and `planner` refuses cross-cell arithmetic by design. The model would have to add 22 of 30 and 26 of 28 raw strings spanning USD and EUR, one of which (`28,654 million USD`, FP153) the registry itself flags as unit-ambiguous. A confidently wrong total is the likely output. | **HIGH** |
| **H5** | **Section-scoped ask** — "What does section B.3 of FP151 say?" | `_board_range_note`'s `\bb\.?\s?(\d{1,2})\b` matches `B.3`; 3 ∉ `BOARD_YEARS` → the turn receives *"Note (computed): B.3 is not in this corpus, which covers board meetings B.11 (2015) through B.43 (2025) completely. **State this definitively.**"* Retrieval will meanwhile do fine — `section_path` is indexed. So the failure mode is a definitive false denial sitting on top of correct excerpts. Window: B.1–B.10, i.e. the standard proposal headings (B.1 description, B.2 project details, B.3 rationale). **Zero gold cases are section-scoped, so the suite is structurally blind to it.** | **HIGH** |
| **H6** | **Multi-fact single-doc** — "Give me FP151's entity, countries and total." | S3 covers this shape with 7 cases at 1.00 — but every one stays inside the 5 fields `registry._fmt` prints. The moment a sixth field is asked the case becomes H7. | **MED** |
| **H7** | **Non-money single-doc field** — implementation period, lifespan, ESS category, project size, instruments, beneficiaries, mitigation/adaptation outcome, co-financing | `registry_v2.json` holds these for 38–236 of 273 documents. `_fmt` prints none of them. `planner.FIELD_ORDER` covers all of them — but `planner.detect` returns `None` below two identifiers. **The same question is structured with two ids and unstructured with one.** Zero gold cases touch any of these fields: `expect.fields` across all 89 cases is only `{gcf_financing 32, accredited_entity 15, countries 5, title 4, total_financing 3}`. | **HIGH** |
| **H8** | **Fields with no structured path at all** — "Who is the executing entity of FP220?" / "Which NDA endorsed it?" | `executing_entity` (193/273) and `national_designated_authority` (101/273) exist in `registry_v2` but appear in neither `_fmt` nor `planner.FIELD_ORDER`. No mechanism reaches them at any arity. | **MED-HIGH** |
| **H9** | **Date / duration** — "When was FP220 approved?" / "on what date?" | Board+year is on the `_fmt` line and `YEAR_BLOCK` licenses treating the board year as the approval year, so the *year* form probably works. A calendar date has no field anywhere. No gold case asks either. | **MED** (year) / **HIGH** (date) |
| **H10** | **Id-format variants outside the noisy class** — `FP#220`, `FP no. 220`, `FP.220`, "proposal 220", "funding proposal number 220" | Verified against `_FP_RE` directly: `FP-220` → `220` ✓, `FP 0086` → `86` ✓, but `FP#220` → ∅, `FP.220` → ∅, `FP no. 220` → ∅, bare "proposal 220" → ∅. A miss is silent: no registry note, no identifier routing, no guard — the turn degrades to open retrieval with no signal that anything was lost. The noisy class (6 cases) tests only forms that *do* match. | **MED-HIGH** |
| **H11** | **French twins of untested shapes** | 13 cases carry French text. They cover S1, S2, S5, S7, S9, S11, S12, S13, S14. **No French case exists for S3 (multi-fact single doc), S4 (board code), S6, S8 (rankable comparison) or S10 (merge trap).** S4 is the sharpest: `resolve_board_code` normalises punctuation, and no French phrasing of a board code has ever been run. | **MED** |
| **H12** | **Theme count** — "How many proposals concern agriculture?" | Two failure modes, both verified by running the regexes. Without a corpus token: nothing fires, `CORE` forbids the count, expect a refusal. **With** one ("how many proposals *in the corpus* concern agriculture?"): `_COVERAGE_ASK_RE` matches `how many (\w+ ){0,2}proposals` and `_CORPUS_TOKEN_RE` matches, so the coverage note fires and ends *"Answer corpus-coverage questions from this note"* — an authoritative note that holds only per-year counts being handed to a thematic question. False-authority misfire, same family as H5. | **HIGH** |
| **H13** | **Within-doc conflict on a non-GCF-request field** — "Are FP251's total financing figures consistent?" | `registry_v2` records a conflicting candidate for `total_financing` in **92 of 273** documents versus **53** for `gcf_funding_requested`. The commonest conflict in the corpus is the one no gold case asks about — all 10 S11 cases ask `gcf_funding_requested`. `_MONEY_FIELDS` does include `total_financing`, so the note fires; nothing measures it. | **MED** |
| **H13b** | **Conflict on a non-money field** | `registry_v2` records conflicting candidates in `implementation_period` (1 doc), `co_financing` (3), `mitigation_outcome` (1), `adaptation_outcome` (1), `beneficiaries_direct` (1). `_conflict_lines` iterates `_MONEY_FIELDS` (`gcf_funding_requested`, `total_financing`, `co_financing`) only, so **four of those five** — everything but `co_financing` — can never produce a warning. Rare, but silent. | **MED** |
| **H14** | **A conflict the registry did not record** | All 10 S11 cases have a `CONFLICT` note line. Conflict-class page retrieval is **8/14**, with three cases at 0/2. So the fallback path — find the contradiction in the excerpts — is not merely untested, it is measurably unable to reach the pages it would need. The gold set's own notes concede the adjacent case: `conf-fp267-gcf` was written when "the registry currently records only 46.10", yet the release-7 note now prints three figures for it, so even that case no longer exercises the fallback. | **HIGH** |
| **H15** | **Enumerating a set** — "Which proposals make up the Subnational Climate Fund?" | S6, `disc-subnational-pair`, is the only case, and it has scored **0.60 in all 6 runs**: the answer names the programme and both entities but neither `FP151` nor `FP152`. One case, one claim, six consecutive failures. Any "which proposals …" question is this shape. | **HIGH** (known) |
| **H16** | **In-range board-inverse** — "Which proposals were approved at B.35?" | `_board_range_note` fires only for boards **outside** 11–43. `registry_note` needs a full code with an addendum. `registry.json` carries `board` per row but there is no `by_board()`. So the out-of-range arm gets a definitive answer (`abs-b44`, `abs-b45` both 1.00) and the in-range arm gets nothing. `agg-2021-boards` tests year→boards, the opposite direction. | **HIGH** |
| **H17** | **Aggregate over a range phrase** — "proposals since 2022", "approved after 2023", "between 2019 and 2021", "before 2015" | `_scan_years` implements seven range forms plus `_outside_corpus_note`, with a long docstring recording a measured regression it fixed. **Zero gold cases exercise any active form.** The three cases containing a range word all say "from 2020", which the code deliberately treats as *not* a range. A documented, unit-tested mechanism (`tests/test_quickfixes.py`, `tests/test_coverage_note.py`) with no end-to-end case. | **HIGH** (unmeasured mechanism) |
| **H18** | **Follow-up over a conflict, or a mid-thread document switch** | S14's 6 cases cover pronoun, unrelated-interleave, two-entity "their", language switch, and two `_extend_registry_note` cases. None follows up on a conflict answer ("and which of those two figures is on the cover page?") and none switches document mid-thread while reusing a pronoun. | **MED** |
| **H19** | **Cross-document contradiction about the same proposal** | The corpus holds package documents and status/addendum documents for the same FP (e.g. FP086's status doc `189_…-respect-fp086-…`). `_conflict_lines` is scoped to conflicts *within* one document (`CONFLICT in this document`). A disagreement between two documents about one proposal has no detector and no case. | **MED** |
| **H20** | **Comparison beyond three documents** | `cmp-three-way` is the widest case. `planner.detect` has no arity cap, so a five-way matrix would build; retrieval fan-out and the 15-hit cap are the risk, and `cmp-three-way`'s own note says it exists to test "per-document fan-out starvation". Untested above 3. | **LOW-MED** |

### 4.3 Severity roll-up

22 entries, scored:

- **Likely-fails — 11** (12 counting H9's date form separately): H1, H2, H3, H4, H5, H7, H12, H14, H15, H16, H17, plus H9 in its calendar-date form.
- **Probably-works-but-unmeasured — 10:** H3b, H6, H8, H9 (year form), H10, H11, H13, H13b, H18, H19.
- **Works-by-construction-but-unmeasured — 1:** H20.

Of the 11 likely-fails, **three are false-authority misfires** (H5, H12, and the H16 asymmetry) —
cases where the system does not merely fail to answer but injects a computed note labelled
authoritative into a turn that contradicts it. Those are the most expensive failures a
"never asserts what it cannot cite" system can have, and none of them is reachable from the current
gold set.

---

## 5. Reconciliation with the known-residue list

Act XIV lists six residue items. Reconciled against this audit at release-7:

| # | Residue item (Act XIV) | Status at release-7 | Note |
| --- | --- | --- | --- |
| 1 | Entity page provenance (2 cases) | **partially closed** | `txt-cmp-fp237-fp195-entity` 0.83 → **1.00** after the `meta_provenance` rebuild (`registry_v2_sha256` changed at release-7). `txt-cmp-fp012-fp074-country` remains 0.83 — but its `bad_citations` are now `p.5` and `p.4`, **not the `p.3` Act XIV describes**. The residue's description no longer matches the residue. |
| 2 | Derived arithmetic (1 claim) | **open, unchanged** | `fu-compare-those`, 1 unsupported money claim, "Evidence gives 150M and 18.5M, not 131.5M difference". Confirmed. |
| 3 | Conflict-class evidence-page retrieval | **open, exactly unchanged** | 8/14 in release-6 **and** release-7. Independently recomputed here from `retrieval.pages_hit`. Overall page coverage 24/36 → 26/36. |
| 4 | Two sampling flappers | **one closed, one open** | `bc-b30-02-add03-trap` 0.50 → **1.00** in release-7. `bc-b27-02-add12` still 0.80 (must_contain `re:Pegasus` unmet; the answer is factually correct and names FP152). |
| 5 | `disc-subnational-pair` at 0.60 | **open, now six runs** | Extended: 0.60 in releases 3, 4, 4-off, 5, 6 **and 7**. Act XIV said "release-3 through release-6". |
| 6 | Two content-abstention fixtures | **open, unchanged** | `abs-offtopic` 0.86 (behaviour check false — the refusal is correct prose that the abstain detector does not recognise); `abs-antarctica` 0.50 (behaviour false + must_contain miss; "None of the retrieved excerpts state…" matches none of the fixture's alternatives, and the answer additionally emits 10 citations and 2 unsupported claims). |

### 5.1 In this audit but on **no** residue list

1. **`fr-agg-2018` regressed 1.00 → 0.80 in release-7** — the `language` check fails. The answer is
   French but quotes the English registry note verbatim inside brackets. Newest release, newest
   regression, unlisted.
2. **The section-code collision (H5)** and **the theme-count coverage-note misfire (H12)** — two
   false-authority note paths readable directly from the regexes. Act XIV's closing boast is
   "nothing on this list was discovered by reading the code"; these two were, and they are not on it.
3. **Two unsupported-claim families not covered by residue item 2** — `agg-2021-boards`'s board→year
   claims (2 unsupported, unverifiable by construction because the mapping lives in `YEAR_BLOCK`,
   not in evidence) and `agg-2020-largest`'s basis meta-sentence. Both cases score 1.00 on
   `checks.score`, so they are invisible to the behaviour headline.
4. **Zero same-configuration repeat runs in releases 3–7** (§2.3) — which means residue item 4's
   "sampling flapper" designation is asserted, not measured, on this evidence base.
5. **Two stale gold-set notes.** `cid-fp0086-padded`'s note says *"KNOWN BUG (baseline expected to
   fail): registry.resolve_fps uses `r'fp\s?(\d{2,3})'`, so 'FP0086' captures '008'"*, and
   `noisy-hyphen-fp220`'s says *"KNOWN GAP: 'FP-220' does not match"*. The live pattern is
   `fp[\s\-]?0*(\d{1,3})(?!\d)` and handles both; both cases score 1.00 in 6/6 runs. Similarly
   `conf-fp267-gcf`'s *"the registry currently records only 46.10"* is contradicted by the
   release-7 note, which prints three figures. Derivation notes that no longer describe the code
   are how a suite quietly stops testing what it claims to test.
6. **`scripts/answer_gold.jsonl` is not checksummed.** `data/eval/CHECKSUMS.sha256` (226 lines)
   anchors all six releases and 60-odd other artifacts but not the fixture that defines them.
   Mitigating: every release row embeds its own `expect`, and a byte-comparison against the current
   gold file shows drift in exactly one case (§2.4) — so the anchor is recoverable, just not declared.

### 5.2 On a residue list but softer than stated

- Residue item 1 is scoped "2 cases"; item 4 is "two flappers"; item 6 is "two fixtures". Counting
  by case rather than by item, the six items cover **9–10 cases**. Two of those are now closed,
  which is genuine progress and should be claimed — but the item count and the case count are
  different denominators and the report uses only the smaller one.

---

## 6. What would strengthen the claim

### (a) Shapes to probe live **before asserting anything** — ordered

Each is a single query; none requires new code. Marked ★ where a concurrent probe may already cover it.

1. ★ **H5** — `What does section B.3 of FP151 say?` Confirm or refute the `_board_range_note`
   false denial. This is the highest-value probe in the list: it is a one-character-class regex bug
   with a definitive-assertion blast radius.
2. ★ **H12** — `How many proposals in the corpus concern agriculture?` Confirm whether the coverage
   note's "answer corpus-coverage questions from this note" instruction produces a per-year count in
   answer to a thematic question.
3. ★ **H1** — `Which proposals does UNDP implement?` and **H2** — `Which funding proposals are in Kenya?`
   Establish the actual degradation shape: honest hedge, partial list presented as complete, or refusal.
4. **H16** — `Which funding proposals were approved at B.35?` Measure the in-range/out-of-range asymmetry.
5. **H4** — `Did 2020 request more GCF funding than 2021 combined?` Determine whether the model sums
   the year note or refuses. Either answer is publishable; a wrong total is a finding.
6. **H17** — `Which funding proposals were approved after 2023?` and `…before 2015?` Exercise the
   range machinery end-to-end for the first time.
7. **H7** — `What is the implementation period of FP220?` vs
   `Compare the implementation periods of FP220 and FP203.` The pair measures the arity asymmetry directly.
8. **H10** — `What is FP#220 about?` and `Tell me about funding proposal number 220.`
9. **H3** — `What is the smallest GCF funding request in the corpus?`

### (b) Gold cases worth adding — ordered by evidence bought per case

| Priority | Case | Shape | Why it earns a slot |
| --- | --- | --- | --- |
| 1 | Section-scoped ask on a live FP (`section B.3 of FP151`) | new L1 sub-shape | Closes the H5 blind spot permanently. The suite currently cannot see a whole query family. |
| 2 | `total_financing` conflict (one of the 92 documents) | S11 | The commonest conflict in the corpus is untested; S11's 10 cases all ask one field. |
| 3 | Entity-inverse and country-inverse, written as **expected-hedge** cases | new L2 sub-shape | Even before a mechanism exists, a case that pins "must scope to the retrieved excerpts, must not present a partial list as complete" turns an unknown into a measured behaviour. |
| 4 | Single-document non-money field (implementation period, ESS category, beneficiaries) | H7 | Three cases would cover the 10 planner fields the suite has never touched. |
| 5 | A range-phrase year aggregate (`after 2023`, `before 2015`) | S12 | Gives seven documented range forms their first end-to-end evidence. |
| 6 | A second and third S6 (multi-document discovery) case | S6 | One case at 0.60 for six runs is a defect report, not a shape measurement. |
| 7 | In-range board-inverse (`approved at B.35`) | S13/new | Balances `abs-b44`/`abs-b45`, which only test the easy arm. |
| 8 | French twins for S4 (board code) and S8 (rankable comparison) | H11 | Two cases close the two widest language gaps. |
| 9 | `FP#220` / "proposal 220" | S1/S2 noisy | Extends the noisy class to forms that actually miss `_FP_RE`. |
| 10 | Follow-up over a conflict answer | S14 | The two hardest L2 shapes have never been composed. |

Also: **add `scripts/answer_gold.jsonl` to `data/eval/CHECKSUMS.sha256`**, and repair the three
stale derivation notes (§5.1 item 5).

### (c) Mechanisms genuinely missing

| Priority | Mechanism | Shape closed | Cost signal |
| --- | --- | --- | --- |
| 1 | **Fix `_board_range_note`'s trigger** so a bare `B.n` inside a section reference cannot fire — require a board-code context or exclude `n < 11` when an FP token is present | H5 | Smallest change on this list, largest correctness gain |
| 2 | **Narrow `_corpus_coverage_note`** so a thematic "how many … proposals" does not receive a per-year-count note labelled authoritative | H12 | Trigger-only change |
| 3 | **`registry.by_country()` + a country note** mirroring `by_year` | H2 | Data is already clean: 178 distinct countries, arrays per row |
| 4 | **`registry.by_entity()` + entity alias normalisation** | H1 | Needs an alias table first — 127 raw strings, UNDP/FAO/IDB each under ≥2 spellings. The normalisation is the real work |
| 5 | **`registry.by_board()` + an in-range board note** | H16 | `board` already on every row; mirrors the year note exactly |
| 6 | **Let `registry._fmt` print the fields the question asked for**, using `planner.fields_for()` on a single-identifier turn | H6, H7, H8 | Reuses two existing components; removes the one-vs-two-identifier asymmetry without a new store |
| 7 | **Widen `_conflict_lines` beyond `_MONEY_FIELDS`** | H13b | Five documents affected today; cheap insurance |
| 8 | **A derived-arithmetic claim shape in the verifier** (operands + operator + result, all cited) | residue 2, and the H4 blocker | Already named as future verifier work in Act XIV |
| 9 | **Corpus-wide extremum note** (min/max over stated `gcf_financing` with a currency guard and an explicit "N of M rows state a figure" denominator) | H3, H3b | Must refuse across currencies, as the matrix already does |
| 10 | **Conflict retrieval as a fallback path** — a doc-scoped probe for the section that disagrees, so a conflict not in `registry_v2` can still be surfaced | H14, residue 3 | The 8/14 page shortfall is the measurement that justifies it |

---

## 7. What the record does support

Stated plainly, because an adversarial review that only subtracts is not calibrated:

- **L1 single-fact lookup by identifier is genuinely solid.** S1+S2+S3 = 23 cases, 48 claims, 48
  supported, **23/23 always-1.00 across every run they appear in**, page retrieval 10/11. Across four
  identifier spellings (`FP151`, `FP 86`, `FP-220`, `FP0086`) and two languages.
- **The cross-currency refusal is real and load-bearing.** S9's 4 cases have never produced a
  ranking across currencies in six releases, and the refusal survives translation (`fr-cmp-currency`).
  This is the single most defensible L2 claim in the suite.
- **Within-document conflict surfacing works, at the note level, without exception** — 10 cases,
  28 claims, 28 supported, 0 contradicted, across three languages and one degraded-input variant.
  The caveat in §1.3 is about *how* it works, not whether.
- **The blind-authoring discipline of release-6 paid for itself.** Twenty cases written without
  sight of the safety gates immediately exposed a page-level retrieval shortfall the 69-case set was
  structurally unable to see. That is the correct result from that exercise, and it is the reason
  this review can be specific about where the next twenty should go.

The honest form of the headline is therefore not "L1 and L2 are covered". It is:

> Over 89 cases, the system holds 0.982 behaviour and 96.5% claim support. Its L1 identifier path and
> its L2 conflict and cross-currency paths are stable across six recorded releases. Its coverage of
> the L1/L2 *space* is narrower than the level names imply: 27 of 273 documents, 5 of 19 structured
> fields, no inverse lookups, and at least eleven query shapes a user would plausibly ask that no case
> tests and that the code suggests would fail.

---

### Provenance

| Artifact | Used for |
| --- | --- |
| `scripts/answer_gold.jsonl` (89 lines) | shape inventory, fixture drift |
| `data/eval/release_release-{3,4,4-off,5,6,7}.jsonl` | scores, claims, mechanisms, retrieval, harness |
| `data/eval/CHECKSUMS.sha256` | integrity anchor; gold-set omission |
| `data/registry.json` (273 docs), `data/registry_v2.json` (273 docs, 19 fact fields) | field/entity/country/conflict inventory |
| `src/gcf_qna/rag/registry.py`, `planner.py`, `retrieve.py`, `lexical.py`, `boards.py` | mechanism inventory |
| `src/gcf_qna/app/chainlit_app.py`, `app/prompts.py` | note triggers, prompt constraints |
| `docs/build-report.html` Acts XIII–XIV, `docs/DEPLOYED.md` | the claim, the residue list |

The release-7 commit message could not be read: this review was run under a no-git-commands
constraint. Reconciliation in §5 is against Act XIV and the release artifacts, which carry
`git_sha 1d2905f6e87dab3a23d423044aa297cca085c522` for release-7.


## 7. Live probe results (merged from the concurrent probe agent)

Fifteen probes through the real production `Pipeline` (conductor + planner, pinned sampling,
`gpt-5.2-2025-12-11`), deterministic verifier pass, ≈$0.11. Raw transcripts:
`scratchpad/l1l2_probes.jsonl` (session scratchpad; verdicts reproduced here in full).

| # | Shape | Question | Verdict |
|---|---|---|---|
| P1 | H1 entity-inverse | UNDP proposals? | HONESTLY-SCOPED — 3 of 41 named, incompleteness stated, all 3 correct |
| P2 | H2 country-inverse | Kenya proposals? | HONESTLY-SCOPED (weak) — 6 of 25, per-item caveats only, never says incomplete |
| P3 | H3 corpus minimum | smallest GCF request? | HONESTLY-SCOPED — refused corpus-wide; truth FP2 = 16,265 |
| P4 | H4 cross-year compare | 2020 vs 2021 totals? | **WRONG** — "Yes, 2020" (truth: 2021 by 2×); verifier scored it supported |
| P5 | multi-fact scalar+list | FP151 entity/countries/total | entity+total correct incl. conflict disclosure; **countries: 5 stated flat, truth 44** |
| P6 | H5 section code | section B.3 of FP172? | scoped honestly, **but the board note injected "B.3 is not in this corpus… State this definitively" — a false authoritative note**; this model ignored it |
| P7 | H12 theme count | agriculture count? | HONESTLY-SCOPED — coverage note fired; model correctly said it has no theme field and refused a count |
| P8 | v2 scalar field | implementation period FP220? | CORRECT+CITED — "12 years" [p. 5] = v2 canonical |
| P9 | v2 sparse field | co-financing FP086? | **WRONG** — derived EUR 13M by subtraction, cited "cover pages"; v2 has no fact; deterministic verifier caught it |
| P10 | H1 in French | Banque mondiale ? | HONESTLY-SCOPED (weak) — 6 of 13, one category error (FP183 is IFAD), no incompleteness statement |
| F11 | H4 forced sum | sum the year note | **HALLUCINATED-COMPLETE-ANSWER** — $29.0B / $85.0B vs truth ≈$1.36B / $2.41B (21×/35× off); root cause below |
| F12 | H1 count form | how many UNDP? | HONESTLY-SCOPED — refused the count, no fabricated number: the scoping rule held under direct pressure |
| F13 | list completeness | FP151: how many countries? list all | **HALLUCINATED-COMPLETE-ANSWER** — "five", cited, verifier scored SUPPORTED; truth 44; root cause below |
| F14 | H5 control | section C.2 of FP172? | CORRECT+CITED — real C.2 content with the component cost table; section retrieval works |
| F15 | year-scoped minimum | least-requesting 2020 FP? | WRONG value (named FP151 18.5M; the note itself prints FP129 = 17,198,843), honestly scoped |

### 7.1 The two findings that outrank everything else in this review

**Silent list truncation is a note-manufactured falsehood the verifier certifies (F13/P5).**
`registry.py` builds the note with `", ".join(r["countries"][:5])` — no count, no ellipsis — inside
a note labelled *authoritative*. Asked "how many countries — list all", the system answers **five**
with a citation, and `classify_deterministic` marks the claim **supported**, because the evidence
really does say it. No layer of the stack asks "is the evidence complete?". Every list field is
exposed; countries is the one users ask.

**Year-note aggregation multiplies a parse bug (F11, P4).** `_year_assist` prints v1 money strings
raw and unnormalised ("28,654 million USD" for FP153); v2 holds normalised floats for the same
fields and the note does not use them. Prompted to sum, the model produced totals 21–35× too high;
unprompted, a flatly wrong direction. There is no summation mechanism and no refusal rule for one.

### 7.2 Probes from §6(a) still unexecuted

H16 (board-inverse in-range), H17 (year ranges end-to-end), the H7 arity pair, and H10 (`FP#220`
formats) were not covered by the fifteen probes and remain open.
