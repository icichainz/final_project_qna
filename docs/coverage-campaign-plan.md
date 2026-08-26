# The Coverage Campaign — working plan

Branch `coverage-campaign` · adopted 2026-08-26 from the external forward plan
(authored at release-9 / `4beb9ad`), reconciled here to the actual state at
`d30da86` (release-10, 126 cases, 425 claims, L3 measured). This file is the
living version: statuses move here, and each phase's exit gate is quoted at
its closure with the run that proved it.

**The goal, in the plan's own decomposition:** mechanisms that can serve every
fact the store holds (Phases 1–2), data that actually holds every document's
facts (Phase 3), measurement that samples the whole corpus rather than ~16%
of it (Phase 4), an honest rule-application L3 (Phase 5), and instruments
that can see all of it (Phase 6). The non-negotiable stays non-negotiable:
never assert what cannot be cited.

**House rules carried over unchanged:** every phase ends in a
same-configuration release pair judged against the measured ±2–3 pp band;
every registry or gold change re-anchors `data/eval/CHECKSUMS.sha256`;
derivation-or-drop for every gold expectation; exact-allowlist note fences;
blind adjudication + owner ratification for anything that touches gold or the
verifier's ground truth.

---

## Phase 0 — commit what is in flight, settle the ledger  ·  **~80% CLOSED**

| item | status |
| --- | --- |
| Answer-cap removal (`MAX_ANSWER_TOKENS` default uncapped, `_answer_cap()`) | **DONE** — committed `d30da86`; validated by release-10's own data (`agg-inv-undp` 1.00, 41/41 named, where capped runs silently stopped at 34) |
| Visible truncation marker off `finish_reason` when a cap IS set | OPEN — small, both answer loops |
| Adjudicate `agg-2020-least` (deterministic CONTRADICTED on a 1.00 case: must a perfect answer report FP129's second print?) | **OWNER** — a ruling, not code; same protocol as ruling 8 |
| Adjudicate `agg-2021-boards` (board→year claims unsupported by construction: mapping into evidence, or expect hedged phrasing?) | **OWNER** |
| PLANNER default drift (config `0`, deployed `.env` `1`) | **OWNER decision, recommendation recorded**: default `1` — the measured configuration is the deployed one; today a lost `.env` line silently un-planners production |
| Gold-set ratification for the 101→126 case waves | **OWNER** — one session can cover this plus the two adjudications above |

Exit gate (amended): the original gate ("agg-inv-undp reaches 1.00") is
already met by release-10; what remains gated is the owner session.

## Phase 1 — L1 mechanisms: every stored fact answerable at arity one  ·  **CLOSED** (`8c0bedc`, gated by release-11)

Source finding: 736 of 3,785 candidate records servable by nothing; most
documents hold at least one field their single-document ask cannot reach.

- Serve fields at arity one: single-identifier turns get the asked-for fields
  appended to the registry line via `planner.fields_for()` — same store, same
  page/section citations the matrix already prints.
- Add `executing_entity`, `national_designated_authority`,
  `financial_instruments` to the field maps (exist for 193/101/70 documents,
  reachable by nothing).
- Guard `_corpus_coverage_note` against thematic asks (last false-authority
  path; probe P7 showed the model copes, so hygiene, not live harm).
- Widen `_conflict_lines` beyond `_MONEY_FIELDS` (the four non-money
  conflicts can then warn at arity one).
- `retrieve.identifier_tokens` widening for FP-variant forms — or keep the
  recorded-as-test acceptance; the registry note already carries those turns.
- Extraction honesty: `suspect` (16) and `llm_fallback` (19) documents say so
  on their registry line, the way lists state their own truncation.

Exit gate: the arity pair returns the same field, same citation, both
arities; zero of the 126 gold questions change behaviour except where a case
pins the new field service.

## Phase 2 — L2 mechanisms off the happy path  ·  **mechanism half CLOSED** (`8c0bedc`: probe_pages 18/18; app wiring = next wave)

- Conflict-fallback probe: on conflict-shaped turns (or when a fired CONFLICT
  note's pages didn't land), issue a doc-scoped query for the disagreeing
  section. Justified by the 8/14 page shortfall, stable across four releases.
- License the comparative sentence over the note's computed year totals
  (the totals and the direction are already printed; release-10's
  `l2x-xyear` case passed — verify whether anything remains here at all).
- Measure a 4–5 document comparison; scale the merged-hit cap with arity if
  cells starve.

Exit gate: conflict-class evidence pages move off 8/14 for the first time; a
4-doc matrix renders with no starved row.

## Phase 2½ (AMENDMENT 1, promoted to its own track) — the sectioned index  ·  **DECIDED against ranked adoption** (release-11 vs 11-v2)

The live retriever loads `data/index/default`, where `section_path` is
`None` on every hit; the section-aware v2 index (125,414 sectioned chunks)
has been staged since plan step 3 and never loaded. Measured symptoms that
plausibly share this root: section-page retrieval reaching top-10 for only
18/92 C.2-bearing documents; procedural steps one-chunk-deeper missing four
times over; the conflict-page shortfall of Phase 2. **Phase 5c hard-depends
on this** (template-compliance presence checks read `section_path`).
Work: activate `INDEX_NAME=v2` (or rebuild default with sections), A/B
retrieval fixtures both ways, and only adopt what measures better — the
same discipline that kept RERANK honest.

## Phase 3 — data: make the registry cover all 273 documents  ·  OPEN, the long pole

31 unrecognized-template documents; 68 with missing core fields; 16 suspect;
19 llm_fallback; plus the newest extraction defects absorbed by amendment:
line-item-as-total (FP048), first-of-several co-financing rows
(FP106/FP115/FP098/FP055/FP245), incoherent v1/v2 values (FP042/FP001/FP047).

- Targeted re-extraction of the 31 unrecognized-template documents (identify
  variants, add recognizers, re-run the VLM on those only).
- Sweep the 68 core-missing documents: extract or record a **confirmed
  absence** — absence-as-fact is what lets those L1 asks answer honestly.
- Adjudicate the 16 suspect + 19 llm_fallback extractions (human hours,
  bounded).
- Approval dates (H9): extract if cover pages print one; otherwise record
  absence and keep the board-year licence; make YEAR_BLOCK ship on
  date-shaped asks.
- French country exonyms beyond accent folding (hand table); Congo stays
  ambiguous by design.
- H19 watch guard: a test that fails the moment a second document per FP
  enters the corpus.

**Operational constraint (verified, in memory):** the VLM endpoint tolerates
exactly one concurrent request — serial passes, batched by template variant.
This phase runs on the owner's machine and clock.

Exit gate: every document has, per core field, a canonical value with page
provenance or a recorded confirmed absence; registry rebuild checksummed; a
release pair shows no gold regression.

## Phase 4 — gold that samples the corpus it claims  ·  Wave 1 ~60% DONE

Already closed by the 101→126 wave: `total_financing` conflicts (×2, one
candidate rejected as an extraction artifact), before-2016 and range forms
(finding: "between Y and Y2" is not an implemented form — middle year
silently missing), French rankable comparison, the three zero-coverage
years, section lookups, both arity arms, plus the L3 procedure class the
original plan did not anticipate.

Wave 1 remainder: French board-code; a follow-up over a conflict answer;
a four-document comparison; an executing-entity case (expected-hedge until
Phase 1 lands, then flipped).

Wave 2 (needs Phase 3): stratified sampling by year × template era until
every year and era has representation; one asserting case per registry_v2
field once Phase 1 serves them; document coverage raised from ~45/273 to a
**declared** target (≥30% defensible — declare it, then measure against it).

Exit gate: release pair on the expanded set; per-shape means published with
the noise band; the coverage claim restated with its denominator.

## Phase 5 — L3, honestly scoped  ·  head start already measured

Already done (release-10): the `procedure` class — retrieval face 59/59
claims, and the judgment boundary held under three traps. What this phase
adds is the **rule-application** face, extending the derived-arithmetic
precedent (deterministic application of a cited rule to cited facts,
SUPPORTED-with-caution / CONTRADICTED, never silent).

- **5a** Ingest the policy segment: GCF investment-criteria framework, E&S
  policy and ESS category definitions, concessionality guidance,
  results-management framework, the funding-proposal template definition.
  `source_type: policy` on every chunk; answers never blend policy and
  proposal text without saying which is which; registry gains a policy index
  (rule id → doc, section, page). **OWNER decision first: sourcing the PDFs
  is new corpus acquisition.**
- **5b** The rule-application claim shape: three checkable premises (rule
  verified against the policy segment; facts verified as today; application
  deterministic where possible), verdicts mirroring derived arithmetic.
- **5c** Starter shapes, easiest-checkable first: definition+stated-value
  join; template-compliance presence checks (**hard dependency: Phase 2½**);
  criteria-coverage enumeration; numerical threshold checks.
- **5d** Process discipline unchanged: blind adjudication rounds and
  anchor-mapped rulings for new L3 failure shapes; each shape class ships
  behind its own gate.

Out of scope, permanently: judgment calls ("should this have been
approved?") — weighing rather than checking gets the honest refusal.

Exit gate: the taxonomy row moves to "L3 · partially covered —
rule-application over ingested procedures, N shapes, measured", with
denominators; what the corpus cannot exercise stays out of scope, stated.

## Phase 6 — instruments  ·  OPEN

- Decide which instrument is wrong for `abs-antarctica` (abstain detector
  vs fixture) and fix that one; give chat-mode refusals a certifiable
  abstain path (`abs-offtopic`).
- `_CONFLICT_RE` lacks the two-word negation "not consistent" (recorded in
  `l2x-conf-fp249`'s notes) — same decide-then-fix treatment.
- Release-pair discipline permanent; `bc-b27-02-add12` stays the resident
  flapper spot-check.

Exit gate: a release where every sub-1.00 score is a true behaviour defect.

---

## Sequencing at `d30da86`

1. **Owner session** (can happen any time, unblocks Phase 0 closure): the
   two adjudications, the PLANNER default, gold ratification, the Phase 5a
   sourcing decision.
2. **Parallel agent tracks, disjoint files**: Phase 1 (registry/planner
   formatting) ∥ Phase 2 (retrieval fallback) ∥ Phase 2½ (index A/B).
3. **Phase 3** starts alongside on the owner's machine (serial VLM);
   nothing in 1–2½ depends on it.
4. **Phase 4 Wave 1 remainder** after Phase 1 (field cases need field
   service); **Wave 2** after Phase 3.
5. **Phase 5a** after the sourcing decision; **5b–5c** after 5a + 2½.
6. **Phase 6** slots into any release preparation.

Every phase closure updates this file and quotes its gate run.


---

## Wave 1 closure record (2026-08-26, `8c0bedc` + release-11 pair)

- **Phase 1 closed**: 1,195 (doc, field) pairs servable at arity one; three
  new fields mapped; non-money conflicts warn; extraction honesty flags on.
  Scorer byte-identical; release-10 → release-11 within the noise band
  (−0.8 pp support / +1.0 pp groundedness). Gate note: release-11 passed at
  exactly its integer threshold (384/404) — zero margin, recorded.
- **Phase 2 mechanism half closed**: `Retriever.probe_pages` recovers 18/18
  conflict-class target pages on demand; app wiring is the next wave's item,
  together with the year-note boards evidence (ruling 10, shipped) and the
  truncation marker (shipped).
- **Phase 2½ decided**: the sectioned index is REFUSED for ranked retrieval —
  answer-level A/B (release-11 vs release-11-v2): 1 better / 6 worse, the
  losses concentrated in the section/procedure cases v2 was meant to serve
  (its chunk boundaries aid addressing, hurt generation context; shared
  subset −2.3 pp support / −4.1 pp groundedness). The binary capability
  stands (named-section addressing 0/136 default vs 136/136 v2) and the
  Phase 5c path is now: extract v2's section map as a small
  {doc → section → pages} artifact and serve those pages FROM THE DEFAULT
  index via probe_pages — page numbering is identical across builds
  (41,550 pairs verified), so no second index is needed at runtime.


## Serving wave + Wave 2 closure record (2026-08-26, `0f86897` + release-13 pair)

- **Phase 4 Wave 2 closed**: 126 → 157 cases; direct document coverage
  53 → 70 of 273 (25.6%); all 11 served-but-unasserted fields asserted;
  all four Phase-3 recognizer families first-tested; the first
  ratified-absence gold case. The 30% target remains declared, ~12
  documents short — a topper rides any future wave under the
  coordinator ruling (aggregate-derivation FP mentions are not "used").
- **The serving wave closed**: enumerations citable (release-13:
  citation presence 100.0% in BOTH arms, 469/469 and 462/462 — the
  release-12-repeat failure shape extinct); confirmed absences served
  (31 of 48 pairs); conflict pages probe-delivered (8/14 → 14/14 after
  five stuck releases); the cross-extractor verification arm standing.
- **Release-13 pair verdict**: support 94.9%/95.0% — exactly ON the 95%
  integer gate, one claim either side across arms; within-pair deltas
  −0.8/+1.0pp, inside the band; failure census entirely known shapes,
  zero attributable to the new mechanisms.
- **The next data round, sized**: the cross-extractor census flags 129
  of 622 canonical money facts (99 documents, 79 digit-misread
  signatures) — adjudication-ready at scratchpad/s4/cross_check_flags.json.

## Remaining open

Phase 5a (owner provides policy PDFs) · Phase 6 (two instrument items) ·
the 129-flag adjudication (owner session) · the 30% coverage topper ·
deferred: registry_named acronym gap, Track D conflict-page ranking.
