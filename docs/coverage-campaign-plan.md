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


## Cross-check round closure (2026-08-26, release-14 pair) — a red gate, fully named

The round: 129 flags adjudicated three-source, 117 corrections applied,
rulings 11-12 signed, the scorer baseline refreshed (640/640 hit texts
frozen), the llm_fallback decay fixed, suite 1766/4/0. The release-14
pair then reads **94.6% / 94.3% — gate FAIL in both arms**, consistent
(not noise), and the failure census decomposes completely:

1. **~4 failures = the transitional store-vs-served divergence, live**
   (answers quoting served pages' old digits, contradicted by the
   corrected store). Fix: the queued corpus cure (re-extract all
   flagged pages + index refresh, next LM Studio window ~3h). The
   ruling-12 pins are written to shrink when it lands.
2. **~6-8 failures = a new named gap: the inverse-listing HEADER
   count claim** ("41 proposals record UNDP") files under no evidence
   key because the header deliberately names no stem — the items
   verify, the count cannot. Fix queued: give the count a verifiable
   scope (design owed to the next serving touch).
3. The remainder: the long-known cited-hedge / meta-sentence shapes
   at a claim population that has doubled since release-10.

The honest state: the store is MORE correct than ever (cross-checked
against the PDFs' own text), and the gate honestly reports that the
served corpus and one note shape have not caught up. Red for fully
named reasons, each with a queued fix, beats green by silence.

## Open items after the round

1. Corpus cure (owner's LM Studio window, ~3h serial + index refresh + release pair — expected to clear cause 1 and shrink the ruling-12 pins)
2. Inverse-header count scope (next serving touch — cause 2)
3. Phase 5a policy PDFs (owner)
4. Phase 6 instruments (unchanged)
5. Deploy + SSH key + leaked keys (the standing trio)


## The corpus cure closure (2026-08-26, release-15 pair) — GREEN

Part 1: 95 pages re-extracted against ratified known answers (87 CURED,
104/109 rows now printed in the served corpus, from 8). Part 2: registry
and index rebuilt over the cured corpus (the rebuilt live index carries
section paths on 125,430 chunks — Phase 2½'s capability, delivered
incidentally); baseline notes re-refreshed; two builder defects killed
en route (settlement-by-declaration, the 0.5% tolerance); the
currency-then-scale reader gap closed; every pin moved as its own
comments promised.

**Release-15 pair: PASS in both arms — 375/394 = 95.2% and 465/486 =
95.7% — with contradicted verdicts collapsed 9-11 → 4-5.** The PDFs'
text layer, the served corpus, the store, the gold set and the scorer
baseline now describe one corpus. The campaign's engineering is
complete.

## Final open items

1. Inverse-listing header count scope (next serving touch)
2. Phase 5a policy PDFs (owner)
3. Phase 6 instrument items (unchanged)
4. Deploy · SSH key · leaked keys (the standing trio)

## The answer-length pass (2026-08-27/28, releases 16-18) — arm 1 GREEN, pair pending credits

Operator report after the release-15 deploy: live answers read bare.
Root-caused to no cap at all — the campaign's own citation discipline
(cite-or-hedge, corpus scope, apposition guard) taught the model to
state the asked fact and stop. The cure is CONTEXT_BLOCK: answer first,
then two or three cited sentences of registry-note context, shipped
behind the registry trigger (comparison/matrix turns keep their
per-item format), so the prompt budget's biggest variant is unchanged.

Two discovery arms sharpened the rule, each on a measured defect:

- **Release-16** (96.5%, contradicted 9): five elaboration sentences
  volunteered one side of a registry-recorded total-financing conflict
  (FP173, FP151, FP251, FP29) → conflict figures only in two-value
  form; note lines never pasted as citations (fr-agg-2020's language
  flunk).
- **Release-17** (96.5%, contradicted 10): the four cured, but the
  acknowledge-without-enumerating habit migrated, and a table figure's
  rounding fought the canonical cover value (156.7M vs 156.8M). Every
  excess contradicted claim in both arms was money; descriptive
  context contradicted nothing → unasked money excluded from the
  added context wholesale.

**Release-18 arm 1, the strongest arm recorded: 575/592 = 97.1%
supported, contradicted 5 (all baseline shapes, none from the
elaboration), presence 88.2%, identifier median 555 chars (was 221).**
The repeat arm aborted on API credit exhaustion (429, ~40 cases in);
its truncated record is void and unanchored. The pair closes with one
re-run (`--record release-18-repeat --force-record`) once the account
is topped up — production shares the key and is equally starved.

Also this window: main fast-forwarded to the campaign tip and deployed
(fp-gcf:94cea40, the release-17 block; the release-18 block redeploys
on pair closure). The GitHub push is still blocked on the SSH key, and
the extracted green theme survives on the server as two stale files
rsync never deletes (`public/custom.css`, `public/theme.json`) — the
removal command awaits the owner, then the next deploy ships clean.

## Final open items (unchanged otherwise)

1. Top up API credits → re-run release-18-repeat → redeploy certified block
2. Inverse-listing header count scope (next serving touch — deliberately
   NOT folded into this pass, to keep release-18's arm 1 valid)
3. Gold regex near-miss `w2a-rbp-fp273-absence` ("the FP273 document"
   defeats the article-noun pattern) — scorer edit, owner ratifies
4. Phase 5a policy PDFs (owner) · Phase 6 instruments
5. SSH key · leaked keys · stale theme files (owner)

## The answer-length pass closes (2026-08-28, release-19 pair) — GREEN

After the credit top-up, release-18-repeat came back red for one named
reason: the inverse note's mid-word title clips invited quote-tidying
(fr-inv-banque-mondiale, 15 of the arm's 31 unsupported claims). _clip
now cuts at word boundaries; baseline re-rendered (640 hit chunks
proven unchanged) and re-anchored.

**Release-19 pair: PASS both arms — 534/548 = 97.4% and 552/572 =
96.5% — contradicted 5 in BOTH arms with identical case lists, the
World Bank case 14/14 and 14/15, identifier median 545/566 chars
(was 221).** Deployed as fp-gcf:7ad53d2. Remaining open items are the
owner trio (SSH key, leaked keys, stale theme files), the gold regex
near-miss awaiting ratification, the inverse-header count scope, and
Phases 5a/6.

## The section probe closes (2026-08-31, release-22 pair) — GREEN

Two serving touches, each earned by a named residue:

1. **The section probe** (releases 20-21): `probe_pages(sections=...)`
   finally got its ask — 'section C.2' + exactly one document fetches
   the section by printed id, two-stage (the id finds the pages, the
   pages are fetched whole, because FP126's table body files under a
   VLM-promoted table-header heading). l1x-sec-c2-fp126: page_rate
   0.0 → 1.0 and full pass in every arm since.
2. **The copy rule** (release-21-repeat red, cured in 22): the French
   listing case produced its second paraphrase habit (per-line entity
   translation after release-18's ellipsis tidying), so `_COPY_RULE`
   now rides on every inverse listing — copy lines EXACTLY as printed,
   translate nothing, state the shared name once in the registry's own
   spelling. A rule line, not evidence; the _NO_SUM_RULE pattern.

**Release-22 pair: PASS both arms — 593/608 = 97.5% and 594/620 =
95.8%; banque-mondiale 14/15 and 13/13; the section case green in both
arms.** Deployed as fp-gcf:4fde815. Standing note: if the French
listing case produces a THIRD paraphrase habit, the byte-substring
standard for listing lines becomes a verifier-protocol question for
the owner, not another serving patch.
