# Wave 1 adjudication — owner review sheet

71 failed claims from `release-1`, labelled **blind** (the verifier's own verdict was withheld) by three independent Opus reviewers, with a fourth independently re-labelling 20 rows to measure agreement.

## Headline

| | count | share |
|---|---|---|
| `verifier_false_positive` | 46 | 65% |
| `not_a_claim` | 12 | 17% |
| `missing_citation` | 6 | 8% |
| `missing_retrieval_evidence` | 5 | 7% |
| `wrong_citation` | 1 | 1% |
| `genuine_answer_error` | 1 | 1% |

**Inter-rater agreement: 17/20 (85%)** on both root cause and `verifier_correct`. All three disagreements are the same taxonomy gap (ruling 1), listed below.

**The verifier was wrong on 59 of 71 flagged claims (83%).** The 57% claim-support figure is therefore mostly measurement artifact, not wrong answers — as suspected, now measured.

## What the 46 false positives authorise (Wave 2 scope)

| mechanism | rows |
|---|---|
| registry-conflict figure the note itself states | 14 |
| doc-level / cover-page citation | 14 |
| exact-value match on held evidence | 7 |
| registry-confirmed closed-world negative | 7 |
| registry-known acronym / full name | 3 |

## DECISION 1 — doc-level citations (highest leverage)

My taxonomy's ruling 4 covers page citations only. ~14 rows cite `[doc]` or `[doc, cover pages]` while the figure sits on a *different held key* of that same document, with the page named in the claim's prose.

- **Reviewers' reading (adopted):** the bracket names the right document, the evidence was held, the value matches → `verifier_false_positive`.
- **Stricter reading:** the citation does not point at the entailing evidence → `wrong_citation`, and the false-positive count drops by ~9-14.

**My recommendation: adopt the reviewers' reading.** Treating coarse-but-correct citation as a verifier hit pushes generation toward citing pages it never held, which is the worse failure. Precision of citation is a separate, reportable metric — not a support question.

**Your call:** ☐ adopt (VFP)  ☐ strict (wrong_citation)  ☐ split by whether the claim's prose names the page

## DECISION 2 — headings/lead-ins that assert something

Rulings 1-2 name *shapes* but justify by "asserts nothing checkable"; some headings do assert checkable, held content. This is the 3-row inter-rater disagreement:

- `claim-8b23b13e` — **not_a_claim** vs **missing_citation** — FP267 (“Eco-DRR”) shows **conflicting figures** for the **total GCF funding requested** in the retrieved excerpts:
- `claim-a72ac18e` — **not_a_claim** vs **missing_citation** — For FP173 (“The Amazon Bioeconomy Fund”), the retrieved excerpts show **conflicting figures within the same document**:
- `claim-c4a42b13` — **not_a_claim** vs **missing_citation** — Dans le FP274 (BRACE), le **montant de financement GCF demandé** apparaît avec **trois valeurs différentes** dans le même document :

**My recommendation: shape wins.** A lead-in whose predicate is completed by the unit below it is an extraction artifact; fixing extraction removes the row entirely rather than teaching the matcher to chase it.

**Your call:** ☐ shape wins (`not_a_claim`)  ☐ test wins (`missing_citation`)

## DECISION 3 — negatives about retrieved content

Ruling 3 covers registry-confirmed *existence* negatives. Two rows assert a true negative about the *retrieved set* ("none of the retrieved excerpts mention X") with no registry note. Reviewers scored these `verifier_false_positive` (the row enumerates the evidence and settles it) except one marked `ambiguous_unscorable`.

**My recommendation: supported when the row's own evidence enumeration settles it**, since the claim is explicitly scoped to what retrieval surfaced.

**Your call:** ☐ supported  ☐ unscorable

## The 12 genuine defects (verifier was right)

These are the real work for generation/retrieval, and they are what remains of the 57%:

| label | claim |
|---|---|
| `missing_citation` | Using the **A.8 “GCF funding requested”** entries: **FP220 (USD 50.0m) > FP173 (USD 23.6m) > FP172 (USD 21.128 |
| `missing_citation` | FP153 (“Mongolian Green Finance Corporation”) has **inconsistent figures** in the retrieved document: |
| `missing_retrieval_evidence` | Parmi les extraits récupérés, c’est la **proposition de financement GCF B.30 (2021), document [105_gcf-b30-02- |
| `missing_citation` | So, based on what retrieval surfaced, FP173’s GCF amount is reported as **either USD 23.6 million or USD 279 m |
| `missing_retrieval_evidence` | Si vous cherchez une proposition qui mentionne explicitement un paquet d’interventions “riziculture intelligen |
| `missing_retrieval_evidence` | **FP151**: *Technical Assistance Facility for the Global Subnational Climate Fund* (IUCN) [124_gcf-b27-02-add1 |
| `missing_citation` | **GCF grant: USD 21.127 million**, |
| `missing_citation` | total financing needed **USD 49.312 million**, |
| `wrong_citation` | Non — d’après les montants figurant dans les extraits, **FP214** demande **38,17 millions d’euros** de finance |
| `genuine_answer_error` | **Total Target Fund Corpus / Offshore Fund corpus (Architectural Fund, Singapore)** — the form/excerpt that di |
| `missing_citation` | Among the retrieved excerpts, the Global Subnational Climate Fund (SoCF/“SoFC”/SnCF Global) is described as be |
| `missing_retrieval_evidence` | **FP152**: *Global Sub-national Climate Fund (Equity)* (Pegasus Capital Advisors LP) [123_gcf-b27-02-add12, p. |

## Spot-check sample (required before Wave 2)

Every `verifier_false_positive` is listed in `data/eval/release_release-1-adjudicated.jsonl` with the reviewer's note naming the page that settles it. Sign off by confirming the three decisions above and spot-checking any rows you wish; the file records `label`, `verifier_correct`, `notes`, `confidence`, and the second opinion where one exists.

