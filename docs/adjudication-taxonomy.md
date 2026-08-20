# Adjudication taxonomy and rulings

Labelling rules for the 71 `release-1` failures. Written before labelling, so
the rulings are not reverse-engineered from whichever labels the rows happened
to attract. Wave 0c's gate exercise scored 3 of 8 rows cleanly decidable, and
**all five undecidable rows failed on label definition, not on missing
evidence** — these rulings close that gap.

## Two independent questions

Each row records both. Conflating them is what made `verifier_false_positive`
carry two meanings at once.

1. **`verifier_correct`** — should the verifier have flagged this claim at all?
   Wave 2's precision and recall are computed from this field alone.
2. **`root_cause`** (the `label`) — what produced the failure, and therefore
   which component has to change.

## Root causes

| Label | Meaning | Implied fix |
|---|---|---|
| `verifier_false_positive` | The held evidence and the claim's own citation support it; the verifier was wrong | verifier matcher |
| `genuine_answer_error` | The answer states something incorrect, contradictory, or materially incomplete | generation / repair |
| `missing_retrieval_evidence` | The evidence needed was never retrieved for that turn | retrieval |
| `missing_citation` | Supporting evidence was held, but the claim cites nothing | generation |
| `wrong_citation` | The claim cites *something*, but not the evidence that supports it | generation |
| `not_a_claim` | The unit is not a factual assertion (lead-in, heading, glue) | claim extraction |
| `registry_conflict` | The document contradicts itself; the failure exposes a source-level conflict | registry / answer policy |
| `ambiguous_unscorable` | The record cannot settle it; the ambiguity is written down | none — escalate |

## Rulings for the four contested shapes

**1. Colon lead-in** — `"The financing terms are as follows:"`, `"Key figures:"`.
Asserts nothing checkable; it introduces the claims that follow.
**Ruling: `not_a_claim`, `verifier_correct: false`.** The fix is in extraction,
not in matching — the verifier cannot support a sentence with no proposition,
and treating these as verifier misses would send the matcher chasing them.

**2. Markdown heading** — `"## Financing"`, `"**Accredited entity**"`.
Same reasoning. **Ruling: `not_a_claim`, `verifier_correct: false`.**

**3. Closed-world negative** — `"FP999 does not exist in this corpus"`,
`"the corpus contains no B.44 proposals"`.
The corpus is closed and the registry is complete for it, so *absence in the
registry is positive evidence of absence*. **Ruling: when the registry confirms
the absence, `verifier_false_positive` — the claim is true and supported.**
Where the assertion is broader than the registry can settle (a negative about
document *content* rather than existence), it is `ambiguous_unscorable` with the
reason written down. This is the one ruling that expands what counts as support,
so it is scoped narrowly: existence, registry-confirmed, nothing else.

**4. Wrong-scope or never-retrieved citation** — the claim carries a bracket,
but it points at a page the turn never held, or at a page that does not state
the figure.
**Ruling: `wrong_citation`, `verifier_correct: true`.** Distinguishing it from
`missing_citation` matters: the answer *did* cite, so the fix is citation
accuracy in generation, not citation presence. It is not
`missing_retrieval_evidence` when the fact was available elsewhere in the held
evidence — check the row's evidence block before choosing.

## Labelling procedure

1. Export **blind** so the verifier's verdict cannot anchor the label:
   `python3 scripts/adjudicate_claims.py export --release data/eval/release_release-1.jsonl --evidence data/eval/release_release-1-evidence.jsonl --output <work>.jsonl --blind`
2. Label from the row alone: claim text, its citations, and the evidence the
   turn actually held. Each row needs `label`, `verifier_correct`, `reviewer`,
   and a `notes` line naming the page or stating why it cannot be scored.
3. Rejoin on `claim_id` and validate:
   `python3 scripts/adjudicate_claims.py import --release ... --inventory <work>.jsonl --output data/eval/release_release-1-adjudicated.jsonl`

**Owner spot-check (required).** Agent labelling is not self-certifying: the
same system that builds the matcher must not be the sole judge of which of its
outputs were wrong. The owner reviews **every** `verifier_false_positive` — the
label that authorises a matcher change — plus a stratified sample of at least
one row per other label, and signs the adjudicated file.

## Rulings 5-7 (added after Wave 1 labelling, owner-approved)

All three reviewers hit the same three gaps independently. The rulings below
close them; rows already labelled under them were re-resolved, and the counts
in `docs/wave1-adjudication-review.md` reflect the amended taxonomy.

**5. Doc-level and cover-page citations.** A bracket naming only the document
(`[doc]`, `[doc, cover pages]`) is **satisfied by any held evidence key of that
document** that entails the claim. It is a coarse citation, not a wrong one.
Rationale: treating it as a verifier hit would push generation toward citing a
specific page it never held, which is the worse failure — an invented page
reads as precision. Citation *precision* is reported separately and never
folded into support. This ruling governs ~14 rows and is the one that most
moves the false-positive count, so it is stated rather than left to taste.

**6. Shape wins for headings and lead-ins.** When a unit has the *form* of a
heading, bold label, or colon lead-in, it is `not_a_claim` **even if it carries
a checkable proposition** — the predicate is completed by the unit below it.
Rationale: the fix is claim extraction, and fixing extraction deletes the row
entirely; teaching the matcher to support half-sentences would be the wrong
component changing. This resolves the three inter-rater disagreements.

**7. Negatives scoped to the retrieved set.** A negative that explicitly scopes
itself to retrieval ("none of the retrieved excerpts mention X", "based on what
retrieval surfaced") is **supported when the row's own evidence enumeration
settles it** — the claim asserts something about the held set, and the held set
is in the row. Unhedged negatives about corpus *content* remain
`ambiguous_unscorable` per ruling 3; only the explicitly scoped form is covered
here.
