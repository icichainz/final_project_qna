"""The generation-side citation pass, pinned by dimension rather than wording.

The red acceptance gate is citation completeness (SUPPORTED *and* carrying a
bracket) at 144/165 = 87.3% against >= 95%, with groundedness at 89.1%: the
evidence supports the sentence, the sentence carries no bracket. The three
audits agree on the dominant shape ('no citation on a factual claim') and on a
secondary one (a claim asserted where retrieval never surfaced the evidence,
4 adjudicated `missing_retrieval_evidence` rows).

The prompt rules added for that are pinned here as PRESENCE OF DIMENSION —
"this variant carries a per-sentence citation rule", not "the prompt equals
this string". String equality would freeze the wording and make every future
sharpening a test edit, which is how a prompt suite stops catching anything.

The verifier is the frozen instrument and is imported READ-ONLY below, to
document what the rules are aimed at: `verify._units` inherits a paragraph's
trailing bracket backwards over that paragraph's sentences, but a bullet NEVER
borrows another bullet's bracket. That asymmetry is why the list rule is its
own sentence in CORE.
"""
import pytest

from gcf_qna.app import prompts
from gcf_qna.app.chainlit_app import _invalid_citations, _note_pages
from gcf_qna.app.prompts import (CHAT_CORE, COMPARISON_BLOCK, CORE,
                                 MATRIX_BLOCK, REGISTRY_BLOCK, SYSTEM_PROMPT,
                                 assemble, assemble_chat)
from gcf_qna.rag import verify

#: Every (kwargs) shape `chainlit_app` and `scripts/eval_answers.py` can build.
VARIANTS = [
    {},
    {"year": True},
    {"registry": True},
    {"comparison": True},
    {"matrix": True},
    {"lang": "French"},
    {"year": True, "registry": True, "comparison": True},
    {"year": True, "registry": True, "comparison": True, "matrix": True,
     "lang": "French"},
]


def _all_variants():
    return [assemble(**kw) for kw in VARIANTS]


# --- the rules ship in every answer variant ---------------------------------

@pytest.mark.parametrize("kw", VARIANTS)
def test_every_answer_variant_carries_the_per_sentence_citation_rule(kw):
    """`verify._units` lets a sentence borrow its paragraph's trailing bracket,
    so the marginal loss is a paragraph with NO bracket and a paragraph whose
    one bracket mis-scopes across documents. The rule names both."""
    p = assemble(**kw)
    assert "CITE AT THE" in p and "SENTENCE" in p
    assert "every sentence stating a document fact" in p
    assert "put the bracket on the sentence that states the fact" in p


@pytest.mark.parametrize("kw", VARIANTS)
def test_every_answer_variant_carries_the_list_item_rule(kw):
    """Bullets and table rows never inherit a sibling's citation in
    `verify._units`; one trailing bracket under a multi-document list leaves
    every other item uncited. Two adjudicated `missing_citation` rows are
    exactly that shape ('**GCF grant: USD 21.127 million**,')."""
    p = assemble(**kw)
    assert "Every bullet, list item and table row carries its own bracket" in p
    assert "never covers a multi-document list" in p


@pytest.mark.parametrize("kw", VARIANTS)
def test_every_answer_variant_carries_cite_or_hedge(kw):
    """The 4 `missing_retrieval_evidence` rows cannot be fixed by finding
    evidence. An explicit 'the retrieved excerpts do not state X' is glue under
    `verify.claim_kind`, so hedging removes the claim from the denominator
    instead of failing inside it."""
    p = assemble(**kw)
    assert "Cite or hedge" in p
    assert "the retrieved excerpts do not state it" in p


@pytest.mark.parametrize("kw", VARIANTS)
def test_every_answer_variant_keeps_the_page_prohibitions(kw):
    """The pre-existing prohibitions are load-bearing and stay: pushing for a
    page number without them trades uncited claims for invented ones, which
    `verify._scopes` reports as 'cited evidence was never retrieved'."""
    p = assemble(**kw)
    assert "never invent a page number" in p
    assert "cite the document id alone rather than guess one" in p
    # corpus-scope hedging and the conflict rule, kept verbatim
    assert "among the retrieved" in p
    assert "present both values with their pages" in p


@pytest.mark.parametrize("kw", VARIANTS)
def test_never_guess_a_page_outranks_the_cite_at_the_sentence_rule(kw):
    """Release-3 measured the two rules colliding and the weaker one winning:
    'every sentence carries its page' is stated once per answer and pushed on
    every line, while 'cite the id alone when the page is not certain' was a
    subordinate clause. Two comparison turns cited pages that the harness gate
    had never retrieved ('[124_gcf-b27-02-add11, p. 3]'). The fix is a
    PRECEDENCE, not another prohibition, so the precedence is what is pinned —
    and it must be fact-general: the earlier wording read as money-only."""
    p = assemble(**kw)
    assert "OUTRANKS" in p
    assert "cite-at-the-sentence rule" in p
    assert "for ANY fact" in p              # not just figures
    assert "cite the document id alone rather than guess one" in p


def test_the_bracket_format_is_language_independent():
    """The same rules in French; only the prose language changes."""
    fr = assemble(lang="French")
    assert "The bracket format is identical in every language" in fr
    assert "French" in fr
    assert "Cite or hedge" in fr


def test_the_rules_name_the_header_format_the_model_actually_reads():
    """`chainlit_app._doc_label` prints '[doc_id, p. N — B.x, year]'. A rule
    that names a format the model never sees is a rule it cannot follow."""
    assert "[doc_id, p. N — B.x, year]" in CORE


# --- triggered blocks stay triggered ----------------------------------------

def test_registry_citation_rule_ships_only_with_the_registry_note():
    """`registry._fmt` ends its line '[stem, cover pages]' and prints each
    figure's own '(p.5, A.8)'. The note carries its provenance; the answer has
    to carry it through."""
    assert "cite the document id it states plus the page printed beside" \
        in REGISTRY_BLOCK
    assert "[12_doc, cover pages]" in REGISTRY_BLOCK
    p = assemble(registry=True)
    assert REGISTRY_BLOCK in p
    for kw in ({}, {"year": True}, {"matrix": True}, {"comparison": True}):
        assert "cover pages]" not in assemble(**kw)


def test_the_registry_fallback_covers_facts_that_are_not_money():
    """`registry._fmt` prints a page beside FIGURES only ('18.5 M USD (p.5,
    A.8)'); entity, countries and title arrive page-less on the same line. The
    old fallback said 'with no page beside the FIGURE' and the model read the
    scope literally: asked which accredited entity two proposals use, it
    invented p.3 for both rather than fall back to the line's own label.
    The fallback is pinned as covering EVERY fact the line states."""
    assert "provenance for EVERY fact" in REGISTRY_BLOCK
    assert "entity, countries, title, figures alike" in REGISTRY_BLOCK
    # the page-less arm, and the fact kinds that actually reach it
    assert "a fact with NO page beside it" in REGISTRY_BLOCK
    assert "country and title normally have none" in REGISTRY_BLOCK
    assert "[12_doc, cover pages]" in REGISTRY_BLOCK
    # and the money-only scope is gone, not merely supplemented
    assert "beside the figure" not in REGISTRY_BLOCK


def test_the_registry_line_outranks_another_notes_page():
    """Both notes print provenance for the same cell, and they disagree: the
    evidence matrix prints the accredited entity at '(p.3, rule A.1.5)' while
    the registry line prints it page-less under '[124_..., cover pages]'.
    `chainlit_app._note_pages` only credits a page whose line also names its
    document, which no matrix row does — so the matrix page is uncitable and
    the rules have to say which note wins, at BOTH sites."""
    assert "outranks any" in REGISTRY_BLOCK
    assert "page another note prints for it" in REGISTRY_BLOCK
    assert "a Registry line for that fact wins" in MATRIX_BLOCK
    # the deference is inert on a turn that ships no registry note
    assert "outranks any" not in assemble(matrix=True)


def test_matrix_citation_rule_ships_only_with_the_matrix():
    """`planner.render` labels rows 'FP220 | field | value (p.7, A.8)' and maps
    the label to its doc id on the block's header line — the answer cites the
    id, not the label."""
    assert "Cite each value you report at the document id" in MATRIX_BLOCK
    assert "the page its own row" in MATRIX_BLOCK
    assert MATRIX_BLOCK in assemble(matrix=True)
    assert "its own row" not in assemble(year=True, registry=True,
                                         comparison=True)


def test_comparison_block_asks_for_a_citation_per_item():
    """The fan-out is where multi-document lists are produced; the adjudicated
    'FP220 (USD 50.0m) > FP173 (USD 23.6m) > FP172 ...' row is one line stating
    three documents' figures under no bracket at all."""
    assert "each item citing its own document and" in COMPARISON_BLOCK
    assert COMPARISON_BLOCK in assemble(comparison=True)
    assert "each item citing its own document" not in assemble()


# --- what must NOT change ---------------------------------------------------

def test_chat_prompt_carries_no_citation_rules():
    """A chat turn ships no excerpts, so there is nothing to cite and a
    citation rule can only invite an invented bracket."""
    for lang in (None, "French", "English"):
        chat = assemble_chat(lang)
        for token in ("Cite or hedge", "CITE AT THE", "p. N", "bracket",
                      "cover pages"):
            assert token not in chat, token
    assert "not supplied by the user" in CHAT_CORE     # unchanged


def test_compatibility_export_still_assembles():
    """`chainlit_app` imports SYSTEM_PROMPT; MATRIX_BLOCK stays out of it."""
    assert SYSTEM_PROMPT == assemble(year=True, registry=True, comparison=True)
    assert MATRIX_BLOCK not in SYSTEM_PROMPT
    assert "CITE AT THE" in SYSTEM_PROMPT


def test_conductor_prompt_is_untouched_by_the_citation_pass():
    """The conductor emits JSON queries and never writes an answer."""
    for token in ("Cite or hedge", "CITE AT THE", "bracket"):
        assert token not in prompts.CONDUCTOR_PROMPT


# --- the length budget ------------------------------------------------------

#: The fully-assembled prompt is 4932 characters at this commit (CORE +
#: comparison + matrix + year + registry + an explicit language directive).
#: It was 4816 before the page-fallback pass; that pass added 338 characters
#: of new rule and paid 222 of them back by tightening eleven existing
#: sentences, which is the trade this budget exists to force.
#:
#: WHY A BUDGET AT ALL: this module's own docstring records the measurement
#: that the answer model DROPS PROCEDURAL RULES IN LONG PROMPTS — three times,
#: which is why blocks are assembled per trigger instead of shipped whole. A
#: citation rule the model has stopped reading scores worse than no rule,
#: because it costs attention the rules above it were getting. The margin here
#: is ~4% (roughly two lines of prose): a wording sharpening passes, a tenth
#: rule group does not. Tripping this is not a failure to route around by
#: raising the number — it is the point at which the next rule has to earn its
#: place by displacing one, or by shipping behind its own trigger.
MAX_PROMPT_CHARS = 5000


def test_the_fully_assembled_prompt_stays_within_budget():
    biggest = max(_all_variants(), key=len)
    assert len(biggest) <= MAX_PROMPT_CHARS, (
        f"assembled prompt is {len(biggest)} chars, over the "
        f"{MAX_PROMPT_CHARS} budget — see the comment above MAX_PROMPT_CHARS: "
        f"displace a rule or put the new one behind a trigger")


def test_each_block_is_shorter_than_the_core_it_supplements():
    """No single triggered block may outgrow CORE: the per-turn assembly only
    protects attention while the conditional half stays the smaller half."""
    for name, block in (("comparison", COMPARISON_BLOCK),
                        ("matrix", MATRIX_BLOCK),
                        ("registry", REGISTRY_BLOCK),
                        ("year", prompts.YEAR_BLOCK)):
        assert len(block) < len(CORE), name


# --- the instrument these rules are aimed at (read-only) --------------------

def test_the_list_rule_is_what_the_verifier_actually_requires():
    """Not a prompt assertion — a demonstration, against the frozen verifier,
    that the CORE list rule targets a real scoring rule rather than a stylistic
    preference. One trailing bracket under a three-item list leaves two items
    with no citation at all; the same list cited per item leaves none."""
    trailing = ("- FP220 requests USD 50,000,000\n"
                "- FP173 requests USD 23,600,000\n"
                "- FP172 requests USD 21,128,000 [21_doc-c, p. 7]\n")
    per_item = ("- FP220 requests USD 50,000,000 [19_doc-a, p. 5]\n"
                "- FP173 requests USD 23,600,000 [20_doc-b, p. 6]\n"
                "- FP172 requests USD 21,128,000 [21_doc-c, p. 7]\n")
    uncited = [c for c in verify.extract_claims(trailing) if not c.cited]
    assert len(uncited) == 2, [c.text for c in uncited]
    assert not [c for c in verify.extract_claims(per_item) if not c.cited]


#: The FP151/FP152 registry note as `registry._fmt` prints it: the money
#: carries '(p.5, A.8)', the accredited entity carries nothing.
_REG151 = ('Registry — FP151: "TA Facility for the Global Subnational Climate '
           'Fund"; accredited entity: International Union for Conservation of '
           'Nature and Natural Resources (IUCN); GCF funding requested: 18.5 M '
           'USD (p.5, A.8) [124_gcf-b27-02-add11, cover pages]')


class _Hit:
    def __init__(self, doc, page):
        self.doc_id, self.page = doc, page


def test_the_page_less_fallback_is_what_the_harness_gate_requires():
    """Not a prompt assertion — a demonstration, against the frozen gate, that
    the fallback the rules dictate is the form that survives it. The evidence
    is the shape release-3 actually shipped for `cmp-fp151-fp152-entity`: the
    registry line states the entity with no page, and retrieval returned pages
    2, 4, 5, 143, 150 of that document — never 3. `_invalid_citations` flags
    the page the answer invented and passes the '[doc, cover pages]' form the
    fallback dictates, which `verify.parse_citations` also reads as a
    document-bearing pointer ('cover'), not as a malformed page-only bracket."""
    doc = "124_gcf-b27-02-add11"
    hits = [_Hit(doc, p) for p in (2, 4, 5, 143, 150)]
    pages = _note_pages([_REG151])
    assert (doc, 5) in pages and (doc, 3) not in pages   # only the money page

    invented = f"The accredited entity is IUCN. [{doc}, p. 3]"
    assert _invalid_citations(invented, hits, pages) == [f"{doc}… p.3"]

    dictated = f"The accredited entity is IUCN. [{doc}, cover pages]"
    assert _invalid_citations(dictated, hits, pages) == []
    (c,) = verify.parse_citations(dictated)
    assert c.doc.startswith(doc) and c.kind == "cover" and c.page is None

    # the money arm of the same line still cites its printed page, and passes
    money = f"FP151 requests **USD 18.5 million** [{doc}, p. 5]"
    assert _invalid_citations(money, hits, pages) == []


def test_the_hedge_wording_the_prompt_asks_for_is_glue_not_a_claim():
    """The cite-or-hedge rule only helps if the hedge it dictates leaves the
    denominator: `verify.claim_kind` must read it as glue. Pinned for both
    languages, since the same rules ship in French answers."""
    for hedge in ("The retrieved excerpts do not state FP151's co-financing.",
                  "Les extraits ne mentionnent pas le cofinancement du FP151.",
                  "Retrieval did not surface a figure for FP151."):
        assert verify.claim_kind(hedge) is None, hedge


def test_every_bracket_form_the_prompt_dictates_parses_to_a_pointer():
    """Each example the rules print must survive `verify.parse_citations` as a
    document-bearing pointer. A page with no id ('[p. 5]') parses 'malformed'
    and `verify._scopes` reports it as never-retrieved, so no rule may ever
    teach the page-only form — which is why the page-uncertain fallback in
    CORE drops the PAGE and keeps the id, never the other way round."""
    forms = ["[01_gcf-b42-02-add17, p. 5]",       # CORE, the sentence form
             "[12_doc, p. 5]",                    # REGISTRY, page from a note
             "[12_doc, cover pages]",             # REGISTRY, page-less (ruling 5)
             "[19_a, p. 5; 20_b, p. 6]"]          # a chained two-document bracket
    for form in forms:
        cits = verify.parse_citations(form)
        assert cits, form
        for c in cits:
            assert c.doc, (form, c)
            assert c.kind in ("page", "cover", "doc"), (form, c.kind)
    assert [c.page for c in verify.parse_citations(forms[-1])] == [5, 6], \
        "a page belongs to the nearest preceding id, not the bracket's first"
