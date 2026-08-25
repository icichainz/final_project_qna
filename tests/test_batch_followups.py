"""Follow-up batch from the gpt-5.2 production battery:

A. Year/board aggregates answered from the registry (retrieval never surfaces
   all of a year's proposals, so excerpt-scoped notes made the model refuse).
B. FP zero-padding: 'FP86' must reach the corpus's one zero-padded doc (fp086).
C. Citation-bracket parsing (pages belong to the nearest preceding doc id) and
   explicit in-message language requests beating wordlist statistics.
"""
import hashlib
import json
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gcf_qna.rag import registry  # noqa: E402
from gcf_qna.rag.lexical import fp_variants  # noqa: E402
from gcf_qna.rag.retrieve import Hit, Retriever, _doc_match  # noqa: E402


# --- B: fp zero-padding ----------------------------------------------------

def test_fp_variants():
    assert fp_variants("fp86") == ["fp086", "fp86"]
    assert fp_variants("fp086") == ["fp086", "fp86"]
    assert fp_variants("fp274") == ["fp274"]
    assert fp_variants("b42") == ["b42"]      # non-FP tokens pass through


def test_doc_match_zero_padding():
    doc = "189_12-status-approved-fps-adding-host-countries-respect-fp086-gcf-b37-07"
    assert _doc_match(doc, "fp86")
    assert _doc_match(doc, "fp086")
    assert not _doc_match("02_gcf-b42-02-add16-funding-proposal-package-fp274", "fp86")


class FakeEmbedder:
    def encode(self, texts, **kw):
        out = []
        for t in texts:
            rng = np.random.default_rng(int(hashlib.sha1(t.encode()).hexdigest()[:8], 16))
            v = rng.standard_normal(16).astype("float32")
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


PADDED_CHUNKS = [
    {"doc_id": "189_12-status-approved-fps-fp086-gcf-b37-07", "page": 1,
     "text": "green cities facility host countries FP086"},
    {"doc_id": "02_gcf-b42-02-add16-funding-proposal-package-fp274", "page": 8,
     "text": "total financing information for the project"},
]


@pytest.fixture
def padded_retriever(tmp_path):
    emb = FakeEmbedder()
    index = faiss.IndexFlatIP(16)
    index.add(emb.encode([c["text"] for c in PADDED_CHUNKS]))
    return Retriever(index, PADDED_CHUNKS, emb, index_dir=tmp_path)


def test_unpadded_fp_resolves_and_routes(padded_retriever):
    hits, conf = padded_retriever.search_with_confidence("What is FP86 about?", 3)
    assert conf == 1.0                       # variant group resolves
    assert any("fp086" in h.doc_id for h in hits)


def test_unresolvable_fp_still_degrades(padded_retriever):
    _, conf = padded_retriever.search_with_confidence("What is FP999 about?", 3)
    assert conf < 1.0


# --- A: registry-powered year/board notes ----------------------------------

FAKE_REGISTRY = {
    "124_gcf-b27-02-add11": {"fp": 151, "year": 2020, "board": 27,
                             "gcf_financing": "18.5 M USD"},
    "123_gcf-b27-02-add12": {"fp": 152, "year": 2020, "board": 27,
                             "gcf_financing": "150 M USD"},
    "02_gcf-b42-02-add16-funding-proposal-package-fp274":
        {"fp": 274, "year": 2025, "board": 42},
}


@pytest.fixture
def fake_registry(monkeypatch):
    monkeypatch.setattr(registry, "_cache", FAKE_REGISTRY)
    # ... and an EMPTY v2: the year note now prefers the v2 canonical figure,
    # so a v1-only fixture that left _cache_v2 alone would quietly read the
    # real data/registry_v2.json and print the corpus's money for these
    # stems. Empty means "v2 knows nothing here" -> the v1 fallback, which is
    # what these cases are pinning.
    monkeypatch.setattr(registry, "_cache_v2", {})
    yield


def test_year_note_lists_registry_rows(fake_registry):
    from gcf_qna.app.chainlit_app import _year_assist
    hits, note = _year_assist("Which proposals were approved in 2020?", [])
    assert "FP151 (18.5 M USD GCF)" in note and "FP152 (150 M USD GCF)" in note
    assert "2 proposals" in note
    assert "authoritative" in note           # the model may answer from it


def test_year_note_registry_unavailable(monkeypatch):
    monkeypatch.setattr(registry, "load",
                        lambda: (_ for _ in ()).throw(FileNotFoundError()))
    from gcf_qna.app.chainlit_app import _year_assist
    _, note = _year_assist("proposals from 2020?", [])
    assert "state this limit" in note        # never a definitive claim
    assert "no registered proposals" not in note


def test_board_range_note():
    from gcf_qna.app.chainlit_app import _board_range_note
    note = _board_range_note("Does the corpus contain proposals from B.44?")
    assert "B.44 is not in this corpus" in note
    assert "B.11 (2015) through B.43 (2025)" in note
    assert _board_range_note("What was decided at B.42?") is None
    assert _board_range_note("no board mentioned") is None


# --- A2: the year note's money, and the sum it must refuse (F11/P4) --------
#
# Measured: the note printed v1's `gcf_financing` raw. For FP153 that string
# is "28,654 million USD" — a print whose mantissa and scale word cannot both
# be true — and asked to total the 2020 note the model answered $29.0B
# against a truth near $1.36B (21x); an unprompted 2020-vs-2021 comparison
# came out backwards and the verifier passed it. Two changes answer that: the
# figure now comes from the v2 canonical fact (which marks a print it could
# not parse), and the note forbids the summation outright.

def _cand(raw, value, page=5, section="A.8", status="canonical"):
    return {"raw": raw, "value": value, "currency": "USD", "unit": None,
            "page": page, "section": section, "status": status}


V1_MONEY = {
    "124_gcf-b27-02-add11": {"fp": 151, "year": 2020, "board": 27,
                             "gcf_financing": "18.5 M USD"},
    # the OCR-garbled print, exactly as schema 1 stored it
    "126_gcf-b27-02-add14": {"fp": 153, "year": 2020, "board": 27,
                             "gcf_financing": "28,654 million USD"},
    # schema 1 never read this one; v2 did
    "125_gcf-b27-02-add10-rev01": {"fp": 150, "year": 2020, "board": 27},
}


@pytest.fixture
def year_money(monkeypatch):
    monkeypatch.setattr(registry, "_cache", V1_MONEY)
    monkeypatch.setattr(registry, "_cache_v2", {
        "124_gcf-b27-02-add11": {"fp": 151, "facts": {
            # v2 disagrees with the v1 text: v2 is the one that gets printed
            "gcf_funding_requested": [_cand("18,500,000 USD", 18_500_000.0)]}},
        "126_gcf-b27-02-add14": {"fp": 153, "facts": {
            "gcf_funding_requested": [_cand("28,654 million USD", None)]}},
        "125_gcf-b27-02-add10-rev01": {"fp": 150, "facts": {
            "gcf_funding_requested": [_cand("256.48 million USD", 256_480_000.0)]}},
    })
    yield


def test_year_note_money_is_the_v2_canonical_print(year_money):
    from gcf_qna.app.chainlit_app import _year_assist
    _, note = _year_assist("Which proposals were approved in 2020?", [])
    assert "FP151 (18,500,000 USD GCF)" in note      # v2 print, not v1's text
    assert "18.5 M USD" not in note
    assert "FP150 (256.48 million USD GCF)" in note  # v2 fills a v1 gap
    # printed as printed: the per-proposal figure is never a reformatted float
    # (the note's one computed number is the 'Computed total' sentence below,
    # which is labelled as computed and never quotes a proposal's figure)
    assert "18500000" not in note and "1.85e" not in note
    assert "FP151 (18,500,000.00" not in note


def test_a_print_v2_could_not_parse_is_quoted_and_flagged(year_money):
    """FP153's single unusable print is the whole of the 21x error."""
    from gcf_qna.app.chainlit_app import _year_assist
    _, note = _year_assist("Which proposals were approved in 2020?", [])
    assert 'FP153 ("28,654 million USD" GCF, unit as printed is ambiguous)' in note
    # the same words registry._money_bit uses, so one figure reads one way
    assert "unit as printed is ambiguous" in note


def test_year_note_falls_back_to_v1_when_v2_has_no_canonical(monkeypatch):
    """'Stated somewhere but not in a template section' is not a canonical
    fact: the note keeps the v1 string rather than promoting a candidate."""
    monkeypatch.setattr(registry, "_cache", V1_MONEY)
    monkeypatch.setattr(registry, "_cache_v2", {
        "124_gcf-b27-02-add11": {"fp": 151, "facts": {
            "gcf_funding_requested": [
                _cand("99 M USD", 99_000_000.0, status="supporting")]}}})
    from gcf_qna.app.chainlit_app import _year_assist
    _, note = _year_assist("Which proposals were approved in 2020?", [])
    assert "FP151 (18.5 M USD GCF)" in note
    assert "99 M USD" not in note


def test_year_note_survives_an_unreadable_v2(monkeypatch, fake_registry):
    """The year note is the answer to a year question; a broken v2 file may
    cost it the better figure, never the note."""
    monkeypatch.setattr(registry, "load_v2",
                        lambda: (_ for _ in ()).throw(ValueError("bad json")))
    from gcf_qna.app.chainlit_app import _year_assist
    _, note = _year_assist("Which proposals were approved in 2020?", [])
    assert "FP151 (18.5 M USD GCF)" in note          # v1 text, unharmed


@pytest.mark.parametrize("q", [
    "Which proposals were approved in 2020?",        # detailed, prints money
    "Which proposals were approved since 2018?",     # wide span, counts only
])
def test_the_year_note_forbids_the_sum(q, fake_registry):
    from gcf_qna.app.chainlit_app import _year_assist, _NO_SUM_RULE
    _, note = _year_assist(q, [])
    assert _NO_SUM_RULE in note
    assert "MUST NOT be summed" in note and "refusing the sum" in note


def test_the_no_sum_rule_publishes_no_page_and_no_document():
    """The note-page scope readers credit a cited page to a document named on
    the SAME line; the year note is one line, so a sentence that parsed as a
    doc id or a '(p.N,' pointer would hand the model a page it never saw."""
    import re

    from gcf_qna.app import chainlit_app as app
    from gcf_qna.rag import verify
    assert app._note_pages([app._NO_SUM_RULE]) == set()
    assert verify.note_page_scopes(app._NO_SUM_RULE) == []
    assert not re.search(r"[\[(][0-9]{1,3}_", app._NO_SUM_RULE)
    assert not re.search(r"\(p\.\d", app._NO_SUM_RULE)


# --- A2b: the one total the note is allowed to state (P4/F11) --------------
#
# The A2 refusal above is right about the printed strings and wrong about the
# question. P4 asked which of 2020 and 2021 requested more GCF funding and got
# the direction BACKWARDS; F11 asked for the sums and got 21x/35x. Neither
# failure is fixable by refusing harder: the direction exists, the corpus knows
# it, and registry v2 holds a normalised float for every print it could parse.
# So the note computes the total itself — same currency, unambiguous prints
# only, everything left out named — and _NO_SUM_RULE licenses THAT figure while
# still forbidding the model's own arithmetic.

#: P4's shape: two years, one comparison, no arithmetic the model may do.
P4_QUESTION = "Which year received more GCF funding in total, 2020 or 2021?"


def _independent_usd_total(year: int):
    """(total, n) re-summed here from the SAME public accessor the note uses.

    Deliberately a second implementation rather than a call into the app's:
    a test that imports the summer it is checking proves the note prints what
    _usd_total returned, not that either of them is the corpus's total.
    """
    total, n = 0.0, 0
    for row in registry.by_year(year):
        if not row.get("fp"):
            continue
        cand = registry.canonical(row["doc_id"], "gcf_funding_requested")
        if (cand and cand.get("value") is not None
                and (cand.get("currency") or "").upper() == "USD"):
            total += float(cand["value"])
            n += 1
    return total, n


def _year_rows(year: int):
    return [r for r in registry.by_year(year) if r.get("fp")]


def test_the_year_note_computes_a_same_currency_total(year_money):
    """Computed from the v2 FLOATS: 18,500,000 + 256,480,000. FP153's
    '28,654 million USD' — the whole of the 21x error — is excluded and said
    to be excluded, not silently dropped."""
    from gcf_qna.app.chainlit_app import _year_assist
    _, note = _year_assist("Which proposals were approved in 2020?", [])
    assert "Computed total for 2020" in note
    assert "2 of the 3 proposals state their GCF request as an unambiguous " \
           "USD figure, and those 2 total USD 274,980,000" in note
    assert "excluded from this total: FP153 (unit as printed is ambiguous)" in note
    # 28,654 million USD would have added ~28.65bn had the strings been parsed
    assert "28,654,000,000" not in note and "28,928,980,000" not in note


def test_a_year_with_nothing_summable_gets_no_total_at_all(fake_registry):
    """The fixture's v2 is empty, so every figure is a v1 string: no float,
    no total. Silence beats a total over two rows and a shrug."""
    from gcf_qna.app.chainlit_app import _year_assist
    _, note = _year_assist("Which proposals were approved in 2020?", [])
    assert "Computed total for" not in note
    assert "FP151 (18.5 M USD GCF)" in note          # the listing is untouched


def test_p4s_question_now_carries_its_own_answer():
    """P4 answered '2020'. The truth is 2021, by roughly 2x — and after this
    change the note states both totals, so the direction is READ, not guessed.

    The two figures are re-derived here from registry.canonical()['value'],
    never from the printed strings the note also carries.
    """
    from gcf_qna.app.chainlit_app import _year_assist, _usd_amount
    _, note = _year_assist(P4_QUESTION, [])
    totals = {}
    for year in (2020, 2021):
        total, n = _independent_usd_total(year)
        rows = _year_rows(year)
        assert (f"Computed total for {year} (computed by the system from the "
                f"registry's normalised values, NOT by adding the strings "
                f"above): {n} of the {len(rows)} proposals state their GCF "
                f"request as an unambiguous USD figure, and those {n} total "
                f"{_usd_amount(total)}") in note, year
        totals[year] = total
    # the direction P4 inverted, now derivable from the note alone
    assert totals[2021] > totals[2020]
    assert totals[2021] / totals[2020] > 2
    # and the figures themselves, pinned against the checksummed registry
    assert "those 19 total USD 1,157,208,843.80" in note
    assert "those 25 total USD 2,395,398,247.26" in note


@pytest.mark.parametrize("year", [2020, 2021])
def test_the_exclusion_list_is_part_of_the_total(year):
    """A total whose coverage is invisible is §7.1's truncated country list
    again ('five', meaning five of 44). Every proposal the sum could not take
    is named with the reason, or counted when the reason is silence."""
    from gcf_qna.app.chainlit_app import _year_total_line
    rows = _year_rows(year)
    line = _year_total_line(year, rows)
    clause = line.split("excluded from this total:")[1]
    silent = 0
    for row in rows:
        cand = registry.canonical(row["doc_id"], "gcf_funding_requested")
        named = bool(re.search(rf"FP{row['fp']}\b", clause))
        if (cand and cand.get("value") is not None
                and (cand.get("currency") or "").upper() == "USD"):
            assert not named, f"FP{row['fp']} is in the total AND excluded"
        elif not (cand and cand.get("raw")) and not row.get("gcf_financing"):
            silent += 1                       # counted, not named: it says so
            assert not named
        else:
            assert named, f"FP{row['fp']} left the total unannounced"
    assert (f"{silent} proposals stating no figure" in clause) or silent == 0


def test_the_recorded_exclusion_lists():
    """The deliverable, spelled out: what the two totals leave behind."""
    from gcf_qna.app.chainlit_app import _year_total_line
    line20 = _year_total_line(2020, _year_rows(2020))
    assert ("excluded from this total: FP132, FP138 (EUR), FP153 (unit as "
            "printed is ambiguous), FP142, FP150 (figure printed above but "
            "not normalised), 6 proposals stating no figure — the figures "
            "listed above for them stand as printed.") in line20
    line21 = _year_total_line(2021, _year_rows(2021))
    assert ("excluded from this total: FP176 (EUR), FP162, FP168 (unit as "
            "printed is ambiguous) — the figures listed above for them "
            "stand as printed.") in line21


def test_the_total_stops_at_two_years(monkeypatch):
    """_TOTAL_YEARS_MAX: 'A vs B' is the shape this answers. A third year's
    listing plus a third exclusion list stops being a note and starts being a
    report — and the per-proposal listing above it is unaffected either way."""
    from gcf_qna.app.chainlit_app import _year_assist
    rows, v2 = {}, {}
    for i, year in enumerate((2020, 2021, 2022)):
        doc = f"9{i}_gcf-b{25 + i * 3}-02-add01"
        rows[doc] = {"fp": 200 + i, "year": year, "board": 25 + i * 3}
        v2[doc] = {"fp": 200 + i, "facts": {"gcf_funding_requested":
                                            [_cand("1,000,000 USD", 1e6)]}}
    monkeypatch.setattr(registry, "_cache", rows)
    monkeypatch.setattr(registry, "_cache_v2", v2)
    _, two = _year_assist("Which was bigger, 2020 or 2021?", [])
    assert two.count("Computed total for") == 2
    _, three = _year_assist("Compare 2020, 2021 or 2022.", [])
    assert "2022 — 1 proposals: FP202" in three          # listing unaffected
    assert "Computed total for" not in three


def test_the_no_sum_rule_licenses_the_notes_total_in_the_sentence_that_forbids_the_models():
    """The licence and the prohibition share one sentence ON PURPOSE. A rule
    that grants the exception here and re-forbids self-summation two sentences
    later is a rule with a gap in the middle of it, and F11's $29.0B is what
    falls through that gap."""
    from gcf_qna.app.chainlit_app import _NO_SUM_RULE
    licensing = [s for s in _NO_SUM_RULE.split(". ") if "Computed total" in s]
    assert len(licensing) == 1, "the exception must not be its own paragraph"
    sentence = licensing[0]
    assert "quote that total with the coverage and the exclusions it states" \
        in sentence
    assert "do NOT add, subtract, extend or convert any figure yourself" \
        in sentence
    assert "a total this note does not print is a total the answer does not " \
           "have" in sentence
    # …and the original refusal is intact, not softened into the exception
    assert "MUST NOT be summed, totalled or averaged" in _NO_SUM_RULE
    assert "refusing the sum" in _NO_SUM_RULE


def test_the_computed_total_publishes_no_page_and_no_document():
    """Mirrors the _NO_SUM_RULE safety test: the note-page readers credit a
    cited page to a document named on the SAME line, and the year note is one
    line — a computed sentence that parsed as a doc id or a '(p.N' pointer
    would hand the model a page it never saw, attached to a number no
    document prints."""
    from gcf_qna.app import chainlit_app as app
    from gcf_qna.rag import verify
    line = app._year_total_line(2020, _year_rows(2020))
    assert app._note_pages([line]) == set()
    assert verify.note_page_scopes(line) == []
    assert not re.search(r"[\[(][0-9]{1,3}_", line)
    assert not re.search(r"\(p\.\d", line)
    # and the whole extended note still publishes nothing either — including
    # its 'Retrieved excerpts dated …' tail, which is built from _doc_label
    # and now carries an FP number and a page next to every stem
    dated = [Hit(text="", doc_id="124_gcf-b27-02-add11", score=1.0, page=84),
             Hit(text="", doc_id="123_gcf-b27-02-add12", score=1.0, page=76)]
    _, note = app._year_assist(P4_QUESTION, dated)
    assert "124_gcf-b27-02-add11, p. 84 — FP151, B.27, 2020" in note
    assert app._note_pages([note]) == set()
    assert verify.note_page_scopes(note) == []
    # the same header inside a citation bracket parses as the stem and the
    # page it always did — the identifier is not a page and not a document
    assert [(c.doc, c.page) for c in
            verify.parse_citations("[124_gcf-b27-02-add11, p. 84 — FP151, "
                                   "B.27, 2020]")] == \
        [("124_gcf-b27-02-add11", 84)]


@pytest.mark.parametrize("cid", ["agg-2020-count", "agg-2020-range",
                                 "agg-2020-largest", "fr-agg-2020",
                                 "fr-agg-2018"])
def test_the_extended_note_still_answers_the_year_gold_cases(cid):
    """The gold regexes these cases score on must still be satisfiable from
    the note — including agg-2020-largest, whose FP150 is EXCLUDED from the
    computed total and whose printed 256.48 million therefore has to survive
    the exclusion clause that names it."""
    import eval_answers as ev

    from gcf_qna.app.chainlit_app import _year_assist
    case = {c["id"]: c for c in ev.load_cases(ev.DEFAULT_CASES)}[cid]
    _, note = _year_assist(case["question"], [])
    assert note is not None
    for pat in case["expect"]["must_contain"]:
        assert re.search(pat[3:], note), f"{cid}: {pat} not answerable"


# --- D: the identifier a discovery answer is scored on (disc-subnational-pair)
#
# 0.60 in six consecutive releases, and the record says why. Retrieval was
# PERFECT: rank 1, both documents inside the top 10, retrieval_score 1.0. No
# note fired (`notes_used == {}`) — the question names no year, no board code
# and no FP id, so nothing in registry_note/_year_assist/_extend_registry_note
# had a trigger, and the top-ranked pages are Brazilian no-objection letters
# naming the two ACCREDITED ENTITIES and no identifier. Of the two numbers the
# case is scored on, FP152 is in no excerpt at all and FP151 is in exactly one
# — the last-ranked of ten, in a heading over a block that lists BOTH
# proposals' names, so it cannot even say which document it belongs to. The
# model named both proposals, cited both stems and wrote neither number,
# because writing them would have been a guess. That is an evidence defect,
# not a generation defect, and _doc_label is where a per-document fact the
# model would otherwise have to invent already lives (board and year are
# there for exactly that reason).

def _recorded(case_id: str, release: str = "release-8"):
    for line in (ROOT / "data" / "eval" / f"release_{release}.jsonl").open():
        row = json.loads(line)
        if row["id"] == case_id:
            return row
    raise AssertionError(f"{case_id} not recorded in {release}")


def _context(rec) -> str:
    """The excerpt block, built exactly as chainlit_app and the harness build
    it (`[{_doc_label(...)}] (score …)` + text)."""
    from gcf_qna.app.chainlit_app import _doc_label
    return "\n\n".join(
        f"[{_doc_label(h['doc'], h['page'])}] (score {h['score']:.2f})\n{h['text']}"
        for h in rec["hits"])


@pytest.mark.parametrize("release", ["release-6", "release-7", "release-8"])
def test_the_subnational_defect_reproduces_from_the_record(release):
    """Before: the failing shape, from three recorded releases."""
    rec = _recorded("disc-subnational-pair", release)
    assert rec["checks"]["score"] == 0.6
    assert rec["retrieval"]["rank"] == 1 and rec["retrieval"]["cover10"]
    assert rec["retrieval_score"] == 1.0            # retrieval did its job
    assert (rec.get("notes_used") or {}) == {}      # nothing computed fired
    assert rec["plan"] == [{"q": rec["plan"][0]["q"], "doc": None}]
    # no doc tag and no FP id in the resolved plan -> _extend_registry_note
    # has nothing to resolve, which is why no registry line was emitted
    from gcf_qna.app.chainlit_app import _turn_doc_ids
    assert _turn_doc_ids(rec["plan"]) == []
    # Neither identifier is available to the answer. FP152 is in no excerpt
    # at all. FP151 is in exactly ONE — the LAST-ranked of ten, an ITAP-reply
    # heading on a page that lists BOTH proposals' names under one
    # 'Proposal name' block, so it does not even establish which document is
    # which. The answer names both proposals, cites both stems, and states
    # neither number: the evidence it was given contained one of the two, in
    # the one place least able to attribute it.
    with_151 = [i for i, h in enumerate(rec["hits"])
                if re.search(r"FP\s?151", h["text"])]
    assert with_151 == [len(rec["hits"]) - 1]
    assert "Proposal name" in rec["hits"][with_151[0]]["text"]
    assert not any(re.search(r"FP\s?152", h["text"]) for h in rec["hits"])
    assert not re.search(r"FP\s?15[12]", rec["answer"])


def test_the_subnational_case_is_now_answerable_from_the_context():
    """After: every must_contain regex of the failing case is satisfiable
    from the excerpt block the app builds for the recorded hits.

    Same proof shape as the coverage note's recorded-turn replay: the model
    is not asserted to be right, it is given the thing it was missing.
    """
    rec = _recorded("disc-subnational-pair")
    context = _context(rec)
    for pat in rec["expect"]["must_contain"]:
        assert re.search(pat[3:], context), f"{pat} still not in the context"
    assert "124_gcf-b27-02-add11, p. 84 — FP151, B.27, 2020" in context
    assert "123_gcf-b27-02-add12, p. 76 — FP152, B.27, 2020" in context


def test_the_header_identifier_is_the_registrys_and_never_a_parse():
    """'124_gcf-b27-02-add11' contains no 151 and never will: the B.27 stems
    carry no FP number, so this has to be a lookup. An id the registry does
    not know gets no FP and no em dash it did not already earn."""
    from gcf_qna.app.chainlit_app import _doc_label, _registry_fp
    assert _registry_fp("124_gcf-b27-02-add11") == 151
    assert "151" not in "124_gcf-b27-02-add11"
    assert _registry_fp("no-such-document") is None
    assert _doc_label("no-such-document", 2) == "no-such-document, p. 2"


def test_the_identifier_survives_an_unreadable_registry(monkeypatch):
    """The header is the citation: a broken registry may cost it the FP
    number, never the label."""
    from gcf_qna.app.chainlit_app import _doc_label
    monkeypatch.setattr(registry, "load",
                        lambda: (_ for _ in ()).throw(FileNotFoundError()))
    assert _doc_label("124_gcf-b27-02-add11", 84) == \
        "124_gcf-b27-02-add11, p. 84 — B.27, 2020"


def test_the_other_discovery_cases_see_no_behaviour_change():
    """The other eight discovery cases score 1.00 and must keep doing so.
    Their questions still fire NO note (nothing about the note triggers
    changed), and their headers gain the registry's own FP number and
    nothing else — the same number `registry._fmt` already prints for them.
    """
    from gcf_qna.app import chainlit_app as app
    from gcf_qna.boards import board_of, year_of
    rows = [json.loads(line) for line in
            (ROOT / "data" / "eval" / "release_release-8.jsonl").open()]
    others = [r for r in rows if r.get("class") == "discovery"
              and r["id"] != "disc-subnational-pair"]
    assert len(others) == 8
    for rec in others:
        q = rec["question"]
        assert rec["checks"]["score"] == 1.0, rec["id"]        # premise
        # zero behaviour change on every note path this turn can reach
        assert app._year_assist(q, [])[1] is None, rec["id"]
        assert app._board_range_note(q) is None, rec["id"]
        assert app._corpus_coverage_note(q) is None, rec["id"]
        for hit in rec["hits"]:
            label = app._doc_label(hit["doc"], hit["page"])
            # everything the header carried before is still there…
            assert label.startswith(f"{hit['doc']}, p. {hit['page']}")
            board, year = board_of(hit["doc"]), year_of(hit["doc"])
            if board and year:
                assert f"B.{board}, {year}" in label
            # …plus the registry's FP for this stem, exactly, or nothing
            fp = (registry.load().get(hit["doc"]) or {}).get("fp")
            if fp:
                assert re.search(rf"— FP{fp}\b", label), label
            else:
                assert "FP" not in label, label


# --- A3: 'B.<n>' is a board code OR a template heading (H5/P6) -------------

@pytest.mark.parametrize("q,why", [
    ("What does section B.3 of FP172 say?", "the recorded P6 question"),
    ("Which section B.3 commitments does FP172 list?", "'section' anywhere"),
    ("Que dit la rubrique B.3 de FP172 ?", "the French heading word"),
    ("What does § B.3 cover?", "the section sign"),
    ("What does B.2(a) of FP172 report?", "a paragraph letter: never a board"),
    ("What is in B.3?", "unframed and ambiguous - no definitive denial"),
    ("Which board approved the B.2(a) figure for FP172?",
     "a board word does not turn a lettered heading into a meeting"),
])
def test_low_numbers_do_not_get_the_out_of_range_note(q, why):
    from gcf_qna.app.chainlit_app import _board_range_note
    assert _board_range_note(q) is None, why


@pytest.mark.parametrize("q", [
    "What was approved at B.3?",
    "Which proposals did the board approve at B.3?",
    "Qu'est-ce qui a été approuvé lors de la réunion B.3 ?",
])
def test_an_explicitly_claimed_low_board_is_still_denied(q):
    """B.1-B.10 are real GCF meetings (2012-2015) and genuinely outside this
    corpus, so a question that frames one AS a meeting keeps the definitive
    note - the guard removes the false positives, not the true ones."""
    from gcf_qna.app.chainlit_app import _board_range_note
    note = _board_range_note(q)
    assert note is not None and "B.3 is not in this corpus" in note


def test_out_of_range_boards_are_unchanged_above_the_section_range():
    """No heading is numbered above 10, so nothing above it is ambiguous -
    including next to the word 'section'."""
    from gcf_qna.app.chainlit_app import _board_range_note
    for q in ("Which funding proposals were approved at B.44?",
              "What does section B.44 say?"):
        assert "B.44 is not in this corpus" in _board_range_note(q)
    assert _board_range_note("What was decided at B.30?") is None
    assert _board_range_note("What does document GCF/B.42/02/Add.16 contain?") is None


def test_several_codes_read_independently():
    from gcf_qna.app.chainlit_app import _board_range_note
    assert "B.9" not in _board_range_note("What about B.9 and B.44?")
    both = _board_range_note("Which proposals were approved at B.9 and B.44?")
    assert "B.9, B.44 are not in this corpus" in both     # numeric order, plural


# --- C: citation-bracket attribution + explicit language -------------------

HITS = [Hit(text="", doc_id="102_gcf-b30-02-add05", score=1.0, page=5),
        Hit(text="", doc_id="103_gcf-b30-03-add04", score=1.0, page=6)]


def test_chained_bracket_pages_attributed_correctly():
    from gcf_qna.app.chainlit_app import _invalid_citations
    ans = "FP173 asks more [102_gcf-b30-02-add05, p. 5; 103_gcf-b30-03-add04, p. 6]."
    assert _invalid_citations(ans, HITS) == []   # was: false-flagged 102 p.6


def test_invented_page_still_flagged():
    from gcf_qna.app.chainlit_app import _invalid_citations
    ans = "See [102_gcf-b30-02-add05, p. 5; 103_gcf-b30-03-add04, p. 99]."
    bad = _invalid_citations(ans, HITS)
    assert len(bad) == 1 and bad[0].startswith("103_") and "p.99" in bad[0]


def test_unretrieved_doc_pages_skipped():
    from gcf_qna.app.chainlit_app import _invalid_citations
    ans = "Registry says [999_some-other-doc, p. 12]."
    assert _invalid_citations(ans, HITS) == []


def test_explicit_language_request_wins():
    from gcf_qna.app.chainlit_app import _detect_lang
    assert _detect_lang("Now back to the first one — which country is it in, "
                        "and présente ta réponse en français.") == "French"
    assert _detect_lang("Réponds en anglais s'il te plaît, quel est le budget ?") == "English"
    assert _detect_lang("Which country is FP172 in?") == "English"
    assert _detect_lang("Quel est le financement total ?") == "French"


# --- E: ruling 10 — board→year facts become citable evidence ----------------
#
# `agg-2021-boards` answered "B.28 (2021)" from the prompt's YEAR_BLOCK board
# table and cited excerpts that print no such string: unsupported BY
# CONSTRUCTION, 2 rows in release-7. The owner ruled (2026-08-26) for putting
# the mapping into evidence rather than for hedged phrasing, so the year note
# now prints the boards of every year it lists — deterministically, from
# BOARD_YEARS — and prints each retrieved excerpt's own board/year on its own
# line, which is the half `verify.build_evidence` can attribute per document.

def _year_note(question, hits=()):
    from gcf_qna.app.chainlit_app import _year_assist
    return _year_assist(question, list(hits))[1]


@pytest.mark.parametrize("question,year", [
    ("Which proposals were approved in 2021?", 2021),   # detailed arm
    ("Which proposals were approved in 2022?", 2022),   # detailed arm
    ("Which proposals were approved since 2018?", 2022),  # wide-span arm
])
def test_every_populated_year_line_names_its_own_boards(question, year):
    """Derived, never hardcoded: the expectation is recomputed from
    BOARD_YEARS, so a board table edit moves the note and this test together."""
    from gcf_qna.boards import BOARD_YEARS
    boards = ", ".join(f"B.{b}" for b, y in sorted(BOARD_YEARS.items())
                       if y == year)
    note = _year_note(question)
    line = next(l for l in note.split("\n") if l.startswith(f"{year} — "))
    assert line.endswith(f" Boards in {year}: {boards}.")


def test_the_boards_are_spelled_out_never_spanned():
    """The ruling's illustration writes 'boards B.28–B.30'. A dash prints the
    two ends and hides everything between them, and `verify._check_years`
    matches board TOKENS: for 2022 (B.31-B.34) the span form would leave a
    'B.32 (2022)' claim exactly as unverifiable as before the ruling."""
    note = _year_note("Which proposals were approved in 2022?")
    for board in ("B.31", "B.32", "B.33", "B.34"):
        assert board in note, board
    assert "–" not in note.split("Boards in 2022:")[1].split("\n")[0]


def test_each_retrieved_excerpt_carries_its_board_on_its_own_line():
    """The line split is not cosmetic. `verify.build_evidence` walks a note
    block line by line and files each line under the FIRST document id it
    names; one long 'Retrieved excerpts dated 2021: a; b; c' sentence filed
    the whole note under `a` and nothing under `b` or `c`."""
    from gcf_qna.app import chainlit_app as app
    from gcf_qna.rag import verify
    dated = [Hit(text="x", doc_id="111_gcf-b28-02-add11", score=1.0, page=82),
             Hit(text="y", doc_id="110_gcf-b29-02-add01", score=1.0, page=97)]
    note = _year_note("Which board meetings took place in 2021?", dated)
    assert "\n111_gcf-b28-02-add11, p. 82 — FP164, B.28, 2021\n" in note
    ev = verify.build_evidence(dated, [note])
    for doc, board in (("111_gcf-b28-02-add11", "B.28"),
                       ("110_gcf-b29-02-add01", "B.29")):
        assert board in ev[(doc, None)] and "2021" in ev[(doc, None)]
    # and the excerpt's own page still holds the PAGE's text, unmixed with the
    # note line: a computed fact must not be readable as something the page
    # printed (the label carries no '(p.N' pointer, so it keys doc-level)
    assert ev[("111_gcf-b28-02-add11", 82)] == "x"
    assert app._note_pages([note]) == set()


def test_the_recorded_agg_2021_boards_turn_now_verifies():
    """Replay of the release-7 turn the ruling names, offline: the recorded
    plan, the recorded answer, the recorded hits — and the two board→year
    claims that were unsupported by construction.

    BEFORE is the note release-7 actually shipped, so the premise is measured
    rather than asserted; AFTER is the note this module builds today."""
    from gcf_qna.app.chainlit_app import _year_assist
    from gcf_qna.rag import verify
    rec = None
    for line in (ROOT / "data" / "eval" / "release_release-7.jsonl").open():
        r = json.loads(line)
        if r["id"] == "agg-2021-boards":
            rec = r
            break
    assert rec is not None
    hits = [Hit(text=h["text"], doc_id=h["doc"], score=h.get("score", 0.0),
                page=h.get("page")) for h in rec["hits"]]
    claims = verify.extract_claims(rec["answer"])
    boards = [c for c in claims if re.match(r"^\*\*B\.\d\d \(2021\)", c.text)]
    assert len(boards) == 3

    def statuses(note):
        ev = verify.build_evidence(hits, [rec["notes_used"]["registry"], note])
        by_text = {v.claim.text: v.status
                   for v in verify.classify_deterministic(claims, ev)}
        return [by_text[c.text] for c in boards]

    before = statuses(rec["notes_used"]["year"])
    assert before.count(verify.UNSUPPORTED) == 2, before
    after = statuses(_year_assist(rec["question"], hits)[1])
    assert after == [verify.SUPPORTED] * 3, after


# --- F: the comparative licence over the computed year totals ---------------

def test_the_comparative_licence_ships_only_when_two_totals_print():
    """Phase 2's open item. The licence is carried by the note, so it costs
    nothing on the turns it does not apply to — and a rule about 'two totals'
    on a note that prints one would be a rule with no referent."""
    from gcf_qna.app import chainlit_app as app
    two = _year_note("Did 2021 request more GCF funding in total than 2020?")
    assert two.count("Computed total for") == 2
    assert app._COMPARE_RULE in two
    one = _year_note("Which proposals were approved in 2021?")
    assert one.count("Computed total for") == 1
    assert app._COMPARE_RULE not in one
    assert app._COMPARE_RULE not in _year_note("proposals approved since 2018")
    assert app._COMPARE_RULE not in app._corpus_coverage_note(
        "How many funding proposals are in the corpus?")


def test_the_comparative_licence_grants_the_ranking_and_nothing_else():
    """_NO_SUM_RULE licenses QUOTING a computed total and forbids arithmetic on
    it; it never said whether two totals may be RANKED, and release-10's
    l2x-xyear got the direction right by inference on a single sample. The
    licence is now stated — and the three figures the note does not print
    (difference, ratio, sum) are refused in the same sentence, the way
    _NO_SUM_RULE keeps its own exception and prohibition together."""
    from gcf_qna.app.chainlit_app import _COMPARE_RULE
    assert "say which year is larger" in _COMPARE_RULE
    assert "quote both totals with the coverage each states" in _COMPARE_RULE
    assert ("their difference, their ratio and their sum are figures this "
            "note does not print") in _COMPARE_RULE
    assert _COMPARE_RULE.count(". ") == 0, "one sentence, no gap in the middle"


def test_the_two_year_note_still_answers_the_comparative_gold_case():
    """l2x-xyear-2021-vs-2020-total: every regex it scores on stays
    answerable from the note the licence now rides on."""
    import eval_answers as ev

    case = {c["id"]: c for c in ev.load_cases(ev.DEFAULT_CASES)}[
        "l2x-xyear-2021-vs-2020-total"]
    note = _year_note(case["question"])
    for pat in case["expect"]["must_contain"]:
        if pat.startswith("re:") and "yes" in pat:
            continue          # the direction is the ANSWER's, not the note's
        assert re.search(pat[3:], note), f"{pat} not answerable from the note"
