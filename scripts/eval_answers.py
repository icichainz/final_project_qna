#!/usr/bin/env python3
"""Answer-level evaluation against scripts/answer_gold.jsonl (step 0 of the
RAG-correctness plan: freeze a baseline before answer behavior changes).

scripts/eval_retrieval.py answers "did we find the document?".  This answers
"did we say the right thing about it?" — expected documents AND expected
evidence pages, plus, with --answers, the generated answer's behavior
(answer / conflict / abstention), required and forbidden strings, language,
and citation validity.

Four modes
----------
  --retrieval-only   (default) deterministic, no API calls. Runs the same
                     retrieval helpers as the app, with the conductor skipped
                     and the question used verbatim, then scores document
                     recall and evidence-page hit rate. Multi-turn cases are
                     skipped (they need history to resolve) and counted apart.
  --answers          additionally calls the chat model per case, assembling
                     the prompt with chainlit_app's shared helpers (registry,
                     year, board-range and weak-signal notes) and scores it.
                     Per-record metadata names the remaining parity gaps.
  --release          --answers over the WHOLE suite plus the release report:
                     required-field coverage against registry v2, claim
                     groundedness / citation completeness / citation presence
                     from rag.verify (see "metric contract" below),
                     latency p50/p95, tokens and estimated cost. Records to
                     data/eval/release_<label>.jsonl. The verifier defaults to
                     deterministic/no-repair, so it adds no API calls.
  --compare A B      diff two recorded runs, per case and per class.

Two scorers beyond the string checks, both deterministic:

* **required-field coverage** — for a case labelled ``expect.fields``, every
  (document, field) cell must be *addressed*: the answer either prints a value
  matching one of registry v2's candidates for that document+field, or says
  plainly that it is not stated.  Cells whose document has no v2 fact for the
  field cannot be scored either way and are reported apart — they are never a
  pass.
* **claim support** — every claim ``rag.verify`` extracts from the answer is
  classified against the evidence THIS harness assembled for that case (the
  same hits and notes the prompt carried).  No API calls, no judge model.

The retrieval floor (100% document recall on the 30 retrieval gold questions)
is re-checked in-process with --gate, so one index load covers both.

Examples
--------
  python scripts/eval_answers.py --retrieval-only --gate
  python scripts/eval_answers.py --answers --sample 12 --record baseline-sample
  python scripts/eval_answers.py --answers --ids conf-fp274-gcf,abs-fp999
  python scripts/eval_answers.py --release --record release-1
  python scripts/eval_answers.py --release --production-planner \\
      --verifier-mode production --verifier-repair --record repair-shadow
  python scripts/eval_answers.py --compare data/eval/answers_baseline_a.jsonl \\
                                           data/eval/answers_baseline_b.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict, defaultdict
from itertools import zip_longest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Import the app module for its helpers without paying the 730 MB index load
# at import time (same trick as tests/conftest.py); the retriever is then
# fetched explicitly, once, via app.get_retriever().
os.environ["PRELOAD"] = "0"

try:                                    # keys live in .env, never in source
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from gcf_qna import config                                    # noqa: E402
from gcf_qna.app.prompts import assemble                      # noqa: E402
from gcf_qna.rag import planner, registry, verify             # noqa: E402

DEFAULT_CASES = ROOT / "scripts" / "answer_gold.jsonl"
GOLD_SET = ROOT / "scripts" / "gold_set.jsonl"
EVAL_DIR = ROOT / "data" / "eval"

CLASS_ORDER = ["identifier", "compact-id", "board-code", "discovery",
               "comparison", "conflict", "french", "noisy", "abstain",
               "aggregate", "followup"]

# Estimated USD price PER TOKEN for the answer model. THESE ARE ESTIMATES: the
# API does not report prices, provider list prices move, and a run may be
# served by a proxy with its own rates. The release report's dollar figure is
# an order-of-magnitude number, not an invoice. One constant, edited in one
# place — nothing else in this file hard-codes a price.
TOKEN_COST_USD = {"prompt": 1.25 / 1_000_000,        # ~$1.25 per 1M input
                  "completion": 10.00 / 1_000_000}   # ~$10.00 per 1M output

# The plan's citation-precision gate: share of extracted claims that BOTH cite
# evidence and are supported by what they cite (docs/claim-support-execution-
# plan.md, "Metric contract"). Gated on citation completeness only.
CLAIM_SUPPORT_GATE = 0.95

# ---------------------------------------------------------------------------
# metric contract (docs/claim-support-execution-plan.md, "Metric contract")
# ---------------------------------------------------------------------------
# Three numbers, two gates:
#
#   groundedness          some evidence the answer path HELD entails the claim,
#                         cited or not          -> gated
#   citation completeness the claim CITES evidence that entails it: cited AND
#                         SUPPORTED             -> gated
#   citation presence     the claim carries any citation, right or wrong
#                         -> REPORTED, NEVER GATED
#
# Citation presence is the one an answer can raise by appending '[anything,
# p.1]'. It is kept visible precisely so it can never be mistaken for
# completeness again (verdict finding 4: counting it as completeness reported
# 141/165 where the contract gives 91/165).
#
# `--compare` builds one row per entry. Two entries naming the same underlying
# record key would double-count a single regression and print two green deltas
# where one exists (binding decision 4), so the tuple is asserted pairwise
# distinct at import — see _assert_distinct_metric_keys below.
#
#            display name          record block  record key                gated
COMPARED_METRICS = (
    ("field_coverage",             "fields", "coverage",                   True),
    ("groundedness",               "claims", "groundedness_rate",          True),
    ("citation_completeness",      "claims", "citation_completeness_rate", True),
    ("citation_presence",          "claims", "citation_presence_rate",     False),
)

# The verdict flags that mean "held evidence entails this claim even though the
# claim's own citation does not". Groundedness is defined over ALL held
# evidence, so it must read the verifier's all-evidence finding rather than
# re-deriving it from citation state (which is what collapsed groundedness onto
# citation support, verdict finding 5). Public flag surface only — this file
# calls no private verify helper.
GROUNDED_FLAGS = ("value-present-elsewhere",)


def n_over_d(num, den) -> str:
    """A claim metric as `n/d`. Never publish a bare rate: a rate whose
    denominator moved is not a comparison (metric contract)."""
    return f"{int(num)}/{int(den)}"


def gate_threshold(den, gate: float = CLAIM_SUPPORT_GATE) -> int:
    """The exact integer numerator a gate needs — ceil(gate * d), computed in
    integers so no float rounding can hand out a pass (binding decision 5)."""
    den = int(den)
    return -((-int(round(gate * 10000)) * den) // 10000)


def _assert_distinct_metric_keys(metrics=COMPARED_METRICS):
    """Startup assertion: no two compared metrics share a name or a record key.

    A plain `assert` would vanish under `python -O`, and this is the check that
    stops a resurrected duplicate (`citation_support_rate` beside
    `citation_completeness_rate`) from double-counting one regression.
    """
    names = [m[0] for m in metrics]
    keys = [(m[1], m[2]) for m in metrics]
    dup_names = sorted({n for n in names if names.count(n) > 1})
    dup_keys = sorted({f"{b}.{k}" for b, k in keys if keys.count((b, k)) > 1})
    if dup_names or dup_keys:
        raise AssertionError(
            "compared metric keys must be pairwise distinct; duplicates: "
            + ", ".join(dup_names + dup_keys))
    return metrics


_assert_distinct_metric_keys()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
_BEHAVIORS = {"answer", "conflict", "abstain"}


def load_cases(path: Path) -> list:
    """Parse + validate the fixture file. Raises ValueError on a bad row."""
    cases = []
    seen = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            c = json.loads(line)
        except ValueError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from None
        for key in ("id", "class", "lang", "question", "expect"):
            if key not in c:
                raise ValueError(f"{path}:{lineno}: missing '{key}'")
        if c["id"] in seen:
            raise ValueError(f"{path}:{lineno}: duplicate id {c['id']!r}")
        seen.add(c["id"])
        if c["lang"] not in ("en", "fr"):
            raise ValueError(f"{path}:{lineno}: lang must be en|fr")
        e = c["expect"]
        if e.get("behavior", "answer") not in _BEHAVIORS:
            raise ValueError(f"{path}:{lineno}: behavior must be one of {_BEHAVIORS}")
        e.setdefault("behavior", "answer")
        for key in ("docs", "pages", "must_contain", "must_not_contain", "fields"):
            e.setdefault(key, [])
            if not isinstance(e[key], list):
                raise ValueError(f"{path}:{lineno}: expect.{key} must be a list")
        e["pages"] = [int(p) for p in e["pages"]]
        c.setdefault("turns", [])
        cases.append(c)
    return cases


def select(cases: list, ids: str = None, sample: int = None,
           classes: str = None) -> list:
    """--ids wins; --sample takes a deterministic class-stratified round robin
    so a small sample still covers conflict / abstain / French."""
    if ids:
        wanted = [i.strip() for i in ids.split(",") if i.strip()]
        by_id = {c["id"]: c for c in cases}
        missing = [i for i in wanted if i not in by_id]
        if missing:
            raise SystemExit(f"unknown case id(s): {', '.join(missing)}")
        return [by_id[i] for i in wanted]
    if classes:
        keep = {c.strip() for c in classes.split(",")}
        cases = [c for c in cases if c["class"] in keep]
    if sample and sample < len(cases):
        buckets = OrderedDict()
        for c in sorted(cases, key=lambda c: (_class_rank(c["class"]), c["id"])):
            buckets.setdefault(c["class"], []).append(c)
        out, i = [], 0
        while len(out) < sample:
            progressed = False
            for rows in buckets.values():
                if i < len(rows):
                    out.append(rows[i])
                    progressed = True
                    if len(out) == sample:
                        break
            if not progressed:
                break
            i += 1
        return out
    return cases


def _class_rank(cls: str) -> int:
    return CLASS_ORDER.index(cls) if cls in CLASS_ORDER else len(CLASS_ORDER)


# ---------------------------------------------------------------------------
# matchers  (shared by the harness and tests/test_eval_answers.py)
# ---------------------------------------------------------------------------
def matches(pattern: str, text: str) -> bool:
    """'re:<regex>' -> case-insensitive regex search; anything else -> a
    case-insensitive substring test."""
    text = text or ""
    if pattern.startswith("re:"):
        return re.search(pattern[3:], text, re.I | re.S) is not None
    return pattern.lower() in text.lower()


# A refusal leads with the refusal, so only the head of the answer is
# inspected: "the excerpts do not state the co-financing" in paragraph three
# of an otherwise substantive answer is not an abstention.
#
# Every French marker is phrase-bound. A bare 'aucun' scored
# "Il n'y a aucun doute : le FP151 demande 18,5 millions USD" as an
# abstention, which then failed the behavior check on a correct answer.
_ABSTAIN_HEAD = 260
_ABSTAIN_RE = re.compile(
    r"does not exist|do(?:es)? not (?:contain|cover|appear|include)|not exist|"
    r"no such|not found|NOT FOUND|not (?:present |listed |available )?in (?:this|the) corpus|"
    r"cannot (?:find|locate|answer)|could not find|unable to (?:find|answer)|"
    r"no (?:information|record|proposal|funding proposal|data|evidence|mention)|"
    r"outside (?:the|this) corpus|not covered|no board meeting|only covers|"
    r"n'existe pas|introuvable|ne figure pas|pas dans (?:le|ce) corpus|"
    r"ne contient pas|hors (?:du|de ce) corpus|je ne (?:peux|trouve)|pas trouv|"
    r"aucune? (?:proposition|document|information|mention|r[ée]sultat|donnée|"
    r"correspondance|élément|extrait|trace)", re.I)

_CONFLICT_RE = re.compile(
    r"conflict|discrepan|inconsisten|contradict|mismatch|diverg|"
    r"differ(?:s|ent)?\b|two different|whereas|while (?:page|p\.)|does not match|"
    r"disagree|contradictoire|incohéren|diffère|deux (?:montants|chiffres|valeurs)",
    re.I)

# A conflict claim has a shape, not just a keyword: two different values in one
# sentence, or two different page numbers cited close together. Without this,
# "FP151 and FP152 have different accredited entities" — an ordinary
# comparison — counted as conflict-surfacing and made the conflict cases free.
_VALUE_RE = re.compile(r"\d{1,3}(?:[.,\s]\d{3})+|\d+[.,]\d+|\d{4,}")
_PAGE_RE = re.compile(r"\bpp?\.?\s*(\d{1,3})\b", re.I)
_PAGE_WINDOW = 80


def _two_values_in_a_sentence(answer: str) -> bool:
    for sent in re.split(r"(?<=[.!?;:])\s+|\n", answer or ""):
        vals = {re.sub(r"\s", "", v) for v in _VALUE_RE.findall(sent)}
        if len(vals) >= 2:
            return True
    return False


def _two_pages_close_together(answer: str) -> bool:
    """Two DIFFERENT page numbers within _PAGE_WINDOW characters. Requiring
    them to differ keeps an ordinary two-document citation ('… p. 5] and …
    p. 5]') from reading as a page-vs-page contradiction."""
    marks = [(m.start(), m.group(1)) for m in _PAGE_RE.finditer(answer or "")]
    for i, (pos, page) in enumerate(marks):
        for pos2, page2 in marks[i + 1:]:
            if pos2 - pos > _PAGE_WINDOW:
                break
            if page2 != page:
                return True
    return False


def looks_abstained(answer: str) -> bool:
    return bool(_ABSTAIN_RE.search((answer or "")[:_ABSTAIN_HEAD]))


def looks_conflicted(answer: str) -> bool:
    if not _CONFLICT_RE.search(answer or ""):
        return False
    return _two_values_in_a_sentence(answer) or _two_pages_close_together(answer)


def behavior_ok(expected: str, answer: str) -> bool:
    if expected == "abstain":
        return looks_abstained(answer)
    if expected == "conflict":
        return looks_conflicted(answer)
    return not looks_abstained(answer)


def language_ok(lang: str, answer: str) -> bool:
    """Reuses the app's own FR/EN heuristic, so the metric moves with the
    behavior it measures."""
    from gcf_qna.app import chainlit_app as app
    got = app._detect_lang(answer or "")
    if lang == "fr":
        return got == "French"
    return got != "French"


# ---------------------------------------------------------------------------
# retrieval scoring
# ---------------------------------------------------------------------------
def doc_eq(hit_doc: str, expected: str) -> bool:
    """Forgiving stem compare: the index splits some documents into
    '<stem>_0' shards, and the fixture labels the registry stem."""
    a, b = (hit_doc or "").lower(), (expected or "").lower()
    return bool(a) and bool(b) and (a == b or a.startswith(b) or b.startswith(a))


def _page(h) -> int:
    try:
        return int(str(getattr(h, "page", None) or 0))
    except (TypeError, ValueError):
        return 0


def score_retrieval(case: dict, hits: list) -> dict:
    """recall@5 / recall@10 (any expected doc), coverage@10 (all expected
    docs), and the expected-evidence-page hit rate."""
    exp = case["expect"]["docs"]
    out = {"n_hits": len(hits), "docs_expected": len(exp),
           "rank": None, "r5": None, "r10": None, "cover10": None,
           "pages_expected": len(case["expect"]["pages"]),
           "pages_hit": None, "page_rate": None,
           "top_docs": list(dict.fromkeys(h.doc_id for h in hits))[:5]}
    if exp:
        rank = next((i + 1 for i, h in enumerate(hits)
                     if any(doc_eq(h.doc_id, d) for d in exp)), None)
        found = {d for d in exp
                 if any(doc_eq(h.doc_id, d) for h in hits[:10])}
        out["rank"] = rank
        out["r5"] = bool(rank and rank <= 5)
        out["r10"] = bool(rank and rank <= 10)
        out["cover10"] = len(found) == len(exp)
        out["docs_found"] = sorted(found)
    pages = case["expect"]["pages"]
    if pages:
        got = {_page(h) for h in hits[:10]
               if not exp or any(doc_eq(h.doc_id, d) for d in exp)}
        hit = [p for p in pages if p in got]
        out["pages_hit"] = len(hit)
        out["page_rate"] = len(hit) / len(pages)
    return out


def retrieval_score(r: dict) -> float:
    """Single number for --compare: mean of the applicable sub-scores."""
    parts = [v for v in (r.get("r5"), r.get("cover10")) if v is not None]
    parts = [float(v) for v in parts]
    if r.get("page_rate") is not None:
        parts.append(r["page_rate"])
    return sum(parts) / len(parts) if parts else 1.0


# ---------------------------------------------------------------------------
# answer scoring
# ---------------------------------------------------------------------------
def score_answer(case: dict, answer: str, hits: list) -> dict:
    from gcf_qna.app import chainlit_app as app
    e = case["expect"]
    contains = {p: matches(p, answer) for p in e["must_contain"]}
    forbidden = {p: (not matches(p, answer)) for p in e["must_not_contain"]}
    bad_cites = app._invalid_citations(answer or "", hits) if hits else []
    checks = {
        "behavior": behavior_ok(e["behavior"], answer),
        "must_contain": contains,
        "must_not_contain": forbidden,
        "language": language_ok(case["lang"], answer),
        "citations": not bad_cites,
        "bad_citations": bad_cites[:4],
    }
    flags = ([checks["behavior"]] + list(contains.values()) + list(forbidden.values())
             + [checks["language"], checks["citations"]])
    checks["score"] = sum(bool(f) for f in flags) / len(flags)
    checks["pass"] = all(flags)
    return checks


# ---------------------------------------------------------------------------
# required-field coverage  (expect.fields x expect.docs, against registry v2)
# ---------------------------------------------------------------------------
# The fixture labels fields in the corpus's own vocabulary ('gcf_financing');
# registry v2 names some of them differently ('gcf_funding_requested'). The
# money half of that mapping already exists in rag.verify and is BORROWED, not
# copied, so a field renamed there is renamed here too.
_V2_FROM_VERIFY = dict(getattr(verify, "_V2_FIELD", None) or {
    "gcf_financing": "gcf_funding_requested",
    "total_financing": "total_financing",
    "co_financing": "co_financing",
    "duration": "implementation_period",
    "beneficiaries": "beneficiaries_direct"})
FIELD_TO_V2 = {"accredited_entity": "accredited_entity",
               "title": "title",
               "countries": "countries",
               "executing_entity": "executing_entity",
               "project_size": "project_size",
               **_V2_FROM_VERIFY}

# 'we do not have this' said out loud, EN and FR. A cell the answer explicitly
# marks not-stated is covered: the answer addressed the field honestly, which
# is the behavior the prompt asks for when the excerpts are silent. Saying
# nothing at all about it is not.
_NOT_STATED_RE = re.compile(
    r"not (?:explicitly |directly |separately )?(?:stated|specified|given|provided|"
    r"listed|disclosed|reported|available|mentioned|shown|indicated|present|"
    r"broken out|found|retrieved)"
    r"|(?:is|are|was|were) not (?:stated|specified|given|provided|available|shown)"
    r"|(?:does|do|did) not (?:state|specify|give|provide|mention|list|report|"
    r"indicate|contain|include|appear)"
    r"|no (?:figure|amount|value|entity|title|number|country|countries|"
    r"information|data)s?\b[^.]{0,30}\b(?:stated|given|provided|available|"
    r"in the excerpts?|retrieved)"
    r"|unavailable|unknown from the excerpts?|silent on"
    r"|non (?:pr[ée]cis[ée]|indiqu[ée]|sp[ée]cifi[ée]|mentionn[ée]|disponible)"
    r"|pas (?:pr[ée]cis[ée]|indiqu[ée]|sp[ée]cifi[ée]|mentionn[ée]|disponible|"
    r"fourni|[ée]tabli|donn[ée])"
    r"|n['’]est pas (?:pr[ée]cis[ée]|indiqu[ée]|mentionn[ée]|sp[ée]cifi[ée]|disponible)"
    r"|ne (?:pr[ée]cise|mentionne|indique|figure|contient|donne)(?:nt)? pas"
    r"|aucun(?:e)? (?:montant|valeur|information|chiffre|donn[ée]e|mention|"
    r"entit[ée]|pays)", re.I)

# list separators the registry prints inside one candidate string
# ('Africa: Angola; Benin; Botswana'). Commas are NOT split on: a title's comma
# yields fragments generic enough to match any answer.
_CAND_SPLIT_RE = re.compile(r"\s*[;:\n]\s*")

_entity_variants = getattr(verify, "_entity_variants", lambda s: [s])


def field_candidates(doc_id: str, field: str):
    """(v2 field name, candidates) registry v2 records for a document+field.

    Public API only (facts/canonical), canonical first. An empty list is the
    'unscorable' signal: the corpus never published this fact for this
    document, so no answer can be graded against it.
    """
    v2f = FIELD_TO_V2.get(field, field)
    try:
        cands = list(registry.facts(doc_id).get(v2f) or [])
    except Exception:
        return v2f, []
    cands.sort(key=lambda c: 0 if c.get("status") == "canonical" else 1)
    return v2f, cands


def _doc_fp(doc_id: str):
    """The FP number a document carries, for attributing answer text to it."""
    for loader in (getattr(registry, "load_v2", None), getattr(registry, "load", None)):
        try:
            row = (loader() or {}).get(doc_id) if loader else None
        except Exception:
            row = None
        if row and row.get("fp"):
            return int(row["fp"])
    m = _FP_TOKEN_RE.search(doc_id or "")
    return int(m.group(1)) if m else None


def _answer_units(answer: str):
    """(text, citations) for every claim-sized unit, reusing verify's splitter
    so a bullet list and a markdown table are cut the same way here as there."""
    fn = getattr(verify, "_units", None)
    if fn:
        return [(text, cits) for text, _kind, cits, _inh in fn(answer or "")]
    return [(s, verify.parse_citations(s))
            for s in verify.split_sentences(answer or "")]


def doc_scope(answer: str, doc_id: str, fp=None):
    """(the answer text that talks about this document, how it was scoped).

    A comparison answer states one value per document; scoring 'did the answer
    give FP152's GCF amount' against the WHOLE answer would pass on FP151's
    figure. A unit belongs to a document when it cites it, names its FP id, or
    prints its stem. When nothing does, the whole answer is used (flagged):
    the document's own registry values still have to appear, so the fallback
    is lenient about attribution, never about the value.
    """
    keep = []
    for text, cits in _answer_units(answer):
        if any(c.doc and doc_eq(c.doc, doc_id) for c in cits):
            keep.append(text)
        elif fp is not None and str(int(fp)) in fp_ids(text):
            keep.append(text)
        elif doc_id and doc_id.lower()[:24] in (text or "").lower():
            keep.append(text)
    if keep:
        return "\n".join(keep), "document"
    return (answer or ""), "answer"


def _candidate_variants(raw: str) -> list:
    out = []
    for part in [raw] + _CAND_SPLIT_RE.split(raw or ""):
        # parentheses are NOT stripped: '... (IUCN)' is where the acronym
        # variant comes from, and trimming the ')' loses it
        part = (part or "").strip(" .,;\t")
        if not part:
            continue
        for v in _entity_variants(part) or []:
            if len(verify.norm_text(v)) >= 4:
                out.append(v)
    return list(dict.fromkeys(out))


def candidate_stated(cand: dict, text: str) -> bool:
    """Does ``text`` print this registry candidate?

    Figures go through verify's own amount matcher — same separator rules,
    same unit-word handling, same currency guard as the verifier — so
    '18.5 M USD', '18,500,000 USD' and 'USD 18.5 million' are one value here
    exactly as they are there. Everything else is a normalized substring test
    over the candidate's printed forms (acronym, elided title, list items).
    """
    raw = str(cand.get("raw") or "")
    if not raw.strip():
        return False
    want = verify.amounts(raw)
    if want:
        got = verify.amounts(text or "")
        if any(verify.amount_matches(w, g) for w in want for g in got):
            return True
    hay = verify.norm_text(text or "")
    return any(verify.norm_text(v) in hay for v in _candidate_variants(raw))


def score_fields(case: dict, answer: str):
    """Per-(document, field) coverage for a case labelled ``expect.fields``.

    None when the case labels no fields (or no documents): not every class
    carries a field contract, and a missing contract is not a zero.
    """
    fields = list(case["expect"].get("fields") or [])
    docs = list(case["expect"].get("docs") or [])
    if not fields or not docs:
        return None
    cells = []
    for doc in docs:
        scope, how = doc_scope(answer, doc, _doc_fp(doc))
        for field in fields:
            v2f, cands = field_candidates(doc, field)
            cell = {"doc": doc, "field": field, "v2_field": v2f, "scoped": how}
            if not cands:
                cell["status"] = "unscorable"
                cell["why"] = "registry v2 records no candidate for this document+field"
            else:
                hit = next((c for c in cands if candidate_stated(c, scope)), None)
                if hit:
                    cell["status"] = "stated"
                    cell["matched"] = str(hit.get("raw"))[:120]
                    cell["page"] = hit.get("page")
                elif _NOT_STATED_RE.search(scope):
                    cell["status"] = "marked-missing"
                else:
                    cell["status"] = "missed"
                    cell["expected"] = [str(c.get("raw"))[:80] for c in cands[:3]]
            cells.append(cell)
    n = defaultdict(int)
    for c in cells:
        n[c["status"]] += 1
    scorable = len(cells) - n["unscorable"]
    covered = n["stated"] + n["marked-missing"]
    return {"cells": cells, "n_cells": len(cells), "n_scorable": scorable,
            "n_stated": n["stated"], "n_marked_missing": n["marked-missing"],
            "n_missed": n["missed"], "n_unscorable": n["unscorable"],
            "n_covered": covered,
            "coverage": (covered / scorable) if scorable else None}


# ---------------------------------------------------------------------------
# claim support  (the plan's citation-precision gate)
# ---------------------------------------------------------------------------
def score_claims(answer: str, hits, notes=None, verdicts=None, evidence=None):
    """Deterministic claim metrics against THIS turn's own evidence.

    The old citation check asked 'does the answer cite a page we retrieved?',
    which a fabricated figure on a real page passes. This asks whether the
    cited evidence states the claim: verify.extract_claims over the answer,
    verify.build_evidence over the very hits and notes the harness put in the
    prompt, verify.classify_deterministic between them.

    Three separate numbers, per the metric contract (COMPARED_METRICS above):

    * groundedness          — some evidence the turn HELD entails the claim,
                              cited or not.
    * citation completeness — the claim cites evidence that entails it:
                              ``cited AND SUPPORTED``. This is the gated one.
    * citation presence     — the claim carries a bracket, right or wrong.
                              Reported, never gated: a fabricated citation to a
                              document that was never retrieved raises presence
                              and CANNOT raise completeness.

    ``verdicts`` and ``evidence`` let the offline verifier reuse its final
    verdicts without another classification pass. With neither supplied this
    remains the original pure-python scorer: no judge model or repair call.
    """
    blocks = [n for n in (notes or []) if n]
    evidence = evidence if evidence is not None else verify.build_evidence(hits or [], blocks)
    claims = verify.extract_claims(answer or "")
    verdicts = (list(verdicts) if verdicts is not None
                else verify.classify_deterministic(claims, evidence))
    n = defaultdict(int)
    failures = []
    for v in verdicts:
        n[v.status] += 1
        cited = bool(getattr(v.claim, "cited",
                             getattr(v.claim, "citations", [])))
        flags = list(getattr(v, "flags", []) or [])
        supported_here = v.status == verify.SUPPORTED
        # Groundedness reads the verifier's all-evidence finding, never the
        # claim's citation state.
        grounded = supported_here or any(f in flags for f in GROUNDED_FLAGS)
        # Citation completeness is 'cited AND supported'. A bracket alone is
        # citation PRESENCE and is counted under its own name.
        n["citation_present"] += cited
        n["grounded"] += grounded
        n["citation_complete"] += cited and supported_here
        if not supported_here:
            failures.append({"status": v.status, "kind": v.claim.kind,
                             "text": v.claim.text[:160], "reason": v.reason[:160],
                             "grounded": grounded, "cited": cited})
    total = len(verdicts)
    supported = n[verify.SUPPORTED]
    return {
        "claims": total,
        "grounded": n["grounded"],
        "groundedness_rate": (n["grounded"] / total) if total else None,
        "groundedness": n_over_d(n["grounded"], total),
        "citation_complete": n["citation_complete"],
        "citation_completeness_rate": (n["citation_complete"] / total) if total else None,
        "citation_completeness": n_over_d(n["citation_complete"], total),
        "citation_present": n["citation_present"],
        "citation_presence_rate": (n["citation_present"] / total) if total else None,
        "citation_presence": n_over_d(n["citation_present"], total),
        "supported": supported,
        "contradicted": n[verify.CONTRADICTED],
        "unsupported": n[verify.UNSUPPORTED],
        # Legacy verdict tally, kept for pre-split artifacts. NOT a compared
        # metric key — `--compare` reads citation_completeness_rate, so this
        # cannot double-count the same regression (binding decision 4).
        "support_rate": (supported / total) if total else None,
        "evidence_keys": [f"{d}|{p if p is not None else '-'}" for d, p in evidence],
        "failures": failures[:6],
    }


def run_offline_verifier(answer: str, hits, notes=None, client=None,
                         mode: str = "deterministic", allow_repair: bool = False):
    """Run the production verifier entry point with explicit API semantics.

    ``deterministic`` is the default and cannot make an extra API call because
    both the judge and repair are disabled. ``production`` enables the same
    LLM adjudication path used by the app; repair remains an independent,
    opt-in switch. The caller owns the client so tests can remain network-free.
    """
    if mode not in ("deterministic", "production"):
        raise ValueError(f"unknown verifier mode: {mode}")
    blocks = [n for n in (notes or []) if n]
    evidence = verify.build_evidence(hits or [], blocks)
    use_llm = mode == "production"
    result = verify.verify_answer(
        answer, evidence,
        client=client if (use_llm or allow_repair) else None,
        use_llm=use_llm,
        allow_repair=bool(allow_repair),
    )
    return result, evidence


# ---------------------------------------------------------------------------
# latency / token aggregation
# ---------------------------------------------------------------------------
def percentile(values, q: float):
    """Nearest-rank percentile — no interpolation, so p95 of a real run is a
    latency that actually happened."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    k = max(0, min(len(xs) - 1, int(-(-len(xs) * q // 1)) - 1))
    return xs[k]


def usage_totals(rows: list) -> dict:
    """Latency percentiles, token totals and the estimated dollar cost of a
    run. Rows with no model call (guard short-circuits, errored cases) simply
    contribute nothing."""
    lat, prompt, completion, calls = [], 0, 0, 0
    for r in rows:
        u = r.get("usage") or {}
        if not u:
            continue
        calls += 1
        if u.get("latency_s") is not None:
            lat.append(float(u["latency_s"]))
        prompt += int(u.get("prompt_tokens") or 0)
        completion += int(u.get("completion_tokens") or 0)
    cost = (prompt * TOKEN_COST_USD["prompt"]
            + completion * TOKEN_COST_USD["completion"])
    return {"calls": calls, "latency": lat,
            "p50": percentile(lat, 0.50), "p95": percentile(lat, 0.95),
            "max": max(lat) if lat else None, "sum": round(sum(lat), 1),
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cost_usd": round(cost, 4)}


# ---------------------------------------------------------------------------
# the pipeline under test
# ---------------------------------------------------------------------------
# Borrowed, never copied: a private duplicate of this pattern went stale the
# moment registry's was fixed for hyphenated and zero-padded ids, so the
# harness would have counted identifiers by yesterday's rules.
_FP_TOKEN_RE = getattr(registry, "_FP_RE", None) or re.compile(
    r"fp[\s\-]?0*(\d{1,3})(?!\d)", re.I)
_BOARD_CODE_RE = re.compile(
    r"b\.?\s?(\d{2})\s*[/.\-]\s*(?:(\d{2})\s*[/.\-]\s*)?add\.?\s?(\d{2})", re.I)


def fp_ids(text: str) -> set:
    """The distinct FP numbers a string names, normalized ('FP86', 'FP086'
    and 'FP-86' are one identifier; 'fp2023' is not an identifier at all)."""
    return {str(int(n)) for n in _FP_TOKEN_RE.findall(text or "")}


def multi_identifier(question: str) -> bool:
    """Stand-in for the app's `decomposed` flag, which the conductor sets.

    The conductor is deliberately skipped here (it is an LLM call and would
    make the retrieval baseline non-deterministic), but COMPARISON_BLOCK is
    gated on it in the app.  A question naming two or more identifiers is the
    deterministic proxy: it is exactly the shape the conductor fans out.
    """
    return len(fp_ids(question)) + len(set(_BOARD_CODE_RE.findall(question))) > 1


class Pipeline:
    """Offline app subset with explicit, per-record parity metadata."""

    def __init__(self, top_k: int = None, comparison_proxy: bool = True,
                 raw_retrieval: bool = False, scope_single_id: bool = False,
                 production_planner: bool = False):
        from gcf_qna.app import chainlit_app as app
        self.app = app
        self.top_k = top_k or config.TOP_K
        self.comparison_proxy = comparison_proxy
        self.raw_retrieval = raw_retrieval
        self.scope_single_id = scope_single_id
        self.production_planner = production_planner
        t0 = time.perf_counter()
        self.retriever = app.get_retriever()
        if self.retriever is None:
            raise SystemExit(f"no index at {app._index_dir()} — build one first "
                             "(see scripts/build_index.py)")
        self.load_seconds = time.perf_counter() - t0
        self.meta = dict(app._retriever_meta)

    def parity(self, *, planner_applicable=False, planner_used=False,
               matrix_in_prompt=False, planner_fallback=None) -> dict:
        """Capabilities that materially affect equivalence with the app."""
        limitations = [
            "LLM conductor is not run; semantic rewrites and chat routing are unavailable",
            "non-planner single-ID retrieval does not run the app's production prescope helper",
        ]
        if planner_fallback:
            limitations.append("deterministic planner failed and raw retrieval was used; "
                               "the production conductor fallback is unavailable")
        return {
            "level": "partial",
            "retrieval_helpers": {
                "rescope_and_tag_resolution": not self.raw_retrieval,
                "production_single_id_prescope": False,
                "single_id_ab_scope_enabled": self.scope_single_id,
            },
            "deterministic_planner": {
                "enabled": self.production_planner,
                "applicable": planner_applicable,
                "used": planner_used,
                "matrix_in_prompt": matrix_in_prompt,
                "fallback": planner_fallback,
            },
            "conductor": {"available": False, "used": False},
            "answer_history_isolation": False,
            "limitations": limitations,
        }

    # -- the registry FP-miss guard, verbatim from the app ------------------
    def fp_guard(self, question: str):
        try:
            if registry.load():
                resolved, missing = registry.resolve_fps(question)
                if missing and not resolved:
                    lang = self.app._detect_lang(question)
                    miss = ", ".join(f"FP{n}" for n in missing)
                    return (f"{miss} n'existe pas dans le corpus (registre de 273 documents)."
                            if lang == "French" else
                            f"{miss} does not exist in this corpus (273-document registry).")
        except Exception:
            pass
        return None

    # -- the query plan, through the app's own tag machinery ----------------
    def plan(self, question: str) -> list:
        """The app's `search_queries`, minus the conductor.

        chainlit_app builds `[{"q": message, "doc": None}]`, lets the
        conductor optionally replace it (and only then runs _rescope_items),
        and finally runs _resolve_doc_tags over whatever survives — that last
        call is on the unconditional path, so a scoped baseline has to go
        through it too. Both guards run here so a regression in either shows
        up in the recorded plan.
        """
        items = [{"q": question, "doc": None}]
        if self.scope_single_id:
            # NOT production behavior: the conductor emits no doc tag for a
            # cold single-topic question (CONDUCTOR_PROMPT rules 1 and 4).
            # Opt-in only, to A/B what doc-scoping would buy.
            ids = fp_ids(question)
            if len(ids) == 1:
                items[0]["doc"] = f"fp{next(iter(ids))}"
        if self.raw_retrieval:
            return items
        items = self.app._rescope_items(items, question, [])
        return self.app._resolve_doc_tags(items)

    def run(self, question: str) -> dict:
        app = self.app
        guard = self.fp_guard(question)
        if guard is not None:
            # the guard answers FROM the registry, so the registry lookup is
            # the evidence this turn held — recording it lets the claim-support
            # scorer audit a guard answer against what produced it
            try:
                reg = registry.registry_note(question)
            except Exception:
                reg = None
            return {"guard": True, "guard_answer": guard, "hits": [],
                    "system": None, "user": None, "weak": False, "plan": [],
                    "notes": {"registry": reg, "year": None, "board": None,
                              "matrix": None},
                    "pipeline_parity": self.parity()}

        plan_obj = None
        planner_applicable = False
        planner_used = False
        planner_fallback = None
        matrix_block = None
        if self.production_planner:
            candidate = planner.detect(question)
            planner_applicable = bool(
                candidate is not None and app._planner_intent(question, candidate))
            if planner_applicable:
                try:
                    matrix = planner.build_matrix(candidate, self.retriever)
                    if not any(c.status not in ("missing", "missing-document")
                               for c in matrix.cells):
                        raise ValueError("no cell carries evidence")
                    matrix_block = planner.render(matrix)
                    plan_obj = candidate
                    planner_used = True
                except Exception as e:
                    planner_fallback = f"{type(e).__name__}: {e}"

        if planner_used:
            items = [{"q": app._plan_query(plan_obj, d), "doc": d.scope}
                     for d in plan_obj.docs if not d.missing]
        else:
            items = self.plan(question)

        decomposed = len(items) > 1 or planner_used
        per_query = self.top_k if not decomposed else max(3, self.top_k // len(items))
        weak = True
        per_lists = []
        for sq in items:
            got, conf = self.retriever.search_with_confidence(
                sq["q"], per_query, sq.get("doc"))
            if conf >= config.MIN_DENSE_SCORE:
                weak = False
            per_lists.append(got)
        seen, hits = set(), []
        for tier in zip_longest(*per_lists):
            for h in tier:
                if h is None:
                    continue
                key = (h.doc_id, _page(h), (h.text or "")[:120])
                if key not in seen:
                    seen.add(key)
                    hits.append(h)
        hits = hits[:15]
        hits, year_note = app._year_assist(question, hits)
        board_note = app._board_range_note(question)
        if board_note:
            year_note = f"{year_note} {board_note}" if year_note else board_note

        context = "\n\n".join(
            f"[{app._doc_label(h.doc_id, h.page)}] (score {h.score:.2f})\n{h.text}"
            for h in hits)
        if year_note:
            context = year_note + "\n\n" + context
        if weak:
            context = ("Note: retrieval confidence for this question is LOW — the "
                       "excerpts below may not actually be relevant. Do not force an "
                       "answer from marginal matches; say plainly that the corpus "
                       "does not appear to cover this.\n\n") + context
        reg_note = None
        try:
            reg_note = registry.registry_note(question)
            if reg_note:
                context = reg_note + "\n\n" + context
        except Exception:
            pass

        if matrix_block:
            context = matrix_block + "\n\n" + context

        system = assemble(year=bool(year_note), registry=bool(reg_note),
                          comparison=(planner_used or
                                      (self.comparison_proxy and multi_identifier(question))),
                          matrix=bool(matrix_block),
                          lang=app._detect_lang(question))
        return {
            "guard": False, "guard_answer": None, "hits": hits,
            "weak": weak, "plan": items,
            "system": system,
            "user": f"Context excerpts:\n{context}\n\nQuestion: {question}",
            "notes": {"registry": reg_note, "year": year_note,
                      "board": board_note, "matrix": matrix_block},
            "pipeline_parity": self.parity(
                planner_applicable=planner_applicable,
                planner_used=planner_used,
                matrix_in_prompt=bool(matrix_block),
                planner_fallback=planner_fallback),
        }


def ask_model(client, system: str, turns: list, user: str):
    """(answer text, call metadata).

    The metadata is the release report's raw material: wall-clock latency of
    the call that succeeded, and the API's OWN token counts — estimating
    tokens from the prompt string would be a second, wronger measurement of
    something the response already reports.
    """
    messages = [{"role": "system", "content": system}]
    messages += [{"role": t["role"], "content": t["content"]} for t in turns]
    messages.append({"role": "user", "content": user})
    last = None
    for attempt in range(3):
        t0 = time.perf_counter()
        try:
            resp = client.chat.completions.create(
                model=config.CHAT_MODEL,
                max_completion_tokens=config.MAX_ANSWER_TOKENS,
                messages=messages,
            )
            dt = time.perf_counter() - t0
            u = getattr(resp, "usage", None)
            pt = int(getattr(u, "prompt_tokens", 0) or 0)
            ct = int(getattr(u, "completion_tokens", 0) or 0)
            meta = {"latency_s": round(dt, 3), "attempts": attempt + 1,
                    "model": config.CHAT_MODEL, "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": int(getattr(u, "total_tokens", 0) or 0) or (pt + ct)}
            return (resp.choices[0].message.content or ""), meta
        except Exception as e:                       # transient 429/5xx
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat call failed after 3 attempts: {last}")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _pct(num, den):
    return "  n/a" if not den else f"{num / den:>4.0%}"


def print_retrieval_table(rows: list, skipped: list):
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r)
    hdr = (f"{'class':12} {'n':>3} {'scored':>6} {'r@5':>6} {'r@10':>6} "
           f"{'cover@10':>9} {'page-hit':>9} {'guard':>6}")
    print(hdr)
    print("-" * len(hdr))
    tot = defaultdict(float)
    for cls in sorted(by_cls, key=lambda c: (_class_rank(c), c)):
        rs = by_cls[cls]
        line = _fmt_retrieval_line(cls, rs)
        print(line)
        for r in rs:
            tot["n"] += 1
            if r["retrieval"]["docs_expected"]:
                tot["scored"] += 1
                tot["r5"] += bool(r["retrieval"]["r5"])
                tot["r10"] += bool(r["retrieval"]["r10"])
                tot["cover10"] += bool(r["retrieval"]["cover10"])
            if r["retrieval"]["pages_expected"]:
                tot["pn"] += 1
                tot["prate"] += r["retrieval"]["page_rate"]
            tot["guard"] += bool(r["guard"])
    print("-" * len(hdr))
    print(f"{'TOTAL':12} {int(tot['n']):>3} {int(tot['scored']):>6} "
          f"{_pct(tot['r5'], tot['scored']):>6} {_pct(tot['r10'], tot['scored']):>6} "
          f"{_pct(tot['cover10'], tot['scored']):>9} "
          f"{_pct(tot['prate'], tot['pn']):>9} {int(tot['guard']):>6}")
    if skipped:
        print(f"\nskipped (multi-turn, need history): {len(skipped)} — "
              + ", ".join(s["id"] for s in skipped))


def _fmt_retrieval_line(cls, rs):
    scored = [r for r in rs if r["retrieval"]["docs_expected"]]
    paged = [r for r in rs if r["retrieval"]["pages_expected"]]
    return (f"{cls:12} {len(rs):>3} {len(scored):>6} "
            f"{_pct(sum(bool(r['retrieval']['r5']) for r in scored), len(scored)):>6} "
            f"{_pct(sum(bool(r['retrieval']['r10']) for r in scored), len(scored)):>6} "
            f"{_pct(sum(bool(r['retrieval']['cover10']) for r in scored), len(scored)):>9} "
            f"{_pct(sum(r['retrieval']['page_rate'] for r in paged), len(paged)):>9} "
            f"{sum(bool(r['guard']) for r in rs):>6}")


def print_answer_table(rows: list):
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r)
    hdr = (f"{'class':12} {'n':>3} {'behavior':>9} {'contains':>9} {'forbid':>7} "
           f"{'lang':>6} {'cites':>6} {'score':>6} {'pass':>5}")
    print(hdr)
    print("-" * len(hdr))
    tot = defaultdict(float)
    for cls in sorted(by_cls, key=lambda c: (_class_rank(c), c)):
        rs = by_cls[cls]
        c = _agg_answer(rs)
        print(f"{cls:12} {len(rs):>3} {_pct(c['behavior'], len(rs)):>9} "
              f"{_pct(c['ct'], c['cn']):>9} {_pct(c['ft'], c['fn']):>7} "
              f"{_pct(c['language'], len(rs)):>6} {_pct(c['citations'], len(rs)):>6} "
              f"{c['score'] / len(rs):>6.2f} {_pct(c['pass'], len(rs)):>5}")
        for k, v in c.items():
            tot[k] += v
        tot["n"] += len(rs)
    print("-" * len(hdr))
    print(f"{'TOTAL':12} {int(tot['n']):>3} {_pct(tot['behavior'], tot['n']):>9} "
          f"{_pct(tot['ct'], tot['cn']):>9} {_pct(tot['ft'], tot['fn']):>7} "
          f"{_pct(tot['language'], tot['n']):>6} {_pct(tot['citations'], tot['n']):>6} "
          f"{tot['score'] / max(1, tot['n']):>6.2f} {_pct(tot['pass'], tot['n']):>5}")


def _metrics(rows: list, key: str, need: str) -> list:
    """The rows' <key> metric blocks, skipping absent ones and the ``{'error':
    ...}`` stub a failed scorer leaves behind."""
    out = []
    for r in rows:
        m = r.get(key)
        if isinstance(m, dict) and need in m:
            out.append(m)
    return out


def print_release_table(rows: list, cases_total: int = None):
    """The release report: one line per class, then the three aggregates the
    plan gates on (field coverage, claim support, cost/latency)."""
    scored = [r for r in rows if not r.get("error") and r.get("checks")]
    errored = [r for r in rows if r.get("error")]
    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["class"]].append(r)

    print("\n" + "=" * 78)
    print(f"RELEASE REPORT — {len(rows)} cases"
          + (f" of {cases_total}" if cases_total and cases_total != len(rows) else "")
          + f", model={config.CHAT_MODEL}, {len(errored)} errored")
    print("=" * 78)
    hdr = (f"{'class':12} {'n':>3} {'err':>4} {'pass':>6} {'behav':>6} "
           f"{'contain':>8} {'forbid':>7} {'lang':>6} {'cites':>6} "
           f"{'field':>6} {'cite-cmpl':>9} {'p50':>7} {'p95':>7}")
    print(hdr)
    print("-" * len(hdr))
    for cls in sorted(by_cls, key=lambda c: (_class_rank(c), c)):
        print(_release_line(cls, by_cls[cls]))
    print("-" * len(hdr))
    print(_release_line("TOTAL", rows))

    fields = _metrics(scored, "fields", "n_cells")
    claims = _metrics(scored, "claims", "claims")
    u = usage_totals(rows)

    print("\nFIELD COVERAGE — (document x required field) cells vs registry v2")
    cells = sum(f["n_cells"] for f in fields)
    scorable = sum(f["n_scorable"] for f in fields)
    stated = sum(f["n_stated"] for f in fields)
    marked = sum(f["n_marked_missing"] for f in fields)
    missed = sum(f["n_missed"] for f in fields)
    unscorable = sum(f["n_unscorable"] for f in fields)
    print(f"  cases with a field contract : {len(fields)}")
    print(f"  cells                       : {cells} "
          f"({scorable} scorable, {unscorable} unscorable)")
    print(f"  covered                     : {stated + marked}/{scorable} "
          f"{_pct(stated + marked, scorable).strip()}   "
          f"[stated {stated}, explicitly marked missing {marked}]")
    print(f"  missed                      : {missed}")
    _print_field_breakdown(fields)

    print("\nCLAIM SUPPORT / QUALITY — factual content and citations scored separately "
          "(deterministic unless verifier mode says otherwise)")
    tot = sum(c["claims"] for c in claims)
    sup = sum(c["supported"] for c in claims)
    # No .get(key, c["supported"]) backfill: substituting the verdict tally for
    # a missing metric prints a fabricated number under the metric's own name
    # (verdict finding 21 — it asserted 91 == 110 == 141). A record that lacks
    # the key reports n/a instead.
    grounded = _sum_claim_metric(claims, "grounded")
    complete = _sum_claim_metric(claims, "citation_complete")
    present = _sum_claim_metric(claims, "citation_present")
    con = sum(c["contradicted"] for c in claims)
    uns = sum(c["unsupported"] for c in claims)
    rate = (complete / tot) if (tot and complete is not None) else None
    need = gate_threshold(tot)
    print(f"  claims                      : {tot} over {len(claims)} answers")
    print(f"  groundedness                : {_metric_line(grounded, tot)}")
    print(f"  citation completeness       : {_metric_line(complete, tot)}  "
          f"gate >= {CLAIM_SUPPORT_GATE:.0%} = {need}/{tot} — "
          f"{'PASS' if tot and complete is not None and complete >= need else 'FAIL'}")
    print(f"  citation presence           : {_metric_line(present, tot)}  "
          f"(reported, never gated)")
    print(f"  legacy supported verdicts   : {sup}")
    print(f"  contradicted                : {con}")
    print(f"  unsupported                 : {uns}")
    _print_claim_breakdown(scored)

    print("\nLATENCY / COST")
    print(f"  model calls                 : {u['calls']}")
    print(f"  latency p50 / p95           : "
          + (f"{u['p50']:.1f}s / {u['p95']:.1f}s" if u["p50"] is not None else "n/a")
          + (f"   (max {u['max']:.1f}s, {u['sum']:.0f}s of model wall-clock)"
             if u["max"] is not None else ""))
    print(f"  tokens                      : prompt {u['prompt_tokens']:,} + "
          f"completion {u['completion_tokens']:,} = {u['total_tokens']:,}")
    print(f"  estimated cost              : ${u['cost_usd']:.2f}  "
          f"(ESTIMATED rates: ${TOKEN_COST_USD['prompt'] * 1e6:.2f}/1M prompt, "
          f"${TOKEN_COST_USD['completion'] * 1e6:.2f}/1M completion)")

    print("\nERRORED CASES")
    if not errored:
        print("  none")
    else:
        for r in errored:
            print(f"  {r['id']:26} {r['error'][:110]}")
    return {"fields": {"cells": cells, "scorable": scorable, "stated": stated,
                       "marked_missing": marked, "missed": missed,
                       "unscorable": unscorable},
            "claims": {"total": tot, "grounded": grounded,
                       "citation_complete": complete,
                       "citation_present": present,
                       "groundedness": _n_over_d_or_none(grounded, tot),
                       "citation_completeness": _n_over_d_or_none(complete, tot),
                       "citation_presence": _n_over_d_or_none(present, tot),
                       "gate_threshold": need,
                       "supported": sup, "contradicted": con,
                       "unsupported": uns, "rate": rate},
            "usage": u, "errors": [r["id"] for r in errored]}


def _sum_claim_metric(claims: list, key: str):
    """Pooled numerator for one claim metric, or None if any record predates
    it. None propagates to `n/a`; it is never silently replaced by another
    metric's count."""
    vals = [c.get(key) for c in claims]
    return None if any(v is None for v in vals) else sum(vals)


def _n_over_d_or_none(num, den):
    return None if num is None else n_over_d(num, den)


def _metric_line(num, den) -> str:
    """`n/d` first, the percentage second and only as a reading aid — the
    contract forbids publishing a claim metric as a bare rate."""
    if num is None:
        return f"n/a ({den} claims; this run predates the metric)"
    return f"{n_over_d(num, den)}" + (f" {num / den:.1%}" if den else "")


def _release_line(label: str, rs: list) -> str:
    scored = [r for r in rs if not r.get("error") and r.get("checks")]
    err = sum(1 for r in rs if r.get("error"))
    c = _agg_answer(scored)
    f = _metrics(scored, "fields", "n_cells")
    cl = _metrics(scored, "claims", "claims")
    n = max(1, len(scored))
    u = usage_totals(rs)
    fcov = _pct(sum(x["n_covered"] for x in f), sum(x["n_scorable"] for x in f))
    # The gated claim column is citation completeness, published as n/d.
    cnum = _sum_claim_metric(cl, "citation_complete")
    cden = sum(x["claims"] for x in cl)
    ccov = "n/a" if cnum is None else n_over_d(cnum, cden)
    return (f"{label:12} {len(rs):>3} {err:>4} {_pct(c['pass'], n):>6} "
            f"{_pct(c['behavior'], n):>6} {_pct(c['ct'], c['cn']):>8} "
            f"{_pct(c['ft'], c['fn']):>7} {_pct(c['language'], n):>6} "
            f"{_pct(c['citations'], n):>6} {fcov:>6} {ccov:>9} "
            + (f"{u['p50']:>6.1f}s" if u["p50"] is not None else f"{'n/a':>7}")
            + (f" {u['p95']:>6.1f}s" if u["p95"] is not None else f" {'n/a':>7}"))


def _print_field_breakdown(fields: list):
    per = defaultdict(lambda: [0, 0, 0])           # field -> covered, scorable, unscorable
    misses = []
    for f in fields:
        for cell in f["cells"]:
            row = per[cell["field"]]
            if cell["status"] == "unscorable":
                row[2] += 1
                continue
            row[1] += 1
            row[0] += cell["status"] in ("stated", "marked-missing")
            if cell["status"] == "missed":
                misses.append(cell)
    for name, (cov, tot, uns) in sorted(per.items()):
        print(f"    {name:20} {cov:>3}/{tot:<3} {_pct(cov, tot).strip():>4}"
              + (f"   ({uns} unscorable)" if uns else ""))
    for cell in misses[:8]:
        print(f"    MISS {cell['field']:18} {cell['doc'][:44]:44} "
              f"expected {'; '.join(cell.get('expected', []))[:60]}")


def _print_claim_breakdown(rows: list):
    """Per class, and the five worst cases, on the SAME metric the header
    gates: citation completeness. Reporting the legacy verdict tally under a
    citation-completeness header is how the two got conflated."""
    per = defaultdict(lambda: [0, 0, False])   # complete, claims, any-missing
    have = [r for r in rows if _metrics([r], "claims", "claims")]
    for r in have:
        c = r["claims"]
        row = per[r["class"]]
        if c.get("citation_complete") is None:
            row[2] = True
        else:
            row[0] += c["citation_complete"]
        row[1] += c.get("claims", 0)
    for cls in sorted(per, key=lambda c: (_class_rank(c), c)):
        cc, tot, missing = per[cls]
        if missing:                            # never print 0/n for 'unknown'
            print(f"    {cls:20} {'n/a':>3}/{tot:<3}")
            continue
        print(f"    {cls:20} {cc:>3}/{tot:<3} {_pct(cc, tot).strip():>4}")

    # Records that predate the metric have no completeness to be worst at, so
    # they are excluded rather than sorted to either end.
    scored = [r for r in have if r["claims"].get("claims")
              and r["claims"].get("citation_completeness_rate") is not None]
    worst = sorted(scored, key=lambda r: r["claims"]["citation_completeness_rate"])
    for r in worst[:5]:
        c = r["claims"]
        if c["citation_completeness_rate"] >= 1.0:
            break
        first = (c["failures"] or [{}])[0]
        print(f"    LOW  {r['id']:26} "
              f"{n_over_d(c['citation_complete'], c['claims'])} "
              f"{first.get('status', '')}: {first.get('reason', '')[:60]}")


def _agg_answer(rs):
    c = defaultdict(float)
    for r in rs:
        ch = r["checks"]
        c["behavior"] += bool(ch["behavior"])
        c["language"] += bool(ch["language"])
        c["citations"] += bool(ch["citations"])
        c["score"] += ch["score"]
        c["pass"] += bool(ch["pass"])
        c["cn"] += len(ch["must_contain"])
        c["ct"] += sum(bool(v) for v in ch["must_contain"].values())
        c["fn"] += len(ch["must_not_contain"])
        c["ft"] += sum(bool(v) for v in ch["must_not_contain"].values())
    return c


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------
def run_gate(pipe: Pipeline, gold_path: Path):
    """The retrieval floor from the plan: the 30 gold questions must stay at
    100% document recall. Same metric as scripts/eval_retrieval.py, run in
    this process so the index is loaded once."""
    gold = [json.loads(l) for l in gold_path.read_text(encoding="utf-8").splitlines() if l]
    stats = defaultdict(lambda: {"n": 0, "r5": 0, "r10": 0, "mrr": 0.0})
    for g in gold:
        hits = pipe.retriever.search(g["q"], top_k=10)
        rank = next((i + 1 for i, h in enumerate(hits)
                     if any(doc_eq(h.doc_id, d) for d in g["expected"])), None)
        s = stats[g["cls"]]
        s["n"] += 1
        s["r5"] += bool(rank and rank <= 5)
        s["r10"] += bool(rank)
        s["mrr"] += (1.0 / rank) if rank else 0.0
    print(f"\n=== retrieval floor gate — {gold_path.name}, {len(gold)} questions ===")
    print(f"{'class':12} {'n':>3} {'recall@5':>9} {'recall@10':>10} {'MRR':>6}")
    tot = {"n": 0, "r5": 0, "r10": 0, "mrr": 0.0}
    for cls, s in sorted(stats.items()):
        print(f"{cls:12} {s['n']:>3} {s['r5'] / s['n']:>9.0%} "
              f"{s['r10'] / s['n']:>10.0%} {s['mrr'] / s['n']:>6.2f}")
        for k in tot:
            tot[k] += s[k]
    print(f"{'TOTAL':12} {tot['n']:>3} {tot['r5'] / tot['n']:>9.0%} "
          f"{tot['r10'] / tot['n']:>10.0%} {tot['mrr'] / tot['n']:>6.2f}")
    ok = tot["r5"] == tot["n"]
    print(f"GATE: {'PASS' if ok else 'FAIL'} — document recall@5 "
          f"{tot['r5'] / tot['n']:.0%} (floor: 100%)")
    return ok


def run_eval(args, cases: list) -> list:
    pipe = Pipeline(top_k=args.k, comparison_proxy=not args.no_comparison_proxy,
                    raw_retrieval=args.raw_retrieval,
                    scope_single_id=args.scope_single_id,
                    production_planner=getattr(args, "production_planner", False))
    print(f"retriever ready in {pipe.load_seconds:.1f}s — "
          f"{pipe.meta.get('n_chunks')} chunks, {pipe.meta.get('embedding_model')}")
    if args.gate:
        run_gate(pipe, GOLD_SET)

    client = None
    if args.answers:
        import openai
        client = openai.OpenAI(base_url=config.OPENAI_BASE_URL or None)

    rows, skipped = [], []
    t0 = time.perf_counter()
    for i, case in enumerate(cases, 1):
        if case["turns"] and not args.answers:
            skipped.append(case)
            continue
        try:
            rec = _run_case(pipe, client, args, case)
        except Exception as e:                  # one bad case never ends a run
            parity = pipe.parity()
            parity["followup"] = {
                "fixture_has_history": bool(case["turns"]),
                "conductor_history_resolution": (
                    "unavailable" if case["turns"] else "not-needed"),
            }
            rec = {"id": case["id"], "class": case["class"], "lang": case["lang"],
                   "question": case["question"], "turns": len(case["turns"]),
                   "mode": "answers" if args.answers else "retrieval-only",
                   "guard": False, "expect": case["expect"], "score": 0.0,
                   "pipeline_parity": parity,
                   "error": f"{type(e).__name__}: {e}"}
            print(f"[{i}/{len(cases)}] {case['id']}  ERROR {rec['error'][:90]}",
                  flush=True)
            rows.append(rec)
            continue
        rows.append(rec)
        if args.verbose:
            r = rec["retrieval"]
            print(f"[{i}/{len(cases)}] {case['id']:26} rank={r['rank']} "
                  f"pages={r['pages_hit']}/{r['pages_expected']} "
                  + (f"score={rec['score']:.2f}" if args.answers else ""))
        elif args.answers:
            u = rec.get("usage") or {}
            print(f"[{i}/{len(cases)}] {case['id']:26} score={rec['score']:.2f}"
                  + (f" {u['latency_s']:.1f}s {u['total_tokens']}tok" if u else ""),
                  flush=True)

    dt = time.perf_counter() - t0
    print(f"\n=== {'answer' if args.answers else 'retrieval-only'} baseline — "
          f"{len(rows)} cases in {dt:.0f}s "
          f"(k={pipe.top_k}, hybrid={getattr(pipe.retriever, 'hybrid_enabled', False)}"
          + (f", model={config.CHAT_MODEL}" if args.answers else "") + ") ===")
    ok = [r for r in rows if not r.get("error")]
    print_retrieval_table(ok, skipped)
    if args.answers:
        print()
        print_answer_table([r for r in ok if r.get("checks")])
        _print_failures([r for r in ok if r.get("checks")])
    if getattr(args, "release", False):
        print_release_table(rows, cases_total=len(cases))
    return rows


def _safe(fn, *a, **kw):
    """Run a scorer; a scorer that raises must not throw away an answer that
    cost a model call. The failure is recorded in place of its metrics."""
    try:
        return fn(*a, **kw)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _metric_rate(metric, key):
    return metric.get(key) if isinstance(metric, dict) else None


def compare_verifier_output(original: dict, returned: dict) -> dict:
    """Machine-readable no-regression checks for a verifier/repair pass.

    One row per entry in COMPARED_METRICS, whose keys are asserted pairwise
    distinct at import: two rows reading the same record key would count one
    regression twice and print an extra green delta. Citation presence is
    reported as a delta but never triggers a regression — it is the metric a
    fabricated bracket can move.
    """
    _assert_distinct_metric_keys()
    checks_a, checks_b = original["checks"], returned["checks"]
    blocks = {"checks": (checks_a, checks_b),
              "fields": (original["fields"], returned["fields"]),
              "claims": (original["claims"], returned["claims"])}
    regressions = []
    if checks_a.get("pass") and not checks_b.get("pass"):
        regressions.append("answer_checks_pass_to_fail")
    deltas = {"answer_score": checks_b.get("score", 0) - checks_a.get("score", 0)}
    for name, block, key, gated in COMPARED_METRICS:
        a, b = blocks[block]
        before, after = _metric_rate(a, key), _metric_rate(b, key)
        deltas[name] = None if before is None or after is None else after - before
        if not gated:
            continue
        if before is not None and after is not None and after < before - 1e-9:
            regressions.append(name + "_decreased")
    return {"no_regression": not regressions, "regressions": regressions,
            "deltas": deltas}


def _answer_metrics(case, answer, hits, notes, *, verdicts=None, evidence=None):
    return {
        "checks": score_answer(case, answer, hits),
        "fields": _safe(score_fields, case, answer),
        "claims": _safe(score_claims, answer, hits, notes,
                        verdicts=verdicts, evidence=evidence),
    }


def _run_case(pipe, client, args, case: dict) -> dict:
    out = pipe.run(case["question"])
    parity = dict(out.get("pipeline_parity") or {
        "level": "unknown", "limitations": ["pipeline did not report capabilities"]})
    parity["followup"] = {
        "fixture_has_history": bool(case["turns"]),
        "conductor_history_resolution": "unavailable" if case["turns"] else "not-needed",
    }
    if case["turns"]:
        parity.setdefault("limitations", []).append(
            "follow-up history is passed to answer generation but is not resolved by the conductor")
    rec = {
        "id": case["id"], "class": case["class"], "lang": case["lang"],
        "question": case["question"], "turns": len(case["turns"]),
        "mode": "answers" if args.answers else "retrieval-only",
        "guard": out["guard"], "weak_signal": out.get("weak"),
        "plan": out.get("plan") or [],
        "pipeline_parity": parity,
        "retrieval": score_retrieval(case, out["hits"]),
        "expect": case["expect"],
    }
    rec["retrieval_score"] = retrieval_score(rec["retrieval"])
    if not args.answers:
        rec["score"] = rec["retrieval_score"]
        return rec

    answer, usage = out["guard_answer"], {}
    if answer is None:
        answer, usage = ask_model(client, out["system"], case["turns"], out["user"])
    notes = out.get("notes") or {}
    evidence_notes = [notes.get("registry"), notes.get("year"), notes.get("matrix")]
    original = _answer_metrics(case, answer, out["hits"], evidence_notes)
    verifier_mode = getattr(args, "verifier_mode", "deterministic")
    allow_repair = bool(getattr(args, "verifier_repair", False))
    final_answer = answer
    verifier_result = None
    verifier_error = None
    evidence = None
    try:
        verifier_result, evidence = run_offline_verifier(
            answer, out["hits"], evidence_notes, client=client,
            mode=verifier_mode, allow_repair=allow_repair)
        final_answer = verifier_result.answer
    except Exception as e:
        # Production verification is best-effort and never discards an answer.
        verifier_error = f"{type(e).__name__}: {e}"

    if verifier_result is not None:
        returned = _answer_metrics(
            case, final_answer, out["hits"], evidence_notes,
            verdicts=verifier_result.verdicts, evidence=evidence)
    else:
        returned = original
    comparison = compare_verifier_output(original, returned)
    rec.update({
        "answer": final_answer,
        "original_answer": answer,
        "checks": returned["checks"],
        "checks_original": original["checks"],
        "model": config.CHAT_MODEL,
        "usage": usage,
        "fields": returned["fields"],
        "fields_original": original["fields"],
        "claims": returned["claims"],
        "claims_original": original["claims"],
        "verification": {
            "mode": verifier_mode,
            "use_llm": verifier_mode == "production",
            "repair_enabled": allow_repair,
            "status": getattr(verifier_result, "status", None),
            "answer_changed": final_answer != answer,
            "repaired": bool(getattr(verifier_result, "repaired", False)),
            "repair_rejected": bool(getattr(verifier_result, "repair_rejected", False)),
            "error": verifier_error,
            "usage_accounting": "answer-generation call only; verifier judge and repair excluded",
            "comparison": comparison,
        },
        "hits": [{"doc": h.doc_id, "page": _page(h), "score": round(h.score, 4)}
                 for h in out["hits"]],
        "notes_used": {k: v for k, v in notes.items() if v},
    })
    rec["score"] = rec["checks"]["score"]
    return rec


def _print_failures(rows: list):
    bad = [r for r in rows if not r["checks"]["pass"]]
    if not bad:
        print("\nall cases pass")
        return
    print(f"\nfailing cases ({len(bad)}/{len(rows)}):")
    for r in bad:
        why = []
        if not r["checks"]["behavior"]:
            why.append(f"behavior!={r['expect']['behavior']}")
        miss = [p for p, ok in r["checks"]["must_contain"].items() if not ok]
        if miss:
            why.append("missing " + "; ".join(miss))
        forb = [p for p, ok in r["checks"]["must_not_contain"].items() if not ok]
        if forb:
            why.append("forbidden " + "; ".join(forb))
        if not r["checks"]["language"]:
            why.append(f"lang!={r['lang']}")
        if not r["checks"]["citations"]:
            why.append("bad cites: " + "; ".join(r["checks"]["bad_citations"]))
        print(f"  {r['id']:26} {r['score']:.2f}  " + " | ".join(why))


def run_compare(path_a: Path, path_b: Path):
    def _load(p):
        return {json.loads(l)["id"]: json.loads(l)
                for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}

    a, b = _load(path_a), _load(path_b)
    shared = [i for i in a if i in b]
    only_a = sorted(set(a) - set(b))
    only_b = sorted(set(b) - set(a))
    print(f"A: {path_a}  ({len(a)} cases)")
    print(f"B: {path_b}  ({len(b)} cases)")
    print(f"compared: {len(shared)}"
          + (f" | only in A: {len(only_a)}" if only_a else "")
          + (f" | only in B: {len(only_b)}" if only_b else ""))
    by_cls = defaultdict(lambda: {"n": 0, "better": 0, "worse": 0, "same": 0,
                                  "da": 0.0, "db": 0.0})
    changed = []
    for i in shared:
        ra, rb = a[i], b[i]
        sa, sb = float(ra.get("score", 0.0)), float(rb.get("score", 0.0))
        s = by_cls[ra.get("class", "?")]
        s["n"] += 1
        s["da"] += sa
        s["db"] += sb
        if sb > sa + 1e-9:
            s["better"] += 1
            changed.append((i, sa, sb, "better"))
        elif sb < sa - 1e-9:
            s["worse"] += 1
            changed.append((i, sa, sb, "worse"))
        else:
            s["same"] += 1
    hdr = (f"{'class':12} {'n':>3} {'A':>6} {'B':>6} {'delta':>7} "
           f"{'better':>7} {'worse':>6} {'same':>5}")
    print()
    print(hdr)
    print("-" * len(hdr))
    tot = defaultdict(float)
    for cls in sorted(by_cls, key=lambda c: (_class_rank(c), c)):
        s = by_cls[cls]
        print(f"{cls:12} {s['n']:>3} {s['da'] / s['n']:>6.2f} {s['db'] / s['n']:>6.2f} "
              f"{(s['db'] - s['da']) / s['n']:>+7.2f} {s['better']:>7} "
              f"{s['worse']:>6} {s['same']:>5}")
        for k, v in s.items():
            tot[k] += v
    print("-" * len(hdr))
    n = max(1, tot["n"])
    print(f"{'TOTAL':12} {int(tot['n']):>3} {tot['da'] / n:>6.2f} {tot['db'] / n:>6.2f} "
          f"{(tot['db'] - tot['da']) / n:>+7.2f} {int(tot['better']):>7} "
          f"{int(tot['worse']):>6} {int(tot['same']):>5}")
    if changed:
        print("\nper-case changes:")
        for i, sa, sb, tag in sorted(changed, key=lambda c: (c[3], c[0])):
            print(f"  {tag:6} {i:26} {sa:.2f} -> {sb:.2f}")
    _compare_extras(a, b, shared)


def _extra_rates(rec: dict):
    """(field coverage cell counts, citation-completeness counts) of one
    recorded row, each as (n, d), or None where the row predates the metric.

    The claim number here is CITATION COMPLETENESS — cited AND supported. It
    used to read the legacy `support_rate`, which in production mode absorbs
    the judge's uncited promotions, and it read it as a bare rate. Counts are
    returned so the comparison can pool them and publish n/d; a run recorded
    before the metric existed contributes nothing rather than being back-filled
    from a different metric.
    """
    f = rec.get("fields")
    c = rec.get("claims")
    f = f if isinstance(f, dict) else {}
    c = c if isinstance(c, dict) else {}
    cov = (f.get("n_covered"), f.get("n_scorable"))
    cite = (c.get("citation_complete"), c.get("claims"))
    return (None if cov[0] is None or not cov[1] else cov,
            None if cite[0] is None or not cite[1] else cite)


def _compare_extras(a: dict, b: dict, shared: list):
    pairs = {"field coverage": [], "citation completeness": []}
    for i in shared:
        fa, ca = _extra_rates(a[i])
        fb, cb = _extra_rates(b[i])
        if fa is not None and fb is not None:
            pairs["field coverage"].append((fa, fb))
        if ca is not None and cb is not None:
            pairs["citation completeness"].append((ca, cb))
    lines = [(name, vs) for name, vs in pairs.items() if vs]
    if not lines:
        return
    print("\n(metrics present in both runs — pooled n/d, not a mean of rates)")
    for name, vs in lines:
        na, da = sum(x[0] for x, _ in vs), sum(x[1] for x, _ in vs)
        nb, db = sum(y[0] for _, y in vs), sum(y[1] for _, y in vs)
        moved = "" if da == db else f"   DENOMINATOR MOVED {da} -> {db}"
        print(f"  {name:22} {len(vs):>3} cases  {n_over_d(na, da):>9} -> "
              f"{n_over_d(nb, db):>9}  {na / da:>6.1%} -> {nb / db:>6.1%}{moved}")


def record_path(label: str, prefix: str = "answers_baseline_") -> Path:
    """Where --record LABEL would write. One place computes this, so the
    pre-flight check and the write can never disagree about the target."""
    return EVAL_DIR / f"{prefix}{label}.jsonl"


def _overwrite_refusal(out: Path) -> str:
    return (f"refusing to overwrite {out}\n"
            f"  It already holds a recorded run, and a recorded run is a "
            f"MEASUREMENT, not a build artifact: re-running --release calls the "
            f"model again and produces a different sample, so the file it "
            f"replaces cannot be reconstructed.\n"
            f"  Use a new --record LABEL, or pass --force-record to overwrite "
            f"it deliberately.")


def record(rows: list, label: str, prefix: str = "answers_baseline_",
           force: bool = False) -> Path:
    """Write a run to data/eval/<prefix><LABEL>.jsonl.

    Refuses an existing path unless ``force``. data/eval/release_release-1.jsonl
    is the only copy of the 66-answer release run every later --compare is
    measured against; the unconditional write this replaced meant one mistyped
    --record label destroyed it in place, silently and unrecoverably. A fresh
    label is unaffected, so nothing about the normal path changes.
    """
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = record_path(label, prefix)
    if out.exists() and not force:
        raise SystemExit(_overwrite_refusal(out))
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ap.add_argument("--retrieval-only", action="store_true",
                    help="default mode: no API calls")
    ap.add_argument("--answers", action="store_true",
                    help="also generate and score answers (costs API calls)")
    ap.add_argument("--release", action="store_true",
                    help="full-suite release run: --answers over every case "
                         "plus the release report (field coverage, claim "
                         "support, latency, tokens, estimated cost); "
                         "--record writes data/eval/release_<LABEL>.jsonl")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), type=Path,
                    help="diff two recorded runs")
    ap.add_argument("--gate", action="store_true",
                    help="also re-check the gold_set retrieval floor in-process")
    ap.add_argument("--sample", type=int, help="evaluate N class-stratified cases")
    ap.add_argument("--ids", help="comma-separated case ids")
    ap.add_argument("--classes", help="comma-separated class filter")
    ap.add_argument("--record", metavar="LABEL",
                    help="write data/eval/answers_baseline_<LABEL>.jsonl")
    ap.add_argument("--force-record", action="store_true",
                    help="allow --record to overwrite an existing recording. "
                         "OFF by default: a recorded run is the only copy of "
                         "that measurement (a re-run is a different sample), "
                         "so overwriting one has to be deliberate.")
    ap.add_argument("--k", type=int, default=config.TOP_K, help="hits per query")
    ap.add_argument("--no-comparison-proxy", action="store_true",
                    help="never ship COMPARISON_BLOCK (the app gates it on the "
                         "conductor's fan-out, which this harness skips)")
    ap.add_argument("--raw-retrieval", action="store_true",
                    help="bypass the app's _rescope_items/_resolve_doc_tags "
                         "guards and search the raw question")
    ap.add_argument("--scope-single-id", action="store_true",
                    help="A/B only, NOT production: doc-scope questions naming "
                         "exactly one FP, to measure what tag resolution buys")
    ap.add_argument("--production-planner", action="store_true",
                    help="run the app's deterministic comparison planner, evidence "
                         "matrix, scoped-query and round-robin retrieval path; the "
                         "LLM conductor fallback remains unavailable")
    ap.add_argument("--verifier-mode", choices=("deterministic", "production"),
                    default="deterministic",
                    help="offline verify.verify_answer mode: deterministic adds no "
                         "API calls (default); production enables LLM adjudication")
    ap.add_argument("--verifier-repair", action="store_true",
                    help="opt in to the verifier repair pass; this can add an API "
                         "call independently of --verifier-mode")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    if args.raw_retrieval and args.production_planner:
        ap.error("--raw-retrieval and --production-planner are mutually exclusive")

    if args.compare:
        run_compare(*args.compare)
        return 0

    if args.release:
        args.answers = True          # a release run IS an answer run, whole suite

    prefix = "release_" if args.release else "answers_baseline_"
    if args.record:
        # Pre-flight, before a single model call: record() would refuse anyway,
        # but only AFTER 66 answers had been generated and paid for — the run
        # would be lost to the very check meant to protect runs.
        out = record_path(args.record, prefix)
        if out.exists() and not args.force_record:
            raise SystemExit(_overwrite_refusal(out))

    cases = select(load_cases(args.cases), ids=args.ids, sample=args.sample,
                   classes=args.classes)
    if not cases:
        raise SystemExit("no cases selected")
    rows = run_eval(args, cases)
    if args.record:
        print(f"\nrecorded -> "
              f"{record(rows, args.record, prefix=prefix, force=args.force_record)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
