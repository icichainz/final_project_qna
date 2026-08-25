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
                     retrieval path the app runs — conductor SKIPPED, the
                     question used verbatim — and scores document recall and
                     evidence-page hit rate.  Multi-turn cases are skipped
                     (they need history to be resolvable) and counted apart.
  --answers          additionally calls the chat model per case, assembling
                     the prompt exactly as chainlit_app does (registry note,
                     year note, board-range note, weak-signal note, per-turn
                     assemble()), and scores the answer.
  --release          --answers over the WHOLE suite plus the release report:
                     required-field coverage against registry v2, claim
                     support from rag.verify (the plan's >=95% citation-
                     precision gate), latency p50/p95, tokens and estimated
                     cost.  Records to data/eval/release_<label>.jsonl.
                     PRODUCTION PARITY IS THE DEFAULT HERE (see below): the
                     planner and the conductor are on unless turned off.
  --compare A B      diff two recorded runs. --require-metrics turns it into a
                     gate: a metric missing or None on either side is a hard
                     failure, and a regression exits 1.
  --rescore-record   recompute a recorded run's claims metrics offline, zero
                     API calls (--evidence supplies the passage text for a
                     record whose hits predate F7).

Production parity (Wave 3)
--------------------------
A release number is only a release number if it came off the path production
runs.  The harness therefore reproduces `chainlit_app.main` turn for turn:
`planner.detect` behind the app's own intent gate and its evidence matrix
(--production-planner), the per-turn LLM conductor with the app's prompt and
rewrite guards (--conductor), the FP-miss guard AFTER the conductor and
returning BEFORE verification, the app's `decomposed` flag rather than a
question-shape proxy, `_answer_messages` history isolation with the resolved-
references note, `verify.verify_answer(use_llm=1)`
(--verifier-mode production), and every model call — conductor, answer,
judge — booked into the turn's usage.

Every one of those is an explicit CLI switch.  None is read from the ambient
environment: `.env` is loaded for the API key alone, and a harness that took
its configuration from the file that ships to production would silently change
what it measures whenever an operator flipped a switch.

`--verifier-mode deterministic` (the default) stays the cheap, zero-API,
per-commit instrument; production mode runs at wave boundaries.

Sampling is pinned (temperature 0 and a fixed seed on every call) and each
record stores the model SNAPSHOT the endpoint served, the `verify.py` blob
sha, the index and registry hashes, and a `pipeline_parity` block whose
`level` reads "full" only when every gap is closed AND the deployed
configuration matches field for field.

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

def _sha256_file(path: Path, missing: str = None):
    """sha256 of a file, or ``missing`` when it is not there."""
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return missing


DEFAULT_CASES = ROOT / "scripts" / "answer_gold.jsonl"
GOLD_SET = ROOT / "scripts" / "gold_set.jsonl"
EVAL_DIR = ROOT / "data" / "eval"

# How the harness decides whether to ship COMPARISON_BLOCK.
#   decomposed  production: the plan fanned out (or the planner built a matrix)
#   proxy       the retired stand-in: >= 2 identifiers in the question
#   off         never ship it
COMPARISON_FLAGS = {"decomposed", "proxy", "off"}

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

# The plan's citation-precision gate: share of extracted claims the turn's own
# evidence supports.
CLAIM_SUPPORT_GATE = 0.95


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
    r"no (?:[\w-]+ ){0,2}(?:information|record|proposals?|funding proposals?|"
    r"data|evidence|mention|board meetings?)|"
    r"outside (?:the|this) corpus|not covered|only covers|"
    r"n'existe pas|introuvable|ne figure pas|pas dans (?:le|ce) corpus|"
    r"ne contient pas|hors (?:du|de ce) corpus|je ne (?:peux|trouve)|pas trouv|"
    r"aucune? (?:proposition|document|information|mention|r[ée]sultat|donnée|"
    r"correspondance|élément|extrait|trace)", re.I)

_CONFLICT_RE = re.compile(
    r"conflict|discrepan|inconsisten|contradict|mismatch|diverg|"
    r"differ(?:s|ent)?\b|two different|whereas|while (?:page|p\.)|does not match|"
    r"disagree|contradictoire|incohéren|diffère|différent|"
    r"(?:deux|trois|plusieurs) (?:montants|chiffres|valeurs|figures)",
    re.I)

# A conflict claim has a shape, not just a keyword: two different values in one
# sentence, or two different page numbers cited close together. Without this,
# "FP151 and FP152 have different accredited entities" — an ordinary
# comparison — counted as conflict-surfacing and made the conflict cases free.
_VALUE_RE = re.compile(r"\d{1,3}(?:[.,\s]\d{3})+|\d+[.,]\d+|\d{4,}")
_PAGE_RE = re.compile(r"\bpp?\.?\s*(\d{1,3})\b", re.I)
_PAGE_WINDOW = 80
_SAME_DOC_LINES = 4


def _two_values_in_a_sentence(answer: str) -> bool:
    for sent in re.split(r"(?<=[.!?;:])\s+|\n", answer or ""):
        vals = {re.sub(r"\s", "", v) for v in _VALUE_RE.findall(sent)}
        if len(vals) >= 2:
            return True
    return False


def _same_doc_two_pages_nearby(answer: str) -> bool:
    """Third conflict shape, for the per-sentence citation style: the SAME
    document cited with two DIFFERENT pages within a few adjacent lines.
    The cite-at-the-sentence prompt puts each conflicting figure on its own
    bullet with its own bracket, so the two values no longer share a sentence
    and the ~46-char doc ids push the page marks past _PAGE_WINDOW — the old
    two shapes read a clearer conflict report as no report at all (measured:
    release-3's five conflict answers all report both figures, all failed).
    Requiring the same document keeps an ordinary two-document comparison
    from matching."""
    marks = []
    for i, line in enumerate((answer or "").splitlines()):
        for m in re.finditer(
                r"([0-9]{1,3}_[\w.\-]+)[^\][]*?\bpp?\.?\s*(\d{1,3})\b", line):
            marks.append((i, m.group(1), int(m.group(2))))
    for i, (l1, d1, p1) in enumerate(marks):
        for l2, d2, p2 in marks[i + 1:]:
            if l2 - l1 > _SAME_DOC_LINES:
                continue
            if d1 == d2 and p1 != p2:
                return True
    return False


def _two_pages_close_together(answer: str) -> bool:
    """Two DIFFERENT page numbers within _PAGE_WINDOW characters, in PROSE.
    Requiring them to differ keeps an ordinary two-document citation
    ('… p. 5] and … p. 5]') from reading as a page-vs-page contradiction —
    and pages inside doc-id brackets are excluded entirely: they belong to
    citations, whose meaning `_same_doc_two_pages_nearby` judges (a compact
    cross-document citation pair is a comparison, not a conflict)."""
    prose = re.sub(r"\[[0-9]{1,3}_[^\]]*\]", " ", answer or "")
    marks = [(m.start(), m.group(1)) for m in _PAGE_RE.finditer(prose)]
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
    return (_two_values_in_a_sentence(answer)
            or _two_pages_close_together(answer)
            or _same_doc_two_pages_nearby(answer))


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
def score_answer(case: dict, answer: str, hits: list, notes=None) -> dict:
    from gcf_qna.app import chainlit_app as app
    e = case["expect"]
    contains = {p: matches(p, answer) for p in e["must_contain"]}
    forbidden = {p: (not matches(p, answer)) for p in e["must_not_contain"]}
    # The app passes _note_pages(notes) so a page a computed note prints
    # ('18.5 M USD (p.5, A.8)') is a legal citation even when retrieval never
    # returned that page. Scoring without it flagged answers for citing the
    # registry's own provenance (measured: 8 of release-3's 13 regressions).
    note_pages = app._note_pages(notes) if notes else frozenset()
    bad_cites = (app._invalid_citations(answer or "", hits, note_pages)
                 if hits else [])
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
# The four keys `--compare` pairs up. They must be DISTINCT: two names for one
# number would double-count a single regression and show four green deltas
# where three exist.
METRIC_KEYS = ("support_rate", "groundedness_rate", "citation_completeness_rate",
               "citation_presence_rate")
assert len(set(METRIC_KEYS)) == len(METRIC_KEYS), "compared metric keys collide"


def grounded_flags(claims, evidence) -> list:
    """Per claim: does ANY evidence this turn held entail it?

    The groundedness definition, fixed before any baseline is taken. At
    verify.py@HEAD `_verify_against` is only ever run over the claim's CITED
    scope, so a cited-but-wrong-page claim fails groundedness and citation
    completeness alike and the split carries no independent information. Here
    the same matcher — verify's own, unmodified — is run over the union of the
    turn's evidence, so "the answer knew this but pointed at the wrong page"
    separates from "nothing we retrieved says this".

    Relaxes no matcher and reads no new source: the text is what the prompt
    carried.
    """
    blob = verify._text_of(evidence, list(evidence))
    out = []
    for c in claims:
        try:
            ok, _missing = verify._verify_against(c, blob)
        except Exception:                            # noqa: BLE001
            ok = False
        out.append(bool(ok))
    return out


def claim_metrics(claims, verdicts, evidence, full_failures: bool = False) -> dict:
    """The claims block: three n/d, their denominators, and the failure list.

    Split out of score_claims so the production path (verify.verify_answer,
    whose verdicts carry judge promotions) and the deterministic path report
    the identical shape.
    """
    grounded = grounded_flags(claims, evidence)
    by_index = {id(v.claim): g for v, g in zip(verdicts, grounded)} \
        if len(verdicts) == len(claims) else {}
    n = defaultdict(int)
    failures = []
    cited = citation_supported = 0
    for i, v in enumerate(verdicts):
        n[v.status] += 1
        has_cite = bool(v.claim.citations)
        cited += bool(has_cite)
        if v.status == verify.SUPPORTED and has_cite:
            citation_supported += 1
        if v.status != verify.SUPPORTED:
            failures.append({"status": v.status, "kind": v.claim.kind,
                             "text": v.claim.text[:160], "reason": v.reason[:160],
                             "cited": has_cite,
                             "grounded": bool(by_index.get(id(v.claim), False)),
                             "source": getattr(v, "source", "deterministic")})
    total = len(verdicts)
    supported = n[verify.SUPPORTED]
    n_grounded = sum(1 for v in verdicts if by_index.get(id(v.claim), False)) \
        if by_index else sum(grounded)
    return {
        "claims": total,
        "supported": supported,
        "contradicted": n[verify.CONTRADICTED],
        "unsupported": n[verify.UNSUPPORTED],
        "support_rate": (supported / total) if total else None,
        "grounded": n_grounded,
        "groundedness_rate": (n_grounded / total) if total else None,
        "citation_supported": citation_supported,
        "citation_completeness_rate": (citation_supported / total) if total else None,
        "cited": cited,
        "citation_presence_rate": (cited / total) if total else None,
        "judge_promotions": sum(1 for v in verdicts
                                if getattr(v, "source", "") == "llm"
                                and v.status == verify.SUPPORTED),
        "evidence_keys": [f"{d}|{p if p is not None else '-'}" for d, p in evidence],
        # F9: the aggregates count every failure, so a truncated list that the
        # inventory export reads as complete is a silent undercount. The count
        # travels WITH the list, and a release run keeps the whole thing.
        "n_failures": len(failures),
        "failures": failures if full_failures else failures[:6],
    }


# verify.adjudicate's own default, and therefore production's judge budget.
# Recorded per case so "the cap bound" can never be an unstated assumption.
JUDGE_MAX_CLAIMS = 12


def verify_production(answer: str, hits, notes=None, client=None,
                      full_failures: bool = False):
    """production's verification pass over one answer.

    Returns (RepairResult, claims block, judge accounting). `verify_answer` is
    called as the app calls it — same entry point, same switches — rather than
    reassembled from its parts, so a change inside it reaches this harness.

    The deterministic pass is repeated here for ONE reason: to count the
    judge's candidate set (F9). It is pure python and costs no call, and it is
    the only way to know whether the 12-claim cap bound without reaching into
    verify.py, which is frozen for this wave.
    """
    blocks = [n for n in (notes or []) if n]
    evidence = verify.build_evidence(hits or [], blocks)
    claims = verify.extract_claims(answer or "")
    det = verify.classify_deterministic(claims, evidence)
    candidates = [v for v in det
                  if v.status == verify.UNSUPPORTED and v.plausible]
    res = verify.verify_answer(answer or "", evidence, client=client,
                               use_llm=True)
    block = claim_metrics([v.claim for v in res.verdicts], res.verdicts,
                          evidence, full_failures=full_failures)
    block["verifier_mode"] = "production"
    block["verify_status"] = res.status
    # Always False now that the repair pathway is removed (pure detector);
    # the keys stay because every frozen release record carries them.
    block["repaired"] = bool(getattr(res, "repaired", False))
    block["repair_rejected"] = bool(getattr(res, "repair_rejected", False))
    judge = {"judge_candidates": len(candidates),
             "judge_max_claims": JUDGE_MAX_CLAIMS,
             "judge_budget_exhausted": int(len(candidates) > JUDGE_MAX_CLAIMS)}
    return res, block, judge


def score_claims(answer: str, hits, notes=None, full_failures: bool = False):
    """Deterministic claim-level verdicts against THIS turn's own evidence.

    The old citation check asked 'does the answer cite a page we retrieved?',
    which a fabricated figure on a real page passes. This asks whether the
    cited evidence states the claim: verify.extract_claims over the answer,
    verify.build_evidence over the very hits and notes the harness put in the
    prompt, verify.classify_deterministic between them. Pure python — no judge
    model, no second API call, no repair pass.
    """
    blocks = [n for n in (notes or []) if n]
    evidence = verify.build_evidence(hits or [], blocks)
    claims = verify.extract_claims(answer or "")
    verdicts = verify.classify_deterministic(claims, evidence)
    out = claim_metrics(claims, verdicts, evidence, full_failures=full_failures)
    # Scoped identity, deterministic mode only: an uncited claim is
    # UNSUPPORTED at verify.py@HEAD, so these two cannot differ here. If they
    # ever do, one of them is measuring something else.
    assert out["supported"] == out["citation_supported"], (
        "deterministic scoped identity broken: supported "
        f"{out['supported']} != citation_supported {out['citation_supported']}")
    return out


# ---------------------------------------------------------------------------
# usage accounting
# ---------------------------------------------------------------------------
def verify_call_role(kwargs: dict) -> str:
    """judge | other, from the system prompt the call carries.

    The judge is the pure detector's only model call; "repair" stopped being
    a possible label when that pathway was removed. "verify-other" survives so
    a call this harness does not recognise is booked loudly, not lost.
    """
    system = ""
    for m in kwargs.get("messages") or []:
        if m.get("role") == "system":
            system = m.get("content")
            break
    if system is verify.ADJUDICATE_PROMPT or system == verify.ADJUDICATE_PROMPT:
        return "judge"
    return "verify-other"


class _MeteredCompletions:
    def __init__(self, inner, sink):
        self._inner, self._sink = inner, sink

    def create(self, **kwargs):
        t0 = time.perf_counter()
        resp = self._inner.create(**kwargs)
        self._sink.append(call_meta(verify_call_role(kwargs), resp,
                                    time.perf_counter() - t0))
        return resp


class _MeteredChat:
    def __init__(self, inner, sink):
        self.completions = _MeteredCompletions(inner.chat.completions, sink)


class MeteredClient:
    """An OpenAI client that books what verify.py spends.

    Only `chat.completions.create` is proxied because that is the only surface
    `verify._complete` touches; anything else is delegated, so a future call
    through another attribute still works and is simply not metered — visibly,
    since the record's call list would then not add up to the response's own
    token totals.
    """

    def __init__(self, inner, sink: list):
        self._inner = inner
        self.chat = _MeteredChat(inner, sink)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def call_meta(role: str, resp, dt: float, attempts: int = 1) -> dict:
    """One model call, priced and attributed.

    `role` is the whole point: a release run that books only the answer call
    reports a cost and a latency production does not have. Conductor, answer,
    judge and repair are separate rows under one turn.
    """
    u = getattr(resp, "usage", None)
    pt = int(getattr(u, "prompt_tokens", 0) or 0)
    ct = int(getattr(u, "completion_tokens", 0) or 0)
    return {"role": role, "latency_s": round(dt, 3), "attempts": attempts,
            "model": config.CHAT_MODEL,
            "snapshot": getattr(resp, "model", None) or config.CHAT_MODEL,
            "prompt_tokens": pt, "completion_tokens": ct,
            "total_tokens": int(getattr(u, "total_tokens", 0) or 0) or (pt + ct)}


def turn_usage(calls: list) -> dict:
    """Roll a turn's calls into the per-case `usage` block.

    `latency_s` stays the ANSWER call alone, so a recorded run is still
    comparable with release-1, which measured nothing else; `turn_latency_s`
    is what the turn actually took across every call it made, and it is what
    the report's p50/p95 are computed from.
    """
    calls = [c for c in calls if c]
    if not calls:
        return {}
    answer = next((c for c in calls if c["role"] == "answer"), None)
    return {"calls": calls,
            "latency_s": (answer or {}).get("latency_s"),
            "turn_latency_s": round(sum(c.get("latency_s") or 0.0 for c in calls), 3),
            "model": config.CHAT_MODEL,
            "snapshot": (answer or calls[0]).get("snapshot"),
            "prompt_tokens": sum(c["prompt_tokens"] for c in calls),
            "completion_tokens": sum(c["completion_tokens"] for c in calls),
            "total_tokens": sum(c["total_tokens"] for c in calls),
            "roles": sorted({c["role"] for c in calls})}


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
    ans_lat, turn_lat, prompt, completion, calls = [], [], 0, 0, 0
    by_role = defaultdict(lambda: {"calls": 0, "prompt_tokens": 0,
                                   "completion_tokens": 0, "latency_s": 0.0})
    for r in rows:
        u = r.get("usage") or {}
        if not u:
            continue
        sub_calls = u.get("calls")
        if sub_calls:
            for c in sub_calls:
                role = c.get("role") or "answer"
                calls += 1
                by_role[role]["calls"] += 1
                by_role[role]["prompt_tokens"] += int(c.get("prompt_tokens") or 0)
                by_role[role]["completion_tokens"] += int(c.get("completion_tokens") or 0)
                by_role[role]["latency_s"] += float(c.get("latency_s") or 0.0)
                prompt += int(c.get("prompt_tokens") or 0)
                completion += int(c.get("completion_tokens") or 0)
            if u.get("turn_latency_s") is not None:
                turn_lat.append(float(u["turn_latency_s"]))
            if u.get("latency_s") is not None:
                ans_lat.append(float(u["latency_s"]))
        else:                       # a run recorded before per-call accounting
            calls += 1
            by_role["answer"]["calls"] += 1
            by_role["answer"]["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
            by_role["answer"]["completion_tokens"] += int(u.get("completion_tokens") or 0)
            prompt += int(u.get("prompt_tokens") or 0)
            completion += int(u.get("completion_tokens") or 0)
            if u.get("latency_s") is not None:
                by_role["answer"]["latency_s"] += float(u["latency_s"])
                ans_lat.append(float(u["latency_s"]))
                turn_lat.append(float(u["latency_s"]))
    cost = (prompt * TOKEN_COST_USD["prompt"]
            + completion * TOKEN_COST_USD["completion"])
    return {"calls": calls, "latency": turn_lat,
            "p50": percentile(turn_lat, 0.50), "p95": percentile(turn_lat, 0.95),
            "answer_p50": percentile(ans_lat, 0.50),
            "answer_p95": percentile(ans_lat, 0.95),
            "max": max(turn_lat) if turn_lat else None,
            "sum": round(sum(turn_lat), 1),
            "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "by_role": {k: dict(v, latency_s=round(v["latency_s"], 1))
                        for k, v in sorted(by_role.items())},
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
    """The RETIRED stand-in for the app's `decomposed` flag.

    It guessed from the question's shape — two or more identifiers — what the
    app decides from the plan the conductor and planner actually produced.
    Kept, and still selectable via --comparison-flag proxy, because a
    zero-API run has no plan to read; it is no longer what --release uses.
    """
    return len(fp_ids(question)) + len(set(_BOARD_CODE_RE.findall(question))) > 1


# ---------------------------------------------------------------------------
# parity metadata
# ---------------------------------------------------------------------------
# Gap 1 of the Wave-3 table. The claim under audit is "production's single-FP
# pre-scoping also runs here"; an audit that answers it with a hard-coded
# True is worth nothing, so the app's own _prescope_single_fp is wrapped in a
# transparent counter and the record carries what it actually did. The wrapper
# calls the original and returns its result unchanged — it is a tally, not a
# behaviour change — and it sits on the app module because that is where
# _rescope_items looks the name up.
_PRESCOPE_STATS = {"calls": 0, "tagged": 0, "wrapped": False}


def _instrument_prescope(app) -> dict:
    """Count calls to the app's single-FP prescope. Idempotent."""
    fn = getattr(app, "_prescope_single_fp", None)
    if fn is None or getattr(fn, "_eval_counted", False):
        return _PRESCOPE_STATS
    def counted(items, msg_text, _orig=fn):
        before = [bool(i.get("doc")) for i in (items or [])]
        out = _orig(items, msg_text)
        _PRESCOPE_STATS["calls"] += 1
        _PRESCOPE_STATS["tagged"] += sum(
            1 for was, item in zip(before, out or []) if not was and item.get("doc"))
        return out
    counted._eval_counted = True
    counted.__doc__ = fn.__doc__
    app._prescope_single_fp = counted
    _PRESCOPE_STATS["wrapped"] = True
    return _PRESCOPE_STATS


class Pipeline:
    """chainlit_app.main() without the Chainlit I/O.

    Every production stage is reachable and every one is a switch, so the same
    object serves the zero-API per-commit run and the production-parity
    release run. What it is NOT is a re-implementation: the planner, the
    conductor's prompt and guards, the prescope, the tag resolver, the
    refs note and the answer-message assembly are all imported from the app.
    """

    def __init__(self, top_k: int = None, comparison_flag: str = "decomposed",
                 raw_retrieval: bool = False, scope_single_id: bool = False,
                 history_mode: str = "isolated",
                 production_planner: bool = False, conductor: bool = False,
                 client=None, pins: dict = None,
                 verifier_mode: str = "deterministic"):
        from gcf_qna.app import chainlit_app as app
        self.app = app
        _instrument_prescope(app)
        self.top_k = top_k or config.TOP_K
        if comparison_flag not in COMPARISON_FLAGS:
            raise SystemExit(f"--comparison-flag must be one of {sorted(COMPARISON_FLAGS)}")
        self.comparison_flag = comparison_flag
        if history_mode not in ("isolated", "prepend"):
            raise SystemExit("--history-mode must be isolated|prepend")
        self.history_mode = history_mode
        self.production_planner = production_planner
        self.planner_stats = {"detected": 0, "intent_ok": 0, "matrix_built": 0,
                              "matrix_failed": 0}
        self.conductor = conductor
        self.client = client
        self.pins = pins if pins is not None else _pinning()
        self.conductor_stats = {"calls": 0, "fanned_out": 0, "chat": 0,
                                "failed": 0}
        if verifier_mode not in ("deterministic", "production"):
            raise SystemExit("--verifier-mode must be deterministic|production")
        self.verifier_mode = verifier_mode
        self.raw_retrieval = raw_retrieval
        self.scope_single_id = scope_single_id
        t0 = time.perf_counter()
        self.retriever = app.get_retriever()
        if self.retriever is None:
            raise SystemExit(f"no index at {app._index_dir()} — build one first "
                             "(see scripts/build_index.py)")
        self.load_seconds = time.perf_counter() - t0
        self.meta = dict(app._retriever_meta)

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

    def _decomposed(self, items: list, question: str, plan=None) -> bool:
        """Did this turn fan out? (the app's `decomposed`)"""
        if self.comparison_flag == "off":
            return False
        if self.comparison_flag == "proxy":
            return multi_identifier(question)
        return len(items) > 1 or plan is not None

    # -- the LLM conductor (config.CONDUCTOR in the app) -------------------
    def conduct(self, question: str, turns=()):
        """(mode, search queries, call metadata) — the app's run_conductor.

        Best-effort exactly as there: any failure leaves the raw message as
        the only query, and the original wording still goes to the answer
        model. The rewrite guards run on the parsed output before it is
        adopted, because an unguarded conductor tag is the contamination they
        exist for.
        """
        app = self.app
        items = [{"q": question, "doc": None}]
        if not self.conductor or self.client is None:
            return "retrieve", items, None
        mode, meta = "retrieve", None
        history = [{"role": m["role"], "content": m["content"]}
                   for m in (turns or [])]
        try:
            convo = ("\n".join(f"{m['role']}: {m['content'][:1200]}"
                                for m in history[-6:])
                     if history else "((no prior conversation))")
            cited = app._cited_docs(history)
            if cited:
                convo += ("\nDocuments cited in conversation: "
                          + ", ".join(cited[-12:]))
            t0 = time.perf_counter()
            resp = self.client.chat.completions.create(
                model=config.CHAT_MODEL,
                max_completion_tokens=300,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": app.CONDUCTOR_PROMPT},
                          {"role": "user", "content":
                           f"Conversation:\n{convo}\n\nLatest message: {question}"}],
                **(self.pins or {}),
            )
            meta = call_meta("conductor", resp, time.perf_counter() - t0)
            self.conductor_stats["calls"] += 1
            data = json.loads(resp.choices[0].message.content or "{}")
            if data.get("mode") == "chat":
                mode = "chat"
                self.conductor_stats["chat"] += 1
            parsed = []
            for item in (data.get("queries") or [])[:6]:
                if isinstance(item, str) and item.strip():
                    parsed.append({"q": item.strip(), "doc": None})
                elif isinstance(item, dict) and (item.get("q") or "").strip():
                    parsed.append({"q": item["q"].strip(),
                                   "doc": item.get("doc") or None})
            parsed = app._rescope_items(parsed, question, cited)
            if parsed:
                items = parsed
                if len(parsed) > 1:
                    self.conductor_stats["fanned_out"] += 1
        except Exception:                            # noqa: BLE001
            self.conductor_stats["failed"] += 1
        return mode, items, meta

    # -- the deterministic comparison planner (config.PLANNER in the app) ---
    def planner_plan(self, question: str):
        """(plan, matrix_block) for a question, or (None, None).

        `detect` fires on any message naming >= 2 documents, comparative or
        not, so the app's intent gate runs behind it; a matrix that carries no
        evidence at all is discarded exactly as `main` discards it, and the
        caller then proceeds as PLANNER=0 would.
        """
        if not self.production_planner:
            return None, None
        plan = planner.detect(question)
        if plan is None:
            return None, None
        self.planner_stats["detected"] += 1
        if not self.app._planner_intent(question, plan):
            return None, None
        self.planner_stats["intent_ok"] += 1
        try:
            matrix = planner.build_matrix(plan, self.retriever)
            if not any(c.status not in ("missing", "missing-document")
                       for c in matrix.cells):
                raise ValueError("no cell carries evidence")
            block = planner.render(matrix)
        except Exception:                            # noqa: BLE001
            self.planner_stats["matrix_failed"] += 1
            return None, None
        self.planner_stats["matrix_built"] += 1
        return plan, block

    def _retrieve(self, items: list, decomposed: bool, original: str = None):
        """(hits, best confidence, weak-signal flag) — the app's fan-out.

        Per-query quota and round-robin merge, verbatim from `main`: the global
        cap must not starve the later documents of a multi-document turn.
        """
        from itertools import zip_longest
        per_query = (self.top_k if not decomposed
                     else max(3, self.top_k // max(1, len(items))))
        best, weak, per_lists = None, True, []
        for sq in items:
            # Mirror the app: on a single-query turn the user's own words get a
            # second dense vote on WHICH PAGES of the settled document rank
            # first. Without this the harness measures a retriever production
            # does not run (chainlit_app.py passes `original` at its one call
            # site); the document set cannot move either way.
            got, conf = self.retriever.search_with_confidence(
                sq["q"], per_query, sq.get("doc"),
                original=original if len(items) == 1 else None)
            best = conf if best is None else max(best, conf)
            if conf >= config.MIN_DENSE_SCORE:
                weak = False
            per_lists.append(got)
        seen, hits = set(), []
        for tier in zip_longest(*per_lists):
            for h in tier:
                if h is None:
                    continue
                key = (h.doc_id, h.page, h.text[:120])
                if key not in seen:
                    seen.add(key)
                    hits.append(h)
        return hits[:15], (best if best is not None else 0.0), weak

    # -- what of production's answer path this harness actually ran ---------
    def parity(self) -> dict:
        """The `pipeline_parity` block, rebuilt per case from observation.

        Every key is either a switch this run was started with or something
        the run measured. `level` is graded in _parity_level, which is the
        only place allowed to say "full".
        """
        return {
            "production_single_id_prescope": bool(_PRESCOPE_STATS["wrapped"]),
            "prescope_calls": _PRESCOPE_STATS["calls"],
            "prescope_tagged": _PRESCOPE_STATS["tagged"],
            "comparison_flag": self.comparison_flag,
            "answer_history_isolation": self.history_mode == "isolated",
            "guard_verification_skipped": True,
            "abstain_keeps_original": True,
            "conductor": {"enabled": bool(self.conductor),
                          "used": self.conductor_stats["calls"] > 0,
                          **self.conductor_stats},
            "planner": {"enabled": bool(self.production_planner),
                        "detected": self.planner_stats["intent_ok"],
                        "matrix_built": self.planner_stats["matrix_built"],
                        "matrix_failed": self.planner_stats["matrix_failed"]},
            "verifier_mode": self.verifier_mode,
            "usage_accounts_judge_and_repair": True,
        }

    def run(self, question: str, turns=()) -> dict:
        app = self.app
        calls = []

        # 1. planner, 2. conductor when the planner declined — the app's order.
        plan, matrix_block = self.planner_plan(question)
        mode, items, cmeta = "retrieve", [{"q": question, "doc": None}], None
        if plan is None:
            mode, items, cmeta = self.conduct(question, turns)
            if cmeta:
                calls.append(cmeta)

        if mode == "chat":
            # conversational turn: answered from history, no retrieval, no
            # evidence — so nothing here is ever verified (see main()).
            lang = app._detect_lang(question)
            system = app.assemble_chat(lang)
            history = [{"role": m["role"], "content": m["content"]}
                       for m in (turns or [])]
            return {"guard": False, "chat": True, "guard_answer": None,
                    "hits": [], "confidence": None, "weak": False,
                    "plan": items, "decomposed": False, "system": system,
                    "context": "", "refs_note": None,
                    "user": question, "calls": calls,
                    "messages": [{"role": "system", "content": system}]
                                + history + [{"role": "user", "content": question}],
                    "notes": {"registry": None, "year": None, "board": None,
                              "matrix": None}}

        # 3. the registry FP-miss guard, AFTER the conductor as in main()
        guard = self.fp_guard(question)
        if guard is not None:
            # the guard answers FROM the registry, so the registry lookup is
            # the evidence this turn held — recording it lets the claim-support
            # scorer audit a guard answer against what produced it
            try:
                reg = registry.registry_note(question)
            except Exception:
                reg = None
            return {"guard": True, "chat": False, "guard_answer": guard,
                    "hits": [], "system": None, "user": None, "weak": False,
                    "plan": items, "decomposed": False, "calls": calls,
                    "notes": {"registry": reg, "year": None, "board": None,
                              "matrix": None}}

        if plan is not None:
            # Authoritative stems and an English query per document: the raw
            # message is the wrong query here, and the conductor that would
            # have translated it was skipped (see _plan_query).
            items = [{"q": app._plan_query(plan, d), "doc": d.scope}
                     for d in plan.docs if not d.missing] or items
        elif self.conductor:
            # the app's step 7 over the conductor's own output: pre-scope a
            # lone untagged query, then registry-resolve every surviving tag
            items = app._resolve_doc_tags(
                app._prescope_single_fp(items, question))
        else:
            items = self.plan(question)
        decomposed = self._decomposed(items, question, plan)
        hits, conf, weak = self._retrieve(items, decomposed, original=question)
        hits, year_note = app._year_assist(question, hits)
        board_note = app._board_range_note(question)
        if board_note:
            year_note = f"{year_note} {board_note}" if year_note else board_note

        coverage_note = app._corpus_coverage_note(question)
        if coverage_note:
            year_note = (f"{year_note} {coverage_note}" if year_note
                         else coverage_note)

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
            # The app's second trigger, on the app's own function: the
            # question's words are not the only evidence of which document the
            # turn is about — a follow-up spells no identifier and its resolved
            # query spells one. Same items retrieval just ran on.
            reg_note = app._extend_registry_note(reg_note, items)
            if reg_note:
                context = reg_note + "\n\n" + context
        except Exception:
            pass
        if matrix_block:
            # ABOVE the registry note and the excerpts, as in the app: the
            # matrix is the complete half of the evidence.
            context = matrix_block + "\n\n" + context

        system = assemble(year=bool(year_note), registry=bool(reg_note),
                          comparison=decomposed, matrix=bool(matrix_block),
                          lang=app._detect_lang(question))
        # The referents a follow-up needs, as ids rather than as prose — the
        # app's own note, built from the same items the retrieval used.
        refs_note = app._resolved_refs_note(items, question)
        user = f"Context excerpts:\n{context}\n\nQuestion: {question}"
        if refs_note:
            user = f"{refs_note}\n\n{user}"
        return {
            "guard": False, "chat": False, "guard_answer": None, "hits": hits,
            "confidence": conf, "weak": weak, "plan": items, "calls": calls,
            "decomposed": decomposed,
            "system": system,
            "context": context,
            "refs_note": refs_note,
            "user": user,
            "messages": app._answer_messages(system, context, question, refs_note),
            "notes": {"registry": reg_note, "year": year_note,
                      "board": board_note, "matrix": matrix_block},
        }


# ---------------------------------------------------------------------------
# run pinning (F10)
# ---------------------------------------------------------------------------
# Every --release run used to be an independent sample of the answer
# distribution: no temperature, no seed, and the ALIAS 'gpt-5.2' recorded in
# place of the snapshot the endpoint actually served. Two runs of the same
# tree were therefore not comparable, and a snapshot rotation would have moved
# every number with nothing in the record to show it. Both are pinned here and
# the served snapshot is read back off each response.
PIN_TEMPERATURE = 0.0
PIN_SEED = 20260819


def _pinning(temperature=PIN_TEMPERATURE, seed=PIN_SEED) -> dict:
    """Sampling parameters sent with EVERY call this harness makes.

    ``None`` for either drops it from the request, which is how a run against
    an endpoint that rejects the parameter is recorded honestly rather than
    silently unpinned.
    """
    out = {}
    if temperature is not None:
        out["temperature"] = temperature
    if seed is not None:
        out["seed"] = seed
    return out


def ask_model(client, system: str, turns: list, user: str, pins: dict = None,
              messages: list = None):
    """(answer text, call metadata).

    The metadata is the release report's raw material: wall-clock latency of
    the call that succeeded, the API's OWN token counts — estimating tokens
    from the prompt string would be a second, wronger measurement of something
    the response already reports — and the model SNAPSHOT the endpoint served,
    read off the response rather than copied from the alias we asked for.

    ``messages`` overrides the default assembly. The default still prepends
    ``turns`` as conversation; production does not (see _answer_messages and
    the history-isolation gap), so the release path passes its own array.
    """
    if messages is None:
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
                **(pins if pins is not None else _pinning()),
            )
            dt = time.perf_counter() - t0
            u = getattr(resp, "usage", None)
            pt = int(getattr(u, "prompt_tokens", 0) or 0)
            ct = int(getattr(u, "completion_tokens", 0) or 0)
            meta = {"latency_s": round(dt, 3), "attempts": attempt + 1,
                    "model": config.CHAT_MODEL,
                    "snapshot": getattr(resp, "model", None) or config.CHAT_MODEL,
                    "system_fingerprint": getattr(resp, "system_fingerprint", None),
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "total_tokens": int(getattr(u, "total_tokens", 0) or 0) or (pt + ct)}
            return (resp.choices[0].message.content or ""), meta
        except Exception as e:                       # transient 429/5xx
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"chat call failed after 3 attempts: {last}")


# ---------------------------------------------------------------------------
# what the user is shown  (chainlit_app._verify_reply)
# ---------------------------------------------------------------------------
def final_answer(original: str, res) -> tuple:
    """(body the app would display, where it came from).

    Mirrors `_verify_reply` exactly: the verifier is a pure detector, so the
    body is ALWAYS the answer as the model wrote it. ``abstain`` keeps that
    body too, led by a banner — the answer is not deleted, it is captioned.
    ``res is None`` (verification off, or it raised) is the same guarantee by
    another route.

    ``res.answer`` is deliberately not read: nothing the verifier returns can
    reach the display, because the repair pathway was removed.
    """
    if res is not None and getattr(res, "status", None) == "abstain":
        return original, "abstain-original"
    return original, "model"


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
           f"{'field':>6} {'claim':>6} {'p50':>7} {'p95':>7}")
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

    print("\nCLAIM SUPPORT — verify.extract_claims vs the turn's own evidence")
    tot = sum(c["claims"] for c in claims)
    sup = sum(c["supported"] for c in claims)
    con = sum(c["contradicted"] for c in claims)
    uns = sum(c["unsupported"] for c in claims)
    grd = sum(c.get("grounded") or 0 for c in claims)
    cis = sum(c.get("citation_supported") or 0 for c in claims)
    cit = sum(c.get("cited") or 0 for c in claims)
    rate = (sup / tot) if tot else 0.0
    need = -(-int(CLAIM_SUPPORT_GATE * 1000) * tot // 1000)   # ceil(0.95 * d)
    print(f"  claims                      : {tot} over {len(claims)} answers")
    print(f"  supported                   : {sup}/{tot} ({rate:.1%})  "
          f"gate >= {need}/{tot} — "
          f"{'PASS' if tot and sup >= need else 'FAIL'}")
    print(f"  groundedness                : {grd}/{tot} "
          f"({(grd / tot if tot else 0):.1%})   "
          f"[any held evidence entails the claim]")
    print(f"  citation completeness       : {cis}/{tot} "
          f"({(cis / tot if tot else 0):.1%})   [cited AND supported]")
    print(f"  citation presence           : {cit}/{tot} "
          f"({(cit / tot if tot else 0):.1%})   [reported, never gated]")
    print(f"  contradicted                : {con}")
    print(f"  unsupported                 : {uns}")
    _print_claim_breakdown(scored)

    skips = [r for r in rows if r.get("claims_skipped")]
    if skips:
        removed = sum(s["claims_skipped"]["claims_removed"] for s in skips)
        sup_removed = sum(s["claims_skipped"]["supported_removed"] for s in skips)
        print("\nDENOMINATOR — claims REMOVED from the population above")
        print(f"  unverified-turn skip: {len(skips)} cases, "
              f"{removed} claims ({sup_removed} of them supported)")
        for s in skips:
            print(f"    {s['id']:26} -{s['claims_skipped']['claims_removed']} "
                  f"claims   {s['claims_skipped']['reason'][:44]}")

    jc = [r for r in rows if r.get("judge_candidates") is not None]
    if any(r.get("verify_status") for r in rows):
        exhausted = sum(int(r.get("judge_budget_exhausted") or 0) for r in jc)
        cand = sum(int(r.get("judge_candidates") or 0) for r in jc)
        print("\nJUDGE BUDGET (F9)")
        print(f"  candidates sent            : {cand} over {len(jc)} cases "
              f"(cap {JUDGE_MAX_CLAIMS}/case)")
        print(f"  budget exhausted           : {exhausted}/{len(jc)} cases "
              f"— gate 0 — {'PASS' if exhausted == 0 else 'FAIL'}")
        st = defaultdict(int)
        for r in rows:
            if r.get("verify_status"):
                st[r["verify_status"]] += 1
        print("  verifier status            : "
              + ", ".join(f"{k} {v}" for k, v in sorted(st.items())))

    print("\nLATENCY / COST")
    print(f"  model calls                 : {u['calls']}")
    print(f"  turn latency p50 / p95      : "
          + (f"{u['p50']:.1f}s / {u['p95']:.1f}s" if u["p50"] is not None else "n/a")
          + (f"   (max {u['max']:.1f}s, {u['sum']:.0f}s of model wall-clock)"
             if u["max"] is not None else "")
          + "   [every call the turn made]")
    print(f"  answer-call p50 / p95       : "
          + (f"{u['answer_p50']:.1f}s / {u['answer_p95']:.1f}s"
             if u.get("answer_p50") is not None else "n/a")
          + "   [comparable with release-1, which measured only this]")
    print(f"  tokens                      : prompt {u['prompt_tokens']:,} + "
          f"completion {u['completion_tokens']:,} = {u['total_tokens']:,}")
    if u.get("by_role"):
        print("  per role                    :")
        for role, r in u["by_role"].items():
            rc = (r["prompt_tokens"] * TOKEN_COST_USD["prompt"]
                  + r["completion_tokens"] * TOKEN_COST_USD["completion"])
            print(f"    {role:14} {r['calls']:>4} calls  "
                  f"{r['prompt_tokens']:>8,}p + {r['completion_tokens']:>7,}c  "
                  f"{r['latency_s']:>7.1f}s  ${rc:.3f}")
    print(f"  estimated cost              : ${u['cost_usd']:.2f}  "
          f"(ESTIMATED rates: ${TOKEN_COST_USD['prompt'] * 1e6:.2f}/1M prompt, "
          f"${TOKEN_COST_USD['completion'] * 1e6:.2f}/1M completion)")

    print("\nERRORED CASES")
    if not errored:
        print("  none")
    else:
        for r in errored:
            print(f"  {r['id']:26} {r['error'][:110]}")
    return {"claims_skipped": {
                "cases": [r["id"] for r in rows if r.get("claims_skipped")],
                "claims_removed": sum(r["claims_skipped"]["claims_removed"]
                                      for r in rows if r.get("claims_skipped")),
                "supported_removed": sum(r["claims_skipped"]["supported_removed"]
                                         for r in rows if r.get("claims_skipped"))},
            "fields": {"cells": cells, "scorable": scorable, "stated": stated,
                       "marked_missing": marked, "missed": missed,
                       "unscorable": unscorable},
            "claims": {"total": tot, "supported": sup, "contradicted": con,
                       "unsupported": uns, "rate": rate, "grounded": grd,
                       "citation_supported": cis, "cited": cit},
            "usage": u, "errors": [r["id"] for r in errored]}


def _release_line(label: str, rs: list) -> str:
    scored = [r for r in rs if not r.get("error") and r.get("checks")]
    err = sum(1 for r in rs if r.get("error"))
    c = _agg_answer(scored)
    f = _metrics(scored, "fields", "n_cells")
    cl = _metrics(scored, "claims", "claims")
    n = max(1, len(scored))
    u = usage_totals(rs)
    fcov = _pct(sum(x["n_covered"] for x in f), sum(x["n_scorable"] for x in f))
    ccov = _pct(sum(x["supported"] for x in cl), sum(x["claims"] for x in cl))
    return (f"{label:12} {len(rs):>3} {err:>4} {_pct(c['pass'], n):>6} "
            f"{_pct(c['behavior'], n):>6} {_pct(c['ct'], c['cn']):>8} "
            f"{_pct(c['ft'], c['fn']):>7} {_pct(c['language'], n):>6} "
            f"{_pct(c['citations'], n):>6} {fcov:>6} {ccov:>6} "
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
    per = defaultdict(lambda: [0, 0])
    have = [r for r in rows if _metrics([r], "claims", "claims")]
    for r in have:
        c = r["claims"]
        per[r["class"]][0] += c.get("supported", 0)
        per[r["class"]][1] += c.get("claims", 0)
    for cls in sorted(per, key=lambda c: (_class_rank(c), c)):
        sup, tot = per[cls]
        print(f"    {cls:20} {sup:>3}/{tot:<3} {_pct(sup, tot).strip():>4}")
    worst = sorted((r for r in have if r["claims"].get("claims")),
                   key=lambda r: r["claims"]["support_rate"])[:5]
    for r in worst:
        c = r["claims"]
        if c["support_rate"] >= 1.0:
            break
        first = (c["failures"] or [{}])[0]
        print(f"    LOW  {r['id']:26} {c['supported']}/{c['claims']} "
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
    flag = "off" if getattr(args, "no_comparison_proxy", False) else \
        getattr(args, "comparison_flag", "decomposed")
    client = None
    if args.answers:
        import openai
        client = openai.OpenAI(base_url=config.OPENAI_BASE_URL or None)
    pins = _pinning(getattr(args, "temperature", PIN_TEMPERATURE),
                    getattr(args, "seed", PIN_SEED))
    pipe = Pipeline(top_k=args.k, comparison_flag=flag,
                    raw_retrieval=args.raw_retrieval,
                    scope_single_id=args.scope_single_id,
                    history_mode=getattr(args, "history_mode", "isolated"),
                    production_planner=bool(getattr(args, "production_planner",
                                                    False)),
                    conductor=bool(getattr(args, "conductor", False)),
                    client=client, pins=pins,
                    verifier_mode=getattr(args, "verifier_mode", "deterministic"))
    print(f"retriever ready in {pipe.load_seconds:.1f}s — "
          f"{pipe.meta.get('n_chunks')} chunks, {pipe.meta.get('embedding_model')}")
    if args.gate:
        run_gate(pipe, GOLD_SET)

    rows, skipped = [], []
    t0 = time.perf_counter()
    for i, case in enumerate(cases, 1):
        if case["turns"] and not args.answers:
            skipped.append(case)
            continue
        try:
            rec = _run_case(pipe, client, args, case)
        except Exception as e:                  # one bad case never ends a run
            rec = {"id": case["id"], "class": case["class"], "lang": case["lang"],
                   "question": case["question"], "turns": len(case["turns"]),
                   "mode": "answers" if args.answers else "retrieval-only",
                   "guard": False, "expect": case["expect"], "score": 0.0,
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
            lat = u.get("turn_latency_s")
            print(f"[{i}/{len(cases)}] {case['id']:26} score={rec['score']:.2f}"
                  + (f" {lat:.1f}s {u['total_tokens']}tok"
                     if lat is not None else ""),
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


def _verify_client(client, sink: list):
    """The client the verifier gets: a metered proxy, or None with no client."""
    if client is None:
        return None
    return MeteredClient(client, sink)


def _safe(fn, *a, **kw):
    """Run a scorer; a scorer that raises must not throw away an answer that
    cost a model call. The failure is recorded in place of its metrics."""
    try:
        return fn(*a, **kw)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _run_case(pipe, client, args, case: dict) -> dict:
    out = pipe.run(case["question"], case.get("turns") or ())
    rec = {
        "id": case["id"], "class": case["class"], "lang": case["lang"],
        "question": case["question"], "turns": len(case["turns"]),
        "mode": "answers" if args.answers else "retrieval-only",
        "guard": out["guard"], "chat": bool(out.get("chat")),
        "weak_signal": out.get("weak"),
        "plan": out.get("plan") or [],
        "decomposed": out.get("decomposed"),
        "refs_note": out.get("refs_note"),
        "matrix": bool((out.get("notes") or {}).get("matrix")),
        "retrieval": score_retrieval(case, out["hits"]),
        "expect": case["expect"],
        "pipeline_parity": pipe.parity(),
    }
    rec["retrieval_score"] = retrieval_score(rec["retrieval"])
    if not args.answers:
        rec["score"] = rec["retrieval_score"]
        return rec

    answer, verdict_result = out["guard_answer"], None
    calls = list(out.get("calls") or [])
    judge_acct = {"judge_candidates": 0, "judge_max_claims": JUDGE_MAX_CLAIMS,
                  "judge_budget_exhausted": 0}
    prod_block = None
    if answer is None:
        isolated = getattr(args, "history_mode", "isolated") == "isolated"
        answer, ameta = ask_model(
            client, out["system"], () if isolated else case["turns"],
            out["user"],
            pins=_pinning(getattr(args, "temperature", PIN_TEMPERATURE),
                          getattr(args, "seed", PIN_SEED)),
            messages=out.get("messages") if isolated else None)
        ameta["role"] = "answer"
        calls.append(ameta)
    usage = turn_usage(calls)
    notes = out.get("notes") or {}
    if (pipe.verifier_mode == "production" and not out["guard"]
            and not out.get("chat")):
        # The app verifies BEFORE the answer becomes history, on the text the
        # model wrote; what the user is shown is decided from the result.
        verdict_result, prod_block, judge_acct = verify_production(
            answer, out["hits"],
            [notes.get("registry"), notes.get("year"), notes.get("matrix")],
            client=_verify_client(client, calls),
            full_failures=bool(getattr(args, "release", False)))
        usage = turn_usage(calls)
    raw_answer = answer
    answer, answer_source = final_answer(raw_answer, verdict_result)
    rec.update({
        "answer": answer,
        "raw_answer": raw_answer,
        "answer_source": answer_source,
        "checks": score_answer(case, answer, out["hits"],
                               [notes.get("registry"), notes.get("year"),
                                notes.get("matrix")]),
        "model": config.CHAT_MODEL,
        "usage": usage,
        "fields": _safe(score_fields, case, answer),
        # F7: the coordinates alone cannot be adjudicated — 'is this claim
        # entailed by the held evidence?' needs the evidence. Recording the
        # text is what makes a release run replayable without re-retrieval.
        "hits": [{"doc": h.doc_id, "page": _page(h), "score": round(h.score, 4),
                  "text": h.text}
                 for h in out["hits"]],
        "notes_used": {k: v for k, v in notes.items() if v},
    })
    note_blocks = [notes.get("registry"), notes.get("year"), notes.get("matrix")]
    full = bool(getattr(args, "release", False))
    if out["guard"] or out.get("chat"):
        # Production returns here, before build_evidence: a guard answer has no
        # verdicts at all. Scoring it anyway is measured and published, never
        # silently folded into the rate — hence the count on its own key.
        would = _safe(score_claims, answer, out["hits"], note_blocks,
                      full_failures=full)
        rec["claims"] = None
        rec["claims_skipped"] = {
            "reason": ("chat-mode turn: production answers from history and "
                       "builds no evidence" if out.get("chat") else
                       "guard-answer: production returns before verification"),
            "claims_removed": int((would or {}).get("claims") or 0),
            "supported_removed": int((would or {}).get("supported") or 0)}
    elif prod_block is not None:
        rec["claims"] = prod_block
        rec["claims_skipped"] = None
    else:
        rec["claims"] = _safe(score_claims, answer, out["hits"], note_blocks,
                              full_failures=full)
        rec["claims_skipped"] = None
    rec.update(judge_acct)
    rec["verify_status"] = getattr(verdict_result, "status", None)
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


def run_compare(path_a: Path, path_b: Path, require_metrics: list = None):
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
    return _compare_extras(a, b, shared, require_metrics)


def _extra_rates(rec: dict):
    """(field coverage, claim support) of one recorded row, or (None, None).

    Both are post-release keys. A run recorded before they existed simply has
    neither, and --compare must keep working against it — which is the whole
    point of reading them with .get and skipping the pair when either side is
    missing.
    """
    f = rec.get("fields")
    c = rec.get("claims")
    out = {"field coverage": (f or {}).get("coverage") if isinstance(f, dict) else None}
    for name, key in (("claim support", "support_rate"),
                      ("groundedness", "groundedness_rate"),
                      ("citation completeness", "citation_completeness_rate"),
                      ("citation presence", "citation_presence_rate")):
        out[name] = (c or {}).get(key) if isinstance(c, dict) else None
    return out


# metric name -> the record key --require-metrics names it by
_REQUIRABLE = {"support_rate": "claim support",
               "groundedness_rate": "groundedness",
               "citation_completeness_rate": "citation completeness",
               "citation_presence_rate": "citation presence",
               "field_coverage": "field coverage"}


def _compare_extras(a: dict, b: dict, shared: list, require_metrics: list = None):
    pairs = defaultdict(list)
    missing = defaultdict(int)
    for i in shared:
        ra, rb = _extra_rates(a[i]), _extra_rates(b[i])
        for name in ra:
            if ra[name] is not None and rb[name] is not None:
                pairs[name].append((ra[name], rb[name]))
            else:
                missing[name] += 1
    lines = [(name, vs) for name, vs in pairs.items() if vs]
    if lines:
        print("\n(metrics present in both runs)")
    for name, vs in lines:
        ma = sum(x for x, _ in vs) / len(vs)
        mb = sum(y for _, y in vs) / len(vs)
        print(f"  {name:22} {len(vs):>3} cases  {ma:>6.1%} -> {mb:>6.1%} "
              f"{mb - ma:>+7.1%}")
    if not require_metrics:
        return None
    # A required metric that is absent, or None on either side, is a hard
    # failure: "no regression" reported over a metric nobody could compute is
    # the failure mode this flag exists to close.
    report = {"no_regression": True, "metrics": {}, "missing": [],
              "cases_compared": len(shared)}
    for key in require_metrics:
        name = _REQUIRABLE.get(key)
        if name is None:
            report["missing"].append(f"{key}: not a comparable metric")
            report["no_regression"] = False
            continue
        vs = pairs.get(name) or []
        if not vs or missing.get(name):
            report["missing"].append(
                f"{key}: {missing.get(name, len(shared))} of {len(shared)} "
                f"cases carry no value on one side")
            report["no_regression"] = False
            if not vs:
                report["metrics"][key] = {"a": None, "b": None, "delta": None}
                continue
        ma = sum(x for x, _ in vs) / len(vs)
        mb = sum(y for _, y in vs) / len(vs)
        report["metrics"][key] = {"a": ma, "b": mb, "delta": mb - ma,
                                  "cases": len(vs)}
        if mb < ma - 1e-9:
            report["no_regression"] = False
    print("\n" + json.dumps(report, indent=1))
    return report


def _evidence_from_backfill(row: dict) -> dict:
    """The Evidence dict a backfilled row reconstructs.

    The backfill's own precondition — every recorded (doc,page,score) triple
    reproduces — is re-checked here rather than trusted: a rescoring built on
    evidence that does not match what the turn held is a different
    measurement wearing the same label.
    """
    if not row.get("evidence_keys_match", False):
        raise ValueError(f"{row.get('case_id')}: backfilled evidence keys do "
                         "not match the recorded ones")
    out = {}
    for e in row.get("evidence") or []:
        out[(e["doc"], e["page"])] = e["text"]
    return out


def _evidence_from_record(row: dict):
    """The Evidence dict a post-F7 record rebuilds from its own hits."""
    hits = row.get("hits") or []
    if not hits or not all("text" in h for h in hits):
        return None
    from gcf_qna.rag.retrieve import Hit
    notes = [v for v in (row.get("notes_used") or {}).values() if v]
    return verify.build_evidence(
        [Hit(text=h["text"], doc_id=h["doc"], score=float(h.get("score") or 0.0),
             page=h.get("page") or None) for h in hits], notes)


def run_rescore(record: Path, out: Path, evidence: Path = None,
                force: bool = False) -> int:
    """Recompute claims metrics over a recorded run. No model is called.

    The recorded block is kept beside the new one under `claims_recorded`:
    the point of a rescoring is the DIFFERENCE, and a file that overwrites the
    old number has destroyed the comparison it exists to make.
    """
    rows = [json.loads(l) for l in
            record.read_text(encoding="utf-8").splitlines() if l.strip()]
    back = {}
    if evidence:
        for line in evidence.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                back[r.get("case_id") or r.get("id")] = r
    if out.exists() and not force:
        raise SystemExit(_overwrite_refusal(out))

    agg = defaultdict(int)
    done = []
    for row in rows:
        answer = row.get("answer")
        if row.get("error") or not answer:
            done.append(row)
            continue
        src = "record"
        ev = _evidence_from_record(row)
        if ev is None:
            b = back.get(row.get("id"))
            if b is None:
                raise SystemExit(
                    f"{row['id']}: hits carry no text and no backfilled "
                    f"evidence row was supplied (--evidence)")
            ev, src = _evidence_from_backfill(b), "backfill"
            if (b.get("answer") or "") != answer:
                raise SystemExit(f"{row['id']}: backfill answer differs from "
                                 "the recorded answer")
        claims = verify.extract_claims(answer)
        verdicts = verify.classify_deterministic(claims, ev)
        new = claim_metrics(claims, verdicts, ev, full_failures=True)
        old = row.get("claims") if isinstance(row.get("claims"), dict) else {}
        row["claims_recorded"] = old or None
        row["claims"] = new
        row["rescored"] = {"evidence_source": src, "api_calls": 0,
                           "verify_blob_sha": _sha256_file(VERIFY_PY)}
        agg["claims"] += new["claims"]
        agg["supported"] += new["supported"]
        agg["grounded"] += new["grounded"]
        agg["citation_supported"] += new["citation_supported"]
        agg["cited"] += new["cited"]
        agg["was_claims"] += int(old.get("claims") or 0)
        agg["was_supported"] += int(old.get("supported") or 0)
        done.append(row)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in done),
                   encoding="utf-8")
    d, wd = agg["claims"], agg["was_claims"]
    print(f"rescored {record.name} -> {out}")
    print(f"  verify.py blob              : {_sha256_file(VERIFY_PY)[:12]}")
    print(f"  claims                      : {wd} recorded -> {d} rescored "
          f"({d - wd:+d})")
    print(f"  supported (as recorded)     : {agg['was_supported']}/{wd}")
    print(f"  supported (rescored)        : {agg['supported']}/{d}")
    print(f"  citation completeness       : {agg['citation_supported']}/{d}")
    print(f"  groundedness                : {agg['grounded']}/{d}")
    print(f"  citation presence           : {agg['cited']}/{d}")
    if agg["supported"] != agg["citation_supported"]:
        print("  NOTE: supported != citation_supported — this file was not "
              "scored in deterministic mode")
    return 0


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


# ---------------------------------------------------------------------------
# run metadata — what has to be true again for a run to be reproducible
# ---------------------------------------------------------------------------
VERIFY_PY = ROOT / "src" / "gcf_qna" / "rag" / "verify.py"


def artifact_hashes() -> dict:
    """The retrieval and registry artifacts a recorded run was scored against.

    Wave 3's gate diffs these against the deployed fingerprint: a run scored
    on a locally rebuilt index is not a production-parity run however faithful
    the code path is, and without the hashes in the record there is nothing to
    catch that after the fact.
    """
    idx = config.INDEX_DIR / os.getenv("INDEX_NAME", "default")
    return {
        "index_dir": str(idx),
        "index_config_sha256": _sha256_file(idx / "config.json"),
        "index_faiss_sha256": _sha256_file(idx / "index.faiss"),
        "index_chunks_sha256": _sha256_file(idx / "chunks.jsonl"),
        "registry_sha256": _sha256_file(ROOT / "data" / "registry.json"),
        "registry_v2_sha256": _sha256_file(ROOT / "data" / "registry_v2.json"),
    }


# Field-for-field, the deployed switches a `level=full` record must match
# (plan, Wave 3: CONDUCTOR/PLANNER/VERIFY/VERIFY_LLM/INDEX_NAME/
# CHAT_MODEL plus the app commit sha).
FINGERPRINT_FILE = ROOT / "docs" / "deployed-env-fingerprint.txt"
DEPLOYED_LOG = ROOT / "docs" / "DEPLOYED.md"


def deployment_fingerprint(path: Path = None) -> dict:
    """What production is running, from tracked files only.

    Wave −1 step 6 captures `docs/deployed-env-fingerprint.txt` over ssh; that
    is an owner action and it has not been performed, so this falls back to
    the two tracked artifacts that do exist — the `docs/DEPLOYED.md` row the
    deploy printed (sha + the PLANNER/VERIFY/VERIFY_LLM lines of the .env
    that shipped) and the local `.env`, which Wave −1 makes the source of
    truth for production's switches because it rsyncs by design. The source is
    recorded so a reader is never left guessing which one answered.
    """
    path = path or FINGERPRINT_FILE
    out = {"source": None, "sha": None, "switches": {},
           "remote_artifacts_verified": False, "notes": []}
    if path.exists():
        try:
            out["source"] = str(path.relative_to(ROOT))
        except ValueError:
            out["source"] = str(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            m = re.match(r"\s*([A-Z_]+)=(\S*)", line)
            if m:
                out["switches"][m.group(1)] = m.group(2)
        m = re.search(r"fp-gcf:([0-9a-f]{7,40})", text)
        if m:
            out["sha"] = m.group(1)
        out["remote_artifacts_verified"] = "local-artifacts" in text
        return out
    out["source"] = "docs/DEPLOYED.md + .env (deployed-env-fingerprint.txt absent)"
    out["notes"].append(
        "docs/deployed-env-fingerprint.txt is missing (Wave −1 step 6 / owner "
        "action 6 not performed): the deployed switches come from the tracked "
        "DEPLOYED.md row and the local .env, and the REMOTE artifact hashes "
        "were not read — no ssh from this harness.")
    try:
        rows = [l for l in DEPLOYED_LOG.read_text(encoding="utf-8").splitlines()
                if re.match(r"\|\s*\d{4}-\d{2}-\d{2}T", l)]
        if rows:
            cells = [c.strip() for c in rows[-1].strip("|").split("|")]
            out["sha"] = cells[1]
            for kv in cells[2].split():
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    out["switches"][k] = v
    except Exception as e:                                   # noqa: BLE001
        out["notes"].append(f"DEPLOYED.md unreadable: {type(e).__name__}: {e}")
    for key in ("CONDUCTOR", "PLANNER", "VERIFY", "VERIFY_LLM",
                "RERANK", "INDEX_NAME", "CHAT_MODEL"):
        out["switches"].setdefault(key, _shipped_default(key))
    return out


def _shipped_default(key: str) -> str:
    """The value production reads when `.env` is silent: the code default."""
    return {"CONDUCTOR": "1" if config.CONDUCTOR else "0",
            "PLANNER": "1" if config.PLANNER else "0",
            "VERIFY": "1" if config.VERIFY else "0",
            "VERIFY_LLM": "1" if config.VERIFY_LLM else "0",
            # Read at call time by retrieve.py, not a config.py switch; the
            # code default is OFF (measured: page-hit 94% -> 88% with it on).
            "RERANK": os.getenv("RERANK", "0"),
            "INDEX_NAME": os.getenv("INDEX_NAME", "default"),
            "CHAT_MODEL": config.CHAT_MODEL}.get(key, "")


def app_source_matches(deployed_sha: str) -> bool:
    """Is the app source this harness imports identical to the deployed one?

    The image is built `COPY src/ src/`, so the deployed artifact is a source
    tree and not a commit: a later commit touching only docs runs the same
    application. Comparing shas would call that a drift; comparing nothing
    would miss a real one.
    """
    if not deployed_sha:
        return False
    import subprocess
    try:
        return subprocess.run(
            ["git", "diff", "--quiet", deployed_sha, "HEAD", "--", "src/"],
            cwd=str(ROOT), capture_output=True, timeout=30).returncode == 0
    except Exception:                                # noqa: BLE001
        return False


def _parity_level(parity: dict, deployed: dict, git_sha: str = None) -> tuple:
    """(level, mismatches) — "full" only when nothing is missing.

    Two independent halves, and BOTH have to hold: every gap in the Wave-3
    table closed in this harness, and the deployed configuration matching the
    harness's pinned one field-for-field. A harness that mirrors production
    perfectly is still not measuring production if production is running
    something else.
    """
    gaps = []
    if not parity.get("production_single_id_prescope"):
        gaps.append("gap1-prescope")
    if parity.get("comparison_flag") != "decomposed":
        gaps.append("gap2-comparison-flag")
    if not parity.get("abstain_keeps_original"):
        gaps.append("gap3-abstain")
    if not parity.get("guard_verification_skipped"):
        gaps.append("gap4-guard")
    if not parity.get("answer_history_isolation"):
        gaps.append("gap5-history-isolation")
    if not (parity.get("planner") or {}).get("enabled"):
        gaps.append("gap6-planner")
    if not (parity.get("conductor") or {}).get("enabled"):
        gaps.append("gap7-conductor")
    if parity.get("verifier_mode") != "production":
        gaps.append("gap8-verifier-config")
    if not parity.get("usage_accounts_judge_and_repair"):
        gaps.append("gap9-usage-accounting")
    want = {"CONDUCTOR": "1" if (parity.get("conductor") or {}).get("enabled") else "0",
            "PLANNER": "1" if (parity.get("planner") or {}).get("enabled") else "0",
            "VERIFY": "1" if parity.get("verifier_mode") == "production" else "0",
            "VERIFY_LLM": "1" if parity.get("verifier_mode") == "production" else "0",
            "RERANK": os.getenv("RERANK", "0"),
            "INDEX_NAME": os.getenv("INDEX_NAME", "default"),
            "CHAT_MODEL": config.CHAT_MODEL}
    drift = [f"{k}: deployed {deployed['switches'].get(k)!r} != harness {v!r}"
             for k, v in want.items() if deployed["switches"].get(k) != v]
    if deployed.get("sha") and git_sha and not git_sha.startswith(deployed["sha"]):
        if app_source_matches(deployed["sha"]):
            note = (f"harness HEAD {git_sha[:7]} is past the deployed "
                    f"{deployed['sha']}, but src/ is byte-identical — the image "
                    f"is built COPY src/ src/, so the deployed application is "
                    f"this source tree")
            notes = deployed.setdefault("notes", [])
            if note not in notes:      # one dict, 66 rows: append once
                notes.append(note)
        else:
            drift.append(f"app source: deployed {deployed['sha']} differs from "
                         f"harness {git_sha[:7]} under src/")
    if gaps:
        return "partial: " + ",".join(gaps), drift
    if not deployed.get("sha"):
        return "partial: unverified-deployment", drift
    if drift:
        return "partial: deployment-drift", drift
    return "full", drift


def run_meta(args) -> dict:
    """Run-level facts stamped onto EVERY recorded row.

    On every row, not in a header line: `--compare` keys recorded files by
    case id, so a header row would need special-casing in every consumer, and
    a run whose rows disagree about which verify.py scored them is a defect
    worth being able to see.
    """
    import subprocess
    try:
        git_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                                 capture_output=True, text=True,
                                 timeout=10).stdout.strip() or None
    except Exception:
        git_sha = None
    return {
        "verify_blob_sha": _sha256_file(VERIFY_PY),
        "git_sha": git_sha,
        "artifacts": artifact_hashes(),
        "harness": {
            "model_alias": config.CHAT_MODEL,
            "temperature": getattr(args, "temperature", PIN_TEMPERATURE),
            "seed": getattr(args, "seed", PIN_SEED),
            "top_k": getattr(args, "k", config.TOP_K),
            "max_answer_tokens": config.MAX_ANSWER_TOKENS,
        },
        "ambient_env": {k: os.getenv(k) for k in
                        ("PLANNER", "CONDUCTOR", "VERIFY", "VERIFY_LLM",
                         "RERANK", "INDEX_NAME", "CHAT_MODEL")},
    }


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
    ap.add_argument("--require-metrics", type=lambda s: [x for x in s.split(",") if x],
                    help="comma-separated metric keys --compare must find on "
                         "BOTH sides; prints a no_regression JSON object and "
                         "exits 1 on a regression, a None or a missing key")
    ap.add_argument("--rescore-record", type=Path, metavar="REC",
                    help="recompute a recorded run's claims metrics offline, "
                         "zero API calls (needs --out)")
    ap.add_argument("--evidence", type=Path,
                    help="backfilled evidence for --rescore-record, when the "
                         "record's own hits carry no text")
    ap.add_argument("--out", type=Path, help="output path for --rescore-record")
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
    ap.add_argument("--verifier-mode", choices=("deterministic", "production"),
                    default="deterministic",
                    help="'deterministic' is the cheap zero-API instrument "
                         "(classify_deterministic only); 'production' runs "
                         "verify.verify_answer(use_llm=1) exactly as the app "
                         "does, judge call included")
    ap.add_argument("--conductor", dest="conductor", action="store_true",
                    default=None,
                    help="run the real per-turn LLM conductor (mode routing + "
                         "English sub-queries), as production does at "
                         "CONDUCTOR=1. Default ON for --release, OFF otherwise; "
                         "costs one extra model call per case")
    ap.add_argument("--no-conductor", dest="conductor", action="store_false",
                    help="disable the conductor even under --release")
    ap.add_argument("--production-planner", dest="production_planner",
                    action="store_true", default=None,
                    help="run the deterministic comparison planner and its "
                         "evidence matrix, as production does at PLANNER=1. "
                         "Default ON for --release, OFF otherwise")
    ap.add_argument("--no-production-planner", dest="production_planner",
                    action="store_false",
                    help="disable the planner even under --release")
    ap.add_argument("--history-mode", choices=("isolated", "prepend"),
                    default="isolated",
                    help="'isolated' is production (_answer_messages: system + "
                         "one user turn, referents via the resolved-refs note); "
                         "'prepend' is the retired harness behaviour that fed "
                         "the fixture's turns to the answer model as history")
    ap.add_argument("--comparison-flag", choices=sorted(COMPARISON_FLAGS),
                    default="decomposed",
                    help="how COMPARISON_BLOCK is gated: 'decomposed' is "
                         "production's own flag (the plan fanned out); 'proxy' "
                         "is the retired identifier-count stand-in; 'off' never "
                         "ships it")
    ap.add_argument("--no-comparison-proxy", action="store_true",
                    help="deprecated alias for --comparison-flag off")
    ap.add_argument("--raw-retrieval", action="store_true",
                    help="bypass the app's _rescope_items/_resolve_doc_tags "
                         "guards and search the raw question")
    ap.add_argument("--scope-single-id", action="store_true",
                    help="A/B only, NOT production: doc-scope questions naming "
                         "exactly one FP, to measure what tag resolution buys")
    ap.add_argument("--seed", type=int, default=PIN_SEED,
                    help="sampling seed sent with every model call; -1 sends "
                         "none (records the run as unpinned)")
    ap.add_argument("--temperature", type=float, default=PIN_TEMPERATURE,
                    help="sampling temperature sent with every model call; "
                         "negative sends none")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)
    if args.seed is not None and args.seed < 0:
        args.seed = None
    if args.temperature is not None and args.temperature < 0:
        args.temperature = None

    if args.compare:
        report = run_compare(*args.compare,
                             require_metrics=args.require_metrics)
        if report is not None and not report.get("no_regression"):
            return 1
        return 0

    if args.rescore_record:
        if not args.out:
            raise SystemExit("--rescore-record needs --out")
        return run_rescore(args.rescore_record, args.out,
                           evidence=args.evidence, force=args.force_record)

    if args.release:
        args.answers = True          # a release run IS an answer run, whole suite
    # Production parity is the DEFAULT for a release run (Wave 3, gaps 6-7):
    # a release number that skipped the planner is not a release number. Both
    # switches stay explicit, so a run can still opt out and say so.
    if args.production_planner is None:
        args.production_planner = bool(args.release)
    if args.conductor is None:
        args.conductor = bool(args.release)
    if args.verifier_mode == "production" and not args.answers:
        raise SystemExit("--verifier-mode production needs --answers/--release: "
                         "it calls the judge model")
    if args.conductor and not args.answers:
        raise SystemExit("--conductor needs --answers/--release: it is a model "
                         "call, and a retrieval-only run must stay zero-API")

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
    meta = run_meta(args)
    deployed = deployment_fingerprint()
    for r in rows:
        r.update(meta)
        parity = r.get("pipeline_parity")
        if isinstance(parity, dict):
            level, drift = _parity_level(parity, deployed, meta.get("git_sha"))
            parity["level"] = level
            parity["deployment"] = dict(deployed, drift=drift)
    if rows and isinstance(rows[0].get("pipeline_parity"), dict):
        print(f"\npipeline_parity: {rows[0]['pipeline_parity']['level']}")
    if args.record:
        print(f"\nrecorded -> "
              f"{record(rows, args.record, prefix=prefix, force=args.force_record)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
