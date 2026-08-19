#!/usr/bin/env python3
"""Answer-level evaluation against scripts/answer_gold.jsonl (step 0 of the
RAG-correctness plan: freeze a baseline before answer behavior changes).

scripts/eval_retrieval.py answers "did we find the document?".  This answers
"did we say the right thing about it?" — expected documents AND expected
evidence pages, plus, with --answers, the generated answer's behavior
(answer / conflict / abstention), required and forbidden strings, language,
and citation validity.

Three modes
-----------
  --retrieval-only   (default) deterministic, no API calls. Runs the same
                     retrieval path the app runs — conductor SKIPPED, the
                     question used verbatim — and scores document recall and
                     evidence-page hit rate.  Multi-turn cases are skipped
                     (they need history to be resolvable) and counted apart.
  --answers          additionally calls the chat model per case, assembling
                     the prompt exactly as chainlit_app does (registry note,
                     year note, board-range note, weak-signal note, per-turn
                     assemble()), and scores the answer.
  --compare A B      diff two recorded runs, per case and per class.

The retrieval floor (100% document recall on the 30 retrieval gold questions)
is re-checked in-process with --gate, so one index load covers both.

Examples
--------
  python scripts/eval_answers.py --retrieval-only --gate
  python scripts/eval_answers.py --answers --sample 12 --record baseline-sample
  python scripts/eval_answers.py --answers --ids conf-fp274-gcf,abs-fp999
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
from gcf_qna.rag import registry                              # noqa: E402

DEFAULT_CASES = ROOT / "scripts" / "answer_gold.jsonl"
GOLD_SET = ROOT / "scripts" / "gold_set.jsonl"
EVAL_DIR = ROOT / "data" / "eval"

CLASS_ORDER = ["identifier", "compact-id", "board-code", "discovery",
               "comparison", "conflict", "french", "noisy", "abstain",
               "aggregate", "followup"]


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
    """chainlit_app.main() with the conductor removed and no Chainlit I/O."""

    def __init__(self, top_k: int = None, comparison_proxy: bool = True,
                 raw_retrieval: bool = False, scope_single_id: bool = False):
        from gcf_qna.app import chainlit_app as app
        self.app = app
        self.top_k = top_k or config.TOP_K
        self.comparison_proxy = comparison_proxy
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

    def run(self, question: str) -> dict:
        app = self.app
        guard = self.fp_guard(question)
        if guard is not None:
            return {"guard": True, "guard_answer": guard, "hits": [],
                    "system": None, "user": None, "weak": False, "plan": [],
                    "notes": {"registry": None, "year": None, "board": None}}

        items = self.plan(question)
        sq = items[0]
        hits, conf = self.retriever.search_with_confidence(
            sq["q"], self.top_k, sq.get("doc"))
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
        weak = conf < config.MIN_DENSE_SCORE
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

        system = assemble(year=bool(year_note), registry=bool(reg_note),
                          comparison=self.comparison_proxy and multi_identifier(question),
                          lang=app._detect_lang(question))
        return {
            "guard": False, "guard_answer": None, "hits": hits,
            "confidence": conf, "weak": weak, "plan": items,
            "system": system,
            "user": f"Context excerpts:\n{context}\n\nQuestion: {question}",
            "notes": {"registry": reg_note, "year": year_note, "board": board_note},
        }


def ask_model(client, system: str, turns: list, user: str) -> str:
    messages = [{"role": "system", "content": system}]
    messages += [{"role": t["role"], "content": t["content"]} for t in turns]
    messages.append({"role": "user", "content": user})
    last = None
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=config.CHAT_MODEL,
                max_completion_tokens=config.MAX_ANSWER_TOKENS,
                messages=messages,
            )
            return resp.choices[0].message.content or ""
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
                    scope_single_id=args.scope_single_id)
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
        out = pipe.run(case["question"])
        rec = {
            "id": case["id"], "class": case["class"], "lang": case["lang"],
            "question": case["question"], "turns": len(case["turns"]),
            "mode": "answers" if args.answers else "retrieval-only",
            "guard": out["guard"], "weak_signal": out.get("weak"),
            "plan": out.get("plan") or [],
            "retrieval": score_retrieval(case, out["hits"]),
            "expect": case["expect"],
        }
        rec["retrieval_score"] = retrieval_score(rec["retrieval"])
        if args.answers:
            answer = out["guard_answer"]
            if answer is None:
                answer = ask_model(client, out["system"], case["turns"], out["user"])
            rec["answer"] = answer
            rec["checks"] = score_answer(case, answer, out["hits"])
            rec["score"] = rec["checks"]["score"]
            rec["model"] = config.CHAT_MODEL
        else:
            rec["score"] = rec["retrieval_score"]
        rows.append(rec)
        if args.verbose:
            r = rec["retrieval"]
            print(f"[{i}/{len(cases)}] {case['id']:26} rank={r['rank']} "
                  f"pages={r['pages_hit']}/{r['pages_expected']} "
                  + (f"score={rec['score']:.2f}" if args.answers else ""))
        elif args.answers:
            print(f"[{i}/{len(cases)}] {case['id']}", flush=True)

    dt = time.perf_counter() - t0
    print(f"\n=== {'answer' if args.answers else 'retrieval-only'} baseline — "
          f"{len(rows)} cases in {dt:.0f}s "
          f"(k={pipe.top_k}, hybrid={getattr(pipe.retriever, 'hybrid_enabled', False)}"
          + (f", model={config.CHAT_MODEL}" if args.answers else "") + ") ===")
    print_retrieval_table(rows, skipped)
    if args.answers:
        print()
        print_answer_table(rows)
        _print_failures(rows)
    return rows


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


def record(rows: list, label: str) -> Path:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / f"answers_baseline_{label}.jsonl"
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
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), type=Path,
                    help="diff two recorded runs")
    ap.add_argument("--gate", action="store_true",
                    help="also re-check the gold_set retrieval floor in-process")
    ap.add_argument("--sample", type=int, help="evaluate N class-stratified cases")
    ap.add_argument("--ids", help="comma-separated case ids")
    ap.add_argument("--classes", help="comma-separated class filter")
    ap.add_argument("--record", metavar="LABEL",
                    help="write data/eval/answers_baseline_<LABEL>.jsonl")
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
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args(argv)

    if args.compare:
        run_compare(*args.compare)
        return 0

    cases = select(load_cases(args.cases), ids=args.ids, sample=args.sample,
                   classes=args.classes)
    if not cases:
        raise SystemExit("no cases selected")
    rows = run_eval(args, cases)
    if args.record:
        print(f"\nrecorded -> {record(rows, args.record)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
