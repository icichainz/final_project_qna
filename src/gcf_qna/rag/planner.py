"""Deterministic comparison planning: document-by-field evidence matrices.

Step 4 of docs/rag-correctness-next-steps.html. When a question names the
documents AND the fields it wants, nothing about the plan needs a model: the
identifiers resolve through the registry, the fields come from a keyword map,
and the cells come from registry v2 facts (page- and section-cited) before any
retrieval happens. The LLM conductor keeps what it is actually good at —
semantic decomposition, pronouns, follow-ups.

The point of the matrix is coverage. A fan-out of k chunks per sub-query can
quietly starve the last document of a three-way comparison, and prose written
over a starved context reads exactly like prose written over a complete one.
Here every (document, field) pair becomes a cell, and every cell is rendered:

  stated             a registry v2 candidate with page and template section
  contradictory      ... and the document ALSO prints a figure that disagrees;
                     both are rendered, neither is silently preferred
  retrieved          no registry fact; the line carries the top passage of a
                     retrieval scoped to that one document
  missing            the document states it nowhere retrieval could see it
  missing-document   the identifier resolves to no document in the corpus

There is deliberately no 'calculated' status: this module never computes
across cells. `Matrix.comparable(field)` says whether a ranking is even
defensible ('EUR vs USD - no conversion rule'), and render() prints that
verdict; arithmetic across currencies is refused rather than approximated.

Everything degrades. No registry_v2.json: every cell takes the retrieval
route. No retriever: a registry-only matrix whose uncovered cells say MISSING.
No registry.json at all: identifiers stay unresolved but are NOT called
missing — the closed-world claim needs the closed world to be loaded — and the
literal id is used as the retrieval scope instead.

`fields_for` is the question->field detection of this module, and it has one
caller outside it: `registry._served_bits`, which serves the SAME fields off
the SAME v2 candidates on a single-identifier turn, where `detect()` returns
None and there is no matrix to serve them. One detector for both arities is
the point — a field word that opens a column here opens a segment there, and
neither arity can quietly know a field the other does not.

    plan = detect(question)                  # None => not our job
    matrix = build_matrix(plan, retriever)   # retriever optional
    block = render(matrix)                   # context for the answer model
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field as _dc_field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gcf_qna.rag import registry

# The board-code pattern keeps ONE definition, in the registry — the same
# reason FP_RE is exported there. A private copy would drift the day the
# corpus grows a code shape the resolver already understands.
_BOARD_CODE_RE = registry._BOARD_CODE_RE

# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------

MONEY_FIELDS = frozenset({"total_financing", "gcf_funding_requested", "co_financing"})
# fields whose cells hold a number: only these can ever be ranked, and only
# when currency and unit class agree
NUMERIC_FIELDS = MONEY_FIELDS | frozenset({
    "beneficiaries_direct", "beneficiaries_indirect", "mitigation_outcome",
    "adaptation_outcome", "implementation_period", "lifespan"})
# one v2 candidate per instrument: the cell is the list, not the first entry.
# `financial_instruments` is the same A.10 table read down its amount column —
# one candidate per instrument, each carrying the instrument in its own section
# ('A.10 Grant', 'A.10 Loan') — so the cell is that list too, and the section
# is what says which amount belongs to which instrument. It is deliberately
# NOT in NUMERIC_FIELDS: a per-instrument breakdown is not one figure, and
# ranking two documents by 'their financial instruments' would be ranking two
# lists whose rows are not even the same instruments.
LIST_FIELDS = frozenset({"instruments", "financial_instruments"})

# Question order for the columns of the matrix; also the order render() prints.
# The three additions of Phase 1 sit beside the field they belong with — the
# two entity/authority fields after the accredited entity, the per-instrument
# amounts after the instruments they price. Insertion, never reordering: two
# questions asking for the same fields must still produce byte-identical
# matrices, and `fields_for` reads its output order off this tuple.
FIELD_ORDER: Tuple[str, ...] = (
    "title", "countries", "accredited_entity", "executing_entity",
    "national_designated_authority", "project_size",
    "total_financing", "gcf_funding_requested", "co_financing", "instruments",
    "financial_instruments",
    "implementation_period", "lifespan", "ess_category",
    "mitigation_outcome", "adaptation_outcome",
    "beneficiaries_direct", "beneficiaries_indirect",
)

# "Compare FP254 and FP248" with no field word still has an obvious answer
# shape: who, what, how much.
DEFAULT_FIELDS: Tuple[str, ...] = (
    "title", "accredited_entity", "gcf_funding_requested", "total_financing")

# EN + FR keyword -> field. Matched against the question with accents stripped
# ('financement', 'entité accréditée', 'période de mise en œuvre'), most
# specific rule first: 'total financing' must not be read as the GCF request.
_FIELD_RULES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    (r"co[\s\-]?financ|cofinanc|contre[\s\-]?partie", ("co_financing",)),
    (r"total (?:project |programme )?financ|financement total|total (?:project )?cost"
     r"|cout total|budget total|total budget|overall (?:cost|financing)",
     ("total_financing",)),
    (r"gcf financ|gcf funding|funding requested|requested gcf|gcf request"
     r"|financement (?:du |de la |accorde par le |octroye par le )?(?:gcf|fvc)"
     r"|montant demande|demande de financement|subvention (?:du )?(?:gcf|fvc)",
     ("gcf_funding_requested",)),
    (r"accredited entit|entite accredit|entite de mise en oeuvre|\bae\b",
     ("accredited_entity",)),
    # The entity that RUNS the project is not the entity accredited to the
    # Fund, and the corpus prints both (193 documents state one, section A.20
    # or A.1.6). Kept clear of the accredited-entity rule's own words: nothing
    # here matches 'accredited entity' or 'entite de mise en oeuvre'.
    (r"executing entit|executing agenc|entites? d'?execution"
     r"|agence d'?execution|entites? chargees? de l'?execution",
     ("executing_entity",)),
    # The country's own authority for the Fund (A.1.4), stated by 101
    # documents and, before this, reachable by nothing.
    (r"national designated authorit|designated authorit|\bnda\b"
     r"|autorite nationale designee|autorite designee",
     ("national_designated_authority",)),
    (r"countr(?:y|ies)|\bpays\b|\bnations?\b|geograph", ("countries",)),
    # A.10 read down its AMOUNT column: 'financial instrument' is the
    # template's own label for the table that prices each instrument, so the
    # phrase asks for the amounts as well as the names. The bare-noun rule
    # below stays names-only on purpose - 'a grant' in a title is not a
    # request for a breakdown, and it is a title word in this corpus.
    (r"financial instrument|instruments? financiers?"
     r"|(?:amount|montant)s? (?:per|by|par) instrument"
     r"|instrument breakdown|repartition par instrument",
     ("instruments", "financial_instruments")),
    (r"instrument|\bgrants?\b|\bloans?\b|\bequity\b|\bpret\b|\bdon\b|\bgarantie\b",
     ("instruments",)),
    (r"beneficiar|beneficiaire|people reached|personnes touchees",
     ("beneficiaries_direct", "beneficiaries_indirect")),
    (r"implementation period|periode de mise en oeuvre|periode d'?execution"
     r"|duree de mise en oeuvre|duree d'?execution|how long.{0,24}(?:implement|last)",
     ("implementation_period",)),
    (r"lifespan|duree de vie|duree totale", ("lifespan",)),
    (r"ess categor|e&s categor|esg categor|environmental and social"
     r"|categorie (?:environnementale|e&s|ess|de risque)|risk categor|safeguard",
     ("ess_category",)),
    (r"project size|programme size|size categor|taille (?:du |de la )?(?:projet|programme)"
     r"|\bsize\b|\btaille\b", ("project_size",)),
    (r"\btitles?\b|\btitres?\b|\bintitule|\bnamed?\b|\bnames\b|\bappele", ("title",)),
    (r"mitigation|attenuation|\bghg\b|tco2|emission", ("mitigation_outcome",)),
    (r"adaptation", ("adaptation_outcome",)),
)
_FIELD_RES = tuple((re.compile(p, re.I), fs) for p, fs in _FIELD_RULES)

# A money word with no field of its own ('financement', 'cost', 'budget',
# 'how big') asks the two questions the template answers: what was requested,
# and what the whole thing costs. Only consulted when no specific money rule
# fired, so 'total financing' does not also drag in the GCF request.
_GENERIC_MONEY_RE = re.compile(
    r"financ|funding|\bfunds?\b|\bcosts?\b|\bcouts?\b|\bbudget\b|\bmontants?\b"
    r"|\bamounts?\b|\bprix\b|\bsize\b|\btaille\b|\bbig(?:ger|gest)?\b"
    r"|\bexpensive\b|\bchere?\b|\bgrand\b|\blarg(?:e|er|est)\b", re.I)

# What a doc-scoped retrieval asks when registry v2 has no candidate. English:
# the index is built over the extracted corpus and the app's conductor already
# translates queries into English.
_FIELD_QUERIES: Dict[str, str] = {
    "title": "project or programme title",
    "countries": "country or countries of operation",
    "accredited_entity": "accredited entity",
    "executing_entity": "executing entity implementation arrangements",
    "national_designated_authority": "national designated authority NDA",
    "project_size": "project size category micro small medium large",
    "total_financing": "total financing GCF plus co-financing amount",
    "gcf_funding_requested": "total GCF funding requested amount",
    "co_financing": "total co-financing amount",
    "instruments": "financial instruments requested grant loan equity guarantee",
    "financial_instruments": "amount requested per financial instrument grant loan equity",
    "implementation_period": "implementation period years months",
    "lifespan": "total project lifespan years",
    "ess_category": "environmental and social safeguards ESS category",
    "mitigation_outcome": "expected mitigation outcome tCO2eq emissions reduced",
    "adaptation_outcome": "expected adaptation outcome beneficiaries",
    "beneficiaries_direct": "direct beneficiaries number of people",
    "beneficiaries_indirect": "indirect beneficiaries number of people",
}

STATUSES = ("stated", "contradictory", "retrieved", "missing", "missing-document")

_UNIT_MULT = {"thousand": 1e3, "million": 1e6, "billion": 1e9}


def _deaccent(s: str) -> str:
    """'entité accréditée' -> 'entite accreditee'; 'œ' -> 'oe'."""
    s = s.replace("œ", "oe").replace("Æ", "AE").replace("æ", "ae")
    return "".join(c for c in unicodedata.normalize("NFKD", s)
                   if not unicodedata.combining(c))


def fields_for(question: str) -> Tuple[List[str], bool]:
    """(fields the question asks for, used_default) — deterministic keyword map."""
    q = _deaccent(question)
    got: List[str] = []
    for rx, fs in _FIELD_RES:
        if rx.search(q):
            got.extend(f for f in fs if f not in got)
    if not (set(got) & MONEY_FIELDS) and _GENERIC_MONEY_RE.search(q):
        got.extend(f for f in ("gcf_funding_requested", "total_financing") if f not in got)
    # FIELD_ORDER, not match order: two questions asking for the same fields in
    # different words must produce byte-identical matrices
    if not got:
        return [f for f in FIELD_ORDER if f in DEFAULT_FIELDS], True
    return [f for f in FIELD_ORDER if f in got], False


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------

@dataclass
class DocRef:
    """One identifier the question named, resolved through the registry."""
    label: str                       # 'FP254' / 'GCF/B.42/02/Add.16'
    kind: str                        # 'fp' | 'board'
    doc_id: Optional[str] = None     # corpus stem, None when unresolved
    fp: Optional[int] = None
    title: Optional[str] = None
    missing: bool = False            # the registry is loaded and has no such doc
    scope: Optional[str] = None      # doc_filter for retrieval
    aliases: List[str] = _dc_field(default_factory=list)

    @property
    def name(self) -> str:
        return self.label + (f" (= {', '.join(self.aliases)})" if self.aliases else "")


def _board_label(b: str, item: Optional[str], add: str) -> str:
    return f"GCF/B.{b}/" + (f"{item}/" if item else "") + f"Add.{add}"


def _identifiers(question: str) -> List[DocRef]:
    """Every FP id and board code in the question, in the order they are written.

    Resolution order per plan step 4: the registry decides existence BEFORE any
    retrieval runs. Duplicates collapse — 'FP274 and GCF/B.42/02/Add.16' name
    one document and must produce one row, not two.
    """
    marks: List[Tuple[int, str, tuple]] = []
    for m in registry.FP_RE.finditer(question):
        marks.append((m.start(), "fp", (int(m.group(1)),)))
    for m in _BOARD_CODE_RE.finditer(question):
        marks.append((m.start(), "board", (m.group(1), m.group(2), m.group(3))))
    marks.sort(key=lambda t: t[0])

    have_registry = bool(registry.load())
    # existence check for FP ids goes through the resolver the app already uses
    _, missing_fps = registry.resolve_fps(question) if have_registry else ([], [])
    missing_fps = set(missing_fps)

    out: List[DocRef] = []
    by_key: Dict[tuple, DocRef] = {}
    by_doc: Dict[str, DocRef] = {}
    for _, kind, key in marks:
        if kind == "fp":
            n = key[0]
            label = f"FP{n}"
            row = registry.by_fp(n) if have_registry else None
            ref = DocRef(label=label, kind="fp", fp=n,
                         doc_id=row["doc_id"] if row else None,
                         title=(row or {}).get("title"),
                         missing=have_registry and (n in missing_fps or row is None),
                         # with no registry loaded the bare id is still a usable
                         # retrieval scope: retrieve.py matches 'fp86' -> 'fp086'
                         scope=(row["doc_id"] if row else (None if have_registry
                                                           else f"fp{n:03d}")))
        else:
            b_, item_, add_ = key
            label = _board_label(b_, item_, add_)
            row = (registry.resolve_board_code(int(b_), int(add_),
                                               int(item_) if item_ else None)
                   if have_registry else None)
            ref = DocRef(label=label, kind="board",
                         doc_id=row["doc_id"] if row else None,
                         fp=(row or {}).get("fp"), title=(row or {}).get("title"),
                         missing=have_registry and row is None,
                         scope=(row["doc_id"] if row else
                                (None if have_registry or not item_ else
                                 f"b{int(b_)}-{item_}-add{add_}")))
        dedupe = (kind, key)
        if dedupe in by_key:
            continue
        by_key[dedupe] = ref
        if ref.doc_id and ref.doc_id in by_doc:      # same document, two names
            by_doc[ref.doc_id].aliases.append(ref.label)
            continue
        if ref.doc_id:
            by_doc[ref.doc_id] = ref
        out.append(ref)
    return out


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """One cell to fill: a document and a field (field None = the whole row is
    unfillable because the identifier resolves to no document)."""
    doc: DocRef
    field: Optional[str]

    @property
    def missing_document(self) -> bool:
        return self.doc.missing


@dataclass
class Plan:
    question: str
    docs: List[DocRef]
    fields: List[str]
    jobs: List[Job]
    default_fields: bool = False

    @property
    def trigger(self) -> str:
        return (f"{len(self.docs)} identifiers x {len(self.fields)} fields"
                + (" (default field set: no field word in the question)"
                   if self.default_fields else ""))


def detect(question: str) -> Optional[Plan]:
    """A comparison plan when the question names >= 2 documents, else None.

    Deterministic by construction: identifiers come from the registry's own
    patterns, fields from a keyword map. One identifier, or none, is left to
    the LLM conductor — a single-document question has no matrix, and a purely
    semantic one has no columns.
    """
    if not question or not question.strip():
        return None
    docs = _identifiers(question)
    if len(docs) < 2:
        return None
    fields, default = fields_for(question)
    jobs: List[Job] = []
    for d in docs:
        if d.missing:
            # ONE row that says the document does not exist, rather than N
            # identical cells. The row is never dropped: a comparison that
            # silently forgets an identifier is the failure this step exists for.
            jobs.append(Job(doc=d, field=None))
            continue
        jobs.extend(Job(doc=d, field=f) for f in fields)
    return Plan(question=question, docs=docs, fields=fields, jobs=jobs,
                default_fields=default)


# ---------------------------------------------------------------------------
# matrix
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    doc: DocRef
    field: Optional[str]
    status: str
    raw: Optional[str] = None
    value: Optional[float] = None
    currency: Optional[str] = None
    unit: Optional[str] = None
    page: Optional[int] = None
    section: Optional[str] = None
    conflicts: List[dict] = _dc_field(default_factory=list)
    extras: List[dict] = _dc_field(default_factory=list)   # list fields
    text: Optional[str] = None                             # retrieved passage
    score: Optional[float] = None
    note: Optional[str] = None

    @property
    def label(self) -> str:
        return f"{self.doc.label} | {self.field or '*'}"


@dataclass
class Comparability:
    ok: bool
    reason: str

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class Matrix:
    plan: Plan
    cells: List[Cell]

    def cell(self, label: str, field: str) -> Optional[Cell]:
        return next((c for c in self.cells
                     if c.doc.label.lower() == label.lower() and c.field == field), None)

    def row(self, label: str) -> List[Cell]:
        return [c for c in self.cells if c.doc.label.lower() == label.lower()]

    def column(self, field: str) -> List[Cell]:
        return [c for c in self.cells if c.field == field]

    @property
    def counts(self) -> Dict[str, int]:
        return {s: sum(1 for c in self.cells if c.status == s) for s in STATUSES}

    def comparable(self, field: str) -> Comparability:
        """Whether the cells of `field` may be ranked, summed or subtracted.

        False with a printable reason whenever a comparison would have to
        invent something: a conversion rate, a missing figure, or a choice
        between two figures the document itself disagrees about.
        """
        cells = self.column(field)
        if not cells:
            return Comparability(False, f"{field}: no cell in this matrix")
        if field not in NUMERIC_FIELDS:
            return Comparability(
                False, f"{field} is a text field - report what each document states; "
                       "there is no ranking to make")
        reasons: List[str] = []
        gaps = [c for c in cells if c.value is None]
        priced = [c for c in cells if c.value is not None]
        if not priced:
            reasons.append("no document states a parsed figure for this field "
                           "- there is nothing to rank")
        elif gaps:
            reasons.append("; ".join(
                f"{c.doc.label} states no comparable figure ({c.status})" for c in gaps)
                + " - ranking the others would drop it from the answer")
        elif len(priced) < 2:
            reasons.append(f"only {priced[0].doc.label} states a figure "
                           "- nothing to compare it with")
        currencies = {c.currency for c in priced}
        if field in MONEY_FIELDS or any(c.currency for c in priced):
            if None in currencies and len(currencies) > 1:
                named = sorted(x for x in currencies if x)
                reasons.append(
                    f"currency not printed for "
                    + ", ".join(c.doc.label for c in priced if not c.currency)
                    + f" - do not assume it is {'/'.join(named)}")
            elif len([x for x in currencies if x]) > 1:
                pairs = ", ".join(f"{c.doc.label} {c.currency}" for c in priced if c.currency)
                reasons.append(
                    " vs ".join(sorted(x for x in currencies if x))
                    + f" - no conversion rule in this corpus ({pairs}); "
                      "give each amount in its own currency, never rank or sum them")
        units = {("duration" if c.unit in ("years", "months") else "amount")
                 for c in priced}
        if len(units) > 1:
            reasons.append("durations and amounts are mixed in this column")
        if field in ("implementation_period", "lifespan"):
            printed = {c.unit for c in priced if c.unit}
            if len(printed) > 1:
                reasons.append(" vs ".join(sorted(printed))
                               + " - the documents print different time units, "
                                 "and no conversion is applied here")
        clashes = [c for c in cells if c.status == "contradictory"]
        if clashes:
            reasons.append("; ".join(
                f"{c.doc.label} prints conflicting figures (p.{c.page} vs "
                + ", ".join(f"p.{x.get('page')}" for x in c.conflicts) + ")"
                for c in clashes) + " - resolve the contradiction before ranking")
        if reasons:
            return Comparability(False, "; ".join(reasons))
        return Comparability(
            True, "same currency and unit, every document states a figure - "
                  "comparable as printed")


def _same_doc(doc_id: str, scope: str) -> bool:
    """Guard against retrieve.py's unscoped fallback.

    Retriever.search() deliberately degrades to a global search when a
    doc_filter matches nothing; a cell must never quote another document's
    page, so hits are re-checked against the scope here.
    """
    a, b = (doc_id or "").lower(), (scope or "").lower()
    return bool(a and b) and (a == b or b in a or a in b)


def _best_supporting(cands: Sequence[dict]) -> Optional[dict]:
    """Earliest non-conflicting candidate — the template block is front matter.

    Used when v2 has candidates but elected none canonical (an LLM-fallback
    document, or a field printed outside any template section).
    """
    ok = [c for c in cands if c.get("status") != "conflicting"]
    return min(ok, key=lambda c: (c.get("page") or 10 ** 6)) if ok else None


def _registry_cell(doc: DocRef, field: str) -> Optional[Cell]:
    key = doc.doc_id or doc.fp
    if key is None:
        return None
    cands = registry.facts(key).get(field) or []
    if not cands:
        return None
    if field in LIST_FIELDS:
        seen, items = set(), []
        for c in cands:
            k = (c.get("raw") or "").strip().lower()
            if k and k not in seen:
                seen.add(k)
                items.append(c)
        head = items[0]
        return Cell(doc=doc, field=field, status="stated", raw=head.get("raw"),
                    page=head.get("page"), section=head.get("section"),
                    extras=items[1:])
    canon = registry.canonical(key, field)
    conflicts = registry.conflicts(key).get(field) or []
    best = canon or _best_supporting(cands)
    if best is None:                       # only conflicting candidates: no anchor
        best = cands[0]
    return Cell(doc=doc, field=field,
                status="contradictory" if conflicts else "stated",
                raw=best.get("raw"), value=best.get("value"),
                currency=best.get("currency"), unit=best.get("unit"),
                page=best.get("page"), section=best.get("section"),
                conflicts=list(conflicts),
                note=None if canon else "no template-section candidate; "
                                        "best available source shown")


def _retrieved_cell(doc: DocRef, field: str, retriever: Any, top_k: int) -> Cell:
    scope = doc.scope or doc.doc_id
    if retriever is None or not scope:
        return Cell(doc=doc, field=field, status="missing",
                    note="no registry fact and no retriever in this run")
    query = _FIELD_QUERIES.get(field, field.replace("_", " "))
    try:
        hits = retriever.search(query, top_k, scope) or []
    except Exception as e:                                   # noqa: BLE001
        return Cell(doc=doc, field=field, status="missing",
                    note=f"retrieval failed ({type(e).__name__})")
    hits = [h for h in hits if _same_doc(getattr(h, "doc_id", ""), scope)]
    if not hits:
        return Cell(doc=doc, field=field, status="missing",
                    note="not in the fact registry; a search inside this document "
                         "returned no passage")
    h = hits[0]
    return Cell(doc=doc, field=field, status="retrieved",
                text=(h.text or "").strip(), page=getattr(h, "page", None),
                score=getattr(h, "score", None), section=None)


def build_matrix(plan: Plan, retriever: Any = None, top_k: int = 3) -> Matrix:
    """Fill every job: registry v2 first, doc-scoped retrieval second, MISSING last.

    `retriever` only needs the public Retriever.search(query, top_k, doc_filter)
    -> [Hit(text, doc_id, score, page)] contract; None gives a registry-only
    matrix (the mode tests and offline evaluation use).
    """
    cells: List[Cell] = []
    for job in plan.jobs:
        if job.field is None or job.doc.missing:
            cells.append(Cell(doc=job.doc, field=job.field, status="missing-document",
                              note="no such document in the 273-document registry"))
            continue
        cell = _registry_cell(job.doc, job.field)
        if cell is None:
            cell = _retrieved_cell(job.doc, job.field, retriever, top_k)
        cells.append(cell)
    return Matrix(plan=plan, cells=cells)


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def _fmt_number(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


def _fmt_amount(cell: Cell) -> Optional[str]:
    """'EUR 38.17 million' / 'USD 40,511,264' / '60 months' — never converted."""
    if cell.value is None:
        return None
    cur = f"{cell.currency} " if cell.currency else ""
    if cell.unit in _UNIT_MULT:
        scaled = cell.value / _UNIT_MULT[cell.unit]
        num = f"{scaled:,.4f}".rstrip("0").rstrip(".")
        return f"{cur}{num} {cell.unit}"
    if cell.unit in ("years", "months"):
        return f"{_fmt_number(cell.value)} {cell.unit}"
    return f"{cur}{_fmt_number(cell.value)}"


def _cite(page: Optional[int], section: Optional[str]) -> str:
    bits = [f"p.{page}" if page else "page not recorded"]
    if section:
        # 'rule:A.8' = the page printed no section id; A.8 is the rule that fired
        bits.append(("rule " + section[5:]) if section.startswith("rule:") else section)
    return "(" + ", ".join(bits) + ")"


def _adds_nothing(raw: str, shown: str) -> bool:
    """Does the source text carry nothing the formatted value already shows?

    Character multiset containment, so '40,511,264 USD' vs 'USD 40,511,264' and
    '38.17 | million eur' vs 'EUR 38.17 million' are one print, while the
    truncated 'USD 258 million (USD' keeps its extra characters and is echoed.
    """
    norm = lambda s: Counter(re.sub(r"[^0-9a-z]", "", (s or "").lower()))  # noqa: E731
    return not (norm(raw) - norm(shown))


def _one_line(s: str, limit: int) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s if len(s) <= limit else s[:limit].rstrip() + " ..."


def _value_text(cell: Cell, snippet: int) -> str:
    if cell.status == "missing-document":
        return "MISSING DOCUMENT - the identifier matches no document in the corpus"
    if cell.status == "missing":
        return "MISSING - " + (cell.note or "not stated anywhere this run could see")
    if cell.status == "retrieved":
        return ("no registry fact; retrieved from this document only: \""
                + _one_line(cell.text or "", snippet) + "\"")
    if cell.field in LIST_FIELDS:
        names = [cell.raw] + [x.get("raw") for x in cell.extras]
        return ", ".join(n for n in names if n)
    amount = _fmt_amount(cell)
    if amount is None:
        return (f"as printed: \"{_one_line(cell.raw or '', snippet)}\""
                + ("" if cell.field not in NUMERIC_FIELDS else
                   " (no parsed value - the page's figure and its unit word "
                   "contradict each other)"))
    if cell.field in MONEY_FIELDS and not cell.currency:
        # a money cell whose page never named a currency: the gap is a fact
        amount += " (currency not printed)"
    if cell.raw and not _adds_nothing(cell.raw, amount):
        return f"{amount} [as printed: \"{_one_line(cell.raw, 90)}\"]"
    return amount


HEADER = (
    "EVIDENCE MATRIX (deterministic plan: identifiers and fields resolved before "
    "retrieval; registry facts first, document-scoped search second)\n"
    "Row format: <identifier> | <field> | <value exactly as the document prints it> "
    "<(page, section)> | <status>\n"
    "stated = printed in this document; contradictory = this document ALSO prints a "
    "figure that disagrees (both are given below the row); retrieved = no registry "
    "fact, the text comes from a search inside that one document; missing = the "
    "document does not state it; missing-document = the identifier resolves to no "
    "document in the corpus.\n"
    "'rule X' in a citation means the page printed no section id and X is the "
    "template rule that matched - cite the page, not the section.\n"
    "Address EVERY row: report a missing row as not stated, never fill it from "
    "another document, and never convert or rank across currencies."
)


def render(matrix: Matrix, snippet: int = 320) -> str:
    """The context block: one line per cell, in question order, then field order.

    This is the 'evidence matrix before prose generation' of plan step 4 —
    every cell is printed, including the empty ones, so the answer model cannot
    leave a document silently unanswered.
    """
    plan = matrix.plan
    lines = [HEADER, ""]
    lines.append(f"Question plan: {len(plan.docs)} document(s) x {len(plan.fields)} "
                 f"field(s) = {len(matrix.cells)} cell(s). Fields: "
                 + ", ".join(plan.fields)
                 + (" [default set - the question named no field]"
                    if plan.default_fields else ""))
    for doc in plan.docs:
        lines.append("")
        head = f"{doc.name} -> " + (doc.doc_id or "UNRESOLVED")
        if doc.title:
            lines.append(head + f" | \"{_one_line(doc.title, 90)}\"")
        else:
            lines.append(head)
        for cell in matrix.row(doc.label):
            body = _value_text(cell, snippet)
            cite = (_cite(cell.page, cell.section)
                    if cell.status in ("stated", "contradictory", "retrieved") else "")
            lines.append(f"{doc.label} | {cell.field or '*'} | {body}"
                         + (f" {cite}" if cite else "") + f" | {cell.status}")
            if cell.note and cell.status == "stated":
                lines.append(f"    note: {cell.note}")
            for x in cell.conflicts:
                alt = Cell(doc=doc, field=cell.field, status="stated",
                           raw=x.get("raw"), value=x.get("value"),
                           currency=x.get("currency"), unit=x.get("unit"),
                           page=x.get("page"), section=x.get("section"))
                lines.append(f"    CONFLICT in the same document: {_value_text(alt, snippet)} "
                             f"{_cite(alt.page, alt.section)} - report BOTH figures with "
                             f"their pages; do not choose one silently")
            for x in cell.extras[:6] if cell.field in LIST_FIELDS else []:
                if x.get("page") != cell.page:
                    lines.append(f"    also: \"{x.get('raw')}\" "
                                 + _cite(x.get("page"), x.get("section")))
    numeric = [f for f in plan.fields if f in NUMERIC_FIELDS]
    if numeric:
        lines.append("")
        lines.append("COMPARABILITY (deterministic; no conversion rule exists in this corpus)")
        for f in numeric:
            v = matrix.comparable(f)
            lines.append(f"{f}: {'COMPARABLE' if v.ok else 'NOT COMPARABLE'} - {v.reason}")
    counts = matrix.counts
    lines.append("")
    lines.append("Cell coverage: " + ", ".join(f"{k} {counts[k]}" for k in STATUSES))
    return "\n".join(lines)


def plan_and_render(question: str, retriever: Any = None,
                    top_k: int = 3) -> Optional[str]:
    """detect + build + render in one call; None when the planner does not fire."""
    plan = detect(question)
    if plan is None:
        return None
    return render(build_matrix(plan, retriever, top_k))
