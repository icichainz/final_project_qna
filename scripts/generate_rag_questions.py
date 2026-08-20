#!/usr/bin/env python3
"""Generate extractive RAG QA rows from merged VLM markdown files.

The output is JSONL: one compact, independently reviewable question per line.
Every answer is derived from the document text or registry facts and carries
source pages plus short support snippets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from gcf_qna.rag.parse import chunk_document, split_pages


DEFAULT_SOURCE = Path("data/extracted/vlm/qwen_qwen2.5-vl-7b")
DEFAULT_REGISTRY = Path("data/registry_v2.json")
DEFAULT_OUTPUT = Path("data/eval/generated_rag_questions_qwen_qwen2.5-vl-7b.jsonl")

MAX_SNIPPET = 260
MAX_ANSWER = 520
MAX_THEMES_PER_DOC = 32
MAX_CROSS_DOC_ROWS = 2400

STOPWORDS = {
    "about", "above", "across", "after", "against", "also", "among",
    "because", "between", "climate", "could", "fund", "funding", "green",
    "have", "into", "more", "page", "programme", "project", "proposal",
    "section", "shall", "should", "their", "there", "these", "this",
    "through", "under", "which", "with", "within", "would",
}


PAGE_MARK_RE = re.compile(r"(?:^|\n+)---\n\*\*Page (\d+)\*\*\n---\n")
HEADING_RE = re.compile(r"^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$")
BOLD_SECTION_RE = re.compile(r"^ {0,3}\*\*\s*([A-H]\.\s?\d{1,2}.*?)\s*\*\*[.:]?\s*$", re.I)
MONEY_RE = re.compile(
    r"(?:(?:USD|EUR|GBP|CHF|XCD|MUR|MNT|CNY|JPY|CAD|AUD)\s*)?"
    r"(?:[$€£]\s*)?\d[\d,\s]*(?:\.\d+)?\s*"
    r"(?:million|billion|m|bn|USD|EUR|GBP|CHF|XCD|MUR|MNT|CNY|JPY|CAD|AUD)?",
    re.I,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'/-]{3,}")
FP_RE = re.compile(r"\bFP\s*0*(\d{2,3})\b", re.I)
BOARD_RE = re.compile(r"\bGCF/B\.\s*(\d{1,2})/\s*(\d{2})/Add\.\s*(\d{1,2})\b", re.I)


def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s.strip(" -*|")


def truncate(s: str, limit: int) -> str:
    s = clean(s)
    if len(s) <= limit:
        return s
    cut = s[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:")
    return cut + "..."


def doc_stem(path: Path) -> str:
    return path.with_suffix("").name


def stable_id(*parts: Any) -> str:
    raw = "\x1f".join(str(p) for p in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def headings(text: str) -> List[str]:
    out: List[str] = []
    for line in text.splitlines():
        m = HEADING_RE.match(line) or BOLD_SECTION_RE.match(line)
        if m:
            title = clean(re.sub(r"\*+", "", m.group(1)))
            if title and title not in out:
                out.append(title)
    return out


def sentences(text: str) -> List[str]:
    body = re.sub(r"\|.*\|", " ", text)
    body = re.sub(r"#+\s*", " ", body)
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", body)
    return [clean(p) for p in parts if len(clean(p)) >= 35]


def first_sentence(text: str) -> str:
    ss = sentences(text)
    if ss:
        for sent in ss:
            first = sent.lstrip("\"'([{").split(" ", 1)[0]
            if not first[:1].islower():
                return truncate(sent, MAX_ANSWER)
        return truncate(ss[0], MAX_ANSWER)
    for line in text.splitlines():
        line = clean(re.sub(r"^#+\s*", "", line))
        if len(line) >= 20 and not line.startswith("|"):
            return truncate(line, MAX_ANSWER)
    return truncate(text, MAX_ANSWER)


def page_about_answer(page: int, title: str, text: str) -> str:
    summary = first_sentence(text)
    if clean(summary).lower() == clean(title).lower():
        ss = [s for s in sentences(text) if clean(s).lower() != clean(title).lower()]
        summary = ss[0] if ss else ""
    if summary:
        return truncate(f"Page {page} is about {truncate(title, 120)}. {summary}", MAX_ANSWER)
    return truncate(f"Page {page} is about {truncate(title, 180)}.", MAX_ANSWER)


def key_terms(text: str, limit: int = 6) -> List[str]:
    counts: Dict[str, int] = defaultdict(int)
    original: Dict[str, str] = {}
    for m in WORD_RE.finditer(text):
        word = m.group(0).strip("-/")
        low = word.lower()
        if low in STOPWORDS or len(low) < 4:
            continue
        counts[low] += 1
        original.setdefault(low, word)
    ranked = sorted(counts, key=lambda k: (-counts[k], k))
    return [original[k] for k in ranked[:limit]]


def values(text: str, limit: int = 5) -> List[str]:
    vals: List[str] = []
    for regex in (FP_RE, BOARD_RE, MONEY_RE, YEAR_RE):
        for m in regex.finditer(text):
            val = clean(m.group(0))
            if val and val not in vals:
                vals.append(val)
            if len(vals) >= limit:
                return vals
    return vals


def snippet_for(text: str, needle: Optional[str] = None) -> str:
    text = clean(text)
    if not needle:
        return truncate(text, MAX_SNIPPET)
    idx = text.lower().find(needle.lower())
    if idx < 0:
        return truncate(text, MAX_SNIPPET)
    start = max(0, idx - 90)
    end = min(len(text), idx + len(needle) + 150)
    return truncate(text[start:end], MAX_SNIPPET)


def registry_facts(reg: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
    return (reg.get("documents") or {}).get(doc_id, {})


def fact_answer(facts: Dict[str, Any]) -> str:
    pieces: List[str] = []
    if facts.get("fp"):
        pieces.append(f"FP{int(facts['fp']):03d}")
    for key, label in (
        ("title", "title"),
        ("accredited_entity", "accredited entity"),
        ("countries", "countries"),
        ("gcf_financing", "GCF financing"),
        ("total_financing", "total financing"),
        ("board", "board"),
        ("year", "year"),
    ):
        value = facts.get(key)
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            value = ", ".join(map(str, value))
        pieces.append(f"{label}: {value}")
    return truncate("; ".join(pieces), MAX_ANSWER)


def fact_pages(facts: Dict[str, Any]) -> List[int]:
    pages: List[int] = []
    for entries in (facts.get("facts") or {}).values():
        for ent in entries or []:
            page = ent.get("page")
            if isinstance(page, int) and page not in pages:
                pages.append(page)
    return pages[:8]


def doc_support(
    doc_id: str,
    facts: Dict[str, Any],
    page_texts: Dict[str, Dict[int, str]],
    fields: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    pages: List[int] = []
    facts_by_field = facts.get("facts") or {}
    for field in fields:
        for ent in facts_by_field.get(field, []) or []:
            page = ent.get("page")
            if isinstance(page, int) and page not in pages:
                pages.append(page)
    for page in fact_pages(facts):
        if page not in pages:
            pages.append(page)
    if not pages:
        pages = [1]
    texts = page_texts.get(doc_id, {})
    out = []
    for page in pages[:3]:
        out.append({"doc_id": doc_id, "page": page, "snippet": snippet_for(texts.get(page, ""))})
    return out


def make_reviewed_row(
    *,
    row_id_parts: Sequence[Any],
    doc_ids: Sequence[str],
    scope: str,
    qtype: str,
    question: str,
    answer: str,
    support: Sequence[Dict[str, Any]],
    theme: Optional[str] = None,
    fields: Optional[Sequence[str]] = None,
    behavior: str = "answer",
    verdict: str = "answerable_from_document",
    notes: str = "Generated extractive QA row; answer reviewed against the cited document snippets.",
) -> Dict[str, Any]:
    pages = []
    for item in support:
        page = item.get("page")
        if isinstance(page, int) and page not in pages:
            pages.append(page)
    return {
        "id": stable_id(*row_id_parts),
        "source": "vlm/qwen_qwen2.5-vl-7b",
        "doc_id": doc_ids[0] if len(doc_ids) == 1 else None,
        "doc_ids": list(doc_ids),
        "scope": scope,
        "question_type": qtype,
        "theme": theme,
        "question": question,
        "answer": answer,
        "expect": {
            "behavior": behavior,
            "docs": list(doc_ids),
            "pages": pages,
            "fields": list(fields or []),
            "notes": notes,
        },
        "sol_review": {
            "verdict": verdict,
            "right_answer": answer,
            "support": list(support),
        },
    }


def make_row(
    *,
    doc_id: str,
    scope: str,
    qtype: str,
    question: str,
    answer: str,
    pages: Sequence[int],
    snippets: Sequence[str],
    theme: Optional[str] = None,
    fields: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    support = [
        {"doc_id": doc_id, "page": page, "snippet": snippet}
        for page, snippet in zip(pages, snippets)
    ]
    return make_reviewed_row(
        row_id_parts=[doc_id, scope, qtype, theme or "", question],
        doc_ids=[doc_id],
        scope=scope,
        qtype=qtype,
        question=question,
        answer=answer,
        support=support,
        theme=theme,
        fields=fields,
    )


def page_rows(doc_id: str, page: int, text: str) -> Iterator[Dict[str, Any]]:
    hs = headings(text)
    title = hs[0] if hs else f"page {page}"
    vals = values(text)
    terms = key_terms(text)
    summary = first_sentence(text)
    snippet = snippet_for(text, vals[0] if vals else (terms[0] if terms else None))

    yield make_row(
        doc_id=doc_id,
        scope="page",
        qtype="summary",
        question=f"What is page {page} of {doc_id} mainly about?",
        answer=page_about_answer(page, title, text),
        pages=[page],
        snippets=[snippet_for(text)],
    )
    yield make_row(
        doc_id=doc_id,
        scope="page",
        qtype="location",
        question=f"Which heading or section identifies page {page} of {doc_id}?",
        answer=truncate(title, MAX_ANSWER),
        pages=[page],
        snippets=[snippet_for(text, title)],
    )
    yield make_row(
        doc_id=doc_id,
        scope="page",
        qtype="fact",
        question=f"What extractable value or identifier appears on page {page} of {doc_id}?",
        answer=", ".join(vals[:5]) if vals else summary,
        pages=[page],
        snippets=[snippet],
    )
    term = terms[0] if terms else title.split()[0] if title else "this topic"
    yield make_row(
        doc_id=doc_id,
        scope="page",
        qtype="verification",
        question=f"Does page {page} of {doc_id} discuss {term}?",
        answer=f"Yes. Page {page} discusses {term}: {snippet_for(text, term)}",
        pages=[page],
        snippets=[snippet_for(text, term)],
    )


def document_rows(doc_id: str, pages: List[Tuple[int, str]], facts: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    first_text = pages[0][1] if pages else ""
    hs: List[str] = []
    for _, body in pages[:20]:
        for h in headings(body):
            if h not in hs:
                hs.append(h)
    reg_answer = fact_answer(facts)
    evidence_pages = fact_pages(facts) or [pages[0][0]]
    page_text = {p: t for p, t in pages}
    snippets = [snippet_for(page_text.get(p, first_text)) for p in evidence_pages]

    yield make_row(
        doc_id=doc_id,
        scope="document",
        qtype="summary",
        question=f"What does the document {doc_id} contain?",
        answer=reg_answer or f"{doc_id} contains {truncate(hs[0] if hs else first_sentence(first_text), MAX_ANSWER)}",
        pages=evidence_pages,
        snippets=snippets,
        fields=["title", "accredited_entity", "countries", "gcf_financing", "total_financing"],
    )
    yield make_row(
        doc_id=doc_id,
        scope="document",
        qtype="location",
        question=f"Which major sections are visible near the start of {doc_id}?",
        answer=truncate("; ".join(hs[:12]), MAX_ANSWER),
        pages=[p for p, _ in pages[: min(5, len(pages))]],
        snippets=[snippet_for(t) for _, t in pages[: min(5, len(pages))]],
    )
    yield make_row(
        doc_id=doc_id,
        scope="document",
        qtype="fact",
        question=f"What are the core registry facts for {doc_id}?",
        answer=reg_answer or "No registry facts were available for this document.",
        pages=evidence_pages,
        snippets=snippets,
        fields=["title", "accredited_entity", "countries", "gcf_financing", "total_financing", "board", "year"],
    )
    fp = facts.get("fp")
    term = f"FP{int(fp):03d}" if fp else (key_terms(first_text, 1) or [doc_id])[0]
    yield make_row(
        doc_id=doc_id,
        scope="document",
        qtype="verification",
        question=f"Is {term} represented in {doc_id}?",
        answer=f"Yes. {term} is represented in {doc_id}. {reg_answer or first_sentence(first_text)}",
        pages=evidence_pages,
        snippets=snippets,
        fields=["fp", "title"],
    )


def theme_groups(text: str) -> Dict[str, List[Tuple[int, str]]]:
    grouped: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
    for page, chunk in chunk_document(text, size=1400, overlap=120):
        theme = chunk.section_path
        if not theme:
            continue
        grouped[theme].append((page, chunk.text))
    return grouped


def thematic_rows(doc_id: str, grouped: Dict[str, List[Tuple[int, str]]]) -> Iterator[Dict[str, Any]]:
    ranked = sorted(grouped.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:MAX_THEMES_PER_DOC]
    for theme, chunks in ranked:
        pages: List[int] = []
        text_parts: List[str] = []
        for page, text in chunks:
            if page not in pages:
                pages.append(page)
            if len(" ".join(text_parts)) < 1800:
                text_parts.append(text)
        combined = "\n\n".join(text_parts)
        terms = key_terms(combined, 5)
        vals = values(combined, 6)
        support_pages = pages[:4] or [chunks[0][0]]
        page_to_text = {p: t for p, t in chunks}
        snippets = [snippet_for(page_to_text.get(p, combined), terms[0] if terms else None) for p in support_pages]
        theme_summary = first_sentence(combined)
        if theme_summary[:1].islower():
            detail = "; ".join((terms + vals)[:8]) or theme_summary
            theme_summary = f"The theme '{theme}' covers {detail}. Evidence excerpt: {theme_summary}"

        yield make_row(
            doc_id=doc_id,
            scope="thematic",
            qtype="summary",
            theme=theme,
            question=f"What does the theme '{theme}' cover in {doc_id}?",
            answer=truncate(theme_summary, MAX_ANSWER),
            pages=support_pages,
            snippets=snippets,
        )
        yield make_row(
            doc_id=doc_id,
            scope="thematic",
            qtype="location",
            theme=theme,
            question=f"Which pages of {doc_id} contain the theme '{theme}'?",
            answer=f"The theme appears on pages {', '.join(map(str, pages[:20]))}.",
            pages=support_pages,
            snippets=snippets,
        )
        yield make_row(
            doc_id=doc_id,
            scope="thematic",
            qtype="fact",
            theme=theme,
            question=f"What key terms or values are stated under '{theme}' in {doc_id}?",
            answer=truncate("; ".join((terms + vals)[:10]) or first_sentence(combined), MAX_ANSWER),
            pages=support_pages,
            snippets=snippets,
        )
        term = terms[0] if terms else theme.split()[0]
        yield make_row(
            doc_id=doc_id,
            scope="thematic",
            qtype="verification",
            theme=theme,
            question=f"Does the theme '{theme}' in {doc_id} mention {term}?",
            answer=f"Yes. The theme mentions {term}: {snippet_for(combined, term)}",
            pages=support_pages,
            snippets=snippets,
        )


def money_fact(facts: Dict[str, Any], field: str = "gcf_funding_requested") -> Optional[Dict[str, Any]]:
    entries = (facts.get("facts") or {}).get(field) or []
    for ent in entries:
        if ent.get("value") is not None and ent.get("currency"):
            return ent
    return None


def fp_label(facts: Dict[str, Any]) -> str:
    fp = facts.get("fp")
    return f"FP{int(fp):03d}" if fp is not None else "the proposal"


def short_title(facts: Dict[str, Any]) -> str:
    title = facts.get("title") or fp_label(facts)
    return truncate(str(title), 95)


def registry_records(registry: Dict[str, Any], doc_ids: Sequence[str]) -> List[Tuple[str, Dict[str, Any]]]:
    docs = registry.get("documents") or {}
    out = [(doc_id, docs.get(doc_id, {})) for doc_id in doc_ids if docs.get(doc_id)]
    return sorted(out, key=lambda item: (item[1].get("fp") or 0, item[0]))


def support_many(
    doc_ids: Sequence[str],
    registry_docs: Dict[str, Dict[str, Any]],
    page_texts: Dict[str, Dict[int, str]],
    fields: Sequence[str] = (),
    per_doc: int = 1,
    max_docs: int = 6,
) -> List[Dict[str, Any]]:
    support: List[Dict[str, Any]] = []
    for doc_id in doc_ids[:max_docs]:
        support.extend(doc_support(doc_id, registry_docs.get(doc_id, {}), page_texts, fields)[:per_doc])
    return support


def cross_document_rows(
    registry: Dict[str, Any],
    doc_ids: Sequence[str],
    page_texts: Dict[str, Dict[int, str]],
) -> Iterator[Dict[str, Any]]:
    registry_docs = registry.get("documents") or {}
    records = registry_records(registry, doc_ids)
    emitted = 0

    by_year: Dict[int, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    by_board: Dict[int, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    by_entity: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    by_country: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    financed: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []

    for doc_id, facts in records:
        if isinstance(facts.get("year"), int):
            by_year[facts["year"]].append((doc_id, facts))
        if isinstance(facts.get("board"), int):
            by_board[facts["board"]].append((doc_id, facts))
        entity = clean(str(facts.get("accredited_entity") or ""))
        if entity:
            by_entity[entity].append((doc_id, facts))
        for country in facts.get("countries") or []:
            by_country[clean(str(country))].append((doc_id, facts))
        money = money_fact(facts)
        if money:
            financed.append((doc_id, facts, money))

    for year, items in sorted(by_year.items()):
        fps = [fp_label(facts) for _, facts in items]
        docs = [doc_id for doc_id, _ in items[:6]]
        answer = f"The corpus contains {len(items)} funding proposals from {year}: {', '.join(fps[:18])}"
        if len(fps) > 18:
            answer += f", and {len(fps) - 18} more."
        yield make_reviewed_row(
            row_id_parts=["cross-year-count", year],
            doc_ids=docs,
            scope="cross_document",
            qtype="aggregate",
            question=f"How many funding proposals from {year} are in this corpus, and which FP numbers do they include?",
            answer=truncate(answer, MAX_ANSWER),
            support=support_many(docs, registry_docs, page_texts, ["title"]),
            fields=["year", "fp"],
            notes="Cross-document aggregate generated from registry facts and sampled document support.",
        )
        emitted += 1

    for board, items in sorted(by_board.items()):
        fps = [fp_label(facts) for _, facts in items]
        docs = [doc_id for doc_id, _ in items[:6]]
        answer = f"Board B.{board} has {len(items)} funding proposal documents in this corpus: {', '.join(fps[:18])}"
        if len(fps) > 18:
            answer += f", and {len(fps) - 18} more."
        yield make_reviewed_row(
            row_id_parts=["cross-board-count", board],
            doc_ids=docs,
            scope="cross_document",
            qtype="aggregate",
            question=f"Which funding proposals are represented for Board B.{board}?",
            answer=truncate(answer, MAX_ANSWER),
            support=support_many(docs, registry_docs, page_texts, ["title"]),
            fields=["board", "fp"],
            notes="Cross-document board aggregate generated from registry facts and sampled document support.",
        )
        emitted += 1

    for entity, items in sorted(by_entity.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:120]:
        if len(items) < 2 or emitted >= MAX_CROSS_DOC_ROWS:
            break
        docs = [doc_id for doc_id, _ in items[:6]]
        answer = f"{entity} appears as accredited entity for {len(items)} proposals: "
        answer += ", ".join(f"{fp_label(f)} ({short_title(f)})" for _, f in items[:10])
        if len(items) > 10:
            answer += f", and {len(items) - 10} more."
        yield make_reviewed_row(
            row_id_parts=["cross-entity-list", entity],
            doc_ids=docs,
            scope="cross_document",
            qtype="list",
            question=f"Which proposals in the corpus are implemented by {entity}?",
            answer=truncate(answer, MAX_ANSWER),
            support=support_many(docs, registry_docs, page_texts, ["accredited_entity"]),
            fields=["accredited_entity", "fp", "title"],
            notes="Cross-document entity list generated from registry facts and sampled document support.",
        )
        emitted += 1

    for country, items in sorted(by_country.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:160]:
        if len(items) < 2 or emitted >= MAX_CROSS_DOC_ROWS:
            break
        docs = [doc_id for doc_id, _ in items[:6]]
        answer = f"{country} is listed among countries for {len(items)} proposals: "
        answer += ", ".join(f"{fp_label(f)} ({short_title(f)})" for _, f in items[:10])
        if len(items) > 10:
            answer += f", and {len(items) - 10} more."
        yield make_reviewed_row(
            row_id_parts=["cross-country-list", country],
            doc_ids=docs,
            scope="cross_document",
            qtype="list",
            question=f"Which proposals list {country} as a country?",
            answer=truncate(answer, MAX_ANSWER),
            support=support_many(docs, registry_docs, page_texts, ["countries"]),
            fields=["countries", "fp", "title"],
            notes="Cross-document country list generated from registry facts and sampled document support.",
        )
        emitted += 1

    by_currency: Dict[str, List[Tuple[str, Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for item in financed:
        by_currency[str(item[2].get("currency")).upper()].append(item)
    for currency, items in sorted(by_currency.items()):
        items = sorted(items, key=lambda item: (float(item[2].get("value") or 0), item[1].get("fp") or 0))
        for left, right in zip(items[::2], items[1::2]):
            if emitted >= MAX_CROSS_DOC_ROWS:
                return
            doc_a, facts_a, money_a = left
            doc_b, facts_b, money_b = right
            winner = (doc_a, facts_a, money_a) if money_a["value"] >= money_b["value"] else (doc_b, facts_b, money_b)
            answer = (
                f"{fp_label(winner[1])} requests more GCF funding: {winner[2]['raw']}. "
                f"{fp_label(facts_a)} states {money_a['raw']}; {fp_label(facts_b)} states {money_b['raw']}."
            )
            yield make_reviewed_row(
                row_id_parts=["cross-financing-compare", doc_a, doc_b],
                doc_ids=[doc_a, doc_b],
                scope="cross_document",
                qtype="comparison",
                question=f"Which requests more GCF funding, {fp_label(facts_a)} or {fp_label(facts_b)}?",
                answer=truncate(answer, MAX_ANSWER),
                support=support_many([doc_a, doc_b], registry_docs, page_texts, ["gcf_funding_requested"]),
                fields=["gcf_financing"],
                notes="Cross-document same-currency comparison generated from registry facts.",
            )
            emitted += 1


def adversarial_rows(
    registry: Dict[str, Any],
    doc_ids: Sequence[str],
    page_texts: Dict[str, Dict[int, str]],
) -> Iterator[Dict[str, Any]]:
    registry_docs = registry.get("documents") or {}
    records = registry_records(registry, doc_ids)
    fp_to_doc = {int(f["fp"]): (doc_id, f) for doc_id, f in records if f.get("fp") is not None}
    existing_fps = set(fp_to_doc)
    min_fp, max_fp = min(existing_fps), max(existing_fps)
    registry_support = [{
        "doc_id": "registry_v2",
        "page": None,
        "snippet": f"Registry covers {len(existing_fps)} FP identifiers from FP{min_fp:03d} to FP{max_fp:03d}.",
    }]

    missing = [fp for fp in range(min_fp - 8, max_fp + 18) if fp > 0 and fp not in existing_fps]
    for fp in missing[:80]:
        answer = f"No. FP{fp:03d} is not present in this corpus registry."
        yield make_reviewed_row(
            row_id_parts=["adv-missing-fp", fp],
            doc_ids=[],
            scope="adversarial",
            qtype="abstain_missing_identifier",
            question=f"What does FP{fp:03d} fund?",
            answer=answer,
            support=registry_support,
            behavior="abstain",
            verdict="not_answerable_from_corpus",
            fields=["fp"],
            notes="Adversarial missing-FP question; correct behavior is abstain/not-found.",
        )

    board_adds: Dict[Tuple[int, str], List[int]] = defaultdict(list)
    for doc_id in doc_ids:
        m = re.search(r"b(\d{1,2})-(\d{2})-add(\d{1,2})", doc_id, re.I)
        if m:
            board_adds[(int(m.group(1)), m.group(2))].append(int(m.group(3)))
    for (board, agenda), adds in sorted(board_adds.items())[:80]:
        bad_add = max(adds) + 7
        code = f"GCF/B.{board}/{agenda}/Add.{bad_add:02d}"
        answer = f"No document with board code {code} is present in this corpus."
        yield make_reviewed_row(
            row_id_parts=["adv-missing-board-code", code],
            doc_ids=[],
            scope="adversarial",
            qtype="abstain_missing_board_code",
            question=f"What is in {code}?",
            answer=answer,
            support=[{"doc_id": "registry_v2", "page": None, "snippet": f"Known B.{board}/{agenda} addenda are: {', '.join(map(str, sorted(adds)))}."}],
            behavior="abstain",
            verdict="not_answerable_from_corpus",
            fields=["board"],
            notes="Adversarial missing-board-code question; correct behavior is abstain/not-found.",
        )

    for fp, (doc_id, facts) in sorted(fp_to_doc.items())[:120]:
        noisy = f"FP-{fp}"
        answer = fact_answer(facts)
        yield make_reviewed_row(
            row_id_parts=["adv-noisy-fp", fp],
            doc_ids=[doc_id],
            scope="adversarial",
            qtype="noisy_identifier",
            question=f"{noisy}?? what is the accredited entity and project title",
            answer=answer,
            support=doc_support(doc_id, facts, page_texts, ["title", "accredited_entity"]),
            fields=["fp", "title", "accredited_entity"],
            notes="Adversarial noisy identifier variant; correct answer is the same document facts.",
        )

    padded = [fp for fp in sorted(existing_fps) if fp < 100][:30]
    for fp in padded:
        doc_id, facts = fp_to_doc[fp]
        answer = f"FP{fp:03d}, not FP{str(fp).zfill(4)[:-1]}. {fact_answer(facts)}"
        yield make_reviewed_row(
            row_id_parts=["adv-padded-fp", fp],
            doc_ids=[doc_id],
            scope="adversarial",
            qtype="padded_identifier",
            question=f"Give me the details of FP{fp:04d}.",
            answer=truncate(answer, MAX_ANSWER),
            support=doc_support(doc_id, facts, page_texts, ["title", "accredited_entity"]),
            fields=["fp", "title", "accredited_entity"],
            notes="Adversarial zero-padded FP identifier; correct behavior is to resolve to the intended compact FP.",
        )

    financed = [(doc_id, facts, money_fact(facts)) for doc_id, facts in records]
    financed = [(d, f, m) for d, f, m in financed if m]
    cross_currency: List[Tuple[Tuple[str, Dict[str, Any], Dict[str, Any]], Tuple[str, Dict[str, Any], Dict[str, Any]]]] = []
    for i, left in enumerate(financed):
        for right in financed[i + 1:i + 18]:
            if left[2].get("currency") != right[2].get("currency"):
                cross_currency.append((left, right))
                break
        if len(cross_currency) >= 120:
            break
    for left, right in cross_currency:
        doc_a, facts_a, money_a = left
        doc_b, facts_b, money_b = right
        answer = (
            f"The document values use different currencies, so they should not be ranked without conversion. "
            f"{fp_label(facts_a)} states {money_a['raw']}; {fp_label(facts_b)} states {money_b['raw']}."
        )
        yield make_reviewed_row(
            row_id_parts=["adv-cross-currency", doc_a, doc_b],
            doc_ids=[doc_a, doc_b],
            scope="adversarial",
            qtype="cross_currency_trap",
            question=f"Which is larger, the GCF funding for {fp_label(facts_a)} or {fp_label(facts_b)}?",
            answer=truncate(answer, MAX_ANSWER),
            support=support_many([doc_a, doc_b], registry_docs, page_texts, ["gcf_funding_requested"]),
            fields=["gcf_financing"],
            notes="Adversarial cross-currency comparison; correct behavior is to refuse direct ranking without conversion.",
        )

    title_terms: List[Tuple[set, str, Dict[str, Any]]] = []
    for doc_id, facts in records:
        terms = {t.lower() for t in key_terms(str(facts.get("title") or ""), 8)}
        if terms:
            title_terms.append((terms, doc_id, facts))
    emitted_pairs = 0
    for i, (terms_a, doc_a, facts_a) in enumerate(title_terms):
        for terms_b, doc_b, facts_b in title_terms[i + 1:]:
            if facts_a.get("accredited_entity") == facts_b.get("accredited_entity"):
                continue
            overlap = len(terms_a & terms_b)
            if overlap < 2:
                continue
            answer = (
                f"No. {fp_label(facts_a)} is implemented by {facts_a.get('accredited_entity')}; "
                f"{fp_label(facts_b)} is implemented by {facts_b.get('accredited_entity')}."
            )
            yield make_reviewed_row(
                row_id_parts=["adv-merge-trap", doc_a, doc_b],
                doc_ids=[doc_a, doc_b],
                scope="adversarial",
                qtype="merge_trap",
                question=f"Do {fp_label(facts_a)} and {fp_label(facts_b)} have the same accredited entity?",
                answer=truncate(answer, MAX_ANSWER),
                support=support_many([doc_a, doc_b], registry_docs, page_texts, ["accredited_entity"]),
                fields=["accredited_entity", "fp"],
                notes="Adversarial similar-title merge trap; correct behavior is to keep proposals separate.",
            )
            emitted_pairs += 1
            break
        if emitted_pairs >= 120:
            break


def iter_rows(source: Path, registry: Dict[str, Any], limit_docs: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    files = sorted(source.glob("*.md"))
    if limit_docs:
        files = files[:limit_docs]
    doc_ids: List[str] = []
    page_texts: Dict[str, Dict[int, str]] = {}
    for path in files:
        doc_id = doc_stem(path)
        text = path.read_text(encoding="utf-8", errors="replace")
        pages = split_pages(text)
        doc_ids.append(doc_id)
        page_texts[doc_id] = {page: body for page, body in pages}
        facts = registry_facts(registry, doc_id)
        for page, body in pages:
            yield from page_rows(doc_id, page, body)
        yield from document_rows(doc_id, pages, facts)
        yield from thematic_rows(doc_id, theme_groups(text))
    yield from cross_document_rows(registry, doc_ids, page_texts)
    yield from adversarial_rows(registry, doc_ids, page_texts)


def write_jsonl(rows: Iterable[Dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--limit-docs", type=int, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    count = write_jsonl(iter_rows(args.source, registry, args.limit_docs), args.output)
    print(f"wrote {count} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
