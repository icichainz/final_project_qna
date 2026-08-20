#!/usr/bin/env python3
"""Review generated RAG QA rows with an OpenAI-compatible judge model.

This does not modify the generated dataset. It writes a sidecar JSONL where
each line is one model-reviewed row with quality scores, issue tags, and an
optional corrected answer. The input row's own evidence snippets are sent to
the reviewer, not the full source document, so the process is cheaper and
keeps the review focused on groundedness.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from gcf_qna import config


DEFAULT_INPUT = Path("data/eval/generated_rag_questions_qwen_qwen2.5-vl-7b.jsonl")
DEFAULT_OUTPUT = Path("data/eval/generated_rag_questions_qwen_qwen2.5-vl-7b.sol_review.jsonl")

SYSTEM_PROMPT = """You are SOL, a strict dataset quality reviewer for Green Climate Fund RAG QA data.

Judge each row only against the provided evidence snippets. Do not use outside knowledge.

For each row, decide:
- accept: question is answerable, answer is grounded, and the row is useful.
- revise: answer or wording should be corrected but the row is salvageable.
- reject: row is not useful, not answerable from evidence, or teaches a bad pattern.

Score 0-5:
- groundedness: answer is supported by evidence.
- answer_correctness: answer is factually right for the evidence.
- question_quality: question is natural and useful for RAG training/eval.
- citation_quality: support snippets/pages are sufficient.

Use issue tags when relevant:
fragmentary_answer, generic_question, unsupported_answer, wrong_answer,
insufficient_evidence, noisy_but_ok, adversarial_ok, duplicate_like,
bad_abstain, should_abstain, multi_doc_support_gap, too_mechanical.

Return ONLY JSON with this schema:
{"reviews":[{"id":"...","decision":"accept|revise|reject","scores":{"groundedness":0,"answer_correctness":0,"question_quality":0,"citation_quality":0},"issue_tags":[],"corrected_answer":null,"notes":"..."}]}
"""


def parse_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"model did not return a JSON object: {text[:200]}")
    return json.loads(text[start : end + 1])


def row_brief(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "scope": row.get("scope"),
        "question_type": row.get("question_type"),
        "theme": row.get("theme"),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "expected_behavior": (row.get("expect") or {}).get("behavior"),
        "docs": row.get("doc_ids") or (row.get("expect") or {}).get("docs") or [],
        "support": (row.get("sol_review") or {}).get("support") or [],
    }


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    out = set()
    for row in iter_jsonl(path):
        rid = row.get("id") or row.get("row_id")
        if rid:
            out.add(rid)
    return out


def select_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    per_doc: int,
    max_rows: Optional[int],
    seed: int,
    include_scopes: Optional[set],
    exclude_ids: set,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    by_doc: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    global_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("id") in exclude_ids:
            continue
        if include_scopes and row.get("scope") not in include_scopes:
            continue
        all_rows.append(row)
        doc_ids = row.get("doc_ids") or []
        if len(doc_ids) == 1:
            by_doc[doc_ids[0]].append(row)
        else:
            global_rows.append(row)

    selected: List[Dict[str, Any]] = []
    for doc_id in sorted(by_doc):
        bucket = by_doc[doc_id]
        grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
        for row in bucket:
            grouped[(row.get("scope"), row.get("question_type"))].append(row)
        picks: List[Dict[str, Any]] = []
        for key in sorted(grouped):
            rng.shuffle(grouped[key])
            picks.extend(grouped[key][:1])
            if len(picks) >= per_doc:
                break
        if len(picks) < per_doc:
            rng.shuffle(bucket)
            for row in bucket:
                if row not in picks:
                    picks.append(row)
                if len(picks) >= per_doc:
                    break
        selected.extend(picks[:per_doc])

    rng.shuffle(global_rows)
    selected.extend(global_rows[: max(25, per_doc * 4)])
    if max_rows is not None:
        if len(selected) < max_rows:
            selected_ids = {row.get("id") for row in selected}
            remainder = [row for row in all_rows if row.get("id") not in selected_ids]
            rng.shuffle(remainder)
            selected.extend(remainder[: max_rows - len(selected)])
        selected = selected[:max_rows]
    return selected


def batches(items: List[Dict[str, Any]], size: int) -> Iterator[List[Dict[str, Any]]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def make_client():
    if not os.getenv("OPENAI_API_KEY") and not config.OPENAI_BASE_URL:
        raise SystemExit("OPENAI_API_KEY is not set and OPENAI_BASE_URL is empty.")
    from openai import OpenAI

    return OpenAI(
        base_url=config.OPENAI_BASE_URL or None,
        timeout=120,
        default_headers={"Accept-Encoding": "identity"},
    )


def review_batch(client, model: str, rows: List[Dict[str, Any]], dry_run: bool) -> Dict[str, Any]:
    payload = {"rows": [row_brief(r) for r in rows]}
    if dry_run:
        return {
            "reviews": [
                {
                    "id": r["id"],
                    "decision": "accept",
                    "scores": {
                        "groundedness": 5,
                        "answer_correctness": 5,
                        "question_quality": 5,
                        "citation_quality": 5,
                    },
                    "issue_tags": ["dry_run"],
                    "corrected_answer": None,
                    "notes": "Dry run placeholder; no model call was made.",
                }
                for r in rows
            ]
        }
    last_error: Optional[Exception] = None
    for attempt in range(1, 4):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
            )
            return parse_json_object(resp.choices[0].message.content or "")
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2 * attempt)
    raise RuntimeError(f"review batch failed after 3 attempts") from last_error


def normalize_review(input_row: Dict[str, Any], review: Dict[str, Any], model: str) -> Dict[str, Any]:
    rid = review.get("id") or input_row.get("id")
    scores = review.get("scores") or {}
    return {
        "id": rid,
        "model": model,
        "source_row": row_brief(input_row),
        "decision": review.get("decision"),
        "scores": {
            "groundedness": scores.get("groundedness"),
            "answer_correctness": scores.get("answer_correctness"),
            "question_quality": scores.get("question_quality"),
            "citation_quality": scores.get("citation_quality"),
        },
        "issue_tags": review.get("issue_tags") or [],
        "corrected_answer": review.get("corrected_answer"),
        "notes": review.get("notes") or "",
        "reviewed_at_unix": int(time.time()),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--model", default=os.getenv("SOL_REVIEW_MODEL", config.CHAT_MODEL))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--per-doc", type=int, default=8, help="sample this many single-doc rows per document")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--scopes", default="", help="comma-separated scope filter, e.g. page,document,thematic")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write placeholder reviews without API calls")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    include_scopes = {s.strip() for s in args.scopes.split(",") if s.strip()} or None
    done = existing_ids(args.output) if args.resume else set()
    rows = select_rows(
        iter_jsonl(args.input),
        per_doc=args.per_doc,
        max_rows=args.max_rows,
        seed=args.seed,
        include_scopes=include_scopes,
        exclude_ids=done,
    )
    if not rows:
        print("no rows selected")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    client = None if args.dry_run else make_client()
    mode = "a" if args.resume else "w"
    written = 0
    with args.output.open(mode, encoding="utf-8") as fh:
        for batch in batches(rows, args.batch_size):
            result = review_batch(client, args.model, batch, args.dry_run)
            reviews_by_id = {r.get("id"): r for r in result.get("reviews", [])}
            for row in batch:
                review = reviews_by_id.get(row["id"], {"id": row["id"], "decision": "reject", "notes": "missing review"})
                fh.write(json.dumps(normalize_review(row, review, args.model), ensure_ascii=False, separators=(",", ":")) + "\n")
                written += 1
            print(f"reviewed {written}/{len(rows)} rows", flush=True)
    print(f"wrote {written} reviews to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
