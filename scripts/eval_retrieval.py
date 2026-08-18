#!/usr/bin/env python3
"""Retrieval evaluation against the gold set (scripts/gold_set.jsonl).

Reports recall@5 / recall@10 (any returned chunk from an expected doc) and
MRR, split by question class. Run with HYBRID=0/1 to A/B retrieval modes.

  python scripts/eval_retrieval.py [--index default]
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gcf_qna import config
from gcf_qna.rag import Embedder, Retriever, load_index


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="default")
    ap.add_argument("--gold", type=Path, default=Path(__file__).parent / "gold_set.jsonl")
    a = ap.parse_args()

    gold = [json.loads(l) for l in a.gold.read_text(encoding="utf-8").splitlines() if l]
    index, chunks, cfg = load_index(config.INDEX_DIR / a.index)
    r = Retriever(index, chunks, Embedder(cfg["embedding_model"]))
    r.search("warmup", top_k=1)

    stats = defaultdict(lambda: {"n": 0, "r5": 0, "r10": 0, "mrr": 0.0})
    t0 = time.perf_counter()
    for g in gold:
        hits = r.search(g["q"], top_k=10)
        docs = [h.doc_id for h in hits]
        rank = next((i + 1 for i, d in enumerate(docs) if d in g["expected"]), None)
        s = stats[g["cls"]]
        s["n"] += 1
        s["r5"] += bool(rank and rank <= 5)
        s["r10"] += bool(rank)
        s["mrr"] += (1.0 / rank) if rank else 0.0
    dt = (time.perf_counter() - t0) / len(gold) * 1000

    mode = "hybrid" if getattr(r, "hybrid_enabled", False) else "dense-only"
    print(f"\n{mode} | {len(gold)} questions | {dt:.0f} ms/query")
    print(f"{'class':12} {'n':>3} {'recall@5':>9} {'recall@10':>10} {'MRR':>6}")
    tot = {"n": 0, "r5": 0, "r10": 0, "mrr": 0.0}
    for cls, s in sorted(stats.items()):
        print(f"{cls:12} {s['n']:>3} {s['r5']/s['n']:>9.0%} {s['r10']/s['n']:>10.0%} {s['mrr']/s['n']:>6.2f}")
        for k in tot: tot[k] += s[k]
    print(f"{'TOTAL':12} {tot['n']:>3} {tot['r5']/tot['n']:>9.0%} {tot['r10']/tot['n']:>10.0%} {tot['mrr']/tot['n']:>6.2f}")


if __name__ == "__main__":
    main()
