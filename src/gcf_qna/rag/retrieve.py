"""Query the index.

Filters FAISS's -1 padding: when the index holds fewer vectors than top_k,
FAISS pads ids with -1, and Python's metadata[-1] silently returns the LAST
chunk — the old codebase fed that unrelated passage to the LLM as context.

Pipeline for one query (plan step 3):

    hybrid candidates (3x top_k, CANDIDATE_POOL_MULT)
        -> dedup by (doc, page, text hash)      BEFORE any per-doc quota
        -> rerank query/passage pairs           cross-encoder, RERANK=1
        -> per-doc quota, round robin           identifier routing only
        -> one slot per page before two         PAGE_DIVERSITY, on
        -> expand into neighbouring same-section chunks, SECTION_EXPAND=1
        -> top_k

Documents are chosen before pages, and by a different text: once a document
is settled — by a doc_filter, or by the identifier router's lexical head —
the caller's `original` message gets a second dense vote on WHICH OF ITS
CHUNKS come first (_probes / _scoped_probes). Nothing there can change the
document set, so it moves page precision without touching document recall.

Every stage ranks on retrieval_text — whatever the build embedded — while
every Hit carries source_text, so a retrieval-side prefix can never reach the
answer model or ground.py. Chunks written before the schema existed have
neither field and fall back to chunk["text"] throughout.

Beside that pipeline sits one query the pipeline cannot ask. `probe_pages`
fetches named pages (or, where the build stored them, named sections) of ONE
named document, and ranks nothing globally: the caller already knows where
the evidence is — the registry's conflict candidates each carry a page — and
the measured failure is not that the page is unindexed but that it loses the
similarity contest to a cover page phrased in the question's own words. It is
a supplement, never a fallback: it selects nothing, degrades to nothing, and
fires only when a caller asks.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from gcf_qna import config
from gcf_qna.rag.embed import Embedder
from gcf_qna.rag.lexical import LexicalIndex, _doc_tokens, fp_variants, tokenize
from gcf_qna.rag.parse import retrieval_text, section_id, source_text

# --- tunables -------------------------------------------------------------
# Read at call time from the environment rather than at import: config.py is
# shared with two other work streams, and these are step-3 experiment knobs,
# not settled configuration.
#
# Defaults follow the measurement on the 62-case answer fixture (v2 index,
# retrieval-only; baseline r@5 88% / page-hit 81%):
#
#   flags                       r@5   r@10  cover@10  page-hit
#   (default)                   96%   96%     96%       94%
#   RERANK=1                    96%   98%     98%       88%
#   SECTION_EXPAND=1            96%   96%     96%       72%
#   RERANK=1 SECTION_EXPAND=1   96%   96%     96%       66%
#   PAGE_DIVERSITY=0            96%   96%     96%       94%
#
# So both extra stages are built, tested and OFF: expansion spends top-k slots
# that were holding required evidence pages, and the ms-marco cross-encoder
# trades evidence pages for a little document coverage. PAGE_DIVERSITY was
# neutral on the fixture and is on because it costs nothing and widens what
# the answer model sees (it stops one page taking three of ten context slots).
#   CANDIDATE_POOL_MULT  candidate pool as a multiple of top_k (3). 4 measured
#     WORSE on the gold set (MRR 1.00 -> 0.98): the pool also sets the depth of
#     the two fusion lists, and a passage that reaches only one list at rank
#     31-40 collects RRF weight it has not earned.
#   RERANK / RERANK_MODEL / RERANK_DEVICE / RERANK_MAX_PAIRS / RERANK_MIN_POOL
#   SECTION_EXPAND / EXPAND_KEEP / EXPAND_TOKENS
#   PAGE_DIVERSITY
#   ORIGINAL_PROBE  give the caller's `original` text a second dense vote in
#     the doc-scoped stages (on). Ignored unless a caller passes `original`,
#     so every caller that does not is unaffected either way. See _probes.
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_on(name: str, default: str = "1") -> bool:
    return os.getenv(name, default) == "1"


def rrf(ranked_lists: List[List[int]], k: int = 60,
        weights: Optional[List[float]] = None) -> Dict[int, float]:
    """Reciprocal Rank Fusion: rank-only merging of incomparable score scales.

    Weights bias whole lists: for identifier queries the dense list is pure
    noise (embeddings are blind to FP codes), so unweighted fusion interleaves
    noise into the top-5. Doubling the lexical weight there restores precision.
    """
    scores: Dict[int, float] = {}
    for li, ranking in enumerate(ranked_lists):
        w = weights[li] if weights else 1.0
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank + 1)
    return scores


@dataclass
class Hit:
    text: str                    # SOURCE text: what the model and ground.py see
    doc_id: str
    score: float
    page: Optional[int] = None   # 1-based; None/0 = unknown (pre-page-aware index)
    section_path: Optional[str] = None
    chunk_index: Optional[int] = None


def _doc_match(doc_id: str, wanted: str) -> bool:
    """Forgiving doc match: the LLM may cite a truncated or partial id."""
    a, b = doc_id.lower(), wanted.lower()
    if a == b or a.startswith(b) or b in a:
        return True
    # zero-padding: a plain 'fp86' tag must reach the 'fp086' document
    m = re.fullmatch(r"fp0*(\d{1,3})", b)
    return bool(m) and f"fp{int(m.group(1)):03d}" in a


def _registry_fp_numbers(query: str) -> List[int]:
    """FP numbers the registry's own pattern binds in `query` (read-only).

    Read rather than copied: `registry.FP_RE` is the single definition of what
    an FP identifier looks like — chainlit_app, planner and the eval harness
    all bind through that one object — and a second spelling of it here is how
    the two halves of H10 drifted apart in the first place. Registry-as-an-
    enhancement holds: an import failure leaves the token scan alone.
    """
    try:
        from gcf_qna.rag import registry
        return [int(n) for n in registry.FP_RE.findall(query)]
    except Exception:            # registry is an enhancement, never a blocker
        return []


def identifier_tokens(query: str) -> List[str]:
    """The FP / board / addendum codes a query names, dotless and sorted.

    Two readers of the same query, because one is not enough. The lexical
    tokenizer sees every form whose punctuation it glues back together —
    'FP220', 'FP#220', 'FP.220', 'FP-220', 'FP 220' — and nothing else:
    'fp\\d{2,3}' is matched against TOKENS, so a word between the marker and
    the number ('FP no. 220', 'proposal 220', 'proposition n° 220') leaves no
    token to match, and an over-padded 'FP 0086' joins into a four-digit
    'fp0086' the pattern rejects outright. Those are exactly the forms H10
    widened `registry.FP_RE` for, and
    `test_registry_resolver.test_the_consumer_that_keeps_its_own_pattern`
    recorded this function as the one runtime consumer left behind: the
    registry note, the FP-miss guard and doc resolution bound them, the
    per-document BM25 head did not.

    A number the token scan already found is never re-emitted. 'FP086' has to
    stay ONE identifier: two tokens for one document would widen
    `_target_docs`' routing limit and make `search_with_confidence` demand two
    lexical resolutions where the corpus can only offer one spelling.

    Board and addendum codes are the tokenizer's alone — `registry.FP_RE` is
    an FP pattern, and 'Add.220' is deliberately not an FP number.
    """
    toks = {t.replace(".", "") for t in tokenize(query)
            if re.fullmatch(r"fp\d{2,3}|b\.?\d{2}|add\.?\d{2}", t)}
    bound = {int(m.group(1)) for t in toks
             if (m := re.fullmatch(r"fp0*(\d{1,3})", t))}
    toks |= {f"fp{n}" for n in _registry_fp_numbers(query) if n not in bound}
    return sorted(toks)


def doc_is_identifier(doc_id: str, id_toks: Sequence[str]) -> bool:
    """True when the DOCUMENT IS the identifier, rather than merely mentioning it.

    Filenames carry the codes ('..._gcf-b42-02-add16-...-fp274'), so a token
    match against the stem separates 'the FP151 package' from 'the FP242 page
    that lists FP151 among other Guyana programmes'.
    """
    toks = set(_doc_tokens(doc_id))
    return any(v in toks for t in id_toks for v in fp_variants(t))


def _page_no(page: Any) -> Optional[int]:
    """A page hint as an int, or None. Hints arrive from the registry
    ('page': 48), from a note line, and from a caller's literal — all three
    spellings of the same number have to select the same chunks."""
    try:
        return int(str(page).strip())
    except (TypeError, ValueError):
        return None


def _section_hit(path: Optional[str], wanted: Sequence[str]) -> bool:
    """Does a chunk's section path contain one of the asked-for sections?

    The stored path is a trail ('C PROJECT DETAILS > C.2 Financing by
    component'); a hint is the printed id ('C.2'). Each component is reduced
    to its id by the same parser the build used, so the comparison is between
    ids and not between two spellings of a heading: 'C.2' matches the C.2
    heading and any 'C.2.1' beneath it, 'C' matches every C.n, and 'C.2'
    never matches 'C.20'.

    On an index built without section paths this is False for every chunk —
    which is the point of measuring the sectioned index before adopting it.
    """
    if not path or not wanted:
        return False
    ids = [(section_id(part.strip()) or "").upper() for part in str(path).split(">")]
    ids = [i for i in ids if i]
    for w in wanted:
        w = str(w).strip().upper()
        if w and any(i == w or i.startswith(w + ".") for i in ids):
            return True
    return False


def _registry_docs(id_toks: Sequence[str]) -> List[str]:
    """Doc ids the metadata registry maps the FP numbers to (read-only)."""
    out: List[str] = []
    try:
        from gcf_qna.rag import registry
        for t in id_toks:
            m = re.fullmatch(r"fp0*(\d{1,3})", t)
            if not m:
                continue
            row = registry.by_fp(int(m.group(1)))
            if row and row.get("doc_id"):
                out.append(row["doc_id"])
    except Exception:            # registry is an enhancement, never a blocker
        return out
    return out


class Reranker:
    """Cross-encoder query/passage scorer, with a dense-order fallback.

    The fallback is not an error path: on a machine that cannot download the
    model, `score()` returns None and the caller keeps the fusion order it
    already had. `scorer` injects a callable for tests.
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None,
                 scorer=None):
        self.model_name = model_name or os.getenv("RERANK_MODEL", DEFAULT_RERANK_MODEL)
        self.device = device or (os.getenv("RERANK_DEVICE") or None)
        self._scorer = scorer
        self._model = None
        self.failed = False

    @property
    def available(self) -> bool:
        return self._scorer is not None or not self.failed

    def _load(self):
        if self._model is None and not self.failed:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.model_name, device=self.device,
                                           max_length=512)
            except Exception as e:
                self.failed = True
                print(f"reranker unavailable, keeping fusion order: {e}", flush=True)
        return self._model

    def score(self, query: str, texts: Sequence[str]) -> Optional[List[float]]:
        if not texts:
            return []
        if self._scorer is not None:
            return [float(s) for s in self._scorer(query, list(texts))]
        model = self._load()
        if model is None:
            return None
        try:
            pairs = [(query, t[:1800]) for t in texts]
            return [float(s) for s in model.predict(pairs, batch_size=32,
                                                    show_progress_bar=False)]
        except Exception as e:
            self.failed = True
            print(f"reranker failed, keeping fusion order: {e}", flush=True)
            return None


class Retriever:
    def __init__(self, index, chunks: List[Dict[str, Any]], embedder: Embedder,
                 index_dir: Optional[Any] = None, reranker: Optional[Reranker] = None):
        self.index = index
        self.chunks = chunks
        self.embedder = embedder
        self.reranker = reranker or Reranker()
        self._by_doc: Optional[Dict[str, List[int]]] = None
        self.hybrid_enabled = False
        if config.HYBRID and index_dir is not None:
            try:
                self.lexical = LexicalIndex(index_dir)
                self.lexical.ensure(chunks)
                self.hybrid_enabled = True
            except Exception as e:   # lexical is an enhancement, never a blocker
                print(f"lexical index unavailable, dense-only: {e}", flush=True)

    # -- chunk access --------------------------------------------------------
    def _hit(self, i: int, score: float) -> Hit:
        c = self.chunks[i]
        return Hit(text=source_text(c), doc_id=c.get("doc_id", "?"),
                   score=float(score), page=c.get("page") or None,
                   section_path=c.get("section_path"), chunk_index=i)

    # -- candidate generation ------------------------------------------------
    def _dense(self, qv, k: int) -> List[Tuple[int, float]]:
        k = max(1, min(int(k), self.index.ntotal))
        scores, ids = self.index.search(qv, k)
        return [(int(i), float(s)) for s, i in zip(scores[0], ids[0])
                if 0 <= i < len(self.chunks)]

    def _doc_index(self) -> Dict[str, List[int]]:
        if self._by_doc is None:
            by_doc: Dict[str, List[int]] = {}
            for i, c in enumerate(self.chunks):
                by_doc.setdefault(c.get("doc_id", ""), []).append(i)
            self._by_doc = by_doc
        return self._by_doc

    def _dense_over(self, qv, ids: Sequence[int], want: int
                    ) -> Optional[List[Tuple[int, float]]]:
        """Dense scores over a chosen subset of chunk ids, or None.

        FAISS scans only the listed vectors, which is both exact and cheap —
        no global pass to filter afterwards. `params=` needs a FAISS new
        enough to accept it, so this returns None rather than raising and
        every caller keeps its own fallback for the old library.
        """
        if len(ids) == 0:
            return []
        try:
            import faiss
            import numpy as np
            arr = np.asarray(sorted(ids), dtype="int64")
            sel = faiss.IDSelectorArray(arr)   # holds a POINTER: arr and sel must
            params = faiss.SearchParameters()  # both stay alive across the search
            params.sel = sel
            scores, found = self.index.search(qv, min(want, len(arr)), params=params)
            return [(int(i), float(s)) for s, i in zip(scores[0], found[0]) if i >= 0]
        except Exception:
            return None

    def _scoped(self, qv, doc_filter: str, want: int) -> List[Tuple[int, float]]:
        """Dense candidates restricted to one document.

        Inside a single document "budget" has no 175k-chunk competition, so
        annex tables that lose the global similarity contest win the local one.
        FAISS scans only the document's own ids, which is both exact and cheap:
        the previous global-pass-and-filter had to re-scan the whole index
        whenever a routed document had too few chunks in the global head, and
        a wider candidate pool made that the common case, not the rare one.
        """
        ids = [i for d, group in self._doc_index().items()
               if _doc_match(d, doc_filter) for i in group]
        if not ids:
            return []
        picked = self._dense_over(qv, ids, want)
        if picked:
            return picked
        # older FAISS (None) or an empty result: fall back to filtering

        def _pass(k: int) -> List[Tuple[int, float]]:
            out: List[Tuple[int, float]] = []
            for i, s in self._dense(qv, k):
                if _doc_match(self.chunks[i].get("doc_id", ""), doc_filter):
                    out.append((i, s))
                    if len(out) >= want:
                        break
            return out

        got = _pass(min(max(200, want * 40), self.index.ntotal))
        if len(got) < want:
            # generic queries may not surface the target doc in any global
            # top-200 — scan the whole index; a flat scan is ~100 ms here
            got = _pass(self.index.ntotal)
        return got

    def _probes(self, query: str, original: Optional[str]) -> List[str]:
        """The texts allowed to rank chunks INSIDE an already-chosen document.

        The app rewrites the message before retrieving — French into English,
        noise into clean prose, "it" into "FP220". That rewrite is what finds
        the DOCUMENT: it carries the identifier and the vocabulary the corpus
        is actually written in, and turning it on is what moved document
        recall from 47/52 to 51/52. It is not what finds the PAGE. Rewriting
        spends the surface forms that pin one page — the user's own numerals,
        their misspelling of the product name, the wording they chose — on
        canonical phrasing that fits the whole document about equally well.
        Measured, same document scope, labelled evidence page in brackets:

          [p.6]  "Quel est le financement GCF du FP172 au Nepal ?"   rank 9
                 "GCF funding amount for project FP172 in Nepal"     absent
          [p.5]  "fp 173 amazon bioecconomy fund -- how much gcf $$$ ?"  rank 6
                 "FP173 Amazon Bioeconomy Fund proposal: how much
                  GCF funding (USD) is requested/approved?"          absent
          [p.40] "wat is teh gcf finacing for fp274?"                rank 2
                 "What is the Green Climate Fund financing for FP274?"  absent

        So both texts vote, and only where the document is already settled:
        the doc_filter path and the identifier-routed path, never the open
        hybrid fusion that CHOOSES documents. The document set cannot move,
        so neither can document recall — the split buys page precision out of
        the ranking stage alone.

        Whether a message IS one document's message is the caller's call, not
        this module's: a turn that fanned out over several documents must
        pass `original=None`, because its message names all of them and would
        probe each document for the others' figures. chainlit_app applies
        that rule at the one call site.
        """
        probes = [query]
        if (original and _env_on("ORIGINAL_PROBE")
                and " ".join(original.split()) != " ".join(query.split())):
            probes.append(original)
        return probes

    def _scoped_probes(self, qvs: Sequence[Any], doc_filter: str, want: int
                       ) -> List[Tuple[int, float]]:
        """Doc-scoped dense candidates from one probe per vector, fused by rank.

        The same identify-then-rank split the identifier router already runs,
        one level down: _scoped is called once per probe over the SAME
        document, so the probes only argue about which of its chunks come
        first. Reciprocal rank rather than a score blend — a chunk both
        probes place near the top is what a page hit looks like, and neither
        cosine is calibrated against the other's question. Each hit keeps the
        best cosine any probe gave it, since that score is what the answer
        prompt prints beside the excerpt.

        One probe is the plain _scoped call, unchanged, so a caller that
        passes no `original` gets exactly the ranking it got before.
        """
        if len(qvs) == 1:
            return self._scoped(qvs[0], doc_filter, want)
        lists: List[List[int]] = []
        best: Dict[int, float] = {}
        for qv in qvs:
            got = self._scoped(qv, doc_filter, want)
            lists.append([i for i, _ in got])
            for i, s in got:
                best[i] = max(best.get(i, s), s)
        fused = sorted(rrf(lists, config.RRF_K).items(),
                       key=lambda kv: kv[1], reverse=True)[:want]
        return [(i, best[i]) for i, _ in fused]

    def _docs_of(self, indices: Sequence[int]) -> List[str]:
        """Distinct doc ids behind a ranked chunk list, order preserved."""
        out: List[str] = []
        for i in indices:
            d = self.chunks[i].get("doc_id", "")
            if d and d not in out:
                out.append(d)
        return out

    def _target_docs(self, lex_rank: Sequence[int], id_toks: Sequence[str],
                     limit: int = 3, head: int = 40) -> List[str]:
        """Documents the identifier routing will search, best first.

        Mention-magnet defect: '34_...fp242' contains the line "Other GCF
        programs in Guyana include: FP189, FP203, FP152, FP151", which wins the
        BM25 head for an FP151 question and takes a routing slot away from the
        FP151 package itself (measured: the real documents fell to rank 6).
        A document that IS the identifier — by filename stem or by registry
        row — is therefore ordered ahead of every document that merely says it.
        """
        # One BM25 head per identifier, merged round robin, ahead of the joined
        # head. A joined "fp220 OR fp203" query is won outright by whichever
        # code the corpus repeats more often, and the other proposal never
        # reaches the candidate list at all (measured: FP203 was missing from a
        # two-proposal comparison). Each identifier now nominates its own.
        lists: List[List[str]] = []
        if self.hybrid_enabled:
            for t in id_toks:
                found = self.lexical.search(" ".join(fp_variants(t)), head)
                if found:
                    lists.append(self._docs_of(found))
        lists.append(self._docs_of(lex_rank[:head]))
        ordered: List[str] = []
        for depth in range(max(len(x) for x in lists)):
            for group in lists:
                if depth < len(group) and group[depth] not in ordered:
                    ordered.append(group[depth])
        reg = _registry_docs(id_toks)

        def owned(doc: str) -> int:
            """How many of the query's identifiers this document IS.

            Counting rather than testing keeps 'GCF/B.42/02/Add.16' precise:
            b42 alone is satisfied by every B.42 addendum, and only the
            document that owns b42 AND add16 answers the question. For a
            two-proposal comparison every candidate owns exactly one code, so
            the count ties and the round-robin order decides — which is how
            both proposals stay in the list.
            """
            n = sum(1 for t in id_toks if doc_is_identifier(doc, [t]))
            return n or (1 if any(_doc_match(doc, r) for r in reg) else 0)

        ranked = sorted(range(len(ordered)), key=lambda k: (-owned(ordered[k]), k))
        return [ordered[k] for k in ranked[:limit]]

    # -- ranking stages ------------------------------------------------------
    @staticmethod
    def _key(chunk: Dict[str, Any]) -> Tuple[Any, ...]:
        text = " ".join(source_text(chunk).split())
        return (chunk.get("doc_id", ""), chunk.get("page"),
                hashlib.sha1(text.encode("utf-8", "replace")).hexdigest())

    def _dedup(self, cands: Sequence[Tuple[int, float]]) -> List[Tuple[int, float]]:
        """Collapse identical passages BEFORE quotas, so a page reprinted in an
        annex cannot spend two of a document's slots on one piece of evidence."""
        seen = set()
        out = []
        for i, s in cands:
            k = self._key(self.chunks[i])
            if k in seen:
                continue
            seen.add(k)
            out.append((i, s))
        return out

    def _rerank(self, query: str, cands: List[Tuple[int, float]], top_k: int
                ) -> List[Tuple[int, float]]:
        """Order candidates by cross-encoder relevance, keeping their retrieval
        scores: the reranker decides rank, not what the answer prompt prints."""
        # A model load to reorder a handful of passages buys nothing, and it
        # would drag a 90 MB download into every unit test that builds a
        # three-chunk index.
        floor = max(top_k + 1, _env_int("RERANK_MIN_POOL", 8))
        if not _env_on("RERANK", "0") or len(cands) < floor or not self.reranker.available:
            return list(cands)
        pool = cands[:_env_int("RERANK_MAX_PAIRS", 64)]
        scores = self.reranker.score(
            query, [retrieval_text(self.chunks[i]) for i, _ in pool])
        if scores is None:
            return list(cands)
        order = sorted(range(len(pool)), key=lambda k: -scores[k])
        return [pool[k] for k in order] + list(cands[len(pool):])

    def _quota(self, ranked: Sequence[Tuple[int, float]], doc_order: Sequence[str],
               limit: int) -> List[Tuple[int, float]]:
        """Round-robin the ranked candidates across the routed documents.

        Sequential filling gave document #1 ranks 1-3 and document #3 ranks
        7-9, so a two-document comparison could push the second answer out of
        the top-5 entirely. One hit per document per sweep keeps every routed
        document represented near the top.
        """
        buckets: Dict[str, List[Tuple[int, float]]] = {}
        for i, s in ranked:
            buckets.setdefault(self.chunks[i].get("doc_id", "?"), []).append((i, s))

        def pos(doc: str) -> int:
            return next((k for k, d in enumerate(doc_order) if _doc_match(doc, d)),
                        len(doc_order))

        order = sorted(buckets, key=lambda d: (pos(d), d))
        out: List[Tuple[int, float]] = []
        depth = 0
        while len(out) < limit and any(len(buckets[d]) > depth for d in order):
            for d in order:
                if depth < len(buckets[d]):
                    out.append(buckets[d][depth])
                    if len(out) >= limit:
                        break
            depth += 1
        return out

    def _diversify(self, ranked: Sequence[Tuple[int, float]]
                   ) -> List[Tuple[int, float]]:
        """Serve every page once before serving any page twice.

        A stable round robin over (doc, page): the ranking is unchanged except
        that the second and third chunk of one page wait behind the first chunk
        of pages that have not been shown yet. Ten context slots spent on three
        pages is how a question needing two figures from two pages ends up with
        one of them (FP274's consistency question drew page 41 three times).
        """
        if not _env_on("PAGE_DIVERSITY"):
            return list(ranked)
        seen: Dict[Tuple[Any, Any], int] = {}
        keyed = []
        for rank, (i, s) in enumerate(ranked):
            c = self.chunks[i]
            key = (c.get("doc_id"), c.get("page"))
            nth = seen.get(key, 0)
            seen[key] = nth + 1
            keyed.append((nth, rank, (i, s)))
        return [t[2] for t in sorted(keyed, key=lambda t: (t[0], t[1]))]

    def _neighbors(self, i: int) -> List[int]:
        """Chunk indices adjacent to i in the same document, same section first.

        Chunks are written in document/page/reading order, so index adjacency
        IS document adjacency. A same-section neighbour on the next page is
        preferred over one on the same page: it adds evidence the citation
        cannot already reach.
        """
        c = self.chunks[i]
        doc, sec, page = c.get("doc_id"), c.get("section_path"), c.get("page")
        out: List[Tuple[int, int]] = []
        for j in (i + 1, i - 1, i + 2, i - 2):
            if not (0 <= j < len(self.chunks)) or j == i:
                continue
            n = self.chunks[j]
            if n.get("doc_id") != doc:
                continue
            same_sec = bool(sec) and n.get("section_path") == sec
            same_page = bool(page) and n.get("page") == page
            if not (same_sec or same_page):
                continue
            rank = 0 if (same_sec and not same_page) else (1 if same_sec else 2)
            out.append((rank, j))
        return [j for _, j in sorted(out, key=lambda t: (t[0], abs(t[1] - i)))]

    def _expand(self, ranked: List[Tuple[int, float]], top_k: int
                ) -> List[Tuple[int, float]]:
        """Give the best hits their neighbouring context, within a token budget.

        The seeds that survive untouched are the first EXPAND_KEEP of top_k, so
        document recall in the top-5 cannot be spent on expansions.
        """
        if not _env_on("SECTION_EXPAND", "0") or len(ranked) < 2:
            return list(ranked)
        keep_n = max(1, int(round(top_k * _env_float("EXPAND_KEEP", 0.6))))
        keep = list(ranked[:keep_n])
        chosen = {i for i, _ in ranked[:top_k]}
        budget = _env_int("EXPAND_TOKENS", 1200)
        extras: List[Tuple[int, float]] = []
        for i, s in keep:
            for j in self._neighbors(i):
                if len(keep) + len(extras) >= top_k or budget <= 0:
                    break
                if j in chosen:
                    continue
                chosen.add(j)
                extras.append((j, s - 1e-4))
                budget -= max(1, len(source_text(self.chunks[j])) // 4)
        rest = [(i, s) for i, s in ranked[keep_n:] if i not in {j for j, _ in extras}]
        return keep + extras + rest

    def _finalize(self, query: str, cands: Sequence[Tuple[int, float]], top_k: int,
                  doc_order: Optional[Sequence[str]] = None) -> List[Hit]:
        ordered = self._dedup(cands)
        ordered = self._rerank(query, ordered, top_k)
        if doc_order:
            ordered = self._quota(ordered, doc_order, max(top_k * 2, top_k))
        ordered = self._diversify(ordered)
        ordered = self._expand(list(ordered), top_k)
        return [self._hit(i, s) for i, s in ordered[:top_k]]

    # -- public API ----------------------------------------------------------
    def search_with_confidence(self, query: str, top_k: int = 5,
                               doc_filter: Optional[str] = None,
                               original: Optional[str] = None):
        """(hits, confidence) — confidence is the best dense cosine for the
        query, the signal behind the no-answer guard. Identifier queries
        return 1.0 only when every identifier resolves: the document match
        is then exact by construction.

        Confidence is read off `query` alone even when `original` is given:
        the rewrite is the text that identifies the document, and the
        weak-signal guard is a statement about the document match, not about
        which of its pages ranked first."""
        import numpy as np
        qv = np.asarray(self.embedder.encode([query]), dtype="float32")
        scores, _ = self.index.search(qv, 1)
        conf = float(scores[0][0]) if scores.size else 0.0
        id_toks = identifier_tokens(query)
        if id_toks and self.hybrid_enabled:
            # 1.0 only when EVERY identifier resolves in the corpus. FTS5 MATCH
            # is OR-joined, so a joined query lets one live token vouch for a
            # dead one ("B.42/02/Add.99": b42 hits, add99 does not) and suppress
            # the weak-signal note on a nonexistent document (review finding #2).
            # An identifier counts as resolved if ANY padding variant does.
            if all(any(self.lexical.search(v, 1) for v in fp_variants(t))
                   for t in id_toks):
                conf = 1.0
        return self.search(query, top_k, doc_filter, original), conf

    def search(self, query: str, top_k: int = 5,
               doc_filter: Optional[str] = None,
               original: Optional[str] = None) -> List[Hit]:
        """Top-k chunks for a query; doc_filter restricts hits to one document.

        `original` is the user's own message when `query` is a rewrite of it.
        It never selects documents — only ranks chunks inside the ones the
        query already chose. See _probes.
        """
        import numpy as np
        probes = self._probes(query, original)
        arr = np.asarray(self.embedder.encode(probes), dtype="float32")
        qv, qvs = arr[:1], [arr[i:i + 1] for i in range(len(probes))]
        mult = _env_int("CANDIDATE_POOL_MULT", 3)
        pool = max(top_k * mult, top_k)

        if doc_filter is not None:
            cands = self._scoped_probes(qvs, doc_filter, pool)
            if not cands:
                # a filter that matches no document (e.g. a fabricated doc tag)
                # must degrade to unscoped search, never to an empty context
                return self.search(query, top_k, None, original)
            return self._finalize(query, cands, top_k)

        if not self.hybrid_enabled:
            return self._finalize(query, self._dense(qv, pool), top_k)

        # hybrid: fuse dense and lexical rankings by reciprocal rank
        n = max(config.CANDIDATES_PER_RETRIEVER, pool)
        dense_rank = [i for i, _ in self._dense(qv, n)]
        # Identifier queries: restrict the lexical search to the id tokens
        # alone. With the full query, common words ("accredited entity")
        # drag topical wrong-doc chunks into BM25's tail, where dual-list
        # membership out-sums the right doc's lexical-only head.
        id_toks = identifier_tokens(query)
        lex_query = (" ".join(v for t in id_toks for v in fp_variants(t))
                     if id_toks else query)
        lex_rank = self.lexical.search(lex_query, n)
        if id_toks and lex_rank:
            # Two-stage for identifier queries: the lexical head IDENTIFIES
            # the document(s); a doc-scoped dense search then ranks chunks
            # semantically WITHIN each. Without this, id-tagged chunks tie
            # in BM25 and the right document's pages are picked ~randomly
            # (observed: FP274's financing question drew pp. 56-187, missed
            # the financing section, answered "no figure stated").
            targets = self._target_docs(lex_rank, id_toks,
                                        limit=max(3, len(id_toks)))
            per = max(3, top_k // max(1, len(targets)))
            routed: List[Tuple[int, float]] = []
            for d in targets:
                routed += self._scoped_probes(qvs, d, per * mult)
            if routed:
                return self._finalize(query, routed, top_k, doc_order=targets)
        weights = [1.0, 2.0] if id_toks else [1.0, 1.0]
        fused = sorted(rrf([dense_rank, lex_rank], config.RRF_K, weights).items(),
                       key=lambda kv: kv[1], reverse=True)[:pool]
        return self._finalize(query, fused, top_k)

    # -- the supplementary probe ---------------------------------------------
    def _probe_ids(self, doc_id: str, pages: Sequence[Any],
                   sections: Sequence[str]) -> List[int]:
        """One document's chunk ids on the asked-for pages or sections."""
        want_pages = {p for p in (_page_no(p) for p in pages) if p is not None}
        want_secs = [s for s in sections if str(s).strip()]
        if not want_pages and not want_secs:
            return []
        out: List[int] = []
        for d, group in self._doc_index().items():
            if not _doc_match(d, doc_id):
                continue
            for i in group:
                c = self.chunks[i]
                if (_page_no(c.get("page")) in want_pages
                        or _section_hit(c.get("section_path"), want_secs)):
                    out.append(i)
        return sorted(out)

    def probe_pages(self, doc_id: str, pages: Sequence[Any] = (), k: int = 4,
                    query: Optional[str] = None,
                    sections: Sequence[str] = ()) -> List[Hit]:
        """One document's chunks on named pages/sections: the second ask.

        `search` answers "which passages best match this question". This
        answers "show me THAT page of THAT document" — a different question,
        and the one a conflict turn has to ask. The registry already knows
        where the disagreeing figure is printed (`registry.conflicts()` gives
        every candidate a `page` and a `section`, and `_conflict_lines`
        prints them as '(p.N, SECTION)'), and the measured shortfall is not
        that the page is missing: all fourteen conflict-class evidence pages
        are indexed in both builds. It is that a second printing of a figure,
        deep in a component table, loses the similarity contest to the cover
        page that says the same thing in the question's own words. No amount
        of reranking the wrong candidate pool fixes that; asking for the page
        by name does.

        The caller decides when to ask, and this module never wires itself in
        — a supplementary query that fires on its own would spend top-k slots
        on every turn that merely mentions a document.

        Three refusals hold it honest:

        * **The document is the caller's.** Matched with `_doc_match`, and an
          empty result stays empty. `search`'s `doc_filter` degrades to an
          unscoped search when nothing matches, which is right for a primary
          query and wrong for a supplement: pages fetched from some other
          document would be cited as this one's.
        * **A page that is not indexed is not invented.** Ask for four pages,
          get back the ones that exist — the caller can see which by the
          pages the hits carry.
        * **`query` orders, it never selects.** With a query the chunks the
          page filter selected are scored by dense cosine against it; without
          one they come back in document reading order, scored 0.0, because a
          caller who asked for pages did not ask for a ranking.

        Slots go one per page before any page takes a second — the rule
        `_diversify` applies to search results, written out again here rather
        than reused because for a page probe the spread IS the request:
        `PAGE_DIVERSITY=0` must not be able to spend all four slots on one
        page of a two-page conflict.

        `sections` takes printed section ids ('C.2') and is inert on an index
        whose chunks carry no `section_path` — which is every chunk of
        `data/index/default`. Pages work on both builds.
        """
        ids = self._probe_ids(doc_id, pages, sections)
        if not ids:
            return []
        scored: List[Tuple[int, float]] = []
        if query:
            import numpy as np
            qv = np.asarray(self.embedder.encode([query]), dtype="float32")
            scored = self._dense_over(qv, ids, len(ids)) or []
        if not scored:
            scored = [(i, 0.0) for i in ids]
        scored = self._dedup(scored)
        seen: Dict[Any, int] = {}
        keyed: List[Tuple[int, int, Tuple[int, float]]] = []
        for rank, (i, s) in enumerate(scored):
            page = self.chunks[i].get("page")
            nth = seen.get(page, 0)
            seen[page] = nth + 1
            keyed.append((nth, rank, (i, s)))
        ordered = [t[2] for t in sorted(keyed, key=lambda t: (t[0], t[1]))]
        return [self._hit(i, s) for i, s in ordered[:max(0, int(k))]]
