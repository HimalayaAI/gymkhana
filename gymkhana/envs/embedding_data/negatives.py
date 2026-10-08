"""Mine and verify hard negatives for retrieval training rows.

The seed-preserving environment keeps whatever negatives the seed dataset
shipped with. Those are often easy ("not mentioned" strings, answers to other
questions) and are never checked for false labels. This module builds the
negative list from a passage pool instead: retrieve near neighbours of the
positive, then drop every candidate that a reranker thinks answers the query.

Retrieval and reranking are plain callables so the logic is testable without
model downloads. ``sentence_transformer_retriever`` and ``cross_encoder_scorer``
build the usual model-backed implementations (optional dependency).
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

# retrieve(queries, passages, top_k) -> for each query, passage indices, best first
Retriever = Callable[[Sequence[str], Sequence[str], int], list[list[int]]]
# rerank([(query, passage), ...]) -> relevance probability in [0, 1] per pair
Reranker = Callable[[Sequence[tuple[str, str]]], list[float]]

MINED_VERIFIED = "mined_verified"


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def select_negatives(
    candidates: Sequence[str],
    candidate_scores: Sequence[float],
    positive_score: float,
    n_negatives: int,
    *,
    abs_threshold: float = 0.5,
    margin: float = 0.1,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Keep the first ``n_negatives`` candidates that are not likely false negatives.

    A candidate is rejected when its relevance score is >= ``abs_threshold`` or
    within ``margin`` of the positive's score. Returns (kept, rejected) with
    rejected as (text, score) pairs, so callers can write an audit trail.
    """
    kept: list[str] = []
    rejected: list[tuple[str, float]] = []
    for text, score in zip(candidates, candidate_scores):
        if score >= abs_threshold or score >= positive_score - margin:
            rejected.append((text, score))
        elif len(kept) < n_negatives:
            kept.append(text)
    return kept, rejected


def mine_verified_negatives(
    rows: Iterable[Mapping[str, Any]],
    *,
    retrieve: Retriever,
    rerank: Reranker,
    extra_passages: Sequence[str] = (),
    n_negatives: int = 5,
    pool_top_k: int = 30,
    abs_threshold: float = 0.5,
    margin: float = 0.1,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return rows with up to ``n_negatives`` verified negatives each.

    Pool = every row's positive plus ``extra_passages``. A candidate is
    rejected when it is the row's own positive, is a positive of any row with
    the same query (duplicate questions are the main source of false
    negatives in pooled corpora), or the reranker scores it >= ``abs_threshold``
    or within ``margin`` of the positive's score. Rows that end up with fewer
    than ``n_negatives`` are kept and flagged ``below_target``; nothing is padded.
    """
    rows = [dict(row) for row in rows]
    pool: list[str] = []
    index: dict[str, int] = {}
    for text in [_norm(r["positive"]) for r in rows] + [_norm(t) for t in extra_passages]:
        if text and text not in index:
            index[text] = len(pool)
            pool.append(text)

    positives_of_query: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        positives_of_query[_norm(row["query"]).casefold()].add(_norm(row["positive"]))

    queries = [_norm(r["query"]) for r in rows]
    ranked = retrieve(queries, pool, pool_top_k) if rows and pool else [[] for _ in rows]

    stats = {"rows": len(rows), "below_target": 0, "candidates": 0, "rejected_false_negative": 0}
    out: list[dict[str, Any]] = []
    for row, query, order in zip(rows, queries, ranked):
        positive = _norm(row["positive"])
        banned = positives_of_query[query.casefold()]
        candidates = [pool[i] for i in order if pool[i] not in banned]
        stats["candidates"] += len(candidates)
        scores = rerank([(query, positive)] + [(query, c) for c in candidates])
        pos_score, cand_scores = scores[0], scores[1:]
        negatives, rejected = select_negatives(
            candidates, cand_scores, pos_score, n_negatives, abs_threshold=abs_threshold, margin=margin
        )
        stats["rejected_false_negative"] += len(rejected)
        row["negatives"] = negatives
        row["negative_types"] = [MINED_VERIFIED] * len(negatives)
        row["below_target"] = len(negatives) < n_negatives
        stats["below_target"] += row["below_target"]
        out.append(row)
    return out, stats


def sentence_transformer_retriever(model_name: str, query_prefix: str = "", doc_prefix: str = "") -> Retriever:
    """Dense retriever; for e5 pass ``query_prefix='query: '``, ``doc_prefix='passage: '``."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def retrieve(queries: Sequence[str], passages: Sequence[str], top_k: int) -> list[list[int]]:
        p = model.encode([doc_prefix + t for t in passages], normalize_embeddings=True, convert_to_numpy=True)
        q = model.encode([query_prefix + t for t in queries], normalize_embeddings=True, convert_to_numpy=True)
        sims = q @ p.T
        return [list(map(int, row.argsort()[::-1][:top_k])) for row in sims]

    return retrieve


def cross_encoder_scorer(model_name: str = "BAAI/bge-reranker-v2-m3", batch_size: int = 32) -> Reranker:
    """Cross-encoder returning sigmoid probabilities."""
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, activation_fn=torch.nn.Sigmoid())

    def rerank(pairs: Sequence[tuple[str, str]]) -> list[float]:
        return [float(s) for s in model.predict(list(pairs), batch_size=batch_size)]

    return rerank


__all__ = [
    "MINED_VERIFIED",
    "Reranker",
    "Retriever",
    "cross_encoder_scorer",
    "mine_verified_negatives",
    "select_negatives",
    "sentence_transformer_retriever",
]
