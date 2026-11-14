"""Hybrid retrieval: fuse keyword (TF-IDF) and semantic (dense) rankings.

Each backend scores on its own scale (TF-IDF cosine and embedding cosine are
not comparable), so scores are normalized per query - each hit is divided by
its backend's top score - then combined with fixed weights. Measured on the
sample-filing evaluation set (scripts/evaluate_retrieval.py), this hybrid
ranking matched or beat either backend alone on recall@5 and MRR (recall@1
and recall@3 were unchanged versus the tuned keyword baseline).
"""
from __future__ import annotations

from dataclasses import replace

from .retriever import RetrievedChunk, Retriever

OVERFETCH = 3


class HybridRetriever:
    """Weighted normalized-score fusion over multiple Retriever backends."""

    def __init__(self, backends: list[tuple[Retriever, float]]) -> None:
        if not backends:
            raise ValueError("HybridRetriever needs at least one backend")
        total = sum(w for _, w in backends)
        if total <= 0:
            raise ValueError("backend weights must sum to a positive value")
        self.backends = [(r, w / total) for r, w in backends]

    def retrieve(self, question: str, k: int = 5) -> list[RetrievedChunk]:
        merged: dict[str, list] = {}  # chunk_id -> [fused_score, chunk]
        for retriever, weight in self.backends:
            hits = retriever.retrieve(question, k=max(k, k * OVERFETCH))
            peak = max((h.score for h in hits), default=0.0)
            for h in hits:
                normalized = h.score / peak if peak else 0.0
                entry = merged.get(h.chunk_id)
                if entry is None:
                    merged[h.chunk_id] = [weight * normalized, h]
                else:
                    entry[0] += weight * normalized
        ranked = sorted(merged.values(), key=lambda e: e[0], reverse=True)
        return [
            replace(chunk, score=round(score, 4))
            for score, chunk in ranked[:k]
            if score > 0
        ]
