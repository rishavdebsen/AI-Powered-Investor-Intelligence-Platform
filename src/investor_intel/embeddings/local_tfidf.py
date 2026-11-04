"""Dependency-free TF-IDF embedder used by the local profile and tests.

Vectors are sparse dicts (term -> weight); cosine similarity runs directly on
them, so the whole retrieval path works with no external services or keys.
"""
from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(["a", "an", "the", "of", "was", "were", "is", "are", "did", "do", "does", "how", "much", "make", "made", "what", "which", "who", "whom", "and", "or", "but", "in", "on", "at", "to", "for", "with", "by", "from", "as", "it", "its", "this", "that", "these", "those", "be", "been"])


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


class TfidfEmbedder:
    def __init__(self) -> None:
        self.df: Counter[str] = Counter()
        self.n_docs = 0

    def fit(self, texts: list[str]) -> None:
        self.df = Counter()
        self.n_docs = len(texts)
        for t in texts:
            self.df.update(set(tokenize(t)))

    def embed(self, text: str) -> dict[str, float]:
        tf = Counter(tokenize(text))
        if not tf or self.n_docs == 0:
            return {}
        vec: dict[str, float] = {}
        for term, count in tf.items():
            idf = math.log((1 + self.n_docs) / (1 + self.df.get(term, 0))) + 1.0
            vec[term] = (count / len(tf)) * idf
        return vec


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(w * b.get(t, 0.0) for t, w in a.items())
    na = math.sqrt(sum(w * w for w in a.values()))
    nb = math.sqrt(sum(w * w for w in b.values()))
    return dot / (na * nb) if na and nb else 0.0
