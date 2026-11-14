"""In-memory vector store over dense list embeddings, with JSON persistence.

Same interface as the sparse TF-IDF store (``memory.py``) so the retriever
and service can swap embedding backends without changing callers. Cosine
similarity runs in pure Python; honest for a portfolio-scale corpus
(hundreds of chunks).
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def cosine_dense(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryDenseVectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[list[float], dict]] = []

    def add(self, vector: list[float], metadata: dict) -> None:
        self._items.append((list(vector), metadata))

    def search(self, query: list[float], k: int = 5) -> list[dict]:
        scored = [(cosine_dense(query, vec), meta) for vec, meta in self._items]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(meta, score=round(s, 4)) for s, meta in scored[:k] if s > 0]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._items))

    def load(self, path: str | Path) -> None:
        self._items = [(vec, meta) for vec, meta in json.loads(Path(path).read_text())]

    def __len__(self) -> int:
        return len(self._items)
