"""In-memory vector store over sparse dict embeddings, with JSON persistence.

Small by design: honest for a portfolio-scale corpus (hundreds of chunks).
The azure profile can swap in Azure AI Search without changing callers.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..embeddings.local_tfidf import cosine


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: list[tuple[dict[str, float], dict]] = []

    def add(self, vector: dict[str, float], metadata: dict) -> None:
        self._items.append((vector, metadata))

    def search(self, query: dict[str, float], k: int = 5) -> list[dict]:
        scored = [(cosine(query, vec), meta) for vec, meta in self._items]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(meta, score=round(s, 4)) for s, meta in scored[:k] if s > 0]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self._items))

    def load(self, path: str | Path) -> None:
        self._items = [(vec, meta) for vec, meta in json.loads(Path(path).read_text())]

    def __len__(self) -> int:
        return len(self._items)
