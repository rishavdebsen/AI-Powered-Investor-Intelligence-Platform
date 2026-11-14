from __future__ import annotations

from typing import Protocol

Vector = dict[str, float] | list[float]


class Embedder(Protocol):
    vector_kind: str  # "sparse" (dict vectors) or "dense" (list vectors)

    def fit(self, texts: list[str]) -> None: ...
    def embed(self, text: str) -> Vector: ...
