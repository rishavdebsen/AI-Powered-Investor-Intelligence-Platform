from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def embed(self, text: str) -> dict[str, float]: ...
