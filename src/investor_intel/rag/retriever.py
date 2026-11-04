from __future__ import annotations

from dataclasses import dataclass

from ..embeddings.base import Embedder
from ..vectorstore.memory import InMemoryVectorStore


@dataclass
class RetrievedChunk:
    text: str
    doc_id: str
    chunk_id: str
    heading: str
    source: str
    score: float


class Retriever:
    def __init__(self, store: InMemoryVectorStore, embedder: Embedder) -> None:
        self.store = store
        self.embedder = embedder

    def retrieve(self, question: str, k: int = 5) -> list[RetrievedChunk]:
        hits = self.store.search(self.embedder.embed(question), k=k)
        return [
            RetrievedChunk(
                text=h["text"],
                doc_id=h["doc_id"],
                chunk_id=h["chunk_id"],
                heading=h.get("heading", ""),
                source=h.get("source", ""),
                score=h["score"],
            )
            for h in hits
        ]
