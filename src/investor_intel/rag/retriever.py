from __future__ import annotations

import re
from dataclasses import dataclass

from ..embeddings.base import Embedder
from ..vectorstore.memory import InMemoryVectorStore

# Capitalized phrases in a question usually name an entity (company, product).
# When those words appear in a document's id/source, that document is very
# likely the intended one, so its chunks get a bounded score boost. This fixes
# cross-company drift where shared boilerplate ("total revenue of ...") made a
# different company's filing outrank the named one.
_ENTITY_PHRASE = re.compile(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*")
_TOKEN = re.compile(r"[a-z0-9]+")

# Sentence-initial words that look like entities but are ordinary question
# words; harmless to keep tiny - unknown words simply never match a doc id.
_QUESTION_WORDS = frozenset([
    "what", "how", "why", "when", "which", "who", "did", "does", "do", "is",
    "are", "was", "were", "in", "for", "compare", "show", "tell", "give",
    "list", "summarize", "describe", "between", "the", "a", "an",
])

ENTITY_BOOST = 0.5
OVERFETCH = 3


def _entity_tokens(question: str) -> set[str]:
    tokens: set[str] = set()
    for phrase in _ENTITY_PHRASE.findall(question):
        for word in phrase.split():
            w = word.lower()
            if w not in _QUESTION_WORDS:
                tokens.add(w)
    return tokens


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
        hits = self.store.search(self.embedder.embed(question), k=max(k, k * OVERFETCH))
        entities = _entity_tokens(question)
        if entities:
            for h in hits:
                doc_tokens = set(_TOKEN.findall(h.get("doc_id", "") + " " + h.get("source", "")))
                overlap = len(entities & doc_tokens)
                if overlap:
                    h["score"] = round(h["score"] * (1 + ENTITY_BOOST * overlap / len(entities)), 4)
            hits.sort(key=lambda h: h["score"], reverse=True)
        return [
            RetrievedChunk(
                text=h["text"],
                doc_id=h["doc_id"],
                chunk_id=h["chunk_id"],
                heading=h.get("heading", ""),
                source=h.get("source", ""),
                score=h["score"],
            )
            for h in hits[:k]
        ]
