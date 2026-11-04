"""Answer questions over retrieved chunks, always with citations.

- With an LLM configured: the model writes the answer, grounded in the
  retrieved context; citations come from retrieval, not from the model.
- Without an LLM (local profile): an extractive fallback quotes the top
  excerpts verbatim. No number in an answer ever originates from the system
  itself - everything traces back to a retrieved chunk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import LLM
from .retriever import RetrievedChunk, Retriever

_PROMPT = (
    "You are an investor-intelligence assistant. Answer the question using only "
    "the provided excerpts from company filings. If the excerpts do not contain "
    "the answer, say so. Do not invent figures.\n\n"
    "Excerpts:\n{context}\n\nQuestion: {question}\nAnswer:"
)


@dataclass
class Answer:
    text: str
    citations: list[dict] = field(default_factory=list)
    retrieved: int = 0
    mode: str = "extractive"


def _citation(chunk: RetrievedChunk) -> dict:
    return {"source": chunk.source, "heading": chunk.heading, "chunk_id": chunk.chunk_id, "score": chunk.score}


def answer_question(question: str, retriever: Retriever, llm: LLM | None = None, k: int = 5) -> Answer:
    chunks = retriever.retrieve(question, k=k)
    if not chunks:
        return Answer("I could not find anything relevant in the ingested documents.", [], 0)
    citations = [_citation(c) for c in chunks]
    if llm is None:
        excerpts = "\n\n".join(f"[{c.source} - {c.heading}] {c.text[:400]}" for c in chunks[:3])
        return Answer(f"Relevant excerpts from the ingested documents:\n\n{excerpts}", citations, len(chunks), "extractive")
    context = "\n\n".join(f"[Source: {c.source}, section: {c.heading}]\n{c.text}" for c in chunks)
    text = llm.complete([
        {"role": "system", "content": "You answer only from the supplied excerpts."},
        {"role": "user", "content": _PROMPT.format(context=context, question=question)},
    ])
    return Answer(text, citations, len(chunks), "llm")
