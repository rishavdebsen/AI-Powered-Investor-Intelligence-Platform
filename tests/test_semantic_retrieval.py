"""Semantic retrieval tests. Skipped automatically when the optional
sentence-transformers dependency is not installed, so CI on the default
dependency-free profile still passes."""
import pytest

pytest.importorskip("sentence_transformers")

from investor_intel.embeddings.local_tfidf import TfidfEmbedder
from investor_intel.embeddings.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)
from investor_intel.ingestion.chunker import chunk_document
from investor_intel.rag.hybrid import HybridRetriever
from investor_intel.rag.retriever import Retriever
from investor_intel.vectorstore.dense import InMemoryDenseVectorStore
from investor_intel.vectorstore.memory import InMemoryVectorStore


def _semantic_retriever():
    chunks = chunk_document(
        "vertex-mobility-fy2025",
        "# Financial Highlights\n\nTotal revenue of $1.9 billion for fiscal year 2025. Revenue grew 34% year over year.",
    )
    chunks += chunk_document(
        "meridian-industries-fy2025",
        "# Financial Highlights\n\nTotal revenue of $4.2 billion for fiscal year 2025. Revenue grew 11% year over year.",
    )
    chunks += chunk_document(
        "vertex-mobility-fy2025",
        "# Risk Factors\n\nBattery-cell supply is concentrated among a few overseas suppliers, and import duties can move costs quickly.",
    )
    emb = SentenceTransformerEmbedder()
    store = InMemoryDenseVectorStore()
    for c in chunks:
        store.add(
            emb.embed(c.text),
            {"text": c.text, "doc_id": c.doc_id, "chunk_id": c.chunk_id, "heading": c.heading, "source": c.doc_id + ".md"},
        )
    return Retriever(store, emb)


def test_paraphrase_question_finds_revenue_chunk():
    retriever = _semantic_retriever()
    hits = retriever.retrieve("How fast is the top line growing?", k=3)
    assert hits
    assert any(h.heading == "Financial Highlights" for h in hits)


def test_named_company_still_ranks_first_with_dense_embeddings():
    retriever = _semantic_retriever()
    hits = retriever.retrieve("What was Meridian Industries total revenue in fiscal 2025?", k=1)
    assert hits and hits[0].doc_id == "meridian-industries-fy2025"


def test_hybrid_matches_semantic_on_paraphrase_and_keeps_named_company_first():
    chunks = chunk_document(
        "vertex-mobility-fy2025",
        "# Financial Highlights\n\nTotal revenue of $1.9 billion for fiscal year 2025. Revenue grew 34% year over year.",
    )
    chunks += chunk_document(
        "meridian-industries-fy2025",
        "# Financial Highlights\n\nTotal revenue of $4.2 billion for fiscal year 2025. Revenue grew 11% year over year.",
    )
    tfidf = TfidfEmbedder()
    tfidf.fit([c.text for c in chunks])
    sparse = InMemoryVectorStore()
    dense_emb = SentenceTransformerEmbedder()
    dense = InMemoryDenseVectorStore()
    for c in chunks:
        meta = {"text": c.text, "doc_id": c.doc_id, "chunk_id": c.chunk_id, "heading": c.heading, "source": c.doc_id + ".md"}
        sparse.add(tfidf.embed(c.text), meta)
        dense.add(dense_emb.embed(c.text), meta)
    hybrid = HybridRetriever([(Retriever(sparse, tfidf), 0.5), (Retriever(dense, dense_emb), 0.5)])
    hits = hybrid.retrieve("What was Meridian Industries total revenue in fiscal 2025?", k=2)
    assert hits and hits[0].doc_id == "meridian-industries-fy2025"
    assert all(h.score > 0 for h in hits)


def test_dense_store_roundtrip(tmp_path):
    store = InMemoryDenseVectorStore()
    store.add([1.0, 0.0], {"text": "a", "doc_id": "d", "chunk_id": "d:0"})
    store.add([0.0, 1.0], {"text": "b", "doc_id": "d", "chunk_id": "d:1"})
    path = tmp_path / "store.json"
    store.save(path)
    loaded = InMemoryDenseVectorStore()
    loaded.load(path)
    hits = loaded.search([1.0, 0.0], k=1)
    assert hits and hits[0]["text"] == "a"
