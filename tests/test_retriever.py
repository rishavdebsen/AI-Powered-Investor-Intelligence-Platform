from investor_intel.embeddings.local_tfidf import TfidfEmbedder
from investor_intel.ingestion.chunker import chunk_document
from investor_intel.rag.retriever import Retriever
from investor_intel.vectorstore.memory import InMemoryVectorStore


def test_retriever_finds_right_section():
    chunks = chunk_document("d1", "# Finance\n\nRevenue was $4.2 billion and net income was $742 million.")
    chunks += chunk_document("d2", "# Risks\n\nThe company depends on a small number of semiconductor suppliers.")
    emb = TfidfEmbedder()
    emb.fit([c.text for c in chunks])
    store = InMemoryVectorStore()
    for c in chunks:
        store.add(emb.embed(c.text), {"text": c.text, "doc_id": c.doc_id, "chunk_id": c.chunk_id, "heading": c.heading, "source": "s"})
    hits = Retriever(store, emb).retrieve("how much revenue and net income did the company report", k=1)
    assert hits and hits[0].doc_id == "d1"
