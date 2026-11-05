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


def _two_company_store():
    chunks = chunk_document(
        "vertex-mobility-fy2025",
        "# Financial Highlights\n\nTotal revenue of $1.9 billion for fiscal year 2025. Revenue grew 34% year over year.",
    )
    chunks += chunk_document(
        "meridian-industries-fy2025",
        "# Financial Highlights\n\nTotal revenue of $4.2 billion for fiscal year 2025. Revenue grew 12% year over year.",
    )
    emb = TfidfEmbedder()
    emb.fit([c.text for c in chunks])
    store = InMemoryVectorStore()
    for c in chunks:
        store.add(
            emb.embed(c.text),
            {"text": c.text, "doc_id": c.doc_id, "chunk_id": c.chunk_id, "heading": c.heading, "source": c.doc_id + ".md"},
        )
    return store, emb


def test_named_company_ranks_first_with_shared_boilerplate():
    store, emb = _two_company_store()
    hits = Retriever(store, emb).retrieve("What was Meridian Industries total revenue in fiscal 2025?", k=1)
    assert hits and hits[0].doc_id == "meridian-industries-fy2025"


def test_unnamed_question_keeps_pure_relevance_order():
    store, emb = _two_company_store()
    hits = Retriever(store, emb).retrieve("total revenue for fiscal year", k=2)
    assert len(hits) == 2
