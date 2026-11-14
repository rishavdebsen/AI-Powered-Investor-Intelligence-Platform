"""Application service: ties ingestion, embedding, retrieval, KPI extraction
and persistence together. One instance per app process.

Retrieval backend is chosen by II_EMBEDDINGS_PROVIDER:
- local (default): dependency-free TF-IDF, no downloads or keys.
- sentence-transformers: local dense embeddings on CPU (optional dependency).
- hybrid: TF-IDF + sentence-transformers fused (see rag/hybrid.py).
- openai: OpenAI-compatible embeddings (wired, not yet verified).
"""
from __future__ import annotations

from pathlib import Path

from .config import Settings
from .db.models import ChunkRow, Document, KPIRecord, QueryLog
from .db.session import make_session_factory
from .embeddings.local_tfidf import TfidfEmbedder
from .ingestion.chunker import chunk_document
from .ingestion.pdf_loader import load_document
from .llm.base import LLM
from .rag.chain import Answer, answer_question
from .rag.hybrid import HybridRetriever
from .rag.kpi import extract_kpis
from .rag.retriever import Retriever
from .vectorstore.dense import InMemoryDenseVectorStore
from .vectorstore.memory import InMemoryVectorStore


def _store_for(embedder) -> object:
    if getattr(embedder, "vector_kind", "sparse") == "dense":
        return InMemoryDenseVectorStore()
    return InMemoryVectorStore()


class InvestorIntelService:
    def __init__(self, settings: Settings, database_url: str | None = None, data_dir: str | None = None) -> None:
        self.settings = settings
        self.data_dir = Path(data_dir or settings.data_dir).resolve()
        self._session_factory = make_session_factory(database_url or settings.database_url)
        # Each index is (embedder, store, fusion weight); one entry for a
        # single backend, two for hybrid.
        self._indexes: list[tuple[object, object, float]] = self._build_indexes()
        self._corpus: list[tuple[str, dict]] = []  # (chunk text, metadata)
        self._retriever = self._make_retriever()
        self._llm: LLM | None = self._build_llm()

    def _build_indexes(self) -> list[tuple[object, object, float]]:
        s = self.settings
        provider = s.embeddings_provider
        if provider == "hybrid":
            weight = min(max(s.hybrid_tfidf_weight, 0.0), 1.0)
            return [
                (TfidfEmbedder(), InMemoryVectorStore(), weight),
                (self._build_sentence_embedder(), InMemoryDenseVectorStore(), 1.0 - weight),
            ]
        embedder = self._build_embedder()
        return [(embedder, _store_for(embedder), 1.0)]

    def _build_sentence_embedder(self):
        from .embeddings.sentence_transformer_embedder import (
            SentenceTransformerEmbedder,
        )

        return SentenceTransformerEmbedder(model_name=self.settings.sentence_transformer_model)

    def _build_embedder(self):
        s = self.settings
        if s.embeddings_provider in ("sentence-transformers", "sentence_transformers"):
            return self._build_sentence_embedder()
        if s.embeddings_provider == "openai" and s.openai_api_key:
            from .embeddings.openai_embedder import OpenAIEmbedder

            return OpenAIEmbedder(api_key=s.openai_api_key, base_url=s.openai_base_url, model=s.openai_embedding_model)
        return TfidfEmbedder()

    def _make_retriever(self):
        backends = [(Retriever(store, embedder), weight) for embedder, store, weight in self._indexes]
        if len(backends) == 1:
            return backends[0][0]
        return HybridRetriever(backends)

    def _resolve_ingest_path(self, path: str | Path) -> Path:
        resolved = Path(path).resolve()
        if not (resolved == self.data_dir or self.data_dir in resolved.parents):
            raise ValueError(f"ingest path must be inside the data directory ({self.data_dir})")
        return resolved

    def _build_llm(self) -> LLM | None:
        s = self.settings
        if s.llm_provider == "openai" and s.openai_api_key:
            from .llm.openai_client import OpenAIChat

            return OpenAIChat(api_key=s.openai_api_key, model=s.openai_chat_model, base_url=s.openai_base_url)
        if s.llm_provider == "azure-openai" and s.azure_openai_api_key:
            from .llm.openai_client import OpenAIChat

            return OpenAIChat(
                api_key=s.azure_openai_api_key,
                model=s.azure_openai_chat_deployment,
                azure_endpoint=s.azure_openai_endpoint,
            )
        return None

    def _rebuild_index(self) -> None:
        for embedder, store, _ in self._indexes:
            embedder.fit([text for text, _ in self._corpus])
            store._items.clear()
            for text, meta in self._corpus:
                store.add(embedder.embed(text), meta)
        self._retriever = self._make_retriever()

    def ingest(self, path: str | Path) -> dict:
        path = self._resolve_ingest_path(path)
        doc_id, text = load_document(path)
        name = Path(path).name
        chunks = chunk_document(doc_id, text)
        kpis = extract_kpis(text)
        with self._session_factory() as session:
            if session.query(Document).filter_by(doc_id=doc_id).first():
                return {"doc_id": doc_id, "name": name, "chunks": 0, "kpis": 0, "duplicate": True}
            session.add(Document(doc_id=doc_id, name=name))
            for c in chunks:
                session.add(ChunkRow(chunk_id=c.chunk_id, doc_id=doc_id, heading=c.heading, text=c.text))
                self._corpus.append(((c.heading + "\n" + c.text) if c.heading else c.text, {"text": c.text, "doc_id": doc_id, "chunk_id": c.chunk_id, "heading": c.heading, "source": name}))
            for kpi in kpis:
                session.add(KPIRecord(doc_id=doc_id, name=kpi.name, value=kpi.value, unit=kpi.unit, snippet=kpi.snippet))
            session.commit()
        self._rebuild_index()
        return {"doc_id": doc_id, "name": name, "chunks": len(chunks), "kpis": len(kpis), "duplicate": False}

    def ask(self, question: str, k: int = 5) -> Answer:
        answer = answer_question(question, self._retriever, llm=self._llm, k=k)
        top_score = answer.citations[0]["score"] if answer.citations else 0.0
        with self._session_factory() as session:
            session.add(QueryLog(question=question, answer=answer.text, mode=answer.mode, score=top_score))
            session.commit()
        return answer

    def list_documents(self) -> list[dict]:
        with self._session_factory() as session:
            return [{"doc_id": d.doc_id, "name": d.name} for d in session.query(Document).all()]

    def list_kpis(self, doc_id: str | None = None) -> list[dict]:
        with self._session_factory() as session:
            q = session.query(KPIRecord)
            if doc_id:
                q = q.filter_by(doc_id=doc_id)
            return [{"doc_id": r.doc_id, "name": r.name, "value": r.value, "unit": r.unit, "snippet": r.snippet} for r in q.all()]

    @property
    def chunk_count(self) -> int:
        return sum(len(store) for _, store, _ in self._indexes) // max(len(self._indexes), 1)
