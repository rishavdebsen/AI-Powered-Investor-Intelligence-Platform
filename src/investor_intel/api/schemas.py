from __future__ import annotations

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    path: str = Field(..., description="Server-side path to a .md, .txt or .pdf document")


class IngestResponse(BaseModel):
    doc_id: str
    name: str
    chunks: int
    kpis: int
    duplicate: bool


class AskRequest(BaseModel):
    question: str
    k: int = 5


class Citation(BaseModel):
    source: str
    heading: str
    chunk_id: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved: int
    mode: str
