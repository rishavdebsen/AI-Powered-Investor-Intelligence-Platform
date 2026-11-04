from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(512))
    ingested_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))


class ChunkRow(Base):
    __tablename__ = "chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    doc_id: Mapped[str] = mapped_column(String(32), index=True)
    heading: Mapped[str] = mapped_column(String(512), default="")
    text: Mapped[str] = mapped_column(Text)


class KPIRecord(Base):
    __tablename__ = "kpis"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(32), index=True)
    name: Mapped[str] = mapped_column(String(128))
    value: Mapped[str] = mapped_column(String(64))
    unit: Mapped[str] = mapped_column(String(32), default="")
    snippet: Mapped[str] = mapped_column(Text, default="")


class QueryLog(Base):
    __tablename__ = "query_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text)
    mode: Mapped[str] = mapped_column(String(16), default="extractive")
    score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
