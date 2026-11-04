"""Structure-aware chunking for markdown-ish financial documents.

Splits on headings first, then packs paragraphs into chunks bounded by
max_chars with a small character overlap for long paragraphs. Chunk IDs are
stable (doc_id + ordinal) so re-ingesting the same file is reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    heading: str
    ordinal: int


def split_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (heading, body) pairs. Body without any
    heading is returned under an empty heading."""
    sections: list[tuple[str, str]] = []
    heading, buf = "", []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            if any(l.strip() for l in buf):
                sections.append((heading, "\n".join(buf).strip()))
            buf = []
            heading = stripped.lstrip("#").strip()
        else:
            buf.append(line)
    if any(l.strip() for l in buf):
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]


def chunk_document(doc_id: str, text: str, max_chars: int = 1200, overlap: int = 150) -> list[Chunk]:
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    chunks: list[Chunk] = []
    ordinal = 0
    for heading, body in split_sections(text):
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
        current = ""
        for para in paragraphs:
            if current and len(current) + len(para) + 2 > max_chars:
                chunks.append(Chunk(f"{doc_id}:{ordinal}", doc_id, current, heading, ordinal))
                ordinal += 1
                current = ""
            while len(para) > max_chars:
                piece, para = para[:max_chars], para[max_chars - overlap:]
                if current:
                    piece = current + "\n\n" + piece
                    current = ""
                chunks.append(Chunk(f"{doc_id}:{ordinal}", doc_id, piece, heading, ordinal))
                ordinal += 1
            current = (current + "\n\n" + para).strip() if current else para
        if current:
            chunks.append(Chunk(f"{doc_id}:{ordinal}", doc_id, current, heading, ordinal))
            ordinal += 1
    return chunks
