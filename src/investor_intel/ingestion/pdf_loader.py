"""Document loading: markdown/text natively, PDF via pymupdf4llm (lazy import
so local development and CI do not need the PDF dependency installed)."""
from __future__ import annotations

import hashlib
from pathlib import Path


def load_document(path: str | Path) -> tuple[str, str]:
    """Return (doc_id, text). doc_id is a short hash of the file contents so
    re-ingestion of unchanged files is idempotent."""
    p = Path(path)
    raw = p.read_bytes()
    doc_id = hashlib.sha256(raw).hexdigest()[:12]
    if p.suffix.lower() == ".pdf":
        import pymupdf4llm  # lazy: optional dependency

        text = pymupdf4llm.to_markdown(str(p))
    else:
        text = raw.decode("utf-8", errors="replace")
    return doc_id, text
