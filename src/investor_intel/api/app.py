from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from ..config import get_settings
from ..service import InvestorIntelService
from .schemas import AskRequest, AskResponse, IngestRequest, IngestResponse


def create_app(database_url: str | None = None, data_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="AI-Powered Investor Intelligence Platform", version="1.0.0")
    service = InvestorIntelService(get_settings(), database_url=database_url, data_dir=data_dir)
    app.state.service = service

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "profile": service.settings.profile, "chunks": service.chunk_count}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest) -> dict:
        try:
            return service.ingest(req.path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"file not found: {req.path}")

    @app.post("/ingest/upload", response_model=IngestResponse)
    async def ingest_upload(file: UploadFile = File(...)) -> dict:  # noqa: B008 - FastAPI idiom
        uploads = service.data_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        dest = uploads / Path(file.filename or "upload.md").name  # basename only
        dest.write_bytes(await file.read())
        try:
            return service.ingest(dest)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="upload could not be stored")

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> dict:
        result = service.ask(req.question, k=req.k)
        return {"answer": result.text, "citations": result.citations, "retrieved": result.retrieved, "mode": result.mode}

    @app.get("/documents")
    def documents() -> list[dict]:
        return service.list_documents()

    @app.get("/kpis")
    def kpis(doc_id: str | None = None) -> list[dict]:
        return service.list_kpis(doc_id)

    return app


app = create_app()
