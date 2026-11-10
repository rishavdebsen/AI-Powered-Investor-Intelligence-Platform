from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from ..analytics import Analytics
from ..config import get_settings
from ..service import InvestorIntelService
from .schemas import AskRequest, AskResponse, IngestRequest, IngestResponse

API_DISTINCT_ID = "api-client"


def create_app(database_url: str | None = None, data_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="AI-Powered Investor Intelligence Platform", version="1.0.0")
    settings = get_settings()
    service = InvestorIntelService(settings, database_url=database_url, data_dir=data_dir)
    app.state.service = service
    analytics = Analytics(settings.mixpanel_project_token, api_host=settings.mixpanel_api_host)
    app.state.analytics = analytics

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "profile": service.settings.profile, "chunks": service.chunk_count}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest) -> dict:
        try:
            result = service.ingest(req.path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"file not found: {req.path}")
        analytics.track(
            API_DISTINCT_ID,
            "Report Uploaded",
            {
                "source": "api-path",
                "chunks": result.get("chunks", 0),
                "kpis": result.get("kpis", 0),
                "duplicate": bool(result.get("duplicate")),
            },
        )
        return result

    @app.post("/ingest/upload", response_model=IngestResponse)
    async def ingest_upload(file: UploadFile = File(...)) -> dict:  # noqa: B008 - FastAPI idiom
        uploads = service.data_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        dest = uploads / Path(file.filename or "upload.md").name  # basename only
        dest.write_bytes(await file.read())
        try:
            result = service.ingest(dest)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="upload could not be stored")
        analytics.track(
            API_DISTINCT_ID,
            "Report Uploaded",
            {
                "source": "api-upload",
                "file_type": dest.suffix.lower(),
                "chunks": result.get("chunks", 0),
                "kpis": result.get("kpis", 0),
                "duplicate": bool(result.get("duplicate")),
            },
        )
        return result

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest) -> dict:
        analytics.track(API_DISTINCT_ID, "Question Asked", {"source": "api", "question_length": len(req.question)})
        result = service.ask(req.question, k=req.k)
        analytics.track(
            API_DISTINCT_ID,
            "Answer Shown",
            {"source": "api", "citations_count": len(result.citations), "answer_length": len(result.text), "mode": result.mode},
        )
        return {"answer": result.text, "citations": result.citations, "retrieved": result.retrieved, "mode": result.mode}

    @app.get("/documents")
    def documents() -> list[dict]:
        return service.list_documents()

    @app.get("/kpis")
    def kpis(doc_id: str | None = None) -> list[dict]:
        return service.list_kpis(doc_id)

    return app


app = create_app()
