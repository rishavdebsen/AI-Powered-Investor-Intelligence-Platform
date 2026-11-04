# AI-Powered Investor Intelligence Platform

Ask questions of company filings and get answers with citations, extract
headline KPIs automatically, and review everything in a dashboard.

**Current verified state:** the local mode works end to end - lexical
(TF-IDF keyword) retrieval with extractive answers that quote the retrieved
excerpts. Semantic embeddings and LLM-written answers (OpenAI / Azure OpenAI)
are implemented behind configuration but are **not yet verified** - no API
key has been run against them. See "Honest status" below.

## Architecture

```
filings (.md/.txt/.pdf)
      |
      v
 ingestion (structure-aware chunking) ---> KPI extraction ---> SQLite / PostgreSQL
      |
      v
 embeddings + vector store
   (local TF-IDF - verified | OpenAI embeddings - optional, unverified)
      |
      v
 retriever ---> answer chain ---> FastAPI (/ask, /ingest, /kpis) ---> Streamlit UI
                     |
              local mode: quotes retrieved excerpts verbatim
              LLM mode (unverified): instructs the model to use only the
              retrieved context - reduces, but cannot eliminate, the risk of
              invented content
```

Every answer carries citations (source file, section, retrieval score).

## Quickstart (local, no API keys needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
uvicorn investor_intel.api.app:app --reload   # API on :8000
streamlit run src/investor_intel/ui/streamlit_app.py  # UI on :8501
```

Ingest the sample filing and ask a question:

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"path": "data/sample/meridian-industries-fy2025.md"}'
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "What was total revenue in fiscal 2025?"}'
```

Server-side ingestion is restricted to the configured data directory
(`II_DATA_DIR`, default `./data`); other paths are rejected. Use
`POST /ingest/upload` (multipart) for arbitrary files. PDF ingestion needs
`requirements-full.txt`. Set `LLM_PROVIDER=openai` (or `azure-openai`) and
`II_EMBEDDINGS_PROVIDER=openai` with the variables in `.env.example` to
switch on the (unverified) semantic/LLM mode.

## Evaluation

`python evaluation/run_eval.py` ingests the sample filing and measures
retrieval hit@3 (did any top-3 chunk come from the expected section?),
writing `evaluation/eval_report.md`. Current measured result: **6/6**.

Scope of that number: the golden set is 6 questions authored for this
project against one fictional sample filing. It is a wiring smoke test, not
an independent benchmark, and not a measure of answer correctness or general
accuracy. Chunk text is embedded together with its section heading, which
closes vocabulary gaps like "supply chain" vs "semiconductor suppliers".

## Testing

`pytest -q` - 15 tests covering chunking, embeddings, retrieval, KPI
extraction, the API flow, ingest path sandboxing, file upload, and PDF
ingestion. CI (`.github/workflows/ci.yml`) installs the full dependency set
(so the PDF test runs there too), then lints, tests, evaluates and builds
both Docker images on every push and PR.

## Deployment

- `docker-compose.yml` runs api + ui locally.
- `docker/api.Dockerfile`, `docker/ui.Dockerfile` build the two services.
- `.github/workflows/cd-aks.yml` (manual trigger only) builds, pushes to
  Azure Container Registry and applies `k8s/` to AKS. It needs repo
  secrets/vars (`AZURE_CREDENTIALS`, `ACR_NAME`, `AKS_CLUSTER`,
  `AKS_RESOURCE_GROUP`) and a pre-created `investor-intel-secrets`
  kubernetes secret - see `k8s/secrets.example.yaml`. Deploying creates paid
  Azure resources.

## Honest status

- Verified locally (no paid services): 15/15 tests pass; retrieval smoke
  eval 6/6 on the small project-authored golden set; ingest -> ask -> KPI
  flow works end to end on the sample filing without API keys; PDF ingestion
  verified with `pymupdf4llm` on a generated test PDF.
- Retrieval today is lexical (TF-IDF keyword matching). Semantic embeddings
  and LLM-written answers are wired behind the OpenAI / Azure OpenAI
  configuration but are NOT verified - no API key has been run against them.
- Docker images are not built locally (no Docker daemon in the dev
  environment); CI builds them on every push, which is their first real
  verification.
- AKS deployment is untested, needs an Azure subscription, and creates paid
  resources.
- The sample filing in `data/sample/` is fictional and exists for testing.
  The evaluation golden set was authored for this project against that
  filing; it is not independent evidence of quality.

## Acknowledgement

The project concept (investor-filings RAG with KPI extraction on Azure) was
inspired by
[Sandesh-hase/AI-Powered-Investor-Intelligence-Platform](https://github.com/Sandesh-hase/AI-Powered-Investor-Intelligence-Platform).
That repository has no license file, so there is no established permission
to reuse its code; this codebase was implemented independently from scratch
rather than copied.

## License

MIT - see [LICENSE](LICENSE).
