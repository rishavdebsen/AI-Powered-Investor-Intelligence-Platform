from fastapi.testclient import TestClient

from investor_intel.api.app import create_app


def make_client():
    return TestClient(create_app(database_url="sqlite:///:memory:"))


def test_health():
    resp = make_client().get("/health")
    assert resp.status_code == 200 and resp.json()["status"] == "ok"


def test_ingest_ask_kpis_flow(sample_path):
    client = make_client()
    ingest = client.post("/ingest", json={"path": str(sample_path)})
    assert ingest.status_code == 200
    body = ingest.json()
    assert body["chunks"] > 0 and body["kpis"] > 0

    dup = client.post("/ingest", json={"path": str(sample_path)}).json()
    assert dup["duplicate"] is True

    ask = client.post("/ask", json={"question": "What was total revenue?"}).json()
    assert ask["retrieved"] > 0 and ask["citations"]
    assert ask["citations"][0]["source"].endswith(".md")

    kpis = client.get("/kpis").json()
    assert any(k["name"] == "revenue" for k in kpis)

    docs = client.get("/documents").json()
    assert len(docs) == 1


def test_ingest_missing_file_is_404():
    assert make_client().post("/ingest", json={"path": "data/nope.md"}).status_code == 404


def test_ingest_outside_data_dir_is_400():
    assert make_client().post("/ingest", json={"path": "/etc/passwd"}).status_code == 400


def test_upload_endpoint_ingests_file(sample_path):
    client = make_client()
    resp = client.post("/ingest/upload", files={"file": (sample_path.name, sample_path.read_bytes())})
    assert resp.status_code == 200
    assert resp.json()["chunks"] > 0
