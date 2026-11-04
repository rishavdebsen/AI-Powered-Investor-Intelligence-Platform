"""PDF ingestion, verified when pymupdf4llm is installed (requirements-full).
CI installs only requirements-dev, so this skips there; it is run locally."""
import pytest

pymupdf = pytest.importorskip("pymupdf", reason="PDF support needs requirements-full.txt")
pytest.importorskip("pymupdf4llm", reason="PDF support needs requirements-full.txt")

from investor_intel.config import get_settings
from investor_intel.service import InvestorIntelService


def test_pdf_ingest_roundtrip(tmp_path):
    pdf = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Financial Highlights\n\n"
        "Total revenue of $4.2 billion for fiscal year 2025. "
        "Net income was $742 million. Diluted earnings per share was $5.16. "
        "Operating margin was 23%.",
        fontsize=11,
    )
    doc.save(str(pdf))
    doc.close()

    service = InvestorIntelService(get_settings(), database_url="sqlite:///:memory:", data_dir=tmp_path)
    result = service.ingest(pdf)
    assert result["chunks"] > 0 and result["kpis"] > 0
    answer = service.ask("What was total revenue?")
    assert answer.retrieved > 0
    kpis = {k["name"]: k for k in service.list_kpis()}
    assert kpis["revenue"]["value"] == "4.2"
