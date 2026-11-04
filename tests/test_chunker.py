import pytest

from investor_intel.ingestion.chunker import chunk_document, split_sections


def test_sections_split_on_headings(sample_text):
    sections = split_sections(sample_text)
    headings = [h for h, _ in sections]
    assert "Financial Highlights" in headings
    assert "Risk Factors" in headings
    assert len(sections) >= 5


def test_chunks_carry_heading_and_stable_ids(sample_text):
    chunks = chunk_document("doc1", sample_text)
    assert chunks
    assert chunks[0].chunk_id == "doc1:0"
    assert all(c.doc_id == "doc1" for c in chunks)
    assert any(c.heading == "Risk Factors" for c in chunks)


def test_long_paragraph_splits_within_max_chars():
    text = "# Big\n\n" + ("sensor demand grew strongly. " * 200)
    chunks = chunk_document("doc2", text, max_chars=600, overlap=60)
    assert len(chunks) > 1
    assert all(len(c.text) <= 700 for c in chunks)


def test_overlap_must_be_smaller_than_max_chars():
    with pytest.raises(ValueError):
        chunk_document("doc3", "text", max_chars=100, overlap=100)
