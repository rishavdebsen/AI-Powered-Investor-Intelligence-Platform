"""Standalone Streamlit demo entrypoint (e.g. Streamlit Community Cloud).

Runs the platform in-process with the local TF-IDF retrieval profile - no
separate FastAPI server and no API keys required. The reviewed deployment
remains the FastAPI + Streamlit split described in the README; this additive
entrypoint exists so the project can be demoed as a single free Streamlit app.
It pre-loads the bundled sample filing on first run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st

from investor_intel.config import get_settings
from investor_intel.service import InvestorIntelService

SAMPLE_FILING = Path(__file__).resolve().parent / "data" / "sample" / "meridian-industries-fy2025.md"


@st.cache_resource
def get_service() -> InvestorIntelService:
    service = InvestorIntelService(get_settings())
    if SAMPLE_FILING.exists():
        service.ingest(SAMPLE_FILING)  # pre-load the bundled sample filing
    return service


service = get_service()

st.set_page_config(page_title="Investor Intelligence", layout="wide")
st.title("AI-Powered Investor Intelligence Platform")
st.caption("Standalone demo: in-process local retrieval (TF-IDF), pre-loaded with the bundled sample filing.")

tab_chat, tab_kpis, tab_docs = st.tabs(["Ask the filings", "KPI dashboard", "Documents"])

with tab_chat:
    question = st.text_input("Ask a question about the ingested filings")
    if question:
        answer = service.ask(question)
        st.text(answer.text)  # plain text: $ figures must not trigger LaTeX
        with st.expander("Citations"):
            for c in answer.citations:
                st.caption(f"{c['source']} - {c['heading']} (score {c['score']})")

with tab_kpis:
    kpis = service.list_kpis()
    if kpis:
        st.dataframe(kpis, use_container_width=True)
    else:
        st.info("No KPIs extracted yet. Ingest a filing first.")

with tab_docs:
    docs = service.list_documents()
    if docs:
        st.dataframe(docs, use_container_width=True)
    else:
        st.info("No documents ingested yet.")
