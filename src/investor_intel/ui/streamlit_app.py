"""Streamlit UI: chat with the filings + a KPI dashboard. Talks to the FastAPI
backend at API_URL (default http://localhost:8000)."""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Investor Intelligence", layout="wide")
st.title("AI-Powered Investor Intelligence Platform")

tab_chat, tab_kpis, tab_docs = st.tabs(["Ask the filings", "KPI dashboard", "Documents"])

with tab_chat:
    question = st.text_input("Ask a question about the ingested filings")
    if question:
        resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=60).json()
        st.text(resp["answer"])  # plain text: $ figures must not trigger LaTeX
        with st.expander("Citations"):
            for c in resp["citations"]:
                st.caption(f"{c['source']} - {c['heading']} (score {c['score']})")

with tab_kpis:
    kpis = requests.get(f"{API_URL}/kpis", timeout=30).json()
    if kpis:
        st.dataframe(kpis, use_container_width=True)
    else:
        st.info("No KPIs extracted yet. Ingest a filing first.")

with tab_docs:
    docs = requests.get(f"{API_URL}/documents", timeout=30).json()
    if docs:
        st.dataframe(docs, use_container_width=True)
    else:
        st.info("No documents ingested yet.")
