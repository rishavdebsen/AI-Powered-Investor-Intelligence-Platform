"""Compare keyword (TF-IDF) retrieval with semantic (sentence-transformers)
retrieval on the sample filings, using paraphrase-heavy test questions.

Usage:
    python evaluation/compare_backends.py            # TF-IDF only if
                                                    # sentence-transformers missing
    python evaluation/compare_backends.py --json out.json

A question counts as retrieved at rank k when one of the top-k chunks comes
from a gold section (e.g. "Outlook" for "what is management expecting next
year?"). Gold sections were chosen by reading the sample filings.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from investor_intel.embeddings.local_tfidf import TfidfEmbedder
from investor_intel.ingestion.chunker import chunk_document
from investor_intel.rag.hybrid import HybridRetriever
from investor_intel.rag.retriever import Retriever
from investor_intel.vectorstore.dense import InMemoryDenseVectorStore
from investor_intel.vectorstore.memory import InMemoryVectorStore

SAMPLE_DIR = ROOT / "data" / "sample"

# (question, gold headings, why it is in the set)
QUESTIONS = [
    ("How fast is the top line growing?", {"Financial Highlights"},
     "paraphrase: 'top line' never appears in the filings"),
    ("What is management expecting for next fiscal year?", {"Outlook"},
     "no 'outlook' keyword in the question"),
    ("Is the company exposed to supply-chain problems?", {"Risk Factors"},
     "filings say 'suppliers', not 'supply-chain'"),
    ("How does the company make money?", {"Business Overview"},
     "asks for the business description in plain words"),
    ("What could hurt profitability?", {"Risk Factors"},
     "margin threats live in Risk Factors, phrased differently"),
    ("Where does the company plan to expand?", {"Outlook", "Growth Drivers"},
     "expansion plans across two possible sections"),
    ("Which parts of the business bring in the most sales?", {"Segment Results"},
     "'segments' never appears in the question"),
    ("Does the company depend on a small number of vendors?", {"Risk Factors"},
     "'vendors' vs filings' 'suppliers'"),
    ("What was total revenue in fiscal 2025?", {"Financial Highlights"},
     "direct keyword question - TF-IDF should already do well"),
    ("How much net income did the company report?", {"Financial Highlights"},
     "direct keyword question - TF-IDF should already do well"),
]


def build_corpus() -> list[tuple[str, dict]]:
    corpus: list[tuple[str, dict]] = []
    for path in sorted(SAMPLE_DIR.glob("*.md")):
        doc_id = path.stem
        text = path.read_text()
        for c in chunk_document(doc_id, text):
            corpus.append((
                (c.heading + "\n" + c.text) if c.heading else c.text,
                {"text": c.text, "doc_id": doc_id, "chunk_id": c.chunk_id,
                 "heading": c.heading, "source": path.name},
            ))
    return corpus


def make_tfidf_retriever(corpus):
    emb = TfidfEmbedder()
    emb.fit([text for text, _ in corpus])
    store = InMemoryVectorStore()
    for text, meta in corpus:
        store.add(emb.embed(text), meta)
    return Retriever(store, emb)


def make_semantic_retriever(corpus, model_name: str):
    from investor_intel.embeddings.sentence_transformer_embedder import (
        SentenceTransformerEmbedder,
    )

    emb = SentenceTransformerEmbedder(model_name)
    emb.fit([text for text, _ in corpus])
    store = InMemoryDenseVectorStore()
    for text, meta in corpus:
        store.add(emb.embed(text), meta)
    return Retriever(store, emb)


def evaluate(name: str, retriever: Retriever) -> dict:
    rows = []
    for question, gold, note in QUESTIONS:
        hits = retriever.retrieve(question, k=5)  # k=5 matches the app default (service.ask)
        headings = [h.heading for h in hits]
        rank = next((i + 1 for i, h in enumerate(hits) if h.heading in gold), None)
        rows.append({
            "question": question,
            "gold_sections": sorted(gold),
            "top5_sections": headings,
            "hit_at_1": rank == 1,
            "hit_at_3": rank is not None and rank <= 3,
            "hit_at_5": rank is not None,
            "reciprocal_rank": round(1.0 / rank, 4) if rank else 0.0,
            "note": note,
        })
    n = len(rows)
    return {
        "backend": name,
        "recall_at_1": round(sum(r["hit_at_1"] for r in rows) / n, 4),
        "recall_at_3": round(sum(r["hit_at_3"] for r in rows) / n, 4),
        "recall_at_5": round(sum(r["hit_at_5"] for r in rows) / n, 4),
        "mrr": round(sum(r["reciprocal_rank"] for r in rows) / n, 4),
        "questions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="write full results to this JSON file")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    corpus = build_corpus()
    print(f"Corpus: {len(corpus)} chunks from {len(list(SAMPLE_DIR.glob('*.md')))} sample filings\n")

    results = [evaluate("tfidf", make_tfidf_retriever(corpus))]
    try:
        semantic = make_semantic_retriever(corpus, args.model)
        results.append(evaluate(f"semantic ({args.model})", semantic))
        tfidf = make_tfidf_retriever(corpus)
        results.append(evaluate("hybrid (tfidf 0.5 + semantic 0.5)",
                                HybridRetriever([(tfidf, 0.5), (semantic, 0.5)])))
    except RuntimeError as exc:
        print(f"Skipping semantic backend: {exc}\n")

    for res in results:
        print(f"== {res['backend']}  recall@1={res['recall_at_1']:.2f} "
              f"recall@3={res['recall_at_3']:.2f} recall@5={res['recall_at_5']:.2f} MRR={res['mrr']:.2f}")
    print()
    for i, (question, gold, note) in enumerate(QUESTIONS):
        marks = "  ".join(
            f"[{res['backend'].split()[0]}: rank "
            + (str(next((j + 1 for j, h in enumerate(res['questions'][i]['top5_sections']) if h in gold), '-')) + "]")
            for res in results
        )
        print(f"{question}\n    gold={sorted(gold)}  {marks}")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"\nFull results written to {args.json}")


if __name__ == "__main__":
    main()
