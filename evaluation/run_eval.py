"""Retrieval evaluation against the golden QA set.

Ingests the sample filing into a fresh in-memory service, retrieves for each
golden question, and reports hit@k by expected section. Writes
evaluation/eval_report.md with the measured numbers so README claims are
reproducible. No API keys required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from investor_intel.config import get_settings
from investor_intel.service import InvestorIntelService

K = 3


def main() -> None:
    service = InvestorIntelService(get_settings(), database_url="sqlite:///:memory:")
    service.ingest(ROOT / "data" / "sample" / "meridian-industries-fy2025.md")
    golden = json.loads((ROOT / "evaluation" / "golden_qa.json").read_text())
    rows, hits = [], 0
    for item in golden:
        chunks = service.ask(item["question"], k=K).citations
        hit = any(c["heading"] == item["expected_heading"] for c in chunks)
        hits += int(hit)
        rows.append((item["question"], item["expected_heading"], "HIT" if hit else "MISS"))
    rate = hits / len(golden) if golden else 0.0
    lines = [
        "# Retrieval evaluation report",
        "",
        f"Corpus: data/sample/meridian-industries-fy2025.md | metric: hit@{K} by expected section",
        "",
        "Golden set: 6 self-authored questions over one fictional sample filing.",
        "This is a wiring smoke test, not an independent benchmark and not a",
        "measure of answer correctness or general accuracy.",
        "",
        "| Question | Expected section | Result |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {q} | {h} | {r} |" for q, h, r in rows]
    lines += ["", f"**hit@{K}: {hits}/{len(golden)} = {rate:.0%}**", ""]
    (ROOT / "evaluation" / "eval_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
