"""Retrieval evaluation against the golden QA sets.

Suite 1 ingests one sample filing and reports hit@k by expected section.
Suite 2 ingests all three sample filings and reports hit@1 and hit@k by
expected company (source file) - it measures the cross-company ranking fix
in rag/retriever.py. Writes evaluation/eval_report.md with the measured
numbers so README claims are reproducible. No API keys required.
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
SAMPLE = ROOT / "data" / "sample"


def suite_single_filing() -> list[str]:
    service = InvestorIntelService(get_settings(), database_url="sqlite:///:memory:")
    service.ingest(SAMPLE / "meridian-industries-fy2025.md")
    golden = json.loads((ROOT / "evaluation" / "golden_qa.json").read_text())
    rows, hits = [], 0
    for item in golden:
        chunks = service.ask(item["question"], k=K).citations
        hit = any(c["heading"] == item["expected_heading"] for c in chunks)
        hits += int(hit)
        rows.append((item["question"], item["expected_heading"], "HIT" if hit else "MISS"))
    rate = hits / len(golden) if golden else 0.0
    lines = [
        "## Suite 1: single filing, section retrieval",
        "",
        "Corpus: data/sample/meridian-industries-fy2025.md | metric: hit@%d by expected section" % K,
        "",
        "Golden set: 6 self-authored questions over one fictional sample filing.",
        "",
        "| Question | Expected section | Result |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {q} | {h} | {r} |" for q, h, r in rows]
    lines += ["", f"**hit@{K}: {hits}/{len(golden)} = {rate:.0%}**", ""]
    return lines


def suite_cross_company() -> list[str]:
    service = InvestorIntelService(get_settings(), database_url="sqlite:///:memory:")
    for f in sorted(SAMPLE.glob("*.md")):
        service.ingest(f)
    golden = json.loads((ROOT / "evaluation" / "golden_cross_company.json").read_text())
    rows, top1, hits = [], 0, 0
    for item in golden:
        chunks = service.ask(item["question"], k=K).citations
        sources = [c["source"].removesuffix(".md") for c in chunks]
        want = item["expected_doc_id"]
        h1 = bool(sources) and sources[0] == want
        hk = want in sources
        top1 += int(h1)
        hits += int(hk)
        rows.append((item["question"], want, "HIT" if h1 else ("top-%d only" % K if hk else "MISS")))
    n = len(golden) or 1
    lines = [
        "## Suite 2: three filings, cross-company ranking",
        "",
        "Corpus: all three filings in data/sample/ | metrics: hit@1 and hit@%d by expected company" % K,
        "",
        "Golden set: %d self-authored questions naming a specific company." % len(golden),
        "Measures whether a question naming one company ranks that company's",
        "filing first even when all filings share financial boilerplate.",
        "",
        "| Question | Expected company | Result |",
        "| --- | --- | --- |",
    ]
    lines += [f"| {q} | {w} | {r} |" for q, w, r in rows]
    lines += ["", f"**hit@1: {top1}/{len(golden)} = {top1/n:.0%} | hit@{K}: {hits}/{len(golden)} = {hits/n:.0%}**", ""]
    return lines


def main() -> None:
    lines = [
        "# Retrieval evaluation report",
        "",
        "Golden sets are authored for this project against fictional sample",
        "filings. These are wiring smoke tests, not an independent benchmark",
        "and not a measure of answer correctness or general accuracy.",
        "",
    ]
    lines += suite_single_filing() + [""] + suite_cross_company() + [""]
    (ROOT / "evaluation" / "eval_report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
