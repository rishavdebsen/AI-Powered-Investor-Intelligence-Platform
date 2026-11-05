# Retrieval evaluation report

Golden sets are authored for this project against fictional sample
filings. These are wiring smoke tests, not an independent benchmark
and not a measure of answer correctness or general accuracy.

## Suite 1: single filing, section retrieval

Corpus: data/sample/meridian-industries-fy2025.md | metric: hit@3 by expected section

Golden set: 6 self-authored questions over one fictional sample filing.

| Question | Expected section | Result |
| --- | --- | --- |
| What was total revenue in fiscal 2025? | Financial Highlights | HIT |
| How fast did revenue grow year over year? | Financial Highlights | HIT |
| What was diluted EPS? | Financial Highlights | HIT |
| Which segment generated the most revenue? | Segment Results | HIT |
| What supply chain risks does the company face? | Risk Factors | MISS |
| What does management expect for fiscal 2026? | Outlook | HIT |

**hit@3: 5/6 = 83%**


## Suite 2: three filings, cross-company ranking

Corpus: all three filings in data/sample/ | metrics: hit@1 and hit@3 by expected company

Golden set: 6 self-authored questions naming a specific company.
Measures whether a question naming one company ranks that company's
filing first even when all filings share financial boilerplate.

| Question | Expected company | Result |
| --- | --- | --- |
| What was Meridian Industries total revenue in fiscal 2025? | meridian-industries-fy2025 | HIT |
| What was Vertex Mobility total revenue in fiscal 2025? | vertex-mobility-fy2025 | HIT |
| What was Aurora Retail Group total revenue in fiscal 2025? | aurora-retail-group-fy2025 | HIT |
| What drove Vertex Mobility revenue growth in fiscal 2025? | vertex-mobility-fy2025 | HIT |
| What are the main risk factors for Aurora Retail Group? | aurora-retail-group-fy2025 | HIT |
| What does Meridian Industries expect for fiscal 2026? | meridian-industries-fy2025 | HIT |

**hit@1: 6/6 = 100% | hit@3: 6/6 = 100%**
