# Retrieval evaluation report

Corpus: data/sample/meridian-industries-fy2025.md | metric: hit@3 by expected section

Golden set: 6 self-authored questions over one fictional sample filing.
This is a wiring smoke test, not an independent benchmark and not a
measure of answer correctness or general accuracy.

| Question | Expected section | Result |
| --- | --- | --- |
| What was total revenue in fiscal 2025? | Financial Highlights | HIT |
| How fast did revenue grow year over year? | Financial Highlights | HIT |
| What was diluted EPS? | Financial Highlights | HIT |
| Which segment generated the most revenue? | Segment Results | HIT |
| What supply chain risks does the company face? | Risk Factors | HIT |
| What does management expect for fiscal 2026? | Outlook | HIT |

**hit@3: 6/6 = 100%**
