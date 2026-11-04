"""KPI extraction from financial text.

Regex-first extraction keeps the local profile fully testable without an LLM.
An LLM-based extractor can be layered on top for messier documents; the regex
layer also doubles as a sanity check on LLM output.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERNS: dict[str, str] = {
    "revenue": r"(?:total\s+)?revenues?\s+(?:of|was|were|reached|grew\s+to)\s+\$?([\d,.]*\d)\s*(billion|million|thousand)?",
    "net_income": r"net\s+income\s+(?:of|was|were|reached)\s+\$?([\d,.]*\d)\s*(billion|million|thousand)?",
    "operating_income": r"operating\s+income\s+(?:of|was|were|reached)\s+\$?([\d,.]*\d)\s*(billion|million|thousand)?",
    "diluted_eps": r"diluted\s+(?:earnings\s+per\s+share|EPS)[^\d]{0,20}\$?([\d,.]*\d)",
    "operating_margin": r"operating\s+margin\s+(?:of|was|reached)\s+([\d.]+)\s*%",
    "yoy_revenue_growth": r"revenues?\s+(?:grew|increased|rose)\s+([\d.]+)\s*%\s*(?:year[- ]over[- ]year|YoY)?",
}


@dataclass
class KPI:
    name: str
    value: str
    unit: str
    snippet: str


def extract_kpis(text: str) -> list[KPI]:
    found: list[KPI] = []
    for name, pattern in _PATTERNS.items():
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            unit = (m.group(2) if m.lastindex and m.lastindex >= 2 and m.group(2) else "%" if "%" in m.group(0) else "").strip()
            found.append(KPI(name=name, value=m.group(1), unit=unit, snippet=m.group(0).strip()))
    return found
