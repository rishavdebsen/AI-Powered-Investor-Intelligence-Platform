import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SAMPLE = ROOT / "data" / "sample" / "meridian-industries-fy2025.md"


@pytest.fixture
def sample_path() -> Path:
    return SAMPLE


@pytest.fixture
def sample_text() -> str:
    return SAMPLE.read_text()
