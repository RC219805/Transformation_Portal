from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "ingest_fuzz_cases"


def load_corpus_cases() -> list[dict[str, Any]]:
    """Load all JSON corpus cases from the ingest_fuzz_cases directory."""
    cases: list[dict[str, Any]] = []
    for p in sorted(_CORPUS_DIR.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            cases.append(json.load(f))
    return cases
