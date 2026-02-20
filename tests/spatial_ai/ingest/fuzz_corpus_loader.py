from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "ingest_fuzz_cases"


def load_corpus_cases() -> list[dict[str, Any]]:
    """Load all JSON corpus cases from the ingest_fuzz_cases directory.

    Raises:
        RuntimeError: If the corpus directory is missing or contains no cases.
            This ensures CI fails loudly rather than silently collecting zero tests.
    """
    if not _CORPUS_DIR.exists():
        raise RuntimeError(
            f"Ingest fuzz corpus directory missing: {_CORPUS_DIR}\n"
            "Ensure tests/data/ingest_fuzz_cases/ is present in the repository."
        )

    cases: list[dict[str, Any]] = []
    for p in sorted(_CORPUS_DIR.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            cases.append(json.load(f))

    if not cases:
        raise RuntimeError(
            f"Ingest fuzz corpus is empty: {_CORPUS_DIR}\n"
            "Add at least one *.json corpus case to preserve firewall coverage."
        )

    return cases
