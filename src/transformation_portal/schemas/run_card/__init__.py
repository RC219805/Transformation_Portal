"""Runtime loader for packaged run-card schemas."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

SUPPORTED_RUN_CARD_VERSIONS = frozenset({"v1", "v2"})
RUN_CARD_SCHEMA_URIS = {
    "v1": "https://rc219805.github.io/Transformation_Portal/docs/schemas/run_card/run_card.v1.schema.json",
    "v2": "https://rc219805.github.io/Transformation_Portal/docs/schemas/run_card/run_card.v2.schema.json",
}


def normalize_run_card_version(version: Any, *, default: str = "v1") -> str:
    """Normalize a supported run-card version identifier."""
    normalized = str(version or default).strip().lower()
    if normalized not in SUPPORTED_RUN_CARD_VERSIONS:
        raise ValueError(f"Unsupported run card schema version: {version!r}")
    return normalized


def get_run_card_schema_uri(version: Any) -> str:
    """Return the canonical schema URI for a supported run-card version."""
    normalized = normalize_run_card_version(version)
    return RUN_CARD_SCHEMA_URIS[normalized]


@lru_cache(maxsize=len(SUPPORTED_RUN_CARD_VERSIONS))
def load_run_card_schema(version: Any) -> dict[str, Any]:
    """Load a packaged run-card schema payload."""
    normalized = normalize_run_card_version(version)
    schema_path = files(__name__).joinpath(f"run_card.{normalized}.schema.json")
    return json.loads(schema_path.read_text(encoding="utf-8"))
