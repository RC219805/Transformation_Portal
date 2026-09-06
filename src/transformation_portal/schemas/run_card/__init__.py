"""Runtime loader for packaged run-card schemas."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

SUPPORTED_RUN_CARD_VERSIONS = frozenset({"v1", "v2"})
RUN_CARD_SCHEMA_URIS = {
    "v1": "https://rc219805.github.io/Transformation_Portal/docs/schemas/run_card/run_card.v1.schema.json",
    "v2": "https://rc219805.github.io/Transformation_Portal/docs/schemas/run_card/run_card.v2.schema.json",
}


def _schema_filename(version: Any) -> str:
    normalized = normalize_run_card_version(version)
    return f"run_card.{normalized}.schema.json"


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


def get_run_card_schema_path(version: Any) -> Path:
    """Return the filesystem path of an installed packaged schema.

    Python wheels are installed as unpacked package files by supported install
    flows, so this is both an introspection aid and a backwards-compatible
    ``Path`` for callers that still accept an explicit schema override.
    """
    resource = files(__name__).joinpath(_schema_filename(version))
    schema_path = Path(str(resource))
    if not schema_path.is_file():
        raise RuntimeError(f"Packaged run-card schema is not filesystem-addressable: {resource}")
    return schema_path


@lru_cache(maxsize=len(SUPPORTED_RUN_CARD_VERSIONS))
def _load_run_card_schema_bytes(version: str) -> bytes:
    """Load immutable schema bytes once per normalized version."""
    return files(__name__).joinpath(_schema_filename(version)).read_bytes()


def load_run_card_schema(version: Any) -> dict[str, Any]:
    """Load a fresh packaged run-card schema payload.

    Only immutable source bytes are cached. Returning a newly decoded object
    prevents a caller from mutating process-global validation behavior.
    """
    normalized = normalize_run_card_version(version)
    return json.loads(_load_run_card_schema_bytes(normalized))
