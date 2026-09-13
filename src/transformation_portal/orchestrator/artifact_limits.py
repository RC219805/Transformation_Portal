"""Shared startup count limit for artifact indexing, export, and publication."""

from __future__ import annotations

import os

DEFAULT_MAX_INDEXED_ARTIFACTS = 200


def configured_max_indexed_artifacts() -> int:
    """Preserve the portal's default, invalid-value fallback, and minimum of one."""
    raw = os.getenv("TP_MAX_INDEXED_ARTIFACTS")
    if raw is None:
        return DEFAULT_MAX_INDEXED_ARTIFACTS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_MAX_INDEXED_ARTIFACTS
