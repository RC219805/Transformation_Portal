"""Helpers for comparing filesystem paths without losing user-facing aliases.

macOS commonly exposes the same temporary directory through both ``/tmp`` and
``/private/tmp``. Replay artifacts should preserve the spelling the operator
supplied, while safety checks still need a canonical real path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def normalize_lexical_path(path_value: Any) -> Path:
    """Return an absolute path without resolving symlinks."""
    path = Path(path_value).expanduser()
    return Path(os.path.abspath(os.fspath(path)))


def resolve_real_path(path_value: Any) -> Path:
    """Return the canonical real path for safety checks."""
    return normalize_lexical_path(path_value).resolve(strict=False)


def relative_to_path_alias(candidate: Any, root: Any) -> Path:
    """Return ``candidate`` relative to ``root`` with alias-aware fallback."""
    candidate_path = normalize_lexical_path(candidate)
    root_path = normalize_lexical_path(root)
    try:
        return candidate_path.relative_to(root_path)
    except ValueError:
        return resolve_real_path(candidate_path).relative_to(resolve_real_path(root_path))


__all__ = [
    "normalize_lexical_path",
    "relative_to_path_alias",
    "resolve_real_path",
]
