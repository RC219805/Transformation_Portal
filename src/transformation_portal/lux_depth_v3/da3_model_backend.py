"""Retired DA3 model backend shim.

The old direct backend returned placeholder zero depth and is intentionally
unavailable. Runtime DA3 selection must flow through the registry-resolved DA3
API backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DA3ModelBackendConfig:
    """Deprecated compatibility shell for legacy imports."""

    model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    device: str = "cpu"
    dtype: str = "float32"
    max_side: int = 896
    cache_dir: Optional[Path] = None


class DA3ModelBackend:
    """Retired backend shim that fails closed."""

    def __init__(self, config: DA3ModelBackendConfig):
        raise RuntimeError(
            "DA3ModelBackend stub is retired and must not be used. " "Use the registry-resolved DA3 API backend."
        )
