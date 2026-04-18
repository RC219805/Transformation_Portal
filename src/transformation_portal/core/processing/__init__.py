"""
Processing utilities for transformation pipelines.

Provides efficient processing strategies for large images.
Compatibility note: retained as an internal/shared helper surface with
direct smoke coverage, but it currently has no production imports.
"""

from .tiling import TileConfig, TiledProcessor

__all__ = [
    "TiledProcessor",
    "TileConfig",
]
