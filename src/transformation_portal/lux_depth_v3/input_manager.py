"""Input modeling helpers for Lux Depth V3 batch execution.

This module keeps the lightweight input records shared by discovery,
grouping, and orchestrator scheduling paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ImageInput:
    """Represents an input image for processing."""

    path: Path
    metadata: Optional[dict] = None

    def __post_init__(self) -> None:
        """Ensure path is a Path object."""
        if not isinstance(self.path, Path):
            self.path = Path(self.path)
