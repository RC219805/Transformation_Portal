"""Production-safe stub material segmentation backend."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from ..protocols.segmentation_backend import SegmentationBackendInfo

logger = logging.getLogger(__name__)


class StubBackend:
    """Stub segmentation backend that returns empty masks.

    This is the default backend to avoid heavy ML dependencies.
    It's production-safe and will never fail, but provides no segmentation.
    """

    @property
    def info(self) -> SegmentationBackendInfo:
        return SegmentationBackendInfo(
            name="Stub Segmentation Backend",
            model_id="stub",
            requires_gpu=False,
            requires_weights=False,
            approximate_memory_mb=0,
            description="Stub backend that returns empty masks (production-safe default)",
        )

    def load(self, device: str = "auto", weights_path: Optional[Path] = None) -> None:
        """No-op load for stub backend."""
        logger.debug("StubBackend.load() called - no model to load")

    def segment(self, image: np.ndarray) -> Dict[str, Tuple[np.ndarray, float]]:
        """Return empty masks dict."""
        logger.debug("StubBackend.segment() returning empty masks")
        return {}
