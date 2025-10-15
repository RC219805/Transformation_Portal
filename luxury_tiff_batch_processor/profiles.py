"""Processing profile definitions for balancing fidelity and throughput."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ProcessingProfile:
    """Describe processing trade-offs for a pipeline run."""

    name: str
    glow_multiplier: float
    chroma_denoise_multiplier: float
    target_bit_depth: int | None
    compression: str | None

    def resolve_glow(self, value: float) -> float:
        """Return the adjusted glow amount for this profile."""

        return value * self.glow_multiplier

    def resolve_chroma_denoise(self, value: float) -> float:
        """Return the adjusted chroma denoise amount for this profile."""

        return value * self.chroma_denoise_multiplier

    def target_dtype(self, source_dtype: np.dtype) -> np.dtype:
        """Return the dtype that should be used for saving results."""

        if self.target_bit_depth is None:
            return np.dtype(source_dtype)

        if self.target_bit_depth >= 16:
            return np.dtype(np.uint16)

        return np.dtype(np.uint8)

    def resolve_compression(self, requested: str) -> str:
        """Return the compression that should be used for this profile."""

        return self.compression or requested


DEFAULT_PROFILE_NAME = "quality"

PROCESSING_PROFILES: Dict[str, ProcessingProfile] = {
    "quality": ProcessingProfile(
        name="quality",
        glow_multiplier=1.0,
        chroma_denoise_multiplier=1.0,
        target_bit_depth=None,
        compression=None,
    ),
    "balanced": ProcessingProfile(
        name="balanced",
        glow_multiplier=0.6,
        chroma_denoise_multiplier=0.5,
        target_bit_depth=16,
        compression=None,
    ),
    "performance": ProcessingProfile(
        name="performance",
        glow_multiplier=0.0,
        chroma_denoise_multiplier=0.0,
        target_bit_depth=8,
        compression="tiff_jpeg",
    ),
}


__all__ = [
    "DEFAULT_PROFILE_NAME",
    "PROCESSING_PROFILES",
    "ProcessingProfile",
]
