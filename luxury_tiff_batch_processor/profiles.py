"""Processing profiles for balancing quality, performance, and file size.

Profiles adjust expensive operations (glow, chroma denoising) and output settings
(bit depth, compression) to optimize for different use cases:

- **quality**: Maximum fidelity, preserves source bit depth, full effect strength
- **balanced**: Good quality with moderate speed, 16-bit output, reduced effects
- **performance**: Fast processing, 8-bit JPEG compression, effects disabled

Example Usage
-------------

    from luxury_tiff_batch_processor import PROCESSING_PROFILES

    profile = PROCESSING_PROFILES["balanced"]
    effective_glow = profile.resolve_glow(0.5)  # Returns 0.3 (60% of requested)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class ProcessingProfile:
    """Configuration for quality/performance trade-offs in image processing.

    Attributes:
        name: Profile identifier.
        glow_multiplier: Factor applied to glow effect (0.0 to disable).
        chroma_denoise_multiplier: Factor applied to chroma denoising (0.0 to disable).
        target_bit_depth: Output bit depth (None preserves source, 8 or 16 to override).
        compression: Forced compression method (None uses user preference).
    """

    name: str
    glow_multiplier: float
    chroma_denoise_multiplier: float
    target_bit_depth: int | None
    compression: str | None

    def resolve_glow(self, value: float) -> float:
        """Apply profile's glow multiplier to requested amount.

        Args:
            value: Requested glow strength.

        Returns:
            Adjusted glow strength based on profile settings.
        """
        return value * self.glow_multiplier

    def resolve_chroma_denoise(self, value: float) -> float:
        """Apply profile's chroma denoise multiplier to requested amount.

        Args:
            value: Requested chroma denoise strength.

        Returns:
            Adjusted chroma denoise strength based on profile settings.
        """
        return value * self.chroma_denoise_multiplier

    def target_dtype(self, source_dtype: np.dtype) -> np.dtype:
        """Determine output dtype based on profile and source.

        Args:
            source_dtype: Original image dtype.

        Returns:
            Target dtype for output file.
        """
        if self.target_bit_depth is None:
            return np.dtype(source_dtype)

        if self.target_bit_depth >= 16:
            return np.dtype(np.uint16)

        return np.dtype(np.uint8)

    def resolve_compression(self, requested: str) -> str:
        """Determine compression method for this profile.

        Args:
            requested: User-requested compression method.

        Returns:
            Compression to use (profile override or user preference).
        """
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
