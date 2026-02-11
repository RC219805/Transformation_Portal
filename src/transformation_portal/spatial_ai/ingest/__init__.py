"""Linear ingest pipeline for Spatial AI Foundation.

This module provides RAW/TIFF → float32 linear light decoding for research workflows.

Architecture (ADR-023):
- Complete isolation from lux_depth_v3.raw_loader (no shared decode logic)
- Linear gamma (1.0) enforcement
- 16-bit → float32 pipeline (no 8-bit collapse)
- HDR preservation (values >1.0 allowed)
- Provenance tracking

Usage:
    >>> from transformation_portal.spatial_ai.ingest import linear_decoder
    >>> result = linear_decoder.decode(
    ...     input_path="scene.tiff",
    ...     gamma=1.0,
    ...     emit_exr=True,
    ...     emit_provenance=True
    ... )
    >>> assert result.linear_rgb.max() > 1.0  # HDR preserved
    >>> assert result.gamma == 1.0  # Linear light
"""

from __future__ import annotations

from .linear_decoder import LinearDecoder, LinearIngestResult, decode

__all__ = [
    "LinearDecoder",
    "LinearIngestResult",
    "decode",
]
