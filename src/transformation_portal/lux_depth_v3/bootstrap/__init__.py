"""Bootstrap heuristics for Materials V3.

This module provides heuristic-based detection for "stuff" (amorphous) materials
that are difficult for standard object segmentation models to handle reliably.

Current bootstrap modules:
- sky_seed: Sky detection using spatial priors and image characteristics
"""

from .sky_seed import detect_sky_seed

__all__ = ["detect_sky_seed"]
