"""
DEPRECATED: This module is deprecated and will be removed in v2.0.0.

Please use `transformation_portal.depth_canonical` instead.

Migration Guide: https://github.com/RC219805/Transformation_Portal/blob/main/docs/migration/depth_v2_migration.md

Deprecation Timeline:
- v1.8.0 (Feb 2026): Deprecation warnings added
- v1.9.0 (Apr 2026): Final reminder warnings
- v2.0.0 (Aug 2026): Module removed

Original Documentation:
Depth Anything V2 Pipeline for Architectural Rendering Enhancement

A production-ready depth-aware image processing pipeline optimized for Apple Silicon.
Provides monocular depth estimation with depth-guided enhancements for architectural visualization.
"""

import warnings

__version__ = "1.0.0"
__author__ = "Transformation Portal"

# Issue deprecation warning on import
warnings.warn(
    "transformation_portal.depth is deprecated and will be removed in v2.0.0. "
    "Use transformation_portal.depth_canonical instead. "
    "See https://github.com/RC219805/Transformation_Portal/blob/main/docs/migration/depth_v2_migration.md",
    FutureWarning,
    stacklevel=2
)

# Import canonical implementations
from ..depth_canonical import DepthPipeline as _CanonicalDepthPipeline
from ..depth_canonical import UnifiedDepthConfig as _UnifiedDepthConfig

# Original imports (for internal compatibility)
from .models.depth_anything_v2 import DepthAnythingV2Model
from .pipeline import ArchitecturalDepthPipeline as _OriginalArchitecturalDepthPipeline
from .utils.cache import DepthCache

# Backward compatibility shims - map old names to new canonical classes
ArchitecturalDepthPipeline = _CanonicalDepthPipeline
DepthConfig = _UnifiedDepthConfig

__all__ = [
    "ArchitecturalDepthPipeline",  # Now points to DepthPipeline from depth_canonical
    "DepthConfig",  # Now points to UnifiedDepthConfig from depth_canonical
    "DepthAnythingV2Model",  # Original class, still available
    "DepthCache",  # Original class, still available
]
