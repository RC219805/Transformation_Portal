"""
DEPRECATED: This module is deprecated and will be removed in v2.0.0.

Please use `transformation_portal.depth_canonical` instead.

Migration Guide: https://github.com/RC219805/Transformation_Portal/blob/main/docs/migration/depth_v2_migration.md

Deprecation Timeline:
- v1.8.0 (Feb 2026): Deprecation warnings added
- v1.9.0 (Apr 2026): Final reminder warnings
- v2.0.0 (Aug 2026): Module removed

Original Documentation:
Phase 3: Depth and Spatial Intelligence

This module provides depth-aware ML pipelines and spatial intelligence processing
for architectural enhancement. Implements depth estimation, atmospheric modeling,
and depth-guided enhancement operations.

Key Components:
- Depth Estimator: Depth Anything V2 integration
- Spatial Processor: Spatial intelligence and scene understanding
- Atmospheric Modeler: Montecito coastal atmospheric conditions
- Depth-Aware Pipeline: Enhancement with depth guidance
- Depth-Guided Filters: Depth-aware image processing

Usage:
    from transformation_portal.depth_intelligence import DepthPipeline

    # Initialize with substrate and baseline
    pipeline = DepthPipeline(substrate, baseline)

    # Estimate depth
    depth_map = pipeline.estimate_depth(image)

    # Apply depth-aware enhancement
    enhanced = pipeline.enhance_with_depth(image, depth_map)

    # Model atmospheric effects (Montecito coastal)
    atmospheric = pipeline.apply_atmospheric_model(image, depth_map)
"""

import warnings

__version__ = "1.0.0"

# Issue deprecation warning on import
warnings.warn(
    "transformation_portal.depth_intelligence is deprecated and will be removed in v2.0.0. "
    "Use transformation_portal.depth_canonical instead. "
    "See https://github.com/RC219805/Transformation_Portal/blob/main/docs/migration/depth_v2_migration.md",
    FutureWarning,
    stacklevel=2
)

# Import canonical implementations for compatibility shims
from ..depth_canonical import ModelRegistry as _CanonicalModelRegistry

# Original imports (only import what exists in this sparse module)
try:
    from .depth_estimator import DepthEstimator, DepthConfig, DepthMap
    _has_depth_estimator = True
except ImportError:
    _has_depth_estimator = False
    DepthEstimator = None
    DepthConfig = None
    DepthMap = None

# Note: Most of depth_intelligence was never fully implemented
# Users should migrate to depth_canonical.ModelRegistry

__all__ = []
if _has_depth_estimator:
    __all__.extend(["DepthEstimator", "DepthConfig", "DepthMap"])
