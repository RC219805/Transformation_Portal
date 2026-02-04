"""
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

from .atmospheric_modeler import (
    AtmosphericConfig,
    AtmosphericModeler,
    MontecitoCoastalModel,
)
from .depth_estimator import DepthConfig, DepthEstimator, DepthMap
from .depth_filters import DepthAwareBlur, DepthAwareSharpen, DepthGuidedFilter
from .depth_pipeline import DepthPipeline, DepthPipelineConfig
from .spatial_processor import SpatialFeatures, SpatialProcessor

__all__ = [
    "DepthEstimator",
    "DepthConfig",
    "DepthMap",
    "SpatialProcessor",
    "SpatialFeatures",
    "AtmosphericModeler",
    "AtmosphericConfig",
    "MontecitoCoastalModel",
    "DepthPipeline",
    "DepthPipelineConfig",
    "DepthGuidedFilter",
    "DepthAwareBlur",
    "DepthAwareSharpen",
]

__version__ = "1.0.0"
