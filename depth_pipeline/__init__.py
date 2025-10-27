"""
Depth Anything V2 Pipeline for Architectural Rendering Enhancement

A production-ready depth-aware image processing pipeline optimized for Apple Silicon.
Provides monocular depth estimation with depth-guided enhancements for architectural visualization.
"""

__version__ = "1.0.0"
__author__ = "Transformation Portal"

from .pipeline import ArchitecturalDepthPipeline
from .models.depth_anything_v2 import DepthAnythingV2Model
from .utils.cache import DepthCache

__all__ = [
    "ArchitecturalDepthPipeline",
    "DepthAnythingV2Model",
    "DepthCache",
]
