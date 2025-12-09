"""
Concrete stage implementations for Lux Depth V2 pipeline.

Provides reusable, cacheable stages for:
- Depth estimation
- Material segmentation
- Image enhancement
- Upscaling
"""

from .depth import DepthEstimationStage
from .materials import MaterialSegmentationStage
from .enhancement import EnhancementStage
from .upscaling import UpscalingStage

__all__ = [
    "DepthEstimationStage",
    "MaterialSegmentationStage",
    "EnhancementStage",
    "UpscalingStage",
]
