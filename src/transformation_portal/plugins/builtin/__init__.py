"""Built-in plugins for Transformation Portal.

These plugins serve as reference implementations demonstrating
the plugin architecture patterns and best practices.

Available built-in plugins:
- GaussianBlurProcessor: Simple image blur processor
- ContrastEnhancer: Contrast adjustment enhancer
- EdgeDepthEstimator: Edge-based depth estimation (demo)
"""

from .processors import GaussianBlurProcessor
from .enhancers import ContrastEnhancer, SharpenEnhancer
from .depth_models import EdgeDepthEstimator

__all__ = [
    'GaussianBlurProcessor',
    'ContrastEnhancer',
    'SharpenEnhancer',
    'EdgeDepthEstimator',
]
