"""
Hyper-Reality Enhancement Module
Part of the Transformation_Portal luxury image processing pipeline
"""

# pylint: disable=possibly-unused-variable
from .hyper_reality_enhancement import (
    HyperRealityProcessor,
    EnhancementConfig,
    QualityMode,
    enhance_image
)

__all__ = [
    'HyperRealityProcessor',
    'EnhancementConfig',
    'QualityMode',
    'enhance_image'
]

__version__ = '3.0.0'
