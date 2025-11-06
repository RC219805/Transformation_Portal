"""
Model wrappers for depth estimation.
"""

from .depth_anything_v2 import DepthAnythingV2Model, ModelBackend, ModelVariant
from .coreml_wrapper import CoreMLDepthModel

__all__ = ["DepthAnythingV2Model", "ModelBackend", "ModelVariant", "CoreMLDepthModel"]
