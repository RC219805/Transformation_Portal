"""
Model wrappers for depth estimation.
"""

from .coreml_wrapper import CoreMLDepthModel
from .depth_anything_v2 import DepthAnythingV2Model, ModelBackend, ModelVariant

__all__ = ["DepthAnythingV2Model", "ModelBackend", "ModelVariant", "CoreMLDepthModel"]
