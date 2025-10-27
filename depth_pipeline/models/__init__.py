"""
Model wrappers for depth estimation.
"""

from .depth_anything_v2 import DepthAnythingV2Model, ModelBackend
from .coreml_wrapper import CoreMLDepthModel

__all__ = ["DepthAnythingV2Model", "ModelBackend", "CoreMLDepthModel"]
