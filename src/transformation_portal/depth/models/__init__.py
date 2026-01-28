"""
Model wrappers for depth estimation.
"""

from .depth_anything_v2 import DepthAnythingV2Model, ModelBackend, ModelVariant
from .coreml_wrapper import CoreMLDepthModel

try:
    from .coreml_exporter import CoreMLExporter, CoreMLDepthEstimator
except ModuleNotFoundError:
    # CoreML tooling is optional (macOS-only / dev-time)
    CoreMLExporter = None
    CoreMLDepthEstimator = None

__all__ = [
    "DepthAnythingV2Model",
    "ModelBackend",
    "ModelVariant",
    "CoreMLDepthModel",
    "CoreMLExporter",
    "CoreMLDepthEstimator",
]
