"""
Model wrappers for depth estimation.
"""

from .coreml_wrapper import CoreMLDepthModel
from .depth_anything_v2 import DepthAnythingV2Model, ModelBackend, ModelVariant

try:
    from .coreml_exporter import CoreMLDepthEstimator, CoreMLExporter
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
