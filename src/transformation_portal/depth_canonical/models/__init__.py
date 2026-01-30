"""Model registry and interfaces for depth estimation."""

from .registry import ModelRegistry, DepthEstimationModel
from .da2_wrapper import DA2ModelWrapper
from .da3_wrapper import DA3ModelWrapper

__all__ = [
    "ModelRegistry",
    "DepthEstimationModel",
    "DA2ModelWrapper",
    "DA3ModelWrapper",
]
