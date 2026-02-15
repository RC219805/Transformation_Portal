"""Model registry and interfaces for depth estimation."""

from .da2_wrapper import DA2ModelWrapper
from .da3_wrapper import DA3ModelWrapper
from .registry import DepthEstimationModel, ModelRegistry

__all__ = [
    "ModelRegistry",
    "DepthEstimationModel",
    "DA2ModelWrapper",
    "DA3ModelWrapper",
]
