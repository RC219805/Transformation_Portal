"""Protocols module for lux_depth_v3.

This module defines the stable protocol interfaces for depth model backends.
Following the v3.0 architecture, all depth backends implement the DepthModel
protocol for swappable, governed model execution.
"""

from .depth_model import BackendCapability, BackendInfo, BackendRole, DepthModel, DepthModelRegistry
from .segmentation_backend import SegmentationBackend, SegmentationBackendInfo

__all__ = [
    "DepthModel",
    "BackendRole",
    "BackendCapability",
    "BackendInfo",
    "DepthModelRegistry",
    "SegmentationBackend",
    "SegmentationBackendInfo",
]
