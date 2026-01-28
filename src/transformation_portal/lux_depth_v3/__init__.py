"""Lux Depth V3 Pipeline - Public API.

This module defines the stable public surface for the lux_depth_v3 package.
Import from this module rather than internal submodules to ensure API stability.

Example:
    >>> from transformation_portal.lux_depth_v3 import EnhanceOrchestrator, DA3Config
    >>> config = DA3Config()
    >>> orchestrator = EnhanceOrchestrator(config)
"""

# Orchestration
from .orchestrator import EnhanceOrchestrator

# Configuration
from .config import (
    DA3Config,
    ModelVariant,
    Preset,
    EnhanceConfig,
    PostprocessingConfig,
    DeviceConfig,
)

# Core processing
from .postprocessing import Postprocessor
from .inference import DA3InferenceEngine, DepthResult

__all__ = [
    # Orchestration
    "EnhanceOrchestrator",
    # Configuration
    "DA3Config",
    "ModelVariant",
    "Preset",
    "EnhanceConfig",
    "PostprocessingConfig",
    "DeviceConfig",
    # Core processing
    "Postprocessor",
    "DA3InferenceEngine",
    "DepthResult",
]

__version__ = "3.0.0-alpha"
