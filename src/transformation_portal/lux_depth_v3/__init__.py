"""
DEPRECATED: This module is deprecated and will be removed in v2.0.0.

Please use `transformation_portal.depth_canonical` instead.

Migration Guide: https://github.com/RC219805/Transformation_Portal/blob/main/docs/migration/depth_v2_migration.md

Deprecation Timeline:
- v1.8.0 (Feb 2026): Deprecation warnings added
- v1.9.0 (Apr 2026): Final reminder warnings
- v2.0.0 (Aug 2026): Module removed

Original Documentation:
Lux Depth V3 Pipeline - Public API.

This module defines the stable public surface for the lux_depth_v3 package.
Import from this module rather than internal submodules to ensure API stability.

Example:
    >>> from pathlib import Path
    >>> from transformation_portal.lux_depth_v3 import EnhanceOrchestrator, EnhanceConfig
    >>> config = EnhanceConfig()
    >>> orchestrator = EnhanceOrchestrator(config, output_root=Path("./output"))
"""

import warnings

__version__ = "3.0.0-alpha"

# Issue deprecation warning on import
warnings.warn(
    "transformation_portal.lux_depth_v3 is deprecated and will be removed in v2.0.0. "
    "Use transformation_portal.depth_canonical instead. "
    "See https://github.com/RC219805/Transformation_Portal/blob/main/docs/migration/depth_v2_migration.md",
    FutureWarning,
    stacklevel=2
)

# Import canonical implementations for compatibility shims
from ..depth_canonical import (
    DepthPipeline as _CanonicalDepthPipeline,
    generate_pbr_maps as _canonical_generate_pbr_maps,
)

# Original imports (still functional)
from .orchestrator import EnhanceOrchestrator
from .config import (
    DA3Config,
    ModelVariant,
    Preset,
    EnhanceConfig,
    PostprocessingConfig,
    DeviceConfig,
)
from .postprocessing import Postprocessor
from .inference import DA3InferenceEngine, DepthResult

# Backward compatibility shims
generate_pbr_maps = _canonical_generate_pbr_maps

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
    # Compatibility shims
    "generate_pbr_maps",  # Now points to depth_canonical.processing.generate_pbr_maps
]
