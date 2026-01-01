"""Enhancement orchestrator for V3 + V2 integration.

This module provides the orchestration layer that combines DA3 depth estimation
with V2's enhancement pipeline (weights, grading, upscaling, exports, reports).
"""

# Lazy imports to avoid circular dependencies and heavy torch imports
__all__ = [
    "CombinedManifest",
    "DepthMetadata",
    "V2Metadata",
    "TimingMetadata",
    "ReproMetadata",
    "InputMetadata",
    "EnhanceOrchestrator",
    "EnhanceConfig",
    "write_depth_u16_png",
    "sanitize_file_stem",
    "validate_extra_args",
]


def __getattr__(name):
    """Lazy import to avoid heavy dependencies on module import."""
    if name in ["CombinedManifest", "DepthMetadata", "V2Metadata", "TimingMetadata", "ReproMetadata", "InputMetadata"]:
        from lux_depth_v3.enhance.manifest import (
            CombinedManifest,
            DepthMetadata,
            V2Metadata,
            TimingMetadata,
            ReproMetadata,
            InputMetadata,
        )

        return locals()[name]
    elif name in ["EnhanceOrchestrator", "EnhanceConfig"]:
        from lux_depth_v3.enhance.orchestrator import EnhanceOrchestrator, EnhanceConfig

        return locals()[name]
    elif name == "write_depth_u16_png":
        from lux_depth_v3.enhance.depth_writer import write_depth_u16_png

        return write_depth_u16_png
    elif name == "sanitize_file_stem":
        from lux_depth_v3.enhance.security import sanitize_file_stem

        return sanitize_file_stem
    elif name == "validate_extra_args":
        from lux_depth_v3.enhance.security import validate_extra_args

        return validate_extra_args
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
