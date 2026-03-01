"""Lux Depth V3 Pipeline - Public API.

This module intentionally exposes all public symbols lazily so importing
``transformation_portal.lux_depth_v3`` does not eagerly load optional ML stacks.
That keeps CLI/help/test import paths stable in CPU-only or partially provisioned
environments.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

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
    # PBR presets
    "STANDARD_QUALITY",
    "PREMIUM_QUALITY",
    "FAST_PREVIEW",
    "WOOD_OPTIMIZED",
    "METAL_OPTIMIZED",
    "GLASS_OPTIMIZED",
    "STONE_OPTIMIZED",
    "FABRIC_OPTIMIZED",
    "get_preset",
    "list_presets",
    # PBR processor
    "PBRConfig",
    "PBRProcessor",
    # v3.0 Contracts
    "DepthArtifact",
    "DepthProvenance",
    "CameraIntrinsics",
    "DepthArtifactWriter",
    "LicenseTier",
    # v3.0 Protocols
    "DepthModel",
    "BackendRole",
    "BackendCapability",
    "BackendInfo",
    "DepthModelRegistry",
]

_LAZY_EXPORTS: Dict[str, Tuple[str, str]] = {
    # Orchestration
    "EnhanceOrchestrator": (".orchestrator", "EnhanceOrchestrator"),
    # Configuration
    "DA3Config": (".config", "DA3Config"),
    "ModelVariant": (".config", "ModelVariant"),
    "Preset": (".config", "Preset"),
    "EnhanceConfig": (".config", "EnhanceConfig"),
    "PostprocessingConfig": (".config", "PostprocessingConfig"),
    "DeviceConfig": (".config", "DeviceConfig"),
    # Core processing
    "Postprocessor": (".postprocessing", "Postprocessor"),
    "DA3InferenceEngine": (".inference", "DA3InferenceEngine"),
    "DepthResult": (".inference", "DepthResult"),
    # PBR presets
    "STANDARD_QUALITY": (".pbr_presets", "STANDARD_QUALITY"),
    "PREMIUM_QUALITY": (".pbr_presets", "PREMIUM_QUALITY"),
    "FAST_PREVIEW": (".pbr_presets", "FAST_PREVIEW"),
    "WOOD_OPTIMIZED": (".pbr_presets", "WOOD_OPTIMIZED"),
    "METAL_OPTIMIZED": (".pbr_presets", "METAL_OPTIMIZED"),
    "GLASS_OPTIMIZED": (".pbr_presets", "GLASS_OPTIMIZED"),
    "STONE_OPTIMIZED": (".pbr_presets", "STONE_OPTIMIZED"),
    "FABRIC_OPTIMIZED": (".pbr_presets", "FABRIC_OPTIMIZED"),
    "get_preset": (".pbr_presets", "get_preset"),
    "list_presets": (".pbr_presets", "list_presets"),
    # PBR processor
    "PBRConfig": (".pbr", "PBRConfig"),
    "PBRProcessor": (".pbr_processor", "PBRProcessor"),
    # v3.0 Contracts
    "DepthArtifact": (".contracts", "DepthArtifact"),
    "DepthProvenance": (".contracts", "DepthProvenance"),
    "CameraIntrinsics": (".contracts", "CameraIntrinsics"),
    "DepthArtifactWriter": (".contracts", "DepthArtifactWriter"),
    "LicenseTier": (".contracts", "LicenseTier"),
    # v3.0 Protocols
    "DepthModel": (".protocols", "DepthModel"),
    "BackendRole": (".protocols", "BackendRole"),
    "BackendCapability": (".protocols", "BackendCapability"),
    "BackendInfo": (".protocols", "BackendInfo"),
    "DepthModelRegistry": (".protocols", "DepthModelRegistry"),
}


def __getattr__(name: str) -> Any:
    """Lazily resolve public exports."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = target
    module = importlib.import_module(module_path, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose lazy exports for autocomplete/introspection."""
    return sorted(set(globals().keys()) | set(__all__))


__version__ = "3.0.0"  # Lux Depth Engine v3.0
