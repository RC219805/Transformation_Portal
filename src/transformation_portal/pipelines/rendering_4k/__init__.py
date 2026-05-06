"""Internal modules for the 4K rendering pipeline decomposition."""

from importlib import import_module

from .types import (
    STAGE_NAMES,
    AIEnhancementConfig,
    ColorGradingConfig,
    DepthConfig,
    DeviceType,
    MaterialResponseConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingResult,
    QualityFeedbackConfig,
    QualityLevel,
    QualityMetrics,
    StageMetrics,
    ToneMappingConfig,
    ToneMappingMethod,
    UpscalingConfig,
)

_STAGE_EXPORTS = {
    "apply_color_grading",
    "apply_material_response",
    "apply_tone_mapping",
    "apply_upscaling",
    "estimate_depth_simple",
}

_QUALITY_EXPORTS = {
    "GPUMemoryManager",
    "QualityAssessor",
}

__all__ = [
    "AIEnhancementConfig",
    "ColorGradingConfig",
    "DepthConfig",
    "DeviceType",
    "GPUMemoryManager",
    "MaterialResponseConfig",
    "OutputConfig",
    "PipelineConfig",
    "ProcessingResult",
    "QualityAssessor",
    "QualityFeedbackConfig",
    "QualityLevel",
    "QualityMetrics",
    "STAGE_NAMES",
    "StageMetrics",
    "ToneMappingConfig",
    "ToneMappingMethod",
    "UpscalingConfig",
    "apply_color_grading",
    "apply_material_response",
    "apply_tone_mapping",
    "apply_upscaling",
    "estimate_depth_simple",
]


def __getattr__(name: str) -> object:
    """Lazily expose extracted helpers without importing optional dependencies."""
    if name in _STAGE_EXPORTS:
        value = getattr(import_module(f"{__name__}.stages"), name)
        globals()[name] = value
        return value

    if name in _QUALITY_EXPORTS:
        value = getattr(import_module(f"{__name__}.quality"), name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _STAGE_EXPORTS | _QUALITY_EXPORTS)
