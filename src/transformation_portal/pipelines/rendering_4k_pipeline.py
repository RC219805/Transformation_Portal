#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy compatibility shim for the decomposed 4K rendering pipeline."""

from __future__ import annotations

from .rendering_4k.pipeline import (
    HAS_CONTROLNET_AUX,
    HAS_QUALITY_BRIDGE,
    HAS_TIFFFILE,
    HAS_TORCH,
    HAS_TQDM,
    HAS_YAML,
    CannyDetector,
    QualityFeedbackBridge,
    Rendering4KPipeline,
    UnifiedQualityMetrics,
)
from .rendering_4k.pipeline import _json_default as _json_default  # noqa: F401
from .rendering_4k.pipeline import (
    create_rag_indexing_callback,
    logger,
    main,
)
from .rendering_4k.quality import GPUMemoryManager, QualityAssessor  # noqa: F401

# isort: off
from .rendering_4k.stages import (  # noqa: F401
    _aces_approximation as _aces_approximation,
    _agx_sigmoid as _agx_sigmoid,
    _apply_local_contrast as _apply_local_contrast,
    _apply_lut as _apply_lut,
    _apply_vibrance as _apply_vibrance,
    _filmic_hable as _filmic_hable,
    _load_cube_lut as _load_cube_lut,
    _simple_box_blur as _simple_box_blur,
    _simple_gaussian_blur as _simple_gaussian_blur,
    _simple_gaussian_blur_2d as _simple_gaussian_blur_2d,
    apply_color_grading as apply_color_grading,
    apply_material_response as apply_material_response,
    apply_tone_mapping as apply_tone_mapping,
    apply_upscaling as apply_upscaling,
    estimate_depth_simple as estimate_depth_simple,
)

# isort: on
from .rendering_4k.types import (  # noqa: F401
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

__all__ = [
    "AIEnhancementConfig",
    "CannyDetector",
    "ColorGradingConfig",
    "DepthConfig",
    "DeviceType",
    "GPUMemoryManager",
    "HAS_CONTROLNET_AUX",
    "HAS_QUALITY_BRIDGE",
    "HAS_TIFFFILE",
    "HAS_TORCH",
    "HAS_TQDM",
    "HAS_YAML",
    "MaterialResponseConfig",
    "OutputConfig",
    "PipelineConfig",
    "ProcessingResult",
    "QualityAssessor",
    "QualityFeedbackBridge",
    "QualityFeedbackConfig",
    "QualityLevel",
    "QualityMetrics",
    "Rendering4KPipeline",
    "STAGE_NAMES",
    "StageMetrics",
    "ToneMappingConfig",
    "ToneMappingMethod",
    "UnifiedQualityMetrics",
    "UpscalingConfig",
    "_aces_approximation",
    "_agx_sigmoid",
    "_apply_local_contrast",
    "_apply_lut",
    "_apply_vibrance",
    "_filmic_hable",
    "_json_default",
    "_load_cube_lut",
    "_simple_box_blur",
    "_simple_gaussian_blur",
    "_simple_gaussian_blur_2d",
    "apply_color_grading",
    "apply_material_response",
    "apply_tone_mapping",
    "apply_upscaling",
    "create_rag_indexing_callback",
    "estimate_depth_simple",
    "logger",
    "main",
]

if __name__ == "__main__":
    raise SystemExit(main())
