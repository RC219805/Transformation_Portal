"""Compatibility tests for the Phase 4A rendering_4k type extraction."""

from __future__ import annotations

import importlib

import pytest
from PIL import Image

pytestmark = [pytest.mark.unit]


PHASE_4A_SYMBOLS = (
    "AIEnhancementConfig",
    "ColorGradingConfig",
    "DepthConfig",
    "DeviceType",
    "MaterialResponseConfig",
    "OutputConfig",
    "PipelineConfig",
    "ProcessingResult",
    "QualityFeedbackConfig",
    "QualityLevel",
    "QualityMetrics",
    "STAGE_NAMES",
    "StageMetrics",
    "ToneMappingConfig",
    "ToneMappingMethod",
    "UpscalingConfig",
)


def test_legacy_rendering_module_reexports_phase_4a_types() -> None:
    legacy = importlib.import_module("transformation_portal.pipelines.rendering_4k_pipeline")
    extracted = importlib.import_module("transformation_portal.pipelines.rendering_4k.types")

    for symbol in PHASE_4A_SYMBOLS:
        assert getattr(legacy, symbol) is getattr(extracted, symbol)


def test_rendering_4k_package_reexports_phase_4a_types() -> None:
    package = importlib.import_module("transformation_portal.pipelines.rendering_4k")
    extracted = importlib.import_module("transformation_portal.pipelines.rendering_4k.types")

    for symbol in PHASE_4A_SYMBOLS:
        assert getattr(package, symbol) is getattr(extracted, symbol)


def test_extracted_pipeline_config_preserves_default_contract() -> None:
    from transformation_portal.pipelines.rendering_4k.types import (
        PipelineConfig,
        QualityLevel,
        ToneMappingConfig,
        ToneMappingMethod,
        UpscalingConfig,
    )

    config = PipelineConfig()

    assert config.quality_level is QualityLevel.HIGH
    assert isinstance(config.tone_mapping, ToneMappingConfig)
    assert config.tone_mapping.method is ToneMappingMethod.AGX
    assert isinstance(config.upscaling, UpscalingConfig)
    assert config.upscaling.target_resolution == (3840, 2160)


def test_extracted_processing_result_quality_score_matches_legacy_contract() -> None:
    from transformation_portal.pipelines.rendering_4k.types import ProcessingResult, QualityMetrics

    image = Image.new("RGB", (2, 2))

    assert ProcessingResult(image=image).quality_score == pytest.approx(0.0)
    assert ProcessingResult(image=image, quality_metrics=QualityMetrics(overall_score=0.87)).quality_score == pytest.approx(
        0.87
    )
