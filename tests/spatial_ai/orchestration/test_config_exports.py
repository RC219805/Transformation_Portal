"""Compatibility tests for the Spatial AI PipelineConfig extraction."""

from __future__ import annotations

import importlib

import pytest

from transformation_portal.spatial_ai.orchestration.error_handler import ErrorRecoveryStrategy

pytestmark = pytest.mark.unit


def test_pipeline_config_identity_is_preserved_across_import_surfaces() -> None:
    config_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.config")
    pipeline_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.pipeline")
    package_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration")

    assert pipeline_mod.PipelineConfig is config_mod.PipelineConfig
    assert package_mod.PipelineConfig is config_mod.PipelineConfig
    assert "PipelineConfig" in package_mod.__all__


def test_spatial_ai_pipeline_accepts_extracted_pipeline_config() -> None:
    config_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.config")
    pipeline_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.pipeline")

    config = config_mod.PipelineConfig(tier="standard", stages=["ingest"])
    pipeline = pipeline_mod.SpatialAIPipeline(config)

    assert pipeline.config is config


def test_spatial_ai_pipeline_keeps_reload_safe_pipeline_config_contract() -> None:
    pipeline_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.pipeline")

    class PipelineConfig:
        tier = "standard"
        stages = ["ingest"]
        ingest = {}
        segmentation = {"cache_policy": "read_write"}
        materials = {}
        reconstruction = {}
        resource_limits = None
        error_strategy = ErrorRecoveryStrategy.RETRY
        use_execution_graph = False

    reload_safe_config = PipelineConfig()
    pipeline = pipeline_mod.SpatialAIPipeline(reload_safe_config)

    assert pipeline.config is reload_safe_config


def test_extracted_pipeline_config_preserves_validation_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    config_mod = importlib.import_module("transformation_portal.spatial_ai.orchestration.config")

    config = config_mod.PipelineConfig(
        tier="apex_research",
        stages=["reconstruct", "materials"],
        segmentation={"cache_policy": "off"},
        materials={"backend": "heuristic", "strict_backend": False},
        error_strategy="retry_cpu_fallback",
    )

    assert config.stages == ["reconstruction", "materials"]
    assert config.segmentation["cache_policy"] == "off"
    assert config.error_strategy is ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK

    with pytest.raises(ValueError, match="segmentation.cache_policy"):
        config_mod.PipelineConfig(tier="standard", stages=["ingest"], segmentation={"cache_policy": "invalid"})

    with pytest.raises(ValueError, match="research tier.*3DGS"):
        config_mod.PipelineConfig(tier="standard", stages=["reconstruct"])

    monkeypatch.delenv("PBRFUSION_PATH", raising=False)
    with pytest.raises(ValueError, match="runtime_missing"):
        config_mod.PipelineConfig(
            tier="standard",
            stages=["ingest", "segment", "materials"],
            materials={"backend": "pbr_fusion", "strict_backend": True},
        )
