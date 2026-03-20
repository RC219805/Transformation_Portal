"""Unit tests for PipelineCoordinator.

Tests the pipeline coordination logic extracted from orchestrator.py
as part of ADR-043 decomposition.

These tests verify:
1. Backend selection with fallback
2. Runtime backend chain resolution
3. Model ID resolution
4. ExecutionPlan generation
5. Backward compatibility with orchestrator
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]


class TestPipelineCoordinatorImports:
    """Test that imports work from the new module."""

    def test_import_from_pipeline_coordinator(self):
        """Test that we can import from the new pipeline_coordinator module."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            BackendSelection,
            ExecutionPlan,
            PipelineCoordinator,
            default_model_id_for_backend,
            derive_model_id_from_backend_instance,
            expected_output_depth_units_for_backend,
            resolve_backend_model_id,
            resolve_runtime_backend_chain,
            select_backend,
        )

        assert PipelineCoordinator is not None
        assert BackendSelection is not None
        assert ExecutionPlan is not None
        assert callable(resolve_runtime_backend_chain)
        assert callable(select_backend)
        assert callable(default_model_id_for_backend)
        assert callable(derive_model_id_from_backend_instance)
        assert callable(resolve_backend_model_id)
        assert callable(expected_output_depth_units_for_backend)


class TestResolveRuntimeBackendChain:
    """Test runtime backend chain resolution."""

    def test_chain_starts_with_primary(self):
        """Test that chain starts with primary backend."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            resolve_runtime_backend_chain,
        )

        config = EnhanceConfig()
        chain = resolve_runtime_backend_chain("da3", config)

        assert chain[0] == "da3"

    def test_chain_includes_fallbacks(self):
        """Test that chain includes fallback backends."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            resolve_runtime_backend_chain,
        )

        config = EnhanceConfig(depth_operational_fallback_chain=("da3", "da2"))
        chain = resolve_runtime_backend_chain("da3", config)

        assert "da2" in chain

    def test_chain_with_synthetic_fallback(self):
        """Test that synthetic is added when allowed."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            resolve_runtime_backend_chain,
        )

        config = EnhanceConfig(allow_synthetic_fallback=True)
        chain = resolve_runtime_backend_chain("da3", config)

        assert "synthetic" in chain

    def test_chain_no_duplicates(self):
        """Test that chain has no duplicate entries."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            resolve_runtime_backend_chain,
        )

        config = EnhanceConfig(depth_operational_fallback_chain=("da3", "da3", "da2"))
        chain = resolve_runtime_backend_chain("da3", config)

        assert len(chain) == len(set(chain))

    def test_chain_normalizes_backend_id(self):
        """Test that backend IDs are normalized."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            resolve_runtime_backend_chain,
        )

        config = EnhanceConfig()
        chain = resolve_runtime_backend_chain("DA3", config)  # uppercase

        assert chain[0] == "da3"


class TestExpectedOutputDepthUnits:
    """Test depth unit determination."""

    def test_depth_pro_returns_meters(self):
        """Test that depth_pro returns meters."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            expected_output_depth_units_for_backend,
        )

        assert expected_output_depth_units_for_backend("depth_pro") == "meters"

    def test_da3_returns_relative(self):
        """Test that da3 returns relative."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            expected_output_depth_units_for_backend,
        )

        assert expected_output_depth_units_for_backend("da3") == "relative"

    def test_da2_returns_relative(self):
        """Test that da2 returns relative."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            expected_output_depth_units_for_backend,
        )

        assert expected_output_depth_units_for_backend("da2") == "relative"


class TestDefaultModelIdForBackend:
    """Test default model ID resolution."""

    def test_depth_pro_model_id(self):
        """Test depth_pro returns correct model ID."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            default_model_id_for_backend,
        )

        assert default_model_id_for_backend("depth_pro") == "apple/ml-depth-pro"

    def test_da2_model_id(self):
        """Test da2 returns correct model ID."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            default_model_id_for_backend,
        )

        assert default_model_id_for_backend("da2") == "depth-anything/Depth-Anything-V2-Small-hf"

    def test_da3_with_model_variant(self):
        """Test da3 uses model variant."""
        from transformation_portal.lux_depth_v3.config import ModelVariant
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            default_model_id_for_backend,
        )

        model_id = default_model_id_for_backend("da3", ModelVariant.METRIC_LARGE)
        assert "depth-anything" in model_id.lower()

    def test_synthetic_model_id(self):
        """Test synthetic returns correct model ID."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            default_model_id_for_backend,
        )

        assert default_model_id_for_backend("synthetic") == "synthetic/depth-analytic-v1"


class TestDeriveModelIdFromBackendInstance:
    """Test model ID extraction from backend instances."""

    def test_none_backend_returns_none(self):
        """Test that None backend returns None."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            derive_model_id_from_backend_instance,
        )

        assert derive_model_id_from_backend_instance("da3", None) is None

    def test_extracts_model_id_attribute(self):
        """Test extraction from model_id attribute."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            derive_model_id_from_backend_instance,
        )

        backend = MagicMock()
        backend.model_id = "test/model-v1"

        result = derive_model_id_from_backend_instance("da3", backend)
        assert result == "test/model-v1"

    def test_extracts_private_model_id(self):
        """Test extraction from _model_id attribute."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            derive_model_id_from_backend_instance,
        )

        backend = MagicMock(spec=["_model_id"])
        backend._model_id = "test/private-model"

        result = derive_model_id_from_backend_instance("da3", backend)
        assert result == "test/private-model"

    def test_depth_pro_fallback(self):
        """Test depth_pro special case."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            derive_model_id_from_backend_instance,
        )

        backend = MagicMock(spec=[])  # No model_id attributes

        result = derive_model_id_from_backend_instance("depth_pro", backend)
        assert result == "apple/ml-depth-pro"


class TestResolveBackendModelId:
    """Test model ID resolution."""

    def test_depth_pro_canonical(self):
        """Test depth_pro always returns canonical ID."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            resolve_backend_model_id,
        )

        # Even with metadata, depth_pro should be canonical
        result = resolve_backend_model_id(
            "depth_pro",
            result_metadata={"model_id": "other/model"},
        )
        assert result == "apple/ml-depth-pro"

    def test_uses_metadata(self):
        """Test that metadata is used when available."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            resolve_backend_model_id,
        )

        result = resolve_backend_model_id(
            "da3",
            result_metadata={"resolved_model_id": "custom/model"},
        )
        assert result == "custom/model"

    def test_falls_back_to_default(self):
        """Test fallback to default model ID."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            resolve_backend_model_id,
        )

        result = resolve_backend_model_id("synthetic")
        assert result == "synthetic/depth-analytic-v1"


class TestBackendSelection:
    """Test BackendSelection data class."""

    def test_is_success_for_success_status(self):
        """Test is_success for success status."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            BackendSelection,
        )

        selection = BackendSelection(
            requested_backend="da3",
            resolved_backend="da3",
            status="success",
        )
        assert selection.is_success is True

    def test_is_success_for_fallback_status(self):
        """Test is_success for fallback status."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            BackendSelection,
        )

        selection = BackendSelection(
            requested_backend="da3",
            resolved_backend="da2",
            status="fallback",
        )
        assert selection.is_success is True

    def test_is_success_for_error_status(self):
        """Test is_success for error status."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            BackendSelection,
        )

        selection = BackendSelection(
            requested_backend="da3",
            resolved_backend=None,
            status="error",
        )
        assert selection.is_success is False

    def test_to_metadata(self):
        """Test conversion to BackendSelectionMetadata."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            BackendSelection,
        )

        selection = BackendSelection(
            requested_backend="da3",
            resolved_backend="da3",
            status="success",
            reason="Backend ready",
            model_id="depth-anything/model",
            device="mps",
        )

        metadata = selection.to_metadata()
        assert metadata.requested_backend == "da3"
        assert metadata.resolved_backend == "da3"
        assert metadata.resolution_status == "success"


class TestExecutionPlan:
    """Test ExecutionPlan data class."""

    def test_plan_fields(self):
        """Test ExecutionPlan fields."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            ExecutionPlan,
        )

        plan = ExecutionPlan(
            stages=["preprocess", "depth", "output"],
            enable_depth=True,
            enable_v2=False,
            quality_tier="apex",
        )

        assert "depth" in plan.stages
        assert plan.enable_depth is True
        assert plan.enable_v2 is False
        assert plan.quality_tier == "apex"

    def test_plan_default_values(self):
        """Test ExecutionPlan default values."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            ExecutionPlan,
        )

        plan = ExecutionPlan(stages=["preprocess"])

        assert plan.enable_depth is True
        assert plan.enable_v2 is True
        assert plan.enable_pbr is False
        assert plan.quality_tier == "standard"


class TestPipelineCoordinatorClass:
    """Test PipelineCoordinator class."""

    def test_coordinator_init(self):
        """Test coordinator initialization."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            PipelineCoordinator,
        )

        config = EnhanceConfig()
        coordinator = PipelineCoordinator(config)

        assert coordinator.config is config

    def test_coordinator_plan_basic(self):
        """Test coordinator plan generation."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            PipelineCoordinator,
        )

        config = EnhanceConfig(enable_v2=True, v2_preset="default")
        coordinator = PipelineCoordinator(config)
        plan = coordinator.plan()

        assert "preprocess" in plan.stages
        assert "depth" in plan.stages
        assert "v2" in plan.stages
        assert "output" in plan.stages

    def test_coordinator_plan_with_pbr(self):
        """Test coordinator plan includes PBR when enabled."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            PipelineCoordinator,
        )

        config = EnhanceConfig(generate_pbr=True)
        coordinator = PipelineCoordinator(config)
        plan = coordinator.plan()

        assert "pbr" in plan.stages

    def test_coordinator_plan_with_materials_v3(self):
        """Test coordinator plan includes Materials V3 when enabled."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            PipelineCoordinator,
        )

        config = EnhanceConfig(enable_materials_v3=True)
        coordinator = PipelineCoordinator(config)
        plan = coordinator.plan()

        assert "materials_v3" in plan.stages

    def test_coordinator_plan_without_depth(self):
        """Test coordinator plan without depth stage."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            PipelineCoordinator,
        )

        config = EnhanceConfig(generate_pbr=True)  # PBR requires depth
        coordinator = PipelineCoordinator(config)
        plan = coordinator.plan(enable_depth=False)

        assert "depth" not in plan.stages
        assert "pbr" not in plan.stages  # PBR depends on depth

    def test_coordinator_resolve_runtime_chain(self):
        """Test coordinator resolve_runtime_chain method."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            PipelineCoordinator,
        )

        config = EnhanceConfig(depth_operational_fallback_chain=("da3", "da2"))
        coordinator = PipelineCoordinator(config)
        chain = coordinator.resolve_runtime_chain("da3")

        assert chain[0] == "da3"
        assert "da2" in chain

    def test_coordinator_default_model_id(self):
        """Test coordinator default_model_id method."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            PipelineCoordinator,
        )

        config = EnhanceConfig()
        coordinator = PipelineCoordinator(config)

        assert coordinator.default_model_id("depth_pro") == "apple/ml-depth-pro"

    def test_coordinator_expected_depth_units(self):
        """Test coordinator expected_depth_units static method."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            PipelineCoordinator,
        )

        assert PipelineCoordinator.expected_depth_units("depth_pro") == "meters"
        assert PipelineCoordinator.expected_depth_units("da3") == "relative"

    def test_coordinator_plan_quality_tier(self):
        """Test that plan captures quality tier."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            PipelineCoordinator,
        )

        config = EnhanceConfig(quality_tier="apex")
        coordinator = PipelineCoordinator(config)
        plan = coordinator.plan()

        assert plan.quality_tier == "apex"
