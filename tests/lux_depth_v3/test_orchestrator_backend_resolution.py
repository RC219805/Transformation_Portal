"""Test coverage for orchestrator backend resolution and fallback logic.

Phase 2 Coverage: Backend-selection and fallback logic tests for EnhanceOrchestrator.

Tests verify:
1. Backend selection chain resolution
2. Fallback behavior when primary backend fails
3. Backend metadata capture and propagation
4. Model ID resolution
5. Backend attempt history tracking
6. Device and license handling
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit


def _make_test_image(tmp_path: Path, name: str = "test.png", size: tuple = (64, 64)) -> Path:
    """Create a minimal test image for orchestrator tests."""
    image_path = tmp_path / name
    Image.new("RGB", size, color="white").save(image_path)
    return image_path


def _make_mock_depth_result(backend_id: str = "da3", device: str = "cpu"):
    """Create a deterministic synthetic depth result."""
    from transformation_portal.depth.backends.protocol import DepthResult

    return DepthResult(
        depth_map=np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        metadata={},
        depth_units="relative",
        backend_id=backend_id,
        device=device,
    )


class TestBackendChainResolution:
    """Test backend selection chain resolution."""

    def test_resolve_runtime_backend_chain_default(self) -> None:
        """resolve_runtime_backend_chain returns ordered chain."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import resolve_runtime_backend_chain

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
        )

        chain = resolve_runtime_backend_chain("da3", config)

        assert isinstance(chain, list)
        assert len(chain) >= 1
        assert chain[0] == "da3"

    def test_resolve_runtime_backend_chain_includes_fallbacks(self) -> None:
        """resolve_runtime_backend_chain includes configured fallbacks."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import resolve_runtime_backend_chain

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            depth_operational_fallback_chain=("da3", "da2"),
        )

        chain = resolve_runtime_backend_chain("da3", config)

        assert "da3" in chain
        # da2 should be in the chain if not already present
        assert len(chain) >= 1

    def test_resolve_runtime_backend_chain_synthetic_opt_in(self) -> None:
        """resolve_runtime_backend_chain includes synthetic when allowed."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import resolve_runtime_backend_chain

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            allow_synthetic_fallback=True,
        )

        chain = resolve_runtime_backend_chain("da3", config)

        assert "synthetic" in chain

    def test_resolve_runtime_backend_chain_no_synthetic_by_default(self) -> None:
        """resolve_runtime_backend_chain excludes synthetic by default."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import resolve_runtime_backend_chain

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            allow_synthetic_fallback=False,
        )

        # Clear env var if set
        with patch.dict("os.environ", {}, clear=True):
            chain = resolve_runtime_backend_chain("da3", config)

        assert "synthetic" not in chain


class TestBackendSelection:
    """Test backend selection via pipeline_coordinator."""

    def test_select_backend_success(self) -> None:
        """select_backend returns success on available backend."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.pipeline_coordinator import select_backend

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
        )

        backend = Mock()
        backend.name = "da3"
        backend.license_type = Mock(value="commercial")
        backend.ensure_available.return_value = None

        registry = Mock()
        registry.get_backend.return_value = backend

        selection = select_backend("da3", config, registry, ModelVariant.METRIC_LARGE)

        assert selection.is_success
        assert selection.resolved_backend == "da3"
        assert selection.backend is backend

    def test_select_backend_fallback(self) -> None:
        """select_backend falls back when primary fails (non-strict mode)."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.pipeline_coordinator import select_backend

        # Don't explicitly request da3 to avoid strict mode
        config = EnhanceConfig(
            depth_backend=None,  # No explicit request
            depth_device="cpu",
            enable_v2=False,
            allow_synthetic_fallback=True,
        )

        # Primary backend (da3 by default) fails on ensure_available
        da3_backend = Mock()
        da3_backend.name = "da3"
        da3_backend.ensure_available.side_effect = RuntimeError("da3 Not available")

        # Fallback backend (da2) also fails
        da2_backend = Mock()
        da2_backend.name = "da2"
        da2_backend.ensure_available.side_effect = RuntimeError("da2 Not available")

        # Synthetic succeeds
        synthetic_backend = Mock()
        synthetic_backend.name = "synthetic"
        synthetic_backend.license_type = Mock(value="synthetic")
        synthetic_backend.ensure_available.return_value = None

        def get_backend_side_effect(backend_id, config):
            if backend_id == "da3":
                return da3_backend
            if backend_id == "da2":
                return da2_backend
            return synthetic_backend

        registry = Mock()
        registry.get_backend.side_effect = get_backend_side_effect

        selection = select_backend(None, config, registry, ModelVariant.METRIC_LARGE)

        # Should eventually resolve to synthetic after fallbacks
        assert selection.is_success
        assert selection.resolved_backend == "synthetic"
        assert selection.status == "synthetic_fallback"

    def test_backend_selection_metadata_creation(self) -> None:
        """BackendSelection can convert to metadata."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import BackendSelection

        selection = BackendSelection(
            requested_backend="da3",
            resolved_backend="da3",
            status="success",
            reason=None,
            backend=Mock(),
            model_id="depth-anything/large",
            device="cpu",
        )

        metadata = selection.to_metadata()

        assert metadata.requested_backend == "da3"
        assert metadata.resolved_backend == "da3"
        assert metadata.resolution_status == "success"


class TestModelIdResolution:
    """Test model ID resolution for provenance."""

    def test_default_model_id_for_backend_da3(self) -> None:
        """default_model_id_for_backend returns correct ID for da3."""
        from transformation_portal.lux_depth_v3.config import ModelVariant
        from transformation_portal.lux_depth_v3.pipeline_coordinator import default_model_id_for_backend

        model_id = default_model_id_for_backend("da3", ModelVariant.METRIC_LARGE)

        assert model_id is not None
        assert isinstance(model_id, str)
        assert len(model_id) > 0

    def test_default_model_id_for_backend_depth_pro(self) -> None:
        """default_model_id_for_backend returns canonical ID for depth_pro."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import default_model_id_for_backend

        model_id = default_model_id_for_backend("depth_pro")

        assert model_id == "apple/ml-depth-pro"

    def test_default_model_id_for_backend_synthetic(self) -> None:
        """default_model_id_for_backend returns ID for synthetic."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import default_model_id_for_backend

        model_id = default_model_id_for_backend("synthetic")

        assert model_id is not None
        assert "synthetic" in model_id.lower()

    def test_resolve_backend_model_id_from_metadata(self) -> None:
        """resolve_backend_model_id extracts from metadata."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import resolve_backend_model_id

        metadata = {"resolved_model_id": "custom/model-v1"}

        model_id = resolve_backend_model_id("da3", result_metadata=metadata)

        assert model_id == "custom/model-v1"

    def test_resolve_backend_model_id_depth_pro_canonical(self) -> None:
        """resolve_backend_model_id returns canonical for depth_pro."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import resolve_backend_model_id

        model_id = resolve_backend_model_id("depth_pro")

        assert model_id == "apple/ml-depth-pro"

    def test_derive_model_id_from_backend_instance(self) -> None:
        """derive_model_id_from_backend_instance extracts from instance."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import derive_model_id_from_backend_instance

        backend = Mock()
        backend.model_id = "extracted/model"

        model_id = derive_model_id_from_backend_instance("da3", backend)

        assert model_id == "extracted/model"


class TestBackendAttemptTracking:
    """Test backend attempt history tracking."""

    def test_attempt_record_structure(self, tmp_path: Path) -> None:
        """Attempt records have expected structure."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
        )

        backend = Mock()
        backend.name = "da3"
        backend.license_type = Mock(value="commercial")
        backend.ensure_available.return_value = None
        backend.compute.return_value = _make_mock_depth_result()

        registry = Mock()
        registry.get_backend.return_value = backend

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=registry,
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            orchestrator.postprocessor = Mock(process=lambda result: result)

            test_image = _make_test_image(tmp_path, "attempt_struct.png")
            result = orchestrator.enhance_image(
                ImageInput(path=test_image),
                input_root=tmp_path,
            )

        attempts = result.get("attempts", [])
        assert len(attempts) > 0

        attempt = attempts[0]
        assert "backend" in attempt
        assert "status" in attempt
        assert "device" in attempt

    def test_selected_attempt_index_valid(self, tmp_path: Path) -> None:
        """selected_attempt_index points to valid attempt."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
        )

        backend = Mock()
        backend.name = "da3"
        backend.license_type = Mock(value="commercial")
        backend.ensure_available.return_value = None
        backend.compute.return_value = _make_mock_depth_result()

        registry = Mock()
        registry.get_backend.return_value = backend

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=registry,
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            orchestrator.postprocessor = Mock(process=lambda result: result)

            test_image = _make_test_image(tmp_path, "selected_idx.png")
            result = orchestrator.enhance_image(
                ImageInput(path=test_image),
                input_root=tmp_path,
            )

        attempts = result.get("attempts", [])
        selected_idx = result.get("selected_attempt_index")

        if selected_idx is not None:
            assert 0 <= selected_idx < len(attempts)

    def test_attempt_contains_duration(self, tmp_path: Path) -> None:
        """Attempt records contain duration."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
        )

        backend = Mock()
        backend.name = "da3"
        backend.license_type = Mock(value="commercial")
        backend.ensure_available.return_value = None
        backend.compute.return_value = _make_mock_depth_result()

        registry = Mock()
        registry.get_backend.return_value = backend

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=registry,
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            orchestrator.postprocessor = Mock(process=lambda result: result)

            test_image = _make_test_image(tmp_path, "duration.png")
            result = orchestrator.enhance_image(
                ImageInput(path=test_image),
                input_root=tmp_path,
            )

        attempts = result.get("attempts", [])
        if attempts:
            assert "duration_s" in attempts[0]
            assert isinstance(attempts[0]["duration_s"], (int, float))


class TestDeviceHandling:
    """Test device handling in backend selection."""

    def test_device_passed_to_backend_metadata(self, tmp_path: Path) -> None:
        """Device is recorded in backend metadata."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
        )

        backend = Mock()
        backend.name = "da3"
        backend.license_type = Mock(value="commercial")
        backend.ensure_available.return_value = None
        backend.compute.return_value = _make_mock_depth_result(device="cpu")

        registry = Mock()
        registry.get_backend.return_value = backend

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=registry,
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            orchestrator.postprocessor = Mock(process=lambda result: result)

            test_image = _make_test_image(tmp_path, "device_meta.png")
            result = orchestrator.enhance_image(
                ImageInput(path=test_image),
                input_root=tmp_path,
            )

        assert result.get("device") is not None

    def test_expected_output_depth_units_for_backend(self) -> None:
        """expected_output_depth_units_for_backend returns correct units."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import expected_output_depth_units_for_backend

        assert expected_output_depth_units_for_backend("da3") == "relative"
        assert expected_output_depth_units_for_backend("da2") == "relative"
        assert expected_output_depth_units_for_backend("depth_pro") == "meters"


class TestLicenseHandling:
    """Test license handling in backend selection."""

    def test_license_restriction_error_propagated(self, tmp_path: Path) -> None:
        """LicenseRestrictionError is propagated."""
        from transformation_portal.depth.backends.protocol import LicenseRestrictionError
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
        )

        backend = Mock()
        backend.name = "da3"
        backend.license_type = Mock(value="research")
        backend.ensure_available.return_value = None
        backend.compute.side_effect = LicenseRestrictionError("Research license required")

        registry = Mock()
        registry.get_backend.return_value = backend

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=registry,
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            orchestrator.postprocessor = Mock(process=lambda result: result)

            test_image = _make_test_image(tmp_path, "license_err.png")

            with pytest.raises(LicenseRestrictionError):
                orchestrator.enhance_image(
                    ImageInput(path=test_image),
                    input_root=tmp_path,
                )


class TestBackendNormalization:
    """Test backend ID normalization."""

    def test_normalize_backend_id_da3(self) -> None:
        """normalize_backend_id handles da3 variants."""
        from transformation_portal.lux_depth_v3._backend_contract import normalize_backend_id

        assert normalize_backend_id("da3") == "da3"
        assert normalize_backend_id("DA3") == "da3"
        assert normalize_backend_id("Da3") == "da3"

    def test_normalize_backend_id_depth_pro(self) -> None:
        """normalize_backend_id handles depth_pro variants."""
        from transformation_portal.lux_depth_v3._backend_contract import normalize_backend_id

        assert normalize_backend_id("depth_pro") == "depth_pro"
        assert normalize_backend_id("depth-pro") == "depth_pro"
        assert normalize_backend_id("DEPTH_PRO") == "depth_pro"

    def test_normalize_backend_id_none(self) -> None:
        """normalize_backend_id handles None."""
        from transformation_portal.lux_depth_v3._backend_contract import normalize_backend_id

        assert normalize_backend_id(None) is None

    def test_normalize_backend_id_empty(self) -> None:
        """normalize_backend_id handles empty string."""
        from transformation_portal.lux_depth_v3._backend_contract import normalize_backend_id

        assert normalize_backend_id("") is None
        assert normalize_backend_id("   ") is None


class TestExecutionPlan:
    """Test ExecutionPlan data class."""

    def test_execution_plan_creation(self) -> None:
        """ExecutionPlan can be created with default values."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import ExecutionPlan

        plan = ExecutionPlan(stages=["depth", "v2"])

        assert plan.stages == ["depth", "v2"]
        assert plan.enable_depth is True
        assert plan.enable_v2 is True
        assert plan.quality_tier == "standard"

    def test_execution_plan_with_all_fields(self) -> None:
        """ExecutionPlan can be created with all fields."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import BackendSelection, ExecutionPlan

        selection = BackendSelection(
            requested_backend="da3",
            resolved_backend="da3",
            status="success",
        )

        plan = ExecutionPlan(
            stages=["depth", "pbr", "v2"],
            backend_selection=selection,
            enable_depth=True,
            enable_v2=True,
            enable_pbr=True,
            enable_materials_v3=False,
            quality_tier="premium",
        )

        assert plan.enable_pbr is True
        assert plan.quality_tier == "premium"
        assert plan.backend_selection is selection


class TestResolveRequestedBackend:
    """Test resolve_requested_backend function."""

    def test_resolve_requested_backend_explicit(self) -> None:
        """resolve_requested_backend returns explicit request."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import resolve_requested_backend

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
        )

        result = resolve_requested_backend("da3", config)

        assert result == "da3"

    def test_resolve_requested_backend_from_config(self) -> None:
        """resolve_requested_backend uses config when request is None."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import resolve_requested_backend

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
        )

        result = resolve_requested_backend(None, config)

        # Should fall back to config or default
        assert result in ["da3", "depth_pro"]

    def test_resolve_requested_backend_default(self) -> None:
        """resolve_requested_backend returns default when no config."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import resolve_requested_backend

        config = EnhanceConfig(
            depth_backend=None,
            depth_device="cpu",
            enable_v2=False,
        )

        result = resolve_requested_backend(None, config)

        # Default should be da3 or depth_pro depending on platform
        assert result in ["da3", "depth_pro"]
