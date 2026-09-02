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

from unittest.mock import MagicMock

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
            ActiveDepthState,
            BackendSelection,
            ExecutionPlan,
            PipelineCoordinator,
            build_active_depth_state,
            build_backend_metadata_for_attempts,
            default_model_id_for_backend,
            derive_model_id_from_backend_instance,
            expected_output_depth_units_for_backend,
            extract_model_artifact_from_attempts,
            extract_model_id_from_attempts,
            get_or_create_depth_backend,
            infer_operational_error_code,
            normalize_sha256,
            resolve_backend_model_artifact,
            resolve_backend_model_id,
            resolve_runtime_backend_chain,
            seed_depth_attempts_from_selection_fallback,
            select_backend,
            typed_nullary_callable,
        )

        assert PipelineCoordinator is not None
        assert ActiveDepthState is not None
        assert BackendSelection is not None
        assert ExecutionPlan is not None
        assert callable(resolve_runtime_backend_chain)
        assert callable(select_backend)
        assert callable(default_model_id_for_backend)
        assert callable(derive_model_id_from_backend_instance)
        assert callable(resolve_backend_model_id)
        assert callable(expected_output_depth_units_for_backend)
        assert callable(normalize_sha256)
        assert callable(typed_nullary_callable)
        assert callable(resolve_backend_model_artifact)
        assert callable(extract_model_id_from_attempts)
        assert callable(extract_model_artifact_from_attempts)
        assert callable(infer_operational_error_code)
        assert callable(seed_depth_attempts_from_selection_fallback)
        assert callable(get_or_create_depth_backend)
        assert callable(build_active_depth_state)
        assert callable(build_backend_metadata_for_attempts)


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


class TestRuntimeBackendStateHelpers:
    """Test runtime backend state helpers extracted from orchestrator."""

    def test_normalize_sha256_accepts_lowercase_hex_only(self):
        """Test digest normalization rejects non-SHA256 values."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            normalize_sha256,
        )

        digest = "A" * 64

        assert normalize_sha256(f" {digest} ") == digest.lower()
        assert normalize_sha256("not-a-digest") is None
        assert normalize_sha256(None) is None

    def test_resolve_backend_model_artifact_uses_depth_pro_metadata(self):
        """Test Depth Pro artifact provenance comes from checkpoint metadata."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            resolve_backend_model_artifact,
        )

        digest = "B" * 64

        artifact = resolve_backend_model_artifact(
            "depth_pro",
            result_metadata={
                "checkpoint": {
                    "path": "/models/depth-pro-local.pt",
                    "sha256": digest,
                }
            },
        )

        assert artifact == {
            "model_artifact_filename": "depth-pro-local.pt",
            "model_artifact_sha256": digest.lower(),
        }

    def test_resolve_backend_model_artifact_uses_depth_pro_backend_fallback(self):
        """Test backend attributes are best-effort fallback provenance."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            resolve_backend_model_artifact,
        )

        digest = "C" * 64
        backend = MagicMock(spec=["_checkpoint_path", "_checkpoint_hash_cached", "_get_checkpoint_hash"])
        backend._checkpoint_path = "/runtime/depth-pro-fallback.pt"
        backend._checkpoint_hash_cached = None
        backend._get_checkpoint_hash.return_value = digest

        artifact = resolve_backend_model_artifact(
            "depth_pro",
            result_metadata={"checkpoint": {}},
            backend=backend,
        )

        assert artifact == {
            "model_artifact_filename": "depth-pro-fallback.pt",
            "model_artifact_sha256": digest.lower(),
        }

    def test_extract_model_id_from_attempts_prefers_selected_attempt(self):
        """Test selected runtime attempt wins over earlier successful attempts."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            extract_model_id_from_attempts,
        )

        attempts = [
            {"backend": "da3", "status": "success", "model_id": "model/first"},
            {"backend": "da3", "status": "success", "model_id": "model/selected"},
        ]

        assert (
            extract_model_id_from_attempts(
                "da3",
                attempts,
                selected_attempt_index=1,
            )
            == "model/selected"
        )

    def test_extract_model_artifact_from_attempts_prefers_selected_attempt(self):
        """Test selected runtime attempt wins for artifact identity."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            extract_model_artifact_from_attempts,
        )

        selected_digest = "D" * 64
        attempts = [
            {
                "backend": "depth_pro",
                "status": "success",
                "model_artifact_filename": "first.pt",
                "model_artifact_sha256": "E" * 64,
            },
            {
                "backend": "depth_pro",
                "status": "success",
                "model_artifact_filename": "selected.pt",
                "model_artifact_sha256": selected_digest,
            },
        ]

        assert extract_model_artifact_from_attempts(
            "depth_pro",
            attempts,
            selected_attempt_index=1,
        ) == {
            "model_artifact_filename": "selected.pt",
            "model_artifact_sha256": selected_digest.lower(),
        }

    def test_infer_operational_error_code_classifies_known_failures(self):
        """Test backend error classification stays stable."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            infer_operational_error_code,
        )

        assert infer_operational_error_code(ImportError("missing")) == "BACKEND_IMPORT_ERROR"
        assert infer_operational_error_code(FileNotFoundError("missing")) == "BACKEND_RESOURCE_MISSING"
        assert infer_operational_error_code(RuntimeError("cuda not available")) == "CUDA_UNAVAILABLE"
        assert infer_operational_error_code(RuntimeError("mps not available")) == "MPS_UNAVAILABLE"
        assert (
            infer_operational_error_code(
                RuntimeError(
                    '{"cuda_available": false, "device": "mps", '
                    '"mps_available": false, '
                    '"reason": "PyTorch MPS backend is not available in this runtime."}'
                )
            )
            == "MPS_UNAVAILABLE"
        )

    def test_seed_depth_attempts_from_selection_fallback_records_depth_pro_failure(self):
        """Test startup fallback is materialized into per-image attempt history."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.manifest import BackendSelectionMetadata
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            seed_depth_attempts_from_selection_fallback,
        )

        metadata = BackendSelectionMetadata(
            requested_backend="depth_pro",
            resolved_backend="da3",
            resolution_status="fallback",
            resolution_reason="Depth Pro unavailable",
            model_id="depth-anything/Depth-Anything-3-Metric-Large",
            device="cpu",
        )
        config = EnhanceConfig(
            depth_device="cpu",
            depth_pro_checkpoint_path="/models/depth-pro-local.pt",
        )

        attempts = seed_depth_attempts_from_selection_fallback(
            metadata,
            {"depth_pro": "checkpoint not found"},
            config,
            ModelVariant.METRIC_LARGE,
        )

        assert attempts == [
            {
                "attempt": 0,
                "backend": "depth_pro",
                "model_id": "apple/ml-depth-pro",
                "device": "cpu",
                "status": "failed",
                "failure_kind": "operational",
                "error_code": "BACKEND_RUNTIME_ERROR",
                "error_message": "checkpoint not found",
                "apex_gate_passed": False,
                "cached": False,
                "duration_s": 0.0,
                "model_artifact_filename": "depth-pro-local.pt",
                "model_artifact_sha256": None,
            }
        ]

    def test_get_or_create_depth_backend_prefers_matching_active_backend(self):
        """Test active backend injection is preserved over stale cache entries."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            get_or_create_depth_backend,
        )

        active_backend = MagicMock()
        active_backend.name = "da3"
        stale_backend = MagicMock()
        registry = MagicMock()
        backend_cache = {"da3": stale_backend}

        result = get_or_create_depth_backend(
            "da3",
            active_backend=active_backend,
            backend_cache=backend_cache,
            registry=registry,
            config=EnhanceConfig(),
        )

        assert result is active_backend
        assert backend_cache["da3"] is active_backend
        registry.get_backend.assert_not_called()

    def test_build_active_depth_state_copies_attempts(self):
        """Test active depth state owns its attempt list copy."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            build_active_depth_state,
        )

        attempts = [{"backend": "da3", "status": "success"}]

        state = build_active_depth_state(None, attempts, 0)

        assert state.backend_metadata is None
        assert state.depth_attempts == attempts
        assert state.depth_attempts is not attempts
        assert state.selected_attempt_index == 0

    def test_build_backend_metadata_for_attempts_records_runtime_fallback(self):
        """Test per-image backend metadata preserves fallback audit semantics."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.manifest import BackendSelectionMetadata
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            build_backend_metadata_for_attempts,
        )

        startup_metadata = BackendSelectionMetadata(
            requested_backend="da3",
            resolved_backend="da3",
            resolution_status="success",
            resolution_reason=None,
            model_id="model/startup",
            device="cpu",
        )
        attempts = [
            {
                "backend": "da3",
                "status": "failed",
                "failure_kind": "operational",
                "error_code": "CUDA_UNAVAILABLE",
            },
            {
                "backend": "da2",
                "status": "success",
                "model_id": "model/selected-da2",
            },
        ]
        resolve_model_id = MagicMock(return_value="model/fallback")

        metadata = build_backend_metadata_for_attempts(
            "da2",
            attempts,
            startup_metadata,
            EnhanceConfig(depth_device="cpu"),
            resolve_model_id,
            selected_attempt_index=1,
        )

        assert metadata.requested_backend == "da3"
        assert metadata.resolved_backend == "da2"
        assert metadata.resolution_status == "fallback"
        assert metadata.resolution_reason == ("Fallback from 'da3' to 'da2' after operational failure (CUDA_UNAVAILABLE)")
        assert metadata.model_id == "model/selected-da2"
        assert metadata.device == "cpu"
        assert metadata.attempts == attempts
        resolve_model_id.assert_not_called()

    def test_carried_da2_device_drives_startup_and_per_image_metadata(self, tmp_path):
        """Canonical DA2 CPU normalization remains truthful in emitted metadata."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
        from transformation_portal.lux_depth_v3.execution_lifecycle import (
            backend_candidate_authority,
            prepare_lux_execution,
        )
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            build_backend_metadata_for_attempts,
            initialize_depth_backend_state,
        )

        input_root = tmp_path / "inputs"
        input_root.mkdir()
        image = input_root / "sample.jpg"
        image.write_bytes(b"not-decoded-during-plan-preparation")
        prepared = prepare_lux_execution(
            EnhanceConfig(
                model_key="da3-metric",
                depth_device="cuda",
                enable_v2=False,
            ),
            input_root,
            [image],
        )

        class Backend:
            def __init__(self, name: str, available: bool) -> None:
                self.name = name
                self.available = available

            def ensure_available(self) -> None:
                if not self.available:
                    raise ImportError("primary unavailable")

        class Registry:
            @staticmethod
            def get_backend(
                backend_id,
                config,
                *,
                candidate_authority,
                canonical_plan_bytes,
            ):
                assert canonical_plan_bytes == prepared.canonical_plan_bytes
                if backend_id == "da2":
                    assert candidate_authority == backend_candidate_authority(prepared.plan, "da2")
                    assert config.depth_device == "cpu"
                return Backend(backend_id, available=backend_id == "da2")

        state = initialize_depth_backend_state(
            prepared.runtime_config,
            ModelVariant.METRIC_LARGE,
            lambda backend_id, **_kwargs: f"model/{backend_id}",
            registry_factory=Registry,
        )

        assert prepared.runtime_config.depth_device == "cuda"
        assert state.backend_metadata.resolved_backend == "da2"
        assert state.backend_metadata.device == "cpu"

        attempts = [{"backend": "da2", "status": "success", "model_id": "model/da2", "device": "cpu"}]
        per_image = build_backend_metadata_for_attempts(
            "da2",
            attempts,
            state.backend_metadata,
            prepared.runtime_config,
            lambda backend_id, **_kwargs: f"model/{backend_id}",
            selected_attempt_index=0,
        )

        assert per_image.device == "cpu"


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

    def test_to_metadata_raises_on_error_selection(self):
        """Test that to_metadata raises ValueError for error selections with None resolved_backend."""
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            BackendSelection,
        )

        selection = BackendSelection(
            requested_backend="da3",
            resolved_backend=None,
            status="error",
        )

        with pytest.raises(ValueError, match="Cannot convert error selection to metadata"):
            selection.to_metadata()


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
