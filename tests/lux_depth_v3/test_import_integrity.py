"""Test import integrity for ADR-043 decomposed modules.

This module verifies that the ADR-043 orchestrator decomposition maintains
proper import boundaries:
- Decomposed modules MUST NOT import from orchestrator (no circular deps)
- Orchestrator MUST re-export all public classes from decomposed modules
- Import paths MUST remain stable for backward compatibility

See: docs/architecture/ADR-043-orchestrator-decomposition.md
"""

from __future__ import annotations

import sys

import pytest

pytestmark = [pytest.mark.unit]


def _snapshot_orchestrator_modules() -> set:
    """Snapshot orchestrator-related modules currently loaded in sys.modules."""
    return {name for name in sys.modules if name.startswith("transformation_portal.lux_depth_v3.orchestrator")}


class TestNoCircularImports:
    """Verify decomposed modules don't create circular imports."""

    def test_execution_engine_does_not_import_orchestrator(self):
        """execution_engine.py should not import from orchestrator."""
        # Snapshot orchestrator modules BEFORE import
        pre = _snapshot_orchestrator_modules()

        from transformation_portal.lux_depth_v3 import execution_engine

        # Ensure orchestrator class is not exposed
        assert "EnhanceOrchestrator" not in dir(execution_engine)

        # Ensure import did NOT load orchestrator (no new orchestrator modules)
        post = _snapshot_orchestrator_modules()
        assert post == pre, f"execution_engine imported orchestrator: {post - pre}"

    def test_config_resolver_does_not_import_orchestrator(self):
        """config_resolver.py should not import from orchestrator."""
        pre = _snapshot_orchestrator_modules()

        from transformation_portal.lux_depth_v3 import config_resolver

        assert "EnhanceOrchestrator" not in dir(config_resolver)

        post = _snapshot_orchestrator_modules()
        assert post == pre, f"config_resolver imported orchestrator: {post - pre}"

    def test_pipeline_coordinator_does_not_import_orchestrator(self):
        """pipeline_coordinator.py should not import from orchestrator."""
        pre = _snapshot_orchestrator_modules()

        from transformation_portal.lux_depth_v3 import pipeline_coordinator

        assert "EnhanceOrchestrator" not in dir(pipeline_coordinator)

        post = _snapshot_orchestrator_modules()
        assert post == pre, f"pipeline_coordinator imported orchestrator: {post - pre}"

    def test_artifact_manager_does_not_import_orchestrator(self):
        """artifact_manager.py should not import from orchestrator."""
        pre = _snapshot_orchestrator_modules()

        from transformation_portal.lux_depth_v3 import artifact_manager

        assert "EnhanceOrchestrator" not in dir(artifact_manager)

        post = _snapshot_orchestrator_modules()
        assert post == pre, f"artifact_manager imported orchestrator: {post - pre}"

    def test_validators_does_not_import_orchestrator(self):
        """validators package should not import from orchestrator."""
        pre = _snapshot_orchestrator_modules()

        from transformation_portal.lux_depth_v3 import validators

        assert "EnhanceOrchestrator" not in dir(validators)

        post = _snapshot_orchestrator_modules()
        assert post == pre, f"validators imported orchestrator: {post - pre}"


class TestOrchestratorReexports:
    """Verify orchestrator re-exports all Phase 6 public classes."""

    def test_orchestrator_exports_execution_engine_classes(self):
        """Orchestrator should re-export all ExecutionEngine classes."""
        from transformation_portal.lux_depth_v3 import orchestrator

        # Phase 6 result data classes
        assert hasattr(orchestrator, "DepthStageResult")
        assert hasattr(orchestrator, "PBRStageResult")
        assert hasattr(orchestrator, "MaterialsV3StageResult")
        assert hasattr(orchestrator, "V2StageResult")

        # Phase 6 artifact result classes
        assert hasattr(orchestrator, "DepthArtifactPaths")
        assert hasattr(orchestrator, "DepthArtifactResult")
        assert hasattr(orchestrator, "EnhancedImageResult")

        # Phase 6 main class
        assert hasattr(orchestrator, "ExecutionEngine")

        # Phase 6 standalone functions
        assert hasattr(orchestrator, "generate_pbr_stage")
        assert hasattr(orchestrator, "run_v2_stage")
        assert hasattr(orchestrator, "persist_depth_artifacts")
        assert hasattr(orchestrator, "persist_enhanced_image")

    def test_orchestrator_exports_config_resolver_classes(self):
        """Orchestrator should re-export ConfigResolver classes."""
        from transformation_portal.lux_depth_v3 import orchestrator

        assert hasattr(orchestrator, "ConfigResolver")
        assert hasattr(orchestrator, "PresetInfo")
        assert hasattr(orchestrator, "ResolvedConfig")
        assert hasattr(orchestrator, "compute_config_fingerprint")
        assert hasattr(orchestrator, "discover_presets")
        assert hasattr(orchestrator, "resolve_preset")

    def test_orchestrator_exports_pipeline_coordinator_classes(self):
        """Orchestrator should re-export PipelineCoordinator classes."""
        from transformation_portal.lux_depth_v3 import orchestrator

        assert hasattr(orchestrator, "PipelineCoordinator")
        assert hasattr(orchestrator, "BackendSelection")
        assert hasattr(orchestrator, "ExecutionPlan")

    def test_orchestrator_exports_artifact_manager_classes(self):
        """Orchestrator should re-export ArtifactManager classes."""
        from transformation_portal.lux_depth_v3 import orchestrator

        assert hasattr(orchestrator, "ArtifactManager")
        assert hasattr(orchestrator, "build_artifact_index")
        assert hasattr(orchestrator, "compute_artifact_merkle_root")
        assert hasattr(orchestrator, "infer_artifact_type")
        assert hasattr(orchestrator, "make_output_key")

    def test_orchestrator_exports_validators(self):
        """Orchestrator should re-export validator functions."""
        from transformation_portal.lux_depth_v3 import orchestrator

        assert hasattr(orchestrator, "validate_run_card_payload")
        assert hasattr(orchestrator, "validate_run_card_backend_semantics")


class TestBackwardCompatibleImports:
    """Verify legacy import paths continue to work."""

    def test_legacy_import_from_orchestrator_execution_engine(self):
        """Legacy imports from orchestrator for execution_engine items work."""
        # These imports should work without raising ImportError
        from transformation_portal.lux_depth_v3.orchestrator import (
            DepthStageResult,
            ExecutionEngine,
            MaterialsV3StageResult,
            PBRStageResult,
            V2StageResult,
            generate_pbr_stage,
            run_v2_stage,
        )

        assert DepthStageResult is not None
        assert ExecutionEngine is not None
        assert MaterialsV3StageResult is not None
        assert PBRStageResult is not None
        assert V2StageResult is not None
        assert callable(generate_pbr_stage)
        assert callable(run_v2_stage)

    def test_legacy_import_from_orchestrator_config_resolver(self):
        """Legacy imports from orchestrator for config_resolver items work."""
        from transformation_portal.lux_depth_v3.orchestrator import (
            ConfigResolver,
            PresetInfo,
            ResolvedConfig,
            compute_config_fingerprint,
            discover_presets,
        )

        assert ConfigResolver is not None
        assert PresetInfo is not None
        assert ResolvedConfig is not None
        assert callable(compute_config_fingerprint)
        assert callable(discover_presets)

    def test_legacy_import_from_orchestrator_artifact_manager(self):
        """Legacy imports from orchestrator for artifact_manager items work."""
        from transformation_portal.lux_depth_v3.orchestrator import (
            ArtifactManager,
            build_artifact_index,
            compute_artifact_merkle_root,
            infer_artifact_type,
            make_output_key,
        )

        assert ArtifactManager is not None
        assert callable(build_artifact_index)
        assert callable(compute_artifact_merkle_root)
        assert callable(infer_artifact_type)
        assert callable(make_output_key)

    def test_new_canonical_import_paths(self):
        """New canonical import paths from decomposed modules work."""
        # artifact_manager canonical imports
        from transformation_portal.lux_depth_v3.artifact_manager import (
            ArtifactManager,
        )

        # config_resolver canonical imports
        from transformation_portal.lux_depth_v3.config_resolver import (
            ConfigResolver,
            PresetInfo,
            ResolvedConfig,
        )

        # execution_engine canonical imports
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthArtifactPaths,
            DepthArtifactResult,
            DepthStageResult,
            EnhancedImageResult,
            ExecutionEngine,
            MaterialsV3StageResult,
            PBRStageResult,
            V2StageResult,
            generate_pbr_stage,
            persist_depth_artifacts,
            persist_enhanced_image,
            run_v2_stage,
        )

        # pipeline_coordinator canonical imports
        from transformation_portal.lux_depth_v3.pipeline_coordinator import (
            BackendSelection,
            ExecutionPlan,
            PipelineCoordinator,
        )

        # validators canonical imports
        from transformation_portal.lux_depth_v3.validators import (
            validate_run_card_backend_semantics,
            validate_run_card_payload,
        )

        # Verify all canonical imports succeeded (import errors would fail above)
        assert DepthStageResult is not None
        assert ExecutionEngine is not None
        assert ConfigResolver is not None
        assert ArtifactManager is not None
        assert PipelineCoordinator is not None
