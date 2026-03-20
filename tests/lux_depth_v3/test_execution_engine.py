"""Unit tests for ExecutionEngine.

Tests the execution engine logic extracted from orchestrator.py
as part of ADR-043 decomposition (Phase 6).

These tests verify:
1. Result data class behaviors
2. PBR stage generation
3. V2 stage execution
4. Import compatibility
5. Backward compatibility with orchestrator
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]


class TestExecutionEngineImports:
    """Test that imports work from the new module."""

    def test_import_result_classes(self):
        """Test that we can import result data classes."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthStageResult,
            MaterialsV3StageResult,
            PBRStageResult,
            V2StageResult,
        )

        assert DepthStageResult is not None
        assert PBRStageResult is not None
        assert MaterialsV3StageResult is not None
        assert V2StageResult is not None

    def test_import_functions(self):
        """Test that we can import standalone functions."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            generate_pbr_stage,
            run_v2_stage,
        )

        assert callable(generate_pbr_stage)
        assert callable(run_v2_stage)

    def test_import_engine_class(self):
        """Test that we can import ExecutionEngine class."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            ExecutionEngine,
        )

        assert ExecutionEngine is not None

    def test_backward_compatible_orchestrator_imports(self):
        """Test that legacy imports from orchestrator still work."""
        try:
            from transformation_portal.lux_depth_v3.orchestrator import (
                DepthStageResult,
                ExecutionEngine,
                MaterialsV3StageResult,
                PBRStageResult,
                V2StageResult,
                _generate_pbr_stage,
                _run_v2_stage,
                generate_pbr_stage,
                run_v2_stage,
            )
        except ImportError as e:
            # Skip if cv2 or other heavy deps are not available
            # This can happen in minimal test environments
            if "cv2" in str(e) or "torch" in str(e):
                pytest.skip(f"Skipping due to missing optional dependency: {e}")
            raise

        # Verify all classes imported correctly
        assert DepthStageResult is not None
        assert ExecutionEngine is not None
        assert MaterialsV3StageResult is not None
        assert PBRStageResult is not None
        assert V2StageResult is not None

        # Verify functions are callable
        assert callable(generate_pbr_stage)
        assert callable(run_v2_stage)
        assert callable(_generate_pbr_stage)
        assert callable(_run_v2_stage)

        # Verify aliases point to the same functions
        assert _generate_pbr_stage is generate_pbr_stage
        assert _run_v2_stage is run_v2_stage


class TestDepthStageResult:
    """Test DepthStageResult data class."""

    def test_default_values(self):
        """Test default initialization."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthStageResult,
        )

        result = DepthStageResult()

        assert result.depth_metadata is None
        assert result.depth_runtime_s == 0.0
        assert result.pbr_assets is None
        assert result.materials_v3_result is None
        assert result.materials_v3_runtime_s == 0.0
        assert result.enhanced_image_path is None
        assert result.backend_selection is None
        assert result.depth_attempts == []
        assert result.selected_attempt_index is None
        assert result.depth_map is None

    def test_success_property_with_metadata(self):
        """Test success property returns True when depth_metadata exists."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthStageResult,
        )
        from transformation_portal.lux_depth_v3.manifest import DepthMetadata

        metadata = DepthMetadata(
            model="da3",
            depth_path="/test/depth.png",
            runtime_seconds=1.5,
            scaling={},
            stats={},
        )
        result = DepthStageResult(depth_metadata=metadata)

        assert result.success is True

    def test_success_property_without_metadata(self):
        """Test success property returns False when depth_metadata is None."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthStageResult,
        )

        result = DepthStageResult()

        assert result.success is False

    def test_was_cached_property_false(self):
        """Test was_cached returns False when no attempts are cached."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthStageResult,
        )

        result = DepthStageResult(
            depth_attempts=[
                {"backend": "da3", "cached": False},
                {"backend": "da2", "cached": False},
            ]
        )

        assert result.was_cached is False

    def test_was_cached_property_true(self):
        """Test was_cached returns True when any attempt is cached."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthStageResult,
        )

        result = DepthStageResult(
            depth_attempts=[
                {"backend": "da3", "cached": True},
            ]
        )

        assert result.was_cached is True

    def test_to_tuple_format(self):
        """Test to_tuple returns correct format."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthStageResult,
        )
        from transformation_portal.lux_depth_v3.manifest import (
            BackendSelectionMetadata,
            DepthMetadata,
        )

        metadata = DepthMetadata(
            model="da3",
            depth_path="/test/depth.png",
            runtime_seconds=1.5,
            scaling={},
            stats={},
        )
        backend_sel = BackendSelectionMetadata(
            requested_backend="da3",
            resolved_backend="da3",
            resolution_status="success",
            resolution_reason=None,
            model_id="depth-anything/Depth-Anything-V3-Large-hf",
            device="cpu",
        )
        result = DepthStageResult(
            depth_metadata=metadata,
            depth_runtime_s=1.5,
            pbr_assets={"normal_path": "/test/normal.png"},
            backend_selection=backend_sel,
            depth_attempts=[{"backend": "da3"}],
        )

        tup = result.to_tuple()

        assert len(tup) == 8
        assert tup[0] == metadata
        assert tup[1] == 1.5
        assert tup[2] == {"normal_path": "/test/normal.png"}
        assert tup[3] is None  # materials_v3_result
        assert tup[4] == 0.0  # materials_v3_runtime_s
        assert tup[5] is None  # enhanced_image_path
        assert tup[6] == backend_sel
        assert tup[7] == [{"backend": "da3"}]


class TestPBRStageResult:
    """Test PBRStageResult data class."""

    def test_default_values(self):
        """Test default initialization."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            PBRStageResult,
        )

        result = PBRStageResult()

        assert result.success is False
        assert result.normal_path is None
        assert result.roughness_path is None
        assert result.ao_path is None
        assert result.runtime_s == 0.0
        assert result.config is None
        assert result.error is None

    def test_to_dict_when_success(self):
        """Test to_dict returns manifest format when successful."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            PBRStageResult,
        )

        result = PBRStageResult(
            success=True,
            normal_path="/test/normal.png",
            roughness_path="/test/roughness.png",
            ao_path="/test/ao.png",
            runtime_s=0.5,
            config={"normal_strength": 1.0},
        )

        d = result.to_dict()

        assert d is not None
        assert d["normal_path"] == "/test/normal.png"
        assert d["roughness_path"] == "/test/roughness.png"
        assert d["ao_path"] == "/test/ao.png"
        assert d["runtime_seconds"] == 0.5
        assert d["config"]["normal_strength"] == 1.0

    def test_to_dict_when_failed(self):
        """Test to_dict returns None when failed."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            PBRStageResult,
        )

        result = PBRStageResult(success=False, error="Test error")

        assert result.to_dict() is None


class TestMaterialsV3StageResult:
    """Test MaterialsV3StageResult data class."""

    def test_default_values(self):
        """Test default initialization."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            MaterialsV3StageResult,
        )

        result = MaterialsV3StageResult()

        assert result.success is False
        assert result.result is None
        assert result.runtime_s == 0.0
        assert result.enhanced_image_path is None
        assert result.mask_artifact_path is None
        assert result.n_operations_applied == 0
        assert result.error is None

    def test_to_tuple_format(self):
        """Test to_tuple returns correct format."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            MaterialsV3StageResult,
        )

        result = MaterialsV3StageResult(
            success=True,
            result={"material_masks": {}},
            runtime_s=2.0,
            enhanced_image_path=Path("/test/enhanced.png"),
        )

        tup = result.to_tuple()

        assert len(tup) == 3
        assert tup[0] == {"material_masks": {}}
        assert tup[1] == 2.0
        assert tup[2] == Path("/test/enhanced.png")


class TestV2StageResult:
    """Test V2StageResult data class."""

    def test_default_values(self):
        """Test default initialization."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            V2StageResult,
        )

        result = V2StageResult()

        assert result.result == {}
        assert result.runtime_s == 0.0
        assert result.report_path is None
        assert result.output_path is None
        assert result.status == "unknown"
        assert result.skipped is False

    def test_success_property_with_ok_status(self):
        """Test success property returns True for ok status."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            V2StageResult,
        )

        result = V2StageResult(status="ok")
        assert result.success is True

    def test_success_property_with_skipped_status(self):
        """Test success property returns True for skipped status."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            V2StageResult,
        )

        result = V2StageResult(status="skipped", skipped=True)
        assert result.success is True

    def test_success_property_with_error_status(self):
        """Test success property returns False for error status."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            V2StageResult,
        )

        result = V2StageResult(status="error")
        assert result.success is False

    def test_to_tuple_format(self):
        """Test to_tuple returns correct format."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            V2StageResult,
        )

        result = V2StageResult(
            result={"status": "ok", "output": "/test/enhanced.png"},
            runtime_s=3.5,
            report_path=Path("/test/report.json"),
        )

        tup = result.to_tuple()

        assert len(tup) == 3
        assert tup[0] == {"status": "ok", "output": "/test/enhanced.png"}
        assert tup[1] == 3.5
        assert tup[2] == Path("/test/report.json")


class TestGeneratePBRStage:
    """Test generate_pbr_stage function."""

    def test_disabled_returns_failure(self):
        """Test that disabled PBR returns failure result."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_engine import (
            generate_pbr_stage,
        )

        config = EnhanceConfig(generate_pbr=False)
        import numpy as np

        depth = np.zeros((100, 100), dtype=np.float32)

        result = generate_pbr_stage(
            depth=depth,
            output_key=Path("test_image"),
            output_root=Path("/tmp/test"),
            config=config,
        )

        assert result.success is False
        assert result.error is not None
        assert "disabled" in result.error.lower()

    def test_enabled_calls_generation(self, tmp_path):
        """Test that enabled PBR calls generation functions."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_engine import (
            generate_pbr_stage,
        )

        config = EnhanceConfig(generate_pbr=True)
        import numpy as np

        depth = np.random.rand(100, 100).astype(np.float32)

        result = generate_pbr_stage(
            depth=depth,
            output_key=Path("test_image"),
            output_root=tmp_path,
            config=config,
        )

        assert result.success is True
        assert result.normal_path is not None
        assert result.roughness_path is not None
        assert result.ao_path is not None
        assert result.runtime_s > 0
        assert result.config is not None


class TestRunV2Stage:
    """Test run_v2_stage function."""

    def test_disabled_returns_skipped(self):
        """Test that disabled V2 returns skipped result."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_engine import (
            run_v2_stage,
        )

        config = EnhanceConfig(enable_v2=False)
        v2_runner = MagicMock()

        result = run_v2_stage(
            v2_runner=v2_runner,
            image_path=Path("/test/image.png"),
            depth_path=None,
            depth_dir=Path("/test/depth"),
            v2_dir=Path("/test/v2"),
            output_key=Path("test_image"),
            v2_log_path=Path("/test/logs/v2.log"),
            config=config,
        )

        assert result.skipped is True
        assert result.status == "skipped"
        v2_runner.run.assert_not_called()

    def test_none_runner_returns_skipped(self):
        """Test that None runner returns skipped result."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_engine import (
            run_v2_stage,
        )

        config = EnhanceConfig(enable_v2=True)

        result = run_v2_stage(
            v2_runner=None,
            image_path=Path("/test/image.png"),
            depth_path=None,
            depth_dir=Path("/test/depth"),
            v2_dir=Path("/test/v2"),
            output_key=Path("test_image"),
            v2_log_path=Path("/test/logs/v2.log"),
            config=config,
        )

        assert result.skipped is True
        assert result.status == "skipped"


class TestExecutionEngine:
    """Test ExecutionEngine class."""

    def test_initialization(self, tmp_path):
        """Test engine initialization."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_engine import (
            ExecutionEngine,
        )

        config = EnhanceConfig(generate_pbr=True)
        engine = ExecutionEngine(config=config, output_root=tmp_path)

        assert engine.config == config
        assert engine.output_root == tmp_path
        assert engine.depth_dir == tmp_path / "depth"
        assert engine.v2_dir == tmp_path / "v2"
        assert engine.pbr_enabled is True

    def test_execute_pbr_stage(self, tmp_path):
        """Test execute_pbr_stage method."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_engine import (
            ExecutionEngine,
        )

        config = EnhanceConfig(generate_pbr=True)
        engine = ExecutionEngine(config=config, output_root=tmp_path)

        import numpy as np

        depth = np.random.rand(100, 100).astype(np.float32)

        result = engine.execute_pbr_stage(
            depth=depth,
            output_key=Path("test_image"),
        )

        assert result.success is True

    def test_execute_v2_stage_disabled(self, tmp_path):
        """Test execute_v2_stage method when disabled."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_engine import (
            ExecutionEngine,
        )

        config = EnhanceConfig(enable_v2=False)
        engine = ExecutionEngine(config=config, output_root=tmp_path)

        result = engine.execute_v2_stage(
            v2_runner=MagicMock(),
            image_path=Path("/test/image.png"),
            depth_path=None,
            output_key=Path("test_image"),
            v2_log_path=tmp_path / "logs" / "v2.log",
        )

        assert result.skipped is True


class TestBackwardCompatibility:
    """Test backward compatibility with orchestrator patterns."""

    def test_result_tuple_unpacking(self):
        """Test that results can be unpacked like legacy tuples."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            V2StageResult,
        )

        result = V2StageResult(
            result={"status": "ok"},
            runtime_s=1.0,
            report_path=Path("/test/report.json"),
        )

        # This is how orchestrator unpacks V2 results
        v2_result, v2_runtime_s, v2_report_path = result.to_tuple()

        assert v2_result == {"status": "ok"}
        assert v2_runtime_s == 1.0
        assert v2_report_path == Path("/test/report.json")

    def test_depth_result_tuple_unpacking(self):
        """Test that depth results can be unpacked like legacy tuples."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthStageResult,
        )
        from transformation_portal.lux_depth_v3.manifest import (
            BackendSelectionMetadata,
            DepthMetadata,
        )

        metadata = DepthMetadata(
            model="da3",
            depth_path="/test/depth.png",
            runtime_seconds=1.5,
            scaling={},
            stats={},
        )
        backend_sel = BackendSelectionMetadata(
            requested_backend="da3",
            resolved_backend="da3",
            resolution_status="success",
            resolution_reason=None,
            model_id="depth-anything/Depth-Anything-V3-Large-hf",
            device="cpu",
        )
        result = DepthStageResult(
            depth_metadata=metadata,
            depth_runtime_s=1.5,
            backend_selection=backend_sel,
        )

        # This is how orchestrator unpacks depth results
        (
            depth_metadata,
            depth_runtime_s,
            pbr_assets,
            materials_v3_result,
            materials_v3_runtime_s,
            enhanced_image_path,
            backend_selection_metadata,
            depth_attempts,
        ) = result.to_tuple()

        assert depth_metadata == metadata
        assert depth_runtime_s == 1.5
        assert pbr_assets is None
        assert materials_v3_result is None
        assert materials_v3_runtime_s == 0.0
        assert enhanced_image_path is None
        assert backend_selection_metadata == backend_sel
        assert depth_attempts == []
