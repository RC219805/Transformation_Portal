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
from unittest.mock import MagicMock

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
        """Test that legacy imports from orchestrator still work.

        Note: The orchestrator has its own _generate_pbr_stage and _run_v2_stage
        methods that are NOT the same as the standalone functions because they have
        different signatures and return types. The standalone functions are the new
        canonical API, while the orchestrator methods preserve backward compatibility.
        """
        from transformation_portal.lux_depth_v3.orchestrator import (
            DepthStageResult,
            ExecutionEngine,
            MaterialsV3StageResult,
            PBRStageResult,
            V2StageResult,
            generate_pbr_stage,
            run_v2_stage,
        )

        # Verify all classes imported correctly
        assert DepthStageResult is not None
        assert ExecutionEngine is not None
        assert MaterialsV3StageResult is not None
        assert PBRStageResult is not None
        assert V2StageResult is not None

        # Verify functions are callable
        assert callable(generate_pbr_stage)
        assert callable(run_v2_stage)


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
        # Verify timing is captured (should be fast but measurable)
        assert 0.0 <= result.runtime_s < 30.0
        assert result.config is not None
        # Verify output files exist
        assert Path(result.normal_path).exists()
        assert Path(result.roughness_path).exists()
        assert Path(result.ao_path).exists()


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


class TestDepthArtifactPaths:
    """Test DepthArtifactPaths data class."""

    def test_default_values(self):
        """Test default initialization."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthArtifactPaths,
        )

        paths = DepthArtifactPaths(depth_path=Path("/test/depth.png"))

        assert paths.depth_path == Path("/test/depth.png")
        assert paths.float_depth_path is None
        assert paths.metadata_path is None

    def test_with_all_values(self):
        """Test initialization with all values."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthArtifactPaths,
        )

        paths = DepthArtifactPaths(
            depth_path=Path("/test/depth.png"),
            float_depth_path=Path("/test/depth.npy"),
            metadata_path=Path("/test/depth_metadata.json"),
        )

        assert paths.depth_path == Path("/test/depth.png")
        assert paths.float_depth_path == Path("/test/depth.npy")
        assert paths.metadata_path == Path("/test/depth_metadata.json")


class TestDepthArtifactResult:
    """Test DepthArtifactResult data class."""

    def test_default_values(self):
        """Test default initialization."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthArtifactResult,
        )

        result = DepthArtifactResult()

        assert result.success is False
        assert result.depth_path is None
        assert result.float_depth_path is None
        assert result.metadata_path is None
        assert result.scaling_stats is None
        assert result.error is None

    def test_success_case(self):
        """Test success initialization."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthArtifactResult,
        )

        result = DepthArtifactResult(
            success=True,
            depth_path=Path("/test/depth.png"),
            metadata_path=Path("/test/depth_metadata.json"),
            scaling_stats={"min": 0.0, "max": 1.0},
        )

        assert result.success is True
        assert result.depth_path == Path("/test/depth.png")
        assert result.scaling_stats == {"min": 0.0, "max": 1.0}


class TestEnhancedImageResult:
    """Test EnhancedImageResult data class."""

    def test_default_values(self):
        """Test default initialization."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            EnhancedImageResult,
        )

        result = EnhancedImageResult()

        assert result.success is False
        assert result.output_path is None
        assert result.format == "png"
        assert result.bit_depth == 8
        assert result.n_operations_applied == 0
        assert result.error is None

    def test_success_case_png(self):
        """Test success case with PNG output."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            EnhancedImageResult,
        )

        result = EnhancedImageResult(
            success=True,
            output_path=Path("/test/enhanced.png"),
            format="png",
            bit_depth=8,
            n_operations_applied=5,
        )

        assert result.success is True
        assert result.output_path == Path("/test/enhanced.png")
        assert result.format == "png"
        assert result.bit_depth == 8
        assert result.n_operations_applied == 5

    def test_success_case_tiff(self):
        """Test success case with TIFF output."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            EnhancedImageResult,
        )

        result = EnhancedImageResult(
            success=True,
            output_path=Path("/test/enhanced.tif"),
            format="tiff",
            bit_depth=16,
            n_operations_applied=3,
        )

        assert result.success is True
        assert result.format == "tiff"
        assert result.bit_depth == 16


class TestPersistDepthArtifacts:
    """Test persist_depth_artifacts function."""

    def test_persist_depth_success(self, tmp_path):
        """Test successful depth artifact persistence."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_engine import (
            persist_depth_artifacts,
        )
        from transformation_portal.lux_depth_v3.manifest import DepthMetadata

        config = EnhanceConfig(depth_quantization="u16", verify_depth_writes=False)
        import numpy as np

        depth_map = np.random.rand(100, 100).astype(np.float32)
        depth_path = tmp_path / "depth" / "test_depth.png"
        depth_path.parent.mkdir(parents=True, exist_ok=True)

        depth_metadata = DepthMetadata(
            model="da3",
            depth_path=str(depth_path),
            runtime_seconds=1.5,
            scaling={},
            stats={},
        )

        result = persist_depth_artifacts(
            depth_map=depth_map,
            depth_path=depth_path,
            float_depth_path=None,
            depth_metadata=depth_metadata,
            config=config,
        )

        assert result.success is True
        assert result.depth_path == depth_path
        assert result.depth_path.exists()
        assert result.metadata_path is not None
        assert result.metadata_path.exists()


class TestPersistEnhancedImage:
    """Test persist_enhanced_image function."""

    def test_persist_enhanced_image_png(self, tmp_path):
        """Test successful 8-bit PNG enhanced image persistence."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_engine import (
            persist_enhanced_image,
        )

        config = EnhanceConfig(emit_master16=False, emit_upscaled16=False)
        import numpy as np

        # Create RGB image
        enhanced_image = np.random.rand(100, 100, 3).astype(np.float32)
        output_path = tmp_path / "temp" / "test_enhanced.png"

        result = persist_enhanced_image(
            enhanced_image=enhanced_image,
            output_path=output_path,
            config=config,
            n_operations_applied=5,
        )

        assert result.success is True
        assert result.format == "png"
        assert result.bit_depth == 8
        assert result.n_operations_applied == 5
        assert result.output_path is not None
        assert result.output_path.exists()


class TestExecutionEngineNewMethods:
    """Test new ExecutionEngine methods for depth and enhanced image persistence."""

    def test_persist_depth_method(self, tmp_path):
        """Test ExecutionEngine.persist_depth method."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_engine import ExecutionEngine
        from transformation_portal.lux_depth_v3.manifest import DepthMetadata

        config = EnhanceConfig(depth_quantization="u16", verify_depth_writes=False)
        engine = ExecutionEngine(config=config, output_root=tmp_path)

        import numpy as np

        depth_map = np.random.rand(100, 100).astype(np.float32)
        depth_metadata = DepthMetadata(
            model="da3",
            depth_path="/test/depth.png",
            runtime_seconds=1.5,
            scaling={},
            stats={},
        )

        result = engine.persist_depth(
            depth_map=depth_map,
            output_key=Path("test_image"),
            depth_metadata=depth_metadata,
        )

        assert result.success is True
        assert result.depth_path is not None
        assert result.depth_path.exists()

    def test_persist_enhanced_method(self, tmp_path):
        """Test ExecutionEngine.persist_enhanced method."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.execution_engine import ExecutionEngine

        config = EnhanceConfig(emit_master16=False, emit_upscaled16=False)
        engine = ExecutionEngine(config=config, output_root=tmp_path)

        import numpy as np

        enhanced_image = np.random.rand(100, 100, 3).astype(np.float32)

        result = engine.persist_enhanced(
            enhanced_image=enhanced_image,
            output_key=Path("test_image"),
            n_operations_applied=3,
        )

        assert result.success is True
        assert result.format == "png"
        assert result.output_path is not None
        assert result.output_path.exists()


class TestNewImports:
    """Test new imports from execution_engine module."""

    def test_import_new_result_classes(self):
        """Test that we can import new result data classes."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            DepthArtifactPaths,
            DepthArtifactResult,
            EnhancedImageResult,
        )

        assert DepthArtifactPaths is not None
        assert DepthArtifactResult is not None
        assert EnhancedImageResult is not None

    def test_import_new_functions(self):
        """Test that we can import new standalone functions."""
        from transformation_portal.lux_depth_v3.execution_engine import (
            persist_depth_artifacts,
            persist_enhanced_image,
        )

        assert callable(persist_depth_artifacts)
        assert callable(persist_enhanced_image)

    def test_backward_compatible_orchestrator_imports_new(self):
        """Test that new imports from orchestrator work."""
        from transformation_portal.lux_depth_v3.orchestrator import (
            DepthArtifactPaths,
            DepthArtifactResult,
            EnhancedImageResult,
            persist_depth_artifacts,
            persist_enhanced_image,
        )

        assert DepthArtifactPaths is not None
        assert DepthArtifactResult is not None
        assert EnhancedImageResult is not None
        assert callable(persist_depth_artifacts)
        assert callable(persist_enhanced_image)
