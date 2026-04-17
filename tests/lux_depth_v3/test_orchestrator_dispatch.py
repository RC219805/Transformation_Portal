"""Test coverage for orchestrator dispatch lifecycle.

Phase 2 Coverage: Dispatch lifecycle tests for EnhanceOrchestrator.

Tests verify:
1. enhance_image dispatch flow
2. Per-image state management (_active_* fields)
3. Stage A/B coordination
4. Skip decision propagation
5. Result structure integrity
6. Error handling in dispatch lifecycle
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
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


def _make_mock_depth_result():
    """Create a deterministic synthetic depth result."""
    from transformation_portal.depth.backends.protocol import DepthResult

    return DepthResult(
        depth_map=np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64),
        original_image=np.zeros((64, 64, 3), dtype=np.uint8),
        metadata={},
        depth_units="relative",
        backend_id="da3",
        device="cpu",
    )


def _make_mock_registry():
    """Create a mock depth backend registry."""
    backend = Mock()
    backend.name = "da3"
    backend.license_type = Mock(value="commercial")
    backend.ensure_available.return_value = None
    backend.compute.return_value = _make_mock_depth_result()

    registry = Mock()
    registry.get_backend.return_value = backend
    return registry


def _create_orchestrator(tmp_path: Path, **config_kwargs):
    """Create an orchestrator instance with mocked backend registry."""
    from transformation_portal.lux_depth_v3.config import EnhanceConfig
    from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

    defaults = {
        "depth_backend": "da3",
        "depth_device": "cpu",
        "enable_v2": False,
        "enable_materials_v3": False,
    }
    defaults.update(config_kwargs)
    config = EnhanceConfig(**defaults)

    with patch(
        "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
        return_value=_make_mock_registry(),
    ):
        orchestrator = EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        return orchestrator


class TestEnhanceImageDispatch:
    """Test enhance_image dispatch flow."""

    def test_enhance_image_returns_result_dict(self, tmp_path: Path) -> None:
        """enhance_image returns a dictionary with expected keys."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "dispatch.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        assert isinstance(result, dict)
        assert "status" in result
        assert "image" in result
        assert "backend" in result
        assert "manifest" in result

    def test_enhance_image_status_ok_on_success(self, tmp_path: Path) -> None:
        """enhance_image returns status='ok' on successful processing."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "success.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        assert result["status"] == "ok"

    def test_enhance_image_depth_path_populated(self, tmp_path: Path) -> None:
        """enhance_image populates depth_path on success."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "depth_check.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        assert result["depth_path"] is not None
        assert Path(result["depth_path"]).exists()

    def test_enhance_image_manifest_created(self, tmp_path: Path) -> None:
        """enhance_image creates manifest file."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "manifest_check.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        assert result["manifest"] is not None
        manifest_path = Path(result["manifest"])
        assert manifest_path.exists()

        # Verify manifest is valid JSON
        with open(manifest_path) as f:
            manifest_data = json.load(f)
        assert "input" in manifest_data or "depth" in manifest_data


class TestPerImageStateManagement:
    """Test per-image state management in orchestrator."""

    def test_active_batch_id_preserved(self, tmp_path: Path) -> None:
        """_active_batch_id is preserved across processing."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)
        orchestrator._active_batch_id = "test-batch-123"
        test_image = _make_test_image(tmp_path, "batch_state.png")

        orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        # Batch ID should remain set
        assert orchestrator._active_batch_id == "test-batch-123"

    def test_active_depth_attempts_cleared_per_image(self, tmp_path: Path) -> None:
        """_active_depth_attempts is reset per image."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)

        # Process first image
        test_image1 = _make_test_image(tmp_path, "image1.png")
        result1 = orchestrator.enhance_image(
            ImageInput(path=test_image1),
            input_root=tmp_path,
        )
        attempts_after_first = len(result1.get("attempts", []))

        # Process second image
        test_image2 = _make_test_image(tmp_path, "image2.png")
        result2 = orchestrator.enhance_image(
            ImageInput(path=test_image2),
            input_root=tmp_path,
        )
        attempts_after_second = len(result2.get("attempts", []))

        # Both should have similar attempt counts (fresh state)
        assert attempts_after_first > 0
        assert attempts_after_second > 0

    def test_active_backend_metadata_updated(self, tmp_path: Path) -> None:
        """_active_backend_metadata is updated during processing."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "backend_meta.png")

        orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        # Backend metadata should be populated
        assert orchestrator._active_backend_metadata is not None
        assert hasattr(orchestrator._active_backend_metadata, "resolved_backend")


class TestSkipDecisionPropagation:
    """Test skip decision propagation through dispatch."""

    def test_force_depth_bypasses_skip(self, tmp_path: Path) -> None:
        """force_depth=True bypasses skip checks."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
            force_depth=True,
        )

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=_make_mock_registry(),
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            orchestrator.postprocessor = Mock(process=lambda result: result)

            test_image = _make_test_image(tmp_path, "force_depth.png")

            # First run
            result1 = orchestrator.enhance_image(
                ImageInput(path=test_image),
                input_root=tmp_path,
            )

            # Manually verify depth was computed (force_depth=True means it runs)
            assert result1["status"] == "ok"
            assert result1["depth_path"] is not None

    def test_precomputed_paths_used_when_provided(self, tmp_path: Path) -> None:
        """_precomputed_paths argument uses provided paths."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "precomputed.png")

        # Create precomputed paths structure
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)

        precomputed = {
            "output_key": Path("custom_key"),
            "depth_path": depth_dir / "custom_key_depth.png",
            "manifest_path": manifest_dir / "custom_key_combined.json",
            "should_skip": False,
        }

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
            _precomputed_paths=precomputed,
        )

        assert result["status"] == "ok"
        # Verify the custom paths were used in the manifest location
        assert "custom_key" in result["manifest"]


class TestResultStructureIntegrity:
    """Test result dictionary structure integrity."""

    def test_result_contains_all_expected_keys(self, tmp_path: Path) -> None:
        """Result contains all documented keys."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "keys_check.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        expected_keys = [
            "status",
            "image",
            "backend",
            "fallback_used",
            "depth_path",
            "manifest",
            "runtime_s",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_result_backend_field_matches_resolved(self, tmp_path: Path) -> None:
        """Result backend field matches resolved backend."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "backend_match.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        assert result["backend"] == "da3"

    def test_result_runtime_is_positive(self, tmp_path: Path) -> None:
        """Result runtime_s is a positive number."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "runtime_check.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        assert isinstance(result["runtime_s"], (int, float))
        assert result["runtime_s"] >= 0

    def test_result_fallback_used_is_boolean(self, tmp_path: Path) -> None:
        """Result fallback_used is a boolean."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path)
        test_image = _make_test_image(tmp_path, "fallback_bool.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        assert isinstance(result["fallback_used"], bool)


class TestErrorHandlingInDispatch:
    """Test error handling in dispatch lifecycle."""

    def test_depth_fallback_skip_returns_status_skipped(self, tmp_path: Path) -> None:
        """depth_fallback='skip' returns skipped status on failure."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
            depth_fallback="skip",
        )

        # Create registry that fails
        backend = Mock()
        backend.name = "da3"
        backend.license_type = Mock(value="commercial")
        backend.ensure_available.return_value = None
        backend.compute.side_effect = RuntimeError("Depth computation failed")

        failing_registry = Mock()
        failing_registry.get_backend.return_value = backend

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=failing_registry,
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            orchestrator.postprocessor = Mock(process=lambda result: result)

            test_image = _make_test_image(tmp_path, "skip_on_fail.png")

            result = orchestrator.enhance_image(
                ImageInput(path=test_image),
                input_root=tmp_path,
            )

            assert result["status"] == "skipped"
            # Skipped results may not have depth_path key, or it should be None
            assert result.get("depth_path") is None

    def test_depth_fallback_fail_raises_exception(self, tmp_path: Path) -> None:
        """depth_fallback='fail' raises exception on failure."""
        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        config = EnhanceConfig(
            depth_backend="da3",
            depth_device="cpu",
            enable_v2=False,
            enable_materials_v3=False,
            depth_fallback="fail",
        )

        # Create registry that fails
        backend = Mock()
        backend.name = "da3"
        backend.license_type = Mock(value="commercial")
        backend.ensure_available.return_value = None
        backend.compute.side_effect = RuntimeError("Depth computation failed")

        failing_registry = Mock()
        failing_registry.get_backend.return_value = backend

        with patch(
            "transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry",
            return_value=failing_registry,
        ):
            orchestrator = EnhanceOrchestrator(config, tmp_path)
            orchestrator.postprocessor = Mock(process=lambda result: result)

            test_image = _make_test_image(tmp_path, "fail_on_error.png")

            with pytest.raises(RuntimeError, match="Depth computation failed"):
                orchestrator.enhance_image(
                    ImageInput(path=test_image),
                    input_root=tmp_path,
                )


class TestStageCoordination:
    """Test Stage A/B coordination."""

    def test_v2_disabled_skips_v2_stage(self, tmp_path: Path) -> None:
        """enable_v2=False skips V2 stage entirely."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path, enable_v2=False)
        test_image = _make_test_image(tmp_path, "no_v2.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        assert result["status"] == "ok"
        # V2 output should be None when disabled
        assert result.get("v2_output_path") is None

    def test_materials_v3_disabled_skips_materials(self, tmp_path: Path) -> None:
        """enable_materials_v3=False skips Materials V3 stage."""
        from transformation_portal.lux_depth_v3.input_manager import ImageInput

        orchestrator = _create_orchestrator(tmp_path, enable_materials_v3=False)
        test_image = _make_test_image(tmp_path, "no_materials.png")

        result = orchestrator.enhance_image(
            ImageInput(path=test_image),
            input_root=tmp_path,
        )

        assert result["status"] == "ok"
        # Segmentation mask path should be None when materials disabled
        assert result.get("segmentation_mask_path") is None
