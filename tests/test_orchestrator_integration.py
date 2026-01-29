"""Tests for EnhanceOrchestrator integration.

Tests orchestrator wiring of preprocessing, inference, and depth writer
components without requiring heavy ML model loading.
"""
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.config import EnhanceConfig, ModelVariant
from transformation_portal.lux_depth_v3.orchestrator import (
    EnhanceOrchestrator,
    make_output_key,
)
from transformation_portal.lux_depth_v3.inference import DepthResult
from transformation_portal.lux_depth_v3.depth_writer import HAS_CV2

# Skip tests if opencv not available
pytestmark = pytest.mark.skipif(not HAS_CV2, reason="opencv-python not installed")


class TestEnhanceOrchestratorInit:
    """Test EnhanceOrchestrator initialization."""

    @patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine')
    @patch('transformation_portal.lux_depth_v3.orchestrator.V2Runner')
    def test_init_creates_output_directories(self, mock_v2_runner, mock_engine, tmp_path):
        """Test that init creates all required output directories."""
        config = EnhanceConfig(model_variant=ModelVariant.METRIC_SMALL)
        orchestrator = EnhanceOrchestrator(config=config, output_root=tmp_path)

        assert (tmp_path / "depth").exists()
        assert (tmp_path / "v2").exists()
        assert (tmp_path / "manifests").exists()
        assert (tmp_path / "logs").exists()
        assert (tmp_path / "zones").exists()

    @patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine')
    @patch('transformation_portal.lux_depth_v3.orchestrator.V2Runner')
    def test_init_initializes_inference_engine(self, mock_v2_runner, mock_engine, tmp_path):
        """Test that inference engine is initialized with correct config."""
        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL,
            depth_device="cpu",
            non_commercial_ok=True,
        )
        orchestrator = EnhanceOrchestrator(config=config, output_root=tmp_path)

        # Verify engine was called with DA3Config
        mock_engine.assert_called_once()
        call_kwargs = mock_engine.call_args[1]
        assert call_kwargs["commercial_use"] is False  # non_commercial_ok=True -> commercial_use=False
        assert call_kwargs["validate_license_strict"] is True


class TestMakeOutputKey:
    """Test output key generation."""

    def test_relative_path_preserved(self, tmp_path):
        """Test that relative paths are preserved in output key."""
        input_path = tmp_path / "subdir" / "image.jpg"
        input_root = tmp_path

        key = make_output_key(input_path, input_root)

        assert key == Path("subdir") / "image"

    def test_flat_naming_for_non_relative(self, tmp_path):
        """Test flat naming when path not relative to root."""
        input_path = Path("/other/path/image.jpg")
        input_root = tmp_path

        key = make_output_key(input_path, input_root)

        # Should fall back to filename
        assert key.name == "image.jpg" or "image" in str(key)


class TestEnhanceOrchestratorPipeline:
    """Test EnhanceOrchestrator.enhance_image pipeline."""

    @patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine')
    @patch('transformation_portal.lux_depth_v3.orchestrator.V2Runner')
    @patch('transformation_portal.lux_depth_v3.orchestrator.Postprocessor')
    def test_enhance_image_validates_input(
        self, mock_postprocessor, mock_v2_runner, mock_engine, tmp_path
    ):
        """Test that enhance_image validates image format."""
        # Create valid input image
        input_image = tmp_path / "input.png"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(input_image)

        # Create mock depth result
        mock_depth = np.random.rand(98, 98).astype(np.float32)
        mock_result = DepthResult(
            depth_map=mock_depth,
            original_image=np.random.rand(100, 100, 3).astype(np.float32),
            metadata={}
        )
        mock_engine.return_value.predict.return_value = mock_result
        mock_postprocessor.return_value.process.return_value = mock_result

        # V2Runner should fail (script doesn't exist)
        mock_v2_runner.return_value.run.side_effect = FileNotFoundError("Script not found")

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL,
            depth_fallback="fail",
        )
        orchestrator = EnhanceOrchestrator(config=config, output_root=tmp_path)

        # Import here to use in input_manager
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        image_input = ImageInput(path=input_image)

        # V2 will fail but should still produce depth output
        result = orchestrator.enhance_image(image_input)

        # Should have called preprocessing (validate + preprocess)
        # Inference engine should have been called with numpy array
        mock_engine.return_value.predict.assert_called_once()

        assert result["status"] == "ok"
        assert "depth_path" in result

    @patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine')
    @patch('transformation_portal.lux_depth_v3.orchestrator.V2Runner')
    def test_enhance_image_fails_on_invalid_input(
        self, mock_v2_runner, mock_engine, tmp_path
    ):
        """Test that enhance_image raises on invalid input."""
        config = EnhanceConfig(model_variant=ModelVariant.METRIC_SMALL)
        orchestrator = EnhanceOrchestrator(config=config, output_root=tmp_path)

        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        image_input = ImageInput(path=tmp_path / "nonexistent.jpg")

        with pytest.raises(RuntimeError, match="validation failed"):
            orchestrator.enhance_image(image_input)

    @patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine')
    @patch('transformation_portal.lux_depth_v3.orchestrator.V2Runner')
    @patch('transformation_portal.lux_depth_v3.orchestrator.Postprocessor')
    def test_depth_fallback_skip(
        self, mock_postprocessor, mock_v2_runner, mock_engine, tmp_path
    ):
        """Test depth_fallback='skip' returns skipped status on failure."""
        # Create valid input image
        input_image = tmp_path / "input.png"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(input_image)

        # Make inference fail
        mock_engine.return_value.predict.side_effect = RuntimeError("Model failed")

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL,
            depth_fallback="skip",
        )
        orchestrator = EnhanceOrchestrator(config=config, output_root=tmp_path)

        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        image_input = ImageInput(path=input_image)

        result = orchestrator.enhance_image(image_input)

        assert result["status"] == "skipped"
        assert "reason" in result

    @patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine')
    @patch('transformation_portal.lux_depth_v3.orchestrator.V2Runner')
    @patch('transformation_portal.lux_depth_v3.orchestrator.Postprocessor')
    def test_depth_fallback_fail_raises(
        self, mock_postprocessor, mock_v2_runner, mock_engine, tmp_path
    ):
        """Test depth_fallback='fail' raises on inference failure."""
        # Create valid input image
        input_image = tmp_path / "input.png"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(input_image)

        # Make inference fail
        mock_engine.return_value.predict.side_effect = RuntimeError("Model failed")

        config = EnhanceConfig(
            model_variant=ModelVariant.METRIC_SMALL,
            depth_fallback="fail",
        )
        orchestrator = EnhanceOrchestrator(config=config, output_root=tmp_path)

        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        image_input = ImageInput(path=input_image)

        with pytest.raises(RuntimeError, match="V3 depth inference failed"):
            orchestrator.enhance_image(image_input)


class TestRunPipeline:
    """Test EnhanceOrchestrator.run_pipeline method."""

    @patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine')
    @patch('transformation_portal.lux_depth_v3.orchestrator.V2Runner')
    def test_run_pipeline_raises_on_missing_input(
        self, mock_v2_runner, mock_engine, tmp_path
    ):
        """Test run_pipeline raises FileNotFoundError for missing input."""
        config = EnhanceConfig(model_variant=ModelVariant.METRIC_SMALL)
        orchestrator = EnhanceOrchestrator(config=config, output_root=tmp_path)

        with pytest.raises(FileNotFoundError, match="not found"):
            orchestrator.run_pipeline(tmp_path / "nonexistent.jpg")

    @patch('transformation_portal.lux_depth_v3.orchestrator.DA3InferenceEngine')
    @patch('transformation_portal.lux_depth_v3.orchestrator.V2Runner')
    @patch('transformation_portal.lux_depth_v3.orchestrator.Postprocessor')
    def test_run_pipeline_delegates_to_enhance_image(
        self, mock_postprocessor, mock_v2_runner, mock_engine, tmp_path
    ):
        """Test run_pipeline correctly delegates to enhance_image."""
        # Create valid input image
        input_image = tmp_path / "input.png"
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(input_image)

        # Create mock depth result
        mock_depth = np.random.rand(98, 98).astype(np.float32)
        mock_result = DepthResult(
            depth_map=mock_depth,
            original_image=np.random.rand(100, 100, 3).astype(np.float32),
            metadata={}
        )
        mock_engine.return_value.predict.return_value = mock_result
        mock_postprocessor.return_value.process.return_value = mock_result
        mock_v2_runner.return_value.run.side_effect = FileNotFoundError("No V2 script")

        config = EnhanceConfig(model_variant=ModelVariant.METRIC_SMALL)
        orchestrator = EnhanceOrchestrator(config=config, output_root=tmp_path)

        result = orchestrator.run_pipeline(input_image)

        assert result["status"] == "ok"
        assert str(input_image) in result["image"]


class TestEnhanceConfigFields:
    """Test that EnhanceConfig has all required fields."""

    def test_config_has_depth_fallback(self):
        """Test EnhanceConfig has depth_fallback field."""
        config = EnhanceConfig()
        assert hasattr(config, 'depth_fallback')
        assert config.depth_fallback == "fail"  # default

    def test_config_has_verify_depth_writes(self):
        """Test EnhanceConfig has verify_depth_writes field."""
        config = EnhanceConfig()
        assert hasattr(config, 'verify_depth_writes')
        assert config.verify_depth_writes is False  # default

    def test_config_has_force_v2(self):
        """Test EnhanceConfig has force_v2 field."""
        config = EnhanceConfig()
        assert hasattr(config, 'force_v2')
        assert config.force_v2 is False  # default

    def test_config_has_v2_timeout(self):
        """Test EnhanceConfig has v2_timeout field."""
        config = EnhanceConfig()
        assert hasattr(config, 'v2_timeout')
        assert config.v2_timeout is None  # default

    def test_config_customization(self):
        """Test EnhanceConfig can be customized."""
        config = EnhanceConfig(
            depth_fallback="v2-auto",
            verify_depth_writes=True,
            force_v2=True,
            v2_timeout=300.0,
        )
        assert config.depth_fallback == "v2-auto"
        assert config.verify_depth_writes is True
        assert config.force_v2 is True
        assert config.v2_timeout == 300.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
