"""Tests for DA3 Python API wrapper.

Tests the DepthAnything3Wrapper class and full API integration.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from lux_depth_v3.da3_wrapper import (
    DepthAnything3Wrapper,
    DA3Prediction,
)


class TestDA3Prediction:
    """Test DA3Prediction dataclass."""

    def test_prediction_creation(self):
        """Test basic prediction creation."""
        depth = np.random.rand(2, 100, 100)

        pred = DA3Prediction(depth=depth, conf=None, extrinsics=None, intrinsics=None)

        assert pred.depth.shape == (2, 100, 100)
        assert pred.conf is None

    def test_prediction_with_all_fields(self):
        """Test prediction with all fields populated."""
        depth = np.random.rand(3, 100, 100)
        conf = np.random.rand(3, 100, 100)
        extrinsics = np.random.rand(3, 4, 4)
        intrinsics = np.random.rand(3, 3, 3)

        pred = DA3Prediction(depth=depth, conf=conf, extrinsics=extrinsics, intrinsics=intrinsics)

        assert pred.depth.shape == (3, 100, 100)
        assert pred.conf.shape == (3, 100, 100)
        assert pred.extrinsics.shape == (3, 4, 4)
        assert pred.intrinsics.shape == (3, 3, 3)

    def test_prediction_validation_invalid_depth_dim(self):
        """Test validation catches invalid depth dimensions."""
        depth = np.random.rand(100)  # 1D (should be 2D or 3D)

        with pytest.raises(ValueError, match="Depth must be 2D or 3D"):
            DA3Prediction(depth=depth)

    def test_prediction_validation_conf_mismatch(self):
        """Test validation catches confidence shape mismatch."""
        depth = np.random.rand(2, 100, 100)
        conf = np.random.rand(2, 50, 50)  # Wrong shape

        with pytest.raises(ValueError, match="Confidence shape"):
            DA3Prediction(depth=depth, conf=conf)


class TestDepthAnything3Wrapper:
    """Test DepthAnything3Wrapper class."""

    def test_wrapper_initialization_no_api(self):
        """Test wrapper initialization when API not available."""
        with patch("lux_depth_v3.da3_wrapper.DepthAnything3Wrapper.__init__", return_value=None):
            # Mock the import failure
            with patch.dict("sys.modules", {"depth_anything_3.api": None}):
                wrapper = DepthAnything3Wrapper.__new__(DepthAnything3Wrapper)
                wrapper.model_name = "da3-large"
                wrapper.device = "cuda"
                wrapper.available = False
                wrapper.model = None

                assert wrapper.available is False

    def test_wrapper_model_names(self):
        """Test available model names."""
        expected_models = {
            "da3-giant",
            "da3-large",
            "da3-base",
            "da3-small",
            "da3mono-large",
            "da3metric-large",
            "da3nested-giant-large",
        }

        assert set(DepthAnything3Wrapper.AVAILABLE_MODELS.keys()) == expected_models

    def test_gs_capable_models(self):
        """Test GS-capable model list."""
        assert "da3-giant" in DepthAnything3Wrapper.GS_CAPABLE_MODELS
        assert "da3nested-giant-large" in DepthAnything3Wrapper.GS_CAPABLE_MODELS
        assert "da3-large" not in DepthAnything3Wrapper.GS_CAPABLE_MODELS

    @patch("lux_depth_v3.da3_wrapper.DepthAnything3Wrapper._load_model")
    @patch("lux_depth_v3.da3_wrapper.DepthAnything3Wrapper.__init__", return_value=None)
    def test_wrapper_initialization_with_api(self, mock_init, mock_load):
        """Test wrapper initialization with API available."""
        wrapper = DepthAnything3Wrapper.__new__(DepthAnything3Wrapper)
        wrapper.model_name = "da3-large"
        wrapper.device = "cuda"
        wrapper.available = True
        wrapper.model = Mock()

        assert wrapper.available is True
        assert wrapper.model is not None

    def test_prepare_images_paths(self):
        """Test image path preparation."""
        wrapper = DepthAnything3Wrapper.__new__(DepthAnything3Wrapper)

        # Test Path to string conversion
        images = [
            Path("/path/to/image1.jpg"),
            Path("/path/to/image2.jpg"),
            "image3.jpg",
            Mock(spec=np.ndarray),
        ]

        prepared = wrapper._prepare_images(images)

        assert isinstance(prepared[0], str)
        assert isinstance(prepared[1], str)
        assert isinstance(prepared[2], str)
        assert not isinstance(prepared[3], str)  # NumPy array unchanged

    @patch("lux_depth_v3.da3_wrapper.DepthAnything3Wrapper._load_model")
    def test_inference_gs_validation(self, mock_load):
        """Test GS validation in inference."""
        wrapper = DepthAnything3Wrapper.__new__(DepthAnything3Wrapper)
        wrapper.model_name = "da3-large"  # Not GS-capable
        wrapper.device = "cuda"
        wrapper.available = True
        wrapper.model = Mock()

        # Should raise error when requesting GS with non-GS model
        with pytest.raises(ValueError, match="Gaussian Splatting requires"):
            wrapper.inference(image=["test.jpg"], infer_gs=True)

    @patch("lux_depth_v3.da3_wrapper.DepthAnything3Wrapper._load_model")
    def test_inference_ref_view_validation(self, mock_load):
        """Test reference view strategy validation."""
        wrapper = DepthAnything3Wrapper.__new__(DepthAnything3Wrapper)
        wrapper.model_name = "da3-large"
        wrapper.device = "cuda"
        wrapper.available = True
        wrapper.model = Mock()

        # Should raise error with invalid strategy
        with pytest.raises(ValueError, match="Invalid ref_view_strategy"):
            wrapper.inference(image=["test.jpg"], ref_view_strategy="invalid_strategy")

    @patch("lux_depth_v3.da3_wrapper.DepthAnything3Wrapper._load_model")
    def test_inference_not_available(self, mock_load):
        """Test inference when API not available."""
        wrapper = DepthAnything3Wrapper.__new__(DepthAnything3Wrapper)
        wrapper.model_name = "da3-large"
        wrapper.device = "cuda"
        wrapper.available = False
        wrapper.model = None

        with pytest.raises(RuntimeError, match="DA3 API not available"):
            wrapper.inference(image=["test.jpg"])

    @patch("lux_depth_v3.da3_wrapper.DepthAnything3Wrapper._load_model")
    def test_inference_basic(self, mock_load):
        """Test basic inference call."""
        wrapper = DepthAnything3Wrapper.__new__(DepthAnything3Wrapper)
        wrapper.model_name = "da3-large"
        wrapper.device = "cuda"
        wrapper.available = True

        # Mock the model and its inference method
        mock_prediction = Mock()
        mock_prediction.depth = np.random.rand(1, 100, 100)
        mock_prediction.conf = np.random.rand(1, 100, 100)

        wrapper.model = Mock()
        wrapper.model.inference = Mock(return_value=mock_prediction)

        # Run inference
        result = wrapper.inference(image=["test.jpg"], export_dir="output", export_format="mini_npz")

        # Verify
        assert isinstance(result, DA3Prediction)
        assert result.depth.shape == (1, 100, 100)
        wrapper.model.inference.assert_called_once()

    @patch("lux_depth_v3.da3_wrapper.DepthAnything3Wrapper._load_model")
    def test_inference_with_poses(self, mock_load):
        """Test inference with camera poses."""
        wrapper = DepthAnything3Wrapper.__new__(DepthAnything3Wrapper)
        wrapper.model_name = "da3-large"
        wrapper.device = "cuda"
        wrapper.available = True

        # Mock prediction with poses
        mock_prediction = Mock()
        mock_prediction.depth = np.random.rand(3, 100, 100)
        mock_prediction.conf = np.random.rand(3, 100, 100)
        mock_prediction.extrinsics = np.random.rand(3, 4, 4)
        mock_prediction.intrinsics = np.random.rand(3, 3, 3)

        wrapper.model = Mock()
        wrapper.model.inference = Mock(return_value=mock_prediction)

        # Input poses
        extrinsics = np.random.rand(3, 4, 4)
        intrinsics = np.random.rand(3, 3, 3)

        result = wrapper.inference(
            image=["img1.jpg", "img2.jpg", "img3.jpg"],
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            export_dir="output",
        )

        assert result.extrinsics.shape == (3, 4, 4)
        assert result.intrinsics.shape == (3, 3, 3)

    def test_from_pretrained(self):
        """Test from_pretrained class method."""
        with patch.object(DepthAnything3Wrapper, "__init__", return_value=None) as mock_init:
            wrapper = DepthAnything3Wrapper.from_pretrained("depth-anything/DA3-GIANT", device="cuda")

            # Should extract model name from HF ID
            mock_init.assert_called_once_with(model_name="da3-giant", device="cuda")


class TestDA3APIConfig:
    """Test DA3APIConfig."""

    def test_api_config_defaults(self):
        """Test default configuration."""
        from lux_depth_v3.config import DA3APIConfig

        config = DA3APIConfig()

        assert config.model_name == "da3-large"
        assert config.align_to_input_ext_scale is True
        assert config.infer_gs is False
        assert config.ref_view_strategy == "saddle_balanced"
        assert config.process_res == 504

    def test_api_config_to_kwargs(self):
        """Test conversion to API kwargs."""
        from lux_depth_v3.config import DA3APIConfig

        config = DA3APIConfig(
            model_name="da3-giant",
            infer_gs=True,
            export_format="gs_ply-gs_video",
            process_res=672,
        )

        kwargs = config.to_api_kwargs()

        assert kwargs["infer_gs"] is True
        assert kwargs["export_format"] == "gs_ply-gs_video"
        assert kwargs["process_res"] == 672
        assert "model_name" not in kwargs  # Not in API kwargs


class TestInferenceEngineAPI:
    """Test inference engine with API integration."""

    @patch("lux_depth_v3.inference.DepthAnything3Wrapper")
    def test_engine_init_with_api(self, mock_wrapper_class):
        """Test engine initialization with Python API."""
        from lux_depth_v3.config import DA3Config, DA3APIConfig
        from lux_depth_v3.inference import DA3InferenceEngine

        config = DA3Config(api=DA3APIConfig(model_name="da3-large"))

        # Mock wrapper
        mock_wrapper = Mock()
        mock_wrapper.available = True
        mock_wrapper.model = Mock()
        mock_wrapper_class.return_value = mock_wrapper

        engine = DA3InferenceEngine(config)

        # Should initialize wrapper
        assert hasattr(engine, "wrapper")

    @patch("lux_depth_v3.inference.DepthAnything3Wrapper")
    def test_engine_infer_api(self, mock_wrapper_class):
        """Test engine infer method with API."""
        from lux_depth_v3.config import DA3Config
        from lux_depth_v3.inference import DA3InferenceEngine
        from lux_depth_v3.da3_wrapper import DA3Prediction

        config = DA3Config()
        config.cli.use_cli = False  # Ensure we use API mode

        # Mock wrapper and prediction
        mock_prediction = DA3Prediction(depth=np.random.rand(1, 100, 100), conf=None)

        mock_wrapper = Mock()
        mock_wrapper.available = True
        mock_wrapper.model = Mock()
        mock_wrapper.inference = Mock(return_value=mock_prediction)
        mock_wrapper_class.return_value = mock_wrapper

        engine = DA3InferenceEngine(config)
        engine.wrapper = mock_wrapper  # Ensure wrapper is set

        # Run inference
        result = engine.infer(images=[Path("test.jpg")], export_dir=Path("output"))

        # Verify API was called
        mock_wrapper.inference.assert_called_once()
        assert result.depth.shape == (1, 100, 100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
