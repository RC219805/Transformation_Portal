"""Tests for DA3InferenceEngine PIL.Image input support.

This test suite validates that DA3InferenceEngine.predict() and infer()
correctly accept and convert PIL.Image inputs alongside numpy arrays.

Coverage target: Fix DA3 inference to accept PIL and convert properly.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.inference import DA3InferenceEngine


def _create_test_pil_image(h: int = 64, w: int = 64, mode: str = "RGB") -> Image.Image:
    """Create a test PIL Image with random content."""
    rng = np.random.default_rng(42)
    if mode == "RGB":
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGB")
    elif mode == "RGBA":
        arr = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
        return Image.fromarray(arr, mode="RGBA")
    elif mode == "L":
        arr = rng.integers(0, 256, (h, w), dtype=np.uint8)
        return Image.fromarray(arr, mode="L")
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def _create_test_numpy_image(h: int = 64, w: int = 64, dtype=np.uint8) -> np.ndarray:
    """Create a test numpy array image."""
    rng = np.random.default_rng(42)
    if dtype == np.uint8:
        return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    elif dtype == np.float32:
        return rng.random((h, w, 3)).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _create_mock_depth(h: int = 64, w: int = 64) -> np.ndarray:
    """Create a mock depth map with deterministic values."""
    rng = np.random.default_rng(42)
    return rng.random((h, w)).astype(np.float32)


class TestPILImageSupport:
    """Test PIL.Image input support for DA3InferenceEngine."""

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_predict_accepts_pil_image(self, mock_torch):
        """Test that predict() accepts PIL.Image and normalizes it correctly."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = DA3InferenceEngine(config="cpu")

        # Create a mock depth result
        mock_depth = _create_mock_depth(64, 64)
        mock_result = {
            "depth": mock_depth,
            "depth_raw": mock_depth,
            "metadata": {"shape": (64, 64)},
        }

        # Mock the model pipeline
        engine._model_loaded = True
        engine.model = MagicMock()
        engine.model.__call__ = MagicMock(return_value={"depth": mock_depth})

        # Mock _estimate_depth_pytorch
        with patch.object(engine, "_estimate_depth_pytorch", return_value=mock_result) as mock_estimate:
            pil_image = _create_test_pil_image(64, 64)
            _ = engine.predict(pil_image)

            # Verify _estimate_depth_pytorch was called with a PIL Image
            assert mock_estimate.called
            call_args = mock_estimate.call_args[0]
            assert isinstance(call_args[0], Image.Image)

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_infer_accepts_pil_image(self, mock_torch):
        """Test that infer() accepts PIL.Image and normalizes it correctly."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = DA3InferenceEngine(config="cpu")

        # Create a mock depth result
        mock_depth = _create_mock_depth(64, 64)
        mock_result = {
            "depth": mock_depth,
            "depth_raw": mock_depth,
            "metadata": {"shape": (64, 64)},
        }

        # Mock the model and inference
        engine._model_loaded = True
        engine.model = MagicMock()

        with patch.object(engine, "_estimate_depth_pytorch", return_value=mock_result) as mock_estimate:
            pil_image = _create_test_pil_image(64, 64)
            result = engine.infer(pil_image)

            # Verify _estimate_depth_pytorch was called
            assert mock_estimate.called
            # Verify result has expected attributes
            assert hasattr(result, "depth_map")
            assert hasattr(result, "original_image")
            assert hasattr(result, "metadata")

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_pil_rgba_converted_to_rgb(self, mock_torch):
        """Test that RGBA PIL images are converted to RGB."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = DA3InferenceEngine(config="cpu")

        mock_depth = _create_mock_depth(64, 64)
        mock_result = {
            "depth": mock_depth,
            "depth_raw": mock_depth,
            "metadata": {"shape": (64, 64)},
        }

        engine._model_loaded = True
        engine.model = MagicMock()

        with patch.object(engine, "_estimate_depth_pytorch", return_value=mock_result) as mock_estimate:
            # Create RGBA image
            rgba_image = _create_test_pil_image(64, 64, mode="RGBA")
            assert rgba_image.mode == "RGBA"

            _ = engine.infer(rgba_image)

            # Verify the PIL image passed to estimate was RGB
            call_args = mock_estimate.call_args[0]
            assert isinstance(call_args[0], Image.Image)
            assert call_args[0].mode == "RGB"

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_pil_grayscale_converted_to_rgb(self, mock_torch):
        """Test that grayscale PIL images are converted to RGB."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = DA3InferenceEngine(config="cpu")

        mock_depth = _create_mock_depth(64, 64)
        mock_result = {
            "depth": mock_depth,
            "depth_raw": mock_depth,
            "metadata": {"shape": (64, 64)},
        }

        engine._model_loaded = True
        engine.model = MagicMock()

        with patch.object(engine, "_estimate_depth_pytorch", return_value=mock_result) as mock_estimate:
            # Create grayscale image
            gray_image = _create_test_pil_image(64, 64, mode="L")
            assert gray_image.mode == "L"

            _ = engine.infer(gray_image)

            # Verify the PIL image passed to estimate was RGB
            call_args = mock_estimate.call_args[0]
            assert isinstance(call_args[0], Image.Image)
            assert call_args[0].mode == "RGB"

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_original_image_stored_as_numpy(self, mock_torch):
        """Test that original_image in result is stored as numpy array."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = DA3InferenceEngine(config="cpu")

        mock_depth = _create_mock_depth(64, 64)
        mock_result = {
            "depth": mock_depth,
            "depth_raw": mock_depth,
            "metadata": {"shape": (64, 64)},
        }

        engine._model_loaded = True
        engine.model = MagicMock()

        with patch.object(engine, "_estimate_depth_pytorch", return_value=mock_result):
            pil_image = _create_test_pil_image(64, 64)
            result = engine.infer(pil_image)

            # Verify original_image is numpy array
            assert isinstance(result.original_image, np.ndarray)
            assert result.original_image.shape == (64, 64, 3)
            assert result.original_image.dtype == np.uint8


class TestNumpyArrayBackwardCompatibility:
    """Test that numpy array inputs still work (backward compatibility)."""

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_predict_accepts_numpy_uint8(self, mock_torch):
        """Test that predict() still accepts uint8 numpy arrays."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = DA3InferenceEngine(config="cpu")

        mock_depth = _create_mock_depth(64, 64)
        mock_result = {
            "depth": mock_depth,
            "depth_raw": mock_depth,
            "metadata": {"shape": (64, 64)},
        }

        engine._model_loaded = True
        engine.model = MagicMock()

        with patch.object(engine, "_estimate_depth_pytorch", return_value=mock_result):
            np_image = _create_test_numpy_image(64, 64, dtype=np.uint8)
            result = engine.predict(np_image)

            assert hasattr(result, "depth_map")
            assert result.depth_map.shape == (64, 64)

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_predict_accepts_numpy_float32(self, mock_torch):
        """Test that predict() still accepts float32 numpy arrays."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = DA3InferenceEngine(config="cpu")

        mock_depth = _create_mock_depth(64, 64)
        mock_result = {
            "depth": mock_depth,
            "depth_raw": mock_depth,
            "metadata": {"shape": (64, 64)},
        }

        engine._model_loaded = True
        engine.model = MagicMock()

        with patch.object(engine, "_estimate_depth_pytorch", return_value=mock_result):
            np_image = _create_test_numpy_image(64, 64, dtype=np.float32)
            result = engine.predict(np_image)

            assert hasattr(result, "depth_map")
            assert result.depth_map.shape == (64, 64)


class TestTypeErrorHandling:
    """Test error handling for unsupported input types."""

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_predict_rejects_unsupported_type(self, mock_torch):
        """Test that predict() raises TypeError for unsupported types."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = DA3InferenceEngine(config="cpu")
        engine._model_loaded = True

        with pytest.raises(TypeError) as excinfo:
            engine.predict({"unsupported": "dict"})

        assert "Expected np.ndarray, PIL.Image" in str(excinfo.value)

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_infer_rejects_unsupported_type(self, mock_torch):
        """Test that infer() raises TypeError for unsupported types."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = DA3InferenceEngine(config="cpu")
        engine._model_loaded = True

        with pytest.raises(TypeError) as excinfo:
            engine.infer([1, 2, 3])  # list is not supported

        assert "Expected numpy array or PIL.Image" in str(excinfo.value)


class TestCoreMLPathNormalization:
    """Test that CoreML backend receives numpy arrays properly."""

    @patch("transformation_portal.lux_depth_v3.inference.TORCH_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.COREML_AVAILABLE", True)
    @patch("transformation_portal.lux_depth_v3.inference.torch")
    def test_coreml_receives_numpy_from_pil_input(self, mock_torch):
        """Test that CoreML backend receives numpy array even when given PIL input."""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False

        engine = DA3InferenceEngine(config="cpu")

        # Force CoreML backend
        from transformation_portal.lux_depth_v3.inference import ModelBackend

        engine.backend = ModelBackend.COREML

        mock_depth = _create_mock_depth(64, 64)
        mock_result = {
            "depth": mock_depth,
            "depth_raw": mock_depth,
            "metadata": {"shape": (64, 64)},
        }

        engine._model_loaded = True
        engine.model = MagicMock()

        with patch.object(engine, "_estimate_depth_coreml", return_value=mock_result) as mock_estimate:
            pil_image = _create_test_pil_image(64, 64)
            _ = engine.infer(pil_image)

            # Verify _estimate_depth_coreml was called with numpy array
            assert mock_estimate.called
            call_args = mock_estimate.call_args[0]
            assert isinstance(call_args[0], np.ndarray)
            assert call_args[0].shape == (64, 64, 3)


class TestDocstringUpdates:
    """Test that docstrings document PIL.Image support."""

    def test_predict_docstring_mentions_pil(self):
        """Test that predict() docstring mentions PIL.Image."""
        docstring = DA3InferenceEngine.predict.__doc__
        assert docstring is not None
        assert "PIL" in docstring

    def test_infer_docstring_mentions_pil(self):
        """Test that infer() docstring mentions PIL.Image."""
        docstring = DA3InferenceEngine.infer.__doc__
        assert docstring is not None
        assert "PIL" in docstring


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
