"""Unit tests for SAM2 backend (Phase 2.1)."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation.contracts import SegmentationResult
from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend


class TestSAM2BackendInitialization:
    """Test SAM2Backend initialization."""

    def test_default_initialization(self):
        """Test backend initialization with defaults."""
        backend = SAM2Backend()
        assert backend.model_size == "base"
        assert backend.device == "cuda"
        assert backend.revision.startswith("NEEDS_VERIFICATION")

    def test_custom_initialization(self):
        """Test backend initialization with custom parameters."""
        backend = SAM2Backend(model_size="large", device="cpu", revision="abc123")
        assert backend.model_size == "large"
        assert backend.device == "cpu"
        assert backend.revision == "abc123"

    def test_invalid_model_size(self):
        """Test validation of model size."""
        with pytest.raises(ValueError, match="Invalid model_size"):
            SAM2Backend(model_size="invalid")

    def test_supported_models(self):
        """Test that supported models are defined."""
        assert "base" in SAM2Backend.SUPPORTED_MODELS
        assert "large" in SAM2Backend.SUPPORTED_MODELS


class TestSAM2BackendSegmentation:
    """Test SAM2Backend segmentation methods."""

    @patch("transformation_portal.spatial_ai.segmentation.sam2_backend.SAM2Backend._load_model")
    @patch("transformation_portal.spatial_ai.segmentation.sam2_backend.SAM2Backend._segment_auto")
    def test_segment_auto_mode(self, mock_segment_auto, mock_load_model):
        """Test segmentation in auto mode."""
        # Setup mock return value
        mock_result = SegmentationResult(
            masks=np.zeros((3, 100, 100), dtype=bool),
            scores=np.array([0.9, 0.8, 0.7], dtype=np.float32),
            metadata=[Mock(area=100, bbox=(0, 0, 10, 10), stability_score=0.9) for _ in range(3)],
        )
        mock_segment_auto.return_value = mock_result

        # Run segmentation
        backend = SAM2Backend()
        image = np.random.rand(100, 100, 3).astype(np.float32)
        result = backend.segment(image=image, gamma=1.0, mode="auto")

        # Verify
        mock_load_model.assert_called_once()
        mock_segment_auto.assert_called_once()
        assert isinstance(result, SegmentationResult)
        assert len(result.masks) == 3

    @patch("transformation_portal.spatial_ai.segmentation.sam2_backend.SAM2Backend._load_model")
    def test_segment_prompted_mode_not_implemented(self, mock_load_model):
        """Test that prompted mode raises NotImplementedError."""
        backend = SAM2Backend()
        image = np.random.rand(100, 100, 3).astype(np.float32)

        with pytest.raises(NotImplementedError, match="Prompted segmentation"):
            backend.segment(
                image=image,
                gamma=1.0,
                mode="points",
                prompts=[{"type": "point", "coords": [50, 50], "label": 1}],
            )

    @patch("transformation_portal.spatial_ai.segmentation.sam2_backend.SAM2Backend._load_model")
    def test_segment_video_mode_not_implemented(self, mock_load_model):
        """Test that video mode raises NotImplementedError."""
        backend = SAM2Backend()
        image = np.random.rand(100, 100, 3).astype(np.float32)
        prev_masks = np.zeros((2, 100, 100), dtype=bool)

        with pytest.raises(NotImplementedError, match="Video tracking"):
            backend.segment(image=image, gamma=1.0, mode="video", prev_masks=prev_masks)

    def test_segment_contract_validation(self):
        """Test that segment validates input contract."""
        backend = SAM2Backend()
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # Wrong dtype

        with pytest.raises(ValueError, match="float32"):
            backend.segment(image=image, gamma=1.0, mode="auto")

    def test_segment_gamma_validation(self):
        """Test that segment enforces gamma=1.0."""
        backend = SAM2Backend()
        image = np.random.rand(100, 100, 3).astype(np.float32)

        with pytest.raises(ValueError, match="gamma=1.0"):
            backend.segment(image=image, gamma=2.2, mode="auto")


class TestSAM2BackendLinearToSRGB:
    """Test linear to sRGB conversion."""

    def test_linear_to_srgb_conversion(self):
        """Test conversion from linear RGB to sRGB uint8."""
        backend = SAM2Backend()

        # Create linear RGB with known values
        linear_rgb = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)

        # Convert
        srgb = backend._linear_to_srgb(linear_rgb)

        # Check dtype and range
        assert srgb.dtype == np.uint8
        assert srgb.min() >= 0
        assert srgb.max() <= 255

    def test_linear_to_srgb_clips_hdr(self):
        """Test that HDR values are clipped."""
        backend = SAM2Backend()

        # Create HDR image (values > 1.0)
        linear_rgb = np.array([[[2.0, 5.0, 10.0]]], dtype=np.float32)

        # Convert
        srgb = backend._linear_to_srgb(linear_rgb)

        # HDR values should be clipped to 255
        assert srgb[0, 0, 0] == 255
        assert srgb[0, 0, 1] == 255
        assert srgb[0, 0, 2] == 255

    def test_linear_to_srgb_preserves_shape(self):
        """Test that conversion preserves image shape."""
        backend = SAM2Backend()

        linear_rgb = np.random.rand(100, 200, 3).astype(np.float32)
        srgb = backend._linear_to_srgb(linear_rgb)

        assert srgb.shape == linear_rgb.shape


class TestSAM2BackendModelLoading:
    """Test model loading behavior."""

    @patch("transformers.AutoProcessor")
    @patch("transformers.AutoModel")
    def test_model_loads_successfully(self, mock_auto_model, mock_auto_processor):
        """Test successful model loading."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_auto_processor.from_pretrained.return_value = mock_processor
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Load model
        backend = SAM2Backend(model_size="base", device="cpu")
        backend._load_model()

        # Verify calls
        assert mock_auto_processor.from_pretrained.called
        assert mock_auto_model.from_pretrained.called
        assert backend._model is not None
        assert backend._processor is not None

    @patch("transformers.AutoProcessor")
    @patch("transformers.AutoModel")
    def test_model_loads_only_once(self, mock_auto_model, mock_auto_processor):
        """Test that model is only loaded once (lazy loading)."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_auto_processor.from_pretrained.return_value = mock_processor
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Load model twice
        backend = SAM2Backend()
        backend._load_model()
        backend._load_model()

        # Should only call from_pretrained once
        assert mock_auto_processor.from_pretrained.call_count == 1
        assert mock_auto_model.from_pretrained.call_count == 1

    def test_model_loading_missing_dependencies(self):
        """Test error handling when dependencies missing."""
        backend = SAM2Backend()

        # Patch the import inside _load_model
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="SAM2 requires transformers"):
                backend._load_model()


class TestSAM2BackendSegmentAuto:
    """Test automatic segmentation implementation."""

    def test_segment_auto_returns_result(self):
        """Test that _segment_auto returns valid SegmentationResult."""
        with patch("transformers.AutoModel"):
            with patch("transformers.AutoProcessor"):
                with patch("torch.no_grad"):
                    # Setup backend with mocked model
                    backend = SAM2Backend()
                    backend._model = MagicMock()
                    backend._processor = MagicMock()

                    # Mock processor return
                    backend._processor.return_value = {"pixel_values": MagicMock()}

                    # Mock model output
                    mock_output = MagicMock()
                    mock_output.pred_masks.cpu.return_value.numpy.return_value = np.random.rand(3, 100, 100)
                    mock_output.iou_scores.cpu.return_value.numpy.return_value = np.array([0.9, 0.8, 0.7])
                    backend._model.return_value = mock_output

                    # Create input
                    from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput

                    seg_input = SegmentationInput(
                        image=np.random.rand(100, 100, 3).astype(np.float32),
                        gamma=1.0,
                        mode="auto",
                    )

                    # Run segmentation
                    result = backend._segment_auto(seg_input)

                    # Verify result type and structure
                    assert isinstance(result, SegmentationResult)
                    assert result.masks.dtype == bool
                    assert len(result.scores) == len(result.masks)
                    assert len(result.metadata) == len(result.masks)
