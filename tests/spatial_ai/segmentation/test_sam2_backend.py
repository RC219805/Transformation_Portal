"""Unit tests for SAM2 backend (Phase 2.1)."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation.contracts import SegmentationInput, SegmentationResult
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
        """Test segmentation in auto mode raises NotImplementedError."""
        backend = SAM2Backend()
        image = np.random.rand(100, 100, 3).astype(np.float32)
        seg_input = SegmentationInput(image=image, gamma=1.0, mode="auto")

        # Configure mock to raise NotImplementedError (reflecting the actual implementation)
        mock_segment_auto.side_effect = NotImplementedError("SAM2 automatic mask generation not yet integrated")

        # Run segmentation should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="automatic mask generation"):
            result = backend.segment(seg_input)

        # Verify _load_model and _segment_auto were called
        mock_load_model.assert_called_once()
        mock_segment_auto.assert_called_once()

    @patch("transformation_portal.spatial_ai.segmentation.sam2_backend.SAM2Backend._load_model")
    def test_segment_prompted_mode_not_implemented(self, mock_load_model):
        """Test that prompted mode raises NotImplementedError."""
        backend = SAM2Backend()
        image = np.random.rand(100, 100, 3).astype(np.float32)
        seg_input = SegmentationInput(
            image=image,
            gamma=1.0,
            mode="points",
            prompts=[{"type": "point", "coords": [50, 50], "label": 1}],
        )

        with pytest.raises(NotImplementedError, match="Prompted segmentation"):
            backend.segment(seg_input)

    @patch("transformation_portal.spatial_ai.segmentation.sam2_backend.SAM2Backend._load_model")
    def test_segment_video_mode_not_implemented(self, mock_load_model):
        """Test that video mode raises NotImplementedError."""
        backend = SAM2Backend()
        image = np.random.rand(100, 100, 3).astype(np.float32)
        prev_masks = np.zeros((2, 100, 100), dtype=bool)
        seg_input = SegmentationInput(image=image, gamma=1.0, mode="video", prev_masks=prev_masks)

        with pytest.raises(NotImplementedError, match="Video tracking"):
            backend.segment(seg_input)

    def test_segment_contract_validation(self):
        """Test that segment validates input contract."""
        backend = SAM2Backend()
        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # Wrong dtype

        with pytest.raises(ValueError, match="float32"):
            seg_input = SegmentationInput(image=image, gamma=1.0, mode="auto")
            backend.segment(seg_input)

    def test_segment_gamma_validation(self):
        """Test that segment enforces gamma=1.0."""
        backend = SAM2Backend()
        image = np.random.rand(100, 100, 3).astype(np.float32)

        with pytest.raises(ValueError, match="gamma=1.0"):
            seg_input = SegmentationInput(image=image, gamma=2.2, mode="auto")
            backend.segment(seg_input)


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

    @pytest.mark.ml
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

    @pytest.mark.ml
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

        # Load model twice (use verified revision to pass validation)
        backend = SAM2Backend(revision="abc123def456abc123def456abc123def456abc1")
        backend._load_model()
        backend._load_model()

        # Should only call from_pretrained once
        assert mock_auto_processor.from_pretrained.call_count == 1
        assert mock_auto_model.from_pretrained.call_count == 1

    def test_model_loading_missing_dependencies(self):
        """Test error handling when dependencies missing."""
        backend = SAM2Backend(revision="abc123def456abc123def456abc123def456abc1")

        # Patch the import inside _load_model
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ImportError, match="SAM2 requires transformers"):
                backend._load_model()

    def test_model_loading_rejects_unverified_revision(self):
        """Test that _load_model raises ValueError for NEEDS_VERIFICATION revisions (ADR-027)."""
        backend = SAM2Backend()  # default revision is NEEDS_VERIFICATION_*
        assert backend.revision.startswith("NEEDS_VERIFICATION")

        with pytest.raises(ValueError, match="unverified placeholder"):
            backend._load_model()

    def test_model_loading_rejects_custom_unverified_revision(self):
        """Test that custom NEEDS_VERIFICATION revisions are also rejected."""
        backend = SAM2Backend(revision="NEEDS_VERIFICATION_CUSTOM_EXPERIMENT")

        with pytest.raises(ValueError, match="NEEDS_VERIFICATION"):
            backend._load_model()


class TestSAM2BackendSegmentAuto:
    """Test automatic segmentation implementation."""

    def test_segment_auto_raises_not_implemented(self):
        """Test that _segment_auto raises NotImplementedError."""
        backend = SAM2Backend()
        backend._model = MagicMock()  # Pretend model is loaded
        backend._processor = MagicMock()

        # Create input
        seg_input = SegmentationInput(
            image=np.random.rand(100, 100, 3).astype(np.float32),
            gamma=1.0,
            mode="auto",
        )

        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="automatic mask generation"):
            backend._segment_auto(seg_input)
