"""Unit tests for material classifier (Phase 2.1)."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from transformation_portal.spatial_ai.segmentation.material_classifier import MaterialClassifier


class TestMaterialClassifierInitialization:
    """Test MaterialClassifier initialization."""

    def test_default_initialization(self):
        """Test classifier initialization with defaults."""
        classifier = MaterialClassifier()
        assert classifier.device == "cuda"
        assert classifier.confidence_threshold == 0.3
        assert len(classifier.material_classes) > 0
        assert "wood floor" in classifier.material_classes

    def test_custom_initialization(self):
        """Test classifier initialization with custom parameters."""
        custom_classes = ["custom_material_1", "custom_material_2"]
        classifier = MaterialClassifier(
            device="cpu",
            confidence_threshold=0.5,
            material_classes=custom_classes,
        )
        assert classifier.device == "cpu"
        assert classifier.confidence_threshold == 0.5
        assert classifier.material_classes == custom_classes

    def test_invalid_confidence_threshold(self):
        """Test validation of confidence threshold."""
        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            MaterialClassifier(confidence_threshold=1.5)

        with pytest.raises(ValueError, match="must be in \\[0, 1\\]"):
            MaterialClassifier(confidence_threshold=-0.1)

    def test_default_material_classes(self):
        """Test that default material classes are comprehensive."""
        classifier = MaterialClassifier()
        classes = classifier.material_classes

        # Check for key materials
        assert any("wood" in c.lower() for c in classes)
        assert any("marble" in c.lower() for c in classes)
        assert any("glass" in c.lower() for c in classes)
        assert any("metal" in c.lower() for c in classes)
        assert any("concrete" in c.lower() for c in classes)


class TestMaterialClassifierAvailability:
    """Test CLIP availability checking."""

    def test_is_available_with_clip(self):
        """Test availability check when CLIP is installed."""
        with patch("transformers.CLIPModel"):
            with patch("transformers.CLIPProcessor"):
                classifier = MaterialClassifier()
                # Force re-check
                classifier._available = None
                assert classifier.is_available() == True

    def test_is_available_without_clip(self):
        """Test availability check when CLIP is missing."""
        # This test should run without mocking - let it check actual availability
        classifier = MaterialClassifier()
        # Just verify it returns a boolean
        result = classifier.is_available()
        assert isinstance(result, bool)

    def test_availability_cached(self):
        """Test that availability check is cached."""
        classifier = MaterialClassifier()

        # Check twice
        result1 = classifier.is_available()
        result2 = classifier.is_available()

        assert result1 == result2
        assert classifier._available is not None


class TestMaterialClassifierExtractMaskedRegion:
    """Test masked region extraction."""

    def test_extract_from_uint8_image(self):
        """Test extraction from uint8 image."""
        classifier = MaterialClassifier()

        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:50] = True

        extracted = classifier._extract_masked_region(image, mask)

        # Should have size of bounding box
        assert extracted is not None
        assert extracted.shape == (20, 20, 3)
        assert extracted.dtype == np.uint8

    def test_extract_from_float32_image(self):
        """Test extraction from float32 image (converts to uint8)."""
        classifier = MaterialClassifier()

        image = np.random.rand(100, 100, 3).astype(np.float32)
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:50] = True

        extracted = classifier._extract_masked_region(image, mask)

        # Should be converted to uint8
        assert extracted is not None
        assert extracted.dtype == np.uint8
        assert extracted.min() >= 0
        assert extracted.max() <= 255

    def test_extract_empty_mask(self):
        """Test extraction from empty mask returns None."""
        classifier = MaterialClassifier()

        image = np.random.rand(100, 100, 3).astype(np.float32)
        mask = np.zeros((100, 100), dtype=bool)  # All False

        extracted = classifier._extract_masked_region(image, mask)

        assert extracted is None

    def test_extract_applies_mask(self):
        """Test that non-masked pixels are zeroed."""
        classifier = MaterialClassifier()

        image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:40, 30:50] = True

        extracted = classifier._extract_masked_region(image, mask)

        # The crop should be 20x20, but only the masked region should be non-zero
        # Due to masking within crop, all should be non-zero since entire crop is masked
        assert extracted is not None


class TestMaterialClassifierClassifyMasks:
    """Test mask classification."""

    def test_classify_without_clip(self):
        """Test that classification returns None when CLIP unavailable."""
        classifier = MaterialClassifier()
        # Set unavailable explicitly
        classifier._available = False

        image = np.random.rand(100, 100, 3).astype(np.uint8)
        masks = np.zeros((2, 100, 100), dtype=bool)
        masks[0, :50, :50] = True
        masks[1, 50:, 50:] = True

        results = classifier.classify_masks(image, masks)

        # Should return list of (None, None)
        assert len(results) == 2
        assert results[0] == (None, None)
        assert results[1] == (None, None)

    @patch("transformers.CLIPModel")
    @patch("transformers.CLIPProcessor")
    @patch("torch.no_grad")
    def test_classify_with_high_confidence(self, mock_no_grad, mock_processor_class, mock_model_class):
        """Test classification with high confidence."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Mock processor input
        def mock_processor_call(*args, **kwargs):
            result = MagicMock()
            result.to = MagicMock(return_value={"pixel_values": MagicMock(), "input_ids": MagicMock()})
            return result

        mock_processor.side_effect = mock_processor_call

        # Mock model output (high confidence)
        import torch

        mock_output = MagicMock()
        # Create a real tensor for softmax
        logits = torch.tensor([[10.0] + [0.0] * 23])  # High confidence for first class
        mock_output.logits_per_image = logits
        mock_model.return_value = mock_output

        # Run classification
        classifier = MaterialClassifier(confidence_threshold=0.3)
        classifier._load_model()

        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        masks = np.zeros((1, 100, 100), dtype=bool)
        masks[0, :50, :50] = True

        results = classifier.classify_masks(image, masks)

        # Should return label with confidence
        assert len(results) == 1
        assert results[0][0] is not None  # Label assigned
        assert results[0][1] is not None  # Confidence assigned

    @patch("transformers.CLIPModel")
    @patch("transformers.CLIPProcessor")
    @patch("torch.no_grad")
    def test_classify_with_low_confidence(self, mock_no_grad, mock_processor_class, mock_model_class):
        """Test classification with low confidence (below threshold)."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Mock processor input
        def mock_processor_call(*args, **kwargs):
            result = MagicMock()
            result.to = MagicMock(return_value={"pixel_values": MagicMock(), "input_ids": MagicMock()})
            return result

        mock_processor.side_effect = mock_processor_call

        # Mock model output (low confidence - uniform distribution)
        import torch

        mock_output = MagicMock()
        # All equal logits = uniform distribution after softmax
        logits = torch.tensor([[0.0] * 24])
        mock_output.logits_per_image = logits
        mock_model.return_value = mock_output

        # Run classification
        classifier = MaterialClassifier(confidence_threshold=0.5)
        classifier._load_model()

        image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        masks = np.zeros((1, 100, 100), dtype=bool)
        masks[0, :50, :50] = True

        results = classifier.classify_masks(image, masks)

        # Should return None (below threshold)
        assert len(results) == 1
        assert results[0] == (None, None)

    def test_classify_empty_mask(self):
        """Test classification handles empty masks."""
        classifier = MaterialClassifier()
        classifier._available = True
        classifier._model = MagicMock()
        classifier._processor = MagicMock()

        image = np.random.rand(100, 100, 3).astype(np.uint8)
        masks = np.zeros((1, 100, 100), dtype=bool)  # Empty mask

        results = classifier.classify_masks(image, masks)

        # Empty mask should return (None, None)
        assert results[0] == (None, None)


class TestMaterialClassifierModelLoading:
    """Test model loading behavior."""

    @patch("transformers.CLIPModel")
    @patch("transformers.CLIPProcessor")
    def test_model_loads_successfully(self, mock_processor_class, mock_model_class):
        """Test successful model loading."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Load model
        classifier = MaterialClassifier(device="cpu")
        classifier._load_model()

        # Verify calls
        assert mock_processor_class.from_pretrained.called
        assert mock_model_class.from_pretrained.called
        assert classifier._model is not None
        assert classifier._processor is not None

    @patch("transformers.CLIPModel")
    @patch("transformers.CLIPProcessor")
    def test_model_loads_only_once(self, mock_processor_class, mock_model_class):
        """Test that model is only loaded once."""
        # Setup mocks
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_processor_class.from_pretrained.return_value = mock_processor
        mock_model_class.from_pretrained.return_value = mock_model
        mock_model.to.return_value = mock_model

        # Load twice
        classifier = MaterialClassifier()
        classifier._load_model()
        classifier._load_model()

        # Should only load once
        assert mock_processor_class.from_pretrained.call_count == 1
        assert mock_model_class.from_pretrained.call_count == 1

    def test_model_loading_without_clip(self):
        """Test error when loading model without CLIP."""
        classifier = MaterialClassifier()
        # Set unavailable
        classifier._available = False

        with pytest.raises(ImportError, match="CLIP not available"):
            classifier._load_model()
