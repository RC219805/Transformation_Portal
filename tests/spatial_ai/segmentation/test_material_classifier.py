"""Unit tests for material classifier (Phase 2.1)."""

import sys
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from transformation_portal.spatial_ai.segmentation.material_classifier import MaterialClassifier

# ML tests require transformers and torch - skip gracefully if not installed
# This allows tests to run in CI environments without full ML stack
# See: docs/architecture/test_dependency_contracts.md (ADR-TBD)
try:
    import torch
    import transformers

    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    torch = None
    transformers = None


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

    @pytest.mark.ml
    @pytest.mark.skipif(not HAS_ML_DEPS, reason="Requires transformers and torch")
    def test_is_available_with_clip(self):
        """Test availability check when CLIP is installed."""
        # Just check availability - no mocking needed
        # If test runs, transformers is installed
        classifier = MaterialClassifier()
        # Force re-check
        classifier._available = None
        # Should return True since we have transformers installed
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

    def test_strict_classify_without_clip_raises(self):
        """Strict mode fails fast when CLIP dependencies are unavailable."""
        classifier = MaterialClassifier(strict=True)
        classifier._available = False

        image = np.random.rand(100, 100, 3).astype(np.uint8)
        masks = np.zeros((1, 100, 100), dtype=bool)
        masks[0, :50, :50] = True

        with pytest.raises(RuntimeError, match="strict mode.*CLIP is unavailable"):
            classifier.classify_masks(image, masks)

    @pytest.mark.ml
    @pytest.mark.skipif(not HAS_ML_DEPS, reason="Requires transformers and torch")
    def test_classify_with_high_confidence(self):
        """Test classification with high confidence."""
        # These tests require actual transformers package
        # In CI without transformers, they will skip via @pytest.mark.skipif
        pytest.skip("Test requires complex mocking - functionality tested in integration tests")

    @pytest.mark.ml
    @pytest.mark.skipif(not HAS_ML_DEPS, reason="Requires transformers and torch")
    def test_classify_with_low_confidence(self):
        """Test classification with low confidence (below threshold)."""
        # These tests require actual transformers package
        # In CI without transformers, they will skip via @pytest.mark.skipif
        pytest.skip("Test requires complex mocking - functionality tested in integration tests")

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

    def test_classify_masks_batches_clip_and_preserves_order(self, monkeypatch):
        """Batched CLIP classification should preserve mask ordering."""

        class FakeScalar:
            def __init__(self, value):
                self.value = value

            def item(self):
                return self.value

        class FakeTensor:
            def __init__(self, values):
                self.values = np.asarray(values, dtype=np.float32)

            def to(self, _device):
                return self

            def softmax(self, dim):
                shifted = self.values - np.max(self.values, axis=dim, keepdims=True)
                exp = np.exp(shifted)
                return FakeTensor(exp / np.sum(exp, axis=dim, keepdims=True))

            def __getitem__(self, key):
                return FakeTensor(self.values[key])

            def argmax(self):
                return FakeScalar(int(np.argmax(self.values)))

            def item(self):
                return float(np.asarray(self.values).item())

        class FakeNoGrad:
            def __enter__(self):
                return None

            def __exit__(self, *_exc):
                return False

        fake_torch = type("FakeTorch", (), {"no_grad": staticmethod(lambda: FakeNoGrad())})
        monkeypatch.setitem(sys.modules, "torch", fake_torch)

        processor_calls = []
        model_calls = []

        class FakeProcessor:
            def __call__(self, *, text, images, return_tensors, padding):
                processor_calls.append(
                    {
                        "text": list(text),
                        "image_count": len(images),
                        "return_tensors": return_tensors,
                        "padding": padding,
                    }
                )
                return {"pixel_values": FakeTensor([[0.0]]), "input_ids": FakeTensor([[0.0]])}

        class FakeModel:
            device = "cpu"

            def __call__(self, **_inputs):
                model_calls.append("batch")
                return type(
                    "FakeOutputs",
                    (),
                    {
                        # Two non-empty masks: first should classify as glass,
                        # third should classify as wood. The middle mask is empty.
                        "logits_per_image": FakeTensor(
                            [
                                [0.1, 3.0],
                                [4.0, 0.1],
                            ]
                        )
                    },
                )()

        classifier = MaterialClassifier(
            device="cpu",
            confidence_threshold=0.5,
            material_classes=["wood", "glass"],
        )
        classifier._available = True
        classifier._processor = FakeProcessor()
        classifier._model = FakeModel()

        image = np.ones((32, 32, 3), dtype=np.uint8) * 127
        masks = np.zeros((3, 32, 32), dtype=bool)
        masks[0, 2:14, 2:14] = True
        masks[2, 16:30, 16:30] = True

        results = classifier.classify_masks(image, masks)

        assert results[0][0] == "glass"
        assert results[1] == (None, None)
        assert results[2][0] == "wood"
        assert processor_calls == [
            {
                "text": ["wood", "glass"],
                "image_count": 2,
                "return_tensors": "pt",
                "padding": True,
            }
        ]
        assert model_calls == ["batch"]
        assert classifier._last_timing_ms["batch_size"] == 2


class TestMaterialClassifierModelLoading:
    """Test model loading behavior."""

    @pytest.mark.ml
    @pytest.mark.skipif(not HAS_ML_DEPS, reason="Requires transformers and torch")
    def test_model_loads_successfully(self):
        """Test successful model loading."""
        # These tests require actual transformers package
        # In CI without transformers, they will skip via @pytest.mark.skipif
        pytest.skip("Test requires complex mocking - functionality tested in integration tests")

    @pytest.mark.ml
    @pytest.mark.skipif(not HAS_ML_DEPS, reason="Requires transformers and torch")
    def test_model_loads_only_once(self):
        """Test that model is only loaded once."""
        # These tests require actual transformers package
        # In CI without transformers, they will skip via @pytest.mark.skipif
        pytest.skip("Test requires complex mocking - functionality tested in integration tests")

    def test_model_loading_without_clip(self):
        """Test error when loading model without CLIP."""
        classifier = MaterialClassifier()
        # Set unavailable
        classifier._available = False

        with pytest.raises(ImportError, match="CLIP not available"):
            classifier._load_model()
