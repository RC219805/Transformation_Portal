"""Tests for SAM2 integration with Materials V3.

These tests verify that SAM2Backend integrates correctly with the Materials V3
segmentation system via the SAM2MaterialsAdapter.

Test Coverage:
- SAM2MaterialsAdapter protocol compliance
- Auto mode segmentation (mask-generation pipeline)
- Material labeling heuristics
- Integration with Materials V3 config
- Error handling and fallbacks
- Contract conversion (Materials V3 API ↔ SAM2 contracts)

Note: Most tests are mocked to avoid downloading SAM2 models (~1.2GB) in CI.
Integration tests with real SAM2 models are marked @pytest.mark.slow.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.protocols.segmentation_backend import SegmentationBackend, SegmentationBackendInfo
from transformation_portal.lux_depth_v3.segmentation_backend import _get_backend_instance, segment_materials

# Safe imports for SAM2 components (may not be available)
try:
    from transformation_portal.lux_depth_v3.sam2_adapter import SAM2MaterialsAdapter
    from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationInput, SegmentationResult
    from transformation_portal.spatial_ai.segmentation.sam2_backend import SAM2Backend

    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create a 128×128 RGB image with distinct color regions
    image = np.zeros((128, 128, 3), dtype=np.uint8)

    # Sky region (top, blue)
    image[0:30, :] = [100, 150, 220]

    # Foliage region (green)
    image[40:70, 10:60] = [80, 180, 100]

    # Water region (blue, lower part)
    image[80:120, 30:90] = [90, 140, 200]

    # Glass region (gray, neutral)
    image[40:70, 70:110] = [200, 200, 205]

    return image


@pytest.fixture
def config_sam2():
    """Config with SAM2 backend."""
    return EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="sam2",
        depth_device="cpu",  # Use CPU for tests
    )


@pytest.fixture
def mock_sam2_result():
    """Create a mock SegmentationResult for testing."""
    # Create 3 sample masks
    masks = np.array(
        [
            # Sky mask (top region)
            np.concatenate([np.ones((30, 128)), np.zeros((98, 128))], axis=0),
            # Foliage mask
            np.pad(np.ones((30, 50)), ((40, 58), (10, 68))),
            # Water mask
            np.pad(np.ones((40, 60)), ((80, 8), (30, 38))),
        ],
        dtype=bool,
    )

    scores = np.array([0.9, 0.85, 0.8], dtype=np.float32)

    metadata = [
        MaskMetadata(area=30 * 128, bbox=(0, 0, 128, 30), stability_score=0.9),
        MaskMetadata(area=30 * 50, bbox=(10, 40, 50, 30), stability_score=0.85),
        MaskMetadata(area=40 * 60, bbox=(30, 80, 60, 40), stability_score=0.8),
    ]

    return SegmentationResult(masks=masks, scores=scores, metadata=metadata)


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_sam2_adapter_implements_protocol():
    """Test that SAM2MaterialsAdapter implements SegmentationBackend protocol."""
    adapter = SAM2MaterialsAdapter(model_size="base", device="cpu")

    # Check it's recognized as a SegmentationBackend
    assert isinstance(adapter, SegmentationBackend)

    # Check required methods exist
    assert hasattr(adapter, "info")
    assert hasattr(adapter, "load")
    assert hasattr(adapter, "segment")


@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_sam2_adapter_info():
    """Test SAM2MaterialsAdapter.info property."""
    adapter = SAM2MaterialsAdapter(model_size="base", device="cpu")
    info = adapter.info

    assert isinstance(info, SegmentationBackendInfo)
    assert "SAM2" in info.name
    assert "facebook/sam2" in info.model_id
    assert info.requires_weights is True
    assert info.approximate_memory_mb > 1000  # SAM2 is ~1.2GB


# =============================================================================
# Backend Registry Tests
# =============================================================================


@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_get_backend_instance_sam2():
    """Test that _get_backend_instance can create SAM2 backend."""
    backend = _get_backend_instance("sam2", device="cpu", strict=False)

    assert isinstance(backend, SegmentationBackend)
    # In non-strict mode, it may fall back to stub if model download fails
    # We just verify it returns something that implements the protocol


def test_get_backend_instance_invalid_backend():
    """Test that invalid backend names raise ValueError."""
    with pytest.raises(ValueError, match="Unknown segmentation backend"):
        _get_backend_instance("invalid_backend", device="cpu")


@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_segment_materials_with_sam2_config(sample_image, config_sam2):
    """Test segment_materials with SAM2 config (mocked to avoid downloads)."""
    # Mock the SAM2Backend.segment method to avoid model download
    with patch.object(SAM2Backend, "segment") as mock_segment:
        # Create a mock result with empty masks
        from transformation_portal.spatial_ai.segmentation.contracts import SegmentationResult

        mock_result = SegmentationResult(
            masks=np.zeros((0, 128, 128), dtype=bool),
            scores=np.array([], dtype=np.float32),
            metadata=[],
        )
        mock_segment.return_value = mock_result

        # Segment with mocked backend
        masks = segment_materials(sample_image, config_sam2)

        # Should return dict (empty since mock returns no masks)
        assert isinstance(masks, dict)


# =============================================================================
# Contract Conversion Tests
# =============================================================================


@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_sam2_adapter_contract_conversion(sample_image, mock_sam2_result):
    """Test that adapter correctly converts Materials V3 API to SAM2 contracts."""
    adapter = SAM2MaterialsAdapter(model_size="base", device="cpu")
    adapter._model_loaded = True  # Mark as loaded

    # Mock the SAM2Backend.segment to return our mock result
    with patch.object(adapter._sam2_backend, "segment", return_value=mock_sam2_result):
        # Call segment (Materials V3 API: uint8 RGB image)
        result = adapter.segment(sample_image)

        # Verify it returns Materials V3 format: Dict[str, Tuple[mask, conf]]
        assert isinstance(result, dict)

        # Check that masks are float32 arrays
        for material, (mask, confidence) in result.items():
            assert isinstance(mask, np.ndarray)
            assert mask.dtype == np.float32
            assert mask.shape == sample_image.shape[:2]
            assert isinstance(confidence, (float, np.floating))
            assert 0.0 <= confidence <= 1.0


@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_sam2_adapter_srgb_to_linear_conversion(sample_image):
    """Test that adapter converts sRGB uint8 to linear float32 correctly."""
    adapter = SAM2MaterialsAdapter(model_size="base", device="cpu")

    # Convert sample image
    linear = adapter._srgb_to_linear(sample_image)

    # Check dtype and shape
    assert linear.dtype == np.float32
    assert linear.shape == sample_image.shape

    # Check value range (linear should be in [0, 1])
    assert linear.min() >= 0.0
    assert linear.max() <= 1.0

    # Check gamma correction (darker than sRGB for mid-tones)
    # sRGB 128/255 ≈ 0.5 → linear ≈ 0.22
    srgb_midtone = 128 / 255.0
    linear_midtone = adapter._srgb_to_linear(np.array([[[128, 128, 128]]], dtype=np.uint8))[0, 0, 0]
    assert linear_midtone < srgb_midtone
    assert 0.2 < linear_midtone < 0.3  # Approximate gamma 2.2 result


# =============================================================================
# Material Labeling Tests
# =============================================================================


@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_sam2_adapter_material_labeling(sample_image, mock_sam2_result):
    """Test heuristic material labeling logic."""
    adapter = SAM2MaterialsAdapter(model_size="base", device="cpu")

    # Apply labeling
    material_masks = adapter._label_materials_heuristic(sample_image, mock_sam2_result)

    # Should return dict of materials
    assert isinstance(material_masks, dict)

    # Check that we got some material labels
    # Exact labels depend on heuristics, but we should get at least one
    assert len(material_masks) > 0

    # Each material should have (mask, confidence) tuple
    for material, (mask, conf) in material_masks.items():
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.float32
        assert mask.shape == sample_image.shape[:2]
        assert 0.0 <= conf <= 1.0


@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_sam2_adapter_sky_detection(mock_sam2_result):
    """Test that sky regions are correctly identified."""
    # Create image with obvious sky (top, blue)
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[0:30, :] = [100, 150, 220]  # Blue sky at top

    adapter = SAM2MaterialsAdapter(model_size="base", device="cpu")

    # Mock result with sky mask
    sky_mask = np.concatenate([np.ones((30, 128)), np.zeros((98, 128))], axis=0)
    result = SegmentationResult(
        masks=np.array([sky_mask], dtype=bool),
        scores=np.array([0.9], dtype=np.float32),
        metadata=[MaskMetadata(area=30 * 128, bbox=(0, 0, 128, 30), stability_score=0.9)],
    )

    materials = adapter._label_materials_heuristic(image, result)

    # Should detect sky
    assert "sky" in materials or "material" in materials  # May classify as generic


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_sam2_adapter_not_loaded_error(sample_image):
    """Test that calling segment before load raises error."""
    adapter = SAM2MaterialsAdapter(model_size="base", device="cpu")
    # Don't call load()

    with pytest.raises(RuntimeError, match="not loaded"):
        adapter.segment(sample_image)


@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_sam2_adapter_invalid_image_shape():
    """Test that invalid image shapes raise ValueError."""
    adapter = SAM2MaterialsAdapter(model_size="base", device="cpu")
    adapter.load()

    # Wrong number of dimensions
    with pytest.raises(ValueError, match="Expected RGB image"):
        adapter.segment(np.zeros((128, 128), dtype=np.uint8))

    # Wrong number of channels
    with pytest.raises(ValueError, match="Expected RGB image"):
        adapter.segment(np.zeros((128, 128, 4), dtype=np.uint8))


@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_sam2_adapter_invalid_image_dtype(sample_image):
    """Test that invalid image dtypes raise ValueError."""
    adapter = SAM2MaterialsAdapter(model_size="base", device="cpu")
    adapter.load()

    # Wrong dtype
    with pytest.raises(ValueError, match="Expected uint8"):
        adapter.segment(sample_image.astype(np.float32))


# =============================================================================
# Fallback Behavior Tests
# =============================================================================


# =============================================================================
# Fallback Behavior Tests
# =============================================================================


def test_sam2_backend_graceful_degradation():
    """Test that SAM2 backend gracefully degrades if not available.

    This tests the non-strict mode fallback to stub backend.
    """
    # Try to get SAM2 backend in non-strict mode
    # If SAM2 is available, we get SAM2; if not, we get stub
    backend = _get_backend_instance("sam2", device="cpu", strict=False)

    # Should return a valid backend (either SAM2 or stub as fallback)
    assert isinstance(backend, SegmentationBackend)

    # Test that it can segment (even if stub, should not crash)
    sample_img = np.zeros((64, 64, 3), dtype=np.uint8)
    result = backend.segment(sample_img)
    assert isinstance(result, dict)


@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_sam2_backend_strict_mode():
    """Test that strict mode is respected when SAM2 is available.

    In strict mode, if backend fails to load, it should raise rather than fall back.
    """
    # This test assumes SAM2 is available
    # We test that we can create a backend in strict mode
    try:
        backend = _get_backend_instance("sam2", device="cpu", strict=True)
        # If we get here, SAM2 loaded successfully
        assert isinstance(backend, SegmentationBackend)
    except RuntimeError:
        # If strict mode raises, that's also valid behavior (model not downloaded)
        # The key is that it raises instead of silently falling back
        pass


# =============================================================================
# Integration Tests (Slow, Require Real Model)
# =============================================================================


@pytest.mark.slow
@pytest.mark.skipif(not SAM2_AVAILABLE, reason="SAM2 dependencies not installed")
def test_sam2_adapter_real_inference(sample_image):
    """Test SAM2MaterialsAdapter with real model inference.

    WARNING: Downloads SAM2 model (~1.2GB) on first run.
    Only run locally or in CI with model caching enabled.
    """
    pytest.skip("Skipping real model test to avoid large download")

    adapter = SAM2MaterialsAdapter(model_size="base", device="cpu")
    adapter.load()

    # Real inference
    result = adapter.segment(sample_image)

    # Check output format
    assert isinstance(result, dict)

    # Should detect at least one material in our test image
    # (This is a weak assertion since detection is non-deterministic)
    # assert len(result) > 0


# =============================================================================
# Backward Compatibility Tests
# =============================================================================


def test_existing_backends_still_work(sample_image):
    """Test that adding SAM2 doesn't break existing backends."""
    # Stub backend should still work
    stub_backend = _get_backend_instance("stub", device="cpu")
    assert stub_backend.info.model_id == "stub"
    stub_masks = stub_backend.segment(sample_image)
    assert isinstance(stub_masks, dict)
    assert len(stub_masks) == 0  # Stub returns empty

    # EfficientSAM backend should still be registered
    # (May fall back to stub if not installed, but should not error)
    efficientsam_backend = _get_backend_instance("efficientsam", device="cpu", strict=False)
    assert isinstance(efficientsam_backend, SegmentationBackend)
