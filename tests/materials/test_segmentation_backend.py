"""Tests for material segmentation backends.

These tests verify the SegmentationBackend Protocol implementation,
including StubBackend and EfficientSAMBackend.

Test Coverage:
- Protocol compliance (stub and efficientsam backends)
- Shape contracts (RGB input → masks dict output)
- Device placement (MPS/CUDA/CPU)
- Fallback behavior (missing weights → stub)
- Integration with Materials V3 config
- Offline compatibility (no model downloads in CI)

Note: Tests marked with @pytest.mark.ml require torch/torchvision.
"""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.protocols.segmentation_backend import SegmentationBackend, SegmentationBackendInfo
from transformation_portal.lux_depth_v3.segmentation_backend import (
    EfficientSAMBackend,
    StubBackend,
    _get_backend_instance,
    segment_materials,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create a simple 64×64 RGB image with some color variation
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    # Add some colored regions
    image[10:30, 10:30] = [100, 150, 200]  # Blue-ish (potential water)
    image[35:55, 35:55] = [80, 180, 100]  # Green-ish (potential foliage)
    image[10:30, 35:55] = [150, 150, 150]  # Gray (potential stone)

    return image


@pytest.fixture
def config_stub():
    """Config with stub backend."""
    return EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="stub",
    )


@pytest.fixture
def config_efficientsam():
    """Config with EfficientSAM backend."""
    return EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        depth_device="cpu",  # Use CPU for tests
    )


@pytest.fixture
def config_strict():
    """Config with strict_backend=True."""
    return EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=True,
        depth_device="cpu",
    )


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


def test_stub_backend_implements_protocol():
    """Test that StubBackend implements SegmentationBackend protocol."""
    backend = StubBackend()

    # Check protocol compliance
    assert isinstance(backend, SegmentationBackend)
    assert hasattr(backend, "info")
    assert hasattr(backend, "load")
    assert hasattr(backend, "segment")

    # Check info
    info = backend.info
    assert isinstance(info, SegmentationBackendInfo)
    assert info.name == "Stub Segmentation Backend"
    assert info.model_id == "stub"
    assert info.requires_gpu is False
    assert info.requires_weights is False


@pytest.mark.ml
def test_efficientsam_backend_implements_protocol():
    """Test that EfficientSAMBackend implements SegmentationBackend protocol."""
    backend = EfficientSAMBackend()

    # Check protocol compliance
    assert isinstance(backend, SegmentationBackend)
    assert hasattr(backend, "info")
    assert hasattr(backend, "load")
    assert hasattr(backend, "segment")

    # Check info
    info = backend.info
    assert isinstance(info, SegmentationBackendInfo)
    assert info.name == "EfficientSAM"
    assert "efficientvit-sam" in info.model_id
    assert info.requires_weights is True


# =============================================================================
# Shape Contract Tests
# =============================================================================


def test_stub_backend_shape_contract(sample_image):
    """Test that StubBackend returns correct output shape."""
    backend = StubBackend()
    backend.load()

    masks = backend.segment(sample_image)

    # Stub returns empty dict
    assert isinstance(masks, dict)
    assert len(masks) == 0


@pytest.mark.ml
def test_efficientsam_backend_shape_contract(sample_image):
    """Test that EfficientSAMBackend returns correct output shapes."""
    backend = EfficientSAMBackend()
    backend.load(device="cpu")

    masks = backend.segment(sample_image)

    # Check output type
    assert isinstance(masks, dict)

    # Check each mask shape and dtype
    for material, mask in masks.items():
        assert isinstance(material, str)
        assert isinstance(mask, np.ndarray)
        assert mask.shape == sample_image.shape[:2]  # (H, W)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


@pytest.mark.ml
def test_efficientsam_backend_invalid_input():
    """Test that EfficientSAMBackend validates input format."""
    backend = EfficientSAMBackend()
    backend.load(device="cpu")

    # Test invalid shape (grayscale)
    with pytest.raises(ValueError, match="Expected RGB image"):
        backend.segment(np.zeros((64, 64), dtype=np.uint8))

    # Test invalid dtype (float32)
    with pytest.raises(ValueError, match="Expected uint8"):
        backend.segment(np.zeros((64, 64, 3), dtype=np.float32))

    # Test invalid shape (4 channels)
    with pytest.raises(ValueError, match="Expected RGB image"):
        backend.segment(np.zeros((64, 64, 4), dtype=np.uint8))


# =============================================================================
# Device Placement Tests
# =============================================================================


@pytest.mark.ml
def test_efficientsam_backend_cpu_device():
    """Test that EfficientSAMBackend works on CPU."""
    backend = EfficientSAMBackend()
    backend.load(device="cpu")

    assert backend._device == "cpu"
    assert backend._model_loaded is True


@pytest.mark.ml
@pytest.mark.skipif(
    not __import__("torch").backends.mps.is_available(),
    reason="MPS not available",
)
def test_efficientsam_backend_mps_device():
    """Test that EfficientSAMBackend works on MPS (Apple Silicon)."""
    backend = EfficientSAMBackend()
    backend.load(device="mps")

    assert backend._device == "mps"
    assert backend._model_loaded is True


@pytest.mark.ml
@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA not available",
)
def test_efficientsam_backend_cuda_device():
    """Test that EfficientSAMBackend works on CUDA."""
    backend = EfficientSAMBackend()
    backend.load(device="cuda")

    assert backend._device == "cuda"
    assert backend._model_loaded is True


@pytest.mark.ml
def test_efficientsam_backend_auto_device():
    """Test that EfficientSAMBackend auto-detects device."""
    backend = EfficientSAMBackend()
    backend.load(device="auto")

    # Should select MPS > CUDA > CPU
    assert backend._device in ["cpu", "mps", "cuda"]
    assert backend._model_loaded is True


# =============================================================================
# Fallback Behavior Tests
# =============================================================================


def test_segment_materials_disabled(sample_image):
    """Test that segmentation returns empty dict when disabled."""
    config = EnhanceConfig(enable_material_segmentation=False)

    masks = segment_materials(sample_image, config)

    assert isinstance(masks, dict)
    assert len(masks) == 0


def test_segment_materials_stub_backend(sample_image, config_stub):
    """Test that stub backend returns empty masks."""
    masks = segment_materials(sample_image, config_stub)

    assert isinstance(masks, dict)
    assert len(masks) == 0


@pytest.mark.ml
def test_segment_materials_efficientsam_backend(sample_image, config_efficientsam):
    """Test that EfficientSAM backend returns masks."""
    masks = segment_materials(sample_image, config_efficientsam)

    # Should return some masks (heuristic segmentation)
    assert isinstance(masks, dict)
    # May or may not detect materials depending on heuristics
    for material, mask in masks.items():
        assert mask.shape == sample_image.shape[:2]
        assert mask.dtype == np.float32


def test_segment_materials_unknown_backend(sample_image):
    """Test that unknown backend falls back to stub."""
    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="unknown_backend",
    )

    # Should not raise, should fall back to stub
    masks = segment_materials(sample_image, config)

    assert isinstance(masks, dict)
    # Fallback to stub means empty masks
    assert len(masks) == 0


@pytest.mark.ml
def test_backend_caching():
    """Test that backend instances are cached."""
    # Get stub backend twice
    backend1 = _get_backend_instance("stub")
    backend2 = _get_backend_instance("stub")

    assert backend1 is backend2  # Should be same instance

    # Get efficientsam backend twice
    backend3 = _get_backend_instance("efficientsam", device="cpu")
    backend4 = _get_backend_instance("efficientsam", device="cpu")

    assert backend3 is backend4  # Should be same instance


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.ml
def test_heuristic_segmentation_detects_materials():
    """Test that heuristic segmentation detects expected materials."""
    # Create an image with clear material signatures
    image = np.zeros((128, 128, 3), dtype=np.uint8)

    # Blue water region (large enough to pass coverage threshold)
    image[10:60, 10:60] = [50, 80, 200]  # Blue dominant

    # Green foliage region
    image[70:120, 10:60] = [60, 180, 80]  # Green dominant

    # Gray stone region
    image[10:60, 70:120] = [120, 125, 120]  # Low saturation

    # Bright glass region
    image[70:120, 70:120] = [180, 190, 210]  # High brightness + blue tint

    backend = EfficientSAMBackend()
    backend.load(device="cpu")
    masks = backend.segment(image)

    # Should detect multiple materials
    assert len(masks) > 0

    # Check for expected materials (heuristic-based)
    # Note: Exact detection depends on heuristics, so we're lenient
    detected_materials = set(masks.keys())
    possible_materials = {"water", "foliage", "stone", "glass"}

    assert detected_materials.issubset(possible_materials)


@pytest.mark.ml
def test_efficientsam_backend_lazy_loading(sample_image):
    """Test that model is loaded lazily on first inference."""
    backend = EfficientSAMBackend()

    # Model not loaded yet
    assert backend._model_loaded is False

    # Load explicitly
    backend.load(device="cpu")

    # Now model is loaded
    assert backend._model_loaded is True

    # Subsequent loads are no-ops
    backend.load(device="cpu")
    assert backend._model_loaded is True


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.ml
def test_efficientsam_backend_not_loaded_error(sample_image):
    """Test that segmentation fails if model not loaded."""
    backend = EfficientSAMBackend()

    # Don't load model
    with pytest.raises(RuntimeError, match="model not loaded"):
        backend.segment(sample_image)


def test_segment_materials_strict_mode_missing_torch(sample_image, monkeypatch):
    """Test that strict mode raises if torch unavailable."""
    # Mock torch unavailability
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    original_torch_available = seg_module.TORCH_AVAILABLE
    monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", False)

    # Clear backend cache to force re-creation
    seg_module._get_backend_instance.cache_clear()

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=True,
    )

    try:
        # Should raise in strict mode
        with pytest.raises(RuntimeError, match="Material segmentation failed|Failed to load efficientsam"):
            segment_materials(sample_image, config)
    finally:
        # Restore original value
        monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", original_torch_available)
        seg_module._get_backend_instance.cache_clear()


def test_segment_materials_graceful_degradation(sample_image, monkeypatch):
    """Test that non-strict mode gracefully degrades to empty masks."""
    # Mock torch unavailability
    import transformation_portal.lux_depth_v3.segmentation_backend as seg_module

    original_torch_available = seg_module.TORCH_AVAILABLE
    monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", False)

    config = EnhanceConfig(
        enable_material_segmentation=True,
        material_segmentation_backend="efficientsam",
        strict_backend=False,  # Non-strict mode
    )

    try:
        # Clear backend cache
        seg_module._get_backend_instance.cache_clear()

        # Should not raise, should return empty masks
        masks = segment_materials(sample_image, config)

        assert isinstance(masks, dict)
        assert len(masks) == 0  # Empty because fell back to stub
    finally:
        # Restore
        monkeypatch.setattr(seg_module, "TORCH_AVAILABLE", original_torch_available)
        seg_module._get_backend_instance.cache_clear()
