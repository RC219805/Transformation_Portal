"""Tests for ModelRegistry."""

import pytest
from transformation_portal.depth_canonical.models import ModelRegistry
from transformation_portal.depth_canonical.config import ModelVariant, DeviceType


def test_model_registry_initialization():
    """Test ModelRegistry initializes correctly."""
    registry = ModelRegistry()
    assert registry is not None


def test_model_registry_supports_da3_variants():
    """Test ModelRegistry recognizes DA3 variants as supported."""
    registry = ModelRegistry()

    assert registry.is_variant_supported(ModelVariant.DA3_METRIC_LARGE)
    assert registry.is_variant_supported(ModelVariant.DA3_METRIC_BASE)
    assert registry.is_variant_supported(ModelVariant.DA3_METRIC_SMALL)


def test_model_registry_supports_da2_variants():
    """Test ModelRegistry recognizes DA2 variants as supported."""
    registry = ModelRegistry()

    assert registry.is_variant_supported(ModelVariant.DA2_LARGE)
    assert registry.is_variant_supported(ModelVariant.DA2_BASE)


@pytest.mark.slow
def test_model_registry_get_model_returns_model():
    """Test get_model returns a model instance (Phase 2).

    Note: This is a slow test as it downloads and loads models.
    """
    registry = ModelRegistry()

    # Get a model (will download if needed)
    model = registry.get_model(
        variant=ModelVariant.DA3_METRIC_SMALL,  # Use small for faster testing
        device=DeviceType.CPU
    )

    # Phase 2: should return a model instance
    assert model is not None
    assert hasattr(model, "estimate")


@pytest.mark.slow
def test_model_registry_caches_models():
    """Test that models are cached and reused."""
    registry = ModelRegistry()

    # Load model twice
    model1 = registry.get_model(
        variant=ModelVariant.DA3_METRIC_SMALL,
        device=DeviceType.CPU
    )

    model2 = registry.get_model(
        variant=ModelVariant.DA3_METRIC_SMALL,
        device=DeviceType.CPU
    )

    # Should be same instance
    assert model1 is model2


def test_model_registry_auto_detects_device():
    """Test device auto-detection works."""
    registry = ModelRegistry()

    # Auto-detect device (should not crash)
    device = registry._auto_detect_device()

    # Should return a valid device
    assert isinstance(device, DeviceType)
    assert device in {DeviceType.CPU, DeviceType.CUDA, DeviceType.MPS, DeviceType.COREML}


def test_model_registry_clear_cache():
    """Test clear_cache works."""
    registry = ModelRegistry()

    # Clear cache (should not crash)
    registry.clear_cache()

    assert len(registry._models) == 0
