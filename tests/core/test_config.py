"""Tests for core config module."""

import pytest
from pathlib import Path

from transformation_portal.core.config import (
    ConfigSchema,
    DeviceConfig,
    PathsConfig,
    PerformanceConfig,
    OutputConfig,
    ValidationConfig,
    validate_config,
    ConfigValidationError,
    load_preset,
    register_preset,
    list_presets,
)


def test_device_config_defaults():
    """Test device config defaults."""
    config = DeviceConfig()
    assert config.device.value == "auto"
    assert config.precision.value == "fp16"
    assert 0.1 <= config.memory_fraction <= 0.95


def test_device_config_validation():
    """Test device config validation."""
    with pytest.raises(ValueError):
        DeviceConfig(memory_fraction=1.5)
    
    with pytest.raises(ValueError):
        DeviceConfig(memory_fraction=0.05)


def test_paths_config():
    """Test paths config."""
    config = PathsConfig(
        input_dir=Path("~/input"),
        output_dir=Path("~/output")
    )
    
    # Paths should be expanded
    assert config.input_dir.is_absolute()
    assert config.output_dir.is_absolute()


def test_performance_config():
    """Test performance config."""
    config = PerformanceConfig(
        tile_size=512,
        tile_overlap=64
    )
    
    assert config.tile_size == 512
    assert config.tile_overlap == 64
    assert config.tile_overlap < config.tile_size


def test_performance_config_validation():
    """Test performance config validation."""
    with pytest.raises(ValueError):
        PerformanceConfig(tile_size=512, tile_overlap=512)


def test_output_config():
    """Test output config."""
    config = OutputConfig()
    
    assert config.save_master is True
    assert config.write_outputs is True
    assert 0.01 <= config.preview_scale <= 1.0


def test_validation_config():
    """Test validation config."""
    config = ValidationConfig()
    
    assert config.enable_validation is True
    assert config.max_input_size_mb > 0
    assert len(config.allowed_extensions) > 0


def test_config_schema():
    """Test full config schema."""
    config = ConfigSchema()
    
    assert config.device is not None
    assert config.paths is not None
    assert config.performance is not None
    assert config.output is not None
    assert config.validation is not None


def test_config_to_dict():
    """Test config serialization."""
    config = ConfigSchema()
    data = config.to_dict()
    
    assert isinstance(data, dict)
    assert "device" in data
    assert "paths" in data


def test_config_from_dict():
    """Test config deserialization."""
    data = {
        "device": {"device": "cpu"},
        "performance": {"batch_size": 2}
    }
    
    config = ConfigSchema.from_dict(data)
    assert config.device.device.value == "cpu"
    assert config.performance.batch_size == 2


def test_validate_config():
    """Test config validation."""
    config = {
        "performance": {
            "tile_size": 512,
            "tile_overlap": 64,
            "batch_size": 4
        }
    }
    
    errors = validate_config(config)
    assert len(errors) == 0


def test_validate_config_errors():
    """Test config validation with errors."""
    config = {
        "device": {
            "memory_fraction": 1.5
        },
        "performance": {
            "tile_size": 512,
            "tile_overlap": 600,
            "batch_size": 0
        }
    }
    
    errors = validate_config(config)
    assert len(errors) > 0


def test_presets():
    """Test preset system."""
    # Should have default presets
    presets = list_presets()
    assert len(presets) > 0
    assert "photo_realistic" in presets
    
    # Load preset
    preset = load_preset("photo_realistic")
    assert preset is not None
    assert "performance" in preset


def test_register_preset():
    """Test preset registration."""
    custom_preset = {
        "performance": {"batch_size": 8},
        "extras": {"test": True}
    }
    
    register_preset("test_preset", custom_preset)
    
    loaded = load_preset("test_preset")
    assert loaded is not None
    assert loaded["extras"]["test"] is True
