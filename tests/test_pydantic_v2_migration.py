"""
Test Pydantic V2 migration - verify no deprecation warnings.

This test ensures that the migration from Pydantic V1 Config to V2 ConfigDict
is complete and no deprecation warnings are raised.
"""

import warnings
import pytest

from transformation_portal.core.config.schemas import (
    ConfigSchema,
    DeviceConfig,
    PathsConfig,
    PerformanceConfig,
    OutputConfig,
    ValidationConfig,
)


def test_no_pydantic_deprecation_warnings():
    """Test that no PydanticDeprecatedSince20 warnings are raised."""
    with warnings.catch_warnings(record=True) as w:
        # Set warnings to always trigger
        warnings.simplefilter("always")
        
        # Create config instances
        ConfigSchema()
        DeviceConfig()
        PathsConfig()
        PerformanceConfig()
        OutputConfig()
        ValidationConfig()
        
        # Check for Pydantic deprecation warnings
        pydantic_warnings = [
            warning for warning in w 
            if "PydanticDeprecatedSince20" in str(warning.category)
        ]
        
        assert len(pydantic_warnings) == 0, (
            f"Found {len(pydantic_warnings)} Pydantic deprecation warnings: "
            f"{[str(w.message) for w in pydantic_warnings]}"
        )


def test_config_dict_features():
    """Test that ConfigDict features work correctly."""
    # Test arbitrary_types_allowed
    config = ConfigSchema()
    assert config is not None
    
    # Test validate_assignment
    config.device.device = "cpu"
    assert config.device.device == "cpu"
    
    # Test extra="allow"
    config.custom_field = "custom_value"
    assert config.custom_field == "custom_value"
    
    # Test extras dict
    config.extras["test_key"] = "test_value"
    assert config.extras["test_key"] == "test_value"


def test_config_serialization_with_v2():
    """Test that config serialization works with V2."""
    config = ConfigSchema()
    
    # Test model_dump (V2 method)
    data = config.model_dump()
    assert isinstance(data, dict)
    assert "device" in data
    assert "paths" in data
    
    # Test to_dict wrapper
    data2 = config.to_dict()
    assert data == data2


def test_config_validation_with_v2():
    """Test that validation still works with V2."""
    # Valid config should work
    config = ConfigSchema(
        device={"device": "cpu", "precision": "fp32"},
        performance={"batch_size": 2}
    )
    assert config.device.device == "cpu"
    assert config.performance.batch_size == 2
    
    # Invalid config should raise errors
    with pytest.raises(Exception):
        ConfigSchema(device={"memory_fraction": 1.5})


def test_field_validators_with_v2():
    """Test that field validators work with V2."""
    # Test DeviceConfig validator
    with pytest.raises(ValueError):
        DeviceConfig(memory_fraction=1.5)
    
    with pytest.raises(ValueError):
        DeviceConfig(memory_fraction=0.05)
    
    # Test PerformanceConfig validator
    with pytest.raises(ValueError):
        PerformanceConfig(tile_size=512, tile_overlap=512)
    
    # Valid values should work
    device_config = DeviceConfig(memory_fraction=0.8)
    assert device_config.memory_fraction == 0.8
    
    perf_config = PerformanceConfig(tile_size=512, tile_overlap=64)
    assert perf_config.tile_overlap == 64
