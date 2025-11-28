"""
Tests for Device Manager
"""

import pytest

# Skip if torch not available (optional ML dependency)
torch = pytest.importorskip("torch")

from transformation_portal.foundation.device_manager import (
    DeviceManager,
    DeviceType,
    DeviceCapabilities,
)


class TestDeviceManager:
    """Test suite for device manager."""

    @pytest.fixture
    def device_manager(self):
        """Create device manager instance."""
        return DeviceManager()

    def test_initialization(self, device_manager):
        """Test device manager initialization."""
        assert device_manager is not None
        assert device_manager.prefer_ane is True
        assert 0.0 < device_manager.memory_fraction <= 1.0

    def test_device_detection(self, device_manager):
        """Test device detection."""
        device_info = device_manager.detect_devices()
        assert device_info is not None
        assert device_info.primary_device is not None

    def test_device_type(self, device_manager):
        """Test detected device type is valid."""
        device_type = device_manager._detect_device_type()
        assert device_type in [DeviceType.MPS, DeviceType.CUDA, DeviceType.CPU]

    def test_capabilities_detection(self, device_manager):
        """Test capabilities detection."""
        device_info = device_manager.detect_devices()
        caps = device_info.capabilities

        assert caps.device_name is not None
        assert caps.total_memory_gb > 0
        assert caps.available_memory_gb > 0
        assert caps.available_memory_gb <= caps.total_memory_gb

    def test_backend_priority(self, device_manager):
        """Test backend priority determination."""
        device_info = device_manager.detect_devices()
        priority = device_info.backend_priority

        assert len(priority) > 0
        # CPU should always be in fallback chain
        assert DeviceType.CPU in priority

    def test_optimization_config(self, device_manager):
        """Test optimization configuration generation."""
        device_info = device_manager.detect_devices()
        config = device_info.optimization_config

        assert "device_type" in config
        assert "precision" in config
        assert "max_batch_size" in config
        assert config["max_batch_size"] > 0

    def test_get_device(self, device_manager):
        """Test getting PyTorch device."""
        device = device_manager.get_device()
        assert isinstance(device, torch.device)

    def test_get_capabilities(self, device_manager):
        """Test getting capabilities."""
        caps = device_manager.get_capabilities()
        assert isinstance(caps, DeviceCapabilities)

    def test_get_optimization_config(self, device_manager):
        """Test getting optimization config."""
        config = device_manager.get_optimization_config()
        assert isinstance(config, dict)

    def test_memory_fraction_validation(self):
        """Test memory fraction validation."""
        # Valid fractions
        dm1 = DeviceManager(memory_fraction=0.5)
        assert dm1.memory_fraction == 0.5

        # Out of range (should be clamped)
        dm2 = DeviceManager(memory_fraction=1.5)
        assert dm2.memory_fraction <= 0.95

        dm3 = DeviceManager(memory_fraction=0.05)
        assert dm3.memory_fraction >= 0.1

    def test_prefer_ane_setting(self):
        """Test ANE preference setting."""
        dm_with_ane = DeviceManager(prefer_ane=True)
        dm_without_ane = DeviceManager(prefer_ane=False)

        assert dm_with_ane.prefer_ane is True
        assert dm_without_ane.prefer_ane is False

    def test_device_info_caching(self, device_manager):
        """Test that device info is cached after first detection."""
        # First detection
        info1 = device_manager.detect_devices()
        # Second detection should return cached info
        info2 = device_manager.detect_devices()

        assert info1 is info2  # Same object reference


class TestDeviceCapabilities:
    """Test device capabilities structure."""

    def test_mps_capabilities(self):
        """Test MPS capabilities if available."""
        if not torch.backends.mps.is_available():
            pytest.skip("MPS not available")

        dm = DeviceManager()
        info = dm.detect_devices()
        caps = info.capabilities

        if caps.device_type == DeviceType.MPS:
            assert caps.unified_memory is True
            assert caps.supports_fp16 is True
            assert caps.gpu_cores > 0

    def test_cuda_capabilities(self):
        """Test CUDA capabilities if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        dm = DeviceManager()
        info = dm.detect_devices()
        caps = info.capabilities

        if caps.device_type == DeviceType.CUDA:
            assert caps.unified_memory is False
            assert caps.supports_fp16 is True
            assert caps.gpu_cores > 0

    def test_cpu_capabilities(self):
        """Test CPU capabilities."""
        # Force CPU by mocking device detection
        dm = DeviceManager()
        caps = dm._detect_cpu_capabilities()

        assert caps.device_type == DeviceType.CPU
        assert caps.performance_cores > 0
        assert caps.total_memory_gb > 0


class TestM4MaxDetection:
    """Test M4 Max specific detection."""

    def test_is_m4_max(self):
        """Test M4 Max detection."""
        dm = DeviceManager()
        is_m4 = dm._is_m4_max()

        # Result depends on actual hardware, just verify it returns bool
        assert isinstance(is_m4, bool)

    def test_m4_max_capabilities(self):
        """Test M4 Max specific capabilities."""
        dm = DeviceManager()
        info = dm.detect_devices()

        # If running on M4 Max, verify specs
        if dm._is_m4_max():
            caps = info.capabilities
            # M4 Max has 16 cores (12P + 4E)
            assert caps.performance_cores + caps.efficiency_cores == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
