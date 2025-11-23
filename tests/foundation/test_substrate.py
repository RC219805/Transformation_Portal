"""
Tests for Computational Substrate - Phase 1 Foundation Architecture
"""

import pytest
import torch

from transformation_portal.foundation import (
    ComputationalSubstrate,
    SubstrateConfig,
    PrecisionMode,
)


class TestComputationalSubstrate:
    """Test suite for computational substrate."""

    @pytest.fixture
    def substrate(self):
        """Create substrate instance for testing."""
        config = SubstrateConfig.for_development()
        config.enable_profiling = True  # Enable for testing
        return ComputationalSubstrate(config)

    def test_initialization(self, substrate):
        """Test substrate initialization."""
        assert substrate is not None
        assert substrate.device is not None
        assert substrate.device_info is not None

    def test_device_detection(self, substrate):
        """Test device detection."""
        device = substrate.get_device()
        assert device is not None
        assert device.type in ["mps", "cuda", "cpu"]

    def test_capabilities_detection(self, substrate):
        """Test hardware capabilities detection."""
        caps = substrate.get_capabilities()
        assert "device_name" in caps
        assert "total_memory_gb" in caps
        assert "performance_cores" in caps
        assert caps["total_memory_gb"] > 0

    def test_tensor_allocation(self, substrate):
        """Test tensor allocation."""
        tensor = substrate.allocate_tensor((100, 100))
        assert tensor is not None
        assert tensor.device == substrate.device
        assert tensor.shape == (100, 100)

    def test_tensor_allocation_with_dtype(self, substrate):
        """Test tensor allocation with specific dtype."""
        tensor = substrate.allocate_tensor((50, 50), dtype=torch.float32)
        assert tensor.dtype == torch.float32

        tensor_fp16 = substrate.allocate_tensor((50, 50), dtype=torch.float16)
        assert tensor_fp16.dtype == torch.float16

    def test_batch_allocation(self, substrate):
        """Test batch tensor allocation."""
        batch_size = 4
        shape = (64, 64)
        tensors = substrate.memory_manager.allocate_batch(
            batch_size, shape
        )
        assert len(tensors) == batch_size
        for tensor in tensors:
            assert tensor.shape == shape

    def test_memory_stats(self, substrate):
        """Test memory statistics."""
        stats = substrate.get_memory_stats()
        assert "device" in stats
        assert "strategy" in stats

    def test_performance_monitoring(self, substrate):
        """Test performance monitoring."""
        # Allocate some tensors
        for _ in range(10):
            tensor = substrate.allocate_tensor((100, 100))
            _ = tensor * 2.0

        # Get performance summary
        summary = substrate.get_performance_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_memory_optimization(self, substrate):
        """Test memory optimization."""
        # Allocate tensors
        tensors = [substrate.allocate_tensor((100, 100)) for _ in range(10)]
        # Assert all tensors are valid
        for tensor in tensors:
            assert tensor is not None
            assert tensor.shape == (100, 100)

        # Run optimization
        substrate.optimize_memory()

        # Should not crash
        assert True

    def test_cache_clearing(self, substrate):
        """Test cache clearing."""
        # Allocate some tensors
        _ = substrate.allocate_tensor((100, 100))

        # Clear cache
        substrate.clear_cache()

        # Should not crash
        assert True

    def test_autocast_context(self, substrate):
        """Test autocast context manager."""
        with substrate.autocast():
            tensor = torch.randn(10, 10, device=substrate.device)
            result = tensor * 2.0
            assert result is not None

    def test_profile_context(self, substrate):
        """Test profiling context manager."""
        with substrate.profile("test_operation"):
            tensor = substrate.allocate_tensor((100, 100))
            _ = tensor * 2.0

        # Check that operation was recorded
        profile = substrate.performance_monitor.collector.get_operation_profile("test_operation")
        assert profile is not None
        assert profile.total_calls >= 1

    def test_to_device(self, substrate):
        """Test moving tensors to device."""
        cpu_tensor = torch.randn(10, 10)
        device_tensor = substrate.to_device(cpu_tensor)

        assert device_tensor.device == substrate.device

    def test_status_reporting(self, substrate):
        """Test status reporting."""
        status = substrate.get_status()
        assert "device" in status
        assert "capabilities" in status
        assert "memory" in status
        assert "configuration" in status

    def test_repr(self, substrate):
        """Test string representation."""
        repr_str = repr(substrate)
        assert "ComputationalSubstrate" in repr_str
        assert "device=" in repr_str

    def test_str(self, substrate):
        """Test human-readable string."""
        str_repr = str(substrate)
        assert "COMPUTATIONAL SUBSTRATE" in str_repr
        assert "Phase 1" in str_repr.upper()


class TestSubstrateConfig:
    """Test suite for substrate configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = SubstrateConfig()
        assert config.precision == PrecisionMode.FP16
        assert config.enable_amp is True

    def test_m4_max_config(self):
        """Test M4 Max optimized configuration."""
        config = SubstrateConfig.for_m4_max()
        assert config.prefer_ane is True
        assert config.memory_fraction == 0.85
        assert config.max_memory_gb == 108.0

    def test_development_config(self):
        """Test development configuration."""
        config = SubstrateConfig.for_development()
        assert config.enable_profiling is True
        assert config.compile_mode is None

    def test_production_config(self):
        """Test production configuration."""
        config = SubstrateConfig.for_production()
        assert config.enable_profiling is False
        assert config.compile_mode == "reduce-overhead"


class TestIntegration:
    """Integration tests for full substrate."""

    def test_end_to_end_workflow(self):
        """Test complete workflow from initialization to computation."""
        # Initialize substrate
        config = SubstrateConfig.for_m4_max()
        substrate = ComputationalSubstrate(config)

        # Allocate tensors
        input_tensor = substrate.allocate_tensor((1, 3, 224, 224))
        weight_tensor = substrate.allocate_tensor((64, 3, 7, 7))

        # Perform computation
        with substrate.autocast():
            # Simple convolution-like operation
            result = torch.conv2d(
                input_tensor,
                weight_tensor,
                padding=3
            )

        # Verify result
        assert result is not None
        assert result.device == substrate.device

        # Get performance stats
        stats = substrate.get_memory_stats()
        assert stats is not None

        # Cleanup
        substrate.clear_cache()

    def test_batch_processing_workflow(self):
        """Test batch processing workflow."""
        substrate = ComputationalSubstrate()

        # Create batch of tensors
        batch_size = 4
        tensors = [
            substrate.allocate_tensor((3, 128, 128))
            for _ in range(batch_size)
        ]

        # Define operation
        def process_fn(x):
            return x * 2.0 + 1.0

        # Process batch
        results = substrate.process_batch(tensors, process_fn)

        # Verify results
        assert len(results) == batch_size
        for result in results:
            assert result.device == substrate.device

    def test_memory_pressure_handling(self):
        """Test handling of memory pressure."""
        substrate = ComputationalSubstrate()

        # Allocate many tensors to create memory pressure
        tensors = []
        for i in range(50):
            try:
                tensor = substrate.allocate_tensor((100, 100, 100))
                tensors.append(tensor)
            except RuntimeError:
                # Expected to eventually hit memory limits
                break

        # Optimize memory
        substrate.optimize_memory()

        # Clear references
        tensors.clear()
        substrate.clear_cache()

        # Should be able to allocate again
        new_tensor = substrate.allocate_tensor((100, 100, 100))
        assert new_tensor is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_tensor_shape(self):
        """Test handling of invalid tensor shapes."""
        substrate = ComputationalSubstrate()

        with pytest.raises((ValueError, RuntimeError, TypeError)):
            substrate.allocate_tensor((-1, 100))  # Invalid negative dimension

    def test_zero_size_tensor(self):
        """Test handling of zero-size tensors."""
        substrate = ComputationalSubstrate()

        # This should work (zero-size tensors are valid in PyTorch)
        tensor = substrate.allocate_tensor((0, 10))
        assert tensor.numel() == 0

    def test_very_large_tensor(self):
        """Test handling of very large tensor allocation."""
        substrate = ComputationalSubstrate()

        # Try to allocate tensor larger than available memory
        # Should either succeed (if memory allows) or raise appropriate error
        try:
            huge_tensor = substrate.allocate_tensor((10000, 10000, 1000))
            # If successful, clean up
            del huge_tensor
            substrate.clear_cache()
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            # Expected for large allocations
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
