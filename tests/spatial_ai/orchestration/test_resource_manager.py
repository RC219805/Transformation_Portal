"""Unit tests for resource manager (Phase 2.4)."""

import gc

import pytest

from transformation_portal.spatial_ai.orchestration.resource_manager import ResourceLimits, ResourceManager

pytestmark = pytest.mark.unit


class TestResourceLimits:
    """Test ResourceLimits dataclass."""

    def test_default_limits(self):
        """Test default resource limits."""
        limits = ResourceLimits()
        assert limits.max_gpu_memory_gb is None
        assert limits.max_cpu_memory_gb is None
        assert limits.max_models_loaded == 3
        assert limits.batch_size == 1
        assert limits.device_preference == ["cuda", "mps", "cpu"]

    def test_custom_limits(self):
        """Test custom resource limits."""
        limits = ResourceLimits(
            max_gpu_memory_gb=8.0,
            max_cpu_memory_gb=16.0,
            max_models_loaded=2,
            batch_size=4,
        )
        assert limits.max_gpu_memory_gb == 8.0
        assert limits.max_cpu_memory_gb == 16.0
        assert limits.max_models_loaded == 2
        assert limits.batch_size == 4

    def test_negative_gpu_memory_rejected(self):
        """Test that negative GPU memory is rejected."""
        with pytest.raises(ValueError, match="positive"):
            ResourceLimits(max_gpu_memory_gb=-1.0)

    def test_negative_cpu_memory_rejected(self):
        """Test that negative CPU memory is rejected."""
        with pytest.raises(ValueError, match="positive"):
            ResourceLimits(max_cpu_memory_gb=-1.0)

    def test_zero_models_rejected(self):
        """Test that zero max_models_loaded is rejected."""
        with pytest.raises(ValueError, match="positive"):
            ResourceLimits(max_models_loaded=0)

    def test_negative_batch_size_rejected(self):
        """Test that negative batch size is rejected."""
        with pytest.raises(ValueError, match="positive"):
            ResourceLimits(batch_size=-1)


class TestResourceManager:
    """Test ResourceManager."""

    def test_initialization(self):
        """Test resource manager initialization."""
        limits = ResourceLimits(max_models_loaded=2)
        manager = ResourceManager(limits)
        assert manager.limits.max_models_loaded == 2
        assert len(manager._loaded_models) == 0

    def test_default_limits(self):
        """Test resource manager with default limits."""
        manager = ResourceManager()
        assert manager.limits.max_models_loaded == 3

    def test_context_manager(self):
        """Test resource manager as context manager."""
        manager = ResourceManager()
        with manager as rm:
            assert rm is manager
            rm.register_model("test_model", {"dummy": "model"})
            assert "test_model" in rm._loaded_models

        # Models should be cleaned up after exit
        assert len(manager._loaded_models) == 0

    def test_select_device_cpu_fallback(self):
        """Test device selection falls back to CPU."""
        manager = ResourceManager()
        device = manager.select_device()
        # Should return one of the valid devices
        assert device in ["cuda", "mps", "cpu"]

    def test_select_device_prefers_cuda(self):
        """Test device selection prefers CUDA if available."""
        try:
            import torch

            if torch.cuda.is_available():
                manager = ResourceManager()
                device = manager.select_device()
                assert device == "cuda"
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_register_model(self):
        """Test model registration."""
        manager = ResourceManager()
        model = {"type": "sam2"}

        manager.register_model("sam2", model)
        assert "sam2" in manager._loaded_models
        assert manager.get_model("sam2") is model

    def test_register_model_enforces_limit(self):
        """Test that max_models_loaded limit is enforced."""
        limits = ResourceLimits(max_models_loaded=2)
        manager = ResourceManager(limits)

        manager.register_model("model1", {"id": 1})
        manager.register_model("model2", {"id": 2})
        assert len(manager._loaded_models) == 2

        # Registering 3rd model should unload oldest
        manager.register_model("model3", {"id": 3})
        assert len(manager._loaded_models) == 2
        assert "model1" not in manager._loaded_models  # Oldest unloaded
        assert "model2" in manager._loaded_models
        assert "model3" in manager._loaded_models

    def test_unload_model(self):
        """Test model unloading."""
        manager = ResourceManager()
        manager.register_model("sam2", {"type": "sam2"})
        assert "sam2" in manager._loaded_models

        manager.unload_model("sam2")
        assert "sam2" not in manager._loaded_models
        assert manager.get_model("sam2") is None

    def test_unload_nonexistent_model(self):
        """Test unloading a model that doesn't exist (no-op)."""
        manager = ResourceManager()
        manager.unload_model("nonexistent")  # Should not raise

    def test_get_model_returns_none_if_not_loaded(self):
        """Test get_model returns None for unloaded models."""
        manager = ResourceManager()
        assert manager.get_model("nonexistent") is None

    def test_get_memory_usage(self):
        """Test memory usage tracking."""
        manager = ResourceManager()
        memory_mb = manager.get_memory_usage_mb()
        assert memory_mb >= 0.0  # Should return non-negative

    def test_get_peak_memory(self):
        """Test peak memory tracking."""
        manager = ResourceManager()
        peak_mb = manager.get_peak_memory_mb()
        assert peak_mb >= 0.0

    def test_cleanup(self):
        """Test cleanup method."""
        manager = ResourceManager()
        manager.register_model("model1", {"id": 1})
        manager.register_model("model2", {"id": 2})
        assert len(manager._loaded_models) == 2

        manager.cleanup()
        assert len(manager._loaded_models) == 0

    def test_repr(self):
        """Test string representation."""
        manager = ResourceManager(ResourceLimits(max_models_loaded=3))
        manager.register_model("sam2", {"type": "sam2"})

        repr_str = repr(manager)
        assert "ResourceManager" in repr_str
        assert "models=1/3" in repr_str

    def test_fifo_model_unloading(self):
        """Test that models are unloaded in FIFO order."""
        limits = ResourceLimits(max_models_loaded=2)
        manager = ResourceManager(limits)

        # Load 3 models, should unload first one
        manager.register_model("model1", {"id": 1, "timestamp": 1})
        manager.register_model("model2", {"id": 2, "timestamp": 2})
        manager.register_model("model3", {"id": 3, "timestamp": 3})

        # model1 should be unloaded (oldest)
        assert "model1" not in manager._loaded_models
        assert "model2" in manager._loaded_models
        assert "model3" in manager._loaded_models

        # Load model4, should unload model2
        manager.register_model("model4", {"id": 4, "timestamp": 4})
        assert "model2" not in manager._loaded_models
        assert "model3" in manager._loaded_models
        assert "model4" in manager._loaded_models

    def test_gpu_cache_clearing(self):
        """Test GPU cache clearing on unload."""
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")

            manager = ResourceManager()
            manager.register_model("test", torch.randn(1000, 1000).cuda())

            # Unload should clear cache
            manager.unload_model("test")

            # Check that garbage collection ran
            # (Can't easily test cache clearing, but it shouldn't raise)
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_context_manager_cleanup_on_exception(self):
        """Test that cleanup happens even on exception."""
        manager = ResourceManager()

        try:
            with manager as rm:
                rm.register_model("test", {"id": 1})
                raise ValueError("Test error")
        except ValueError:
            pass

        # Cleanup should have happened
        assert len(manager._loaded_models) == 0
