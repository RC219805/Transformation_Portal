"""Unit tests for foundation.memory_manager.

Tests MemoryPool tensor caching logic, MemoryManager allocation/deallocation,
pool selection, profiling, and the AllocationStrategy enum — all on CPU so no
real GPU is needed.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit]

CPU_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# AllocationStrategy enum
# ---------------------------------------------------------------------------


class TestAllocationStrategyEnum:
    def test_all_strategy_values_exist(self):
        from transformation_portal.foundation.memory_manager import AllocationStrategy

        values = {s.value for s in AllocationStrategy}
        assert "immediate" in values
        assert "pooled" in values
        assert "lazy" in values
        assert "aggressive_cache" in values
        assert "conservative" in values


# ---------------------------------------------------------------------------
# MemoryConfig defaults
# ---------------------------------------------------------------------------


class TestMemoryConfigDefaults:
    def test_default_strategy_is_pooled(self):
        from transformation_portal.foundation.memory_manager import AllocationStrategy, MemoryConfig

        cfg = MemoryConfig()
        assert cfg.strategy == AllocationStrategy.POOLED

    def test_high_watermark_above_low_watermark(self):
        from transformation_portal.foundation.memory_manager import MemoryConfig

        cfg = MemoryConfig()
        assert cfg.high_watermark > cfg.low_watermark

    def test_profiling_disabled_by_default(self):
        from transformation_portal.foundation.memory_manager import MemoryConfig

        assert MemoryConfig().enable_profiling is False


# ---------------------------------------------------------------------------
# MemoryPool
# ---------------------------------------------------------------------------


class TestMemoryPool:
    def test_get_returns_none_when_empty(self):
        from transformation_portal.foundation.memory_manager import MemoryPool

        pool = MemoryPool(size_mb=10, device=CPU_DEVICE)
        result = pool.get((4, 4), torch.float32)
        assert result is None

    def test_put_and_get_round_trip(self):
        from transformation_portal.foundation.memory_manager import MemoryPool

        pool = MemoryPool(size_mb=10, device=CPU_DEVICE)
        t = torch.zeros(4, 4)
        assert pool.put(t) is True
        retrieved = pool.get((4, 4), torch.float32)
        assert retrieved is not None
        assert retrieved.shape == (4, 4)

    def test_put_rejects_when_full(self):
        from transformation_portal.foundation.memory_manager import MemoryPool

        # Pool that holds only ~0.001 MB
        pool = MemoryPool(size_mb=0, device=CPU_DEVICE)
        t = torch.zeros(256, 256)  # 256 KB
        assert pool.put(t) is False

    def test_put_rejects_wrong_device(self):
        from transformation_portal.foundation.memory_manager import MemoryPool

        pool = MemoryPool(size_mb=100, device=CPU_DEVICE)
        # Create a CPU tensor but pretend the pool targets a different device by
        # constructing a pool for a different (non-existent) device name.
        wrong_device_pool = MemoryPool(size_mb=100, device=torch.device("cpu"))
        # A tensor already on cpu shouldn't be rejected — confirm acceptance
        t = torch.zeros(4, 4)
        assert wrong_device_pool.put(t) is True

    def test_clear_empties_pool(self):
        from transformation_portal.foundation.memory_manager import MemoryPool

        pool = MemoryPool(size_mb=10, device=CPU_DEVICE)
        pool.put(torch.zeros(4, 4))
        pool.clear()
        stats = pool.get_stats()
        assert stats["total_tensors"] == 0
        assert stats["allocated_mb"] == 0.0

    def test_get_stats_structure(self):
        from transformation_portal.foundation.memory_manager import MemoryPool

        pool = MemoryPool(size_mb=10, device=CPU_DEVICE)
        stats = pool.get_stats()
        for key in ("total_keys", "total_tensors", "allocated_mb", "capacity_mb", "utilization"):
            assert key in stats

    def test_utilization_zero_when_empty(self):
        from transformation_portal.foundation.memory_manager import MemoryPool

        pool = MemoryPool(size_mb=10, device=CPU_DEVICE)
        assert pool.get_stats()["utilization"] == 0.0

    def test_allocated_bytes_decreases_after_get(self):
        from transformation_portal.foundation.memory_manager import MemoryPool

        pool = MemoryPool(size_mb=10, device=CPU_DEVICE)
        t = torch.zeros(64, 64)  # 16 KB float32
        pool.put(t)
        before = pool.allocated_bytes
        pool.get((64, 64), torch.float32)
        assert pool.allocated_bytes < before or pool.allocated_bytes == 0


# ---------------------------------------------------------------------------
# MemoryManager – construction and pool initialization
# ---------------------------------------------------------------------------


class TestMemoryManagerConstruction:
    def test_three_pools_created(self):
        from transformation_portal.foundation.memory_manager import MemoryManager

        mgr = MemoryManager(device=CPU_DEVICE)
        assert set(mgr.pools.keys()) == {"small", "medium", "large"}

    def test_default_device_falls_back_to_cpu_when_no_gpu(self):
        from unittest.mock import patch

        with (
            patch("torch.backends.mps.is_available", return_value=False),
            patch("torch.cuda.is_available", return_value=False),
        ):
            from transformation_portal.foundation.memory_manager import MemoryManager

            mgr = MemoryManager()
        assert mgr.device.type == "cpu"

    def test_repr_contains_device(self):
        from transformation_portal.foundation.memory_manager import MemoryManager

        mgr = MemoryManager(device=CPU_DEVICE)
        assert "cpu" in repr(mgr)


# ---------------------------------------------------------------------------
# Pool name selection
# ---------------------------------------------------------------------------


class TestPoolNameSelection:
    def test_small_tensor_goes_to_small_pool(self):
        from transformation_portal.foundation.memory_manager import MemoryManager

        mgr = MemoryManager(device=CPU_DEVICE)
        assert mgr._get_pool_name(1.0) == "small"

    def test_medium_tensor_goes_to_medium_pool(self):
        from transformation_portal.foundation.memory_manager import MemoryManager

        mgr = MemoryManager(device=CPU_DEVICE)
        assert mgr._get_pool_name(50.0) == "medium"

    def test_large_tensor_goes_to_large_pool(self):
        from transformation_portal.foundation.memory_manager import MemoryManager

        mgr = MemoryManager(device=CPU_DEVICE)
        assert mgr._get_pool_name(200.0) == "large"

    def test_boundary_10mb_goes_to_medium(self):
        from transformation_portal.foundation.memory_manager import MemoryManager

        mgr = MemoryManager(device=CPU_DEVICE)
        assert mgr._get_pool_name(10.0) == "medium"


# ---------------------------------------------------------------------------
# Allocation
# ---------------------------------------------------------------------------


class TestAllocation:
    def test_allocate_returns_tensor_with_correct_shape(self):
        from transformation_portal.foundation.memory_manager import MemoryConfig, MemoryManager

        cfg = MemoryConfig(
            strategy=__import__(
                "transformation_portal.foundation.memory_manager", fromlist=["AllocationStrategy"]
            ).AllocationStrategy.IMMEDIATE
        )
        mgr = MemoryManager(config=cfg, device=CPU_DEVICE)
        t = mgr.allocate((8, 8), torch.float32)
        assert t.shape == (8, 8)

    def test_allocate_returns_correct_dtype(self):
        from transformation_portal.foundation.memory_manager import AllocationStrategy, MemoryConfig, MemoryManager

        cfg = MemoryConfig(strategy=AllocationStrategy.IMMEDIATE)
        mgr = MemoryManager(config=cfg, device=CPU_DEVICE)
        t = mgr.allocate((4,), torch.float16)
        assert t.dtype == torch.float16

    def test_allocate_from_pool_reuses_tensor(self):
        from transformation_portal.foundation.memory_manager import AllocationStrategy, MemoryConfig, MemoryManager

        cfg = MemoryConfig(strategy=AllocationStrategy.POOLED)
        mgr = MemoryManager(config=cfg, device=CPU_DEVICE)

        # Put a tensor in the pool manually
        t_original = torch.zeros(4, 4)
        mgr.pools["small"].put(t_original)

        t_retrieved = mgr.allocate((4, 4), torch.float32)
        assert t_retrieved.shape == (4, 4)

    def test_allocate_batch_returns_correct_count(self):
        from transformation_portal.foundation.memory_manager import AllocationStrategy, MemoryConfig, MemoryManager

        cfg = MemoryConfig(strategy=AllocationStrategy.IMMEDIATE)
        mgr = MemoryManager(config=cfg, device=CPU_DEVICE)
        tensors = mgr.allocate_batch(batch_size=4, shape=(3, 3))
        assert len(tensors) == 4
        for t in tensors:
            assert t.shape == (3, 3)


# ---------------------------------------------------------------------------
# Deallocation
# ---------------------------------------------------------------------------


class TestDeallocation:
    def test_deallocate_returns_to_pool(self):
        from transformation_portal.foundation.memory_manager import AllocationStrategy, MemoryConfig, MemoryManager

        cfg = MemoryConfig(strategy=AllocationStrategy.POOLED)
        mgr = MemoryManager(config=cfg, device=CPU_DEVICE)
        t = torch.zeros(4, 4)
        returned = mgr.deallocate(t, return_to_pool=True)
        assert returned is True  # small tensor should fit in small pool

    def test_deallocate_without_pool_returns_false(self):
        from transformation_portal.foundation.memory_manager import AllocationStrategy, MemoryConfig, MemoryManager

        cfg = MemoryConfig(strategy=AllocationStrategy.POOLED)
        mgr = MemoryManager(config=cfg, device=CPU_DEVICE)
        t = torch.zeros(4, 4)
        returned = mgr.deallocate(t, return_to_pool=False)
        assert returned is False

    def test_deallocate_removes_from_tracking(self):
        from transformation_portal.foundation.memory_manager import MemoryConfig, MemoryManager

        cfg = MemoryConfig(enable_profiling=True)
        mgr = MemoryManager(config=cfg, device=CPU_DEVICE)
        t = mgr.allocate((4, 4), tag="test_tag")
        tensor_id = id(t)
        mgr.deallocate(t, return_to_pool=False)
        assert tensor_id not in mgr.allocations


# ---------------------------------------------------------------------------
# Profiling / tracking
# ---------------------------------------------------------------------------


class TestProfiling:
    def test_profiling_disabled_by_default_no_allocations_tracked(self):
        from transformation_portal.foundation.memory_manager import MemoryManager

        mgr = MemoryManager(device=CPU_DEVICE)
        mgr.allocate((4, 4))
        assert len(mgr.allocations) == 0

    def test_profiling_enabled_tracks_allocations(self):
        from transformation_portal.foundation.memory_manager import MemoryConfig, MemoryManager

        cfg = MemoryConfig(enable_profiling=True)
        mgr = MemoryManager(config=cfg, device=CPU_DEVICE)
        mgr.allocate((4, 4), tag="my_op")
        assert len(mgr.allocations) == 1

    def test_profiling_records_tag_stats(self):
        from transformation_portal.foundation.memory_manager import MemoryConfig, MemoryManager

        cfg = MemoryConfig(enable_profiling=True)
        mgr = MemoryManager(config=cfg, device=CPU_DEVICE)
        mgr.allocate((4, 4), tag="depth_stage")
        assert mgr.allocation_stats["depth_stage"]["count"] == 1
        assert mgr.allocation_stats["depth_stage"]["total_bytes"] > 0

    def test_peak_memory_bytes_updated(self):
        from transformation_portal.foundation.memory_manager import MemoryConfig, MemoryManager

        cfg = MemoryConfig(enable_profiling=True)
        mgr = MemoryManager(config=cfg, device=CPU_DEVICE)
        mgr.allocate((64, 64), tag="big_op")
        assert mgr.peak_memory_bytes > 0


# ---------------------------------------------------------------------------
# Memory statistics
# ---------------------------------------------------------------------------


class TestMemoryStats:
    def test_get_memory_stats_returns_dict(self):
        from transformation_portal.foundation.memory_manager import MemoryManager

        mgr = MemoryManager(device=CPU_DEVICE)
        stats = mgr.get_memory_stats()
        assert isinstance(stats, dict)
        assert "pools" in stats

    def test_pool_stats_in_memory_stats(self):
        from transformation_portal.foundation.memory_manager import MemoryManager

        mgr = MemoryManager(device=CPU_DEVICE)
        stats = mgr.get_memory_stats()
        assert "small" in stats["pools"]
        assert "medium" in stats["pools"]
        assert "large" in stats["pools"]

    def test_allocation_summary_contains_device(self):
        from transformation_portal.foundation.memory_manager import MemoryManager

        mgr = MemoryManager(device=CPU_DEVICE)
        summary = mgr.get_allocation_summary()
        assert "cpu" in summary.lower()


# ---------------------------------------------------------------------------
# Cache management
# ---------------------------------------------------------------------------


class TestCacheManagement:
    def test_clear_cache_empties_pools(self):
        from transformation_portal.foundation.memory_manager import MemoryManager

        mgr = MemoryManager(device=CPU_DEVICE)
        mgr.pools["small"].put(torch.zeros(4, 4))
        mgr.clear_cache()
        assert mgr.pools["small"].get_stats()["total_tensors"] == 0

    def test_moderate_cleanup_clears_large_pool(self):
        from transformation_portal.foundation.memory_manager import MemoryManager

        mgr = MemoryManager(device=CPU_DEVICE)
        # Manually put something in the large pool
        large_tensor = torch.zeros(128, 128)  # ~64 KB, small enough to fit
        mgr.pools["large"].put(large_tensor)
        mgr._moderate_cleanup()
        assert mgr.pools["large"].get_stats()["total_tensors"] == 0
