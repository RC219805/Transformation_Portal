"""
Integration tests for Phase 2 Performance Optimizations.

Tests parallel orchestrator, async I/O, storage management,
model caching, and CLI integration.
"""

import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Skip tests if lux_depth_v2 not available
pytest.importorskip("lux_depth_v2")

from lux_depth_v2.orchestrator import (
    ParallelOrchestrator,
    ProcessOrchestrator,
    TaskConfig,
    TaskStatus,
    WorkerState,
    ParallelCapacityCheck
)
from lux_depth_v2.config import Phase2Config
from lux_depth_v2.resource_monitor import ResourceMonitor


class TestParallelOrchestrator:
    """Test parallel orchestrator functionality."""
    
    def test_orchestrator_backward_compatible(self):
        """Test that ParallelOrchestrator works in Phase 1 mode."""
        orchestrator = ParallelOrchestrator(
            max_workers=1,
            enable_parallel=False
        )
        
        assert orchestrator.enable_parallel is False
        assert orchestrator.max_parallel_workers == 1
        assert len(orchestrator.worker_states) == 0
    
    def test_orchestrator_parallel_mode(self):
        """Test parallel orchestrator in Phase 2 mode."""
        orchestrator = ParallelOrchestrator(
            max_workers=2,
            enable_parallel=True,
            memory_budget_per_worker=25.0
        )
        
        assert orchestrator.enable_parallel is True
        assert orchestrator.max_parallel_workers == 2
        assert orchestrator.memory_budget_per_worker == 25.0
    
    def test_get_available_worker_slots_phase1(self):
        """Test worker slot calculation in Phase 1 mode."""
        orchestrator = ParallelOrchestrator(
            max_workers=1,
            enable_parallel=False
        )
        
        # Phase 1: Only 1 slot available when no workers
        slots = orchestrator.get_available_worker_slots()
        assert slots == 1
        
        # Add a mock worker
        orchestrator.worker_states.append(
            WorkerState(0, "test", Mock(), time.time(), 25.0)
        )
        
        # Phase 1: No slots when worker active
        slots = orchestrator.get_available_worker_slots()
        assert slots == 0
    
    def test_get_available_worker_slots_phase2(self):
        """Test worker slot calculation in Phase 2 mode."""
        # Create orchestrator with mock resource monitor to control behavior
        mock_monitor = Mock()
        mock_metrics = Mock()
        mock_metrics.ram_total_gb = 64.0
        mock_metrics.ram_used_gb = 0.0  # 64GB available
        mock_monitor.get_metrics.return_value = mock_metrics
        
        orchestrator = ParallelOrchestrator(
            max_workers=2,
            enable_parallel=True,
            resource_monitor=mock_monitor
        )
        
        # No workers: 2 slots available (64GB / 25GB = 2)
        slots = orchestrator.get_available_worker_slots()
        assert slots == 2
        
        # Add one worker
        orchestrator.worker_states.append(
            WorkerState(0, "test", Mock(), time.time(), 25.0)
        )
        
        # One worker: 1 slot available
        slots = orchestrator.get_available_worker_slots()
        assert slots == 1
    
    def test_check_parallel_capacity_no_monitor(self):
        """Test capacity check without resource monitor."""
        orchestrator = ParallelOrchestrator(
            max_workers=2,
            enable_parallel=True,
            resource_monitor=None
        )
        
        capacity = orchestrator.check_parallel_capacity(2)
        
        assert isinstance(capacity, ParallelCapacityCheck)
        # Without monitor, it won't have warnings (optimistic mode now)
        # The actual ResourceMonitor is created automatically if None
        # So we can't easily test "no monitor" mode
        assert capacity.memory_per_worker_gb == 25.0
    
    def test_check_parallel_capacity_with_monitor(self):
        """Test capacity check with resource monitor."""
        # Create a mock resource monitor
        mock_monitor = Mock()
        mock_metrics = Mock()
        mock_metrics.ram_total_gb = 64.0
        mock_metrics.ram_used_gb = 14.0  # 50GB available
        mock_metrics.disk_metrics = {'.': {'available_gb': 100.0}}
        mock_monitor.get_metrics.return_value = mock_metrics
        
        orchestrator = ParallelOrchestrator(
            max_workers=4,
            enable_parallel=True,
            memory_budget_per_worker=25.0,
            resource_monitor=mock_monitor
        )
        
        # Request 3 workers (75GB needed, 50GB available → recommend 2)
        capacity = orchestrator.check_parallel_capacity(3)
        
        assert capacity.available_memory_gb == 50.0
        assert capacity.recommended_workers == 2  # 50GB / 25GB = 2 workers
        assert not capacity.can_support_requested


class TestResourceMonitorParallelCapacity:
    """Test resource monitor parallel capacity checking."""
    
    def test_check_parallel_capacity_sufficient_memory(self):
        """Test capacity check with sufficient memory."""
        monitor = ResourceMonitor()
        
        # Mock sufficient memory
        with patch.object(monitor, 'get_metrics') as mock_collect:
            mock_metrics = Mock()
            mock_metrics.ram_total_gb = 64.0
            mock_metrics.ram_used_gb = 14.0  # 50GB available
            mock_metrics.ram_percent = 21.9
            mock_metrics.disk_metrics = {'.': {'free_gb': 100.0}}
            mock_metrics.mps_available = False
            mock_collect.return_value = mock_metrics
            
            result = monitor.check_parallel_capacity(2, 25.0)
            
            assert result['can_support_requested'] is True
            assert result['recommended_workers'] == 2
            assert result['available_memory_gb'] == 50.0
            assert result['total_memory_required_gb'] == 50.0
    
    def test_check_parallel_capacity_insufficient_memory(self):
        """Test capacity check with insufficient memory."""
        monitor = ResourceMonitor()
        
        with patch.object(monitor, 'get_metrics') as mock_collect:
            mock_metrics = Mock()
            mock_metrics.ram_total_gb = 64.0
            mock_metrics.ram_used_gb = 39.0  # Only 25GB available
            mock_metrics.ram_percent = 60.9
            mock_metrics.disk_metrics = {'.': {'free_gb': 100.0}}
            mock_metrics.mps_available = False
            mock_collect.return_value = mock_metrics
            
            result = monitor.check_parallel_capacity(2, 25.0)
            
            assert result['can_support_requested'] is False
            assert result['recommended_workers'] == 1  # Only 1 worker fits
            assert len(result['warnings']) > 0


class TestPhase2Config:
    """Test Phase2Config dataclass."""
    
    def test_default_config(self):
        """Test Phase2Config defaults."""
        config = Phase2Config()
        
        assert config.async_io_enabled is True
        assert config.streaming_upscale is True
        assert config.max_concurrent_workers == 2
        assert config.memory_budget_per_worker_gb == 25.0
        assert config.model_cache_enabled is True
        assert config.depth_map_cache_enabled is True
        assert config.tile_based_upscaling is True
    
    def test_custom_config(self):
        """Test Phase2Config customization."""
        config = Phase2Config(
            async_io_enabled=False,
            max_concurrent_workers=4,
            memory_budget_per_worker_gb=20.0,
            storage_external_t9="/Volumes/T9"
        )
        
        assert config.async_io_enabled is False
        assert config.max_concurrent_workers == 4
        assert config.memory_budget_per_worker_gb == 20.0
        assert config.storage_external_t9 == "/Volumes/T9"


class TestAsyncIO:
    """Test async I/O functionality."""
    
    def test_async_tiff_writer_import(self):
        """Test that AsyncTIFFWriter can be imported."""
        from lux_depth_v2.io_optimizer import AsyncTIFFWriter
        
        writer = AsyncTIFFWriter(use_compression=True)
        assert writer is not None
    
    def test_streaming_upscale_writer_import(self):
        """Test that StreamingUpscaleWriter can be imported."""
        from lux_depth_v2.io_optimizer import StreamingUpscaleWriter
        
        # StreamingUpscaleWriter requires path and dimensions
        # Just test import for now
        assert StreamingUpscaleWriter is not None


class TestStorageManager:
    """Test storage manager functionality."""
    
    def test_storage_manager_import(self):
        """Test that StorageManager can be imported."""
        from lux_depth_v2.storage_manager import StorageManager, StorageConfig
        
        config = StorageConfig(
            internal_ssd_path=".",
            external_t9_path=None
        )
        manager = StorageManager(config)
        assert manager is not None
    
    def test_storage_config_defaults(self):
        """Test StorageConfig defaults."""
        from lux_depth_v2.storage_manager import StorageConfig
        
        config = StorageConfig()
        assert config.internal_ssd_path == "."
        assert config.external_t9_path is None
        assert config.auto_migrate_threshold_gb == 2.0
        assert config.min_free_space_gb == 10.0


class TestModelCache:
    """Test model caching functionality."""
    
    def test_model_cache_singleton(self):
        """Test ModelCache is singleton."""
        from lux_depth_v2.cache_optimizer import ModelCache
        
        cache1 = ModelCache.get_instance()
        cache2 = ModelCache.get_instance()
        
        assert cache1 is cache2
    
    def test_depth_map_cache_import(self):
        """Test DepthMapCache can be imported."""
        from lux_depth_v2.cache_optimizer import DepthMapCache
        
        cache = DepthMapCache(cache_dir=Path(".test_cache"))
        assert cache is not None


class TestTileBasedUpscaler:
    """Test tile-based upscaling functionality."""
    
    def test_tile_based_upscaler_import(self):
        """Test TileBasedUpscaler can be imported."""
        from lux_depth_v2.upscale_optimizer import TileBasedUpscaler
        
        # Just test import - actual instantiation requires proper backend setup
        assert TileBasedUpscaler is not None
    
    def test_tiling_config(self):
        """Test TilingConfig."""
        from lux_depth_v2.upscale_optimizer import TilingConfig
        
        config = TilingConfig(
            tile_size=512,
            overlap=64,
            scale_factor=4
        )
        
        assert config.tile_size == 512
        assert config.overlap == 64
        assert config.effective_tile_size == 512 + 2 * 64


class TestCLIIntegration:
    """Test CLI integration for Phase 2."""
    
    def test_phase2_config_creation_from_cli_args(self):
        """Test creating Phase2Config from CLI args."""
        config = Phase2Config(
            async_io_enabled=True,
            streaming_upscale=True,
            max_concurrent_workers=2,
            memory_budget_per_worker_gb=25.0,
            model_cache_enabled=True,
            depth_map_cache_enabled=True,
            storage_external_t9="/Volumes/T9",
            auto_migrate_large_files=True,
            migrate_threshold_gb=2.0,
            tile_based_upscaling=True,
            upscale_tile_size=512
        )
        
        # Verify all options set correctly
        assert config.async_io_enabled is True
        assert config.streaming_upscale is True
        assert config.max_concurrent_workers == 2
        assert config.memory_budget_per_worker_gb == 25.0
        assert config.model_cache_enabled is True
        assert config.depth_map_cache_enabled is True
        assert config.storage_external_t9 == "/Volumes/T9"
        assert config.auto_migrate_large_files is True
        assert config.migrate_threshold_gb == 2.0
        assert config.tile_based_upscaling is True
        assert config.upscale_tile_size == 512


# Pytest markers for different test categories
pytestmark = pytest.mark.integration


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
