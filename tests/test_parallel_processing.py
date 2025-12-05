"""Tests for parallel processing module"""

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from utils.parallel_processor import (
    ParallelProcessor,
    ProcessingMode,
    WorkerConfig,
    process_images_parallel
)


def _cpu_test_process_fn(x):
    """Test function for CPU parallel processing (must be module-level for pickling)"""
    time.sleep(0.01)
    return x ** 2


def _image_process_fn(path):
    """Test function for image parallel processing (must be module-level)"""
    return str(path)


class TestWorkerConfig:
    """Test WorkerConfig dataclass"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = WorkerConfig()
        assert config.num_workers == -1
        assert config.memory_limit_gb == 8.0
        assert config.mode == ProcessingMode.AUTO
        assert config.gpu_ids is None
        assert config.batch_size == 1
        
    def test_custom_config(self):
        """Test custom configuration"""
        config = WorkerConfig(
            num_workers=4,
            memory_limit_gb=16.0,
            mode=ProcessingMode.MULTI_GPU,
            gpu_ids=[0, 1]
        )
        assert config.num_workers == 4
        assert config.memory_limit_gb == 16.0
        assert config.mode == ProcessingMode.MULTI_GPU
        assert config.gpu_ids == [0, 1]


class TestParallelProcessor:
    """Test ParallelProcessor class"""
    
    def test_hardware_detection(self):
        """Test hardware detection"""
        processor = ParallelProcessor()
        assert processor.num_cpus > 0
        assert processor.available_memory_gb > 0
        
    def test_cpu_mode_configuration(self):
        """Test CPU mode configuration"""
        config = WorkerConfig(mode=ProcessingMode.MULTI_CPU, num_workers=4)
        processor = ParallelProcessor(config)
        assert processor.mode == ProcessingMode.MULTI_CPU
        assert processor.num_workers == 4
        
    def test_sequential_processing(self):
        """Test sequential processing"""
        config = WorkerConfig(mode=ProcessingMode.SINGLE_THREADED)
        processor = ParallelProcessor(config)
        
        items = [1, 2, 3, 4, 5]
        
        def process_fn(x):
            return x * 2
            
        results = processor.process_batch(items, process_fn)
        
        assert len(results) == 5
        for idx, (result, error) in enumerate(results):
            assert error is None
            assert result == items[idx] * 2
            
        assert processor.stats.completed_tasks == 5
        assert processor.stats.failed_tasks == 0
        
    def test_cpu_parallel_processing(self):
        """Test CPU parallel processing"""
        config = WorkerConfig(mode=ProcessingMode.MULTI_CPU, num_workers=2)
        processor = ParallelProcessor(config)
        
        items = list(range(10))
        
        results = processor.process_batch(items, _cpu_test_process_fn)
        
        assert len(results) == 10
        for idx, (result, error) in enumerate(results):
            if error is not None:
                print(f"Error for item {idx}: {error}")
            assert result == items[idx] ** 2
            
    def test_error_handling(self):
        """Test error handling in processing"""
        config = WorkerConfig(mode=ProcessingMode.SINGLE_THREADED)
        processor = ParallelProcessor(config)
        
        items = [1, 2, 3, 4, 5]
        
        def process_fn(x):
            if x == 3:
                raise ValueError(f"Error processing {x}")
            return x * 2
            
        results = processor.process_batch(items, process_fn)
        
        assert len(results) == 5
        assert processor.stats.completed_tasks == 4
        assert processor.stats.failed_tasks == 1
        
        for idx, (result, error) in enumerate(results):
            if items[idx] == 3:
                assert error is not None
                assert result is None
            else:
                assert error is None
                assert result == items[idx] * 2
                
    def test_progress_callback(self):
        """Test progress callback"""
        config = WorkerConfig(mode=ProcessingMode.SINGLE_THREADED)
        processor = ParallelProcessor(config)
        
        items = [1, 2, 3]
        progress_calls = []
        
        def progress_callback(completed, total):
            progress_calls.append((completed, total))
            
        def process_fn(x):
            return x * 2
            
        processor.process_batch(items, process_fn, progress_callback)
        
        assert len(progress_calls) == 3
        assert progress_calls[0] == (1, 3)
        assert progress_calls[1] == (2, 3)
        assert progress_calls[2] == (3, 3)
        
    def test_empty_batch(self):
        """Test processing empty batch"""
        processor = ParallelProcessor()
        results = processor.process_batch([], lambda x: x)
        assert results == []
        
    def test_statistics(self):
        """Test statistics collection"""
        config = WorkerConfig(mode=ProcessingMode.SINGLE_THREADED)
        processor = ParallelProcessor(config)
        
        items = list(range(10))
        processor.process_batch(items, lambda x: x * 2)
        
        stats = processor.get_stats()
        assert stats.total_tasks == 10
        assert stats.completed_tasks == 10
        assert stats.failed_tasks == 0
        assert stats.total_time_seconds > 0
        assert stats.throughput_per_hour > 0
        
    @patch('utils.parallel_processor.TORCH_AVAILABLE', True)
    @patch('utils.parallel_processor.torch')
    def test_gpu_detection(self, mock_torch):
        """Test GPU detection"""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 2
        
        config = WorkerConfig(mode=ProcessingMode.AUTO)
        processor = ParallelProcessor(config)
        
        assert processor.gpu_available
        assert processor.num_gpus == 2


class TestConvenienceFunction:
    """Test convenience function"""
    
    def test_process_images_parallel(self):
        """Test convenience function for image processing"""
        image_paths = [Path(f"image_{i}.jpg") for i in range(5)]
            
        results = process_images_parallel(
            image_paths,
            _image_process_fn,
            num_workers=2,
            use_gpu=False,
            progress=False
        )
        
        assert len(results) == 5
        for (result, error), path in zip(results, image_paths):
            if error is not None:
                print(f"Error for {path}: {error}")
            assert result == str(path)


class TestMemoryAwareness:
    """Test memory-aware scheduling"""
    
    def test_memory_limit_configuration(self):
        """Test memory limit configuration"""
        config = WorkerConfig(memory_limit_gb=4.0)
        assert config.memory_limit_gb == 4.0
        
    def test_available_memory_detection(self):
        """Test available memory detection"""
        processor = ParallelProcessor()
        assert processor.available_memory_gb > 0


class TestGPULoadBalancing:
    """Test GPU load balancing"""
    
    def test_gpu_id_assignment(self):
        """Test GPU ID assignment"""
        config = WorkerConfig(
            mode=ProcessingMode.MULTI_GPU,
            gpu_ids=[0, 1, 2],
            num_workers=6
        )
        processor = ParallelProcessor(config)
        assert processor.gpu_ids == [0, 1, 2]
        
    def test_auto_gpu_ids(self):
        """Test automatic GPU ID assignment"""
        with patch('utils.parallel_processor.TORCH_AVAILABLE', True):
            with patch('utils.parallel_processor.torch') as mock_torch:
                mock_torch.cuda.is_available.return_value = True
                mock_torch.cuda.device_count.return_value = 4
                
                config = WorkerConfig(mode=ProcessingMode.AUTO)
                processor = ParallelProcessor(config)
                
                assert processor.gpu_ids == [0, 1, 2, 3]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
