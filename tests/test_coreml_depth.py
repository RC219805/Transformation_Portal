"""Tests for CoreML depth estimation module"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from depth_pipeline.coreml_exporter import (
    CoreMLExporter,
    CoreMLDepthEstimator
)


class TestCoreMLExporter:
    """Test CoreMLExporter class"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
            
    def test_exporter_initialization(self, temp_cache_dir):
        """Test exporter initialization"""
        exporter = CoreMLExporter(temp_cache_dir)
        assert exporter.cache_dir == temp_cache_dir
        assert exporter.cache_dir.exists()
        
    def test_default_cache_dir(self):
        """Test default cache directory"""
        exporter = CoreMLExporter()
        assert exporter.cache_dir == Path("weights/coreml")
        
    @patch('depth_pipeline.coreml_exporter.TORCH_AVAILABLE', False)
    @patch('depth_pipeline.coreml_exporter.COREML_AVAILABLE', False)
    def test_export_without_dependencies(self, temp_cache_dir):
        """Test export fails gracefully without dependencies"""
        exporter = CoreMLExporter(temp_cache_dir)
        result = exporter.export_depth_model()
        assert result is None
        
    def test_list_models_empty(self, temp_cache_dir):
        """Test listing models when none exist"""
        exporter = CoreMLExporter(temp_cache_dir)
        models = exporter.list_models()
        assert len(models) == 0
        
    def test_get_model_size(self, temp_cache_dir):
        """Test model size calculation"""
        exporter = CoreMLExporter(temp_cache_dir)
        
        test_file = temp_cache_dir / "test_model.mlpackage"
        test_file.mkdir()
        (test_file / "model.bin").write_bytes(b"0" * 1024 * 1024)
        
        size_mb = exporter._get_model_size(test_file)
        assert size_mb >= 1.0
        
    @patch('depth_pipeline.coreml_exporter.COREML_AVAILABLE', False)
    def test_benchmark_without_coreml(self, temp_cache_dir):
        """Test benchmark fails gracefully without CoreML"""
        exporter = CoreMLExporter(temp_cache_dir)
        result = exporter.benchmark_model(Path("dummy.mlpackage"))
        assert result == {}


class TestCoreMLDepthEstimator:
    """Test CoreMLDepthEstimator class"""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
            
    @patch('depth_pipeline.coreml_exporter.TORCH_AVAILABLE', False)
    @patch('depth_pipeline.coreml_exporter.COREML_AVAILABLE', False)
    def test_estimator_fails_without_dependencies(self, temp_cache_dir):
        """Test estimator fails gracefully without dependencies"""
        with pytest.raises(RuntimeError):
            CoreMLDepthEstimator(cache_dir=temp_cache_dir, prefer_coreml=False)
            
    @patch('depth_pipeline.coreml_exporter.TORCH_AVAILABLE', True)
    @patch('depth_pipeline.coreml_exporter.torch')
    @patch('depth_pipeline.coreml_exporter.COREML_AVAILABLE', False)
    def test_pytorch_fallback(self, mock_torch, temp_cache_dir):
        """Test PyTorch fallback when CoreML unavailable"""
        mock_model = MagicMock()
        
        with patch('transformers.AutoModel') as mock_auto_model:
            mock_auto_model.from_pretrained.return_value = mock_model
            
            estimator = CoreMLDepthEstimator(
                cache_dir=temp_cache_dir,
                prefer_coreml=True
            )
            
            assert not estimator.use_coreml
            assert estimator.model is not None
            
    def test_resize_depth(self, temp_cache_dir):
        """Test depth map resizing"""
        with patch('depth_pipeline.coreml_exporter.TORCH_AVAILABLE', True):
            with patch('depth_pipeline.coreml_exporter.torch'):
                with patch('transformers.AutoModel') as mock_auto_model:
                    mock_model = MagicMock()
                    mock_auto_model.from_pretrained.return_value = mock_model
                    
                    estimator = CoreMLDepthEstimator(
                        cache_dir=temp_cache_dir,
                        prefer_coreml=False
                    )
                    
                    depth = np.random.rand(100, 100).astype(np.float32)
                    resized = estimator._resize_depth(depth, (200, 200))
                    
                    assert resized.shape == (200, 200)
                    assert resized.min() >= 0.0
                    assert resized.max() <= 1.0


class TestDepthEstimation:
    """Test depth estimation functionality"""
    
    @pytest.fixture
    def mock_estimator(self):
        """Create mock estimator"""
        with patch('depth_pipeline.coreml_exporter.TORCH_AVAILABLE', True):
            with patch('depth_pipeline.coreml_exporter.torch'):
                with patch('transformers.AutoModel') as mock_auto_model:
                    mock_model = MagicMock()
                    mock_model.eval.return_value = None
                    
                    mock_output = MagicMock()
                    mock_depth = np.random.rand(518, 518).astype(np.float32)
                    mock_output.squeeze.return_value.cpu.return_value.numpy.return_value = mock_depth
                    mock_model.return_value = mock_output
                    
                    mock_auto_model.from_pretrained.return_value = mock_model
                    
                    yield CoreMLDepthEstimator(prefer_coreml=False)
                    
    def test_estimate_shape(self, mock_estimator):
        """Test depth estimation output shape"""
        image = np.random.randint(0, 255, (1024, 768, 3), dtype=np.uint8)
        
        depth = mock_estimator.estimate(image)
        
        assert depth.shape == (1024, 768)
        
    def test_estimate_range(self, mock_estimator):
        """Test depth values are normalized"""
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        
        depth = mock_estimator.estimate(image)
        
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0


class TestBenchmarking:
    """Test benchmarking functionality"""
    
    @pytest.fixture
    def mock_estimator(self):
        """Create mock estimator for benchmarking"""
        with patch('depth_pipeline.coreml_exporter.TORCH_AVAILABLE', True):
            with patch('depth_pipeline.coreml_exporter.torch'):
                with patch('transformers.AutoModel') as mock_auto_model:
                    mock_model = MagicMock()
                    mock_model.eval.return_value = None
                    
                    mock_output = MagicMock()
                    mock_depth = np.random.rand(518, 518).astype(np.float32)
                    mock_output.squeeze.return_value.cpu.return_value.numpy.return_value = mock_depth
                    mock_model.return_value = mock_output
                    
                    mock_auto_model.from_pretrained.return_value = mock_model
                    
                    yield CoreMLDepthEstimator(prefer_coreml=False)
                    
    def test_benchmark_results(self, mock_estimator):
        """Test benchmark returns valid results"""
        results = mock_estimator.benchmark(num_iterations=5)
        
        assert 'backend' in results
        assert 'model' in results
        assert 'mean_ms' in results
        assert 'std_ms' in results
        assert 'throughput_per_hour' in results
        assert results['iterations'] == 5
        
    def test_benchmark_metrics(self, mock_estimator):
        """Test benchmark metrics are reasonable"""
        results = mock_estimator.benchmark(num_iterations=5)
        
        assert results['mean_ms'] > 0
        assert results['std_ms'] >= 0
        assert results['min_ms'] > 0
        assert results['max_ms'] >= results['min_ms']
        assert results['throughput_per_hour'] > 0


class TestModelMapping:
    """Test model name mapping"""
    
    def test_model_id_mapping(self):
        """Test model ID mapping"""
        exporter = CoreMLExporter()
        
        model_names = [
            "depth_anything_v2_small",
            "depth_anything_v2_base",
            "depth_anything_v2_large"
        ]
        
        for name in model_names:
            with patch('transformers.AutoModel') as mock_auto_model:
                mock_model = MagicMock()
                mock_auto_model.from_pretrained.return_value = mock_model
                
                model = exporter._load_pytorch_model(name)
                assert model is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
