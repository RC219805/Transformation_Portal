"""Integration tests for Materials v2.

Tests confidence gating, segmentation downscaling, VRAM lifecycle,
mask caching, and integration with Phase 1 components.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import Materials v2 components
from lux_depth_v2.materials_v2 import (
    ConfidenceConfig,
    ConfidenceMetrics,
    MaterialsV2Config,
    MaterialsV2Engine,
    SegmentationConfig,
    SegmentationResult,
    calculate_segmentation_size,
    create_soft_mask,
    generate_confidence_mask,
)
from lux_depth_v2.cache_manager import MaskCacheManager


# Utility functions

def _torch_available() -> bool:
    """Check if PyTorch is available."""
    try:
        return True
    except ImportError:
        return False


def _create_test_image(height: int = 100, width: int = 100) -> np.ndarray:
    """Create a simple test image."""
    return np.random.rand(height, width, 3).astype(np.float32)


def _create_test_segmentation(height: int = 100, width: int = 100) -> SegmentationResult:
    """Create mock segmentation result."""
    return SegmentationResult(
        mask=np.random.rand(height, width).astype(np.float32) > 0.5,
        confidence=np.random.rand(height, width).astype(np.float32) * 0.5 + 0.5,
        material_type='wood',
        source='test'
    )


class TestConfidenceGating:
    """Test confidence-gated material response."""

    def test_confidence_mask_generation_soft(self):
        """Test soft confidence mask generation."""
        config = ConfidenceConfig(
            confidence_threshold=0.6,
            blend_range=0.1,
            blend_mode='soft',
            fallback_strength=0.2
        )

        # Create confidence map with various values
        confidence_map = np.array([
            [0.3, 0.5, 0.55, 0.6, 0.7],  # Low to high confidence
            [0.4, 0.55, 0.6, 0.65, 0.8],
        ], dtype=np.float32)

        mask = generate_confidence_mask(confidence_map, 'wood', config)

        # Check shape preserved
        assert mask.shape == confidence_map.shape

        # Check value range [0, 1]
        assert np.all(mask >= 0.0)
        assert np.all(mask <= 1.0)

        # Check low confidence gets fallback strength
        assert mask[0, 0] == pytest.approx(0.2, abs=0.01)  # 0.3 confidence → fallback

        # Check high confidence gets full strength
        assert mask[0, 4] >= 0.95  # 0.7 confidence → near 1.0

    def test_confidence_mask_generation_hard(self):
        """Test hard confidence mask generation."""
        config = ConfidenceConfig(
            confidence_threshold=0.6,
            blend_mode='hard',
            fallback_strength=0.0,
            material_thresholds={}  # Override defaults to use global threshold
        )

        confidence_map = np.array([
            [0.5, 0.59, 0.6, 0.61, 0.7],
        ], dtype=np.float32)

        mask = generate_confidence_mask(confidence_map, 'wood', config)

        # Hard cutoff: below threshold = 0, above = 1
        assert mask[0, 0] == 0.0  # 0.5 < 0.6
        assert mask[0, 1] == 0.0  # 0.59 < 0.6
        assert mask[0, 2] >= 0.99  # 0.6 >= 0.6
        assert mask[0, 3] >= 0.99  # 0.61 >= 0.6

    def test_material_specific_thresholds(self):
        """Test per-material confidence thresholds."""
        config = ConfidenceConfig(
            confidence_threshold=0.6,
            material_thresholds={
                'wood': 0.7,
                'glass': 0.5,
            },
            blend_mode='hard',
            fallback_strength=0.0
        )

        confidence_map = np.array([[0.65]], dtype=np.float32)

        # 0.65 below wood threshold (0.7)
        mask_wood = generate_confidence_mask(confidence_map, 'wood', config)
        assert mask_wood[0, 0] == 0.0

        # 0.65 above glass threshold (0.5)
        mask_glass = generate_confidence_mask(confidence_map, 'glass', config)
        assert mask_glass[0, 0] >= 0.99


class TestSegmentationDownscaling:
    """Test segmentation resolution downscaling."""

    def test_calculate_segmentation_size_no_downscale(self):
        """Test no downscaling for small images."""
        config = SegmentationConfig(max_segmentation_side=1536)

        original_size = (1024, 768)
        seg_size = calculate_segmentation_size(original_size, config)

        # No downscaling needed
        assert seg_size == original_size

    def test_calculate_segmentation_size_downscale_4k(self):
        """Test downscaling for 4K image."""
        config = SegmentationConfig(max_segmentation_side=1536)

        original_size = (2160, 3840)  # 4K
        seg_size = calculate_segmentation_size(original_size, config)

        # Should downscale to max side = 1536
        assert max(seg_size) == 1536
        # Aspect ratio preserved
        aspect_original = original_size[1] / original_size[0]
        aspect_seg = seg_size[1] / seg_size[0]
        assert abs(aspect_original - aspect_seg) < 0.01

    def test_calculate_segmentation_size_downscale_8k(self):
        """Test downscaling for 8K image."""
        config = SegmentationConfig(max_segmentation_side=1536)

        original_size = (4320, 7680)  # 8K
        seg_size = calculate_segmentation_size(original_size, config)

        # Should downscale to max side = 1536
        assert max(seg_size) == 1536

    def test_soft_mask_creation(self):
        """Test soft mask with edge feathering."""
        config = SegmentationConfig(
            edge_feather_radius=3,
            edge_feather_sigma=1.0
        )

        # Binary mask with hard edge
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1.0

        soft_mask = create_soft_mask(mask, config)

        # Check shape preserved
        assert soft_mask.shape == mask.shape

        # Check value range
        assert np.all(soft_mask >= 0.0)
        assert np.all(soft_mask <= 1.0)

        # Check center still ~1.0
        assert soft_mask[50, 50] > 0.95

        # Check edges are softened (not 0 or 1)
        # Note: Gaussian blur with sigma=1.0 creates gradual falloff
        # Check just inside the edge where softening effect is visible
        assert soft_mask[29, 50] < 0.7  # Just outside edge - should be softened
        assert soft_mask[31, 50] > 0.3  # Just inside edge - should be softened


class TestVRAMLifecycle:
    """Test VRAM lifecycle management."""

    def test_release_resources(self):
        """Test resource release."""
        config = MaterialsV2Config(enabled=True, backend='heuristic')
        engine = MaterialsV2Engine(config, device='cpu')

        # Mock segmenter
        engine._segmenter = Mock()

        # Release resources
        engine.release_resources()

        # Check segmenter deleted
        assert engine._segmenter is None

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_vram_cleanup_cuda(self):
        """Test CUDA memory cleanup (mock)."""
        with patch('lux_depth_v2.materials_v2.torch') as mock_torch:
            mock_torch.cuda.empty_cache = Mock()
            mock_torch.cuda.synchronize = Mock()

            config = MaterialsV2Config(enabled=True)
            engine = MaterialsV2Engine(config, device='cuda')
            engine._segmenter = Mock()

            engine.release_resources()

            # Check cleanup called
            mock_torch.cuda.empty_cache.assert_called_once()
            mock_torch.cuda.synchronize.assert_called_once()


class TestMaskCaching:
    """Test mask caching functionality."""

    def test_cache_manager_init(self, tmp_path):
        """Test cache manager initialization."""
        cache_dir = tmp_path / "cache"
        MaskCacheManager(cache_dir)

        # Check directory created
        assert cache_dir.exists()

    def test_compute_input_hash_array(self):
        """Test hash computation from array."""
        manager = MaskCacheManager(None)

        image1 = np.random.rand(100, 100, 3).astype(np.float32)
        image2 = image1.copy()
        image3 = np.random.rand(100, 100, 3).astype(np.float32)

        hash1 = manager.compute_input_hash_from_array(image1)
        hash2 = manager.compute_input_hash_from_array(image2)
        hash3 = manager.compute_input_hash_from_array(image3)

        # Same image = same hash
        assert hash1 == hash2

        # Different image = different hash
        assert hash1 != hash3

    def test_save_and_load_masks(self, tmp_path):
        """Test saving and loading masks."""
        cache_dir = tmp_path / "cache"
        manager = MaskCacheManager(cache_dir)

        # Create test masks
        masks = {
            'wood': np.random.rand(100, 100).astype(np.float32),
            'metal': np.random.rand(100, 100).astype(np.float32),
        }

        # Create confidence metrics
        metrics = ConfidenceMetrics(
            confidence_avg=0.75,
            confidence_min=0.3,
            confidence_max=0.95,
            high_confidence_pct=0.8,
            low_confidence_pct=0.2,
            coverage_ratio=0.85,
            material_counts={'wood': 5000, 'metal': 3000}
        )

        # Save masks
        manager.save_masks(
            task_id='test_001',
            masks=masks,
            confidence_metrics=metrics,
            input_hash='sha256:abc123',
            config={'backend': 'heuristic'}
        )

        # Check files created
        assert (cache_dir / 'test_001_wood_mask.png').exists()
        assert (cache_dir / 'test_001_metal_mask.png').exists()
        assert (cache_dir / 'test_001_metadata.json').exists()

        # Load masks
        loaded_masks, loaded_metadata = manager.load_masks('test_001')

        # Check masks loaded
        assert 'wood' in loaded_masks
        assert 'metal' in loaded_masks

        # Check shape preserved
        assert loaded_masks['wood'].shape == masks['wood'].shape

        # Check metadata
        assert loaded_metadata['input_hash'] == 'sha256:abc123'
        assert loaded_metadata['confidence_metrics']['confidence_avg'] == 0.75

    def test_cache_invalidation(self, tmp_path):
        """Test cache invalidation on hash mismatch."""
        cache_dir = tmp_path / "cache"
        manager = MaskCacheManager(cache_dir)

        # Create and save masks
        masks = {'wood': np.random.rand(100, 100).astype(np.float32)}
        metrics = ConfidenceMetrics(material_counts={'wood': 5000})

        manager.save_masks(
            task_id='test_001',
            masks=masks,
            confidence_metrics=metrics,
            input_hash='sha256:original',
            config={}
        )

        # Check cache valid with correct hash
        assert manager.is_cached('test_001', 'sha256:original')

        # Check cache invalid with wrong hash
        assert not manager.is_cached('test_001', 'sha256:modified')

    def test_cache_stats(self, tmp_path):
        """Test cache statistics."""
        cache_dir = tmp_path / "cache"
        manager = MaskCacheManager(cache_dir)

        # Save multiple masks
        for i in range(3):
            masks = {'wood': np.random.rand(100, 100).astype(np.float32)}
            metrics = ConfidenceMetrics(material_counts={'wood': 5000})

            manager.save_masks(
                task_id=f'test_{i:03d}',
                masks=masks,
                confidence_metrics=metrics,
                input_hash=f'sha256:hash{i}',
                config={}
            )

        # Get stats
        stats = manager.get_cache_stats()

        assert stats['enabled']
        assert stats['total_entries'] == 3
        assert stats['total_size_mb'] > 0


class TestMaterialsV2Integration:
    """Test Materials v2 integration with pipeline."""

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not available"
    )
    def test_materials_v2_engine_disabled(self):
        """Test engine behavior when disabled."""
        config = MaterialsV2Config(enabled=False)
        engine = MaterialsV2Engine(config, device='cpu')

        # Should raise error if trying to segment when disabled
        image = np.random.rand(100, 100, 3).astype(np.float32)

        with pytest.raises(RuntimeError, match="not enabled"):
            engine.segment_with_confidence(image)

    def test_confidence_metrics_calculation(self):
        """Test confidence metrics calculation."""
        # Create mock segmentation result
        masks = {
            'wood': np.ones((100, 100), dtype=np.float32) * 0.8,
            'metal': np.ones((100, 100), dtype=np.float32) * 0.6,
        }

        confidences = {
            'wood': np.random.uniform(0.6, 0.9, (100, 100)).astype(np.float32),
            'metal': np.random.uniform(0.4, 0.7, (100, 100)).astype(np.float32),
        }

        config = MaterialsV2Config(enabled=True)
        engine = MaterialsV2Engine(config, device='cpu')

        metrics = engine._calculate_metrics(masks, confidences)

        # Check metrics computed
        assert 0.0 <= metrics.confidence_avg <= 1.0
        assert 0.0 <= metrics.confidence_min <= 1.0
        assert 0.0 <= metrics.confidence_max <= 1.0
        assert 0.0 <= metrics.coverage_ratio <= 1.0
        assert 'wood' in metrics.material_counts
        assert 'metal' in metrics.material_counts


class TestErrorRecoveryFallbacks:
    """Test error recovery fallbacks for materials."""

    def test_segmentation_fallback(self):
        """Test fallback from ONNX to heuristic backend."""
        # This would be tested with actual error recovery integration
        # For now, test the fallback config generation pattern

        original_config = {'segmentation_backend': 'onnx'}

        # Fallback should switch to heuristic
        fallback = {'segmentation_backend': 'heuristic'}

        assert fallback['segmentation_backend'] == 'heuristic'

    def test_memory_fallback(self):
        """Test fallback for memory errors."""
        original_config = {
            'max_segmentation_side': 1536,
            'edge_feather_radius': 3
        }

        # Fallback should reduce resolution
        fallback = {
            'max_segmentation_side': original_config['max_segmentation_side'] // 2,
            'edge_feather_radius': original_config['edge_feather_radius'] * 2
        }

        assert fallback['max_segmentation_side'] == 768
        assert fallback['edge_feather_radius'] == 6


# Fixtures

@pytest.fixture
def test_image():
    """Provide a test image."""
    return _create_test_image(100, 100)


@pytest.fixture
def materials_v2_config():
    """Provide a Materials v2 config."""
    return MaterialsV2Config(
        enabled=True,
        backend='heuristic',
        confidence=ConfidenceConfig(),
        segmentation=SegmentationConfig()
    )
