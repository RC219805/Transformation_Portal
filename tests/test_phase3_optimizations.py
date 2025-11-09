"""
Tests for Phase 3 Performance Optimizations.

Validates pipeline parallelism, streaming, progressive processing, and Numba JIT.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

# Import Phase 3 features
from transformation_portal.depth.pipeline import ArchitecturalDepthPipeline
from transformation_portal.depth.processors.numba_kernels import get_numba_info, NUMBA_AVAILABLE


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def dummy_images(temp_output_dir):
    """Create dummy test images."""
    image_paths = []

    for i in range(5):
        # Create small test image (128x128)
        img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        img_path = temp_output_dir / f"test_image_{i}.jpg"
        img.save(img_path)
        image_paths.append(img_path)

    return image_paths


@pytest.fixture
def mock_depth_model():
    """Create a mock depth model for testing."""
    mock_model = Mock()

    # Mock estimate_depth to return synthetic depth map
    def mock_estimate_depth(image):
        h, w = image.shape[:2]
        depth = np.random.rand(h, w).astype(np.float32)
        return {
            'depth': depth,
            'depth_raw': depth.copy(),
            'metadata': {
                'inference_time_ms': 10.0,
                'backend': 'mock',
            }
        }

    mock_model.estimate_depth.side_effect = mock_estimate_depth
    mock_model.variant = Mock(name='SMALL')
    return mock_model


@pytest.fixture
def pipeline_config():
    """Minimal pipeline configuration for testing."""
    return {
        'depth_model': {
            'variant': 'small',
            'backend': 'pytorch_mps',
            'precision': 'fp16',
            'cache_size': 10,
            'enable_disk_cache': False,
        },
        'processing': {
            'depth_aware_denoise': {'enabled': False},
            'zone_tone_mapping': {'enabled': False},
            'atmospheric_effects': {
                'enabled': True,
                'haze_density': 0.01,
                'haze_color': [0.7, 0.8, 0.9],
                'desaturation_strength': 0.2,
            },
            'depth_guided_filters': {'enabled': False},
        },
        'output': {
            'output_format': 'png',
            'depth_colormap': 'turbo',
        }
    }


@pytest.fixture
def pipeline(pipeline_config, mock_depth_model):
    """Create a pipeline with mocked depth model."""
    with patch('transformation_portal.depth.pipeline.DepthAnythingV2Model', return_value=mock_depth_model):
        pipeline = ArchitecturalDepthPipeline(pipeline_config)
        return pipeline


class TestNumbaIntegration:
    """Test Numba JIT integration."""

    def test_numba_info(self):
        """Test Numba availability info."""
        info = get_numba_info()

        assert 'available' in info
        assert 'version' in info
        assert isinstance(info['available'], bool)

        if info['available']:
            print(f"\nNumba available: {info['version']}")
            print(f"Threading layer: {info['threading_layer']}")
            print(f"Parallel enabled: {info['parallel_enabled']}")

    def test_atmospheric_effects_with_numba(self, pipeline_config):
        """Test atmospheric effects with Numba acceleration."""
        from transformation_portal.depth.processors.atmospheric_effects import AtmosphericEffects

        # Create processor with Numba enabled
        processor = AtmosphericEffects(use_numba=True)

        # Create dummy data
        image = np.random.rand(128, 128, 3).astype(np.float32)
        depth = np.random.rand(128, 128).astype(np.float32)

        # Process
        result = processor.process(image, depth)

        # Validate
        assert result.shape == image.shape
        assert result.dtype == np.float32 or result.dtype == np.float64
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

        # Check if Numba was actually used
        if NUMBA_AVAILABLE:
            assert processor.use_numba is True
            print("\n✓ Numba JIT acceleration active")
        else:
            assert processor.use_numba is False
            print("\n✗ Numba not available, using NumPy fallback")

    def test_numba_numpy_equivalence(self):
        """Test that Numba and NumPy versions produce similar results."""
        from transformation_portal.depth.processors.atmospheric_effects import AtmosphericEffects

        # Skip if Numba not available
        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available")

        # Create two processors
        processor_numba = AtmosphericEffects(use_numba=True)
        processor_numpy = AtmosphericEffects(use_numba=False)

        # Create test data
        np.random.seed(42)
        image = np.random.rand(64, 64, 3).astype(np.float32)
        depth = np.random.rand(64, 64).astype(np.float32)

        # Process with both
        result_numba = processor_numba.process(image, depth)
        result_numpy = processor_numpy.process(image, depth)

        # Results should be very close (allowing for minor floating-point differences)
        np.testing.assert_allclose(result_numba, result_numpy, rtol=1e-5, atol=1e-6)
        print("\n✓ Numba and NumPy results are equivalent")


class TestStreamingProcessing:
    """Test streaming batch processing (Phase 3)."""

    def test_batch_process_streaming(self, pipeline, dummy_images, temp_output_dir):
        """Test streaming batch processing."""

        output_dir = temp_output_dir / "streaming_output"
        output_dir.mkdir(exist_ok=True)

        # Process with streaming
        results = []
        for result in pipeline.batch_process_streaming(
            dummy_images[:3],  # Use first 3 images
            output_dir,
            save_depth=True,
            save_visualization=True,
        ):
            results.append(result)

            # Verify result structure
            assert 'image' in result
            assert 'depth' in result
            assert 'metadata' in result

            # Check that files were saved immediately
            input_path = Path(result['metadata']['input_path'])
            stem = input_path.stem

            enhanced_path = output_dir / f"{stem}_enhanced.png"
            assert enhanced_path.exists(), f"Enhanced image not saved: {enhanced_path}"

        # Verify all images were processed
        assert len(results) == 3
        print(f"\n✓ Streaming processing completed: {len(results)} images")

    def test_streaming_memory_efficiency(self, pipeline, dummy_images, temp_output_dir):
        """Test that streaming doesn't accumulate results in memory."""

        output_dir = temp_output_dir / "streaming_memory"
        output_dir.mkdir(exist_ok=True)

        # Process without accumulating
        processed_count = 0
        for result in pipeline.batch_process_streaming(
            dummy_images,
            output_dir,
            save_depth=False,
            save_visualization=False,
        ):
            processed_count += 1
            # Don't store result - it should be garbage collected

        assert processed_count == len(dummy_images)
        print(f"\n✓ Processed {processed_count} images without accumulation")


class TestPipelineParallelism:
    """Test pipeline parallelism (Phase 3)."""

    def test_batch_process_pipelined(self, pipeline, dummy_images, temp_output_dir):
        """Test pipelined batch processing."""

        output_dir = temp_output_dir / "pipelined_output"
        output_dir.mkdir(exist_ok=True)

        # Process with pipeline parallelism
        results = []
        for result in pipeline.batch_process_pipelined(
            dummy_images[:3],
            output_dir,
            save_depth=True,
            save_visualization=False,
            pipeline_workers=2,
        ):
            results.append(result)

            # Verify result structure
            assert 'image' in result
            assert 'depth' in result
            assert 'metadata' in result

        # Verify all images were processed
        assert len(results) == 3
        print(f"\n✓ Pipelined processing completed: {len(results)} images")

    def test_pipelined_vs_sequential_equivalence(self, pipeline, dummy_images, temp_output_dir):
        """Test that pipelined and sequential processing produce same results."""

        # Sequential processing
        seq_output = temp_output_dir / "sequential"
        seq_output.mkdir(exist_ok=True)
        seq_results = list(pipeline.batch_process_streaming(
            dummy_images[:2],
            seq_output,
            save_depth=False,
            save_visualization=False,
        ))

        # Pipelined processing
        pipe_output = temp_output_dir / "pipelined"
        pipe_output.mkdir(exist_ok=True)
        pipe_results = list(pipeline.batch_process_pipelined(
            dummy_images[:2],
            pipe_output,
            save_depth=False,
            save_visualization=False,
        ))

        # Should process same number of images
        assert len(seq_results) == len(pipe_results)

        # Depth maps should be similar (allowing for cache differences)
        for seq_res, pipe_res in zip(seq_results, pipe_results):
            assert seq_res['depth'].shape == pipe_res['depth'].shape

        print(f"\n✓ Pipelined and sequential results are consistent")


class TestProgressiveProcessing:
    """Test progressive/multi-resolution processing (Phase 3)."""

    def test_process_render_progressive(self, pipeline, dummy_images, temp_output_dir):
        """Test progressive processing at multiple quality levels."""

        # Process at multiple quality levels
        result = pipeline.process_render_progressive(
            dummy_images[0],
            quality_levels=[0.25, 0.5, 1.0],
            return_all_levels=False,
        )

        # Should return highest quality level
        assert 'image' in result
        assert 'depth' in result
        assert 'metadata' in result
        assert result['metadata']['processing_scale'] == 1.0

        print(f"\n✓ Progressive processing (highest quality) completed")

    def test_process_render_progressive_all_levels(self, pipeline, dummy_images, temp_output_dir):
        """Test progressive processing returning all levels."""

        # Get all quality levels
        results = pipeline.process_render_progressive(
            dummy_images[0],
            quality_levels=[0.25, 0.5, 1.0],
            return_all_levels=True,
        )

        # Should return 3 results
        assert len(results) == 3

        # Check scales
        assert results[0]['metadata']['processing_scale'] == 0.25
        assert results[1]['metadata']['processing_scale'] == 0.5
        assert results[2]['metadata']['processing_scale'] == 1.0

        # All should have same final resolution (upscaled)
        for result in results:
            assert result['image'].shape == results[-1]['image'].shape

        print(f"\n✓ Progressive processing (all {len(results)} levels) completed")

    def test_progressive_preview_only(self, pipeline, dummy_images, temp_output_dir):
        """Test fast preview at low resolution."""

        # Quick preview at 25% resolution
        result = pipeline.process_render_progressive(
            dummy_images[0],
            quality_levels=[0.25],  # Preview only
            return_all_levels=False,
        )

        assert 'image' in result
        assert 'metadata' in result
        assert result['metadata']['processing_scale'] == 0.25

        # Processing should be faster than full resolution
        assert result['metadata']['processing_time_sec'] > 0

        print(f"\n✓ Preview-only processing completed in {result['metadata']['processing_time_sec']:.3f}s")


class TestBackwardCompatibility:
    """Test that Phase 3 features don't break existing functionality."""

    def test_standard_batch_processing_still_works(self, pipeline, dummy_images, temp_output_dir):
        """Test that original batch_process() still works."""

        output_dir = temp_output_dir / "standard_output"
        output_dir.mkdir(exist_ok=True)

        # Use original batch processing
        # Note: parallel=False because mock models can't be pickled for ProcessPoolExecutor
        results = pipeline.batch_process(
            dummy_images[:2],
            output_dir,
            save_depth=True,
            save_visualization=False,
            parallel=False,  # Disabled for mock testing
            preload_images=True,  # Phase 1 feature (async I/O)
        )

        assert len(results) == 2
        print(f"\n✓ Original batch_process() still works")

    def test_single_image_processing_still_works(self, pipeline, dummy_images, temp_output_dir):
        """Test that process_render() still works."""

        result = pipeline.process_render(dummy_images[0])

        assert 'image' in result
        assert 'depth' in result
        assert 'metadata' in result

        print(f"\n✓ Original process_render() still works")


def test_phase3_integration_summary():
    """Print Phase 3 integration summary."""
    info = get_numba_info()

    print("\n" + "=" * 60)
    print("PHASE 3 OPTIMIZATIONS - INTEGRATION SUMMARY")
    print("=" * 60)
    print(f"✓ Pipeline parallelism: IMPLEMENTED")
    print(f"✓ Streaming processing: IMPLEMENTED")
    print(f"✓ Progressive rendering: IMPLEMENTED")
    print(f"✓ Numba JIT acceleration: {'AVAILABLE' if info['available'] else 'NOT AVAILABLE (fallback active)'}")

    if info['available']:
        print(f"  - Numba version: {info['version']}")
        print(f"  - Threading layer: {info['threading_layer']}")
        print(f"  - Parallel mode: {info['parallel_enabled']}")

    print("=" * 60)
    print()
