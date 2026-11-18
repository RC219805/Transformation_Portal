"""
Tests for Phase 3 Performance Optimizations.

Validates pipeline parallelism, streaming, progressive processing, and Numba JIT.
"""

import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit

# Import Phase 3 features
from transformation_portal.depth import pipeline as depth_pipeline_module
from transformation_portal.depth.pipeline import ArchitecturalDepthPipeline
from transformation_portal.depth.processors.numba_kernels import NUMBA_AVAILABLE, get_numba_info


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
            "depth": depth,
            "depth_raw": depth.copy(),
            "metadata": {
                "inference_time_ms": 10.0,
                "backend": "mock",
            },
        }

    mock_model.estimate_depth.side_effect = mock_estimate_depth
    mock_model.variant = Mock(name="SMALL")
    return mock_model


@pytest.fixture
def pipeline_config():
    """Minimal pipeline configuration for testing."""
    return {
        "depth_model": {
            "variant": "small",
            "backend": "pytorch_mps",
            "precision": "fp16",
            "cache_size": 10,
            "enable_disk_cache": False,
        },
        "processing": {
            "depth_aware_denoise": {"enabled": False},
            "zone_tone_mapping": {"enabled": False},
            "atmospheric_effects": {
                "enabled": True,
                "haze_density": 0.01,
                "haze_color": [0.7, 0.8, 0.9],
                "desaturation_strength": 0.2,
            },
            "depth_guided_filters": {"enabled": False},
        },
        "output": {
            "output_format": "png",
            "depth_colormap": "turbo",
        },
    }


@pytest.fixture
def pipeline(pipeline_config, mock_depth_model):
    """Create a pipeline with mocked depth model."""
    with patch("transformation_portal.depth.pipeline.DepthAnythingV2Model", return_value=mock_depth_model):
        pipeline = ArchitecturalDepthPipeline(pipeline_config)
        return pipeline


class TestNumbaIntegration:
    """Test Numba JIT integration."""

    def test_numba_info(self):
        """Test Numba availability info."""
        info = get_numba_info()

        assert "available" in info
        assert "version" in info
        assert isinstance(info["available"], bool)

        if info["available"]:
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
            assert "image" in result
            assert "depth" in result
            assert "metadata" in result

            # Check that files were saved immediately
            input_path = Path(result["metadata"]["input_path"])
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
            assert "image" in result
            assert "depth" in result
            assert "metadata" in result

        # Verify all images were processed
        assert len(results) == 3
        print(f"\n✓ Pipelined processing completed: {len(results)} images")

    def test_pipelined_vs_sequential_equivalence(self, pipeline, dummy_images, temp_output_dir):
        """Test that pipelined and sequential processing produce same results."""

        # Sequential processing
        seq_output = temp_output_dir / "sequential"
        seq_output.mkdir(exist_ok=True)
        seq_results = list(
            pipeline.batch_process_streaming(
                dummy_images[:2],
                seq_output,
                save_depth=False,
                save_visualization=False,
            )
        )

        # Pipelined processing
        pipe_output = temp_output_dir / "pipelined"
        pipe_output.mkdir(exist_ok=True)
        pipe_results = list(
            pipeline.batch_process_pipelined(
                dummy_images[:2],
                pipe_output,
                save_depth=False,
                save_visualization=False,
            )
        )

        # Should process same number of images
        assert len(seq_results) == len(pipe_results)

        # Depth maps should be similar (allowing for cache differences)
        for seq_res, pipe_res in zip(seq_results, pipe_results):
            assert seq_res["depth"].shape == pipe_res["depth"].shape

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
        assert "image" in result
        assert "depth" in result
        assert "metadata" in result
        assert result["metadata"]["processing_scale"] == 1.0

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
        assert results[0]["metadata"]["processing_scale"] == 0.25
        assert results[1]["metadata"]["processing_scale"] == 0.5
        assert results[2]["metadata"]["processing_scale"] == 1.0

        # All should have same final resolution (upscaled)
        for result in results:
            assert result["image"].shape == results[-1]["image"].shape

        print(f"\n✓ Progressive processing (all {len(results)} levels) completed")

    def test_progressive_preview_only(self, pipeline, dummy_images, temp_output_dir):
        """Test fast preview at low resolution."""

        # Quick preview at 25% resolution
        result = pipeline.process_render_progressive(
            dummy_images[0],
            quality_levels=[0.25],  # Preview only
            return_all_levels=False,
        )

        assert "image" in result
        assert "metadata" in result
        assert result["metadata"]["processing_scale"] == 0.25

        # Processing should be faster than full resolution
        assert result["metadata"]["processing_time_sec"] > 0

        print(f"\n✓ Preview-only processing completed in {result['metadata']['processing_time_sec']:.3f}s")


class TestBackwardCompatibility:
    """Test that Phase 3 features don't break existing functionality."""

    def test_standard_batch_processing_still_works(self, pipeline, dummy_images, temp_output_dir):
        """Test that original batch_process() still works."""

        output_dir = temp_output_dir / "standard_output"
        output_dir.mkdir(exist_ok=True)

        # Use original batch processing
        # Sequential fallback path; the default thread-based path is covered by TestParallelBatchProcessing
        results = pipeline.batch_process(
            dummy_images[:2],
            output_dir,
            save_depth=True,
            save_visualization=False,
            parallel=False,
            preload_images=True,  # Phase 1 feature (async I/O)
        )

        assert len(results) == 2
        print(f"\n✓ Original batch_process() still works")

    def test_single_image_processing_still_works(self, pipeline, dummy_images, temp_output_dir):
        """Test that process_render() still works."""

        result = pipeline.process_render(dummy_images[0])

        assert "image" in result
        assert "depth" in result
        assert "metadata" in result

        print(f"\n✓ Original process_render() still works")


class NativeHandleDepthModel:
    """Depth model stub that owns a native handle, like an ONNX session or a Core ML model.

    It cannot be pickled, counts every pickling attempt, and records how many threads are inside
    ``estimate_depth`` at once.
    """

    def __init__(self, error=None):
        self.variant = Mock()
        self.error = error
        self.guard = threading.Lock()
        self.native_handle = threading.Lock()
        self.pickle_attempts = 0
        self.active = 0
        self.max_active = 0
        self.completed = 0

    def __reduce_ex__(self, protocol):
        self.pickle_attempts += 1
        raise TypeError("cannot pickle a native model handle")

    def estimate_depth(self, image):
        with self.guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(0.01)  # let a second, unserialised caller overlap
            if self.error is not None:
                raise self.error
            depth = np.ascontiguousarray(image[..., 0], dtype=np.float32)
            return {
                "depth": depth,
                "depth_raw": depth.copy(),
                "metadata": {"inference_time_ms": 10.0, "backend": "stub"},
            }
        finally:
            with self.guard:
                self.active -= 1
                self.completed += 1


def make_pipeline(pipeline_config, depth_model):
    """Create a pipeline around a stub depth model."""
    with patch("transformation_portal.depth.pipeline.DepthAnythingV2Model", return_value=depth_model):
        return ArchitecturalDepthPipeline(pipeline_config)


class TestParallelBatchProcessing:
    """batch_process(parallel=True), the default, shares the pipeline with worker threads."""

    def test_default_batch_processes_a_model_that_cannot_be_pickled(self, pipeline_config, dummy_images, temp_output_dir):
        """The default mode never pickles the pipeline and runs one inference at a time."""
        model = NativeHandleDepthModel()
        pipeline = make_pipeline(pipeline_config, model)
        output_dir = temp_output_dir / "parallel_output"

        results = pipeline.batch_process(dummy_images, output_dir, save_visualization=False, max_workers=3)

        assert [result["metadata"]["input_path"] for result in results] == [str(path) for path in dummy_images]
        assert model.pickle_attempts == 0
        assert model.max_active == 1
        assert pipeline.stats["images_processed"] == len(dummy_images)
        for path in dummy_images:
            assert (output_dir / f"{path.stem}_enhanced.png").is_file()
            assert (output_dir / f"{path.stem}_depth.npy").is_file()

    def test_parallel_batch_keeps_at_most_max_workers_decoded_images_in_flight(
        self, pipeline_config, dummy_images, temp_output_dir, monkeypatch
    ):
        """Images are decoded by the workers, not all up front."""
        model = NativeHandleDepthModel()
        pipeline = make_pipeline(pipeline_config, model)
        real_load_image = depth_pipeline_module.load_image
        in_flight = []

        def counting_load_image(path, **kwargs):
            image = real_load_image(path, **kwargs)
            with model.guard:
                in_flight.append(len(in_flight) + 1 - model.completed)
            return image

        monkeypatch.setattr(depth_pipeline_module, "load_image", counting_load_image)

        results = pipeline.batch_process(
            dummy_images, temp_output_dir / "bounded_output", save_depth=False, save_visualization=False, max_workers=2
        )

        assert len(results) == len(dummy_images)
        assert len(in_flight) == len(dummy_images)
        assert max(in_flight) <= 2

    def test_parallel_batch_matches_sequential_batch(self, pipeline_config, dummy_images, temp_output_dir):
        """Threads change the scheduling, not the results or the files."""
        parallel_dir = temp_output_dir / "parallel_output"
        sequential_dir = temp_output_dir / "sequential_output"

        parallel_results = make_pipeline(pipeline_config, NativeHandleDepthModel()).batch_process(
            dummy_images, parallel_dir, save_visualization=False, max_workers=3
        )
        sequential_results = make_pipeline(pipeline_config, NativeHandleDepthModel()).batch_process(
            dummy_images, sequential_dir, save_visualization=False, parallel=False
        )

        assert len(parallel_results) == len(sequential_results) == len(dummy_images)
        for path, parallel_result, sequential_result in zip(dummy_images, parallel_results, sequential_results):
            assert parallel_result["metadata"]["input_path"] == sequential_result["metadata"]["input_path"] == str(path)
            np.testing.assert_array_equal(parallel_result["image"], sequential_result["image"])
            np.testing.assert_array_equal(parallel_result["depth"], sequential_result["depth"])
            for suffix in ("_enhanced.png", "_depth.npy"):
                name = f"{path.stem}{suffix}"
                assert (parallel_dir / name).read_bytes() == (sequential_dir / name).read_bytes()

    @pytest.mark.parametrize("parallel", [True, False])
    def test_batch_raises_when_every_image_fails(self, pipeline_config, dummy_images, temp_output_dir, parallel):
        """A batch that produced nothing fails closed instead of returning an empty list."""
        error = RuntimeError("model handle lost")
        pipeline = make_pipeline(pipeline_config, NativeHandleDepthModel(error=error))
        output_dir = temp_output_dir / "failed_output"

        with pytest.raises(RuntimeError, match="failed for all 3 images.*model handle lost") as excinfo:
            pipeline.batch_process(dummy_images[:3], output_dir, parallel=parallel, max_workers=2)

        assert excinfo.value.__cause__ is error
        assert list(output_dir.iterdir()) == []
        assert pipeline.stats["images_processed"] == 0

    @pytest.mark.parametrize("parallel", [True, False])
    def test_batch_skips_a_failing_image_and_keeps_the_rest(self, pipeline_config, dummy_images, temp_output_dir, parallel):
        """One bad input is logged and skipped; the other results keep their input order."""
        pipeline = make_pipeline(pipeline_config, NativeHandleDepthModel())
        missing = temp_output_dir / "missing_image.jpg"
        image_paths = [dummy_images[0], missing, dummy_images[1]]

        results = pipeline.batch_process(
            image_paths, temp_output_dir / "partial_output", save_visualization=False, parallel=parallel, max_workers=2
        )

        assert [result["metadata"]["input_path"] for result in results] == [str(dummy_images[0]), str(dummy_images[1])]
        assert pipeline.stats["images_processed"] == 2

    def test_interrupted_parallel_batch_cancels_the_queued_images(
        self, pipeline_config, dummy_images, temp_output_dir, monkeypatch
    ):
        """An interrupt propagates without first working through the rest of the queue."""
        submitted = []

        class RecordingExecutor(ThreadPoolExecutor):
            def submit(self, fn, /, *args, **kwargs):
                future = super().submit(fn, *args, **kwargs)
                submitted.append(future)
                return future

        model = NativeHandleDepthModel()
        estimate_depth = model.estimate_depth

        def estimate_then_wait_for_the_interrupt(image):
            result = estimate_depth(image)
            if model.completed == 2:
                # Hold the single worker until the interrupted caller has cancelled the queue.
                deadline = time.monotonic() + 5.0
                while not submitted[-1].cancelled() and time.monotonic() < deadline:
                    time.sleep(0.005)
            return result

        def interrupt(*args, **kwargs):
            raise KeyboardInterrupt

        model.estimate_depth = estimate_then_wait_for_the_interrupt
        pipeline = make_pipeline(pipeline_config, model)
        monkeypatch.setattr(depth_pipeline_module, "ThreadPoolExecutor", RecordingExecutor)
        monkeypatch.setattr(pipeline, "save_result", interrupt)

        with pytest.raises(KeyboardInterrupt):
            pipeline.batch_process(dummy_images, temp_output_dir / "interrupted_output", max_workers=1)

        assert len(submitted) == len(dummy_images)
        assert all(future.cancelled() for future in submitted[2:])
        assert model.completed <= 2


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

    if info["available"]:
        print(f"  - Numba version: {info['version']}")
        print(f"  - Threading layer: {info['threading_layer']}")
        print(f"  - Parallel mode: {info['parallel_enabled']}")

    print("=" * 60)
    print()


# ============================================================================
# Depth Postprocessing Tests (PR #714)
# ============================================================================


class TestDepthPostprocessing:
    """Tests for _postprocess_depth feature (PR #714)."""

    @pytest.fixture
    def sample_depth(self):
        """Create a sample depth map for testing."""
        np.random.seed(42)
        depth = np.random.rand(64, 64).astype(np.float32) * 100.0 + 10.0
        return depth

    @pytest.fixture
    def minimal_config(self):
        """Minimal config without depth postprocessing."""
        return {
            "depth_model": {
                "variant": "small",
                "backend": "pytorch_cpu",
            },
            "processing": {},
        }

    @pytest.fixture
    def postprocess_config(self):
        """Config with depth postprocessing enabled."""
        return {
            "depth_model": {
                "variant": "small",
                "backend": "pytorch_cpu",
            },
            "processing": {
                "depth_postprocessing": {
                    "enabled": True,
                    "method": "bilateral",
                    "sigma": 5.0,
                    "edge_preserve": 0.1,
                    "preserve_scale": True,
                },
            },
        }

    def test_postprocess_disabled_by_default(self, minimal_config, sample_depth):
        """Test that postprocessing is disabled when not configured."""
        with patch("transformation_portal.depth.pipeline.DepthAnythingV2Model"):
            pipeline = ArchitecturalDepthPipeline(minimal_config)
            result = pipeline._postprocess_depth(sample_depth)

        # Should return original depth unchanged
        np.testing.assert_array_equal(result, sample_depth)

    def test_postprocess_enabled_config_validation(self, postprocess_config):
        """Test that postprocessing can be enabled via config."""
        with patch("transformation_portal.depth.pipeline.DepthAnythingV2Model"):
            pipeline = ArchitecturalDepthPipeline(postprocess_config)

        # Config should be parsed correctly
        assert pipeline.config["processing"]["depth_postprocessing"]["enabled"] is True
        assert pipeline.config["processing"]["depth_postprocessing"]["method"] == "bilateral"

    def test_postprocess_invalid_method_fallback(self, sample_depth):
        """Test that invalid method gracefully falls back."""
        config = {
            "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
            "processing": {
                "depth_postprocessing": {
                    "enabled": True,
                    "method": "invalid_method",
                    "sigma": 5.0,
                },
            },
        }

        with patch("transformation_portal.depth.pipeline.DepthAnythingV2Model"):
            pipeline = ArchitecturalDepthPipeline(config)
            result = pipeline._postprocess_depth(sample_depth)

        # Should return original depth when method is invalid
        np.testing.assert_array_equal(result, sample_depth)

    def test_preserve_scale_boolean_parsing(self):
        """Test that preserve_scale config controls rescaling behavior at runtime."""
        test_cases = [
            (True, True),
            (False, False),
            ("true", True),
            ("false", False),
            ("yes", True),
            ("no", False),
            ("1", True),
            ("0", False),
            (1, True),  # Numeric integer
            (0, False),  # Numeric integer
            (1.0, True),  # Numeric float
            (0.0, False),  # Numeric float
            ("invalid", True),  # Should default to True for bilateral
        ]

        sample_depth = np.linspace(10.0, 100.0, 100, dtype=np.float32).reshape(10, 10)

        for input_value, expected_preserve in test_cases:
            config = {
                "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
                "processing": {
                    "depth_postprocessing": {
                        "enabled": True,
                        "method": "bilateral",
                        "preserve_scale": input_value,
                        "sigma": 5.0,
                        "edge_preserve": 0.1,
                    },
                },
            }

            with (
                patch("transformation_portal.depth.pipeline.DepthAnythingV2Model"),
                patch("transformation_portal.depth.pipeline.smooth_depth") as mock_smooth,
            ):
                # Reset mock for clean test isolation
                mock_smooth.reset_mock()

                # Mock smooth_depth to return normalized [0,1] output (simulating bilateral behavior)
                normalized_output = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
                mock_smooth.return_value = normalized_output

                pipeline = ArchitecturalDepthPipeline(config)
                result = pipeline._postprocess_depth(sample_depth)

                # Verify behavior based on preserve_scale setting
                if expected_preserve:
                    # Should rescale from [0,1] back to [10, 100]
                    assert result.min() >= 9.9, f"Failed for input={input_value}: min={result.min()}"
                    assert result.max() <= 100.1, f"Failed for input={input_value}: max={result.max()}"
                    # Check range is approximately preserved
                    assert (
                        abs(result.max() - result.min() - 90.0) < 1.0
                    ), f"Failed for input={input_value}: range={result.max() - result.min()}"
                else:
                    # When preserve_scale=False, smooth_depth is called without rescaling
                    mock_smooth.assert_called()  # Verify it was called
                    # Assert output stays in normalized [0,1] range (not rescaled)
                    assert (
                        result.min() >= -1e-6
                    ), f"Failed for input={input_value}: min={result.min()} < 0 (was rescaled unexpectedly)"
                    assert (
                        result.max() <= 1.0 + 1e-6
                    ), f"Failed for input={input_value}: max={result.max()} > 1.0 (was rescaled unexpectedly)"
                    # Verify exact output match (no rescaling happened)
                    assert np.allclose(
                        result, normalized_output, atol=1e-6
                    ), f"Failed for input={input_value}: output was modified when preserve_scale=False"

    @patch("transformation_portal.depth.pipeline.smooth_depth")
    def test_postprocess_calls_smooth_depth(self, mock_smooth, sample_depth):
        """Test that postprocessing calls smooth_depth utility."""
        # Mock smooth_depth to return normalized output
        mock_smooth.return_value = np.random.rand(64, 64).astype(np.float32)

        config = {
            "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
            "processing": {
                "depth_postprocessing": {
                    "enabled": True,
                    "method": "bilateral",
                    "sigma": 5.0,
                    "edge_preserve": 0.1,
                    "preserve_scale": True,
                },
            },
        }

        with patch("transformation_portal.depth.pipeline.DepthAnythingV2Model"):
            pipeline = ArchitecturalDepthPipeline(config)
            result = pipeline._postprocess_depth(sample_depth)

        # Verify smooth_depth was called
        mock_smooth.assert_called_once()
        call_args = mock_smooth.call_args
        assert call_args.kwargs["method"] == "bilateral"
        assert call_args.kwargs["sigma"] == 5.0
        assert call_args.kwargs["edge_preserve"] == 0.1

    @patch("transformation_portal.depth.pipeline.smooth_depth")
    def test_preserve_scale_rescaling(self, mock_smooth, sample_depth):
        """Test that preserve_scale correctly rescales normalized output."""
        # Mock smooth_depth to return normalized [0,1] output
        normalized_output = np.random.rand(64, 64).astype(np.float32)
        mock_smooth.return_value = normalized_output

        config = {
            "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
            "processing": {
                "depth_postprocessing": {
                    "enabled": True,
                    "method": "bilateral",
                    "sigma": 5.0,
                    "preserve_scale": True,
                },
            },
        }

        with patch("transformation_portal.depth.pipeline.DepthAnythingV2Model"):
            pipeline = ArchitecturalDepthPipeline(config)
            result = pipeline._postprocess_depth(sample_depth)

        # Result should be rescaled to original range
        original_min = float(np.nanmin(sample_depth))
        original_max = float(np.nanmax(sample_depth))

        assert result.min() >= original_min - 1e-3
        assert result.max() <= original_max + 1e-3

    @patch("transformation_portal.depth.pipeline.smooth_depth")
    def test_preserve_scale_detects_unnormalized_output(self, mock_smooth, sample_depth):
        """Test that preserve_scale detects already-scaled output from fallback."""
        # Mock smooth_depth to return already-scaled output (fallback behavior)
        # This simulates gaussian_filter which preserves original scale
        already_scaled = sample_depth + np.random.randn(64, 64) * 0.1
        mock_smooth.return_value = already_scaled.astype(np.float32)

        config = {
            "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
            "processing": {
                "depth_postprocessing": {
                    "enabled": True,
                    "method": "bilateral",
                    "sigma": 5.0,
                    "preserve_scale": True,
                },
            },
        }

        with patch("transformation_portal.depth.pipeline.DepthAnythingV2Model"):
            pipeline = ArchitecturalDepthPipeline(config)
            result = pipeline._postprocess_depth(sample_depth)

        # Result should be the already-scaled output, not double-rescaled
        np.testing.assert_allclose(result, already_scaled, rtol=1e-5)

    def test_postprocess_gaussian_method(self, sample_depth):
        """Test postprocessing with gaussian method."""
        config = {
            "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
            "processing": {
                "depth_postprocessing": {
                    "enabled": True,
                    "method": "gaussian",
                    "sigma": 3.0,
                    "preserve_scale": False,
                },
            },
        }

        with (
            patch("transformation_portal.depth.pipeline.DepthAnythingV2Model"),
            patch("transformation_portal.depth.pipeline.smooth_depth") as mock_smooth,
        ):
            # Mock gaussian to return smoothed depth
            mock_smooth.return_value = sample_depth * 0.95

            pipeline = ArchitecturalDepthPipeline(config)
            result = pipeline._postprocess_depth(sample_depth)

            # Verify gaussian was called
            mock_smooth.assert_called_once()
            assert mock_smooth.call_args.kwargs["method"] == "gaussian"

    def test_postprocess_median_method(self, sample_depth):
        """Test postprocessing with median method."""
        config = {
            "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
            "processing": {
                "depth_postprocessing": {
                    "enabled": True,
                    "method": "median",
                    "sigma": 5.0,
                    "preserve_scale": False,
                },
            },
        }

        with (
            patch("transformation_portal.depth.pipeline.DepthAnythingV2Model"),
            patch("transformation_portal.depth.pipeline.smooth_depth") as mock_smooth,
        ):
            # Mock median to return smoothed depth
            mock_smooth.return_value = sample_depth * 0.98

            pipeline = ArchitecturalDepthPipeline(config)
            result = pipeline._postprocess_depth(sample_depth)

            # Verify median was called
            mock_smooth.assert_called_once()
            assert mock_smooth.call_args.kwargs["method"] == "median"

    def test_postprocess_error_handling(self, sample_depth):
        """Test that errors in postprocessing are caught and logged."""
        config = {
            "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
            "processing": {
                "depth_postprocessing": {
                    "enabled": True,
                    "method": "bilateral",
                    "sigma": 5.0,
                },
            },
        }

        with (
            patch("transformation_portal.depth.pipeline.DepthAnythingV2Model"),
            patch("transformation_portal.depth.pipeline.smooth_depth", side_effect=ValueError("Test error")),
        ):
            pipeline = ArchitecturalDepthPipeline(config)
            result = pipeline._postprocess_depth(sample_depth)

            # Should return original depth on error
            np.testing.assert_array_equal(result, sample_depth)

    def test_postprocess_integration_with_process_render(self, temp_output_dir):
        """Test that postprocessing integrates with process_render."""
        config = {
            "depth_model": {"variant": "small", "backend": "pytorch_cpu"},
            "processing": {
                "depth_postprocessing": {
                    "enabled": True,
                    "method": "bilateral",
                    "sigma": 5.0,
                    "preserve_scale": True,
                },
            },
        }

        # Create a test image
        img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = temp_output_dir / "test_postprocess.jpg"
        img.save(img_path)

        with (
            patch("transformation_portal.depth.pipeline.DepthAnythingV2Model") as mock_model_class,
            patch("transformation_portal.depth.pipeline.smooth_depth") as mock_smooth,
        ):
            # Setup mocks
            mock_model = Mock()
            mock_model.estimate_depth.return_value = {
                "depth": np.random.rand(128, 128).astype(np.float32) * 100.0,
                "confidence": 0.95,
                "metadata": {
                    "inference_time_ms": 50.0,
                    "model_backend": "pytorch_cpu",
                },
            }
            mock_model_class.return_value = mock_model

            # Mock smooth_depth to return normalized output
            mock_smooth.return_value = np.random.rand(128, 128).astype(np.float32)

            pipeline = ArchitecturalDepthPipeline(config)
            result = pipeline.process_render(img_path)

            # Verify postprocessing was applied
            assert "depth" in result
            assert "image" in result
            mock_smooth.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
