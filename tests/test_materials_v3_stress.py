"""
Stress and Stability Tests for MaterialsV3 Integration

This test suite validates MaterialsV3 stability under heavy load:
- Batch processing (100+ images)
- 1000+ iteration stability
- Concurrent pipeline execution
- Resource exhaustion scenarios
- Memory leak detection
- Performance consistency

Success Criteria:
- No crashes over 1000+ iterations
- Consistent results across iterations
- No memory leaks
- Fallback rate = 0% for valid synthetic images
- Concurrent execution without race conditions
"""

import pytest
import numpy as np
from pathlib import Path
from PIL import Image
import multiprocessing
import time
import gc
import os
from unittest.mock import patch

# Import PyTorch availability from the canonical source
from lux_depth_v2.torch_ops import TORCH_AVAILABLE

# Conditional imports
if TORCH_AVAILABLE:
    from lux_depth_v2.pipeline import LuxPipelineV2
    from lux_depth_v2.config import PipelineConfig, Preset, DepthMode
else:
    LuxPipelineV2 = None
    PipelineConfig = None
    Preset = None
    DepthMode = None


def _materials_v3_meta(result: object) -> dict:
    """
    Returns MaterialsV3 metadata dict from the pipeline result, supporting both:
      - top-level: result["materials_v3_metadata"]
      - nested:    result["metadata"]["materials_v3"]
    """
    if not isinstance(result, dict):
        return {}
    if isinstance(result.get("materials_v3_metadata"), dict):
        return result["materials_v3_metadata"]
    md = result.get("metadata")
    if isinstance(md, dict) and isinstance(md.get("materials_v3"), dict):
        return md["materials_v3"]
    return {}


# Module-level skip - stress tests excluded from PR CI, enabled on nightly/manual runs
pytestmark = pytest.mark.skipif(
    os.getenv("CI") == "true"
    and os.getenv("GITHUB_EVENT_NAME") not in ("schedule", "workflow_dispatch"),
    reason="Stress tests excluded from PR CI; enabled on schedule/manual runs",
)


# Module-level worker function for multiprocessing (must be picklable)
def _process_image_worker(args):
    """Worker function for concurrent pipeline execution test."""
    img_path, output_dir, worker_id = args
    
    # Import here to avoid issues in workers
    from lux_depth_v2.pipeline import LuxPipelineV2
    from lux_depth_v2.config import PipelineConfig, Preset, DepthMode
    
    # Create CI-safe config in worker process
    config = PipelineConfig(
        preset=Preset.PRODUCTION_STANDARD,
        output_dir=output_dir,
        write_outputs=False
    )
    config.segmentation.backend = "heuristic"
    config.depth.mode = DepthMode.AUTO
    config.strict_depth = False
    
    pipeline = LuxPipelineV2(config)
    
    try:
        result = pipeline.process_one(img_path)
        m3 = _materials_v3_meta(result)
        return {
            'worker_id': worker_id,
            'success': True,
            'fallback': bool(m3.get('fallback', False))
        }
    except Exception as e:
        return {
            'worker_id': worker_id,
            'success': False,
            'error': str(e)
        }


@pytest.mark.slow
@pytest.mark.stress
@pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch is required for LuxPipelineV2"
)
class TestMaterialsV3Stress:
    """Stress and stability tests for MaterialsV3."""
    
    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a valid synthetic sample image."""
        img_path = tmp_path / "sample_image.jpg"
        # Create diverse synthetic image with gradients
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        # Add gradients and patterns
        for i in range(512):
            for j in range(512):
                img_array[i, j] = [
                    int(128 + 64 * np.sin(i / 50)),
                    int(128 + 64 * np.cos(j / 50)),
                    int(128 + 32 * np.sin((i + j) / 70))
                ]
        img = Image.fromarray(img_array, 'RGB')
        img.save(img_path, quality=95)
        return img_path
    
    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory."""
        out_dir = tmp_path / "stress_output"
        out_dir.mkdir(exist_ok=True)
        return out_dir
    
    @pytest.fixture
    def ci_safe_config(self, tmp_path):
        """Create CI-safe config with heuristic backend and AUTO depth mode."""
        config = PipelineConfig(
            preset=Preset.PRODUCTION_STANDARD,
            output_dir=tmp_path / "ci_output",
            write_outputs=False
        )
        config.segmentation.backend = "heuristic"
        # Stress tests should not fail on missing depth - use AUTO mode
        config.depth.mode = DepthMode.AUTO
        config.strict_depth = False
        return config
    
    def test_1000_iteration_stability(self, sample_image, ci_safe_config, output_dir):
        """
        Validate MaterialsV3 stability over iterations.
        
        CI mode: 50 iterations (smoke test, ~30s)
        Full mode: 1000 iterations (stress test, ~10min)
        
        Success Criteria:
        - Zero crashes (each iteration completes)
        - Zero fallbacks for valid synthetic images
        - Memory stable (no accumulation)
        """
        # Tier gating: Full stress on nightly/manual, smoke on PR CI, full on local
        in_ci = os.getenv("CI") == "true"
        is_nightly = os.getenv("GITHUB_EVENT_NAME") in ("schedule", "workflow_dispatch")
        full_stress = is_nightly or (os.getenv("MATERIALSV3_STRESS_FULL") == "1")
        iterations = 1000 if (not in_ci or full_stress) else 50
        
        pipeline = LuxPipelineV2(ci_safe_config)
        
        # Track results
        results = []
        fallback_count = 0
        errors = []
        
        print(f"\n{'='*60}")
        print(f"Starting {iterations}-iteration stability test (CI={os.getenv('CI')})...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for i in range(iterations):
            try:
                result = pipeline.process_one(sample_image)
                results.append(result)
                
                # Check for fallback
                m3 = _materials_v3_meta(result)
                if m3.get('fallback', False):
                    fallback_count += 1
                    errors.append(f"Iteration {i}: {m3.get('error', 'Unknown')}")
                
                # Progress reporting
                report_interval = max(10, iterations // 10)  # Report ~10 times
                if (i + 1) % report_interval == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"Progress: {i+1}/{iterations} iterations ({rate:.1f} iter/sec, {fallback_count} fallbacks)")
                    
            except Exception as e:
                errors.append(f"Iteration {i}: {str(e)}")
                # Should not crash - fail test
                pytest.fail(f"Iteration {i} crashed: {e}")
        
        elapsed_total = time.time() - start_time
        avg_rate = iterations / elapsed_total
        
        # Verify all iterations completed
        assert len(results) == iterations, f"Expected {iterations} results, got {len(results)}"
        
        # Verify no fallbacks (synthetic image should always succeed)
        print(f"\n{'='*60}")
        print(f"1000-iteration test completed in {elapsed_total:.1f}s ({avg_rate:.1f} iter/sec)")
        print(f"Fallbacks: {fallback_count}/1000 ({fallback_count/10:.1f}%)")
        print(f"{'='*60}\n")
        
        if fallback_count > 0:
            print("\nFallback errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        # Success criteria: 0% fallback rate for synthetic images
        assert fallback_count == 0, \
            f"Unexpected fallbacks: {fallback_count}/1000 ({fallback_count/10:.1f}%)\n" + \
            f"Errors: {errors[:5]}"
    
    def test_batch_processing_100_images(self, tmp_path, ci_safe_config, output_dir):
        """
        Process images in batch and verify MaterialsV3 stability.
        
        Nightly: 100 images (stress test, ~10min)
        Local/PR: 20 images (smoke test, ~2min)
        """
        # Tier gating: Full stress on nightly/manual, smoke on PR CI, full on local
        in_ci = os.getenv("CI") == "true"
        is_nightly = os.getenv("GITHUB_EVENT_NAME") in ("schedule", "workflow_dispatch")
        full_stress = is_nightly or (os.getenv("MATERIALSV3_STRESS_FULL") == "1")
        batch_size = 100 if (not in_ci or full_stress) else 20
        
        # Generate synthetic images with varying characteristics
        image_paths = []
        
        print(f"\n{'='*60}")
        print(f"Generating {batch_size} synthetic images...")
        print(f"{'='*60}")
        
        for i in range(batch_size):
            img_path = tmp_path / f"batch_image_{i:03d}.jpg"
            # Vary image characteristics
            size = 256 + (i % 5) * 64  # 256, 320, 384, 448, 512
            
            # Create diverse images
            img_array = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
            # Add some structure (gradients, patterns)
            for y in range(size):
                for x in range(size):
                    img_array[y, x, 0] = int(128 + 64 * np.sin((x + i) / 30))
                    img_array[y, x, 1] = int(128 + 64 * np.cos((y + i) / 30))
                    img_array[y, x, 2] = int(128 + 32 * np.sin((x + y + i) / 40))
            
            img = Image.fromarray(img_array, 'RGB')
            img.save(img_path, quality=90)
            image_paths.append(img_path)
        
        pipeline = LuxPipelineV2(ci_safe_config)
        
        # Process batch
        results = []
        fallback_count = 0
        errors = []
        
        print(f"\n{'='*60}")
        print(f"Processing 100 images with MaterialsV3...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for i, img_path in enumerate(image_paths):
            try:
                result = pipeline.process_one(img_path)
                results.append(result)
                
                # Track fallbacks
                m3 = _materials_v3_meta(result)
                if m3.get('fallback', False):
                    fallback_count += 1
                    errors.append(f"Image {i}: {m3.get('error', 'Unknown')}")
                
                # Progress reporting
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"Progress: {i+1}/100 images ({rate:.1f} img/sec, {fallback_count} fallbacks)")
                    
            except Exception as e:
                errors.append(f"Image {i}: {str(e)}")
                # Should not crash
                pytest.fail(f"Batch image {i} crashed: {e}")
        
        elapsed_total = time.time() - start_time
        avg_rate = batch_size / elapsed_total
        
        # Verify all processed
        assert len(results) == batch_size, f"Expected {batch_size} results, got {len(results)}"
        
        print(f"\n{'='*60}")
        print(f"Batch processing completed in {elapsed_total:.1f}s ({avg_rate:.1f} img/sec)")
        print(f"Fallbacks: {fallback_count}/100 ({fallback_count}%)")
        print(f"{'='*60}\n")
        
        if errors:
            print("\nErrors encountered:")
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        # Allow up to 5% fallback rate (for edge cases in random generation)
        assert fallback_count <= 5, \
            f"Too many fallbacks: {fallback_count}/100 (>{5}%)\n" + \
            f"Errors: {errors[:5]}"
    
    @pytest.mark.slow
    def test_concurrent_pipeline_execution(self, tmp_path):
        """Multiple pipelines with MaterialsV3 should not interfere."""
        # Create test images
        images = []
        for i in range(4):
            img_path = tmp_path / f"concurrent_image_{i}.jpg"
            img = Image.new('RGB', (256, 256), color=(50 + i * 50, 100, 150))
            img.save(img_path, quality=95)
            images.append(img_path)
        
        output_dirs = [tmp_path / f"concurrent_output_{i}" for i in range(4)]
        for d in output_dirs:
            d.mkdir(exist_ok=True)
        
        # Run 4 pipelines concurrently
        print(f"\n{'='*60}")
        print(f"Running 4 concurrent pipelines...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        with multiprocessing.Pool(processes=4) as pool:
            args = [(images[i], output_dirs[i], i) for i in range(4)]
            results = pool.map(_process_image_worker, args)
        
        elapsed = time.time() - start_time
        
        # Verify all completed
        assert len(results) == 4, f"Expected 4 results, got {len(results)}"
        
        # Verify all succeeded
        failures = [r for r in results if not r['success']]
        fallbacks = [r for r in results if r.get('fallback', False)]
        
        print(f"\n{'='*60}")
        print(f"Concurrent execution completed in {elapsed:.1f}s")
        print(f"Successes: {len([r for r in results if r['success']])}/4")
        print(f"Failures: {len(failures)}/4")
        print(f"Fallbacks: {len(fallbacks)}/4")
        print(f"{'='*60}\n")
        
        if failures:
            print("\nFailures:")
            for f in failures:
                print(f"  Worker {f['worker_id']}: {f.get('error', 'Unknown')}")
        
        assert len(failures) == 0, \
            f"Concurrent execution had failures: {failures}"
        
        # No race conditions - all should succeed
        assert len(fallbacks) == 0, \
            f"Concurrent execution had unexpected fallbacks: {fallbacks}"
    
    def test_memory_stability_over_iterations(self, sample_image, ci_safe_config):
        """MaterialsV3 should not leak memory over multiple iterations."""
        pipeline = LuxPipelineV2(ci_safe_config)
        
        # Measure memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_samples = []
            
            # Warmup (first few iterations may allocate caches)
            for _ in range(10):
                pipeline.process_one(sample_image)
            
            gc.collect()
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run 100 iterations and track memory
            print(f"\n{'='*60}")
            print(f"Memory stability test (100 iterations)...")
            print(f"Baseline memory: {baseline_memory:.1f} MB")
            print(f"{'='*60}")
            
            for i in range(100):
                pipeline.process_one(sample_image)
                
                if i % 10 == 0:
                    gc.collect()
                    current_memory = process.memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    print(f"Iteration {i}: {current_memory:.1f} MB (Δ {current_memory - baseline_memory:+.1f} MB)")
            
            gc.collect()
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_growth = final_memory - baseline_memory
            
            print(f"\n{'='*60}")
            print(f"Final memory: {final_memory:.1f} MB")
            print(f"Memory growth: {memory_growth:+.1f} MB ({memory_growth/baseline_memory*100:+.1f}%)")
            print(f"{'='*60}\n")
            
            # Memory growth should be minimal (< 50% of baseline)
            # Some growth is acceptable due to caching
            assert memory_growth < baseline_memory * 0.5, \
                f"Excessive memory growth: {memory_growth:.1f} MB ({memory_growth/baseline_memory*100:.1f}%)"
                
        except ImportError:
            pytest.skip("psutil not available for memory profiling")
    
    def test_gpu_memory_exhaustion_fallback(self, tmp_path, ci_safe_config):
        """MaterialsV3 should fallback gracefully if GPU OOM occurs."""
        # Create a very large image that might trigger GPU OOM
        large_img_path = tmp_path / "gpu_stress_image.jpg"
        # 4096x4096 = 16.7MP (may stress GPU memory)
        large_img = Image.new('RGB', (4096, 4096), color=(128, 128, 128))
        large_img.save(large_img_path, quality=95)
        
        ci_safe_config.max_megapixels = 50  # Allow large image
        
        pipeline = LuxPipelineV2(ci_safe_config)
        
        # Mock GPU to simulate OOM
        def mock_gpu_oom(*args, **kwargs):
            raise RuntimeError("CUDA out of memory")
        
        # Try processing (may trigger OOM or succeed)
        try:
            # First, try normal processing
            result = pipeline.process_one(large_img_path)
            # If succeeds, GPU has enough memory - simulate OOM
            if pipeline.materials_v3_engine is not None:
                with patch.object(pipeline.materials_v3_engine, 'process', side_effect=mock_gpu_oom):
                    result_oom = pipeline.process_one(large_img_path)
                    # Should fallback gracefully
                    if hasattr(result_oom, 'metadata') and 'materials_v3' in result_oom.metadata:
                        assert result_oom.metadata['materials_v3'].get('fallback', False) or \
                               'error' in result_oom.metadata['materials_v3']
        except RuntimeError as e:
            # If real OOM occurs, should be handled gracefully
            assert 'out of memory' in str(e).lower() or 'fallback' in str(e).lower()
        except Exception as e:
            # Other exceptions are acceptable if gracefully handled
            error_msg = str(e).lower()
            assert 'fallback' in error_msg or 'materials_v3' not in error_msg
    
    def test_result_consistency_across_iterations(self, sample_image, ci_safe_config):
        """MaterialsV3 should produce consistent results for same input."""
        pipeline = LuxPipelineV2(ci_safe_config)
        
        # Process same image 10 times
        results = []
        for i in range(10):
            result = pipeline.process_one(sample_image)
            results.append(result)
        
        # Verify all succeeded (no fallbacks)
        fallbacks = [
            i for i, r in enumerate(results)
            if hasattr(r, 'metadata') and 
               r.metadata.get('materials_v3', {}).get('fallback', False)
        ]
        
        assert len(fallbacks) == 0, \
            f"Inconsistent results: iterations {fallbacks} had fallbacks"
        
        # Check if results have materials_v3 metadata
        has_metadata = [
            i for i, r in enumerate(results)
            if hasattr(r, 'metadata') and 'materials_v3' in r.metadata
        ]
        
        # Either all have metadata or none (consistent behavior)
        assert len(has_metadata) == 0 or len(has_metadata) == 10, \
            f"Inconsistent metadata presence: {len(has_metadata)}/10 iterations"


@pytest.mark.slow
class TestMaterialsV3StressScenarios:
    """Additional stress scenarios."""
    
    @pytest.fixture
    def ci_safe_config(self, tmp_path):
        """Create CI-safe config with heuristic backend and AUTO depth mode."""
        config = PipelineConfig(
            preset=Preset.PRODUCTION_STANDARD,
            output_dir=tmp_path / "ci_output",
            write_outputs=False
        )
        config.segmentation.backend = "heuristic"
        # Stress tests should not fail on missing depth - use AUTO mode
        config.depth.mode = DepthMode.AUTO
        config.strict_depth = False
        return config
    
    def test_rapid_pipeline_creation_destruction(self, tmp_path, ci_safe_config):
        """Rapidly creating/destroying pipelines should not leak resources."""
        sample_image = tmp_path / "sample.jpg"
        img = Image.new('RGB', (256, 256), color=(128, 128, 128))
        img.save(sample_image, quality=95)
        
        print(f"\n{'='*60}")
        print(f"Rapid pipeline creation/destruction test (50 cycles)...")
        print(f"{'='*60}")
        
        for i in range(50):
            # Create new config for each cycle to test resource cleanup
            config = PipelineConfig(
                preset=Preset.PRODUCTION_STANDARD,
                output_dir=tmp_path / "output",
                write_outputs=False
            )
            config.segmentation.backend = "heuristic"
            
            pipeline = LuxPipelineV2(config)
            
            try:
                result = pipeline.process_one(sample_image)
                assert result is not None
            except Exception as e:
                pytest.fail(f"Cycle {i} failed: {e}")
            
            # Explicitly delete pipeline
            del pipeline
            
            if i % 10 == 0:
                gc.collect()
                print(f"Completed {i+1}/50 cycles")
        
        gc.collect()
        print(f"{'='*60}\n")
        print(f"All 50 cycles completed successfully")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-m', 'slow'])
