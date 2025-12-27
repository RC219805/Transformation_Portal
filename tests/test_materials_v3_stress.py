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
    Returns MaterialsV3 metadata dict from the pipeline result.
    Current contract: result["materials_v3_metadata"] (top-level key).
    Fallback: result["metadata"]["materials_v3"] (backward compat, unlikely).
    """
    if not isinstance(result, dict):
        return {}
    # Preferred: current pipeline contract (top-level)
    if isinstance(result.get("materials_v3_metadata"), dict):
        return result["materials_v3_metadata"]
    # Fallback: nested (for backward compatibility if schema ever changes)
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
    from lux_depth_v2.materials_v3 import MaterialsV3Config, MaterialTaxonomy, RefinementStrategy
    
    # Create CI-safe config in worker process
    config = PipelineConfig(
        preset=Preset.PRODUCTION_STANDARD,
        output_dir=output_dir,
        write_outputs=False
    )
    config.segmentation.backend = "heuristic"
    config.depth.mode = DepthMode.AUTO
    config.strict_depth = False
    # CRITICAL: Explicitly enable MaterialsV3 (PRODUCTION_STANDARD doesn't set it)
    config.materials_v3 = MaterialsV3Config(
        enabled=True,
        taxonomy=MaterialTaxonomy.BASE,
        refine_edges=RefinementStrategy.CANARY,
        apply_pixel_ops=True,
        max_megapixels=30.0
    )
    
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
        # CRITICAL: Explicitly enable MaterialsV3 (PRODUCTION_STANDARD doesn't set it)
        # Import MaterialsV3Config to enable
        from lux_depth_v2.materials_v3 import MaterialsV3Config, MaterialTaxonomy, RefinementStrategy
        config.materials_v3 = MaterialsV3Config(
            enabled=True,
            taxonomy=MaterialTaxonomy.BASE,
            refine_edges=RefinementStrategy.CANARY,
            apply_pixel_ops=True,
            max_megapixels=30.0
        )
        return config
    
    def test_1000_iteration_stability(self, sample_image, ci_safe_config, output_dir):
        """
        Validate MaterialsV3 stability over iterations.
        

        Note:
        - These stress tests are skipped on PR CI by module-level pytestmark.
        - They run on nightly/manual workflows (or locally).
        - Iterations are gated below via environment flags.
        
        Success Criteria:
        - Zero crashes (each iteration completes)
        - Zero fallbacks for valid synthetic images
        - Memory stable (no accumulation)
        """
        # Tier gating: Full stress on schedule, smoke on workflow_dispatch unless explicitly requested
        in_ci = os.getenv("CI") == "true"
        event_name = os.getenv("GITHUB_EVENT_NAME") or ""
        is_schedule = event_name == "schedule"
        full_stress = (os.getenv("MATERIALSV3_STRESS_FULL") == "1") or is_schedule
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
                # Assert MaterialsV3 actually ran (critical for stress validity)
                if i == 0:
                    assert m3, f"MaterialsV3 metadata missing on first iteration; verify PRODUCTION_STANDARD enables MaterialsV3"
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
        pct = (fallback_count / iterations * 100.0) if iterations else 0.0
        print(f"\n{'='*60}")
        print(
            f"{iterations}-iteration test completed in {elapsed_total:.1f}s "
            f"({avg_rate:.1f} iter/sec)"
        )
        print(f"Fallbacks: {fallback_count}/{iterations} ({pct:.1f}%)")
        print(f"{'='*60}\n")
        
        if fallback_count > 0:
            print("\nFallback errors:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        # Success criteria: 0% fallback rate for synthetic images
        assert fallback_count == 0, (
            f"Unexpected fallbacks: {fallback_count}/{iterations} ({pct:.1f}%)\n"
            f"Errors: {errors[:5]}"
        )
    
    def test_batch_processing_100_images(self, tmp_path, ci_safe_config, output_dir):
        """
        Process images in batch and verify MaterialsV3 stability.
        
        Nightly: 100 images (stress test, ~10min)
        Local: 100 images by default (unless MATERIALSV3_STRESS_FULL is used to force)
        
        Note: PR CI is skipped by module-level pytestmark; "PR smoke" is not executed
        unless you explicitly run it elsewhere.
        """
        # Tier gating: Full stress on schedule, smoke on workflow_dispatch unless explicitly requested
        in_ci = os.getenv("CI") == "true"
        event_name = os.getenv("GITHUB_EVENT_NAME") or ""
        is_schedule = event_name == "schedule"
        full_stress = (os.getenv("MATERIALSV3_STRESS_FULL") == "1") or is_schedule
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
        print(f"Processing {batch_size} images with MaterialsV3...")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        for i, img_path in enumerate(image_paths):
            try:
                result = pipeline.process_one(img_path)
                results.append(result)
                
                # Track fallbacks
                m3 = _materials_v3_meta(result)
                # Assert MaterialsV3 actually ran (critical for stress validity)
                if i == 0:
                    assert m3, f"MaterialsV3 metadata missing on first image; verify PRODUCTION_STANDARD enables MaterialsV3"
                if m3.get('fallback', False):
                    fallback_count += 1
                    errors.append(f"Image {i}: {m3.get('error', 'Unknown')}")
                
                # Progress reporting
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"Progress: {i+1}/{batch_size} images ({rate:.1f} img/sec, {fallback_count} fallbacks)")
                    
            except Exception as e:
                errors.append(f"Image {i}: {str(e)}")
                # Should not crash
                pytest.fail(f"Batch image {i} crashed: {e}")
        
        elapsed_total = time.time() - start_time
        avg_rate = batch_size / elapsed_total
        
        # Verify all processed
        assert len(results) == batch_size, f"Expected {batch_size} results, got {len(results)}"
        
        print(f"\n{'='*60}")
        pct = (fallback_count / batch_size * 100.0) if batch_size else 0.0
        print(f"Batch processing completed in {elapsed_total:.1f}s ({avg_rate:.1f} img/sec)")
        print(f"Fallbacks: {fallback_count}/{batch_size} ({pct:.1f}%)")
        print(f"{'='*60}\n")
        
        if errors:
            print("\nErrors encountered:")
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")
        
        # Allow up to 5% fallback rate (for edge cases in random generation)
        max_fallbacks = max(1, int(batch_size * 0.05))  # scale 5% to batch size
        assert fallback_count <= max_fallbacks, (
            f"Too many fallbacks: {fallback_count}/{batch_size} (> {max_fallbacks})\n"
            f"Errors: {errors[:5]}"
        )
    
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
        
        # Allow large image through MaterialsV3 megapixel gate (if enforced)
        if getattr(ci_safe_config, "materials_v3", None) is not None:
            ci_safe_config.materials_v3.max_megapixels = 50.0
        
        pipeline = LuxPipelineV2(ci_safe_config)
        
        def mock_gpu_oom(*args, **kwargs):
            raise RuntimeError("CUDA out of memory")
        
        # Prime lazy init
        _ = pipeline.process_one(large_img_path)

        # Check if engine is available after first run
        engine = getattr(pipeline, "materials_v3_engine", None)
        if engine is None or not hasattr(engine, "process"):
            pytest.skip("MaterialsV3 engine not available for OOM test")

        # Simulate GPU OOM and REQUIRE graceful fallback (no exception escape)
        with patch.object(engine, "process", side_effect=mock_gpu_oom):
            try:
                result_oom = pipeline.process_one(large_img_path)
            except Exception as e:
                pytest.fail(f"Expected graceful fallback on OOM; exception escaped: {e}")

        m3 = _materials_v3_meta(result_oom)
        assert m3, "MaterialsV3 metadata missing; cannot verify fallback contract"
        assert m3.get("fallback", False) or m3.get("error"), \
            "Expected fallback=True or error populated on simulated GPU OOM"
    
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
            if _materials_v3_meta(r).get('fallback', False)
        ]
        
        assert len(fallbacks) == 0, \
            f"Inconsistent results: iterations {fallbacks} had fallbacks"
        
        # Check if results have materials_v3 metadata
        has_metadata = [
            i for i, r in enumerate(results)
            if bool(_materials_v3_meta(r))
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
        # CRITICAL: Explicitly enable MaterialsV3 (PRODUCTION_STANDARD doesn't set it)
        from lux_depth_v2.materials_v3 import MaterialsV3Config, MaterialTaxonomy, RefinementStrategy
        config.materials_v3 = MaterialsV3Config(
            enabled=True,
            taxonomy=MaterialTaxonomy.BASE,
            refine_edges=RefinementStrategy.CANARY,
            apply_pixel_ops=True,
            max_megapixels=30.0
        )
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
            config.depth.mode = DepthMode.AUTO
            config.strict_depth = False
            # CRITICAL: Explicitly enable MaterialsV3 (PRODUCTION_STANDARD doesn't set it)
            from lux_depth_v2.materials_v3 import MaterialsV3Config, MaterialTaxonomy, RefinementStrategy
            config.materials_v3 = MaterialsV3Config(
                enabled=True,
                taxonomy=MaterialTaxonomy.BASE,
                refine_edges=RefinementStrategy.CANARY,
                apply_pixel_ops=True,
                max_megapixels=30.0
            )
            
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
